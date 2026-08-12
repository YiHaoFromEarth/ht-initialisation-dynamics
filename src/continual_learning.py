import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EWC:
    def __init__(self, model, ewc_lambda=1000):
        """
        Args:
            model: Your GeneralMLP instance.
            ewc_lambda: Regularization strength (hyperparameter).
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self._precision_matrices = {}

    def on_task_end(self, dataset, device, num_samples=300):
        """
        Calculates the Fisher Information Matrix diagonal and stores task weights.
        Args:
            dataset: TensorDataset for the task just completed.
            num_samples: Number of samples to use for Fisher estimation (default matches GPM).
        """
        self.model.eval()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        # Fisher estimation: Use a subset of the task data
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        for i, (input, target) in enumerate(loader):
            if i >= num_samples:
                break

            input, target = input.to(device), target.to(device)
            self.model.zero_grad()
            output = self.model(input)

            # The Fisher Information is the variance of the score function (gradient of log-likelihood)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data**2 / num_samples

        # Store mean weights and the Fisher diagonal (precision matrix)
        for n, p in copy.deepcopy(self.params).items():
            self._precision_matrices[f"{n}_{len(self._means)}"] = precision_matrices[n]
            self._means[f"{n}_{len(self._means)}"] = p.data

    def penalty(self, model):
        """Calculates the weighted squared penalty between current and past weights."""
        loss = 0
        for n, p in model.named_parameters():
            # Sum penalties across all previous tasks
            for task_key in self._means:
                if n in task_key:
                    _precision = self._precision_matrices[task_key]
                    _mean = self._means[task_key]
                    # Math: loss = lambda/2 * Fisher * (theta - theta_old)^2
                    loss += (_precision * (p - _mean) ** 2).sum()
        return loss * (self.ewc_lambda / 2)


class SpectralNormGain:
    def __init__(self, name="weight", n_power_iterations=1, gain=1.0):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.gain = gain

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # Power iteration to estimate the largest singular value (sigma)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of W is the same as W^T, so we iterate
                    v.data = F.normalize(torch.mv(weight.t(), u), dim=0, eps=1e-12)
                    u.data = F.normalize(torch.mv(weight, v), dim=0, eps=1e-12)
                if self.n_power_iterations > 0:
                    u.data.copy_(u)
                    v.data.copy_(v)

        sigma = torch.dot(u, torch.mv(weight, v))
        # Apply the gain: W_new = gain * (W / sigma)
        return weight * (self.gain / sigma)

    def __call__(self, module, inputs):
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    @staticmethod
    def apply(module, name, n_power_iterations, gain):
        for fn in module._forward_pre_hooks.values():
            if isinstance(fn, SpectralNormGain) and fn.name == name:
                return fn

        fn = SpectralNormGain(name, n_power_iterations, gain)
        weight = getattr(module, name)

        # Initialize the u and v vectors for power iteration
        with torch.no_grad():
            d = weight.size(0)
            u = F.normalize(weight.new_empty(d).normal_(0, 1), dim=0, eps=1e-12)
            v = F.normalize(
                weight.new_empty(weight.size(1)).normal_(0, 1), dim=0, eps=1e-12
            )

        # Delete the original weight and replace with buffers/parameters
        delattr(module, name)
        module.register_parameter(name + "_orig", nn.Parameter(weight.detach()))
        module.register_buffer(name + "_u", u)
        module.register_buffer(name + "_v", v)

        # Add the hook to re-calculate weight before every forward pass
        module.register_forward_pre_hook(fn)
        return fn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Calculate the 'e_w' perturbation
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the 'peak'
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to original weights
        self.base_optimizer.step()  # update based on the peak gradient
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


class GPM:
    def __init__(self, variance_threshold=0.97, orthog_method="qr"):
        self.variance_threshold = variance_threshold
        self.orthog_method = orthog_method.lower()

        if self.orthog_method not in ["qr", "lowdin"]:
            raise ValueError("orthog_method must be either 'qr' or 'lowdin'")

        self.global_bases = {}

    def _orthogonalize(self, W, eps=1e-8):
        if self.orthog_method == "qr":
            Q, R_qr = torch.linalg.qr(W)
            # Phase correction matching old working script
            d_diag = torch.diagonal(R_qr, dim1=-2, dim2=-1)
            ph = d_diag / (torch.abs(d_diag) + eps)
            return Q * ph.unsqueeze(0)
        elif self.orthog_method == "lowdin":
            Gram = W.T @ W
            eigenvalues, eigenvectors = torch.linalg.eigh(Gram)
            e_clamped = torch.clamp(eigenvalues, min=eps)
            inv_sqrt_matrix = (
                eigenvectors @ torch.diag(1.0 / torch.sqrt(e_clamped)) @ eigenvectors.T
            )
            return W @ inv_sqrt_matrix

    def update_basis(self, layer_id, live_activations, masking_fn=None):
        if live_activations.dim() > 2:
            live_activations = live_activations.flatten(start_dim=1)

        # Always orient R as [in_features, batch_size]
        R = live_activations.T.to(torch.float32)

        # 1. Total Raw Variance Energy
        total_variance_sq = torch.sum(R**2).item()
        if total_variance_sq < 1e-12:
            diagnostics = {
                "total_variance_sq": total_variance_sq,
                "norm_projected_sq": 0.0,
                "residual_variance_sq": 0.0,
            }
            return 0, diagnostics

        # 2. Residual Projection & Variance Accounting
        if layer_id in self.global_bases:
            current_basis = self.global_bases[layer_id].to(torch.float32)
            R_proj = current_basis @ (current_basis.T @ R)
            R_hat = R - R_proj
            norm_projected_sq = torch.sum(R_proj**2).item()
        else:
            R_hat = R
            norm_projected_sq = 0.0

        residual_variance_sq = torch.sum(R_hat**2).item()

        # Package raw energy metrics
        diagnostics = {
            "total_variance_sq": total_variance_sq,
            "norm_projected_sq": norm_projected_sq,
            "residual_variance_sq": residual_variance_sq,
        }

        target_energy = self.variance_threshold * total_variance_sq

        # If existing memory already satisfies the energy threshold
        if norm_projected_sq >= target_energy:
            return 0, diagnostics

        # 3. SVD on Isolated Residual Activations
        U, S_vals, _ = torch.linalg.svd(R_hat, full_matrices=False)
        s_squared = S_vals**2

        cum_residual_var = torch.cumsum(s_squared, dim=0)
        total_accounted_var = norm_projected_sq + cum_residual_var

        satisfying_mask = total_accounted_var >= target_energy
        if not torch.any(satisfying_mask):
            # FIX: If threshold isn't met due to float precision, take all components
            k = S_vals.shape[0]
        else:
            k = torch.where(satisfying_mask)[0][0].item() + 1

        task_basis = U[:, :k].clone().to(torch.float32)

        # 4. Optional Soft Masking & Re-Orthogonalization
        if masking_fn is not None:
            # CRITICAL FIX: Pass R_hat.T (residual activations) instead of live_activations!
            R_hat_float32 = R_hat.T.to(torch.float32)
            mask = masking_fn(task_basis, R_hat_float32)

            task_basis_masked = task_basis * mask

            # CRITICAL FIX: Re-normalize each column to unit length before QR!
            col_norms = torch.norm(task_basis_masked, dim=0, keepdim=True)
            col_norms[col_norms == 0] = 1.0
            task_basis_masked = task_basis_masked / col_norms

            new_basis = self._orthogonalize(task_basis_masked)
        else:
            new_basis = task_basis

        # 5. Append to Global Memory Vault
        if layer_id not in self.global_bases:
            self.global_bases[layer_id] = new_basis
        else:
            self.global_bases[layer_id] = torch.cat(
                [self.global_bases[layer_id], new_basis], dim=1
            )

        return k, diagnostics

    def project_gradient(self, layer_id, grad):
        if layer_id not in self.global_bases or self.global_bases[layer_id] is None:
            return grad

        basis = self.global_bases[layer_id]

        if grad.dim() == 2:  # Weight matrix [out_dim, in_dim]
            return grad - (grad @ basis) @ basis.T
        else:  # Bias vector [hidden_dim]
            return grad - basis @ (basis.T @ grad)

    def project_model_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.global_bases:
                param.grad.copy_(self.project_gradient(name, param.grad))

    def get_basis_ranks(self):
        return {
            layer_id: (basis.shape[1] if basis is not None else 0)
            for layer_id, basis in self.global_bases.items()
        }

    def get_total_basis_rank(self):
        return sum(self.get_basis_ranks().values())


def rmt_resolvent_masking(task_basis, live_activations, gamma=0.0001, beta=0.01):
    """
    Method 4: Pure RMT Resolvent Similarity (Streamlined HTGPM)

    Args:
        task_basis (torch.Tensor): Singular vectors U[:, :k] of shape [N_dim, k].
        live_activations (torch.Tensor): Live input activations [Batch, N_dim].
        gamma (float): Spectral distance past the edge (delta = gamma * lambda_max).
        beta (float): Spectral blur factor for imaginary shift (eta = beta * mean_var).

    Returns:
        torch.Tensor: Continuous soft mask of shape [N_dim, k] on the original device.
    """
    device = task_basis.device

    # 1. Convert inputs to NumPy arrays matching your original logic
    # live_activations arrives as [Batch, N_dim]
    R_hat = live_activations.detach().cpu().numpy()
    raw_basis = task_basis.detach().cpu().numpy()  # [N_dim, k]

    N_dim, k = raw_basis.shape
    if k == 0 or N_dim == 0:
        return torch.ones_like(task_basis)

    # 2. Covariance Operator Frame [N_dim x N_dim]
    A_op = R_hat.T @ R_hat

    # 3. Complete Eigenspectrum Isolation
    eigenvalues, eigenvectors = np.linalg.eigh(A_op)
    lambda_max = eigenvalues[-1]

    active_eigs = eigenvalues[eigenvalues > 1e-5]
    mean_active_variance = np.mean(active_eigs) if len(active_eigs) > 0 else 1.0

    # 4. Dimensionless Tuning Dials & Complex Resolvent Shift
    delta = gamma * lambda_max
    eta_dynamic = beta * mean_active_variance

    z = (lambda_max + delta) + 1j * eta_dynamic
    resolvent_diagonal = 1.0 / (eigenvalues - z)

    # 5. Reconstitute Complex Spatial Resolvent via Functional Calculus
    G_complex = eigenvectors @ np.diag(resolvent_diagonal) @ eigenvectors.T
    resolvent_envelope_matrix = np.abs(G_complex)

    # 6. Pure Scale-Invariant Similarity
    diag_G = np.diagonal(resolvent_envelope_matrix)
    normalization_matrix = np.sqrt(np.outer(diag_G, diag_G))
    rmt_similarity = resolvent_envelope_matrix / (normalization_matrix + 1e-9)

    # 7. Mask Extraction Loop Across Singular Vector Columns
    mask_columns = []

    for col_idx in range(k):
        v = raw_basis[:, col_idx]
        v_norm = np.linalg.norm(v) + 1e-12
        v_unit = v / v_norm

        # Inverse Participation Ratio (IPR) to detect hub concentration
        ipr_val = np.sum(v_unit**4)
        effective_hubs_count = int(np.ceil(1.0 / ipr_val))

        # Isolate physical hub locations via 2nd power energy rank
        neuron_contributions = np.abs(v_unit) ** 2
        sort_idx = np.argsort(neuron_contributions)[::-1]
        rmt_hub_indices = sort_idx[:effective_hubs_count]

        # Aggregate Greens function envelope across hubs
        soft_mask = np.zeros(N_dim)
        for hub in rmt_hub_indices:
            envelope = rmt_similarity[hub, :]
            soft_mask = np.maximum(soft_mask, envelope)

        # Bounding Enforcement
        max_val = np.max(soft_mask)
        if max_val > 0:
            soft_mask /= max_val

        # Store soft mask squared as per Method 4 formulation
        mask_columns.append(soft_mask**2)

    # Stack columns to shape [N_dim, k]
    M_rmt_total = np.stack(mask_columns, axis=1)

    # Convert back to PyTorch tensor on original device
    return torch.tensor(M_rmt_total, dtype=task_basis.dtype, device=device)


def ipr_delta_d_masking_fn(task_basis, live_activations, alpha_0=1.5, gamma=4.0):
    """
    Simplified Multifractal IPR + Delta D Masking Strategy

    Args:
        task_basis (torch.Tensor): Singular vectors U[:, :k] of shape [N_dim, k].
        live_activations (torch.Tensor): Live input activations [Batch, N_dim].
        alpha_0 (float): Baseline power-law exponent for standard/monofractal states (Delta D = 0).
        gamma (float): Sensitivity multiplier scaling exponent decay relative to Delta D.

    Returns:
        torch.Tensor: Continuous mask of shape [N_dim, k] on original device.
    """
    device = task_basis.device
    raw_basis = task_basis.detach().cpu().numpy()  # [N_dim, k]

    N_dim, k = raw_basis.shape
    if k == 0 or N_dim == 0:
        return torch.ones_like(task_basis)

    M_multifractal = np.zeros((N_dim, k), dtype=np.float32)
    log_N = np.log(N_dim) if N_dim > 1 else 1.0

    for col in range(k):
        v = raw_basis[:, col]
        v_unit = v / (np.linalg.norm(v) + 1e-12)
        mag = np.abs(v_unit)

        # 1. Energy Distribution & Dimensions (D1 and D2)
        p = (mag**2) / (np.sum(mag**2) + 1e-12)

        d1 = (-np.sum(p * np.log(p + 1e-12))) / log_N
        ipr = np.sum(p**2)
        d2 = -np.log(max(ipr, 1e-12)) / log_N

        # 2. Singularity Spread
        mf_delta = max(0.0, d1 - d2)

        # 3. Hubs & Margins
        num_hubs = max(1, int(np.round(1.0 / ipr)))
        sort_idx = np.argsort(mag)[::-1]
        hub_idx, margin_idx = sort_idx[:num_hubs], sort_idx[num_hubs:]

        M_multifractal[hub_idx, col] = 1.0

        # 4. Simple Direct Exponential Adjustment
        # alpha_0 sets the baseline; gamma scales down exponent as multifractality increases
        alpha_dynamic = max(0.1, alpha_0 - gamma * mf_delta)

        # 5. Tail Envelope
        if len(margin_idx) > 0:
            max_mar, min_mar = mag[sort_idx[num_hubs]], mag[sort_idx[-1]]
            denom = max_mar - min_mar
            if denom > 1e-8:
                norm_margin = (mag[margin_idx] - min_mar) / denom
                M_multifractal[margin_idx, col] = np.power(norm_margin, alpha_dynamic)

    return torch.tensor(M_multifractal, dtype=task_basis.dtype, device=device)


def htgpm_powerlaw_resolvent_masking_fn(
    task_basis, live_activations, alpha=2.0, eta=1e-3
):
    """
    Original HTGPM Resolvent Power-Law Masking Function

    Transforms the Resolvent complex spatial metric into a topological
    distance matrix, then applies a power-law spatial decay profile around
    hubs isolated via Inverse Participation Ratio (IPR).

    Args:
        task_basis (torch.Tensor): Singular vectors U[:, :k] of shape [N_dim, k].
        live_activations (torch.Tensor): Live input activations [Batch, N_dim].
        alpha (float): Tail index / localization length parameter (xi = eta_decay = alpha).
        eta (float): Regularization shift for Resolvent complex edge (lambda_max + 1j * eta).

    Returns:
        torch.Tensor: Power-law soft mask of shape [N_dim, k] on the original device.
    """
    device = task_basis.device
    raw_basis = task_basis.detach().cpu().numpy()  # [N_dim, k]
    R_hat = live_activations.detach().cpu().numpy().T  # Shape: [N_dim, Batch]

    N_dim, k = raw_basis.shape
    if k == 0 or N_dim == 0:
        return torch.ones_like(task_basis)

    # 1. Empirical Matrix Operator A = R_hat @ R_hat.T
    A = R_hat @ R_hat.T

    # Extract spectral edge via Hermitian eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(A)
    lambda_max = eigenvalues[-1]

    # 2. Compute Complex-Shifted Resolvent Matrix: G(z) = (A - zI)^-1
    # Stabilized using np.linalg.solve instead of direct inv
    z = lambda_max + 1j * eta
    system_matrix = A - z * np.eye(N_dim, dtype=np.complex128)
    Resolvent = np.linalg.solve(system_matrix, np.eye(N_dim, dtype=np.complex128))
    resolvent_envelope_matrix = np.abs(Resolvent)

    # 3. Scale-Invariant Cauchy-Schwarz Normalization to RMT Distance Topology (0 to 1)
    diag_G = np.diagonal(resolvent_envelope_matrix)
    normalization_matrix = np.sqrt(np.outer(diag_G, diag_G))
    rmt_similarity = resolvent_envelope_matrix / (normalization_matrix + 1e-9)
    rmt_distance_matrix = np.clip(1.0 - rmt_similarity, 0.0, 1.0)

    # 4. Mask Processing Loop Across Columns
    mask_columns = []
    xi_resolvent = alpha  # Localization length
    eta_resolvent = alpha  # Power-law algebraic decay exponent

    for col_idx in range(k):
        v = raw_basis[:, col_idx]
        v_unit = v / (np.linalg.norm(v) + 1e-12)

        # Compute Inverse Participation Ratio (IPR) for Hub Isolation
        ipr_val = np.sum(v_unit**4)
        effective_hubs_count = int(np.ceil(1.0 / ipr_val))

        # Isolate hub indices using 2nd power energy rank
        neuron_contributions = np.abs(v_unit) ** 2
        sort_idx = np.argsort(neuron_contributions)[::-1]
        rmt_hub_indices = sort_idx[:effective_hubs_count]

        # Apply continuous Power-Law decay around hubs: (1 + d / xi)^(-eta)
        soft_mask = np.zeros(N_dim, dtype=np.float32)
        for hub in rmt_hub_indices:
            d_from_hub = rmt_distance_matrix[hub, :]
            envelope = (1.0 + d_from_hub / xi_resolvent) ** (-eta_resolvent)
            soft_mask = np.maximum(soft_mask, envelope)

        # Bounding enforcement
        max_val = np.max(soft_mask)
        if max_val > 0:
            soft_mask /= max_val

        mask_columns.append(soft_mask)

    M_powerlaw = np.stack(mask_columns, axis=1)

    return torch.tensor(M_powerlaw, dtype=task_basis.dtype, device=device)
