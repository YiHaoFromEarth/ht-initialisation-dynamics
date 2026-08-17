import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GPM:
    """Streamlined Gradient Projection Memory for sequential Conv2D and Linear backbones."""

    def __init__(self, variance_threshold=0.97):
        self.variance_threshold = variance_threshold
        self.global_bases = {}

    def extract_representation_matrix(self, live_activations, module, max_patches=2000):
        """Constructs representation matrix R matching the parameter input dimension."""
        if isinstance(module, nn.Conv2d):
            k_h, k_w = module.kernel_size
            patches = F.unfold(
                live_activations,
                kernel_size=(k_h, k_w),
                padding=module.padding,
                stride=module.stride,
            )  # [cite: 2]
            c_in_k = module.in_channels * k_h * k_w
            R = patches.transpose(0, 1).contiguous().view(c_in_k, -1)  # [cite: 2]

            if R.size(1) > max_patches:
                idx = torch.randperm(R.size(1), device=R.device)[:max_patches]
                R = R[:, idx]
            return R.to(torch.float32)

        elif isinstance(module, nn.Linear):
            if live_activations.dim() > 2:
                live_activations = live_activations.flatten(start_dim=1)
            return live_activations.T.to(torch.float32)

        raise TypeError(f"Unsupported module: {type(module)}")

    def update_basis(self, layer_id, module, live_activations, threshold=None):
        """Computes SVD on isolated residual activations and appends new orthogonal bases."""
        th = threshold if threshold is not None else self.variance_threshold
        R = self.extract_representation_matrix(live_activations, module)

        total_variance_sq = torch.sum(R**2).item()
        if total_variance_sq < 1e-12:
            return 0, {
                "total_variance_sq": total_variance_sq,
                "norm_projected_sq": 0.0,
                "residual_variance_sq": 0.0,
            }

        # 1. Project out existing memory
        if layer_id in self.global_bases:
            current_basis = self.global_bases[layer_id].to(
                R.device, dtype=torch.float32
            )
            R_proj = current_basis @ (current_basis.T @ R)  # [cite: 2]
            R_hat = R - R_proj  # [cite: 2]
            norm_projected_sq = torch.sum(R_proj**2).item()
        else:
            R_hat = R
            norm_projected_sq = 0.0

        residual_variance_sq = torch.sum(R_hat**2).item()
        diagnostics = {
            "total_variance_sq": total_variance_sq,
            "norm_projected_sq": norm_projected_sq,
            "residual_variance_sq": residual_variance_sq,
        }

        target_energy = th * total_variance_sq
        if norm_projected_sq >= target_energy:
            return 0, diagnostics

        # 2. SVD on Residual Space
        U, S_vals, _ = torch.linalg.svd(R_hat, full_matrices=False)  # [cite: 2]
        cum_residual_var = torch.cumsum(S_vals**2, dim=0)
        total_accounted_var = norm_projected_sq + cum_residual_var

        satisfying_mask = total_accounted_var >= target_energy  # [cite: 2]
        k = (
            S_vals.shape[0]
            if not torch.any(satisfying_mask)
            else torch.where(satisfying_mask)[0][0].item() + 1
        )
        new_basis = U[:, :k].clone().to(torch.float32)  # [cite: 2]

        # 3. Memory Update
        if layer_id not in self.global_bases:
            self.global_bases[layer_id] = new_basis
        else:
            combined = torch.cat(
                [self.global_bases[layer_id], new_basis], dim=1
            )  # [cite: 2]
            Q, _ = torch.linalg.qr(combined)
            self.global_bases[layer_id] = Q

        return k, diagnostics

    def project_gradient(self, layer_id, grad):
        """Projects gradients onto the orthogonal complement of the stored subspace."""
        if layer_id not in self.global_bases:
            return grad

        basis = self.global_bases[layer_id].to(grad.device, dtype=grad.dtype)

        if grad.dim() == 4:  # Conv2D: [C_out, C_in, Kh, Kw]
            c_out, c_in, k_h, k_w = grad.shape
            grad_mat = grad.view(c_out, c_in * k_h * k_w)
            grad_proj = grad_mat - (grad_mat @ basis) @ basis.T  # [cite: 2]
            return grad_proj.view(c_out, c_in, k_h, k_w)
        elif grad.dim() == 2:  # Linear: [out_features, in_features]
            return grad - (grad @ basis) @ basis.T  # [cite: 2]
        return grad - basis @ (basis.T @ grad)

    def project_model_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.global_bases:
                param.grad.copy_(self.project_gradient(name, param.grad))

    def get_basis_ranks(self):
        return {
            layer_id: basis.shape[1] for layer_id, basis in self.global_bases.items()
        }

    def get_total_basis_rank(self):
        return sum(self.get_basis_ranks().values())


class CLMetricsTracker:
    def __init__(self, max_tasks=20):
        self.max_tasks = max_tasks
        # defaultdict-style dynamic storage to avoid rigid pre-allocation
        self.history = {}
        self._total_steps_logged = 0

    def _ensure_key_exists(self, key):
        """Lazy initialization for new dynamic metric keys."""
        if key not in self.history:
            # Pad retroactively with None for previous steps if a metric is added late
            self.history[key] = [None] * self._total_steps_logged

    def log(self, step, acc_list, **extra_metrics):
        """
        Logs a single evaluation step.

        Args:
            step: Global step index.
            acc_list: List of task accuracies [acc_t0, acc_t1, ...]
            **extra_metrics: Arbitrary method-specific key-value pairs.
                             Examples:
                               basis_rank=[12, 18, 5]
                               cum_rank=35
                               effective_rank={"layer1": 4.2, "layer2": 8.1}
        """
        self._ensure_key_exists("step")
        self.history["step"].append(step)

        # 1. Log Task Accuracies
        for t_idx in range(self.max_tasks):
            col_key = f"task_{t_idx}_acc"
            self._ensure_key_exists(col_key)
            if t_idx < len(acc_list):
                self.history[col_key].append(float(acc_list[t_idx]))
            else:
                self.history[col_key].append(None)

        # 2. Dynamically Log Extra Method-Specific Metrics
        for metric_name, val in extra_metrics.items():
            if isinstance(val, (list, tuple)):
                # Handle sequence inputs (e.g., basis_rank per task or per layer)
                for idx, sub_val in enumerate(val):
                    sub_key = f"{metric_name}_{idx}"
                    self._ensure_key_exists(sub_key)
                    self.history[sub_key].append(
                        None if sub_val is None else float(sub_val)
                    )
            elif isinstance(val, dict):
                # Handle dictionary inputs (e.g., {"layer1": 4.5, "layer2": 8.2})
                for dict_key, sub_val in val.items():
                    sub_key = f"{metric_name}_{dict_key}"
                    self._ensure_key_exists(sub_key)
                    self.history[sub_key].append(
                        None if sub_val is None else float(sub_val)
                    )
            else:
                # Handle single scalar values
                self._ensure_key_exists(metric_name)
                self.history[metric_name].append(None if val is None else float(val))

        # 3. Pad any keys that were created previously but NOT passed in this log call
        self._total_steps_logged += 1
        for key in self.history:
            if len(self.history[key]) < self._total_steps_logged:
                self.history[key].append(None)

    def save_to_csv(self, filepath="cl_experiment_metrics.csv"):
        """Exports all recorded columns into a clean, flat CSV file."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Gather headers dynamically, prioritizing 'step' first
        headers = ["step"] + [k for k in self.history if k != "step"]

        # Drop columns that are completely empty (all None)
        active_headers = [
            h for h in headers if any(val is not None for val in self.history[h])
        ]

        num_rows = self._total_steps_logged

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(active_headers)

            for r_idx in range(num_rows):
                row_data = [
                    "" if self.history[h][r_idx] is None else self.history[h][r_idx]
                    for h in active_headers
                ]
                writer.writerow(row_data)

        print(f"Metrics successfully exported to: '{filepath}'")


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
