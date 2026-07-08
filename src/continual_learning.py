from xml.parsers.expat import model

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


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
            for task_key in self._means.keys():
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
        super(SAM, self).__init__(params, defaults)
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


def spectral_norm_with_gain(module, gain=1.2, name="weight", n_power_iterations=1):
    SpectralNormGain.apply(module, name, n_power_iterations, gain)
    return module


def update_GPM_bases(model, images, threshold, feature_list=None, activation_fn=torch.tanh):
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    # FIX: Explicitly flatten the raw images to 2D (Batch, 784)
    flattened_images = images.view(images.size(0), -1)
    all_inputs = [flattened_images]

    for h in list(reps.values())[:-1]:
        # Hidden activations from nn.Linear are already 2D, but we ensure it
        activated_h = activation_fn(h)
        all_inputs.append(activated_h.view(h.size(0), -1))

    if feature_list is None:
        feature_list = [None] * len(all_inputs)

    current_ranks = []
    for i, activation in enumerate(all_inputs):
        X = activation.cpu().numpy()
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        s_sq = S**2
        s_sum = np.sum(s_sq)

        # Vectorized threshold calculation (cleaner than the manual loop)
        cumulative_variance = np.cumsum(s_sq) / s_sum
        k = np.argmax(cumulative_variance >= threshold) + 1
        current_ranks.append(k)  # Store k for logging

        task_basis = Vh[:k].T

        if feature_list[i] is None:
            feature_list[i] = task_basis
        else:
            # We must bring the existing GPU tensor back to CPU numpy for the SVD merge
            old_basis = feature_list[i].cpu().numpy()
            combined = np.concatenate((old_basis, task_basis), axis=1)
            U_new, _, _ = np.linalg.svd(combined, full_matrices=False)
            feature_list[i] = U_new[:, : min(combined.shape[0], combined.shape[1])]

    # CRITICAL: Pre-load the bases to the GPU here, NOT in the training loop
    device = next(model.parameters()).device
    tensor_bases = [
        torch.tensor(b, dtype=torch.float32, device=device) if b is not None else None
        for b in feature_list
    ]

    return tensor_bases, current_ranks


def apply_GPM_projection(linear_layers, feature_list, ema_buffers, alpha=0.01,
                         use_gating=True, beta=0.99, tau=0.01):
    """
    Applies an asymmetric directional gate to Leaky GPM regulated by an
    Exponential Moving Average (EMA) and custom noise-floor threshold.

    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: The pre-loaded GPU tensors of basis matrices.
        ema_buffers: Persistent dictionary tracking the historical alignment per layer.
        alpha: Maximum permitted leakage factor for cooperative steps.
        use_gating: Boolean flag to toggle the temporal gating protection.
        beta: Momentum factor for the EMA tracking corridor (low-pass filter).
        tau: Gating threshold parameter to cut through the background noise floor.
    """
    if feature_list is None:
        return

    with torch.no_grad():
        for i, layer in enumerate(linear_layers):
            if i >= len(feature_list) or feature_list[i] is None:
                continue

            grad = layer.weight.grad
            if grad is None:
                continue

            basis = feature_list[i]  # Shape: (In_Features, K)
            layer_key = f'layer_{i}'

            # 1. Isolate the parallel component (the projection onto past subspace)
            # Math: g_parallel = g @ B @ B.T
            parallel_grad = torch.mm(torch.mm(grad, basis), basis.t())

            # 2. Isolate the orthogonal component
            # Math: g_orthogonal = g - g_parallel
            ortho_grad = grad - parallel_grad

            # 3. Compute Directional Gating Logic
            effective_alpha = alpha

            if use_gating:
                # Isolate the current weight landscape inside the same past subspace
                parallel_weight = torch.mm(torch.mm(layer.weight, basis), basis.t())

                # Compute the true Frobenius Inner Product (Matrix Dot Product)
                dot_product = torch.sum(parallel_grad * parallel_weight).item()

                # Calculate normalized Frobenius cosine similarity (-1.0 to +1.0)
                p_grad_norm = torch.linalg.norm(parallel_grad).item()
                p_weight_norm = torch.linalg.norm(parallel_weight).item()

                current_raw_alignment = 0.0
                if p_grad_norm > 0 and p_weight_norm > 0:
                    current_raw_alignment = dot_product / (p_grad_norm * p_weight_norm)

                # Update the temporal moving average buffer
                if layer_key not in ema_buffers:
                    ema_buffers[layer_key] = current_raw_alignment
                else:
                    ema_buffers[layer_key] = (beta * ema_buffers[layer_key]) + ((1.0 - beta) * current_raw_alignment)

                # The gate now requires the historical trend line to clear the custom floor tau
                if ema_buffers[layer_key] <= tau:
                    effective_alpha = 0.0

            # 4. Recombine components using the gated alpha value
            gated_grad = ortho_grad + effective_alpha * parallel_grad

            # Update the gradient in-place
            layer.weight.grad.copy_(gated_grad)


def update_PowerGPM_bases(model, images, gamma=0.5, q_thresh=0.2, energy_cutoff=0.99,
                          feature_list=None, activation_fn=torch.tanh):
    """
    Extracts structurally meaningful basis vectors using fractional energy truncation,
    preventing subspace saturation while calculating power-law leakage.

    Args:
        model: The active neural network module.
        images: Calibration image batch tensor.
        gamma: Exponent tuning the steepness of the power-law protection curve.
        q_thresh: Fractional order power to flatten the power-law spectrum for truncation.
        energy_cutoff: Cumulative variance percentage to retain (e.g., 0.99).
        feature_list: Persistent list tracking [basis_matrix, leakage_vector] per layer.
    """
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    flattened_images = images.view(images.size(0), -1)
    all_inputs = [flattened_images]

    for h in list(reps.values())[:-1]:
        activated_h = activation_fn(h)
        all_inputs.append(activated_h.view(h.size(0), -1))

    if feature_list is None:
        feature_list = [None] * len(all_inputs)

    current_ranks = []

    for i, activation in enumerate(all_inputs):
        X = activation.cpu().numpy()
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        # 1. Compute the compressed fractional energy spectrum
        # Using a small q power flattens the power law, allowing us to find the true noise floor
        fractional_energy = S ** (2 * q_thresh)
        cumulative_fractional_var = np.cumsum(fractional_energy) / np.sum(fractional_energy)

        # Determine the truncation cutoff index k
        k = np.argmax(cumulative_fractional_var >= energy_cutoff) + 1
        current_ranks.append(k)

        # 2. Extract only the active structural directions and singular values
        S_truncated = S[:k]
        task_basis = Vh[:k].T  # Shape: (In_Features, k)

        # 3. Calculate scale-invariant leakage vector ONLY across the active coordinates
        relative_energy = (S_truncated / (S_truncated.max() + 1e-10)) ** gamma
        leakage_vector = 1.0 - relative_energy

        if feature_list[i] is None:
            feature_list[i] = (task_basis, leakage_vector)
        else:
            # Reconcile sequential tasks cleanly on host CPU memory
            old_basis_tensor, old_leak = feature_list[i]
            old_basis = old_basis_tensor.detach().cpu().numpy()

            combined = np.concatenate((old_basis, task_basis), axis=1)
            U_new, S_new, _ = np.linalg.svd(combined, full_matrices=False)

            # Run the fractional check on the combined space to keep dimensions tight
            frac_energy_new = S_new ** (2 * q_thresh)
            cum_frac_var_new = np.cumsum(frac_energy_new) / np.sum(frac_energy_new)
            k_new = np.argmax(cum_frac_var_new >= energy_cutoff) + 1

            merged_basis = U_new[:, :k_new]
            rel_energy_new = (S_new[:k_new] / (S_new[:k_new].max() + 1e-10)) ** gamma
            merged_leak = 1.0 - rel_energy_new

            feature_list[i] = (merged_basis, merged_leak)

    # Pre-load the pruned, scale-aware matrices to the GPU device
    device = next(model.parameters()).device
    tensor_bases = []
    for item in feature_list:
        if item is not None:
            b_mat, l_vec = item
            t_basis = torch.tensor(b_mat, dtype=torch.float32, device=device)
            t_leak = torch.tensor(l_vec, dtype=torch.float32, device=device)
            tensor_bases.append((t_basis, t_leak))
        else:
            tensor_bases.append(None)

    return tensor_bases, current_ranks


def apply_PowerGPM_projection(linear_layers, feature_list):
    """
    Applies a scale-aware, non-linear subspace projection. Rather than a flat,
    scalar leakage ceiling, the attenuation factor decays smoothly along the
    power-law spectrum of past activations.

    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: Persistent list containing (basis_tensor, leak_tensor) tuples.
    """
    if feature_list is None:
        return

    with torch.no_grad():
        for i, layer in enumerate(linear_layers):
            if i >= len(feature_list) or feature_list[i] is None:
                continue

            grad = layer.weight.grad
            if grad is None:
                continue

            # Unpack your multi-scale spectral tuple
            basis, leakage = feature_list[i]
            # basis shape: (In_Features, Full_Rank)
            # leakage shape: (Full_Rank)

            # 1. Transform the gradient into your orthogonal coordinate basis
            # Math: G_transformed = G @ B
            grad_in_subspace = torch.mm(grad, basis)

            # 2. Apply scale-aware damping vector elements directly along the channels
            # Multiplying by the leakage scales down coordinates matching macro spikes
            # while leaving the high-ID background elements perfectly intact
            gated_subspace_grad = grad_in_subspace * leakage

            # 3. Project back out to standard parameter space
            # Math: G_gated = G_orthogonal + (G_parallel * leak) => G @ (I - B @ diag(1-leak) @ B.T)
            gated_grad = torch.mm(gated_subspace_grad, basis.t())

            # Update the parameter gradient tensor in-place
            layer.weight.grad.copy_(gated_grad)


def update_SparseGPM_bases(
    model,
    images,
    global_threshold,
    local_threshold=1.0,
    xi=0.0,
    feature_list=None,
    activation_fn=torch.tanh,
):
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    flattened_images = images.view(images.size(0), -1)
    all_inputs = [flattened_images]

    for h in list(reps.values())[:-1]:
        activated_h = activation_fn(h)
        all_inputs.append(activated_h.view(h.size(0), -1))

    if feature_list is None:
        feature_list = [None] * len(all_inputs)

    current_ranks = []

    for i, activation in enumerate(all_inputs):
        # 1. Shape Transformation: (Samples, N_dim) -> (N_dim, Samples)
        # Matches textbook GPM formulation: R_l = [x_1, x_2, ..., x_ns]
        R = activation.cpu().numpy().T
        N_dim = R.shape[0]

        # Calculate total raw Frobenius norm variance of the incoming domain
        total_variance_sq = np.sum(R ** 2)
        if total_variance_sq < 1e-12:
            current_ranks.append(0)
            continue

        # 2. EQUATION 8: RESIDUAL PROJECTION LAYER ELIMINATION
        if feature_list[i] is not None:
            old_basis = feature_list[i]
            if torch.is_tensor(old_basis):
                old_basis = old_basis.cpu().numpy()

            # Project onto accumulated memory: R_proj = M @ (M.T @ R)
            R_proj = old_basis @ (old_basis.T @ R)
            # Isolate pure unexplained innovations: R_hat = R - R_proj
            R_hat = R - R_proj
            norm_projected_sq = np.sum(R_proj ** 2)
        else:
            R_hat = R
            old_basis = None
            norm_projected_sq = 0.0

        # 3. EQUATION 9: SVD ON ISOLATED RESIDUAL COMPONENT
        U, S, Vh = np.linalg.svd(R_hat, full_matrices=False)
        s_sq = S**2

        # Compute minimum rank 'k' accounting for historical compensation
        # ||R_proj||^2 + ||(R_hat)_k||^2 >= epsilon_th * ||R||^2
        cumulative_residual_variance = np.cumsum(s_sq)
        total_accounted_variance = norm_projected_sq + cumulative_residual_variance

        target_energy = global_threshold * total_variance_sq

        # Determine if old tasks already fully satisfy the threshold requirements
        if norm_projected_sq >= target_energy:
            k = 0
        else:
            k = np.argmax(total_accounted_variance >= target_energy) + 1

        current_ranks.append(k)

        if k == 0:
            # No new structural directions found; maintain current matrix
            continue

        # Extract left singular vectors spanning the newly discovered innovation space
        raw_basis_vectors = U[:, :k]  # Shape: (N_dim, k)

        # 4. SPARSE ENVELOPE INTERCEPTION (SparseGPM Variant Condition)
        if local_threshold >= 1.0:
            # Faithful GPM Baseline Path
            task_basis = raw_basis_vectors
        else:
            # Custom Heavy-Tailed Pruning Matrix Operations
            if xi > 0:
                norms = np.linalg.norm(R_hat, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                R_normalized = R_hat / norms

                cosine_similarity = R_normalized @ R_normalized.T
                cosine_distance = 1.0 - np.abs(cosine_similarity)
                cosine_distance = np.nan_to_num(cosine_distance, nan=0.0)

            sparse_basis_columns = []
            # Iterate through individual column basis vectors to identify local hubs
            for col_idx in range(k):
                v = raw_basis_vectors[:, col_idx]
                neuron_contributions = np.abs(v) ** 2

                sort_idx = np.argsort(neuron_contributions)[::-1]
                sorted_contribs = neuron_contributions[sort_idx]

                cum_neuron_variance = np.cumsum(sorted_contribs)
                num_neurons_kept = np.argmax(cum_neuron_variance >= local_threshold) + 1
                keep_indices = sort_idx[:num_neurons_kept]

                if xi > 0:
                    soft_mask = np.zeros(N_dim)
                    for hub in keep_indices:
                        d_from_hub = cosine_distance[hub, :]
                        envelope = (1.0 + d_from_hub / xi) ** (-1.0)
                        soft_mask = np.maximum(soft_mask, envelope)
                    v_processed = v * soft_mask
                else:
                    v_processed = np.zeros_like(v)
                    v_processed[keep_indices] = v[keep_indices]

                v_norm = np.linalg.norm(v_processed)
                if v_norm > 0:
                    v_processed /= v_norm
                sparse_basis_columns.append(v_processed)

            task_basis = np.stack(sparse_basis_columns, axis=1)

            # Re-orthogonalize modified entries via QR Decomposition
            if task_basis.shape[1] > 0:
                Q, R_qr = np.linalg.qr(task_basis)
                d_diag = np.diag(R_qr)
                ph = d_diag / (np.abs(d_diag) + 1e-12)
                task_basis = Q * ph

        # 5. LINE 21: DIRECT COLUMN APPENDING (NO JOINT RESAMPLING)
        if old_basis is None:
            feature_list[i] = task_basis
        else:
            if task_basis.shape[1] > 0:
                # Direct structural column concatenation as dictated by GPM paper
                feature_list[i] = np.concatenate((old_basis, task_basis), axis=1)

    # Convert memory registers back to runtime PyTorch tensors
    device = next(model.parameters()).device
    tensor_bases = [
        (
            torch.tensor(b, dtype=torch.float32, device=device)
            if b is not None and not torch.is_tensor(b)
            else b
        )
        for b in feature_list
    ]

    return tensor_bases, current_ranks


def update_HTGPM_bases(
    model,
    images,
    global_threshold,
    alpha=1.2,          # Tail index matching the heavy-tailed initialization
    eta=1e-3,           # Regularization parameter for Resolvent edge shifting
    feature_list=None,
    activation_fn=torch.tanh,
):
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    flattened_images = images.view(images.size(0), -1)
    all_inputs = [flattened_images]

    for h in list(reps.values())[:-1]:
        activated_h = activation_fn(h)
        all_inputs.append(activated_h.view(h.size(0), -1))

    if feature_list is None:
        feature_list = [None] * len(all_inputs)

    current_ranks = []

    for i, activation in enumerate(all_inputs):
        # 1. Shape Transformation: (Samples, N_dim) -> (N_dim, Samples)
        # R_l = [x_1, x_2, ..., x_ns] as formulated in classical GPM
        R = activation.cpu().numpy().T
        N_dim = R.shape[0]

        # Calculate total raw Frobenius norm variance of the incoming domain
        total_variance_sq = np.sum(R ** 2)
        if total_variance_sq < 1e-12:
            current_ranks.append(0)
            continue

        # 2. RESIDUAL PROJECTION LAYER ELIMINATION
        if feature_list[i] is not None:
            old_basis = feature_list[i]
            if torch.is_tensor(old_basis):
                old_basis = old_basis.cpu().numpy()

            # Project onto accumulated memory: R_proj = M @ (M.T @ R)
            R_proj = old_basis @ (old_basis.T @ R)
            # Isolate pure unexplained innovations
            R_hat = R - R_proj
            norm_projected_sq = np.sum(R_proj ** 2)
        else:
            R_hat = R
            old_basis = None
            norm_projected_sq = 0.0

        # 3. SVD ON ISOLATED RESIDUAL COMPONENT
        U, S, Vh = np.linalg.svd(R_hat, full_matrices=False)
        s_sq = S**2

        # Compute minimum rank 'k' accounting for historical compensation
        # ||R_proj||^2 + ||(R_hat)_k||^2 >= epsilon_th * ||R||^2
        cumulative_residual_variance = np.cumsum(s_sq)
        total_accounted_variance = norm_projected_sq + cumulative_residual_variance

        target_energy = global_threshold * total_variance_sq

        # Determine if old tasks already fully satisfy the threshold requirements
        if norm_projected_sq >= target_energy:
            k = 0
        else:
            k = np.argmax(total_accounted_variance >= target_energy) + 1

        current_ranks.append(k)

        if k == 0:
            # No new structural directions found; maintain current matrix states
            continue

        # Extract left singular vectors spanning the newly discovered innovation space
        raw_basis_vectors = U[:, :k]  # Shape: (N_dim, k)

        # =====================================================================
        # 4. HEAVY-TAILED RANDOM MATRIX THEORY OPERATOR ASSEMBLY (HTGPM)
        # =====================================================================

        # A. Construct the Empirical Matrix Operator and find its spectral edge
        A = R_hat @ R_hat.T
        eigenvalues = np.linalg.eigvalsh(A)
        lambda_max = eigenvalues[-1]

        # B. Compute the Complex-Shifted Resolvent Matrix: G(z) = (A - zI)^-1
        z = lambda_max + 1j * eta
        Resolvent = np.linalg.inv(A - z * np.eye(N_dim))
        resolvent_envelope_matrix = np.abs(Resolvent)

        # C. Transform Resolvent entries into a stable RMT Distance Topology (0 to 1)
        # Uses Cauchy-Schwarz style spatial normalization to eliminate magnitude biases
        diag_G = np.diagonal(resolvent_envelope_matrix)
        normalization_matrix = np.sqrt(np.outer(diag_G, diag_G))
        rmt_similarity = resolvent_envelope_matrix / (normalization_matrix + 1e-9)
        rmt_distance_matrix = 1.0 - rmt_similarity

        # D. Mask Processing Loop
        sparse_basis_columns = []
        xi_resolvent = alpha   # Localization length matches tail index
        eta_resolvent = alpha  # Algebraic decay matches stable density index

        for col_idx in range(k):
            v = raw_basis_vectors[:, col_idx]
            v = v / (np.linalg.norm(v) + 1e-12) # Strict vector normalization

            # Diagnostic Gatekeeper: Compute Inverse Participation Ratio (4th power)
            ipr_val = np.sum(v ** 4)
            # Safe boundary constraint: ceiling target calculation
            effective_hubs_count = int(np.ceil(1.0 / ipr_val))

            # Isolate physical hub locations using energy rank (2nd power)
            neuron_contributions = np.abs(v) ** 2
            sort_idx = np.argsort(neuron_contributions)[::-1]
            rmt_hub_indices = sort_idx[:effective_hubs_count]

            # Apply alpha-parameterized continuous Power-Law decay around hubs
            soft_mask = np.zeros(N_dim)
            for hub in rmt_hub_indices:
                d_from_hub = rmt_distance_matrix[hub, :]

                # Power-law profile matching heavy-tailed statistics
                envelope = (1.0 + d_from_hub / xi_resolvent) ** (-eta_resolvent)

                # Constructive structural masking over multiple anchors
                soft_mask = np.maximum(soft_mask, envelope)

            # Bounding enforcement
            if np.max(soft_mask) > 0:
                soft_mask /= np.max(soft_mask)

            # Intercept basis geometry using the RMT soft envelope
            v_processed = v * soft_mask

            # Restore unit normal orientation to maintain projection integrity
            v_norm = np.linalg.norm(v_processed)
            if v_norm > 0:
                v_processed /= v_norm

            sparse_basis_columns.append(v_processed)

        task_basis = np.stack(sparse_basis_columns, axis=1)

        # Re-orthogonalize modified entries via clean QR Decomposition
        if task_basis.shape[1] > 0:
            Q, R_qr = np.linalg.qr(task_basis)
            d_diag = np.diag(R_qr)
            ph = d_diag / (np.abs(d_diag) + 1e-12)
            task_basis = Q * ph

        # 5. DIRECT COLUMN APPENDING TO ACCUMULATED MEMORY
        if old_basis is None:
            feature_list[i] = task_basis
        else:
            if task_basis.shape[1] > 0:
                feature_list[i] = np.concatenate((old_basis, task_basis), axis=1)

    # Convert memory registers back to runtime PyTorch tensors
    device = next(model.parameters()).device
    tensor_bases = [
        (
            torch.tensor(b, dtype=torch.float32, device=device)
            if b is not None and not torch.is_tensor(b)
            else b
        )
        for b in feature_list
    ]

    return tensor_bases, current_ranks


def apply_SparseGPM_projection(linear_layers, feature_list):
    """Applies standard, clean orthogonal gradient projections.

    Because the feature_list contains hyper-sparse basis coordinates from the
    HT localization, this standard math automatically isolates parameter memory.
    """
    if feature_list is None:
        return

    with torch.no_grad():
        for i, layer in enumerate(linear_layers):
            if i >= len(feature_list) or feature_list[i] is None:
                continue

            grad = layer.weight.grad
            if grad is None:
                continue

            basis = feature_list[i]  # Shape: (In_Features, K)

            # Clean Standard GPM Math: G_parallel = G @ B @ B.T
            parallel_grad = torch.mm(torch.mm(grad, basis), basis.t())

            # G_projected = G - G_parallel
            projected_grad = grad - parallel_grad

            # In-place gradient update
            layer.weight.grad.copy_(projected_grad)
