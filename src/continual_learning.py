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


def update_MSG_bases(model, images, k_max=30, window_size=5, gamma=2.0, alpha_target=1.2, feature_list=None):
    """
    Multifractal Singularity Gating (MSG) Base Optimizer.

    Args:
        model: The network model exposing 'get_pre_activations' and 'state_dict'.
        images: Batch of mini-batch images [B, C, H, W] to run the dynamic stress test.
        k_max: The signal horizon ceiling (Hyperparameter 1). Default 30.
        window_size: Width of sliding window for OLS calculus (Hyperparameter 2). Default 5.
        gamma: Exponential protection scaling coefficient (Hyperparameter 3). Default 2.0.
        alpha_target: Manifold anchor target matching our corridor boundary. Default 1.2.
        feature_list: List of stored continuous projection operators per layer.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        reps = model.get_pre_activations(images)
        weights = model.get_layer_weights()

    layer_keys = list(reps.keys())

    # Initialize list to hold raw NumPy operators during this boundary calculation step
    numpy_feature_list = [None] * len(layer_keys)

    current_ranks = []
    half_w = window_size // 2

    for i, key in enumerate(layer_keys):
        h = reps[key].detach()          # Shape: [B, D_out]
        W = weights[key].detach()        # Shape: [D_out, D_in]

        # 1. Analytic Jacobian calculation
        tanh_deriv = 1.0 - torch.tanh(h).pow(2)     # Shape: [B, D_out]
        mean_deriv = torch.mean(tanh_deriv, dim=0)   # Shape: [D_out]

        # J_batch maps sensitivity across input coordinates
        J_batch = mean_deriv.unsqueeze(1) * W       # Shape: [D_out, D_in]

        # 2. Decompose Jacobian operator
        _, S_jac, Vh_jac = torch.linalg.svd(J_batch, full_matrices=False)

        # Explicitly migrate spectral outputs to CPU for NumPy OLS processing
        S_vals = S_jac.cpu().numpy()
        V = Vh_jac.cpu().numpy().T  # Shape: [D_in, Rank]

        actual_k_max = min(k_max, len(S_vals) - half_w - 1)
        current_ranks.append(actual_k_max)

        # 3. Discrete log-log OLS calculus loop
        log_k = np.log(np.arange(1, len(S_vals) + 1))
        log_S = np.log(S_vals)
        phi = np.zeros(V.shape[1])

        for k_idx in range(half_w, actual_k_max):
            x_win = log_k[k_idx - half_w : k_idx + half_w + 1]
            y_win = log_S[k_idx - half_w : k_idx + half_w + 1]

            slope, _ = np.polyfit(x_win, y_win, 1)
            alpha_local = -slope

            # Continuous Protection Mapping
            phi[k_idx] = np.exp(-gamma * max(0.0, alpha_local - alpha_target))

        # Guarantee safety for dominant structural skyscrapers at the head
        phi[:half_w] = 1.0

        # Add this temporary diagnostic print right before 'task_operator = (V * phi) @ V.T'
        if k_idx == actual_k_max - 1:
            print(f"Layer {key} | Raw Slopes sample: {alpha_local:.2f} | Phi sample: {phi[k_idx]:.4f}")
            print(f"Layer {key} | Non-zero Phi components: {np.sum(phi > 0.01)} out of {len(phi)}")

        # 4. Construct continuous soft gating projection matrix operator
        task_operator = (V * phi) @ V.T

        # 5. Envelope consolidation rule with type safety
        if feature_list is None or feature_list[i] is None:
            numpy_feature_list[i] = task_operator
        else:
            # Explicitly bring the existing GPU tensor back to host memory before envelope merge
            old_operator = feature_list[i].detach().cpu().numpy()
            numpy_feature_list[i] = np.maximum(old_operator, task_operator)

    # Re-wrap directly to model tensor parameters device destination
    tensor_operators = [
        torch.tensor(op, dtype=torch.float32, device=device) if op is not None else None
        for op in numpy_feature_list
    ]

    return tensor_operators, current_ranks


def apply_MSG_projection(linear_layers, feature_list):
    """
    Applies continuous, multifractal soft gating to the gradients.

    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: Pre-computed continuous protection operators P from update_MSG_bases.
                      Each operator has shape [D_in, D_in].
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

            # P is the pre-loaded GPU tensor operator matrix [D_in, D_in]
            P = feature_list[i]

            # Attenuate the gradient components along protected axes:
            # grad has shape [D_out, D_in], P has shape [D_in, D_in]
            gated_grad = grad - torch.mm(grad, P)

            # Update the gradient in-place
            layer.weight.grad.copy_(gated_grad)
