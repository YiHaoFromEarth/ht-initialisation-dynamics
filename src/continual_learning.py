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


def apply_GPM_projection(linear_layers, feature_list, alpha=0.01, use_gating=True):
    """
    Applies an asymmetric directional gate to Leaky GPM.
    If the parallel gradient is cooperative (Frobenius inner product > 0), allows alpha leakage.
    If destructive (inner product <= 0), clamps leakage to 0.0 to prevent memory erasure.

    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: The pre-loaded GPU tensors of basis matrices.
        alpha: Maximum permitted leakage factor for cooperative steps.
        use_gating: Boolean flag to toggle the directional gating protection.
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

                # If the gradient points AGAINST the weights, it's destructive -> kill the leak
                if dot_product <= 0:
                    effective_alpha = 0.0

            # 4. Recombine components using the gated alpha value
            gated_grad = ortho_grad + effective_alpha * parallel_grad

            # Update the gradient in-place
            layer.weight.grad.copy_(gated_grad)
