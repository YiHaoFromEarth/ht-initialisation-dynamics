import torch
import numpy as np
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


def update_GPM_bases(model, images, threshold, feature_list=None):
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    all_inputs = [images]
    for h in list(reps.values())[:-1]:
        all_inputs.append(torch.tanh(h))

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


def apply_GPM_projection(linear_layers, feature_list):
    """
    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: The pre-loaded GPU tensors from update_GPM_bases.
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

            basis = feature_list[i]  # Already a GPU tensor!

            # Math: g_proj = g - (g @ B @ B.T)
            proj_grad = grad - torch.mm(torch.mm(grad, basis), basis.t())
            layer.weight.grad.copy_(proj_grad)
