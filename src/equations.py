import torch
import numpy as np

# --- 1. Activation Functions (phi) ---


def relu(h):
    """Standard Rectified Linear Unit."""
    return torch.relu(h)


def tanh(h):
    """Hyperbolic Tangent - common in Qu et al. for signal centering."""
    return torch.tanh(h)


def sigmoid(h):
    """Sigmoid activation."""
    return torch.sigmoid(h)


# --- 2. Activation Derivatives (phi') ---


def relu_prime(h):
    """
    Derivative of ReLU.
    Returns 1.0 for h > 0, and 0.0 otherwise.
    """
    return (h > 0).float()


def tanh_prime(h):
    """
    Derivative of Tanh: 1 - tanh(h)^2.
    Maximum value of 1.0 at h=0 (the linear regime).
    """
    return 1.0 - torch.tanh(h) ** 2


def sigmoid_prime(h):
    """
    Derivative of Sigmoid: sigmoid(h) * (1 - sigmoid(h)).
    Maximum value of 0.25 at h=0.
    """
    s = torch.sigmoid(h)
    return s * (1.0 - s)


# --- 3. Analysis Functions ---

def hill_estimator(W, k_percent=0.01):
    if torch.is_tensor(W):
        W = W.to(torch.float32).numpy().flatten()

    # Take absolute values and sort descending
    w_sorted = np.sort(np.abs(W))[::-1]

    # Select top k entries
    k = int(len(w_sorted) * k_percent)
    if k < 2:
        return np.nan

    # Hill formula
    log_diffs = np.log(w_sorted[:k] / w_sorted[k])
    xi = np.mean(log_diffs)

    return 1 / xi if xi > 0 else np.inf

def frobenius_norm(W):
    return torch.linalg.norm(W).item()

def relative_frobenius_norm(W, W0):
    return frobenius_norm(W - W0) / frobenius_norm(W0)

def spectral_norm(W):
    return torch.linalg.matrix_norm(W, ord=2).item()

def spectral_gap(W):
    singular_values = torch.linalg.svdvals(W)
    spectral_gap = singular_values[0] / singular_values[1]
    return spectral_gap.item()

def layerwise_cosine(W, W0):
    inner_product = torch.sum(W * W0).item()
    return 1 - (inner_product / (frobenius_norm(W) * frobenius_norm(W0)))

def stable_rank(W):
    # Frobenius norm squared
    fro_norm_sq = frobenius_norm(W)**2
    # Spectral norm (largest singular value) squared
    spectral_norm_sq = torch.linalg.norm(W, ord=2)**2

    return (fro_norm_sq / spectral_norm_sq).item()

def levy_alpha_moment(x, alpha):
    """
    The empirical q_l parameter as defined in Qu et al. Equation [4].
    Calculates the alpha-th moment of the post-activation activity.
    """
    return torch.mean(torch.abs(x) ** alpha)
