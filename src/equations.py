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


def mcculloch_estimator(W):
    """
    Refined McCulloch estimator for a single weight matrix.
    Operates natively in PyTorch for speed and to avoid alpha > 2.0.
    """
    if torch.is_tensor(W):
        W = W.detach().float().flatten()
    else:
        W = torch.from_numpy(W).float().flatten()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = W.to(device)

    # 1. Calculate quantiles
    q = torch.quantile(W, torch.tensor([0.05, 0.25, 0.75, 0.95], device=device))

    v0_5 = q[3] - q[0]  # 95th - 5th (Full width)
    v0_25 = q[2] - q[1] # 75th - 25th (Interquartile range)

    if v0_5 == 0: return 2.0

    # 2. Dispersion Ratio
    nu = v0_25 / v0_5

    # 3. CORRECTED Reference Table for Symmetric Stable Distributions
    # These values map (nu) -> alpha for the ratio (q75-q25)/(q95-q05)
    nu_ref = torch.tensor([
        0.042, 0.067, 0.095, 0.126, 0.158, 0.191, 0.225,
        0.257, 0.289, 0.317, 0.343, 0.365, 0.383, 0.398, 0.410
    ], device=device)

    alpha_ref = torch.tensor([
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    ], device=device)

    # 4. Interpolation logic
    nu = torch.clamp(nu, min=nu_ref[0], max=nu_ref[-1])
    idx = torch.searchsorted(nu_ref, nu)
    idx = torch.clamp(idx, 1, len(nu_ref) - 1)

    x0, x1 = nu_ref[idx-1], nu_ref[idx]
    y0, y1 = alpha_ref[idx-1], alpha_ref[idx]

    alpha = y0 + (nu - x0) * (y1 - y0) / (x1 - x0)

    return alpha.item()


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
    fro_norm_sq = frobenius_norm(W) ** 2
    # Spectral norm (largest singular value) squared
    spectral_norm_sq = torch.linalg.norm(W, ord=2) ** 2

    return (fro_norm_sq / spectral_norm_sq).item()


def levy_alpha_moment(x, alpha):
    """
    The empirical q_l parameter as defined in Qu et al. Equation [4].
    Calculates the alpha-th moment of the post-activation activity.
    """
    return torch.mean(torch.abs(x) ** alpha)
