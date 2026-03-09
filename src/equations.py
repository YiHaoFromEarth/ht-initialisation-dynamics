import torch

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
# These are essential for the Jacobian J = diag(phi'(h)) * W

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
    return 1.0 - torch.tanh(h)**2

def sigmoid_prime(h):
    """
    Derivative of Sigmoid: sigmoid(h) * (1 - sigmoid(h)).
    Maximum value of 0.25 at h=0.
    """
    s = torch.sigmoid(h)
    return s * (1.0 - s)

# --- 3. The Lévy "Energy" Metric ---

def levy_alpha_moment(x, alpha):
    """
    The empirical q_l parameter as defined in Qu et al. Equation [4].
    Calculates the alpha-th moment of the post-activation activity.
    """
    return torch.mean(torch.abs(x)**alpha)
