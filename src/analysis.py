import torch
import numpy as np

def get_singular_values(matrix):
    """
    Extracts singular values and converts to normalized squared eigenvalues
    for RMT analysis (λ = s^2 / M).
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    # Use SVD values for numerical stability
    N, M = matrix.shape
    s = torch.linalg.svdvals(matrix.float())
    # RMT usually analyzes the eigenvalues of the covariance matrix WW^T
    eigenvalues = s**2 / M  # Normalize by M for RMT scaling
    return eigenvalues.detach().cpu().numpy()

def marchenko_pastur_pdf(x, Q, sigma=1.0):
    """
    Theoretical Marchenko-Pastur Density.
    Q = N/M (Aspect ratio)
    sigma = variance of the entries
    """
    # Boundary points of the MP distribution
    lambda_plus = (sigma**2) * (1 + np.sqrt(Q))**2
    lambda_minus = (sigma**2) * (1 - np.sqrt(Q))**2

    # PDF Calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        pdf = (1/(2 * np.pi * sigma**2 * Q * x)) * np.sqrt(
            np.maximum(0, (lambda_plus - x) * (x - lambda_minus))
        )
    return np.nan_to_num(pdf), lambda_minus, lambda_plus
