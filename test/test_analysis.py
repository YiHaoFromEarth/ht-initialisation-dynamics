import torch
import numpy as np
import pytest
from src import get_singular_values, marchenko_pastur_pdf

def test_singular_values_identity():
    """Verify singular values for a known matrix (Identity)."""
    # For an IxI identity matrix, all singular values are 1, so all eigenvalues (s^2) are 1 / M.
    size = 100
    identity = torch.eye(size)
    evs = get_singular_values(identity)

    assert np.allclose(evs, 1.0 / size), f"Eigenvalues of Identity matrix should all be 1/{size}"
    assert len(evs) == size

def test_singular_values_scaling():
    """Verify that scaling the matrix scales the eigenvalues by the square."""
    matrix = torch.randn(50, 50)
    evs_1 = get_singular_values(matrix)
    evs_2 = get_singular_values(matrix * 2.0)

    # (2W)(2W)^T = 4(WW^T) -> eigenvalues should be 4x larger
    assert np.allclose(evs_2, evs_1 * 4.0)

def test_marchenko_pastur_integration():
    """Verify the MP PDF integrates to (approximately) 1.0."""
    Q = 0.5
    sigma = 1.0
    x_range = np.linspace(0.001, 5.0, 10000)
    pdf, l_min, l_max = marchenko_pastur_pdf(x_range, Q, sigma)

    # Numerical integration using trapezoidal rule
    area = np.trapezoid(pdf, x_range)
    assert np.isclose(area, 1.0, atol=1e-2), f"MP PDF area should be 1.0, got {area}"

def test_rmt_convergence():
    """
    Physical Unit Test: Does a large Gaussian matrix match the MP curve?
    This bridges the gap between your code and RMT theory.
    """
    N, M = 1000, 2000 # Q = 0.5
    Q = N / M
    # Standard Gaussian entries with variance 1/M
    sigma = 1 / np.sqrt(M)
    matrix = torch.randn(N, M) * sigma

    evs = get_singular_values(matrix)

    # Check if the largest eigenvalue is near the theoretical Bulk Edge (lambda_plus)
    # lambda_plus = sigma^2 * (1 + sqrt(Q))^2
    lambda_plus_theory = (sigma**2) * (1 + np.sqrt(Q))**2

    # We allow a small margin for the Tracy-Widom fluctuation at the edge
    assert np.max(evs) < lambda_plus_theory * 1.2, "Largest eigenvalue significantly exceeded MP bulk"
    assert np.mean(evs) == pytest.approx(sigma**2, rel=0.1), "Mean eigenvalue should match entry variance"
