import torch
import torch.nn as nn
import numpy as np
import pytest
import os
import yaml
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src import (
    get_singular_values,
    extract_rmt_parameters,
    gaussian_broadening,
    fit_mp_to_density,
    get_layer_fingerprint,
    evaluate_spectral_perturbation,
    get_layer_from_checkpoint,
    run_spectral_analysis
)

# --- 1. A Minimal Model for Testing ---
class ToyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            # Class 0 expert is 10x stronger than Class 1 expert
            self.fc.weight.copy_(torch.tensor([[10.0, 0.0],
                                              [0.0, 1.0]]))
    def forward(self, x):
        return self.fc(x)

def test_singular_values_identity():
    """
    Verify raw singular values (nu) for an Identity matrix.
    Per PhysRevE 106 054124, we analyze nu directly, not s^2/M.
    """
    size = 100
    identity = torch.eye(size)

    # Extract nu using the updated function
    nu = get_singular_values(identity)

    # 1. Identity singular values are exactly 1.0
    assert np.allclose(nu, 1.0), "Singular values of Identity should be 1.0"

    # 2. Dimensions must match the smaller axis (m)
    assert len(nu) == size, f"Expected {size} singular values"

def test_extract_rmt_parameters_toy():
    """
    Verify extraction of n, m, and sigma_tilde from a controlled toy matrix.
    """
    # Create a 200x100 matrix for clear n, m distinction
    # Using a constant value to make variance calculation predictable
    # Var(0.1) = 0.0, so we add a bit of noise
    n_true, m_true = 200, 100
    toy_matrix = torch.randn(n_true, m_true) * 0.05

    n, m, sigma_tilde = extract_rmt_parameters(toy_matrix)

    # 1. Check Dimensions (n is max, m is min)
    assert n == n_true, f"Expected n={n_true}, got {n}"
    assert m == m_true, f"Expected m={m_true}, got {m}"

    # 2. Check Sigma Tilde Scaling
    # sigma_tilde = std(W) * sqrt(n)
    expected_std = toy_matrix.std().item()
    expected_sigma_tilde = expected_std * np.sqrt(n_true)

    assert np.isclose(sigma_tilde, expected_sigma_tilde), \
        f"Sigma_tilde {sigma_tilde} does not match expected {expected_sigma_tilde}"

def test_gaussian_broadening_sum():
    """
    Verify that the broadened PDF integrates approximately to 1.
    """
    nu = np.random.normal(loc=1.0, scale=0.1, size=100)
    x_range = np.linspace(0, 2, 1000)
    dx = x_range[1] - x_range[0]

    pdf = gaussian_broadening(nu, x_range, a=5)
    total_area = np.trapz(pdf, x_range)

    # Area should be close to 1.0 (probability distribution) [cite: 227]
    assert np.isclose(total_area, 1.0, atol=1e-2), f"PDF area is {total_area}, expected 1.0"

def test_mp_fitting_and_outlier_detection():
    """
    Verify that the solver recovers the correct nu_max and identifies
    injected outliers (learned signal).
    """
    # 1. Setup Synthetic Bulk (N=1000, M=500, sigma_tilde=1.5)
    n, m = 1000, 500
    sigma_tilde_true = 1.5
    Q = n / m
    nu_min_theory = sigma_tilde_true * (1 - np.sqrt(1/Q))
    nu_max_theory = sigma_tilde_true * (1 + np.sqrt(1/Q))

    # Generate random bulk + 5 clear outliers
    bulk = np.random.uniform(nu_min_theory, nu_max_theory, 995)
    outliers = np.array([nu_max_theory + 2.0, nu_max_theory + 3.0,
                         nu_max_theory + 4.0, nu_max_theory + 5.0,
                         nu_max_theory + 6.0])
    nu_synthetic = np.concatenate([bulk, outliers])

    # 2. Broaden the spectrum to create the fitting target
    x_range = np.linspace(0, nu_synthetic.max() + 1, 2000)
    empirical_pdf = gaussian_broadening(nu_synthetic, x_range, a=15)

    # 3. Solve for RMT Parameters
    v_max_fit, s_tilde_fit = fit_mp_to_density(x_range, empirical_pdf, nu_synthetic)

    # 4. Assertions
    # The fitted nu_max should be close to the theoretical boundary
    assert np.isclose(v_max_fit, nu_max_theory, atol=0.2), \
        f"Fitted nu_max {v_max_fit} deviated too far from theory {nu_max_theory}"

    # Count outliers beyond the fitted bulk
    detected_outliers = np.sum(nu_synthetic > v_max_fit)

    # We expect to find at least our 5 injected outliers
    # (Some high-end bulk values might occasionally drift over)
    assert detected_outliers >= 5, f"Failed to detect all 5 outliers, found {detected_outliers}"
    print(f"Success: Detected {detected_outliers} singular values as 'Learned Information'")

def test_fingerprint_logic():
    # --- 1. Define Toy Matrices ---
    # W_0: Simple Identity (Uniform)
    W_0 = torch.tensor([[1.0, 0.0],
                        [0.0, 1.0]])

    # W_t: Scaled and "Spiky" (Concentrated)
    # We move the first element and keep the second at 1.0
    W_t = torch.tensor([[2.0, 0.0],
                        [0.0, 1.0]])

    # --- 2. Manual Ground Truth Calculations ---
    # S_t will be [2.0, 1.0] | V_t[:, 0] will be [1.0, 0.0]

    # Displacement: sqrt((2-1)^2 + 0 + 0 + (1-1)^2) / sqrt(1^2 + 1^2) = 1 / sqrt(2)
    expected_displacement = 1.0 / np.sqrt(2.0)

    # Effective Rank: (2^2 + 1^2) / (2^2) = 5 / 4 = 1.25
    expected_eff_rank = 1.25

    # IPR of [1, 0]: (1^4 + 0^4) / (1^2 + 0^2)^2 = 1 / 1 = 1.0 (Perfectly Spiky)
    expected_ipr = 1.0

    # Participation Ratio: (2 + 1)^2 / (2 * (2^2 + 1^2)) = 9 / (2 * 5) = 0.9
    expected_part_ratio = 0.9

    # Max Singular Val: 2.0
    expected_max_s = 2.0

    # --- 3. Execute Function ---
    results = get_layer_fingerprint(W_0, W_t)

    # --- 4. Assertions ---
    tol = 1e-6
    try:
        assert abs(results['displacement'] - expected_displacement) < tol
        assert abs(results['effective_rank'] - expected_eff_rank) < tol
        assert abs(results['ipr'] - expected_ipr) < tol
        assert abs(results['participation_ratio'] - expected_part_ratio) < tol
        assert abs(results['max_singular_val'] - expected_max_s) < tol
        print("ALL TESTS PASSED: Metric logic is mathematically sound.")
    except AssertionError as e:
        print("TEST FAILED: Discrepancy found in metric calculations.")
        for k, v in results.items():
            print(f"  {k}: {v}")
        raise e

def test_rigorous_perturbation():
    # 1. Setup Data with "Tie-Breaker" noise
    # We add 0.1 to the 'wrong' dimension so if the expert is ablated,
    # the argmax will definitely flip to the wrong class.
    x_test = torch.tensor([
        [1.0, 0.1], # Strong signal for Class 0, tiny leak for Class 1
        [0.1, 1.0]  # Strong signal for Class 1, tiny leak for Class 0
    ])
    y_test = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x_test, y_test), batch_size=2)
    model = ToyLinear()

    # 2. Baseline Check (Must be 100%)
    # Sample 0: [10*1, 1*0.1] = [10.0, 0.1] -> Correct (0)
    # Sample 1: [10*0.1, 1*1] = [1.0, 1.0] -> TIE!
    # Let's adjust weights slightly to ensure baseline is perfect.
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[10.0, 0.0], [0.0, 5.0]]))

    # 3. Test ABLATE k=1 (Remove the 10.0 expert)
    # W becomes [[0, 0], [0, 5]].
    # Sample 0: [0, 5*0.1] = [0.0, 0.5] -> WRONG (Argmax 1 instead of 0)
    res_ablate = evaluate_spectral_perturbation(model, loader, 'fc.weight', k=1, mode='ablate')

    # 4. Test RANK-K k=1 (Keep ONLY the 10.0 expert)
    # W becomes [[10, 0], [0, 0]].
    # Sample 1: [10*0.1, 0] = [1.0, 0.0] -> WRONG (Argmax 0 instead of 1)
    res_rank = evaluate_spectral_perturbation(model, loader, 'fc.weight', k=1, mode='rank-k')

    # 5. Assertions
    print(f"Rigorous Test Results:")
    print(f"  Ablate k=1 Acc: {res_ablate['accuracy']:.2f} (Expected 0.50)")
    print(f"  Rank-1 k=1 Acc: {res_rank['accuracy']:.2f} (Expected 0.50)")

    assert res_ablate['accuracy'] == 0.5, "Ablate k=1 should destroy exactly one class prediction."
    assert res_rank['accuracy'] == 0.5, "Rank-k=1 should destroy exactly one class prediction."

    # Final check: is original model still [10, 5]?
    assert model.fc.weight[0,0] == 10.0, "Original model was mutated!"
    print("RIGOROUS TEST PASSED: SVD perturbation is mathematically deterministic.")

def test_extraction_io():
    test_path = Path("temp_test_model.pth")
    layer_name = "test_layer.weight"
    dummy_weights = torch.randn(5, 5)

    # Simulate the ml_library.py nested save format
    torch.save({'model_state': {layer_name: dummy_weights}}, test_path)

    try:
        extracted = get_layer_from_checkpoint(test_path, layer_name)

        # Assertions
        assert isinstance(extracted, torch.Tensor), "Output must be a torch.Tensor"
        assert extracted.shape == (5, 5), "Extracted shape mismatch"
        assert torch.equal(extracted, dummy_weights), "Data corruption during extraction"
        print("Extraction IO test passed.")

    finally:
        if test_path.exists():
            os.remove(test_path)

def test_wrapper_integration():
    run_dir = Path("temp_run_dir")
    run_dir.mkdir(exist_ok=True)
    config_path = run_dir / "config.yaml"

    # 1. Create a minimal mock config matching load_master_config requirements
    mock_config = {
        'model': {'class_name': 'GeneralMLP', 'params': {'kwargs': {'input_size': 784, 'hidden_size': 32, 'num_classes': 10, 'depth': 2}}},
        'optimizer': {'class_name': 'SGD', 'params': {'lr': 0.01}},
        'data': {'dataset_name': 'MNIST', 'batch_size': 2},
        'hyperparams': {'epochs': 1}
    }

    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)

    # 2. Create dummy checkpoint
    # We use a real state dict from a dummy model to ensure keys match
    from src import GeneralMLP
    dummy_model = GeneralMLP(input_size=784, hidden_size=32, num_classes=10, depth=2)
    torch.save({'model_state': dummy_model.state_dict()}, run_dir / "final_model.pth")

    try:
        # 3. Run analysis for k=0 and k=1
        # We only care that it executes and returns the list of dicts
        k_vals = [0, 1]
        results = run_spectral_analysis(run_dir, config_path, 'features.0.weight', k_vals, mode='ablate')

        # Assertions
        assert isinstance(results, list), "Output must be a list"
        assert len(results) == len(k_vals), "Result length mismatch"
        assert all(isinstance(r, dict) for r in results), "Results must be dictionaries"
        assert 'accuracy' in results[0], "Dictionary missing accuracy key"

        print("Wrapper integration test passed.")

    finally:
        # Cleanup
        for file in run_dir.glob("*"):
            os.remove(file)
        run_dir.rmdir()
