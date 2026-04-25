import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import pytest
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.utils import get_layer_from_checkpoint
from src.analysis import (
    ModelTracker,
    get_singular_values,
    evaluate_spectral_perturbation,
    run_spectral_analysis,
)
from src.equations import hill_estimator


# --- 1. A Minimal Model for Testing ---
class ToyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            # Class 0 expert is 10x stronger than Class 1 expert
            self.fc.weight.copy_(torch.tensor([[10.0, 0.0], [0.0, 1.0]]))

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


def test_rigorous_perturbation():
    # 1. Setup Data with "Tie-Breaker" noise
    # We add 0.1 to the 'wrong' dimension so if the expert is ablated,
    # the argmax will definitely flip to the wrong class.
    x_test = torch.tensor(
        [
            [1.0, 0.1],  # Strong signal for Class 0, tiny leak for Class 1
            [0.1, 1.0],  # Strong signal for Class 1, tiny leak for Class 0
        ]
    )
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
    res_ablate = evaluate_spectral_perturbation(
        model, loader, "fc.weight", k=1, mode="ablate"
    )

    # 4. Test RANK-K k=1 (Keep ONLY the 10.0 expert)
    # W becomes [[10, 0], [0, 0]].
    # Sample 1: [10*0.1, 0] = [1.0, 0.0] -> WRONG (Argmax 0 instead of 1)
    res_rank = evaluate_spectral_perturbation(
        model, loader, "fc.weight", k=1, mode="rank-k"
    )

    # 5. Assertions
    print(f"Rigorous Test Results:")
    print(f"  Ablate k=1 Acc: {res_ablate['accuracy']:.2f} (Expected 0.50)")
    print(f"  Rank-1 k=1 Acc: {res_rank['accuracy']:.2f} (Expected 0.50)")

    assert res_ablate["accuracy"] == 0.5, (
        "Ablate k=1 should destroy exactly one class prediction."
    )
    assert res_rank["accuracy"] == 0.5, (
        "Rank-k=1 should destroy exactly one class prediction."
    )

    # Final check: is original model still [10, 5]?
    assert model.fc.weight[0, 0] == 10.0, "Original model was mutated!"
    print("RIGOROUS TEST PASSED: SVD perturbation is mathematically deterministic.")


def test_extraction_io():
    test_path = Path("temp_test_model.pth")
    layer_name = "test_layer.weight"
    dummy_weights = torch.randn(5, 5)

    # Simulate the ml_library.py nested save format
    torch.save({"model_state": {layer_name: dummy_weights}}, test_path)

    try:
        extracted = get_layer_from_checkpoint(test_path, layer_name)

        # Assertions
        assert isinstance(extracted, torch.Tensor), "Output must be a torch.Tensor"
        assert extracted.shape == (5, 5), "Extracted shape mismatch"
        assert torch.equal(extracted, dummy_weights), (
            "Data corruption during extraction"
        )
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
        "model": {
            "class_name": "GeneralMLP",
            "params": {
                "kwargs": {
                    "input_size": 784,
                    "hidden_size": 32,
                    "num_classes": 10,
                    "depth": 2,
                }
            },
        },
        "optimizer": {"class_name": "SGD", "params": {"lr": 0.01}},
        "data": {"dataset_name": "MNIST", "batch_size": 2},
        "hyperparams": {"epochs": 1},
    }

    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)

    # 2. Create dummy checkpoint
    # We use a real state dict from a dummy model to ensure keys match
    from src.architectures import GeneralMLP

    dummy_model = GeneralMLP(input_size=784, hidden_size=32, num_classes=10, depth=2)
    torch.save({"model_state": dummy_model.state_dict()}, run_dir / "final_model.pth")

    try:
        # 3. Run analysis for k=0 and k=1
        # We only care that it executes and returns the list of dicts
        k_vals = [0, 1]
        results = run_spectral_analysis(
            run_dir, config_path, "features.0.weight", k_vals, mode="ablate"
        )

        # Assertions
        assert isinstance(results, list), "Output must be a list"
        assert len(results) == len(k_vals), "Result length mismatch"
        assert all(isinstance(r, dict) for r in results), "Results must be dictionaries"
        assert "accuracy" in results[0], "Dictionary missing accuracy key"

        print("Wrapper integration test passed.")

    finally:
        # Cleanup
        for file in run_dir.glob("*"):
            os.remove(file)
        run_dir.rmdir()


def test_hill_toy_example():
    """Test using the toy list we manually calculated."""
    # Data: [0.1, 0.2, 0.5, 0.8, 1.2, 2.5, 5.0, 12.0, 45.0, 150.0]
    # Sorted: 150, 45, 12, 5, ...
    # k=3 (top 10%), threshold = 5
    toy_weights = [0.1, 0.2, 0.5, 0.8, 1.2, 2.5, 5.0, 12.0, 45.0, 150.0]

    # k_percent=0.3 to force k=3
    alpha = hill_estimator(toy_weights, k_percent=0.3)

    # Manual calc: 1/((ln(30)+ln(9)+ln(2.4))/3) approx 0.463
    assert alpha == pytest.approx(0.463, rel=1e-3)


def test_hill_pareto_distribution():
    """Test with a true Power Law (Pareto) where alpha is known."""
    np.random.seed(42)
    true_alpha = 1.5
    # Pareto distribution in numpy: samples = (np.random.pareto(a) + 1) * scale
    # The 'a' in numpy's pareto is the same as our tail index alpha
    samples = np.random.pareto(true_alpha, size=10000) + 1

    # For a large sample, 1% tail should be quite accurate
    estimated_alpha = hill_estimator(samples, k_percent=0.01)

    # Allow 10% relative error due to stochastic sampling
    assert estimated_alpha == pytest.approx(true_alpha, rel=0.1)


def test_hill_normal_distribution():
    """Test that Gaussian weights result in a high alpha (thin tails)."""
    np.random.seed(42)
    # Normal distribution has exponential tails, so Hill alpha should be large
    samples = np.random.normal(0, 1, size=10000)

    estimated_alpha = hill_estimator(samples, k_percent=0.01)

    # Alpha for a Normal distribution usually estimates > 2 or 3
    # as k -> 0, signifying it is not heavy-tailed.
    assert estimated_alpha > 2.0


def test_model_tracker_math_accuracy():
    """
    Unit test to verify that ModelTracker correctly calculates:
    1. Net Drift (Mean of diff)
    2. L1 Distance (Mean of abs diff)
    3. L2 Distance (RMS of diff)
    """

    # 1. Setup a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Two parameters of size 4 to make mental math easy
            self.param1 = nn.Parameter(torch.zeros(4))
            self.param2 = nn.Parameter(torch.ones(4))

    model = DummyModel()
    lags = [1, 2]
    tracker = ModelTracker(model, lags=lags)

    # 2. Define controlled updates (History Step 0: [0,0,0,0])
    tracker.update(model)

    # 3. Update 1: Move param1 by a specific vector [1, -1, 2, -2]
    # Sum = 0 (Net Drift should be 0)
    # Abs Sum = 6 (L1 should be 6/4 = 1.5)
    # Sq Sum = 1+1+4+4 = 10 (L2 should be sqrt(10/4) = sqrt(2.5) approx 1.581)
    with torch.no_grad():
        model.param1.add_(torch.tensor([1.0, -1.0, 2.0, -2.0]))

    tracker.update(model)  # This triggers tau=1 calculation

    # 4. Verify Tau=1 Results for param1
    param1_data = tracker.raw_data["param1"][1][0]
    net, l1, l2 = param1_data

    assert pytest.approx(net, abs=1e-6) == 0.0, (
        "Net Drift should be 0 for symmetric movement"
    )
    assert pytest.approx(l1, abs=1e-6) == 1.5, "L1 Distance calculation failed"
    assert pytest.approx(l2, abs=1e-6) == np.sqrt(2.5), (
        "L2/RMS Energy calculation failed"
    )

    # 5. Update 2: Move param1 again by [2, 2, 2, 2]
    # Current state is [3, 1, 4, 0]
    # Past state (tau=2) was [0, 0, 0, 0]
    # Diff for tau=2 is [3, 1, 4, 0]
    # Net: (3+1+4+0)/4 = 2.0
    # L1: (3+1+4+0)/4 = 2.0
    # L2: sqrt((9+1+16+0)/4) = sqrt(6.5) approx 2.549
    with torch.no_grad():
        model.param1.add_(torch.tensor([2.0, 2.0, 2.0, 2.0]))

    tracker.update(model)  # This triggers tau=1 and tau=2 calculation

    # Verify Tau=2 Results
    net_t2, l1_t2, l2_t2 = tracker.raw_data["param1"][2][0]
    assert pytest.approx(net_t2, abs=1e-6) == 2.0
    assert pytest.approx(l1_t2, abs=1e-6) == 2.0
    assert pytest.approx(l2_t2, abs=1e-6) == np.sqrt(6.5)


def test_dataframe_conversion_logic():
    """Verifies that the scale and step centering logic is consistent."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    model = SimpleModel()
    tracker = ModelTracker(model, lags=[1])

    # Add two updates
    tracker.update(model)
    model.p.data += 1.0
    tracker.update(model)

    # scale=100 (e.g., 100 steps per epoch)
    df = tracker.to_dataframe(alpha=1.2, sigma=0.25, scale=100)

    # Check if the step centering works: (0 + 1 - 0.5) * 100 = 50
    assert df["step"].iloc[0] == 50
    assert df["time_lag"].iloc[0] == 100
    assert df["alpha"].iloc[0] == 1.2


if __name__ == "__main__":
    # If running manually without pytest command
    pytest.main([__file__])
