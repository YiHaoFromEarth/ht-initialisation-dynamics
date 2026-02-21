import torch
import numpy as np
from scipy.stats import levy_stable
from torch import nn
from src import init_heavy_tailed, apply_heavy_tailed_init  # Adjust path as needed

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.fc = nn.Linear(10, 5)
        self.nested = nn.Sequential(
            nn.Linear(5, 2)
        )
        self.non_param = nn.Tanh() # Should be ignored by init

    def forward(self, x):
        return self.nested(self.fc(self.conv(x)))

def test_alpha_fit():
    """Verify that the generated weights converge to the requested alpha."""
    alpha_req = 1.2
    g_req = 0.5
    # Use a large tensor for better statistical fitting
    tensor = torch.empty(500, 500)

    init_heavy_tailed(tensor, alpha=alpha_req, g=g_req, base_seed=42)

    # ...but DOWNSAMPLE for the FIT logic.
    # 5,000 to 10,000 samples is plenty for a statistical check.
    data = tensor.numpy().flatten()
    downsampled_data = np.random.choice(data, size=5000, replace=False)

    # We fix beta (f1) to 0 and location (f2) to 0.
    # Some scipy versions prefer keyword arguments like floc and fscale.
    # This syntax fixes beta=0 and loc=0.
    alpha_fit, beta_fit, loc_fit, scale_fit = levy_stable.fit(downsampled_data, fbeta=0, floc=0)

    # Assert alpha is within a reasonable tolerance (stochastic margin)
    assert np.isclose(alpha_fit, alpha_req, atol=0.05), f"Alpha fit {alpha_fit} deviated from {alpha_req}"
    print(f"\nRequested alpha: {alpha_req}, Fitted alpha: {alpha_fit:.3f}")

    n_eff = (tensor.shape[0] * tensor.shape[1])**0.5
    expected_scale = g_req / (2 * n_eff)**(1/alpha_req)

    # Assert g is within a reasonable tolerance (stochastic margin)
    assert np.isclose(scale_fit, expected_scale, rtol=0.1), f"Scale fit {scale_fit} deviated from expected {expected_scale}"
    print(f"Expected scale: {expected_scale:.4f}, Fitted scale: {scale_fit:.4f}")

def test_scaling_logic():
    """Verify that the scale parameter adjusts correctly for different layer sizes."""
    alpha = 1.5
    g = 1.0

    t_small = torch.empty(10, 10)
    t_large = torch.empty(100, 100)

    init_heavy_tailed(t_small, alpha, g, base_seed=1)
    init_heavy_tailed(t_large, alpha, g, base_seed=1)

    std_small = t_small.std().item()
    std_large = t_large.std().item()

    # In HT distributions, larger N_eff leads to a smaller scale (1/N_eff^(1/alpha))
    # Therefore, the standard deviation (or dispersion) should be smaller for larger tensors
    assert std_large < std_small, "Scaling logic failed: larger layers should have smaller per-entry scales."
    print(f"\nSmall tensor std: {std_small:.3f}, Large tensor std: {std_large:.3f}")

def test_reproducibility_and_offset():
    """Ensure that the same seed/offset produces the same weights, and different ones don't."""
    t1 = torch.empty(50, 50)
    t2 = torch.empty(50, 50)
    t3 = torch.empty(50, 50)

    # Same seed, same offset
    init_heavy_tailed(t1, 1.2, 0.5, seed_offset=10, base_seed=100)
    init_heavy_tailed(t2, 1.2, 0.5, seed_offset=10, base_seed=100)
    # Same seed, different offset
    init_heavy_tailed(t3, 1.2, 0.5, seed_offset=11, base_seed=100)

    assert torch.equal(t1, t2), "Identical seeds failed to produce identical weights."
    assert not torch.equal(t1, t3), "Different seed_offsets produced identical weights (collision)."

def test_apply_init_coverage():
    """Verify that all parametric layers are modified from their default init."""
    model = ToyModel()

    # Capture initial weights
    original_conv = model.conv.weight.clone()
    original_fc = model.fc.weight.clone()
    original_nested = model.nested[0].weight.clone()

    # Apply HT Init
    apply_heavy_tailed_init(model, alpha=1.2, g=0.5, base_seed=123)

    # Assert that weights have changed
    assert not torch.equal(model.conv.weight, original_conv), "Conv layer was not initialized."
    assert not torch.equal(model.fc.weight, original_fc), "FC layer was not initialized."
    assert not torch.equal(model.nested[0].weight, original_nested), "Nested Linear layer was not initialized."

def test_layer_uniqueness():
    """Verify that different layers get different weight values (seed_offset check)."""
    model = ToyModel()
    # Add two identical layers
    model.layer_a = nn.Linear(10, 10)
    model.layer_b = nn.Linear(10, 10)

    apply_heavy_tailed_init(model, alpha=1.2, g=0.5, base_seed=99)

    # If the seed_offset is working, these identical shapes should have unique values
    assert not torch.equal(model.layer_a.weight, model.layer_b.weight), \
        "Layers share identical weights! Check your seed_offset logic."

def test_deterministic_model_init():
    """Verify that a fixed base_seed produces the exact same model every time."""
    model1 = ToyModel()
    model2 = ToyModel()

    apply_heavy_tailed_init(model1, alpha=1.2, g=0.5, base_seed=42)
    apply_heavy_tailed_init(model2, alpha=1.2, g=0.5, base_seed=42)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model init is not deterministic for the same base_seed."
