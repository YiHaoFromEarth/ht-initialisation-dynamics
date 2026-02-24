import torch
import numpy as np
import copy
from pathlib import Path
from .ml_library import load_master_config, get_dataset_class, get_transform, get_universal_loader, model_factory

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

def get_layer_fingerprint(W_init, W_final):
    """
    Calculates fundamental spectral and dynamical metrics for a weight layer.

    Args:
        W_init (torch.Tensor): Weights at epoch 0.
        W_final (torch.Tensor): Weights at the target epoch.
    """
    # 1. Ensure tensors are on CPU and float
    W_0 = W_init.detach().cpu().float()
    W_t = W_final.detach().cpu().float()

    # 2. Singular Value Decomposition
    U_0, S_0, V_0 = torch.svd(W_0)
    U_t, S_t, V_t = torch.svd(W_t)

    # Relative Frobenius Displacement (Distance Traveled)
    # Proves the "Frozen" vs "Rich" regimes
    displacement = torch.norm(W_t - W_0) / torch.norm(W_0)

    # Effective Rank (Stable Rank)
    # Measures dimensionality/utility of the width
    eff_rank = (torch.sum(S_t**2)) / (S_t[0]**2)

    # Inverse Participation Ratio (IPR) of Top Vector
    # Measures localization (Spiky vs. Distributed)
    v_top = V_t[:, 0]
    ipr = torch.sum(v_top**4) / (torch.sum(v_top**2)**2)

    # Participation Ratio of Singular Values
    # Measures how 'flat' the eigenvalue distribution is
    part_ratio = (torch.sum(S_t)**2) / (len(S_t) * torch.sum(S_t**2))

    return {
        'displacement': displacement.item(),
        'effective_rank': eff_rank.item(),
        'ipr': ipr.item(),
        'participation_ratio': part_ratio.item(),
        'max_singular_val': S_t[0].item()
    }

def get_layer_from_checkpoint(model_path, layer_key):
    """
    Retrieves a specific weight matrix from a saved checkpoint.

    Args:
        model_path (str or Path): Path to the .pth file.
        layer_key (str): The state_dict key (e.g., 'features.0.weight').
    """
    # Load to CPU to avoid filling up VRAM during analysis
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle the nested 'model_state' or 'model_state_dict' structure
    # common in your ml_library.py exports
    state_dict = checkpoint.get('model_state', checkpoint)

    if layer_key not in state_dict:
        available = list(state_dict.keys())
        raise KeyError(f"Layer '{layer_key}' not found. Available: {available}")

    return state_dict[layer_key].detach().float()

def evaluate_spectral_perturbation(model, loader, layer_key, k, mode='ablate', device='cpu'):
    """
    Evaluates model accuracy after surgically modifying a layer's spectral components.

    Args:
        model: The instantiated PyTorch model.
        loader: The DataLoader (usually test/val).
        layer_key: The string name of the layer (e.g., 'features.0.weight').
        k: The number of singular values to modify.
        mode: 'ablate' (remove top-k) or 'rank-k' (keep only top-k).
    """
    # 1. Work on a deep copy to avoid 'polluting' the original model weights
    eval_model = copy.deepcopy(model).to(device)
    eval_model.eval()

    # 2. Extract the weight matrix
    # Format: features.0.weight -> getattr(model, 'features')[0].weight
    parts = layer_key.split('.')
    target = eval_model
    for part in parts[:-1]:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)

    W = target.weight.data.float()
    U, S, V = torch.svd(W)

    # 3. Apply Spectral Transformation
    S_mod = S.clone()
    if mode == 'ablate':
        # Remove the 'Experts' (top k)
        S_mod[:k] = 0.0
    elif mode == 'rank-k':
        # Keep ONLY the 'Experts' (top k), zero the bulk
        S_mod[k:] = 0.0
    else:
        raise ValueError("Mode must be 'ablate' or 'rank-k'")

    # 4. Reconstruct and Inject
    W_new = U @ torch.diag(S_mod) @ V.t()
    target.weight.data.copy_(W_new)

    # 5. Run Validation Loop
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = eval_model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    return {
        'layer': layer_key,
        'k': k,
        'mode': mode,
        'accuracy': correct / total
    }

def run_spectral_analysis(run_dir, config_path, layer_key, k_values, mode='ablate', device='cpu'):
    """
    Automated parent wrapper to run spectral perturbations on a specific experiment run.

    Args:
        run_dir (str or Path): The directory containing 'final_model.pth' and 'run_config.json'.
        config_path (str or Path): Path to the master YAML used for the run.
        layer_key (str): The layer to perturb (e.g., 'features.0.weight').
        k_values (list): List of k values to iterate through.
        mode (str): 'ablate' or 'rank-k'.
    """
    run_dir = Path(run_dir)

    # 1. Load configuration and setup environment
    cfg = load_master_config(config_path)
    data_cfg = cfg['data_config']

    # Resolve transforms and dataset class
    dataset_class = get_dataset_class(data_cfg['dataset_name'])
    data_cfg['transform'] = get_transform(data_cfg.get('transforms', []))

    # 2. Setup Data Loader (Test set for evaluation)
    is_omniglot = "Omniglot" in data_cfg['dataset_name']
    test_key = 'background' if is_omniglot else 'train'

    loader = get_universal_loader(
        dataset_class,
        data_cfg,
        **{test_key: False},
        download=True,
        root='./data'
    )

    # 3. Instantiate and Load Model
    m_args = cfg['model_params'].get('args', [])
    m_kwargs = cfg['model_params'].get('kwargs', {})
    model = model_factory(cfg['model_class'], *m_args, **m_kwargs)

    checkpoint_path = run_dir / "final_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Execute Parameter Sweep
    results = []
    print(f"Starting {mode} sweep on {layer_key} for {run_dir.name}...")

    for k in k_values:
        # Call the child function (assumed to be imported or in same file)
        res = evaluate_spectral_perturbation(model, loader, layer_key, k, mode, device)
        results.append(res)
        print(f"  k={k} | Accuracy: {res['accuracy']:.4f}")

    return results
