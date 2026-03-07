import torch
import numpy as np
import copy
from pathlib import Path
from scipy.optimize import curve_fit
from .ml_library import load_master_config, get_dataset_class, get_transform, get_universal_loader, model_factory

def get_singular_values(matrix):
    """
    Extracts singular values from a matrix.
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    # Paper analyzes singular values directly
    s = torch.linalg.svdvals(matrix.float())
    return s.detach().cpu().numpy()

def extract_rmt_parameters(matrix):
    """
    Extracts the dimensions (n, m) and the empirical rescaled
    variance (sigma_tilde) for Marchenko-Pastur fitting.

    Args:
        matrix: torch.Tensor or np.ndarray of shape (rows, cols)

    Returns:
        n (int): Larger dimension
        m (int): Smaller dimension
        sigma_tilde (float): Empirical variance parameter (sigma * sqrt(n))
    """
    # 1. Dimensions: Paper assumes m <= n
    shape = matrix.shape
    n = max(shape)
    m = min(shape)

    # 2. Empirical Sigma Tilde:
    # The paper uses sigma as the variance of matrix entries[cite: 76, 83].
    # In practice, we calculate it from the weights of the current layer.
    if isinstance(matrix, torch.Tensor):
        sigma_empirical = torch.std(matrix).item()
    else:
        sigma_empirical = np.std(matrix)

    # Per Eq (4), sigma_tilde = sigma * sqrt(n)
    sigma_tilde = sigma_empirical * np.sqrt(n)

    return n, m, sigma_tilde

def fit_mp_to_density(x_range, empirical_pdf, nu_raw):
    """
    Fits the modified MP law to the broadened empirical density
    to resolve the bulk boundary (nu_max).
    """
    # 1. Fixed Anchors
    nu_min = np.min(nu_raw) # [cite: 124]
    n, m, sigma_init = extract_rmt_parameters(nu_raw)
    Q = n / m

    # 2. Optimization Wrapper
    def mp_wrapper(x, v_max, s_tilde):
        # Faithful Eq (4) implementation [cite: 78-83]
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = Q / (np.pi * (s_tilde**2) * x)
            term2 = np.sqrt(np.maximum(0, (v_max**2 - x**2) * (x**2 - nu_min**2)))
            return np.nan_to_num(term1 * term2)

    # 3. Initial Guesses [cite: 83]
    v_max_guess = sigma_init * (1 + np.sqrt(1/Q))
    p0 = [v_max_guess, sigma_init]

    # 4. Fitting against the BROADENED density
    # We restrict the fit to the bulk region (x < v_max_guess * 1.1)
    # so the outliers in the tail don't pull the bulk fit off-center.
    fit_mask = x_range < (v_max_guess * 1.1)

    try:
        popt, _ = curve_fit(
            mp_wrapper,
            x_range[fit_mask],
            empirical_pdf[fit_mask],
            p0=p0,
            bounds=([nu_min + 1e-5, 1e-5], [np.inf, np.inf])
        )
        v_max_fit, s_tilde_fit = popt
    except RuntimeError:
        # Fallback to theory if optimization fails
        v_max_fit, s_tilde_fit = v_max_guess, sigma_init

    return v_max_fit, s_tilde_fit

def gaussian_broadening(nu, x_range, a=15):
    """
    Computes the smoothed PDF of singular values using adaptive
    Gaussian broadening per Equation (7) of the paper.

    Args:
        nu (array): The raw singular values extracted from the layer.
        x_range (array): The x-axis points where you want to evaluate the PDF.
        a (int): The window size for adaptive smoothing (paper uses 15).
    """
    # 1. Sort singular values (nu) in ascending order for neighbor calculation
    nu_sorted = np.sort(nu)
    m = len(nu_sorted)

    # 2. Calculate Adaptive Widths (sigma_k)
    # sigma_k = (nu_{k+a} - nu_{k-a}) / 2
    # We pad the array to handle the boundaries (0 and m-1)
    padded_nu = np.pad(nu_sorted, (a, a), mode='edge')
    sigmas = (padded_nu[2*a:] - padded_nu[:m]) / 2.0

    # Ensure no zero-width sigmas to avoid division by zero
    sigmas = np.maximum(sigmas, 1e-6)

    # 3. Compute PDF: Average of Gaussian kernels
    pdf = np.zeros_like(x_range)
    for k in range(m):
        # Calculate Gaussian centered at nu_k with width sigma_k
        diff = x_range - nu_sorted[k]
        kernel = (1.0 / (np.sqrt(2 * np.pi) * sigmas[k])) * \
                 np.exp(-0.5 * (diff / sigmas[k])**2)
        pdf += kernel

    return pdf / m

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

    # Get singular values
    s = torch.linalg.svdvals(W_t.detach().float())
    s2 = s**2
    p = s2 / s2.sum() # Normalized power

    # 1. Hill Estimator (Simplified)
    # We look at the top 10% of singular values to estimate the tail
    k = max(int(len(s) * 0.1), 5)
    s_top = s[:k]
    hill = 1.0 / (torch.log(s_top / s_top[-1]).mean())

    # 2. Raw Spectral Entropy
    entropy = -torch.sum(p * torch.log(p + 1e-10))

    # 3. Dominance Ratio
    dominance = s[0] / s.sum()

    return {
        'displacement': displacement.item(),
        'effective_rank': eff_rank.item(),
        'ipr': ipr.item(),
        'participation_ratio': part_ratio.item(),
        'max_singular_val': S_t[0].item(),
        'hill_alpha': hill.item(),
        'spectral_entropy': entropy.item(),
        'dominance_ratio': dominance.item(),
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
