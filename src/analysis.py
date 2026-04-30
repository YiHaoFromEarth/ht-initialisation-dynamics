import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
import json
import re
import gc
import dcor
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.stats import spearmanr
from .rmt import fit_marcenkoPastur
from .equations import mcculloch_estimator
from .utils import (
    load_master_config,
    get_dataset_class,
    get_transform,
    get_universal_loader,
    model_factory,
    apply_spectral_filter_to_model,
    evaluate_model,
)

class ModelTracker:
    def __init__(self, model, lags=[1, 2, 4, 8, 16, 32, 64, 128]):
        self.lags = sorted(lags)
        self.max_lag = max(self.lags)
        self.num_lags = len(self.lags)

        # 1. Metadata setup
        self.param_metadata = []
        start_idx = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                num_params = p.numel()
                self.param_metadata.append({
                    "name": name,
                    "start": start_idx,
                    "end": start_idx + num_params,
                    "shape": p.shape,
                })
                start_idx += num_params

        self.sizes = [m["end"] - m["start"] for m in self.param_metadata]
        self.num_layers = len(self.param_metadata)
        total_params = start_idx

        # Infer device dynamically to avoid hardcoding
        self.device = next(model.parameters()).device

        # 2. TENSOR BUFFER SETUP
        self.history_buffer = torch.zeros((self.max_lag + 1, total_params), device=self.device)
        self.history_filled = 0
        self.cursor = 0

        # Kept entirely on GPU per your request
        self.results_buffer = []

    @torch.no_grad()
    def update(self, model, flat_grads=None):
        # 1. Capture current state (Avoid creating detached views, just read flat)
        current_flat = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])

        self.history_buffer[self.cursor] = current_flat
        current_idx = self.cursor
        self.cursor = (self.cursor + 1) % (self.max_lag + 1)
        self.history_filled = min(self.history_filled + 1, self.max_lag + 1)

        if self.history_filled <= 1:
            return

        # 2. Identify Valid Lags
        valid_lags = [tau for tau in self.lags if self.history_filled > tau]
        num_valid = len(valid_lags)

        if num_valid == 0:
            # Pad with zeros if no lags are ready
            self.results_buffer.append(torch.zeros((self.num_lags, self.num_layers, 7), device=self.device))
            return

        # Handle Gradients
        if flat_grads is not None:
            g_layers = torch.split(flat_grads, self.sizes)

        # 3. VECTORIZE ACROSS ALL LAGS
        # Fetch all past histories at once -> Shape: (num_valid, total_params)
        past_indices = [(current_idx - tau) % (self.max_lag + 1) for tau in valid_lags]
        past_flats = self.history_buffer[past_indices]

        # Compute differences for all lags simultaneously
        diffs = current_flat.unsqueeze(0) - past_flats

        # Split into layers -> Tuples of (num_valid, layer_size) tensors
        d_layers = torch.split(diffs, self.sizes, dim=1)
        p_layers = torch.split(past_flats, self.sizes, dim=1)
        c_layers = torch.split(current_flat, self.sizes)

        # Pre-calculate current norms to save O(Lags * Layers) redundant math
        c_norms = [torch.norm(c) for c in c_layers]

        nets_list, abs_means_list, rmss_list = [], [], []
        linfs_list, cos_dists_list, snrs_list, align_list = [], [], [], []

        # 4. Compute Metrics per layer (Loops `L` times instead of `L * Lags` times)
        for l_idx, (d, c, p, c_norm) in enumerate(zip(d_layers, c_layers, p_layers, c_norms)):
            # d is shape (num_valid, layer_size)
            nets_list.append(d.mean(dim=1))
            abs_means_list.append(d.abs().mean(dim=1))
            rmss_list.append(torch.sqrt((d**2).mean(dim=1)))
            linfs_list.append(d.abs().max(dim=1).values)
            snrs_list.append(d.mean(dim=1).abs() / (d.std(dim=1) + 1e-12))

            # Cosine distance using ultra-fast Matrix Multiplication instead of dot products
            # p @ c -> (num_valid, layer_size) @ (layer_size,) -> (num_valid,)
            dot_cp = torch.matmul(p, c)
            p_norms = torch.norm(p, dim=1)
            norm_cp = (c_norm * p_norms) + 1e-12
            cos_dists_list.append(1.0 - torch.clamp(dot_cp / norm_cp, -1.0, 1.0))

            # Gradient alignment
            if flat_grads is not None:
                g = g_layers[l_idx]
                dot_gd = torch.matmul(d, -g)
                g_norm = torch.norm(g)
                d_norms = torch.norm(d, dim=1)
                norm_gd = (g_norm * d_norms) + 1e-12
                align_list.append(dot_gd / norm_gd)
            else:
                align_list.append(torch.zeros(num_valid, device=self.device))

        # Stack the results into a single matrix of shape (num_valid, num_layers, 7)
        metrics = torch.stack([
            torch.stack(nets_list, dim=1),
            torch.stack(abs_means_list, dim=1),
            torch.stack(rmss_list, dim=1),
            torch.stack(linfs_list, dim=1),
            torch.stack(cos_dists_list, dim=1),
            torch.stack(snrs_list, dim=1),
            torch.stack(align_list, dim=1)
        ], dim=2)

        # Pad remaining invalid lags with zeros to maintain strict (num_lags, num_layers, 7) shape
        if num_valid < self.num_lags:
            pad = torch.zeros((self.num_lags - num_valid, self.num_layers, 7), device=self.device)
            metrics = torch.cat([metrics, pad], dim=0)

        # 5. Append to GPU buffer
        self.results_buffer.append(metrics)

    def to_dataframe(self, alpha, sigma, seed, scale=1):
        """
        Instantaneous Pandas conversion using NumPy masks instead of nested Python loops.
        This runs only once at the end of training.
        """
        if not self.results_buffer:
            return pd.DataFrame()

        # Only now do we pay the CPU transfer tax
        all_data = torch.stack(self.results_buffer).cpu().numpy()
        num_steps, num_lags, num_layers, num_metrics = all_data.shape

        # Create flat 1D arrays for highly optimized DataFrame construction
        steps_flat = np.repeat(np.arange(num_steps), num_lags * num_layers)
        lags_flat = np.tile(np.repeat(self.lags, num_layers), num_steps)
        layer_names = [m["name"] for m in self.param_metadata]
        layers_flat = np.tile(layer_names, num_steps * num_lags)

        data_flat = all_data.reshape(-1, num_metrics)

        # Apply your logic: Only log if step_idx >= tau
        mask = steps_flat >= lags_flat

        return pd.DataFrame({
            "alpha_init": alpha,
            "sigma_init": sigma,
            "seed": seed,
            "layer": layers_flat[mask],
            "time_lag": (lags_flat[mask] * scale).astype(int),
            "step": ((steps_flat[mask] + 1 - (lags_flat[mask] // 2)) * scale),
            "net_drift": data_flat[mask, 0],
            "abs_mean_dist": data_flat[mask, 1],
            "rms_dist": data_flat[mask, 2],
            "l_inf_dist": data_flat[mask, 3],
            "cos_dist": data_flat[mask, 4],
            "snr": data_flat[mask, 5],
            "grad_weight_alignment": data_flat[mask, 6],
        })


def get_layer_fingerprint(W, W0):
    # --- 1. PRE-CALCULATION & CHEAP METRICS ---
    W_abs = torch.abs(W)
    fro_norm = torch.linalg.norm(W).item()
    fro_norm0 = torch.linalg.norm(W0).item()

    cheap_metrics = {
        "frobenius": fro_norm,
        "max_weight_ratio": torch.max(W_abs).item()
        / (torch.mean(W_abs).item() + 1e-12),
        "est_alpha": mcculloch_estimator(W),  # Assumes this is element-wise/quantile
    }

    # --- 2. THE SVD CLUSTER (ONE SVD CALL) ---
    # We use CPU for SVD as it's often more stable for large matrices in parallel workers
    # but torch.linalg.svdvals is very fast.
    s = torch.linalg.svdvals(W)
    s_list = s.detach().cpu().numpy()
    s_sq = s_list**2
    s_max = s_list[0]

    # Effective Rank Entropy
    s_normed = s_list / (np.sum(s_list) + 1e-12)
    entropy = -np.sum(s_normed * np.log(s_normed + 1e-12))

    spectral_metrics = {
        "spectral_norm": s_max,
        "stable_rank": (fro_norm**2) / (s_max**2 + 1e-12),
        "participation_ratio": (np.sum(s_sq) ** 2) / (np.sum(s_sq**2) + 1e-12),
        "effective_rank": np.exp(entropy),
        "spectral_gap": s_max / (s_list[1] + 1e-12) if len(s_list) > 1 else 1.0,
        "condition_number": s_max / (s_list[-1] + 1e-12),
    }

    # --- 3. THE HISTORICAL CLUSTER (W vs W0) ---
    dist_abs = torch.linalg.norm(W - W0).item()
    dot_product = torch.sum(W * W0).item()
    cos_sim = dot_product / (fro_norm * fro_norm0 + 1e-12)

    # Sign Agreement (Physics: "Domain flipping")
    sign_agree = torch.mean((torch.sign(W) == torch.sign(W0)).float()).item()

    history_metrics = {
        "diff_norm_abs": dist_abs,
        "cosine_sim": cos_sim,
        "sign_agreement": sign_agree,
    }

    # Merge all into one record
    return {**cheap_metrics, **spectral_metrics, **history_metrics}


def process_single_run(cfg_path):
    """Worker function to process one run folder."""
    run_dir = cfg_path.parent
    ckpt_dir = run_dir / "checkpoints"
    records = []

    # --- Seed Extraction ---
    # Look for the '_s' suffix in the folder name (e.g., ..._20260427_185703_s0)
    seed_match = re.search(r"_s(\d+)$", run_dir.name)
    seed_val = int(seed_match.group(1)) if seed_match else 0

    # Load Metadata
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    metadata = {
        "alpha_init": cfg["ht_config"].get("alpha"),
        "sigma_init": cfg["ht_config"].get("g"),
        "seed": seed_val,
        "run_name": run_dir.name,  # Helpful for debugging later
    }
    final_epoch = cfg["hyperparams"].get("epochs")

    # 2. Identify and Sort Checkpoints
    all_ckpts = list(ckpt_dir.glob("weights_epoch_*.pth"))
    # Sort numerically so Epoch 0 is always index [0]
    all_ckpts.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x.name).group(1)))

    # Pre-load Epoch 0 reference for this specific seed
    w0_dict = {}
    if not all_ckpts:
        print(f"Skipping {run_dir.name}: No checkpoints found.")
        return []

    first_ckpt = torch.load(all_ckpts[0], map_location="cpu", weights_only=True)
    w0_state = first_ckpt.get("model_state", first_ckpt.get("state_dict", first_ckpt))

    for k, v in w0_state.items():
        if "weight" in k:
            # Store as float32 for metric precision
            w0_dict[k] = v.to(torch.float32).clone()

    del first_ckpt, w0_state

    # Append final_model to process it last
    final_pth = ckpt_dir / "final_model.pth"
    if final_pth.exists():
        all_ckpts.append(final_pth)

    # 3. Process Checkpoints
    for ckpt in all_ckpts:
        if "final_model" in ckpt.name:
            epoch_val = final_epoch
        else:
            match = re.search(r"epoch_(\d+)", ckpt.name)
            epoch_val = int(match.group(1)) if match else -1

        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get(
            "model_state", checkpoint.get("state_dict", checkpoint)
        )

        for key, tensor in state_dict.items():
            if "weight" in key and key in w0_dict:
                w0 = w0_dict[key]

                # Handle precision artifacts for final_model
                if "final_model" in ckpt.name:
                    w_np = tensor.to(torch.bfloat16).to(torch.float32)
                else:
                    w_np = tensor.to(torch.float32)

                # Calculate Metrics
                fingerprint = get_layer_fingerprint(w_np, w0)

                # Store flat record
                row = {**metadata, "epoch": epoch_val, "layer": key}
                row.update(fingerprint)
                records.append(row)

        del checkpoint, state_dict

    print(f"Finished Seed {seed_val} for Alpha {metadata['alpha_init']}")
    del w0_dict
    gc.collect()
    return records


def collect_sweep_metrics(
    sweep_dir, max_workers=4, output_path="sweep_metrics.parquet"
):
    """
    Iterates through a sweep, identifies seeds from folder names,
    calculates metrics per layer/epoch/seed, and saves to Parquet.
    """
    sweep_path = Path(sweep_dir)
    config_files = list(sweep_path.rglob("run_config.json"))

    all_records = []

    # max_workers=4 is a safe start for HPC.
    # Don't exceed the number of CPU cores assigned to your job/session.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map() handles passing the config_files to the worker function
        results = list(
            tqdm(
                executor.map(process_single_run, config_files),
                total=len(config_files),
                desc="Parallel Sweep Processing",
            )
        )

    # results is a list of lists (one list per run) -> Flatten it
    for run_records in results:
        all_records.extend(run_records)

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    return df


def aggregate_displacement_sweep(
    sweep_dir, output_name="master_displacement_database.parquet"
):
    """
    Concatenates individual run-level Parquet files into a master research database.
    Assumes each run directory contains a 'displacement_log.parquet' and 'run_config.json'.
    """
    sweep_path = Path(sweep_dir)
    all_dfs = []

    # 1. Locate all config files to identify valid runs
    config_files = list(sweep_path.rglob("run_config.json"))
    print(f"Found {len(config_files)} potential runs. Commencing aggregation...")

    for cfg_path in config_files:
        run_dir = cfg_path.parent
        parquet_path = run_dir / "displacement_log.parquet"

        if not parquet_path.exists():
            print(f"Skipping {run_dir.name}: No parquet log found.")
            continue

        # 2. Load the metadata from config
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Pulling the key variables for your thesis analysis
        alpha = cfg["ht_config"].get("alpha")
        sigma = cfg["ht_config"].get("g")
        seed = cfg.get("seed", "unknown")

        # 3. Load the run-level Parquet
        # This is already structured with steps, layers, and displacements
        run_df = pd.read_parquet(parquet_path)

        # 4. Inject metadata for global identification
        run_df["alpha"] = alpha
        run_df["sigma"] = sigma
        run_df["seed"] = seed

        all_dfs.append(run_df)

    # 5. The "Big Bang" Concatenation
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)

        # Enforce strict numeric types for physics queries
        master_df["alpha"] = pd.to_numeric(master_df["alpha"], errors="coerce")
        master_df["sigma"] = pd.to_numeric(master_df["sigma"], errors="coerce")

        # Save using ZSTD for better compression on these repetitive signals
        master_df.to_parquet(
            output_name, engine="pyarrow", compression="zstd", index=False
        )

        print("--- SUCCESS ---")
        print(f"Master database saved to: {output_name}")
        print(f"Total Rows: {len(master_df):,}")
        print(f"Columns tracked: {list(master_df.columns)}")

        return master_df
    else:
        print("Error: No dataframes found to combine.")
        return None


def collect_correlations_from_json(
    sweep_dir, method="spearman", output_name="layer_correlations.csv"
):
    """
    Computes a 10x10 correlation matrix for each lag time (delta_t) across runs.
    Methods: 'pearson', 'spearman', or 'dcor'
    """
    sweep_path = Path(sweep_dir)
    all_results = []

    config_files = list(sweep_path.rglob("run_config.json"))

    for cfg_path in config_files:
        run_dir = cfg_path.parent
        tamsd_json_path = run_dir / "tamsd_results.json"
        if not tamsd_json_path.exists():
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Metadata extraction
        alpha = cfg["ht_config"].get("alpha", "unknown")
        g_val = cfg["ht_config"].get("g", "unknown")
        metadata = {
            "alpha": alpha,
            "sigma": g_val,
            "run_id": f"alpha_{alpha}_g_{g_val}",
        }

        with open(tamsd_json_path, "r") as f:
            raw_data = json.load(f)

        layers = sorted([l for l in raw_data.keys() if l != "GLOBAL_MODEL"])
        lags = sorted(map(int, raw_data[layers[0]].keys()))
        num_layers = len(layers)

        for tau in lags:
            tau_str = str(tau)

            # --- STEP 1: Vectorized Data Loading ---
            # Create a 2D matrix [Layers x Steps]
            # This is done once per tau, rather than inside the i/j loop
            data_matrix = np.array([raw_data[l][tau_str] for l in layers])

            # --- STEP 2: Matrix Correlation ---
            if method == "pearson":
                # np.corrcoef calculates the entire matrix in one optimized call
                corr_matrix = np.corrcoef(data_matrix)

            else:
                # For Spearman or dcor, we still need a loop, but we use the
                # pre-computed data_matrix to avoid repetitive casting
                corr_matrix = np.eye(num_layers)
                for i in range(num_layers):
                    for j in range(i + 1, num_layers):
                        vec_i, vec_j = data_matrix[i], data_matrix[j]

                        if method == "spearman":
                            val, _ = spearmanr(vec_i, vec_j)
                        elif method == "dcor":
                            val = dcor.distance_correlation(vec_i, vec_j)
                        else:
                            val = 0

                        corr_matrix[i, j] = corr_matrix[j, i] = val

            # --- STEP 3: Flattening for Storage ---
            for i, l_i in enumerate(layers):
                for j, l_j in enumerate(layers):
                    all_results.append(
                        {
                            **metadata,
                            "delta_t": tau,
                            "method": method,
                            "layer_A": l_i,
                            "layer_B": l_j,
                            "correlation": corr_matrix[i, j],
                        }
                    )

        print(f"Processed Correlations ({method}): {metadata['run_id']}")
        del raw_data  # Free RAM

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_name, index=False)
        return df
    return None


def get_hill_plot(weights, max_k_fraction=0.1, min_k=10):
    # 1. Standardize and Flatten
    if torch.is_tensor(weights):
        weights = weights.to(torch.float32).numpy()

    w_abs = np.abs(weights).flatten()
    # Filter out zeros to avoid log errors
    w_abs = w_abs[w_abs > 0]

    # 2. Sort Descending
    w_sorted = np.sort(w_abs)[::-1]
    n = len(w_sorted)

    if n < min_k:
        return np.array([]), np.array([])

    # 3. Determine range of k
    max_k = int(n * max_k_fraction)
    ks = np.arange(min_k, max_k)

    # 4. Efficient Cumulative Calculation
    # Hill formula: 1/k * sum(ln(w_i)) - ln(w_k)
    log_w = np.log(w_sorted)
    cumsum_log_w = np.cumsum(log_w)

    # Get the sum of logs up to each k
    sums_to_k = cumsum_log_w[ks - 1]
    # Get the log of the threshold weight at each k
    log_thresholds = log_w[ks]

    # Calculate xi (the inverse alpha)
    xis = (sums_to_k / ks) - log_thresholds

    # 5. Final Alpha calculation with safety for zeros/negatives
    # Using np.divide to handle any potential zeros gracefully
    alphas = np.divide(1.0, xis, out=np.full_like(xis, np.inf), where=xis > 0)

    return ks, alphas


def evaluate_spectral_perturbation(
    model, loader, layer_key, k, mode="ablate", device="cpu"
):
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
    parts = layer_key.split(".")
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
    if mode == "ablate":
        # Remove the 'Experts' (top k)
        S_mod[:k] = 0.0
    elif mode == "rank-k":
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

    return {"layer": layer_key, "k": k, "mode": mode, "accuracy": correct / total}


def run_spectral_analysis(
    run_dir, config_path, layer_key, k_values, mode="ablate", device="cpu"
):
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
    data_cfg = cfg["data_config"]

    # Resolve transforms and dataset class
    dataset_class = get_dataset_class(data_cfg["dataset_name"])
    data_cfg["transform"] = get_transform(data_cfg.get("transforms", []))

    # 2. Setup Data Loader (Test set for evaluation)
    is_omniglot = "Omniglot" in data_cfg["dataset_name"]
    test_key = "background" if is_omniglot else "train"

    loader = get_universal_loader(
        dataset_class, data_cfg, **{test_key: False}, download=True, root="./data"
    )

    # 3. Instantiate and Load Model
    m_args = cfg["model_params"].get("args", [])
    m_kwargs = cfg["model_params"].get("kwargs", {})
    model = model_factory(cfg["model_class"], *m_args, **m_kwargs)

    checkpoint_path = run_dir / "final_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
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


def calculate_true_mle(weights_list, input_sample, activation="relu", device="cpu"):
    """
    Calculates the True Maximal Lyapunov Exponent (MLE) by propagating a
    perturbation through the Jacobian chain.

    Args:
        weights_list: List of torch.Tensors [W1, W2, ..., WL]
        input_sample: A single input vector (shape [1, input_dim])
        activation: 'relu', 'tanh', or 'sigmoid'
    """
    # 1. Initialize the perturbation vector q with unit norm
    # This represents the initial direction of the perturbation
    input_dim = weights_list[0].shape[1]
    q = torch.randn(1, input_dim).to(device)
    q = q / torch.norm(q)

    # 2. Setup activation derivatives
    def get_phi_prime(h, mode):
        if mode == "relu":
            return (h > 0).float()
        if mode == "tanh":
            return 1.0 - torch.tanh(h) ** 2
        if mode == "sigmoid":
            s = torch.sigmoid(h)
            return s * (1.0 - s)
        return torch.ones_like(h)

    total_log_expansion = 0.0
    x = input_sample.to(device)
    num_layers = len(weights_list)

    with torch.no_grad():
        for W in weights_list:
            W = W.to(device)

            # --- FORWARD PASS (State x) ---
            h = torch.matmul(x, W.t())

            # --- PERTURBATION PASS (Direction q) ---
            # The Jacobian J = diag(phi'(h)) * W
            # We find the new direction z = J * q
            phi_prime = get_phi_prime(h, activation)

            # This is the crucial part: z is the result of the perturbation
            # being transformed by the layer weight and filtered by activation
            z = phi_prime * torch.matmul(q, W.t())

            # --- MEASURE & NORMALIZE (The QR step) ---
            # The expansion factor at this layer is the norm of the new vector
            expansion = torch.norm(z)

            # Accumulate the log of the expansion
            total_log_expansion += torch.log(expansion + 1e-10).item()

            # Update q: Normalize the direction for the next layer
            # This ensures we don't hit numerical 'inf' or '0'
            q = z / (expansion + 1e-10)

            # Update x for the next layer's pre-activations
            if activation == "relu":
                x = torch.relu(h)
            elif activation == "tanh":
                x = torch.tanh(h)
            else:
                x = torch.sigmoid(h)

    # Average log-expansion across the network depth
    return total_log_expansion / num_layers


def spectral_kl_divergence(W_original, W_filtered, epsilon=1e-10):
    """
    Measures the Information Loss between the original and filtered weight
    spectra using KL Divergence.
    """
    # 1. Get the spectra (Singular Values)
    # We treat the singular values as a probability distribution of "Energy"
    s_orig = torch.linalg.svdvals(W_original)
    s_filt = torch.linalg.svdvals(W_filtered)

    # 2. Normalize into probability distributions (sum to 1)
    # We add epsilon to ensure numerical stability (no log(0))
    p = (s_orig + epsilon) / (s_orig.sum() + epsilon)
    q = (s_filt + epsilon) / (s_filt.sum() + epsilon)

    # 3. Calculate KL Divergence: sum(p * log(p/q))
    # Higher value = Higher information loss / Spectrum distortion
    kl_div = torch.sum(p * torch.log(p / q))

    return kl_div.item()


def run_spectral_scan(
    model,
    test_loader,
    layer_key_func,
    window_size_perc=0.1,
    kernel="uniform",
    num_centers=15,
    iterations=1,
    device="cpu",
    criterion=nn.CrossEntropyLoss(),
):
    """
    Orchestrates a full spectral scan across different centers and kernel types.

    Returns:
        pd.DataFrame: A table containing center, kernel_type, mean_acc, and std_acc.
    """
    scan_results = []
    # Generate centers from 0 (Outliers/Head) to 1.0 (Bulk/Tail)
    # Note: We cap it at 1.0 - window_size/2 to keep the window inside the spectrum
    half_win = window_size_perc / 2
    centers = np.linspace(half_win, 1.0 - half_win, num_centers)

    original_weights = {
        n: m.weight.data.detach().clone()
        for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d)) and layer_key_func(n)
    }

    for c in centers:
        # 1. Generate the modified model using our parent function
        # This uses deepcopy internally, so the original 'model' stays 'Blessed'
        reconstructed_model = apply_spectral_filter_to_model(
            model,
            layer_key_func,
            center_perc=c,
            window_size_perc=window_size_perc,
            kernel_type=kernel,
        )

        # 2. KL Divergence Calculation (Spectral Info Loss)
        kl_values = []
        for name, weight_orig in original_weights.items():
            # Extract the corresponding layer from the reconstructed model
            # We use r_module.weight.data to get the filtered tensor
            r_module = dict(reconstructed_model.named_modules())[name]
            weight_recon = r_module.weight.data

            # Use our 'Atomic' KL function
            kl_val = spectral_kl_divergence(weight_orig, weight_recon)
            kl_values.append(kl_val)

        # Average KL across targeted layers (Mean Spectral Distortion)
        mean_kl = np.mean(kl_values) if kl_values else 0.0

        # 3. Performance Evaluation using your new consolidated function
        # We run the consolidated function 'iterations' times
        iter_accs = []
        for _ in range(iterations):
            metrics = evaluate_model(
                reconstructed_model, test_loader, device, criterion
            )
            # Multiplying by 100 to keep your percentage format if desired
            iter_accs.append(metrics["acc"] * 100)

        mean_acc = np.mean(iter_accs)
        std_acc = np.std(iter_accs) if iterations > 1 else 0.0

        # 4. Store the metadata for plotting
        scan_results.append(
            {
                "center_perc": c,
                "kernel_type": kernel,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "kl_divergence": mean_kl,
                "window_size": window_size_perc,
            }
        )

    return pd.DataFrame(scan_results)


def get_rmt_threshold_percentage(weight_tensor, broadener):
    """
    Returns the rank percentage where the RMT bulk ends.
    Anything below this percentage (closer to 0.0) is the Tail/Outliers.
    Anything above this (closer to 1.0) is the Bulk.
    """
    # 1. Get singular values
    nu = torch.linalg.svdvals(weight_tensor).cpu().numpy()

    # 2. Fit MP using the authors' official logic
    # We use range_of_y_to_fit=0.7 as per your provided code
    a_fit, nuMin_fit, nuMax_fit, _ = fit_marcenkoPastur(
        nu, broadener, range_of_y_to_fit=0.7
    )

    # 3. Find how many singular values are LARGER than the theoretical nuMax
    # Because SVD is sorted descending: [Outliers, Tail, nuMax, Bulk...]
    num_outside_bulk = np.sum(nu > nuMax_fit)

    # 4. Return as a percentage of the total rank
    return num_outside_bulk / len(nu)
