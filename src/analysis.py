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
from collections import deque
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from .equations import (
    mcculloch_estimator,
    relative_frobenius_norm,
    spectral_norm,
    spectral_gap,
    layerwise_cosine,
    stable_rank,
)
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
    """
    Optimized tracker for high-end GPUs.
    Tracks multiple lag times (tau) across all layers simultaneously.
    """

    def __init__(self, model, lags=[1, 2, 4, 8, 16, 32, 64, 128]):
        self.lags = sorted(lags)
        self.max_lag = max(self.lags)
        self.layer_names = [n for n, p in model.named_parameters() if p.requires_grad]

        # History of flattened weight vectors (keep on GPU!)
        self.history = deque(maxlen=self.max_lag + 1)

        # Storage for raw data: {layer_name: {tau: [list_of_displacements]}}
        self.raw_data = {
            name: {tau: [] for tau in self.lags} for name in self.layer_names
        }

    @torch.no_grad()
    def update(self, model):
        # Capture state on GPU
        current_state = {
            n: p.detach().clone().view(-1)
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.history.append(current_state)

        # Calculate displacements
        for tau in self.lags:
            if len(self.history) > tau:
                past_state = self.history[-(tau + 1)]
                for name in self.layer_names:
                    diff = current_state[name] - past_state[name]

                    # 1. Net Movement (The Drift / Center of Mass)
                    net_diff = torch.mean(diff).item()

                    # 2. Absolute Movement (The Total Activity)
                    abs_diff = torch.mean(torch.abs(diff)).item()

                    # 3. Root Mean Square Movement (The Energy / Variance)
                    rms_diff = torch.sqrt(torch.mean(diff ** 2)).item()

                    # Log as a tuple or dict
                    self.raw_data[name][tau].append((net_diff, abs_diff, rms_diff))

    def to_dataframe(self, alpha, sigma, scale=1):
        """
        Converts internal raw_data into a flattened DataFrame.
        'scale' allows converting Epochs -> Steps automatically.
        """
        rows = []
        for layer, lags in self.raw_data.items():
            for tau, values in lags.items():
                unified_tau = int(tau) * scale
                for step_idx, (net, l1, l2) in enumerate(values):
                    rows.append({
                        "alpha": alpha,
                        "sigma": sigma,
                        "layer": layer,
                        "time_lag": unified_tau,
                        "step": (step_idx + tau - (tau / 2)) * scale, # Centers the epoch movement
                        "net_drift": net,
                        "l1_dist": l1,
                        "l2_dist": l2
                    })
        return pd.DataFrame(rows)


def get_singular_values(matrix):
    """
    Extracts singular values from a matrix.
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    # Paper analyzes singular values directly
    s = torch.linalg.svdvals(matrix.float())
    return s.numpy()


def get_layer_fingerprint(W):
    """
    Modular fingerprinting for a single weight matrix.
    Add any new metric by simply adding a key-value pair to the dict.
    """
    if torch.is_tensor(W):
        W_torch = W.float()
    else:
        W_torch = torch.from_numpy(W).float()

    # --- EXTENSIBLE METRIC SECTION ---
    # You can add new metrics here easily.
    fingerprint = {
        "alpha": mcculloch_estimator(W_torch),
        "spectral_norm": spectral_norm(W_torch),
        "spectral_gap": spectral_gap(W_torch),
        "stable_rank": stable_rank(W_torch),
    }

    return fingerprint


def get_difference_fingerprint(W, W0):
    """
    Modular fingerprinting for the difference between a snapshot and the initial.
    Add any new metric by simply adding a key-value pair to the dict.
    """
    if torch.is_tensor(W):
        W_torch = W.float()
        W0_torch = W0.float()
    else:
        W_torch = torch.from_numpy(W).float()
        W0_torch = torch.from_numpy(W0).float()

    # --- EXTENSIBLE METRIC SECTION ---
    # You can add new metrics here easily.
    fingerprint = {
        "frobenius": relative_frobenius_norm(W_torch, W0_torch),
        "layerwise_cosine": layerwise_cosine(W_torch, W0_torch),
    }

    return fingerprint


def collect_sweep_metrics(sweep_dir, output_name="sweep_metrics.csv"):
    """
    Iterates through a sweep, calculates metrics file-by-file,
    and saves to a CSV.
    """
    sweep_path = Path(sweep_dir)
    records = []

    # 1. Walk through the alpha/sigma folders
    # We use rglob to find all run_config files to identify run directories
    config_files = list(sweep_path.rglob("run_config.json"))
    print(f"Found {len(config_files)} runs. Starting metric extraction...")

    for cfg_path in config_files:
        run_dir = cfg_path.parent
        ckpt_dir = run_dir / "checkpoints"

        # Load Metadata
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Pull parameters
        metadata = {
            "init_alpha": cfg["ht_config"].get("alpha"),
            "init_sigma": cfg["ht_config"].get("g"),
        }
        final_epoch = cfg["hyperparams"].get("epochs")

        # 2. Identify and Sort Checkpoints
        all_ckpts = list(ckpt_dir.glob("weights_epoch_*.pth"))

        # Sort by epoch number to ensure we hit 0 first
        all_ckpts.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x.name).group(1)))

        # Pre-load Epoch 0 to serve as our reference for cumulative difference
        w0_dict = {}
        first_ckpt = torch.load(all_ckpts[0], map_location="cpu", weights_only=True)
        w0_state = first_ckpt.get(
            "model_state", first_ckpt.get("state_dict", first_ckpt)
        )
        for k, v in w0_state.items():
            if "weight" in k:
                w0_dict[k] = v.to(torch.float32).clone()

        # Add final_model to the end of the list after sorting
        final_pth = ckpt_dir / "final_model.pth"
        if final_pth.exists():
            all_ckpts.append(final_pth)

        for ckpt in all_ckpts:
            # Determine epoch number
            if "final_model" in ckpt.name:
                epoch_val = final_epoch
            else:
                match = re.search(r"epoch_(\d+)", ckpt.name)
                epoch_val = int(match.group(1)) if match else -1

            # 3. STREAMING LOAD: Open, Compute, Discard
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state", checkpoint)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            for key, tensor in state_dict.items():
                if "weight" in key and key in w0_dict:
                    # Calculate metric immediately
                    # If final snapshot, we downcast to bfloat16 first to avoid artifacts
                    w0 = w0_dict[key].numpy()
                    if "final_model" in ckpt.name:
                        w_np = tensor.to(torch.bfloat16).to(torch.float32).numpy()
                    else:
                        w_np = tensor.to(torch.float32).numpy()

                    fingerprint = get_layer_fingerprint(w_np)
                    diff_print = get_difference_fingerprint(w_np, w0)

                    row = {**metadata, "epoch": epoch_val, "layer": key}
                    row.update(fingerprint)
                    row.update(diff_print)
                    records.append(row)

            # 4. MEMORY CLEANUP: Clear the heavy objects
            del checkpoint
            del state_dict
            # We don't gc.collect() every single file (too slow),
            # but every run is a good balance.

        print(f"Finished: {run_dir.parent.name}")
        del w0_dict
        gc.collect()

    # 5. Save the lightweight results
    df = pd.DataFrame(records)
    df.to_csv(output_name, index=False)
    print(f"Done! Results saved to {output_name}")
    return df


def aggregate_displacement_sweep(sweep_dir, output_name="master_displacement_database.parquet"):
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
        master_df["alpha"] = pd.to_numeric(master_df["alpha"], errors='coerce')
        master_df["sigma"] = pd.to_numeric(master_df["sigma"], errors='coerce')

        # Save using ZSTD for better compression on these repetitive signals
        master_df.to_parquet(output_name, engine='pyarrow', compression='zstd', index=False)

        print("--- SUCCESS ---")
        print(f"Master database saved to: {output_name}")
        print(f"Total Rows: {len(master_df):,}")
        print(f"Columns tracked: {list(master_df.columns)}")

        return master_df
    else:
        print("Error: No dataframes found to combine.")
        return None


def collect_correlations_from_json(sweep_dir, method="spearman", output_name="layer_correlations.csv"):
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
        if not tamsd_json_path.exists(): continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Metadata extraction
        alpha = cfg["ht_config"].get("alpha", "unknown")
        g_val = cfg["ht_config"].get("g", "unknown")
        metadata = {"alpha": alpha, "sigma": g_val, "run_id": f"alpha_{alpha}_g_{g_val}"}

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
                    all_results.append({
                        **metadata,
                        "delta_t": tau,
                        "method": method,
                        "layer_A": l_i,
                        "layer_B": l_j,
                        "correlation": corr_matrix[i, j]
                    })

        print(f"Processed Correlations ({method}): {metadata['run_id']}")
        del raw_data # Free RAM

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


"""This section contains contains functions for random matrix theory (RMT) analysis. Adapted from 10.1103/PhysRevE.106.054124"""
from typing import List, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from numba import jit, prange
from scipy.special import erf
from scipy.optimize import curve_fit
import scipy.stats
from functools import partial
import powerlaw
from tqdm import tqdm


"""Spectra broadening"""


class Broadener(ABC):
    """Used to broaden a spectrum as an alternative to histograms
    and to unfold it."""

    @abstractmethod
    def broaden_spectrum(self, x: np.ndarray, svals: np.ndarray):
        """Return broadened spectrum from svals at positions x."""

    @abstractmethod
    def unfold_spectrum(self, svals: np.ndarray):
        """Return unfolded spectrum from svals."""


class GaussBroadening(Broadener):
    """Uses a Gaussian kernel with widths of the Gaussians based
    on the density of the singular values (via window size winSize).
    This spectrum is then used for unfolding.
    The method {'replicate' or 'drop'} decides how to deal with
    the first and last winSize values. For replicate, the end
    values are repeated to include them all while for drop, these
    values will not contribute to the broadened spectrum.
    Broadener also handles unfolding of the spectrum."""

    def __init__(self, winSize: float, method: str = "replicate"):
        self.winSize = winSize
        if method in {"replicate", "drop"}:
            self.method = method
        else:
            raise ValueError('Method not recognized. Use "replicate" or "drop".')

    def broaden_spectrum(self, x: np.ndarray, svals: np.ndarray) -> np.ndarray:
        svals = self.preprocessSvals(svals)
        nSvals = np.size(svals)
        # standard deviations for boradening
        stdevs = (
            (svals[2 * self.winSize : :] - svals[: -2 * self.winSize :]) / 2
        ).reshape((1, -1))
        # means of each window
        means = svals[self.winSize : -self.winSize :].reshape((1, -1))
        # distribution as sum of gaussians
        xMat = x.reshape((-1, 1))
        # generate all gaussians
        pdf = GaussBroadening.gaussian(xMat, stdevs, means)
        pdf = np.sum(pdf, axis=1) / nSvals

        return pdf

    def unfold_spectrum(self, svals: np.ndarray) -> np.ndarray:
        # unfold via the cdf
        x = (svals[2 * self.winSize : -2 * self.winSize]).reshape((-1, 1))
        means = (svals[self.winSize : -self.winSize]).reshape((1, -1))
        stdvs = ((svals[2 * self.winSize :] - svals[: -2 * self.winSize]) / 2).reshape(
            (1, -1)
        )

        unfolded = 0.5 * (1 + erf((x - means) / (np.sqrt(2) * stdvs)))
        unfolded = np.sum(unfolded, axis=1)

        return np.sort(unfolded)

    def preprocessSvals(self, svals: np.ndarray) -> np.ndarray:
        if self.method == "replicate":
            svals = copy.deepcopy(svals)
            svals = np.pad(svals, (self.winSize, self.winSize), "edge")
        return svals

    @staticmethod
    def gaussian(x: np.ndarray, sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return (
            1
            / np.sqrt(2 * np.pi * (sigma + 1e-15) ** 2)
            * np.exp(-((x - mu) ** 2 + 1e-15) / (2 * (sigma + 1e-15) ** 2))
        )


"""Theory curves..."""


def marcenkoPastur(x: np.ndarray, a: float, nuMax: float, nuMin: float) -> np.ndarray:
    """Modified Marcenko-Pastur distribution for three independent parameters
    nuMin, nuMax, and a."""
    y = a / x * np.sqrt((nuMax**2 - x**2) * (x**2 - nuMin**2))
    y[y < 0] = 0.0
    y[np.imag(y) != 0] = 0.0
    y[np.isnan(y)] = 0.0

    return y


def wignerSurmise(s: np.ndarray) -> np.ndarray:
    """Wigner surmise formula for spacing of unfolded spectra of
    GOE matrices."""
    return np.pi * s / 2 * np.exp(-1.0 * np.pi * s**2 / 4)


def wignerSurmise_cdf(s: np.ndarray) -> np.ndarray:
    """CDF of Wigner surmise formula for spacing of unfolded spectra of
    GOE matrices."""
    return 1 - np.exp(-s * s * np.pi / 4)


def power_law_cdf(x: np.ndarray, alpha: float, xmin: float):
    """Cumulative distribution for a powerlaw tail."""
    return 1 - (x / xmin) ** (1 - alpha)


"""Data processing..."""


def get_ipr(svec: np.ndarray):
    """Computes the inverse participation ratio of a vector (svec)."""
    ipr = np.sum(svec**4)
    return ipr


def pdf_from_spectrum(
    svals: np.ndarray, nSamples: int, broadener: Broadener
) -> List[np.ndarray]:
    """Uses broadening (defined by broadener) for the spectrum (svals) and returns the probability density as
    an array pdf evaluated at positions x. Parameter nSamples determines the number of point between consecutive
    values in sort(svals) to sample the pdf at."""
    svals = np.sort(svals)
    nSvals = np.size(svals)
    # determine x values with a number "nSamples" points between each
    # consecutive singular values
    offset = 500 * (svals[1] - svals[0])
    if svals[1] - svals[0] > 0:
        x = np.arange(-offset, svals[0], (svals[1] - svals[0]) / nSamples)
    else:
        x = np.arange(svals[0] - 0.1, svals[0], 0.01)
    for i in range(nSvals - 1):
        if svals[i + 1] - svals[i] > 0:
            x = np.concatenate(
                [
                    x,
                    np.arange(
                        svals[i], svals[i + 1], (svals[i + 1] - svals[i]) / nSamples
                    ),
                ]
            )
        else:
            x = np.concatenate([x, [svals[i + 1]]])
    if svals[-1] - svals[-2] > 0:
        x = np.concatenate(
            [
                x,
                np.arange(
                    svals[-1],
                    svals[-1] + 500 * (svals[-1] - svals[-2]),
                    (svals[-1] - svals[-2]) / nSamples,
                ),
            ]
        )
    else:
        x = np.concatenate([x, np.arange(svals[-1], svals[-1] + 0.1, 0.01)])
    # evaluate broadend
    pdf = broadener.broaden_spectrum(x, svals)

    return [x, pdf]


def fit_marcenkoPastur(
    svals: np.ndarray,
    broadener: np.ndarray,
    nSamples: int = 10,
    range_of_y_to_fit: float = 0.7,
    iNuMin: int = 0,
    initialParameters: Union[np.ndarray, None] = None,
    xMin: float = 0.0,
):
    """Fix nuMin by the svals and then fit 2 parameter modified Marcenko-Pastur to the data."""
    # get pdf by broadening
    x, pdf = pdf_from_spectrum(svals, nSamples, broadener)
    condition_keep = np.array(x > xMin)
    x = x[condition_keep]
    pdf = pdf[condition_keep]

    # fit
    if initialParameters is None:
        initialParameters = np.array([0.5, 2.0])
    lowerBounds = (0.0, 0.0)
    upperBounds = (np.inf, np.inf)
    parameterBounds = [lowerBounds, upperBounds]

    # fix nuMin to actual min in svals
    nuMin = np.max([np.sort(svals)[iNuMin], xMin])
    fit_fun = partial(marcenkoPastur, nuMin=nuMin)
    # restrict range
    x_pdfMax = x[np.argmax(pdf)]
    pdfMax = np.max(pdf)
    # print(np.argmax(pdf), x_pdfMax, pdfMax)
    condition_keep = np.array(x <= x_pdfMax) | np.array(
        pdf > range_of_y_to_fit * pdfMax
    )
    pdf = pdf[condition_keep]
    x = x[condition_keep]

    try:
        (a, nuMax), pcov = curve_fit(
            fit_fun, x, pdf, initialParameters, bounds=parameterBounds
        )
    except:
        (a, nuMax), pcov = curve_fit(
            fit_fun, x, pdf, initialParameters, bounds=parameterBounds, maxfev=5000
        )

    return a, nuMin, nuMax, pcov


def unfold_spectrum(svals: np.ndarray, broadener: Broadener) -> np.ndarray:
    """Computed the unfolded spectrum."""
    return broadener.unfold_spectrum(svals)


def level_spacings(svals: np.ndarray, broadener: Broadener) -> np.ndarray:
    """Computes the level spacing of the unfolded spectrum (via broadening)."""
    unfolded = np.sort(unfold_spectrum(svals, broadener))
    return unfolded[1:] - unfolded[:-1]


def cdf_from_spectrum(svals: np.ndarray) -> List[np.ndarray]:
    """Computed the cdf from a spectrum, retruning x=svals and cdf."""
    svals = np.sort(svals)
    nValues = np.size(svals)
    cdf = np.arange(0, nValues, 1) / nValues
    return [svals, cdf]


def level_number_variance(
    svals: np.ndarray,
    broadener: Broadener,
    L: np.ndarray,
    tol: float,
    maxIterations: int,
    minIterations: int,
) -> List[np.ndarray]:
    """
    Computes the spectral regidity in an iterative fashion.
    Code adapted from empyricalRMT python package:
    https://pypi.org/project/empyricalRMT/
    and adjusted to our needs
    """
    unfolded = broadener.unfold_spectrum(svals)
    unfolded = np.sort(unfolded)
    print(
        "xi_min=",
        np.min(unfolded),
        " xi_max=",
        np.max(unfolded),
        " n=",
        np.size(unfolded),
    )

    L_vals, sigma = _sigma_iter_converge(
        unfolded=unfolded,
        L=L,
        tol=tol,
        max_L_iters=maxIterations,
        min_L_iters=minIterations,
    )

    return L_vals, sigma


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _sigma_iter_converge(
    unfolded: np.ndarray, L: np.ndarray, tol: float, max_L_iters: int, min_L_iters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Code adapted from empyricalRMT python package:
    https://pypi.org/project/empyricalRMT/
    and adjusted to our needs
    """
    # the copy and different variable is needed here in the parallel context
    # https://github.com/numba/numba/issues/3652
    L_vals = np.copy(L)
    sigma = np.empty(L_vals.shape, dtype=np.float64)
    for i in prange(L_vals.shape[0]):
        # tol_modified = tol + tol * (L[i] / 5.0)
        tol_modified = tol
        sigma[i] = _sigma_iter_converge_L(
            unfolded, L_vals[i], tol_modified, max_L_iters, min_L_iters
        )
    return L_vals, sigma


@jit(nopython=True, cache=True, fastmath=True)
def _sigma_iter_converge_L(
    unfolded: np.ndarray, L: float, tol: float, max_iters: int, min_iters: int
) -> Any:
    """
    Code adapted from empyricalRMT python package:
    https://pypi.org/project/empyricalRMT/
    and adjusted to our needs
    """
    level_mean = 0.0
    level_sq_mean = 0.0
    sigma = 0.0
    size = min_iters
    # hold the last `size` running averages
    sigmas = np.zeros((size), dtype=np.float64)

    c = np.random.uniform(np.min(unfolded) + L / 2, np.max(unfolded) - L / 2)
    start, end = c - L / 2, c + L / 2
    n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
    n_within_sq = n_within * n_within
    level_mean, level_sq_mean = n_within, n_within_sq
    sigma = level_sq_mean - level_mean * level_mean
    sigmas[0] = sigma

    # we'll use the fact that for x = [x_0, x_1, ... x_n-1], the
    # average a_k == (k*a_(k-1) + x_k) / (k+1) for k = 0, ..., n-1
    k = 0
    while True:
        k += 1
        c = np.random.uniform(np.min(unfolded) + L / 2, np.max(unfolded) - L / 2)
        start, end = c - L / 2, c + L / 2
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        level_mean = (k * level_mean + n_within) / (k + 1)
        level_sq_mean = (k * level_sq_mean + n_within_sq) / (k + 1)
        sigma = level_sq_mean - level_mean * level_mean
        sigmas[k % size] = sigma
        if np.abs(np.max(sigmas) - np.min(sigmas)) < tol and k > min_iters:
            break
        if k > max_iters:
            break
    return sigma


"""Default KS test for Wigner surmise"""


def ksTest_wigner(svals: np.ndarray, broadener: Broadener):
    """Tests level spacing of unfolded singular values vs wigner surmise with
    a Kolmogorov-Smirnov test. Takes the unaltered singular values and
    a Broadener which also defines unfolding."""
    s = level_spacings(svals, broadener)
    statistic, pValue = scipy.stats.kstest(s, wignerSurmise_cdf)
    return pValue


def fit_Brody_bootstrap(svals: np.ndarray, broadener: Broadener, nSamples: int = 1000):
    """Using bootstrapping, determine mean in beta and error of the mean
    from fitting Brody distributions to the unfolded level spacing cdfs."""

    def brody(x, b):
        return 1 - np.exp(
            -1.0 * scipy.special.gamma((b + 2) / (b + 1)) ** (1 + b) * x ** (1 + b)
        )

    def fit_Brody(s: np.ndarray):
        initialParameters = [1.0]
        parameterBounds = [[0], [np.inf]]

        x, y = cdf_from_spectrum(s)
        b, pcov = curve_fit(brody, x, y, initialParameters, bounds=parameterBounds)
        return b

    s = level_spacings(svals, broadener)

    betas = run_on_bootstarp(nSamples, fit_Brody, data=s)
    beta_error = np.std(betas)
    beta = np.mean(betas)

    return beta, beta_error


def run_on_bootstarp(nSamples: int, func, data: np.ndarray):
    """Evaluates the function func for nSample different bootstrap
    samples from data."""
    return [func(bootstrapSample(data)) for _ in range(nSamples)]


def bootstrapSample(data: Union[np.ndarray, List[float]]):
    """Randomly choose a bootstrap sample from data with
    the same length as data."""
    return np.random.choice(data, size=np.size(data))


"""KS test statistics for Porter-Thomas test."""


class CDF:
    """CDF takes discrete values C_y = C(C_x) and
    interpolates to all values C(x) when called.
    Good approximation for sufficently dense C_x."""

    def __init__(self, C_x, C_y):
        self.x = C_x
        self.y = C_y

    def __call__(self, x: float):
        return np.interp(x, self.x, self.y)


def ks_Cbar(N: int, nSamples: int):
    """Get normed vector CDF for Porter-Thomas eigenvectors."""
    xi = np.random.randn(N, nSamples)
    xi = xi / np.sqrt(np.sum(xi**2, axis=0))
    xi = xi.reshape(
        -1,
    )
    Cbar_x = np.sort(xi)
    Cbar_y = np.arange(len(xi)) / (len(xi) - 1)

    return CDF(Cbar_x, Cbar_y)


def ks_D(cdf: CDF, N: int, nSamples: int):
    """Monte-Carlo samples from Porter-Thomas vectors for nSamples samples
    and computes typical KS-distance D to given cdf for each sample.."""
    xi = np.random.randn(N, nSamples)
    xi = xi / np.sqrt(np.sum(xi**2, axis=0))
    Ds = []
    for i in range(nSamples):
        cdf_x_i = np.sort(xi[:, i])
        cdf_y_i = np.arange(len(xi[:, i])) / (len(xi[:, i]) - 1)
        Ds.append(np.max(np.abs(cdf(cdf_x_i) - cdf_y_i)))
    return Ds


def ks_test_statistic_normedPT(N: int, nSamples: int):
    """Returns ks test statistic for Porter-Thomas ks-test.
    Cbar is the cdf of normed PT vectors. C(D) is the cdf
    used to obtain the p-value as 1-p=C(D) for a given
    ks distance D of a sample."""
    Cbar = ks_Cbar(N, nSamples)
    D = np.sort(ks_D(Cbar, N, nSamples))
    C = CDF(D, np.arange(len(D)) / (len(D) - 1))

    return Cbar, C, D


def ks_test_normedPT(x: np.ndarray, ks_C: CDF, Cbar: CDF):
    """Given the ks-test statistics ks_C(D) and underlying cdf Cbar,
    performs a ks test for x against the given distribution and
    returns the corresponding p-value."""
    x = x.reshape(
        -1,
    )
    cdf_x_i = np.sort(x)
    cdf_y_i = np.arange(len(x)) / (len(x) - 1)
    D = np.max(np.abs(Cbar(cdf_x_i) - cdf_y_i))
    pvalue = 1 - ks_C(D)
    return pvalue


def ks_Cbar_pooled(N: int, nSamples: int, pooling_window: int):
    """Get normed vector CDF for a pool of Porter-Thomas eigenvectors."""
    xi = np.array([])
    for _ in range(2 * pooling_window):
        _xi = np.random.randn(N, nSamples)
        _xi = _xi / np.sqrt(np.sum(_xi**2, axis=0))
        _xi = _xi.reshape(
            -1,
        )
        xi = np.concatenate([xi, _xi])
    Cbar_x = np.sort(xi)
    Cbar_y = np.arange(len(xi)) / (len(xi) - 1)

    return CDF(Cbar_x, Cbar_y)


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def ks_D_pooled(
    cdf_x: np.ndarray, cdf_y: np.ndarray, N: int, nSamples: int, pooling_window: int
):
    """Monte-Carlo samples from a pool of Porter-Thomas vectors for nSamples samples
    and computes typical KS-distance D to given cdf for each sample.."""
    Ds = np.zeros(
        nSamples,
    )
    xi = np.zeros(
        N * 2 * pooling_window,
    )
    for i in range(nSamples):
        for j in range(2 * pooling_window):
            _xi = np.random.randn(
                N,
            )
            _xi = _xi / np.sqrt(np.sum(_xi**2))
            _xi = _xi.reshape(
                -1,
            )
            xi[j * N : (j + 1) * N] = _xi
        cdf_x_i = np.sort(xi)
        cdf_y_i = np.arange(len(xi)) / (len(xi) - 1)
        Ds[i] = np.max(np.abs(np.interp(cdf_x_i, cdf_x, cdf_y) - cdf_y_i))
    return Ds


def ks_test_statistic_normedPT_pooled(N: int, nSamples: int, pooling_window: int):
    """Returns ks test statistic for Porter-Thomas ks-test.
    Cbar is the cdf of a pool of normed PT vectors. C(D) is the cdf
    used to obtain the p-value as 1-p=C(D) for a given
    ks distance D of a sample."""
    print("Get Cbar...\n")
    Cbar = ks_Cbar_pooled(N, nSamples, pooling_window)
    print("Get D values for statistic...\n")
    D = np.sort(ks_D_pooled(Cbar.x, Cbar.y, N, nSamples, pooling_window))
    print("Get KS statistics...\n")
    C = CDF(D, np.arange(len(D)) / (len(D) - 1))

    return Cbar, C, D


"""Hill estimator tools"""


@jit(nopython=True)
def movmean(data, navg=3):
    """Computes a moving average of data with window navg."""
    midindices = range(navg // 2, data.size - navg // 2 + 1)
    mean = np.zeros(len(midindices))
    for i, imid in enumerate(midindices):
        mean[i] = np.mean(data[imid - navg // 2 : imid + navg // 2 + 1])
    return mean


def hill_estimator_avg(data, navg, avg_inverse=True, njump=1):
    """Definiton of a Hill estimator via local inverse slopes."""
    n = np.size(data)
    M = np.reshape(data, (n,))
    # i) normalize and sort data in decending order
    M = np.flip(np.sort(M))  # /np.std(M)
    # ii) compute CDF
    M, cdf_M = cdf_from_spectrum(M)
    cdf_M = (1 - cdf_M).reshape((-1,))
    # iii) estimate local inverse slope
    zeta = -np.log(M[1:] / M[:-1]) / np.log(cdf_M[1:] / cdf_M[:-1])
    if njump > 1:
        zeta = zeta[: -(njump - 1)]
    for i in range(2, njump + 1):
        zeta_i = -np.log(M[i:] / M[:-i]) / np.log(cdf_M[i:] / cdf_M[:-i])
        zeta = np.vstack([zeta, zeta_i[: -(njump - i)] if njump - i > 0 else zeta_i])
    if njump > 1:
        zeta = np.mean(zeta, axis=0)
    # iv) compute running averages with window nAvg
    if avg_inverse:
        zeta_mean = 1 / movmean(zeta, navg)
    else:
        zeta_mean = movmean(1 / zeta, navg)
    g_inv = 1 / movmean(M[:-njump], navg)
    return g_inv, zeta_mean


""" Power law fit - p-value """


@dataclass
class PowerLawFitResult:
    p: float
    alpha: float
    xmin: float
    D: float
    Ds: List[float]
    ks_C: CDF
    n: int
    nTail: int
    s_min: float
    s_max: float
    fit: powerlaw.Fit

    def is_powerlaw(self):
        return self.p >= 0.1


def tail_powerlaw_fit(
    s: np.ndarray, nSamples=2500, savePath: str = None, load_path_list: list = None
):
    """Performs a power law tail fit to s and computes the p-value for the fit."""
    s = np.sort(copy.deepcopy(s))

    Ds, ks_C = powerlaw_test_statistic(
        copy.deepcopy(s),
        savePath=savePath,
        load_path_list=load_path_list,
        nSamples=nSamples,
    )

    alpha, xmin, D, fit = fit_powerlaw(s, return_fit_obj=True)

    p = 1.0 - ks_C(D)

    return PowerLawFitResult(
        fit=fit,
        p=p,
        alpha=alpha,
        xmin=xmin,
        D=D,
        Ds=Ds,
        ks_C=ks_C,
        n=np.size(s),
        nTail=np.size(s[s > xmin]),
        s_min=np.min(s),
        s_max=np.max(s),
    )


def powerlaw_test_statistic(s, savePath=None, load_path_list=None, nSamples=2500):
    """
    Saves resutlting ks distances Ds to savePath. If load_path_list is provided it loads
    Ds from all paths in this list and concatenates them. All other arguments are ignored
    if load_path_list is provided! Be careful with this feature, there are no checks if
    the loaded ks distances belong to the provides s values! This feature can be used
    to easily compute the ks statistics for a large number of samples in parallel.

    Computes the tail power law KS stistics according to [Clauset,Shalizi & Newman 2009]:
    1. fit powerlaw to tail of s to get alpha, xmin
    2. generate bootstrap samples:
        2.1 get number of values in s below xmin = nBulk, total size is n
        2.2 draw nBulk values from s[s<xmin] using bootstrapping with replacement
        2.3 draw n-nBulk values from the fitted power-law
        2.4 do this at least 2500 times to generate samples
    3. compute the KS distance of each sample to its own fit result cdf
    4. get ks statistics from all the distances D (as 1-p is the cdf from D)"""

    if load_path_list is None:
        alpha, xmin, _ = fit_powerlaw(s)

        n = np.size(s)
        s_bulk = s[s < xmin]
        nBulk = np.size(s_bulk)

        Ds = []
        for _ in tqdm(range(nSamples)):
            deciders = np.random.rand(n)
            nBulk_draw = np.sum(deciders < (nBulk / n))
            nTail_draw = n - nBulk_draw

            sample_tail = draw_power_law(alpha, xmin, shape=(nTail_draw,))
            sample_bulk = np.random.choice(s_bulk, size=nBulk_draw)
            sample = np.concatenate([sample_bulk, sample_tail])
            _, _, D = fit_powerlaw(sample)
            Ds.append(D)

        Ds = np.sort(Ds)
        if savePath is not None:
            np.save(savePath, Ds)
    else:
        Ds = np.array([])
        for path in load_path_list:
            Ds = np.concatenate([Ds, np.load(path)])
        Ds = np.sort(Ds)
    return Ds, CDF(Ds, np.arange(len(Ds)) / (len(Ds)))


def draw_power_law(alpha, xmin, shape):
    """Draw a numpy array of shape "shape" from a power law tail distribution
    with alpha and xmin."""
    pwlData = np.random.rand(*shape) ** (1 / (1 - alpha)) * xmin
    return pwlData.reshape(*shape)


def fit_powerlaw(s: np.ndarray, return_fit_obj=False):
    """Fits power law tail using the power_laws package and returns
    alpha, xmin"""
    results = powerlaw.Fit(s, xmax=np.max(s), verbose=False)
    if return_fit_obj:
        return (
            results.power_law.alpha,
            results.power_law.xmin,
            results.power_law.D,
            results,
        )
    return results.power_law.alpha, results.power_law.xmin, results.power_law.D


def power_law_cdf(x: np.ndarray, alpha: float, xmin: float):
    """Power law cdf."""
    return 1 - (x / xmin) ** (1 - alpha)


def fit_truncated_powerlaw(s: np.ndarray, return_fit_obj=False):
    """Perfrom a truncated power law fit."""
    results = PowerlawFitWithLambda(
        s, xmax=np.max(s), xmin_distribution="truncated_power_law"
    )
    if return_fit_obj:
        return (
            results.truncated_power_law.alpha,
            results.truncated_power_law.xmin,
            results.truncated_power_law.Lambda,
            results,
        )
    return (
        results.truncated_power_law.alpha,
        results.truncated_power_law.xmin,
        results.truncated_power_law.Lambda,
    )


class PowerlawFitWithLambda(powerlaw.Fit):
    """Modified powerlaw fit class to also return lambda for the truncated power law fit."""

    def __repr__(self):
        return f"fit='{self.xmin_distribution().name}', xmin={self.xmin}, alpha={self.alpha}, lambda={self.Lambda}, D={self.D}"

    def find_xmin(self, xmin_distance=None):
        """
        Returns the optimal xmin beyond which the scaling regime of the power
        law fits best. The attribute self.xmin of the Fit object is also set.

        The optimal xmin beyond which the scaling regime of the power law fits
        best is identified by minimizing the Kolmogorov-Smirnov distance
        between the data and the theoretical power law fit.
        This is the method of Clauset et al. 2007.
        """
        from numpy import unique, asarray, argmin, nan, repeat, arange

        # Much of the rest of this function was inspired by Adam Ginsburg's plfit code,
        # specifically the mapping and sigma threshold behavior:
        # http://code.google.com/p/agpy/source/browse/trunk/plfit/plfit.py?spec=svn359&r=357
        if not self.given_xmin:
            possible_xmins = self.data
        else:
            possible_ind = min(self.given_xmin) <= self.data
            possible_ind *= self.data <= max(self.given_xmin)
            possible_xmins = self.data[possible_ind]
        xmins, xmin_indices = unique(possible_xmins, return_index=True)
        # Don't look at last xmin, as that's also the xmax, and we want to at least have TWO points to fit!
        xmins = xmins[:-1]
        xmin_indices = xmin_indices[:-1]

        if xmin_distance is None:
            xmin_distance = self.xmin_distance

        if len(xmins) <= 0:
            print(
                "Less than 2 unique data values left after xmin and xmax "
                "options! Cannot fit. Returning nans.",
                file=sys.stderr,
            )
            from numpy import nan, array

            self.xmin = nan
            self.D = nan
            self.V = nan
            self.Asquare = nan
            self.Kappa = nan
            self.alpha = nan
            self.sigma = nan
            self.n_tail = nan
            setattr(self, xmin_distance + "s", array([nan]))
            self.alphas = array([nan])
            self.sigmas = array([nan])
            self.in_ranges = array([nan])
            self.xmins = array([nan])
            self.noise_flag = True
            return self.xmin

        def fit_function(xmin, idx, num_xmins):
            # print('xmin progress: {:02d}%'.format(int(idx/num_xmins * 100)), end='\r')
            pl = self.xmin_distribution(
                xmin=xmin,
                xmax=self.xmax,
                discrete=self.discrete,
                estimate_discrete=self.estimate_discrete,
                fit_method=self.fit_method,
                data=self.data,
                parameter_range=self.parameter_range,
                parent_Fit=self,
            )
            if not hasattr(pl, "sigma"):
                pl.sigma = nan
            if not hasattr(pl, "alpha"):
                pl.alpha = nan
            return (
                getattr(pl, xmin_distance),
                pl.alpha,
                pl.sigma,
                pl.in_range(),
                pl,
                pl.Lambda,
            )

        num_xmins = len(xmins)
        fits = asarray(
            list(
                map(
                    fit_function, xmins, arange(num_xmins), repeat(num_xmins, num_xmins)
                )
            )
        )
        # logging.warning(fits.shape)
        setattr(self, xmin_distance + "s", fits[:, 0])
        self.alphas = fits[:, 1]
        self.sigmas = fits[:, 2]
        self.in_ranges = fits[:, 3].astype(bool)
        self.xmins = xmins
        self.fit_objs = fits[:, 4]
        self.lambdas = fits[:, 5]

        good_values = self.in_ranges

        if self.sigma_threshold:
            good_values = good_values * (self.sigmas < self.sigma_threshold)

        if good_values.all():
            min_D_index = argmin(getattr(self, xmin_distance + "s"))
            self.noise_flag = False
        elif not good_values.any():
            min_D_index = argmin(getattr(self, xmin_distance + "s"))
            self.noise_flag = True
        else:
            from numpy.ma import masked_array

            masked_Ds = masked_array(
                getattr(self, xmin_distance + "s"), mask=~good_values
            )
            min_D_index = masked_Ds.argmin()
            self.noise_flag = False

        if self.noise_flag:
            print("No valid fits found.", file=sys.stderr)

        # Set the Fit's xmin to the optimal xmin
        self.xmin = xmins[min_D_index]
        setattr(self, xmin_distance, getattr(self, xmin_distance + "s")[min_D_index])
        self.alpha = self.alphas[min_D_index]
        self.sigma = self.sigmas[min_D_index]
        self.Lambda = self.lambdas[min_D_index]
        self._min_D_index = min_D_index

        # Update the fitting CDF given the new xmin, in case other objects, like
        # Distributions, want to use it for fitting (like if they do KS fitting)
        self.fitting_cdf_bins, self.fitting_cdf = self.cdf()

        return self.xmin
