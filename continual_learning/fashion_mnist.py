import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import itertools
import pandas as pd
import gc  # Added for memory cleanup
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Referencing your project structure verbatim
sys.path.append(os.path.abspath(".."))
from src.architectures import GeneralMLP # Verbatim reference to architectures.py
from src.utils import set_seed, apply_heavy_tailed_init # Verbatim reference to utils.py

def save_physics_snapshot(model, input_batch, output_dir, t_idx, epoch, alpha, g):
    model.eval()

    # Capture physics data as before
    pre_acts = model.get_pre_activations(input_batch)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    layer_physics = {}
    for idx, layer in enumerate(linear_layers):
        layer_key = f"layer_{idx}" if idx < len(linear_layers) - 1 else "classifier"
        h = pre_acts[idx] if idx < len(linear_layers) - 1 else pre_acts["classifier"]

        # Compute Jacobian using float32 for precision
        W = layer.weight.data.float()
        d_act = 1.0 - torch.tanh(h).pow(2).float()
        D_avg = d_act.mean(dim=0)
        J = D_avg.unsqueeze(1) * W

        layer_physics[layer_key] = {
            "pre_activations": h.float().cpu(),
            "jacobian": J.cpu()
        }

    snapshot = {
        "metadata": {"task": t_idx + 1, "epoch": epoch, "alpha": alpha, "g": g},
        "state_dict": model.state_dict(), # Native PyTorch format
        "physics": layer_physics         # Your specific research metrics
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"snapshot_T{t_idx+1}_E{epoch}.pt"
    torch.save(snapshot, file_path)
    return file_path

def update_GPM_bases(model, images, threshold, feature_list=None):
    model.eval()
    with torch.no_grad():
        reps = model.get_pre_activations(images)

    all_inputs = [images]
    for h in list(reps.values())[:-1]:
        all_inputs.append(torch.tanh(h))

    if feature_list is None:
        feature_list = [None] * len(all_inputs)

    current_ranks = []
    for i, activation in enumerate(all_inputs):
        X = activation.cpu().numpy()
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        s_sq = S**2
        s_sum = np.sum(s_sq)

        # Vectorized threshold calculation (cleaner than the manual loop)
        cumulative_variance = np.cumsum(s_sq) / s_sum
        k = np.argmax(cumulative_variance >= threshold) + 1
        current_ranks.append(k) # Store k for logging

        task_basis = Vh[:k].T

        if feature_list[i] is None:
            feature_list[i] = task_basis
        else:
            # We must bring the existing GPU tensor back to CPU numpy for the SVD merge
            old_basis = feature_list[i].cpu().numpy()
            combined = np.concatenate((old_basis, task_basis), axis=1)
            U_new, _, _ = np.linalg.svd(combined, full_matrices=False)
            feature_list[i] = U_new[:, :min(combined.shape[0], combined.shape[1])]

    # CRITICAL: Pre-load the bases to the GPU here, NOT in the training loop
    device = next(model.parameters()).device
    tensor_bases = [
        torch.tensor(b, dtype=torch.float32, device=device) if b is not None else None
        for b in feature_list
    ]

    return tensor_bases, current_ranks


def apply_GPM_projection(linear_layers, feature_list):
    """
    Args:
        linear_layers: A pre-cached list of nn.Linear modules.
        feature_list: The pre-loaded GPU tensors from update_GPM_bases.
    """
    if feature_list is None:
        return

    with torch.no_grad():
        for i, layer in enumerate(linear_layers):
            if i >= len(feature_list) or feature_list[i] is None:
                continue

            grad = layer.weight.grad
            if grad is None:
                continue

            basis = feature_list[i] # Already a GPU tensor!

            # Math: g_proj = g - (g @ B @ B.T)
            proj_grad = grad - torch.mm(torch.mm(grad, basis), basis.t())
            layer.weight.grad.copy_(proj_grad)

# --- 1. CONFIGURATION ---
num_tasks = 5  # Standard for Split Fashion-MNIST (2 classes per task)
seeds = [42, 69, 67, 0, 1]  # Multiple seeds for robustness
alpha_list = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
g_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
depth = 9
hidden_size = 784
bias = False
activation_name = "tanh"
optimiser = "sgd"
batch_size = 128
lr = 1e-3
epochs = 30
snapshot_epochs = [0, 14, 29]
GPM_THRESHOLD = 0.97

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA LOADING & SPLITTING ---
def get_split_tasks(dataset, num_tasks=5):
    """Filters dataset into subsets of 2 classes per task."""
    imgs = torch.stack([img for img, _ in dataset]).to(device).view(-1, 784)
    lbls = torch.tensor([lbl for _, lbl in dataset]).to(device)

    task_bundles = []
    # Create 5 tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
    for i in range(num_tasks):
        c1, c2 = 2*i, 2*i + 1
        mask = (lbls == c1) | (lbls == c2)
        task_bundles.append((imgs[mask], lbls[mask]))
    return task_bundles

print("Fast-loading Fashion-MNIST to GPU and splitting by class...")
f_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
f_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())

# Pre-split the data into 5 tasks
train_tasks = get_split_tasks(f_train, num_tasks)
test_tasks = get_split_tasks(f_test, num_tasks)

# --- 3. GRID SWEEP EXECUTION ---
for seed in seeds:
    print(f"\n=== Starting experiments for seed {seed} ===")
    for alpha, g in itertools.product(alpha_list, g_list):
        set_seed(seed) # Assuming your set_seed function is defined globally
        run_name = f"split_fashion_alpha_{alpha}_g_{g}_lr_{lr}"
        output_dir = Path(f"../continual_learning/sweep_fashion_gpm{GPM_THRESHOLD}_fc{depth+1}/{run_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata Logging
        config_params = {
            "alpha": alpha, "g": g, "seed": seed, "depth": depth,
            "hidden_size": hidden_size, "lr": lr, "batch_size": batch_size,
            "activation": activation_name, "num_tasks": num_tasks,
            "snapshot_epochs": snapshot_epochs, "bias": bias,
            "scenario": "Split Fashion-MNIST"
        }
        with open(output_dir / f"run_config_seed_{seed}.json", "w") as f:
            json.dump(config_params, f, indent=4)

        # Note: Ensure GeneralMLP and apply_heavy_tailed_init are available in your scope
        model = GeneralMLP(784, hidden_size, 10, depth, activation_name, bias=bias).to(device)
        apply_heavy_tailed_init(model, alpha=alpha, g=g, base_seed=seed)

        # 1. CACHE THE LAYERS ONCE
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        results_history = []
        gpm_feature_bases = None

        for t_idx in range(num_tasks):
            print(f"\n[{run_name}] --- Task {t_idx + 1} ---")
            curr_train_imgs, curr_train_lbls = train_tasks[t_idx]
            train_loader = DataLoader(TensorDataset(curr_train_imgs, curr_train_lbls),
                                    batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                model.train()
                total_train_loss, train_correct, train_total = 0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    # --- NEW: GPM PROJECTION STEP ---
                    if t_idx > 0:
                        apply_GPM_projection(linear_layers, gpm_feature_bases)

                    optimizer.step()

                    total_train_loss += loss.item()
                    train_correct += (outputs.argmax(1) == labels).sum().item()
                    train_total += labels.size(0)

                # Snapshot saving (Capturing first batch of current task)
                # if epoch in snapshot_epochs:
                #     save_physics_snapshot(model, curr_train_imgs[:batch_size],
                #                         output_dir / "checkpoints", t_idx, epoch, alpha, g)

                # Evaluation phase: Check current and all previous tasks
                model.eval()
                epoch_metrics = {
                    "alpha": alpha, "g": g, "epoch": epoch + 1, "task_id": t_idx + 1,
                    "train_loss": total_train_loss / len(train_loader),
                    "train_acc": train_correct / train_total,
                }

                # Pre-fill columns for all possible 5 tasks
                for i in range(num_tasks):
                    epoch_metrics[f"task_{i+1}_acc"] = np.nan

                with torch.no_grad():
                    for prev_t_idx in range(t_idx + 1):
                        val_imgs, val_lbls = test_tasks[prev_t_idx]
                        outputs = model(val_imgs)
                        acc = (outputs.argmax(1) == val_lbls).float().mean().item()
                        epoch_metrics[f"task_{prev_t_idx+1}_acc"] = acc

                results_history.append(epoch_metrics)

                # Live tracking printout
                # print(f"Ep {epoch+1:02d} | T1: {epoch_metrics['task_1_acc']:.4f} | "
                #     f"Curr: {epoch_metrics[f'task_{t_idx+1}_acc']:.4f} | Loss: {epoch_metrics['train_loss']:.4f}")

            # --- AFTER TASK COMPLETION: UPDATE GPM MEMORY ---
            print(f"Updating GPM memory for Task {t_idx+1}...")
            # Use a small subset (e.g., 300 samples) of the current task for SVD
            svd_samples = curr_train_imgs[:300]
            gpm_feature_bases, task_ranks = update_GPM_bases(model, svd_samples, GPM_THRESHOLD, gpm_feature_bases)

            if results_history:
                for i, r in enumerate(task_ranks):
                    results_history[-1][f"layer_{i}_rank"] = r

                # Also log a "Total Capacity" metric (sum of ranks across layers)
                results_history[-1]["total_rank_sum"] = sum(task_ranks)

            # Printout for immediate feedback
            # for i, r in enumerate(task_ranks):
            #     print(f"  Layer {i} Rank: {r}")

        # Save and Cleanup
        df = pd.DataFrame(results_history)
        df.to_csv(output_dir / f"results_log_seed_{seed}.csv", index=False)

        del model, optimizer, results_history, df
        torch.cuda.empty_cache()
        gc.collect()
