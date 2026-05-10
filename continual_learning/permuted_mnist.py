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
from src.architectures import GeneralMLP  # Verbatim reference to architectures.py
from src.utils import (
    set_seed,
    apply_heavy_tailed_init,
)  # Verbatim reference to utils.py


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
            "jacobian": J.cpu(),
        }

    snapshot = {
        "metadata": {"task": t_idx + 1, "epoch": epoch, "alpha": alpha, "g": g},
        "state_dict": model.state_dict(),  # Native PyTorch format
        "physics": layer_physics,  # Your specific research metrics
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"snapshot_T{t_idx + 1}_E{epoch}.pt"
    torch.save(snapshot, file_path)
    return file_path


# --- 1. CONFIGURATION ---
num_tasks = 10
seeds = [42]  # Multiple seeds for robustness
alpha_list = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
g_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
depth = 9
hidden_size = 784
bias = False
activation_name = "tanh"
optimiser = "sgd"
batch_size = 128
lr = 1e-2
epochs = 10
snapshot_epochs = [9]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. OPTIMIZED DATA LOADING ---
def get_gpu_data(dataset):
    """Loads raw tensors to GPU once to avoid repetitive overhead[cite: 7]."""
    imgs = torch.stack([img for img, _ in dataset]).to(device).view(-1, 784)
    lbls = torch.tensor([lbl for _, lbl in dataset]).to(device)
    return imgs, lbls


print("Fast-loading MNIST to GPU...")
mnist_train = datasets.MNIST(
    "./data", train=True, download=True, transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    "./data", train=False, download=True, transform=transforms.ToTensor()
)

train_imgs, train_lbls = get_gpu_data(mnist_train)
test_imgs_raw, test_lbls = get_gpu_data(mnist_test)


def generate_permutations(num_tasks, num_pixels=784, seed=42):
    rng = np.random.RandomState(seed)
    perms = [torch.arange(num_pixels)]
    for _ in range(num_tasks - 1):
        perms.append(torch.from_numpy(rng.permutation(num_pixels)))
    return perms


# --- 3. GRID SWEEP EXECUTION ---
for seed in seeds:
    print(f"\n=== Starting experiments for seed {seed} ===")
    permutations = generate_permutations(num_tasks, seed=seed)

    # SPEED OPTIMIZATION: Pre-permute the test set for every task once.
    # This removes the pixel-shuffling bottleneck inside the triple-nested loop.
    permuted_test_bundles = []
    for p in permutations:
        permuted_test_bundles.append(test_imgs_raw[:, p])

    for alpha, g in itertools.product(alpha_list, g_list):
        set_seed(seed)  # Strict determinism per run
        run_name = f"alpha_{alpha}_g_{g}_lr_{lr}"
        output_dir = Path(f"/import/taiji1/yhao4499/honours/training_runs/sweep_permuted_fc{depth + 1}/{run_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata Logging
        config_params = {
            "alpha": alpha,
            "g": g,
            "seed": seed,
            "depth": depth,
            "hidden_size": hidden_size,
            "lr": lr,
            "batch_size": batch_size,
            "activation": activation_name,
            "num_tasks": num_tasks,
            "snapshot_epochs": snapshot_epochs,
            "bias": bias,
        }
        with open(output_dir / f"run_config_seed_{seed}.json", "w") as f:
            json.dump(config_params, f, indent=4)

        model = GeneralMLP(784, hidden_size, 10, depth, activation_name, bias=bias).to(
            device
        )
        apply_heavy_tailed_init(model, alpha=alpha, g=g, base_seed=seed)

        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        results_history = []

        snapshot_inputs = train_imgs[:batch_size, permutations[0]]
        save_physics_snapshot(
            model, snapshot_inputs, output_dir / "checkpoints", 0, 0, alpha, g
        )

        for t_idx in range(num_tasks):
            print(f"\n[{run_name}] --- Task {t_idx + 1}/{num_tasks} ---")

            # Prepare training loader for current permutation
            curr_train_ds = TensorDataset(
                train_imgs[:, permutations[t_idx]], train_lbls
            )
            train_loader = DataLoader(
                curr_train_ds, batch_size=batch_size, shuffle=True
            )

            for epoch in range(epochs):
                model.train()
                total_train_loss, train_correct, train_total = 0, 0, 0

                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    train_correct += (outputs.argmax(1) == labels).sum().item()
                    train_total += labels.size(0)

                # Snapshot saving
                if epoch in snapshot_epochs:
                    # Use a fixed batch from the current task for snapshot consistency
                    snapshot_inputs = train_imgs[:batch_size, permutations[t_idx]]
                    save_physics_snapshot(
                        model,
                        snapshot_inputs,
                        output_dir / "checkpoints",
                        t_idx,
                        epoch,
                        alpha,
                        g,
                    )

                # Evaluation phase: Track ALL tasks
                model.eval()

                # FIX: Initialize all task columns with NaN to ensure consistent CSV structure
                epoch_metrics = {
                    "alpha": alpha,
                    "g": g,
                    "epoch": epoch + 1,
                    "task_id": t_idx + 1,
                    "train_loss": total_train_loss / len(train_loader),
                    "train_acc": train_correct / train_total,
                }
                # Pre-fill columns for all possible tasks
                for i in range(num_tasks):
                    epoch_metrics[f"task_{i + 1}_acc"] = np.nan

                with torch.no_grad():
                    for prev_t_idx in range(t_idx + 1):
                        # Use the pre-permuted test tensors for maximum speed
                        inputs = permuted_test_bundles[prev_t_idx]
                        labels = test_lbls

                        # Batch evaluation for speed
                        outputs = model(inputs)
                        acc = (outputs.argmax(1) == labels).float().mean().item()
                        epoch_metrics[f"task_{prev_t_idx + 1}_acc"] = acc

                results_history.append(epoch_metrics)

                # Live tracking printout
                print(
                    f"Ep {epoch + 1:02d} | T1: {epoch_metrics['task_1_acc']:.4f} | "
                    f"Curr: {epoch_metrics[f'task_{t_idx + 1}_acc']:.4f} | Loss: {epoch_metrics['train_loss']:.4f}"
                )

        # Final Save and Memory Cleanup
        df = pd.DataFrame(results_history)
        df.to_csv(output_dir / f"results_log_seed_{seed}.csv", index=False)

        del model, optimizer, results_history, df
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Run {run_name} complete and memory cleared.")
