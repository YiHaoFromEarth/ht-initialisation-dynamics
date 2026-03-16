import torch
import torch.nn as nn
import pandas as pd
import json
import sys
import itertools
import yaml
from datetime import datetime
from torchinfo import summary
from pathlib import Path
from tqdm import tqdm
from .utils import (
    model_factory,
    set_seed,
    optimizer_factory,
    apply_heavy_tailed_init,
    setup_experiment,
    TeeLogger,
)


def train_model(
    model_input,
    model_params,
    optim_class,
    optim_params,
    hyperparams,
    data_config,
    loaders,
    seed,
    output_root,
    log_freq=10,
):
    """
    Foundational orchestrator to initialize, train, and log an ML experiment.
    """
    # Store the original terminal handle
    original_stdout = sys.stdout
    logger = None

    try:
        # 1. Strict Determinism
        set_seed(seed)

        device = hyperparams.get("device", "cpu")

        # 2. Flexible Model Acquisition
        if isinstance(model_input, nn.Module):
            # We were passed a pre-initialized model instance
            model = model_input
            model_name = model.__class__.__name__
        else:
            # Standard path: Initialize from class and params
            m_args = model_params.get("args", [])
            m_kwargs = model_params.get("kwargs", {})
            model = model_factory(model_input, *m_args, **m_kwargs)
            model_name = model_input.__name__
        if model is None:
            return None  # Graceful exit on init failure
        model = model.to(device)

        # 3. Directory and Metadata Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{model_name}_LR{optim_params.get('lr', 'N/A')}_BS{data_config['batch_size']}_{timestamp}_s{seed}"
        run_dir = Path(output_root) / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Start logging console output to file
        sys.stdout = TeeLogger(run_dir / "console_output.txt")
        print(f"Logging initialized in: {run_dir}")

        # 4. Optimizer Initialization (via Factory)
        optimizer = optimizer_factory(optim_class, model.parameters(), **optim_params)
        if optimizer is None:
            return None

        # 5. Architecture Documentation
        with open(run_dir / "architecture.txt", "w") as f:
            # Assumes MNIST/Omniglot-style 1-channel input
            stats = summary(
                model, input_size=(data_config["batch_size"], 1, 28, 28), verbose=0
            )
            f.write(str(stats))

        # 6. Training Loop Logic
        criterion = nn.CrossEntropyLoss()
        history = []

        for epoch in range(1, hyperparams["epochs"] + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            # Training Phase
            for inputs, labels in loaders["train"]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                if "grad_clip" in hyperparams:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hyperparams["grad_clip"]
                    )

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # 7. Periodic Logging and Test Evaluation
            if epoch % log_freq == 0 or epoch == 1:
                model.eval()  # Set model to evaluation mode (Freezes Dropout/Batchnorm)
                test_loss, test_correct, test_total = 0.0, 0, 0

                with torch.no_grad():  # Disable gradient calculation for efficiency
                    for inputs, labels in loaders["test"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()

                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss / len(loaders["train"]),
                    "train_acc": train_correct / train_total,
                    "test_loss": test_loss / len(loaders["test"]),
                    "test_acc": test_correct / test_total,
                }

                history.append(metrics)
                print(
                    f"Epoch {epoch} | Train Acc: {metrics['train_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}"
                )

        # 8. Final Artifact Export
        pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "hyperparams": hyperparams,
                "seed": seed,
            },
            run_dir / "final_model.pth",
        )

        # Inside train_model, before json.dump:
        # We create a copy so we don't break the actual 'live' dictionary
        saveable_data_config = data_config.copy()

        # Convert the Compose object into a readable string
        if "transform" in saveable_data_config:
            saveable_data_config["transform"] = str(saveable_data_config["transform"])

        # Save exact configuration for audit trail
        config_dump = {
            "model": model_name,
            "model_params": model_params,
            "optimizer": optim_class.__name__,
            "optimizer_params": optim_params,
            "hyperparams": hyperparams,
            "data_config": saveable_data_config,
            "seed": seed,
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config_dump, f, indent=4)

        return model, run_dir

    except Exception as e:
        # If it crashes, print the error so it's caught in the log too
        print(f"\nFATAL ERROR during run: {e}")
        raise e  # Re-raise so we see the traceback

    finally:
        # RESTORE THE TERMINAL
        sys.stdout = original_stdout
        if logger is not None:
            logger.close()
        print(f"--- Run Complete. Console logging detached ---")


def train_model_ht(
    model_input,
    ht_config,
    model_params,
    optim_class,
    optim_params,
    hyperparams,
    data_config,
    loaders,
    seed,
    output_root,
    log_freq=10,
):
    """
    Foundational orchestrator to initialize, train, and log an ML experiment.
    """

    def save_half_precision(state_dict, path):
        """Helper to cast weights to bfp16 before saving to disk."""
        half_state = {
            k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
            for k, v in state_dict.items()
        }
        torch.save(half_state, path)

    # Store the original terminal handle
    original_stdout = sys.stdout
    logger = None

    try:
        # 1. Strict Determinism
        set_seed(seed)

        device = hyperparams.get("device", "cpu")

        # 2. Flexible Model Acquisition
        if isinstance(model_input, nn.Module):
            # We were passed a pre-initialized model instance
            model = model_input
            model_name = model.__class__.__name__
        else:
            # Standard path: Initialize from class and params
            m_args = model_params.get("args", [])
            m_kwargs = model_params.get("kwargs", {})
            model = model_factory(model_input, *m_args, **m_kwargs)
            model_name = model_input.__name__
        if model is None:
            return None  # Graceful exit on init failure

        # 3. Directory and Metadata Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{model_name}_LR{optim_params.get('lr', 'N/A')}_BS{data_config['batch_size']}_{timestamp}_s{seed}"
        run_dir = Path(output_root) / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Start logging console output to file
        sys.stdout = TeeLogger(run_dir / "console_output.txt")
        print(f"Logging initialized in: {run_dir}")

        # 3.1. Apply Heavy-Tailed Initialization (The Research Hook)
        # We check if ht_config exists; if not, it stays as standard Gaussian/Kaiming
        if ht_config and ht_config.get("enabled", False):
            alpha = ht_config.get("alpha", 1.2)
            g = ht_config.get("g", 1.0)

            # We pass the same 'seed' used in set_seed to ensure HT weights
            # are tied to the experimental trial
            print(
                f"Overwriting weights with Heavy-Tailed Distribution (α={alpha}, g={g})"
            )
            model = apply_heavy_tailed_init(model, alpha, g, base_seed=seed)

        # Now move to device after weights are set
        model = model.to(device)

        # 4. Optimizer Initialization (via Factory)
        optimizer = optimizer_factory(optim_class, model.parameters(), **optim_params)
        if optimizer is None:
            return None

        # 5. Architecture Documentation
        with open(run_dir / "architecture.txt", "w") as f:
            # Dynamically infer input shape from the data loader
            # This prevents hard-coding (1, 28, 28) for MNIST vs (3, 32, 32) for CIFAR-10
            try:
                # Get one batch from the training loader
                example_data, _ = next(iter(loaders["train"]))
                # Extract shape: (BatchSize, Channels, Height, Width)
                input_shape = example_data.shape
            except (KeyError, StopIteration):
                # Fallback to config if loader is unavailable
                # Assumes 3072 for CIFAR-10 flattened or (3, 32, 32)
                batch_size = data_config.get("batch_size", 32)
                if data_config.get("dataset_name") == "CIFAR-10":
                    input_shape = (batch_size, 3, 32, 32)
                else:
                    input_shape = (batch_size, 1, 28, 28)
            stats = summary(model, input_size=input_shape, verbose=0)
            f.write(str(stats))

        # 5.1. Initial Weight Snapshot
        if hyperparams.get("save_weights_history", False) and 0 in hyperparams.get(
            "weight_log_epochs", []
        ):
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            save_half_precision(
                model.state_dict(), checkpoint_dir / f"weights_epoch_0.pth"
            )

        # 6. Training Loop Logic
        criterion = nn.CrossEntropyLoss()
        history = []

        # --- Epoch 0: Initial Evaluation ---
        model.eval()
        train_loss, train_correct, train_total = 0.0, 0, 0
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            # Evaluate on train set
            for inputs, labels in loaders["train"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # Evaluate on test set
            for inputs, labels in loaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        metrics_0 = {
            "epoch": 0,
            "train_loss": train_loss / len(loaders["train"]),
            "train_acc": train_correct / train_total,
            "test_loss": test_loss / len(loaders["test"]),
            "test_acc": test_correct / test_total,
        }
        history.append(metrics_0)
        sys.stdout.log.write(
            f"Epoch 0 | Train Acc: {metrics_0['train_acc']:.4f} | Test Acc: {metrics_0['test_acc']:.4f}\n"
        )
        sys.stdout.log.flush()

        pbar = tqdm(
            range(1, hyperparams["epochs"] + 1),
            file=sys.stdout.terminal,  # Write bar to original terminal
            desc=f"alpha={ht_config.get('alpha', 'N/A')} g={ht_config.get('g', 'N/A')}",
        )
        for epoch in pbar:
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            # Training Phase
            for inputs, labels in loaders["train"]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                if "grad_clip" in hyperparams:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hyperparams["grad_clip"]
                    )

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            current_train_acc = train_correct / train_total
            current_train_loss = train_loss / len(loaders["train"])

            # 7. Periodic Logging and Test Evaluation
            if epoch % log_freq == 0:
                model.eval()
                test_loss, test_correct, test_total = 0.0, 0, 0

                with torch.no_grad():
                    # A. Standard Cross-Entropy Validation
                    for inputs, labels in loaders["test"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()

                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss / len(loaders["train"]),
                    "train_acc": train_correct / train_total,
                    "test_loss": test_loss / len(loaders["test"]),
                    "test_acc": test_correct / test_total,
                }

                history.append(metrics)
                sys.stdout.log.write(
                    f"Epoch {epoch} | Train Acc: {metrics['train_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}\n"
                )
                sys.stdout.log.flush()

            # --- New: Periodic Weight Saving for Physics Analysis ---
            if hyperparams.get(
                "save_weights_history", False
            ) and epoch in hyperparams.get("weight_log_epochs", []):
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                save_half_precision(
                    model.state_dict(), checkpoint_dir / f"weights_epoch_{epoch}.pth"
                )

            pbar.set_postfix(
                {
                    "T-Acc": f"{current_train_acc:.4f}",
                    "Loss": f"{current_train_loss:.3f}",
                }
            )

        # 8. Final Artifact Export
        pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "hyperparams": hyperparams,
                "seed": seed,
            },
            run_dir / "checkpoints/final_model.pth",
        )

        # Inside train_model, before json.dump:
        # We create a copy so we don't break the actual 'live' dictionary
        saveable_data_config = data_config.copy()

        # Convert the Compose object into a readable string
        if "transform" in saveable_data_config:
            saveable_data_config["transform"] = str(saveable_data_config["transform"])

        # Save exact configuration for audit trail
        config_dump = {
            "model": model_name,
            "model_params": model_params,
            "optimizer": optim_class.__name__,
            "optimizer_params": optim_params,
            "hyperparams": hyperparams,
            "data_config": saveable_data_config,
            "ht_config": ht_config,
            "seed": seed,
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config_dump, f, indent=4)

        return model, run_dir

    except Exception as e:
        # If it crashes, print the error so it's caught in the log too
        print(f"\nFATAL ERROR during run: {e}")
        raise e  # Re-raise so we see the traceback

    finally:
        # RESTORE THE TERMINAL
        sys.stdout = original_stdout
        if logger is not None:
            logger.close()
        print(f"--- Run Complete. Console logging detached ---")


def run_experiment(config_path="config.yaml", num_seeds=1):
    """
    Executes an experiment sweep by reading a config file.

    Args:
        config_path (str/Path): Path to the master YAML configuration.
        num_seeds (int): Number of consecutive seeds to run in the sweep.
    """
    # 1. Load Master Configuration
    _, loaders, cfg = setup_experiment(
        config_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    base_output = cfg["full_config"]["experiment_metadata"]["output_root"]

    # 4. Multi-Seed Sweep Execution
    starting_seed = cfg["full_config"]["hyperparams"].get("seed", 0)
    seeds = range(starting_seed, starting_seed + num_seeds)
    ht_config = cfg["full_config"].get("heavy_tail", {})

    print(f"Starting experiment with {num_seeds} seeds, outputting to: {base_output}")
    for s in seeds:
        print(f"\n--- STARTING SEED {s} ---")

        # --- RUN A: HEAVY-TAILED EXPERIMENT ---
        print(f"Running HT Experiment (alpha={ht_config['alpha']})...")
        train_model_ht(
            model_input=cfg["model_class"],
            ht_config=ht_config,
            model_params=cfg["model_params"],
            optim_class=cfg["optim_class"],
            optim_params=cfg["optim_params"],
            hyperparams=cfg["hyperparams"],
            data_config=cfg["data_config"],
            loaders=loaders,
            seed=s,
            output_root=f"{base_output}/HT_alpha_{ht_config['alpha']}",
            log_freq=cfg["hyperparams"].get("log_freq", 10),
        )


def run_parameter_sweep(config_path="sweep_config.yaml", num_seeds=1):
    """
    Iterates through a grid of alpha and g values across multiple seeds.
    Saves results into structured subfolders: output_root/alpha_g/seed_k/
    """
    # 1. Load Configuration and Setup Environment
    _, loaders, cfg = setup_experiment(
        config_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    base_output = Path(cfg["full_config"]["experiment_metadata"]["output_root"])

    # 2. Extract Sweep Parameters from YAML
    ht_settings = cfg["full_config"].get("heavy_tail", {})
    alpha_list = ht_settings.get("alpha", [2.0])
    g_list = ht_settings.get("g", [1.0])

    # Standardize to lists for iteration
    if not isinstance(alpha_list, list):
        alpha_list = [alpha_list]
    if not isinstance(g_list, list):
        g_list = [g_list]

    starting_seed = cfg["full_config"]["hyperparams"].get("seed", 0)
    seeds = range(starting_seed, starting_seed + num_seeds)

    # 3. Execution: Seed -> (Alpha, G) Grid
    sweep_grid = list(itertools.product(alpha_list, g_list))
    print(
        f"Total Sweep Load: {num_seeds} seeds * {len(sweep_grid)} pairs = {num_seeds * len(sweep_grid)} runs."
    )

    for s in seeds:
        print(f"\n{'=' * 20}\nSTARTING SEED: {s}\n{'=' * 20}")

        for alpha, g in sweep_grid:
            run_label = f"alpha_{alpha}_g_{g}"
            output_dir = base_output / run_label

            print(f"\n>>> Running: {run_label} | Seed: {s}")

            current_ht_config = ht_settings.copy()
            current_ht_config["alpha"] = alpha
            current_ht_config["g"] = g

            # Execute training using the verified library wrapper
            train_model_ht(
                model_input=cfg["model_class"],
                ht_config=current_ht_config,
                model_params=cfg["model_params"],
                optim_class=cfg["optim_class"],
                optim_params=cfg["optim_params"],
                hyperparams=cfg["hyperparams"],
                data_config=cfg["data_config"],
                loaders=loaders,
                seed=s,
                output_root=str(output_dir),
                log_freq=cfg["hyperparams"].get("log_freq", 10),
            )

    print(f"\nUniversality sweep complete. Results archived in: {base_output}")


if __name__ == "__main__":
    # 1. Load Master Configuration
    # Check if a config file was passed via CLI
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ht_config = cfg.get("heavy_tail", {})
    alpha = ht_config.get("alpha", 1.2)
    g = ht_config.get("g", 1.0)

    if isinstance(alpha, list) or isinstance(g, list):
        run_parameter_sweep(
            config_path, num_seeds=int(sys.argv[2]) if len(sys.argv) > 2 else 1
        )
    else:
        run_experiment(
            config_path, num_seeds=int(sys.argv[2]) if len(sys.argv) > 2 else 1
        )
