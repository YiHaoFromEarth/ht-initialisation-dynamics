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
from .analysis import ModelTracker
from .utils import (
    model_factory,
    set_seed,
    optimizer_factory,
    apply_heavy_tailed_init,
    setup_experiment,
    evaluate_model,
    TeeLogger,
)


class Callback:
    """Base class for all training extensions."""

    def on_train_begin(self, model=None, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_before_step(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, epoch=None, model=None, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = [c for c in (callbacks or []) if c is not None]

    def fire(self, hook_name, **kwargs):
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                method(**kwargs)


class TAMSDCallback(Callback):
    def __init__(self, model, track_step=True, track_epoch=True):
        self.step_tracker = (
            ModelTracker(model, lags=[1, 2, 4, 8, 16, 32]) if track_step else None
        )
        self.epoch_tracker = (
            ModelTracker(model, lags=[1, 2, 4, 8, 16, 32, 64, 128])
            if track_epoch
            else None
        )

    def on_before_step(self, model=None, **kwargs):
        if self.step_tracker:
            grads = torch.cat(
                [
                    p.grad.detach().view(-1)
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
            self.step_tracker.update(model, flat_grads=grads)

    def on_epoch_end(self, epoch=None, model=None, **kwargs):
        if self.epoch_tracker:
            self.epoch_tracker.update(model)

    def on_train_end(self, **kwargs):
        if not self.step_tracker or not self.epoch_tracker:
            return

        seed = kwargs.get("seed", "N/A")
        ht_config = kwargs.get("ht_config", {})
        run_dir = kwargs.get("run_dir", Path("."))

        a_val, s_val = ht_config.get("alpha", "N/A"), ht_config.get("g", "N/A")

        df_step = self.step_tracker.to_dataframe(a_val, s_val, seed, scale=1)
        df_epoch = self.epoch_tracker.to_dataframe(
            a_val, s_val, seed, scale=60000 // 1024
        )
        df = pd.concat([df_step, df_epoch], ignore_index=True)

        float_cols = [
            "net_drift",
            "abs_mean_dist",
            "rms_dist",
            "l_inf_dist",
            "cos_dist",
            "snr",
            "grad_weight_alignment",
        ]
        df[float_cols] = df[float_cols].astype("float32")

        for col in ["layer", "alpha_init", "sigma_init", "seed"]:
            df[col] = df[col].astype("category")

        df.to_parquet(
            run_dir / "displacement_log.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        print(f"Physics artifacts saved to {run_dir}")


class WeightHistoryCallback(Callback):
    def __init__(self, run_dir, weight_log_epochs):
        self.checkpoint_dir = Path(run_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.weight_log_epochs = weight_log_epochs

    def _save(self, model, epoch):
        """Helper to avoid duplicating the bfloat16 casting logic."""
        state = {
            k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
            for k, v in model.state_dict().items()
        }
        torch.save(state, self.checkpoint_dir / f"weights_epoch_{epoch}.pth")

    def on_train_begin(self, model=None, **kwargs):
        if 0 in self.weight_log_epochs:
            self._save(model, 0)

    def on_epoch_end(self, epoch=None, model=None, **kwargs):
        if epoch in self.weight_log_epochs:
            self._save(model, epoch)


class ProgressCallback(Callback):
    def __init__(self, epochs, log_freq, loaders, device, criterion, ht_config):
        self.log_freq = log_freq
        self.loaders = loaders
        self.device = device
        self.criterion = criterion
        self.history = []
        desc = f"alpha={ht_config.get('alpha', 'N/A')} g={ht_config.get('g', 'N/A')}"
        self.pbar = tqdm(range(1, epochs + 1), desc=desc, file=sys.stdout.terminal)

    def on_train_begin(self, model=None, **kwargs):
        train_m = evaluate_model(
            model, self.loaders["train"], self.device, self.criterion
        )
        test_m = evaluate_model(
            model, self.loaders["test"], self.device, self.criterion
        )
        self._record_and_print(0, train_m, test_m)

    def on_epoch_end(self, epoch=None, model=None, **kwargs):
        metrics = kwargs.get("metrics", {})
        self.pbar.set_postfix(
            {
                "T-Acc": f"{metrics['train_acc']:.4f}",
                "Loss": f"{metrics['train_loss']:.3f}",
            }
        )
        self.pbar.update(1)

        if epoch % self.log_freq == 0:
            test_m = evaluate_model(
                model, self.loaders["test"], self.device, self.criterion
            )
            self._record_and_print(epoch, metrics, test_m)

    def _record_and_print(self, epoch, train_m, test_m):
        m = {
            "epoch": epoch,
            "train_loss": train_m.get("train_loss", train_m.get("loss")),
            "train_acc": train_m.get("train_acc", train_m.get("acc")),
            "test_loss": test_m["loss"],
            "test_acc": test_m["acc"],
        }
        self.history.append(m)
        sys.stdout.log.write(
            f"Epoch {epoch} | Train Acc: {m['train_acc']:.4f} | Test Acc: {m['test_acc']:.4f}\n"
        )
        sys.stdout.log.flush()

    def on_train_end(self, **kwargs):
        run_dir = kwargs.get("run_dir", Path("."))
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        config_dump = kwargs.get("config_dump", {})

        pd.DataFrame(self.history).to_csv(run_dir / "train_log.csv", index=False)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
            },
            run_dir / "checkpoints/final_model.pth",
        )

        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config_dump, f, indent=4)


def train_single_epoch(model, loader, optimizer, criterion, device, epoch, callbacks):
    """
    Core training loop for a single epoch.
    Add-ons are handled via callbacks.
    """
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    callbacks.fire("on_epoch_begin", epoch=epoch)

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        callbacks.fire("on_batch_begin", batch_idx=batch_idx)

        # 1. Forward Pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 2. Backward Pass
        loss.backward()

        # 3. The Modifier Hook (Grad Clipping / TAMSD live here)
        # We pass the model so callbacks can access model.parameters()
        callbacks.fire("on_before_step", model=model)

        # 4. Weight Update
        optimizer.step()

        # 5. Internal Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # 6. Post-Step Hook (Batch logging)
        callbacks.fire("on_batch_end", model=model, loss=loss.item())

    # Final epoch calculations
    avg_loss = train_loss / len(loader)
    avg_acc = train_correct / train_total

    # 7. Final Hook (Validation / Checkpointing / Heavy-Tailed Analysis)
    # We pass metrics as a dict so callbacks can use them for logging
    metrics = {"train_loss": avg_loss, "train_acc": avg_acc}
    callbacks.fire("on_epoch_end", epoch=epoch, model=model, metrics=metrics)

    return metrics


def setup_experiment_dir(output_root, model_name, lr, batch_size, seed):
    """Handles folder creation and console logging setup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_LR{lr}_BS{batch_size}_{timestamp}_s{seed}"
    run_dir = Path(output_root) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = run_dir / "console_output.txt"
    sys.stdout = TeeLogger(log_file)
    print(f"Logging initialized in: {run_dir}")

    return run_dir


def build_research_model(model_input, model_params, ht_config, seed, device):
    """Initializes the model and applies custom research initializations."""
    if isinstance(model_input, nn.Module):
        model = model_input
    else:
        m_args = model_params.get("args", [])
        m_kwargs = model_params.get("kwargs", {})
        model = model_factory(model_input, *m_args, **m_kwargs)

    if ht_config and ht_config.get("enabled"):
        print(f"Applying Heavy-Tailed Init (α={ht_config.get('alpha')})")
        model = apply_heavy_tailed_init(
            model, ht_config["alpha"], ht_config["g"], base_seed=seed
        )

    return model.to(device)


def document_architecture(model, run_dir, loaders=None, data_config=None):
    """
    Summarizes the model architecture and saves it to a text file.
    Infers input shape from the data loader or falls back to config.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Infer input shape
    input_shape = None

    # Try to get shape from actual data first (most robust)
    if loaders and "train" in loaders:
        try:
            example_data, _ = next(iter(loaders["train"]))
            input_shape = example_data.shape
        except (StopIteration, AttributeError):
            pass

    # 2. Fallback logic if loader failed or wasn't provided
    if input_shape is None:
        data_config = data_config or {}
        batch_size = data_config.get("batch_size", 32)

        if data_config.get("dataset_name") == "CIFAR-10":
            input_shape = (batch_size, 3, 32, 32)
        else:
            # Default to MNIST-style
            input_shape = (batch_size, 1, 28, 28)

    # 3. Generate and save summary
    try:
        stats = summary(model, input_size=input_shape, verbose=0)
        with open(run_dir / "architecture.txt", "w") as f:
            f.write(str(stats))
        print(f"Architecture documented at {run_dir / 'architecture.txt'}")
    except Exception as e:
        print(f"Failed to document architecture: {e}")


def train_model(
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
    track_model=False,
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

        model_name = (
            model_input.__name__
            if not isinstance(model_input, nn.Module)
            else model_input.__class__.__name__
        )
        run_dir = setup_experiment_dir(
            output_root,
            model_name,
            optim_params.get("lr"),
            data_config["batch_size"],
            torch.seed,
        )

        model = build_research_model(model_input, model_params, ht_config, seed, device)
        optimizer = optimizer_factory(optim_class, model.parameters(), **optim_params)
        criterion = nn.CrossEntropyLoss()

        document_architecture(model, run_dir, loaders=loaders, data_config=data_config)

        logger_cb = ProgressCallback(
            hyperparams["epochs"], log_freq, loaders, device, criterion, ht_config
        )

        callbacks = CallbackList(
            [
                logger_cb,
                TAMSDCallback(model) if track_model else None,
                WeightHistoryCallback(run_dir, hyperparams.get("weight_log_epochs", []))
                if hyperparams.get("save_weights_history")
                else None,
            ]
        )

        callbacks.fire("on_train_begin", config=ht_config)

        for epoch in range(1, hyperparams["epochs"] + 1):
            train_single_epoch(
                model, loaders["train"], optimizer, criterion, device, epoch, callbacks
            )

        # 8. Final Artifact Export
        saveable_config = {
            "model": model_name,
            "model_params": model_params,
            "ht_config": ht_config,
            "data_config": {
                k: (str(v) if k == "transform" else v) for k, v in data_config.items()
            },
        }

        callbacks.fire(
            "on_train_end",
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            config_dump=saveable_config,
            ht_config=ht_config,
            seed=seed,
        )

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
        print("--- Run Complete. Console logging detached ---")


def run_experiment(config_path="config.yaml", num_seeds=1, start_seed=None):
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
    if start_seed is None:
        start_seed = cfg["full_config"]["hyperparams"].get("seed", 0)
    seeds = range(start_seed, start_seed + num_seeds)
    ht_config = cfg["full_config"].get("heavy_tail", {})

    print(f"Starting experiment with {num_seeds} seeds, outputting to: {base_output}")
    for s in seeds:
        print(f"\n--- STARTING SEED {s} ---")

        # --- RUN A: HEAVY-TAILED EXPERIMENT ---
        print(f"Running HT Experiment (alpha={ht_config['alpha']})...")
        train_model(
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


def run_parameter_sweep(config_path="sweep_config.yaml", num_seeds=1, start_seed=None):
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

    if start_seed is None:
        start_seed = cfg["full_config"]["hyperparams"].get("seed", 0)
    seeds = range(start_seed, start_seed + num_seeds)

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
            train_model(
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
                track_model=cfg["hyperparams"].get("track_model", False),
            )

    print(f"\nUniversality sweep complete. Results archived in: {base_output}")


if __name__ == "__main__":
    # Argument 1: Config Path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    # Argument 2: Starting Seed (The "offset" for PBS Arrays)
    # Default to the seed in the YAML if not provided
    start_seed = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Argument 3: Number of seeds to run in this specific job
    # Default to 1 (perfect for PBS Arrays)
    num_to_run = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ht_config = cfg.get("heavy_tail", {})
    alpha = ht_config.get("alpha", 1.2)
    g = ht_config.get("g", 1.0)

    if isinstance(alpha, list) or isinstance(g, list):
        run_parameter_sweep(config_path, num_seeds=num_to_run, start_seed=start_seed)
    else:
        run_experiment(config_path, num_seeds=num_to_run, start_seed=start_seed)
