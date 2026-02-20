import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime
from scipy.stats import levy_stable
from torchinfo import summary
from pathlib import Path
from ml_library import (
    model_factory,
    load_master_config,
    get_universal_loader,
    get_dataset_class,
    get_transform,
    set_seed,
    optimizer_factory,
    get_hooked_features,
    HookManager,
    TeeLogger,
)

def init_heavy_tailed(tensor, alpha, g, seed_offset=0, base_seed=0):
    """
    Overwrites a tensor's data with heavy-tailed weights using per-layer seeds.
    """
    with torch.no_grad():
        # 1. Calculate the effective N based on tensor shape
        if tensor.dim() == 4:  # Conv layer
            n_eff = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
        else:  # Linear layer
            n_eff = (tensor.shape[0] * tensor.shape[1])**0.5

        # 2. Localized Seed: ensures unique outlier placement per layer
        # This prevents correlated 'super-paths' through the network depth.
        local_rng = np.random.RandomState(base_seed + seed_offset)

        # 3. Generate stable samples with the local RNG
        scale = g / (2 * n_eff)**(1/alpha)
        samples = levy_stable.rvs(alpha, 0, scale=scale, size=tensor.shape, random_state=local_rng)

        # 4. Copy to PyTorch
        tensor.copy_(torch.from_numpy(samples).float())

def apply_heavy_tailed_init(model, alpha, g, base_seed=0):
    """
    Scans a model and applies HT initialization to all weight tensors.
    """
    print(f"Applying HT Init: alpha={alpha}, g={g}, seed={base_seed}")
    with torch.no_grad():
        # Dedicated counter ensures seed consistency across different model types
        weight_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Use Fan-in (input dimension) for more stable HT scaling
                # n_eff = param.shape[1]
                init_heavy_tailed(param, alpha, g, seed_offset=weight_idx, base_seed=base_seed)
                weight_idx += 1
            elif 'bias' in name:
                # Heavy tails work best when centered at zero
                param.zero_()
    return model

def train_model_ht(model_input, ht_config, model_params, optim_class, optim_params,
                hyperparams, data_config, loaders, seed, output_root, log_freq=10):
    """
    Foundational orchestrator to initialize, train, and log an ML experiment.
    """
    # Store the original terminal handle
    original_stdout = sys.stdout
    logger = None

    try:
        # 1. Strict Determinism
        set_seed(seed)

        device = hyperparams.get('device', 'cpu')

        # 2. Flexible Model Acquisition
        if isinstance(model_input, nn.Module):
            # We were passed a pre-initialized model instance
            model = model_input
            model_name = model.__class__.__name__
        else:
            # Standard path: Initialize from class and params
            m_args = model_params.get('args', [])
            m_kwargs = model_params.get('kwargs', {})
            model = model_factory(model_input, *m_args, **m_kwargs)
            model_name = model_input.__name__
        if model is None: return None # Graceful exit on init failure

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
        if ht_config and ht_config.get('enabled', False):
            alpha = ht_config.get('alpha', 1.2)
            g = ht_config.get('g', 1.0)

            # We pass the same 'seed' used in set_seed to ensure HT weights
            # are tied to the experimental trial
            print(f"Overwriting weights with Heavy-Tailed Distribution (α={alpha}, g={g})")
            model = apply_heavy_tailed_init(model, alpha, g, base_seed=seed)

        # Now move to device after weights are set
        model = model.to(device)

        # --- 3.2 Dynamic Hook Initialization (New) ---
        # Retrieve path from hyperparams or find the last layer in model.features
        hook_path = hyperparams.get('hook_layers')

        if not hook_path:
            # Default: Target the very last index of the Sequential 'features' block
            last_idx = len(list(model.features.children())) - 1
            hook_path = f"features.{last_idx}"

        feature_hook = HookManager()
        try:
            feature_hook.attach(model, hook_path)
            print(f"Research Probe successfully attached to: {hook_path}")
        except Exception as e:
            print(f"Warning: Failed to attach hook to {hook_path}. Error: {e}")

        # Visual Hook Summary: Verifying the actual binding
        print("\n" + "="*60)
        print("RESEARCH PROBE: ARCHITECTURAL BINDING CONFIRMATION")
        print("="*60)

        # 1. Identify the physical module target
        try:
            actual_target = model.get_submodule(hook_path)
        except AttributeError:
            actual_target = model
            for part in hook_path.split('.'):
                actual_target = actual_target[int(part)] if part.isdigit() else getattr(actual_target, part)

        # 2. Print every layer in 'features' explicitly by index
        if hasattr(model, 'features'):
            for i, module in enumerate(model.features):
                name = f"features.{i}"
                is_hooked = " <--- [PROBE ATTACHED HERE]" if module is actual_target else ""
                print(f"[{name:.<25}]: {module.__class__.__name__:<15}{is_hooked}")

        # 3. Print the classifier separately
        if hasattr(model, 'classifier'):
            is_hooked = " <--- [PROBE ATTACHED HERE]" if model.classifier is actual_target else ""
            print(f"[{'classifier':.<25}]: {model.classifier.__class__.__name__:<15}{is_hooked}")

        print("="*60 + "\n")

        # 4. Optimizer Initialization (via Factory)
        optimizer = optimizer_factory(optim_class, model.parameters(), **optim_params)
        if optimizer is None: return None

        # 5. Architecture Documentation
        with open(run_dir / "architecture.txt", "w") as f:
            # Assumes MNIST/Omniglot-style 1-channel input
            stats = summary(model, input_size=(data_config['batch_size'], 1, 28, 28), verbose=0)
            f.write(str(stats))
            # Log the hooked layer for reproducibility
            f.write(f"\n\nEvaluator Hook Layer: {hook_path}\n")

        # 6. Training Loop Logic
        criterion = nn.CrossEntropyLoss()
        history = []

        for epoch in range(1, hyperparams['epochs'] + 1):
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0

                # Training Phase
                for inputs, labels in loaders['train']:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    if 'grad_clip' in hyperparams:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['grad_clip'])

                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                # 7. Periodic Logging and Test Evaluation
                if epoch % log_freq == 0 or epoch == 1:
                    model.eval()
                    test_loss, test_correct, test_total = 0.0, 0, 0

                    with torch.no_grad():
                        # A. Standard Cross-Entropy Validation
                        for inputs, labels in loaders['test']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                            test_loss += loss.item()
                            _, predicted = outputs.max(1)
                            test_total += labels.size(0)
                            test_correct += predicted.eq(labels).sum().item()

                    # B. Prototypical Few-Shot Evaluation
                    # Requires 'test_dataset' to be passed in the loaders dict
                    fs_mean = "N/A"
                    if 'test_dataset' in loaders:
                        # We run a smaller subset (100 episodes) during training for speed
                        fs_accuracies = evaluate_few_shot(
                            model,
                            feature_hook,
                            device,
                            loaders['test_dataset'],
                            n_way=5,
                            k_shot=1,
                            n_episodes=100,
                        )
                        fs_mean = np.mean(fs_accuracies)

                    metrics = {
                        'epoch': epoch,
                        'train_loss': train_loss / len(loaders['train']),
                        'train_acc': train_correct / train_total,
                        'test_loss': test_loss / len(loaders['test']),
                        'test_acc': test_correct / test_total,
                        'fs_1shot_mean': fs_mean # Added for tracking HT embedding quality
                    }

                    history.append(metrics)
                    fs_str = f" | FS 1-Shot: {fs_mean:.4f}" if isinstance(fs_mean, float) else ""
                    print(f"Epoch {epoch} | Train Acc: {metrics['train_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}{fs_str}")

        # 8. Final Artifact Export
        pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
        torch.save({
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'hyperparams': hyperparams,
            'seed': seed
        }, run_dir / "final_model.pth")

        # Inside train_model, before json.dump:
        # We create a copy so we don't break the actual 'live' dictionary
        saveable_data_config = data_config.copy()

        # Convert the Compose object into a readable string
        if 'transform' in saveable_data_config:
            saveable_data_config['transform'] = str(saveable_data_config['transform'])

        # Save exact configuration for audit trail
        config_dump = {
            'model': model_name,
            'model_params': model_params,
            'optimizer': optim_class.__name__,
            'optimizer_params': optim_params,
            'hyperparams': hyperparams,
            'data_config': saveable_data_config,
            'seed': seed
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config_dump, f, indent=4)

        # --- Extraction Logic Verification ---
        model.eval()
        with torch.no_grad():
            # Create a dummy input matching your dataset shape
            dummy_input = torch.randn(1, 1, 28, 28).to(device)
            raw_features = get_hooked_features(model, feature_hook, dummy_input)

            feat_max = raw_features.abs().max().item()

            print("\n" + "="*40)
            print("BACKBONE EXTRACTION VERIFICATION")
            print(f"Max Absolute Feature Value: {feat_max:.4f}")
            print("="*40 + "\n")

        # 9. Prototypical Few-Shot Evaluation
        print("--- Running Final Few-Shot Sweep ---")
        fs_1s = evaluate_few_shot(model, feature_hook, device, loaders['test_dataset'], n_way=5, k_shot=1, n_episodes=500)
        fs_5s = evaluate_few_shot(model, feature_hook, device, loaders['test_dataset'], n_way=5, k_shot=5, n_episodes=500)

        np.savetxt(f"{run_dir}/fs_1shot_raw.csv", fs_1s, delimiter=",")
        np.savetxt(f"{run_dir}/fs_5shot_raw.csv", fs_5s, delimiter=",")

        return model, run_dir

    except Exception as e:
        # If it crashes, print the error so it's caught in the log too
        print(f"\nFATAL ERROR during run: {e}")
        raise e # Re-raise so we see the traceback

    finally:
        # RESTORE THE TERMINAL
        sys.stdout = original_stdout
        if logger is not None:
            logger.close()
        print(f"--- Run Complete. Console logging detached ---")

if __name__ == "__main__":
    # 1. Load Master Configuration
    # Check if a config file was passed via CLI
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"
    cfg = load_master_config(config_path)
    data_cfg = cfg['data_config']
    base_output = cfg['full_config']['experiment_metadata']['output_root']

    # 2. Resolve Dataset and Transforms
    dataset_class = get_dataset_class(data_cfg['dataset_name'])
    data_cfg['transform'] = get_transform(data_cfg.get('transforms', []))

    # 3. Setup Data Loaders and Raw Dataset for Prototypical Eval
    is_omniglot = "Omniglot" in data_cfg['dataset_name']
    train_key = 'background' if is_omniglot else 'train'

    # We capture the raw test dataset to pass into the loaders dict
    test_ds = dataset_class(root='./data', **{train_key: False}, download=True, transform=data_cfg['transform'])

    loaders = {
        'train': get_universal_loader(dataset_class, data_cfg, **{train_key: True}, download=True, root='./data'),
        'test': get_universal_loader(dataset_class, data_cfg, **{train_key: False}, download=True, root='./data'),
        'test_dataset': test_ds  # Required for Prototypical Evaluation hook
    }

    # 4. Five-Seed Sweep
    seeds = range(5)
    ht_config = cfg['full_config'].get('heavy_tail', {})

    for s in seeds:
        print(f"\n--- STARTING SEED {s} ---")

        # --- RUN A: HEAVY-TAILED EXPERIMENT ---
        print(f"Running HT Experiment (alpha={ht_config['alpha']})...")

        train_model_ht(
            model_input=cfg['model_class'],
            ht_config=ht_config,
            model_params=cfg['model_params'],
            optim_class=cfg['optim_class'],
            optim_params=cfg['optim_params'],
            hyperparams=cfg['hyperparams'],
            data_config=cfg['data_config'],
            loaders=loaders,
            seed=s,
            output_root=f"{base_output}/HT_alpha_{ht_config['alpha']}",
            log_freq=cfg['hyperparams'].get('log_freq', 10)
        )

        # --- RUN B: GAUSSIAN BASELINE (alpha=2.0) ---
        print(f"Running Gaussian Baseline (alpha=2.0)...")
        # Reuse the same g but force alpha to 2.0
        gauss_config = ht_config.copy()
        gauss_config['alpha'] = 2.0

        train_model_ht(
            model_input=cfg['model_class'],
            ht_config=gauss_config,
            model_params=cfg['model_params'],
            optim_class=cfg['optim_class'],
            optim_params=cfg['optim_params'],
            hyperparams=cfg['hyperparams'],
            data_config=cfg['data_config'],
            loaders=loaders,
            seed=s,
            output_root=f"{base_output}/Gaussian_Baseline",
            log_freq=cfg['hyperparams'].get('log_freq', 10)
        )
