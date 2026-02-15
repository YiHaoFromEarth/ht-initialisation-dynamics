import torch.nn as nn
import logging
import random
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import json
import architectures
import yaml
from pathlib import Path
from torchinfo import summary
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU

    # Force CUDA to use deterministic algorithms (may be slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_master_config(file_path):
    """
    Parses a master YAML and returns classes and all config sections.
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Dynamic Class Resolution
    model_name = config['model']['class_name']
    model_class = getattr(architectures, model_name)

    optim_name = config['optimizer']['class_name']
    optim_class = getattr(optim, optim_name)

    # 2. Extract standard dictionaries
    # We return the whole 'config' so the script can access any custom sections
    return {
        'model_class': model_class,
        'optim_class': optim_class,
        'model_params': config['model']['params'],
        'optim_params': config['optimizer']['params'],
        'data_config': config['data'],
        'hyperparams': config['hyperparams'],
        'full_config': config # The 'Extra Sections' are here
    }

def get_universal_loader(dataset_class, data_config, **dataset_kwargs):
    """
    Universal wrapper that pulls settings from a data_config dictionary.
    """
    # Extract values from the config with defaults
    batch_size = data_config.get('batch_size', 128)
    use_gpu = data_config.get('use_gpu', True)
    fast_load = data_config.get('fast_load', True)
    transform = data_config.get('transform', None)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    # Initialize raw dataset
    dataset_raw = dataset_class(transform=transform, **dataset_kwargs)

    if fast_load and torch.cuda.is_available() and use_gpu:
        print(f"Fast-loading {dataset_class.__name__} to {device}...")
        imgs = torch.stack([img for img, _ in dataset_raw]).to(device)
        try:
            lbls = torch.tensor([lbl for _, lbl in dataset_raw]).to(device)
        except Exception:
            lbls = torch.tensor(dataset_raw._labels).to(device)

        dataset = TensorDataset(imgs, lbls)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=dataset_kwargs.get('train', True))
    else:
        loader = DataLoader(
            dataset_raw,
            batch_size=batch_size,
            shuffle=dataset_kwargs.get('train', True),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return loader

def get_transform(transform_list):
    """Converts a list of dicts from YAML into a Compose object."""
    composed_list = []
    for item in transform_list:
        (name, params), = item.items()
        t_class = getattr(transforms, name)

        if params is None:
            composed_list.append(t_class())
        elif isinstance(params, list):
            composed_list.append(t_class(*params))
        elif isinstance(params, dict):
            composed_list.append(t_class(**params))

    return transforms.Compose(composed_list)

def get_dataset_class(name):
    """Maps string names to torchvision dataset classes."""
    return getattr(datasets, name)

def model_factory(model_class, *args, **kwargs):
    """
    Universal factory to instantiate any PyTorch model with graceful error handling.

    Args:
        model_class: The class definition (not an instance).
        *args: Positional arguments for the model constructor.
        **kwargs: Keyword arguments for the model constructor.

    Returns:
        Instance of model_class or None if instantiation fails.
    """
    try:
        # Standard instantiation
        model = model_class(*args, **kwargs)

        # Verify it's actually a PyTorch module
        if not isinstance(model, nn.Module):
            raise TypeError(f"The provided class {model_class.__name__} is not a torch.nn.Module.")

        logging.info(f"Successfully initialized {model_class.__name__} with {len(args)} args and {len(kwargs)} kwargs.")
        return model

    except TypeError as e:
        logging.error(f"Argument mismatch for {model_class.__name__}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error initializing {model_class.__name__}: {e}")
        return None

def optimizer_factory(optimizer_class, model_params, **kwargs):
    """
    Modular factory to instantiate PyTorch optimizers.

    Args:
        optimizer_class: The class (e.g., optim.SGD, optim.Adam).
        model_params: model.parameters() from the initialized model.
        **kwargs: Hyperparameters like lr, momentum, weight_decay, etc.

    Returns:
        An instantiated optimizer.
    """
    try:
        # Standard instantiation: first arg is always the model parameters
        optimizer = optimizer_class(model_params, **kwargs)

        logging.info(f"Initialized {optimizer_class.__name__} with {kwargs}")
        return optimizer

    except TypeError as e:
        logging.error(f"Hyperparameter mismatch for {optimizer_class.__name__}: {e}")
        # Hint: This usually happens if you pass 'momentum' to Adam or 'betas' to SGD
        return None
    except Exception as e:
        logging.error(f"Failed to initialize optimizer {optimizer_class.__name__}: {e}")
        return None

def train_model(model_input, model_params, optim_class, optim_params,
                hyperparams, data_config, loaders, seed, output_root, log_freq=10):
    """
    Foundational orchestrator to initialize, train, and log an ML experiment.
    """
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
    model = model.to(device)

    # 3. Directory and Metadata Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_LR{optim_params.get('lr', 'N/A')}_BS{data_config['batch_size']}_{timestamp}_s{seed}"
    run_dir = Path(output_root) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4. Optimizer Initialization (via Factory)
    optimizer = optimizer_factory(optim_class, model.parameters(), **optim_params)
    if optimizer is None: return None

    # 5. Architecture Documentation
    with open(run_dir / "architecture.txt", "w") as f:
        # Assumes MNIST/Omniglot-style 1-channel input
        stats = summary(model, input_size=(data_config['batch_size'], 1, 28, 28), verbose=0)
        f.write(str(stats))

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
                model.eval() # Set model to evaluation mode (Freezes Dropout/Batchnorm)
                test_loss, test_correct, test_total = 0.0, 0, 0

                with torch.no_grad(): # Disable gradient calculation for efficiency
                    for inputs, labels in loaders['test']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()

                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss / len(loaders['train']),
                    'train_acc': train_correct / train_total,
                    'test_loss': test_loss / len(loaders['test']),
                    'test_acc': test_correct / test_total
                }

                history.append(metrics)
                print(f"Epoch {epoch} | Train Acc: {metrics['train_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}")

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

    return model, run_dir

if __name__ == "__main__":
    from architectures import GeneralMLP
    # SINGLE SOURCE OF TRUTH
    data_config = {
        'dataset_name': 'MNIST',
        'batch_size': 128,
        'use_gpu': True,
        'fast_load': True,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

    # Use it for Loader Init
    loaders = {
        'train': get_universal_loader(datasets.MNIST, data_config, root='./data', train=True, download=True),
        'test': get_universal_loader(datasets.MNIST, data_config, root='./data', train=False, download=True)
    }

    # 2. Define Params (The "What it is")
    model_params = {
        'kwargs': {
            'input_size': 784,
            'hidden_size': 512,
            'num_classes': 10,
            'depth': 2,
            'activation_name': 'tanh',
            'dropout_p': 0.2
        }
    }

    # 3. Define Hyperparams (The "How it learns")
    hyperparams = {
        'epochs': 5,
        'batch_size': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'grad_clip': 1.0
    }

    # 4. Define Optimizer Params
    optim_class = optim.SGD
    optim_params = {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    }

    # 5. Execute Run
    print(f"Starting test run on {hyperparams['device']}...")

    try:
        model, run_dir = train_model(
            model_input=GeneralMLP,
            model_params=model_params,
            optim_class=optim_class,
            optim_params=optim_params,
            hyperparams=hyperparams,
            data_config=data_config,
            loaders=loaders,
            seed=42,
            output_root='./results_test',
            log_freq=1
        )

        print(f"\nTest successful!")
        print(f"Results saved to: {run_dir}")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
