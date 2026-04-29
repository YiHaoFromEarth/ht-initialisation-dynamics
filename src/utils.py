import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import random
import numpy as np
import logging
import json
import pandas as pd
import re
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from . import architectures
from scipy.stats import levy_stable
from pathlib import Path
from copy import deepcopy


class HookManager:
    def __init__(self):
        self.captured_output = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        # We detach to avoid interfering with the computational graph
        self.captured_output = output.detach()

    def attach(self, model, layer_path):
        """
        Attaches to a layer based on a string path (e.g., 'features.6')
        or a direct attribute name.
        """
        # Remove any existing hook before attaching a new one
        self.remove()

        # Navigate the model hierarchy to find the target module
        target_layer = model
        for part in layer_path.split("."):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)

        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        """Cleanly detaches the hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class TeeLogger:
    """Clones stdout to a file while still printing to terminal."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This is needed for compatibility with some python print behaviors
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_hooked_features(model, hook_manager, imgs):
    # Trigger the forward pass
    # The hook inside the manager catches the data before the final activation
    model(imgs)
    # Flatten the captured pre-activation experts
    return torch.flatten(hook_manager.captured_output, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Force CUDA to use deterministic algorithms (may be slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_master_config(file_path):
    """
    Parses a master YAML and returns classes and all config sections.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Dynamic Class Resolution
    model_name = config["model"]["class_name"]
    model_class = getattr(architectures, model_name)

    optim_name = config["optimizer"]["class_name"]
    optim_class = getattr(optim, optim_name)

    # 2. Extract standard dictionaries
    # We return the whole 'config' so the script can access any custom sections
    return {
        "model_class": model_class,
        "optim_class": optim_class,
        "model_params": config["model"]["params"],
        "optim_params": config["optimizer"]["params"],
        "data_config": config["data"],
        "hyperparams": config["hyperparams"],
        "full_config": config,  # The 'Extra Sections' are here
    }


def get_universal_loader(dataset_class, data_config, **dataset_kwargs):
    """
    Universal wrapper that pulls settings from a data_config dictionary.
    """
    # Extract values from the config with defaults
    batch_size = data_config.get("batch_size", 128)
    use_gpu = data_config.get("use_gpu", True)
    fast_load = data_config.get("fast_load", True)
    transform = data_config.get("transform", None)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    # Initialize raw dataset
    dataset_raw = dataset_class(transform=transform, **dataset_kwargs)

    if fast_load and torch.cuda.is_available() and use_gpu:
        print(f"Fast-loading {dataset_class.__name__} to {device}...")
        # Check if we need to convert to tensor on the fly for fast-loading
        first_item = dataset_raw[0][0]
        if not isinstance(first_item, torch.Tensor):
            from torchvision.transforms import ToTensor

            converter = ToTensor()
            imgs = torch.stack([converter(img) for img, _ in dataset_raw]).to(device)
        else:
            imgs = torch.stack([img for img, _ in dataset_raw]).to(device)
        try:
            lbls = torch.tensor([lbl for _, lbl in dataset_raw]).to(device)
        except Exception:
            lbls = torch.tensor(dataset_raw._labels).to(device)

        dataset = TensorDataset(imgs, lbls)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=dataset_kwargs.get("train", True)
        )
    else:
        loader = DataLoader(
            dataset_raw,
            batch_size=batch_size,
            shuffle=dataset_kwargs.get("train", True),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    return loader


def get_transform(transform_list):
    """Converts a list of dicts from YAML into a Compose object."""
    composed_list = []
    for item in transform_list:
        ((name, params),) = item.items()
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
            raise TypeError(
                f"The provided class {model_class.__name__} is not a torch.nn.Module."
            )

        logging.info(
            f"Successfully initialized {model_class.__name__} with {len(args)} args and {len(kwargs)} kwargs."
        )
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


def init_heavy_tailed(tensor, alpha, g, seed_offset=0, base_seed=0):
    """
    Overwrites a tensor's data with heavy-tailed weights using per-layer seeds.
    """
    with torch.no_grad():
        # 1. Calculate the effective N based on tensor shape
        if tensor.dim() == 4:  # Conv layer
            n_eff = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
        else:  # Linear layer
            n_eff = (tensor.shape[0] * tensor.shape[1]) ** 0.5

        # 2. Localized Seed: ensures unique outlier placement per layer
        # This prevents correlated 'super-paths' through the network depth.
        local_rng = np.random.RandomState(base_seed + seed_offset)

        # 3. Generate stable samples with the local RNG
        scale = g / (2 * n_eff) ** (1 / alpha)
        samples = levy_stable.rvs(
            alpha, 0, scale=scale, size=tensor.shape, random_state=local_rng
        )

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
            if "weight" in name and param.dim() >= 2:
                # Use Fan-in (input dimension) for more stable HT scaling
                # n_eff = param.shape[1]
                init_heavy_tailed(
                    param, alpha, g, seed_offset=weight_idx, base_seed=base_seed
                )
                weight_idx += 1
            elif "bias" in name:
                # Heavy tails work best when centered at zero
                param.zero_()
    return model


def get_layer_from_checkpoint(model_path, layer_key):
    """
    Retrieves a specific weight matrix from a saved checkpoint,
    automatically handles nested state_dicts, and casts to float32.

    Args:
        model_path (str or Path): Path to the .pth file.
        layer_key (str): The state_dict key (e.g., 'features.0.weight').

    Returns:
        torch.Tensor: The weight matrix as a float32 tensor on CPU.
    """
    # Load to CPU. weights_only=True is safer and faster for simple weight extraction.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # Handle nested 'model_state' or 'model_state_dict' or raw state_dict
    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, dict):
        # Fallback for other common nesting keys
        state_dict = checkpoint.get("state_dict", checkpoint)

    if layer_key not in state_dict:
        available = list(state_dict.keys())
        # Filter for only 'weight' keys to make the error message more readable
        weights_only_keys = [k for k in available if "weight" in k]
        raise KeyError(
            f"Layer '{layer_key}' not found. Available weights: {weights_only_keys}"
        )

    # 1. Detach from any graph
    # 2. .float() converts bfloat16/half/double to standard float32
    # 3. .cpu() ensures it is ready for numpy conversion
    return state_dict[layer_key].detach().float().cpu()


def get_all_layers_from_checkpoint(model_path):
    """
    Retrieves all parameter matrices (weights/biases) from a saved checkpoint
    and returns them in a dictionary for easy parsing and analysis.

    Args:
        model_path (str or Path): Path to the .pth file.

    Returns:
        dict: A dictionary mapping layer names (str) to torch.Tensors (float32 on CPU).
    """
    # 1. Load to CPU using the same safety protocols as your existing utils
    # weights_only=True is preferred for security and speed during analysis
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # 2. Extract the actual state_dict using your project's nesting logic
    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)

    # 3. Filter for parameters and standardize the format
    layers_dict = {}
    for key, tensor in state_dict.items():
        # Capturing weights and biases as they define the 'layers' in analysis
        if isinstance(tensor, torch.Tensor):
            # detach: isolates from any computational graph
            # float(): ensures bfloat16/half weights are cast to standard float32
            # cpu(): guarantees they are ready for numpy/plotting tools
            layers_dict[key] = tensor.detach().float().cpu()

    return layers_dict


def get_checkpoint_map(run_path):
    """
    Scans the checkpoints directory and returns a sorted mapping of
    epoch/step numbers to their corresponding file paths.

    Returns:
        dict: {int_step: Path_to_file} sorted by step ascending.
    """
    checkpoint_dir = Path(run_path) / "checkpoints"
    ckpt_map = {}

    if not checkpoint_dir.exists():
        return {}

    # 1. Capture intermediate epoch weights
    # Matches 'weights_epoch_10.pth' -> extract 10
    for ckpt in checkpoint_dir.glob("weights_epoch_*.pth"):
        match = re.search(r"epoch_(\d+)", ckpt.name)
        if match:
            ckpt_map[int(match.group(1))] = ckpt

    # 2. Capture the final model if it exists
    # We pull the epoch count from run_config to give it the correct 'step' key
    config_path = Path(run_path) / "run_config.json"
    if (checkpoint_dir / "final_model.pth").exists() and config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
            final_epoch = cfg["hyperparams"].get("epochs")
            ckpt_map[final_epoch] = checkpoint_dir / "final_model.pth"

    # 3. Return as a dict sorted by the keys (epochs/steps)
    return dict(sorted(ckpt_map.items()))


def collect_run_snapshots(run_path):
    """
    Scans a run folder, extracts weight matrices using modular helpers,
    and returns a DataFrame of the weights history for physics analysis.
    """
    run_path = Path(run_path)

    # 1. Retrieve the validated timeline of checkpoints
    # This replaces the manual globbing and regex matching
    ckpt_map = get_checkpoint_map(run_path)

    snapshot_data = []

    # 2. Iterate through the mapping (sorted by epoch/step)
    for epoch, file_path in ckpt_map.items():
        # 3. Extract all layers using the standardized float32/CPU helper
        layers = get_all_layers_from_checkpoint(file_path)

        for key, tensor in layers.items():
            # Physics analysis typically targets weight matrices
            if "weight" in key:
                # Convert the standardized tensor to a flattened numpy array
                snapshot_data.append({
                    "epoch": epoch,
                    "layer": key,
                    "weights": tensor.numpy().flatten()
                })

    # 4. Consolidate into a structured DataFrame
    df = pd.DataFrame(snapshot_data)

    # 5. Final Sort: Essential for sequential visualizations (e.g., Ridge Plots)
    if not df.empty:
        df = df.sort_values(by=["layer", "epoch"]).reset_index(drop=True)

    return df


def collect_sweep_learning_curves(sweep_dir):
    """
    Iterates through the sweep and pulls train/test loss and accuracy
    from every 'train_log.csv' into one master DataFrame.
    """
    sweep_path = Path(sweep_dir)
    all_run_logs = []

    # 1. Find all run_config files to locate the runs
    configs = list(sweep_path.rglob("run_config.json"))
    print(f"Found {len(configs)} training logs. Aggregating...")

    for cfg_path in configs:
        run_dir = cfg_path.parent
        log_path = run_dir / "train_log.csv"

        if not log_path.exists():
            print(f"Warning: Log missing for {run_dir.name}")
            continue

        # 2. Load Metadata
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        a_init = cfg["ht_config"].get("alpha")
        s_init = cfg["ht_config"].get("g")

        # 3. Load the actual CSV data
        run_log = pd.read_csv(log_path)

        # 4. Inject metadata into every row of this run
        run_log["alpha"] = a_init
        run_log["sigma"] = s_init

        all_run_logs.append(run_log)

    # 5. Combine into master Frame
    if not all_run_logs:
        raise ValueError("No training logs found in the provided directory.")

    master_df = pd.concat(all_run_logs, ignore_index=True)

    print(f"Success! Combined {len(master_df)} training steps into master log.")
    return master_df


def setup_experiment(config_path, checkpoint_path=None, device="cpu"):
    """
    Standardizes model and data loading using the project's factory
    and configuration structures.
    """
    # 1. Load Master Configuration
    cfg = load_master_config(config_path)
    data_cfg = cfg["data_config"]
    model_params = cfg["model_params"]
    model_input = cfg["model_class"]

    # 2. Setup Data (Transforms & Loaders)
    dataset_class = get_dataset_class(data_cfg["dataset_name"])
    data_cfg["transform"] = get_transform(data_cfg.get("transforms", []))

    loaders = {
        "train": get_universal_loader(
            dataset_class, data_cfg, train=True, download=True, root="./data"
        ),
        "test": get_universal_loader(
            dataset_class, data_cfg, train=False, download=True, root="./data"
        ),
    }

    # 3. Flexible Model Acquisition (Factory Path)
    m_args = model_params.get("args", [])
    m_kwargs = model_params.get("kwargs", {})

    # Initialize using your existing factory logic
    model = model_factory(model_input, *m_args, **m_kwargs)

    if model is None:
        raise ValueError(f"Failed to initialize model {model_input} from factory.")

    model = model.to(device)

    # 4. Load Weights (If provided)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle state_dict unpacking (supports bfp16 or full precision)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode for consistent metrics
        print(f"Successfully loaded checkpoint: {Path(checkpoint_path).name}")

    return model, loaders, cfg


def spectral_filter(
    weight_tensor, center_perc, window_size_perc, kernel_type="uniform"
):
    """
    Isolates a window of singular values using either a Rectangular or Gaussian filter.
    The window_size_perc represents the FWHM for Gaussian kernels.
    """
    U, S, Vh = torch.linalg.svd(weight_tensor, full_matrices=False)

    num_s = len(S)
    rank_axis = torch.linspace(0, 1, steps=num_s).to(S.device)

    if kernel_type == "uniform":
        # Hard cutoff: 1.0 within [center - width/2, center + width/2]
        half_width = window_size_perc / 2
        mask = (
            (rank_axis >= center_perc - half_width)
            & (rank_axis <= center_perc + half_width)
        ).float()

    elif kernel_type == "gaussian":
        # Gaussian filter where FWHM = window_size_perc
        # Relationship: sigma = FWHM / (2 * sqrt(2 * ln(2)))
        sigma = window_size_perc / (2 * np.sqrt(2 * np.log(2)))

        # We use the standard Gaussian form: exp(-(x - mu)^2 / (2 * sigma^2))
        mask = torch.exp(-((rank_axis - center_perc) ** 2) / (2 * sigma**2))

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Apply mask and reconstruct matrix
    S_filtered = S * mask
    return U @ torch.diag(S_filtered) @ Vh


def apply_spectral_filter_to_model(
    model, layer_key_func, center_perc, window_size_perc, kernel_type="uniform"
):
    """
    Applies spectral reconstruction to specific layers of a model.

    Args:
        model (nn.Module): The live model instance.
        layer_key_func (callable): A function that returns True for layers to be filtered
                                   (e.g., lambda name: 'features.0' in name).
        center_perc (float): Center of the spectral window (0.0 to 1.0).
        window_size_perc (float): FWHM or Width of the window.
        kernel_type (str): 'uniform' or 'gaussian'.

    Returns:
        nn.Module: A copy of the model with altered weights.
    """
    # Create a deep copy to avoid mutating the original model in the loop
    altered_model = deepcopy(model)
    altered_model.eval()  # Ensure eval mode for consistent behavior

    with torch.no_grad():  # No gradients needed for reconstruction analysis
        for name, module in altered_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and layer_key_func(name):
                # 1. Access the raw weight tensor
                W_orig = module.weight.data

                # 2. Call your 'Atomic' math function from equations.py
                # This performs the SVD, filtering, and reconstruction
                W_filtered = spectral_filter(
                    W_orig, center_perc, window_size_perc, kernel_type=kernel_type
                )

                # 3. Replace the weights in-place
                module.weight.copy_(W_filtered)

    return altered_model


@torch.no_grad()
def evaluate_model(model, loader, device, criterion):
    """
    Standardized evaluation for a single data loader.
    Returns a dictionary of metrics for easy logging.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Calculate final averages
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {"loss": avg_loss, "acc": accuracy}
