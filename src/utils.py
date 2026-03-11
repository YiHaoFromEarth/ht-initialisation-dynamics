import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import random
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from . import architectures
from scipy.stats import levy_stable
from pathlib import Path

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
        for part in layer_path.split('.'):
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

def setup_experiment(config_path, checkpoint_path=None, device='cpu'):
    """
    Standardizes model and data loading using the project's factory
    and configuration structures.
    """
    # 1. Load Master Configuration
    cfg = load_master_config(config_path)
    data_cfg = cfg['data_config']
    model_params = cfg['model_params']
    model_input = cfg['model_class']

    # 2. Setup Data (Transforms & Loaders)
    dataset_class = get_dataset_class(data_cfg['dataset_name'])
    data_cfg['transform'] = get_transform(data_cfg.get('transforms', []))

    loaders = {
        'train': get_universal_loader(dataset_class, data_cfg, train=True, download=True, root='./data'),
        'test': get_universal_loader(dataset_class, data_cfg, train=False, download=True, root='./data')
    }

    # 3. Flexible Model Acquisition (Factory Path)
    m_args = model_params.get('args', [])
    m_kwargs = model_params.get('kwargs', {})

    # Initialize using your existing factory logic
    model = model_factory(model_input, *m_args, **m_kwargs)

    if model is None:
        raise ValueError(f"Failed to initialize model {model_input} from factory.")

    model = model.to(device)

    # 4. Load Weights (If provided)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle state_dict unpacking (supports bfp16 or full precision)
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode for consistent metrics
        print(f"Successfully loaded checkpoint: {Path(checkpoint_path).name}")

    return model, loaders, cfg

def spectral_filter(weight_tensor, center_perc, window_size_perc, kernel_type='uniform'):
    """
    Isolates a window of singular values using either a Rectangular or Gaussian filter.
    The window_size_perc represents the FWHM for Gaussian kernels.
    """
    U, S, Vh = torch.linalg.svd(weight_tensor, full_matrices=False)

    num_s = len(S)
    rank_axis = torch.linspace(0, 1, steps=num_s).to(S.device)

    if kernel_type == 'uniform':
        # Hard cutoff: 1.0 within [center - width/2, center + width/2]
        half_width = window_size_perc / 2
        mask = ((rank_axis >= center_perc - half_width) &
                (rank_axis <= center_perc + half_width)).float()

    elif kernel_type == 'gaussian':
        # Gaussian filter where FWHM = window_size_perc
        # Relationship: sigma = FWHM / (2 * sqrt(2 * ln(2)))
        sigma = window_size_perc / (2 * np.sqrt(2 * np.log(2)))

        # We use the standard Gaussian form: exp(-(x - mu)^2 / (2 * sigma^2))
        mask = torch.exp(-((rank_axis - center_perc)**2) / (2 * sigma**2))

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Apply mask and reconstruct matrix
    S_filtered = S * mask
    return U @ torch.diag(S_filtered) @ Vh
