import torch

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
