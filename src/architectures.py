import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class GeneralMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth,
                 activation_name='tanh', bias=True, dropout_p=0.0):
        super(GeneralMLP, self).__init__()

        # Mapping string to PyTorch activation modules
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid
        }
        activation = activations.get(activation_name.lower(), nn.Tanh)

        layers = []

        # 1. Input Layer
        layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        layers.append(activation())
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        # 2. Hidden Layers (Depth defines number of hidden layers)
        # Depth=1 means just Input->Hidden->Output
        # Depth=2 means Input->Hidden->Hidden->Output
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(activation())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

        # 3. Output Layer (Logits)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        # Flatten image inputs (e.g., 28x28 -> 784)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)

    def get_features(self, x):
        """Useful for few-shot evaluation later."""
        x = x.view(x.size(0), -1)
        return self.features(x)

class ResearchMLP(GeneralMLP):
    def __init__(self, *args, **kwargs):
        # Initialize parent to build standard layer stack
        super(ResearchMLP, self).__init__(*args, **kwargs)

    def get_features(self, x):
        """
        Extracts MLP features, halting BEFORE the final activation layer.
        Preserves raw directional spikes for Cosine similarity evaluation.
        """
        # 1. Flatten the input (e.g., 28x28 -> 784)
        x = x.view(x.size(0), -1)

        # 2. Identify all components in the feature extractor
        layers = list(self.features.children())

        # 3. Locate the index of the very last activation (Tanh/ReLU/Sigmoid)
        last_act_idx = -1
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], (nn.Tanh, nn.ReLU, nn.Sigmoid)):
                last_act_idx = i
                break

        # 4. Execute the pass up to, but not including, that final activation
        # This ensures the output is the raw linear projection of the final hidden layer
        if last_act_idx != -1:
            for i in range(last_act_idx):
                x = layers[i](x)
        else:
            # Fallback if no activation is found
            x = self.features(x)

        return x

    def forward(self, x):
        """
        Standard training pass using the full parent logic.
        Maintains activations for gradient stability during backprop.
        """
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)

class GeneralCNN(nn.Module):
    def __init__(self, input_channels, base_channels, num_classes, depth,
                 activation_name='tanh', dropout_p=0.5, bias=False):
        super(GeneralCNN, self).__init__()

        # 1. Setup Activations
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid
        }
        activation = activations.get(activation_name.lower(), nn.Tanh)

        layers = []
        current_channels = input_channels
        out_channels = base_channels

        # 2. Build Convolutional Feature Extractor
        for i in range(depth):
            layers.append(nn.Conv2d(current_channels, out_channels,
                                    kernel_size=3, padding=1, bias=bias))
            layers.append(activation())
            layers.append(nn.MaxPool2d(2))
            current_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_p)

        # 3. Dynamic Classifier Resolution
        # We calculate the flattening dimension based on 28x28 input
        # Each MaxPool(2) halves the H and W. For 28x28:
        # depth 1 -> 14x14 | depth 2 -> 7x7 | depth 3 -> 3x3
        spatial_dim = 28 // (2 ** depth)
        flatten_dim = current_channels * (spatial_dim ** 2)

        self.classifier = nn.Linear(flatten_dim, num_classes, bias=bias)

    def get_features(self, x):
        """Extracts flattened post-activation features for Few-Shot eval."""
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.get_features(x)
        x = self.dropout(x)
        return self.classifier(x)

class ResearchCNN(GeneralCNN):
    def __init__(self, *args, **kwargs):
        # 1. Initialize the parent exactly as defined
        super(ResearchCNN, self).__init__(*args, **kwargs)

        # Surgical Strike: Filter out only the VERY LAST MaxPool2d
        all_layers = list(self.features.children())

        # We find the index of the last MaxPool2d and remove it
        for i in range(len(all_layers) - 1, -1, -1):
            if isinstance(all_layers[i], nn.MaxPool2d):
                all_layers.pop(i)
                break

        self.features = nn.Sequential(*all_layers)

        # Recalculate dimensions: Spatial size is doubled compared to parent
        depth = kwargs.get('depth')
        spatial_dim = 28 // (2 ** (depth - 1))

        # Find the last channel count from the final conv layer in our new list
        last_conv = [m for m in all_layers if isinstance(m, nn.Conv2d)][-1]
        flatten_dim = last_conv.out_channels * (spatial_dim ** 2)

        self.classifier = nn.Linear(flatten_dim, kwargs.get('num_classes'),
                                    bias=kwargs.get('bias', False))

    def get_features(self, x):
        """
        Extracts features manually, stopping BEFORE the final activation.
        This preserves the raw spectral spikes for the Cosine evaluator.
        """
        # 1. Identify the index of the very last activation
        layers = list(self.features.children())
        last_act_idx = -1
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], (nn.Tanh, nn.ReLU)):
                last_act_idx = i
                break

        # 2. Run the forward pass up to (but not including) that last activation
        # If no activation found, just run the whole block
        if last_act_idx != -1:
            for i in range(last_act_idx):
                x = layers[i](x)
        else:
            x = self.features(x)

        return torch.flatten(x, 1)

    def forward(self, x):
        """Standard training pass: Uses the full sequence INCLUDING final activation."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

if __name__ == "__main__":
    # 1. Initialize two models for comparison
    alpha_val = 1.2
    g_val = 1.0

    model_ht = CleanCNN(alpha=alpha_val, g=g_val)
    model_gauss = CleanCNN(alpha=2.0, g=g_val) # Alpha=2.0 is exactly Gaussian

    # 2. Extract weights from the first layer
    # features[0] is the first Conv2d layer
    w_ht = model_ht.features[0].weight.detach().cpu().numpy().flatten()
    w_gauss = model_gauss.features[0].weight.detach().cpu().numpy().flatten()

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Linear scale histogram (The "Spike and Tail" view)
    axes[0].hist(w_ht, bins=100, color='crimson', alpha=0.6, label=f'Heavy-Tailed (α={alpha_val})', density=True)
    axes[0].hist(w_gauss, bins=100, color='gray', alpha=0.4, label='Gaussian (α=2.0)', density=True)
    axes[0].set_title("Weight Distribution (Linear Scale)")
    axes[0].set_yscale('log') # Log scale on Y is essential to see the tails
    axes[0].legend()

    # Right: The "Outlier" scatter (The "Fractal" view)
    # This shows the magnitude of weights across the flattened index
    axes[1].scatter(range(len(w_ht)), w_ht, s=1, color='crimson', alpha=0.5, label='HT Weights')
    axes[1].set_title("Weight Magnitudes (Outlier Check)")
    axes[1].set_ylabel("Weight Value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # 4. Print Summary Statistics
    print(f"--- Initialization Check ---")
    print(f"HT Layer 1 - Max: {w_ht.max():.4f}, Min: {w_ht.min():.4f}, Std: {w_ht.std():.4f}")
    print(f"Gaussian Layer 1 - Max: {w_gauss.max():.4f}, Min: {w_gauss.min():.4f}, Std: {w_gauss.std():.4f}")
