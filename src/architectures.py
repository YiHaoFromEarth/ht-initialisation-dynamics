import torch
import torch.nn.functional as F
from torch import nn


class GeneralMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 784,
        num_classes: int = 10,
        depth: int = 9,
        activation_name: str = "tanh",
        bias: bool = False,
        dropout_p: float = 0.0,
    ):
        """
        Total linear layers = depth + 1 (Indices: 0, 1, ..., depth).
        - Layer 0: Input -> Hidden
        - Layer 1 to depth-1: Hidden -> Hidden
        - Layer depth: Hidden -> num_classes (Readout)
        """
        super().__init__()
        self.depth = depth
        self.num_layers = depth + 1

        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        act_cls = activations.get(activation_name.lower(), nn.Tanh)

        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout_p > 0 else None

        # Build all layers uniformly from 0 to depth
        for i in range(self.num_layers):
            d_in = input_size if i == 0 else hidden_size
            d_out = num_classes if i == self.depth else hidden_size

            self.linears.append(nn.Linear(d_in, d_out, bias=bias))

            # Non-linearities apply to all layers EXCEPT the final output logits
            if i < self.depth:
                self.activations.append(act_cls())
                if dropout_p > 0:
                    self.dropouts.append(nn.Dropout(dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i in range(self.num_layers):
            x = self.linears[i](x)
            if i < self.depth:
                x = self.activations[i](x)
                if self.dropouts is not None:
                    x = self.dropouts[i](x)
        return x

    def get_layer_inputs(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Returns {layer_idx: x^{l-1}} where layer_inputs[l] is the exact matrix fed
        INTO self.linears[l]. Used directly for GPM basis construction.
        """
        x = x.view(x.size(0), -1)
        layer_inputs = {}
        curr = x
        for i in range(self.num_layers):
            layer_inputs[i] = curr.detach()
            curr = self.linears[i](curr)
            if i < self.depth:
                curr = self.activations[i](curr)
                if self.dropouts is not None:
                    curr = self.dropouts[i](curr)
        return layer_inputs

    def get_pre_activations(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Returns {layer_idx: h^l} where pre_activations[l] = W^l x^{l-1}.
        Used directly for diagonal derivative quenching D^l = diag(phi'(h^l)).
        """
        x = x.view(x.size(0), -1)
        pre_acts = {}
        curr = x
        for i in range(self.num_layers):
            h = self.linears[i](curr)
            pre_acts[i] = h.detach()
            if i < self.depth:
                curr = self.activations[i](h)
                if self.dropouts is not None:
                    curr = self.dropouts[i](curr)
            else:
                curr = h
        return pre_acts


class AlexNetCIFAR(nn.Module):
    """5-layer AlexNet architecture for CIFAR-100 adapted for Continual Learning (GPM).

    Supports configurable activation functions ('relu' or 'tanh').
    """

    def __init__(
        self,
        num_classes=100,
        activation="relu",
        bias=False,
        use_dropout=True,
    ):
        super().__init__()
        self.bias = bias
        self.use_dropout = use_dropout
        self.activation_name = activation.lower()

        if self.activation_name == "relu":
            self.act = F.relu
        elif self.activation_name == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. Choose 'relu' or 'tanh'."
            )

        # Stage 1: Conv1 (3x32x32 -> 64x29x29 -> MaxPool 64x14x14)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.2) if use_dropout else nn.Identity()

        # Stage 2: Conv2 (64x14x14 -> 128x12x12 -> MaxPool 128x6x6)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout(0.2) if use_dropout else nn.Identity()

        # Stage 3: Conv3 (128x6x6 -> 256x5x5 -> MaxPool 256x2x2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Stage 4: FC1 (256*2*2 = 1024 -> 2048)
        self.fc1 = nn.Linear(256 * 2 * 2, 2048, bias=bias)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.drop4 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Stage 5: FC2 (2048 -> 2048)
        self.fc2 = nn.Linear(2048, 2048, bias=bias)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.drop5 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Classification Head (No BatchNorm)
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extracts penultimate representations (after FC2 post-activations)."""
        x = self.drop1(self.pool(self.act(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool(self.act(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool(self.act(self.bn3(self.conv3(x)))))

        x = torch.flatten(x, 1)

        x = self.drop4(self.act(self.bn_fc1(self.fc1(x))))
        x = self.drop5(self.act(self.bn_fc2(self.fc2(x))))
        return x

    def get_layer_inputs(self, x):
        """Captures the exact input representation fed INTO each parametric layer."""
        layer_inputs = {}

        # 1. Conv1 Input [B, 3, 32, 32]
        layer_inputs["conv1"] = x.detach()
        x = self.drop1(self.pool(self.act(self.bn1(self.conv1(x)))))

        # 2. Conv2 Input [B, 64, 14, 14]
        layer_inputs["conv2"] = x.detach()
        x = self.drop2(self.pool(self.act(self.bn2(self.conv2(x)))))

        # 3. Conv3 Input [B, 128, 6, 6]
        layer_inputs["conv3"] = x.detach()
        x = self.drop3(self.pool(self.act(self.bn3(self.conv3(x)))))

        # Flatten spatial feature map (256 * 2 * 2 = 1024)
        x = torch.flatten(x, 1)

        # 4. FC1 Input [B, 1024]
        layer_inputs["fc1"] = x.detach()
        x = self.drop4(self.act(self.bn_fc1(self.fc1(x))))

        # 5. FC2 Input [B, 2048]
        layer_inputs["fc2"] = x.detach()
        x = self.drop5(self.act(self.bn_fc2(self.fc2(x))))

        # 6. Classifier Input [B, 2048]
        layer_inputs["classifier"] = x.detach()

        return layer_inputs

    def get_pre_activations(self, x):
        """Captures the pre-activation outputs (h = Wx) before BN and non-linearity."""
        pre_activations = {}

        h1 = self.conv1(x)
        pre_activations["conv1"] = h1.detach()
        x = self.drop1(self.pool(self.act(self.bn1(h1))))

        h2 = self.conv2(x)
        pre_activations["conv2"] = h2.detach()
        x = self.drop2(self.pool(self.act(self.bn2(h2))))

        h3 = self.conv3(x)
        pre_activations["conv3"] = h3.detach()
        x = self.drop3(self.pool(self.act(self.bn3(h3))))

        x = torch.flatten(x, 1)

        h_fc1 = self.fc1(x)
        pre_activations["fc1"] = h_fc1.detach()
        x = self.drop4(self.act(self.bn_fc1(h_fc1)))

        h_fc2 = self.fc2(x)
        pre_activations["fc2"] = h_fc2.detach()
        x = self.drop5(self.act(self.bn_fc2(h_fc2)))

        pre_activations["classifier"] = self.classifier(x).detach()

        return pre_activations


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(
        self,
        num_classes=100,
        in_channels=3,
        base_planes=64,
        bias=False,
        dropout_p=0.0,
    ):
        super().__init__()
        self.in_planes = base_planes
        self.dropout_p = dropout_p

        # 1. CIFAR Stem: 3x3 Conv, stride 1, no MaxPool to preserve spatial resolution
        self.conv1 = nn.Conv2d(
            in_channels,
            base_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(base_planes)

        # 2. Residual Stages
        self.layer1 = self._make_layer(
            BasicBlock, base_planes, num_blocks=2, stride=1, bias=bias
        )
        self.layer2 = self._make_layer(
            BasicBlock, base_planes * 2, num_blocks=2, stride=2, bias=bias
        )
        self.layer3 = self._make_layer(
            BasicBlock, base_planes * 4, num_blocks=2, stride=2, bias=bias
        )
        self.layer4 = self._make_layer(
            BasicBlock, base_planes * 8, num_blocks=2, stride=2, bias=bias
        )

        # 3. Global Pooling & Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(
            base_planes * 8 * BasicBlock.expansion, num_classes, bias=True
        )

    def _make_layer(self, block, planes, num_blocks, stride, bias):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.get_features(x)
        return self.classifier(feat)

    def get_features(self, x):
        """Extracts penultimate pooled and flattened representation before the classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return out

    def get_layer_inputs(self, x):
        """Captures the exact input representation matrix (x) fed INTO every

        parametric Conv2d and Linear layer. Keys match named module paths.
        """
        layer_inputs = {}

        # Stem Conv1 input
        layer_inputs["conv1"] = x.detach()
        out = F.relu(self.bn1(self.conv1(x)))

        # Process Stages (layer1 to layer4)
        stages = [
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
        ]

        for stage_name, stage in stages:
            for b_idx, block in enumerate(stage):
                block_prefix = f"{stage_name}.{b_idx}"

                # 1. Conv1 input inside residual block
                layer_inputs[f"{block_prefix}.conv1"] = out.detach()
                conv1_out = F.relu(block.bn1(block.conv1(out)))

                # 2. Conv2 input inside residual block
                layer_inputs[f"{block_prefix}.conv2"] = conv1_out.detach()
                conv2_out = block.bn2(block.conv2(conv1_out))

                # 3. Shortcut Conv input (if downsampling projection exists)
                if len(block.shortcut) > 0:
                    layer_inputs[f"{block_prefix}.shortcut.0"] = out.detach()
                    shortcut_out = block.shortcut(out)
                else:
                    shortcut_out = block.shortcut(out)

                out = F.relu(conv2_out + shortcut_out)

        # Global Pooling + Classifier Input
        out_pooled = self.avgpool(out)
        out_flat = torch.flatten(out_pooled, 1)
        out_flat = self.dropout(out_flat)

        layer_inputs["classifier"] = out_flat.detach()
        return layer_inputs

    def get_pre_activations(self, x):
        """Captures pre-activation outputs (h = Wx) before Batch Normalization,

        ReLU activations, or pooling operations.
        """
        pre_activations = {}

        # Stem Conv1
        h_conv1 = self.conv1(x)
        pre_activations["conv1"] = h_conv1.detach()
        out = F.relu(self.bn1(h_conv1))

        # Process Stages
        stages = [
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
        ]

        for stage_name, stage in stages:
            for b_idx, block in enumerate(stage):
                block_prefix = f"{stage_name}.{b_idx}"

                h_c1 = block.conv1(out)
                pre_activations[f"{block_prefix}.conv1"] = h_c1.detach()
                conv1_out = F.relu(block.bn1(h_c1))

                h_c2 = block.conv2(conv1_out)
                pre_activations[f"{block_prefix}.conv2"] = h_c2.detach()
                conv2_out = block.bn2(h_c2)

                if len(block.shortcut) > 0:
                    h_sc = block.shortcut[0](out)
                    pre_activations[f"{block_prefix}.shortcut.0"] = h_sc.detach()
                    shortcut_out = block.shortcut[1](h_sc)
                else:
                    shortcut_out = block.shortcut(out)

                out = F.relu(conv2_out + shortcut_out)

        out_pooled = self.avgpool(out)
        out_flat = torch.flatten(out_pooled, 1)
        out_flat = self.dropout(out_flat)

        pre_activations["classifier"] = self.classifier(out_flat).detach()
        return pre_activations
