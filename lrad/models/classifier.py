"""Classifier architectures with intermediate activation extraction.

Both architectures expose activations from every hidden layer so that
decoders can be trained to reconstruct the input from any depth.
"""

import torch
import torch.nn as nn
from typing import Optional


class CNNClassifier(nn.Module):
    """Convolutional classifier with per-block activation extraction.

    Each block: Conv2d -> BatchNorm -> ReLU (stride=2 downsampling).
    Final: AdaptiveAvgPool -> Linear -> logits.

    Args:
        channels: List of channel dimensions, e.g. [1, 16, 32, 64].
                  channels[0] = input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        input_size: Spatial size of input images (assumed square).
    """

    def __init__(self, channels: list[int], num_classes: int, input_size: int = 28):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
            ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

        # Compute spatial sizes at each layer for decoder construction
        with torch.no_grad():
            dummy = torch.zeros(1, channels[0], input_size, input_size)
            _, acts = self._forward_with_activations(dummy)
            self.spatial_sizes = [input_size] + [a.shape[2] for a in acts]

    def _forward_with_activations(self, x: torch.Tensor):
        activations = []
        for block in self.blocks:
            x = block(x)
            activations.append(x)
        pooled = self.pool(x).flatten(1)
        logits = self.fc(pooled)
        return logits, activations

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning (logits, list_of_activations)."""
        return self._forward_with_activations(x)

    def freeze(self) -> None:
        """Freeze all parameters (no gradient computation)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return (f"CNNClassifier(channels={self.channels}, "
                f"spatial_sizes={self.spatial_sizes})")


class MLPClassifier(nn.Module):
    """Fully-connected classifier with per-layer activation extraction.

    Each hidden layer: Linear -> BatchNorm1d -> ReLU.
    Final: Linear -> logits.

    For heatmap generation, activations are 1D vectors that get reshaped
    back to spatial maps by the corresponding MLP decoder.

    Args:
        input_dim: Flattened input dimension (e.g. 784 for 28×28).
        hidden_dims: List of hidden layer widths, e.g. [512, 256, 128].
        num_classes: Number of output classes.
        input_size: Original spatial size (for metadata only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        input_size: int = 28,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.input_size = input_size

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ))

        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass. Input is (B, C, H, W), gets flattened internally."""
        x = x.flatten(1)  # (B, input_dim)
        activations = []

        for layer in self.layers:
            x = layer(x)
            activations.append(x)

        logits = self.fc(x)
        return logits, activations

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return (f"MLPClassifier(input_dim={self.input_dim}, "
                f"hidden_dims={self.hidden_dims})")
