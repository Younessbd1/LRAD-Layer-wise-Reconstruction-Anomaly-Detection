"""Decoder architectures for reconstructing images from classifier activations.

CNNDecoder: Uses transposed convolutions to upsample spatial feature maps
            back to the original image resolution. Produces pixel-aligned
            reconstruction, so the error map is a direct spatial heatmap.

MLPDecoder: Takes a 1D hidden activation vector from an MLP classifier and
            reconstructs the flattened image via linear layers, then reshapes
            to (C, H, W). The pixel-wise error after reshape gives the heatmap.
"""

import torch
import torch.nn as nn


class CNNDecoder(nn.Module):
    """Transpose-convolution decoder from a specific CNN layer back to image space.

    Mirrors the encoder's downsampling path in reverse using ConvTranspose2d.
    output_padding is computed automatically to match target spatial sizes.

    Args:
        channels: Channel dimensions from input to the layer being decoded.
                  e.g. if decoding from block 2 of [1, 16, 32, 64], pass [1, 16, 32].
        spatial_sizes: Spatial sizes at each level (including input).
                       e.g. [28, 14, 7] for a 2-block encoder on 28×28 input.
    """

    def __init__(self, channels: list[int], spatial_sizes: list[int]):
        super().__init__()
        assert len(channels) == len(spatial_sizes), (
            f"channels ({len(channels)}) and spatial_sizes ({len(spatial_sizes)}) "
            f"must have the same length"
        )

        KERNEL, STRIDE, PADDING = 3, 2, 1

        rev_ch = channels[::-1]
        rev_sz = spatial_sizes[::-1]

        layers = []
        for i in range(len(rev_ch) - 1):
            is_last = (i == len(rev_ch) - 2)

            # Compute output_padding to match target spatial size
            raw_out = (rev_sz[i] - 1) * STRIDE - 2 * PADDING + KERNEL
            out_pad = 1 if rev_sz[i + 1] != raw_out else 0

            layers.append(nn.ConvTranspose2d(
                rev_ch[i], rev_ch[i + 1],
                kernel_size=KERNEL, stride=STRIDE,
                padding=PADDING, output_padding=out_pad,
            ))

            if not is_last:
                layers.append(nn.BatchNorm2d(rev_ch[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self._source_channels = rev_ch[0]
        self._target_size = rev_sz[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode activation map back to image space. Returns (B, C, H, W)."""
        return self.net(x)

    def __repr__(self):
        return (f"CNNDecoder(in_ch={self._source_channels}, "
                f"target_size={self._target_size})")


class MLPDecoder(nn.Module):
    """Linear decoder from an MLP hidden layer back to image space.

    Reconstructs the flattened image via a symmetric series of Linear layers,
    then reshapes to (C, H, W).

    Args:
        activation_dim: Dimension of the hidden activation vector.
        output_dim: Flattened image dimension (e.g. 784 = 1×28×28).
        hidden_dims: Intermediate decoder widths (optional, for deeper decoders).
        output_shape: Tuple (C, H, W) for final reshape.
    """

    def __init__(
        self,
        activation_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        output_shape: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        self.output_shape = output_shape

        if hidden_dims is None:
            hidden_dims = []

        dims = [activation_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if not is_last:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode 1D activation to image. Returns (B, C, H, W)."""
        flat = self.net(x)
        return flat.view(-1, *self.output_shape)

    def __repr__(self):
        return f"MLPDecoder(output_shape={self.output_shape})"
