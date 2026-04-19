"""Nested (single-block) decoders for the cascade architecture.

Unlike the parallel decoders in `decoder.py`, each nested decoder D_k inverts
ONE classifier block only:
  - D_0: a_0 → image           (only D_0 reaches pixel space)
  - D_k: a_k → a_{k-1}          for k >= 1

To reconstruct from level k at inference, decoders are chained:
    x̂_k = D_0(D_1(...D_k(a_k)))

This makes each decoder small (one ConvTranspose / one Linear) and forces the
heavy lifting of "going back to pixel space" to be learned exactly once, by D_0.
"""

import torch
import torch.nn as nn


class CNNNestedDecoder(nn.Module):
    """Single ConvTranspose2d block that inverts one classifier block.

    Args:
        in_channels: channels of input activation a_k.
        out_channels: channels of target a_{k-1} (or image if is_image_target).
        in_size: spatial size of a_k.
        out_size: spatial size of target.
        is_image_target: True only for D_0. Adds Sigmoid to bound output to [0,1].
                         For intermediate decoders, no output activation is used —
                         we let MSE pull raw values toward the post-BN+ReLU target.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        is_image_target: bool = False,
    ):
        super().__init__()
        KERNEL, STRIDE, PADDING = 3, 2, 1

        raw_out = (in_size - 1) * STRIDE - 2 * PADDING + KERNEL
        out_pad = 1 if out_size != raw_out else 0

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=KERNEL, stride=STRIDE,
            padding=PADDING, output_padding=out_pad,
        )
        self.activation = nn.Sigmoid() if is_image_target else nn.Identity()

        self.is_image_target = is_image_target
        self._in_ch = in_channels
        self._out_ch = out_channels
        self._in_sz = in_size
        self._out_sz = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.deconv(x))

    def __repr__(self):
        tag = "→image" if self.is_image_target else "→activation"
        return (f"CNNNestedDecoder({self._in_ch}@{self._in_sz}² → "
                f"{self._out_ch}@{self._out_sz}², {tag})")


class MLPNestedDecoder(nn.Module):
    """Single Linear layer that inverts one MLP classifier layer.

    Args:
        in_dim: dimension of input activation a_k.
        out_dim: dimension of target (a_{k-1} dim, or input_dim for image).
        output_shape: if provided, reshape final output to (C, H, W) — only for D_0.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        output_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.output_shape = output_shape
        self.is_image_target = output_shape is not None
        self.activation = nn.Sigmoid() if self.is_image_target else nn.Identity()

        self._in_dim = in_dim
        self._out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        if self.output_shape is not None:
            out = out.view(-1, *self.output_shape)
        return out

    def __repr__(self):
        tag = f"→image{self.output_shape}" if self.is_image_target else "→activation"
        return f"MLPNestedDecoder({self._in_dim} → {self._out_dim}, {tag})"
