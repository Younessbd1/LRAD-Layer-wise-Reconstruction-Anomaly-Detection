"""Multi-scale convolutional decoders that mirror DeepCNNClassifier stages.

A decoder takes the activation map of one classifier stage and reconstructs
the original input image. Reconstruction error in image space is the
anomaly signal LRAD operates on, so:

  * **Output is pixel-aligned with the input** — no resampling on the
    inference path.
  * **Decoder depth matches the encoder depth** — a stage at /32 resolution
    needs more upsampling steps than a stage at /4. We compute that
    automatically from the spatial-size schedule.
  * **Sigmoid output range [0, 1]** — input images are normalised to [0, 1]
    before the loss is taken, so squared error is bounded and comparable
    across decoders.
  * **Upsample → Conv → BN → ReLU**, not strided ConvTranspose. Bilinear
    upsample + 3x3 conv avoids the checkerboard artefact that contaminates
    error maps when transposed convs are used; this materially improves
    pixel-AUROC on textures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class _UpBlock(nn.Module):
    """Bilinear upsample (×2) → 3x3 Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.up(x))))


class _RefineBlock(nn.Module):
    """Same-resolution 3x3 Conv → BN → ReLU, used between upsamples."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNNDecoder(nn.Module):
    """Reconstruct an input image from one classifier stage's activations.

    Args:
        in_ch:        channel dimension of the source activation.
        out_ch:       channel dimension of the input image (1 grayscale,
                      3 RGB).
        in_size:      spatial size of the source activation (square).
        out_size:     spatial size of the target image (square).
        base_ch:      channel width at the deepest decoder layer (kept
                      shallow to avoid dwarfing the encoder).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        in_size: int,
        out_size: int,
        base_ch: int = 128,
    ):
        super().__init__()
        if out_size % in_size != 0:
            raise ValueError(
                f"out_size ({out_size}) must be a power-of-2 multiple of in_size ({in_size})"
            )

        ratio = out_size // in_size
        n_up = 0
        r = ratio
        while r > 1:
            if r % 2 != 0:
                raise ValueError(f"upsampling ratio {ratio} must be a power of 2")
            r //= 2
            n_up += 1

        # Project source activations to the decoder's working width.
        layers: list[nn.Module] = [_RefineBlock(in_ch, base_ch)]
        ch = base_ch

        # Halve the channel count at each upsample step (common decoder pattern).
        for i in range(n_up):
            next_ch = max(base_ch // (2 ** (i + 1)), 32)
            layers.append(_UpBlock(ch, next_ch))
            layers.append(_RefineBlock(next_ch, next_ch))
            ch = next_ch

        # Final 1x1 to image space + sigmoid.
        layers.append(nn.Conv2d(ch, out_ch, 1, 1, 0))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._in_size = in_size
        self._out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Defensive: guarantee exact target spatial size if rounding ever drifts.
        if out.shape[-1] != self._out_size or out.shape[-2] != self._out_size:
            out = F.interpolate(out, size=(self._out_size, self._out_size),
                                mode="bilinear", align_corners=False)
        return out

    def __repr__(self):
        return (
            f"CNNDecoder(in={self._in_ch}@{self._in_size}, "
            f"out={self._out_ch}@{self._out_size})"
        )


def build_decoders(
    stage_channels: Sequence[int],
    spatial_sizes: Sequence[int],
    in_channels: int,
    base_ch: int = 128,
) -> list[CNNDecoder]:
    """Construct one decoder per classifier stage.

    spatial_sizes follows DeepCNNClassifier's convention:
        [input_size, stage_0_size, stage_1_size, ...]

    so spatial_sizes[i + 1] is the spatial size of the activation feeding
    decoder i, and spatial_sizes[0] is the reconstruction target.
    """
    if len(spatial_sizes) != len(stage_channels) + 1:
        raise ValueError(
            f"spatial_sizes ({len(spatial_sizes)}) must equal "
            f"len(stage_channels) + 1 ({len(stage_channels) + 1})"
        )
    target = spatial_sizes[0]
    return [
        CNNDecoder(
            in_ch=ch,
            out_ch=in_channels,
            in_size=spatial_sizes[i + 1],
            out_size=target,
            base_ch=base_ch,
        )
        for i, ch in enumerate(stage_channels)
    ]
