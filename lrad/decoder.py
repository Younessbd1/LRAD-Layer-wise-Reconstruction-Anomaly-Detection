"""Per-block decoders that upsample classifier activations back to the
input resolution.

These are visualization-only: at evaluation time they let us render
``recon_Lk`` and ``error_Lk`` for every conv block, but they are NOT
part of the OOD score (the score still comes from the classifier's
predictive entropy / max softmax probability).

Each ``BlockDecoder`` takes a single block's activation map ``(C, H, W)``
and upsamples it to ``(3, image_size, image_size)`` via a stack of
ConvTranspose2d → BN → ReLU stages, ending with a 1×1 conv + Sigmoid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import FacialCNN


class BlockDecoder(nn.Module):
    """Upsample activations from one conv block to the original image size."""

    def __init__(
        self,
        in_channels: int,
        in_size: int,
        out_size: int = 64,
        out_channels: int = 3,
        min_channels: int = 16,
    ):
        super().__init__()
        if in_size <= 0 or out_size <= 0:
            raise ValueError("in_size and out_size must be positive")
        if out_size % in_size != 0:
            raise ValueError(
                f"out_size {out_size} must be a multiple of in_size {in_size}"
            )
        n_up = int(round(math.log2(out_size // in_size)))

        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(n_up):
            new_ch = max(min_channels, ch // 2)
            layers += [
                nn.ConvTranspose2d(ch, new_ch, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(new_ch),
                nn.ReLU(inplace=True),
            ]
            ch = new_ch
        if n_up == 0:
            # Already at target resolution — apply a small refinement conv.
            ref = max(min_channels, in_channels)
            layers += [
                nn.Conv2d(in_channels, ref, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ref),
                nn.ReLU(inplace=True),
            ]
            ch = ref
        layers += [
            nn.Conv2d(ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_decoders(model: FacialCNN, image_size: int = 64) -> nn.ModuleList:
    """One BlockDecoder per conv block of ``model``."""
    sizes = model.block_spatial_sizes
    channels = model.block_out_channels
    return nn.ModuleList([
        BlockDecoder(in_channels=c, in_size=s, out_size=image_size)
        for c, s in zip(channels, sizes)
    ])
