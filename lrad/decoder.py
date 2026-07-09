"""Per-block decoders that upsample classifier activations back to the
input resolution.

Each model's per-block reconstructions ``f̂_k`` are what the OOD score is
built on: across a deep ensemble they feed the bias/variance decomposition
(``lrad.ensemble``), and the anomaly score is the bias term ``(x − f̄_k)²``.
For a single model they also drive the per-block ``recon_Lk`` / ``error_Lk``
plots and the fused reconstruction-error score.

Each ``BlockDecoder`` takes a single block's activation map ``(C, H, W)``
and grows it to ``(3, image_size, image_size)`` with a stack of ×2
learnable upsampling stages — ConvTranspose2d(4×4, stride 2, padding 1)
→ BN → ReLU — ending in a 1×1 conv + Sigmoid. The 4×4/stride-2/pad-1
kernel doubles H and W exactly, with no output-size ambiguity.

The stack halves the channels at every stage, down to ``min_channels``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import FacialCNN


def _up_stage(in_ch: int, out_ch: int) -> list[nn.Module]:
    """One ×2 learnable upsampling stage: ConvTranspose2d → BN → ReLU."""
    return [
        nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


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
            layers += _up_stage(ch, new_ch)
            ch = new_ch
        if n_up == 0:
            # Activation is already at image resolution (block 0 on a small
            # input): nothing to upsample, just one refinement conv so the
            # decoder still has a learnable stage before the output head.
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


def build_decoders(
    model: FacialCNN,
    image_size: int = 64,
) -> nn.ModuleList:
    """One BlockDecoder per conv block of ``model``."""
    sizes = model.block_spatial_sizes
    channels = model.block_out_channels
    return nn.ModuleList([
        BlockDecoder(in_channels=c, in_size=s, out_size=image_size)
        for c, s in zip(channels, sizes)
    ])
