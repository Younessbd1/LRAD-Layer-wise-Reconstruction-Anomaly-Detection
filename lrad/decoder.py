"""Per-block decoders that upsample classifier activations back to the
input resolution.

Each model's per-block reconstructions ``f̂_k`` are what the OOD score is
built on: across a deep ensemble they feed the bias/variance decomposition
(``lrad.ensemble``), and the anomaly score is the bias term ``(x − f̄_k)²``.
For a single model they also drive the per-block ``recon_Lk`` / ``error_Lk``
plots and the fused reconstruction-error score.

Each ``BlockDecoder`` takes a single block's activation map ``(C, H, W)``
and grows it to ``(3, image_size, image_size)`` with a stack of ×2
upsampling stages, ending in a 1×1 conv + Sigmoid. Two upsampling stages
are available, selected by ``upsample``:

  * ``"bilinear"`` (default) — Upsample(×2, bilinear) → Conv3×3 → BN → ReLU.
    Resize-then-convolve, so it cannot produce the checkerboard artefacts
    that a strided transposed conv leaves behind. Smoother reconstructions
    mean a cleaner reconstruction-error signal, which is why it is the
    default.
  * ``"transpose"`` — ConvTranspose2d(4×4, stride 2) → BN → ReLU. The
    learnable upsampling we started from; kept as an option to reproduce
    the original decoder and to compare the two error signatures.

Both stacks share the same channel schedule (halve the channels at every
stage, down to ``min_channels``) so the two variants stay parameter-
comparable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import FacialCNN

UPSAMPLE_MODES: tuple[str, ...] = ("bilinear", "transpose")


def _up_stage(in_ch: int, out_ch: int, mode: str) -> list[nn.Module]:
    """One ×2 upsampling stage, either resize-then-conv or transposed conv.

    ``transpose`` uses a 4×4 kernel with stride 2 and padding 1, the usual
    choice that doubles H and W exactly without leaving an output-size
    ambiguity. ``bilinear`` upsamples first, then refines with a plain 3×3
    conv, so no learnable kernel ever strides over the output grid.
    """
    if mode == "transpose":
        up: nn.Module = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False,
        )
        return [up, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return [
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
        upsample: str = "bilinear",
    ):
        super().__init__()
        if in_size <= 0 or out_size <= 0:
            raise ValueError("in_size and out_size must be positive")
        if out_size % in_size != 0:
            raise ValueError(
                f"out_size {out_size} must be a multiple of in_size {in_size}"
            )
        if upsample not in UPSAMPLE_MODES:
            raise ValueError(
                f"upsample must be one of {UPSAMPLE_MODES}, got {upsample!r}"
            )
        self.upsample = upsample
        n_up = int(round(math.log2(out_size // in_size)))

        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(n_up):
            new_ch = max(min_channels, ch // 2)
            layers += _up_stage(ch, new_ch, upsample)
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
    upsample: str = "bilinear",
) -> nn.ModuleList:
    """One BlockDecoder per conv block of ``model``.

    ``upsample`` picks the upsampling stage for every decoder (see
    :class:`BlockDecoder`); all blocks share the same mode so an ensemble
    stays architecturally uniform.
    """
    sizes = model.block_spatial_sizes
    channels = model.block_out_channels
    return nn.ModuleList([
        BlockDecoder(in_channels=c, in_size=s, out_size=image_size,
                     upsample=upsample)
        for c, s in zip(channels, sizes)
    ])
