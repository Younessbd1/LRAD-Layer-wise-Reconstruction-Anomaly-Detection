"""From-scratch CNN with two task heads for CelebA OOD detection.

Trunk depth follows ``channels``; the default 5-block layout is shown
below (no pretrained weights)::

    Input (B, 3, 64, 64)
      Block1: Conv3x3(3 -> 32)  + BN + ReLU + MaxPool2x2  -> (32, 32, 32)
      Block2: Conv3x3(32 -> 64) + BN + ReLU + MaxPool2x2  -> (64, 16, 16)
      Block3: Conv3x3(64 -> 128)+ BN + ReLU + MaxPool2x2  -> (128, 8, 8)
      Block4: Conv3x3(128->256) + BN + ReLU + MaxPool2x2  -> (256, 4, 4)
      Block5: Conv3x3(256->256) + BN + ReLU               -> (256, 4, 4)
      AdaptiveAvgPool -> (256,)
      head_gender: Linear(256 -> 2)     # Male vs Female (softmax + CE)
      head_attrs:  Linear(256 -> 6)     # Young, Smiling, ... (sigmoid + BCE)

Forward returns a dict with keys ``gender_logits`` (B, 2) and
``attr_logits`` (B, 6). The loss combines CrossEntropyLoss on gender_logits
with BCEWithLogitsLoss on attr_logits.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _conv_block(
    in_ch: int,
    out_ch: int,
    pool: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class FacialCNN(nn.Module):
    """Multi-head CNN: shared conv trunk + (gender, attrs) heads.

    Each conv block exposes its post-activation tensor. ``forward(x)``
    returns logits as before; ``forward_with_activations(x)`` additionally
    returns the list of per-block activations, used downstream by
    per-block reconstruction decoders.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 256),
        n_attrs: int = 6,
        n_gender: int = 2,
        input_size: int = 64,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("need at least 2 conv blocks")
        self.channels = list(channels)
        self.block_out_channels = list(channels)
        self.n_attrs = n_attrs
        self.n_gender = n_gender
        self.input_size = input_size

        blocks: list[nn.Module] = []
        prev = in_channels
        # Pool after every block except the last to preserve a small
        # spatial map for the GAP layer.
        for i, ch in enumerate(channels):
            pool = i < len(channels) - 1
            blocks.append(_conv_block(prev, ch, pool=pool))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_gender = nn.Linear(prev, n_gender)
        self.head_attrs = nn.Linear(prev, n_attrs)

        self._init_weights()

        # Probe spatial sizes once so per-block decoders can be built
        # without another forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            _, acts = self.forward_features(dummy)
            self.block_spatial_sizes = [int(a.shape[-1]) for a in acts]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the conv trunk and return (final_feature_map, [acts_per_block])."""
        acts: list[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            acts.append(x)
        return x, acts

    def _heads(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.pool(feat).flatten(1)
        return {
            "gender_logits": self.head_gender(h),
            "attr_logits": self.head_attrs(h),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat, _ = self.forward_features(x)
        return self._heads(feat)

    def forward_with_activations(
        self, x: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        feat, acts = self.forward_features(x)
        return self._heads(feat), acts


def build_model(cfg: dict) -> FacialCNN:
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("dataset", {})
    return FacialCNN(
        in_channels=mcfg.get("in_channels", 3),
        channels=mcfg.get("channels", (32, 64, 128, 256, 256)),
        n_attrs=mcfg.get("n_attrs", 6),
        n_gender=mcfg.get("n_gender", 2),
        input_size=mcfg.get("input_size", dcfg.get("image_size", 64)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
