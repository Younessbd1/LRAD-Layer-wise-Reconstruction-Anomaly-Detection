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
      head_attrs:  Linear(256 -> 6)     # Arched_Eyebrows, ... (sigmoid + BCE)

Forward returns a dict with keys ``gender_logits`` (B, 2) and
``attr_logits`` (B, 6). The loss combines CrossEntropyLoss on gender_logits
with BCEWithLogitsLoss on attr_logits.

**Every head is optional.** A disabled head is ``None`` and its key is
simply missing from the forward dict; the defaults (both supervised heads
on, no cutpaste head) are the plain CelebA configuration, so existing
configs and checkpoints are unaffected. ``cutpaste_head: true`` adds the
CutPaste pretext head (:mod:`lrad.cutpaste`) used by the ablation's
cutpaste arms — intact (0) vs altered (1), or the 3-way intact/box/scar
split when ``cutpaste_classes: 3``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _conv_block(
    in_ch: int,
    out_ch: int,
    pool: bool = True,
    kernel_size: int = 3,
) -> nn.Sequential:
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                  padding=kernel_size // 2, bias=False),
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
        kernel_size: int = 3,
        cutpaste_head: bool = False,
        gender_head: bool = True,
        attrs_head: bool = True,
        cutpaste_classes: int = 2,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("need at least 2 conv blocks")
        self.channels = list(channels)
        self.block_out_channels = list(channels)
        self.input_size = input_size
        self.kernel_size = int(kernel_size)
        self.cutpaste_head = bool(cutpaste_head)
        self.gender_head = bool(gender_head)
        self.attrs_head = bool(attrs_head)
        self.cutpaste_classes = int(cutpaste_classes)

        # A head that is switched off reports zero outputs, so callers that
        # size buffers off ``n_attrs`` (e.g. the per-attribute accuracy
        # accumulator in lrad.train) allocate nothing rather than tracking a
        # target that does not exist.
        self.n_attrs = int(n_attrs) if self.attrs_head else 0
        self.n_gender = int(n_gender) if self.gender_head else 0

        if not (self.gender_head or self.attrs_head or self.cutpaste_head):
            raise ValueError(
                "the trunk needs at least one head to train against — "
                "enable one of gender_head / attrs_head / cutpaste_head"
            )
        if self.cutpaste_head and self.cutpaste_classes < 2:
            raise ValueError(
                f"cutpaste_classes must be >= 2, got {self.cutpaste_classes}"
            )

        blocks: list[nn.Module] = []
        prev = in_channels
        # Pool after every block except the last to preserve a small
        # spatial map for the GAP layer. ``kernel_size`` only changes the
        # receptive field of each conv (padding keeps H, W), so the block
        # spatial layout — and therefore the decoder geometry — is the same
        # for every ensemble member regardless of the variant.
        for i, ch in enumerate(channels):
            pool = i < len(channels) - 1
            blocks.append(_conv_block(prev, ch, pool=pool,
                                      kernel_size=self.kernel_size))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Both supervised heads default to on, which is the CelebA
        # configuration and keeps every existing checkpoint loading
        # unchanged.
        self.head_gender = (
            nn.Linear(prev, self.n_gender) if self.gender_head else None
        )
        self.head_attrs = (
            nn.Linear(prev, self.n_attrs) if self.attrs_head else None
        )
        # Optional CutPaste pretext head: intact (0) vs altered (1), or the
        # 3-way intact / box / scar split when cutpaste_classes == 3. Off by
        # default so checkpoints from runs without it keep loading cleanly.
        self.head_cutpaste = (
            nn.Linear(prev, self.cutpaste_classes)
            if self.cutpaste_head else None
        )

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
        # Keys for disabled heads are ABSENT rather than present-and-empty,
        # so consumers must opt in with ``"gender_logits" in out`` instead of
        # silently averaging a zero-width tensor into a score.
        out: dict[str, torch.Tensor] = {}
        if self.head_gender is not None:
            out["gender_logits"] = self.head_gender(h)
        if self.head_attrs is not None:
            out["attr_logits"] = self.head_attrs(h)
        if self.head_cutpaste is not None:
            out["cutpaste_logits"] = self.head_cutpaste(h)
        return out

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
        # A config may centre-crop after resizing, so the tensor the trunk
        # sees is crop_size, not image_size; fall back through both.
        input_size=mcfg.get(
            "input_size",
            dcfg.get("crop_size") or dcfg.get("image_size", 64),
        ),
        kernel_size=mcfg.get("kernel_size", 3),
        cutpaste_head=mcfg.get("cutpaste_head", False),
        gender_head=mcfg.get("gender_head", True),
        attrs_head=mcfg.get("attrs_head", True),
        cutpaste_classes=mcfg.get("cutpaste_classes", 2),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
