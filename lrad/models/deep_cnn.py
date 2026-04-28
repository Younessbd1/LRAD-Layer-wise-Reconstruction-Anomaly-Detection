"""DeepCNNClassifier — ResNet-style backbone tuned for anomaly detection.

Inductive biases relevant to LRAD anomaly detection:

  * **Multi-stage feature pyramid.** Four explicit stages (each ending with
    a stride-2 downsampling) give the LRAD pipeline four well-separated
    spatial scales. Small, localized defects are caught by shallow stages;
    large semantic deviations are caught by deeper ones.

  * **Locality-preserving stem.** A 7x7 stride-2 conv + 3x3 stride-2 max-pool
    maintains a strong texture prior at high resolution, then aggressively
    downsamples. Empirically much better than a stack of 3x3 stride-2 convs
    for industrial-inspection inputs that have fine textures (Real-IAD).

  * **Residual basic blocks.** Identity shortcuts let early features survive
    deep in the network — important because LRAD's reconstruction targets
    the input itself, so we want activations that still carry input
    information at every depth.

  * **No global pooling inside the feature path.** The classifier head
    pools at the very end; every intermediate stage keeps spatial
    structure, which is what the decoders need.

  * **Padded 3x3 convolutions everywhere.** Spatial sizes are predictable
    and per-stage feature maps line up cleanly with the decoder grid.

The pretext head is intentionally small (1 GAP + 1 Linear). It's only there
to drive feature learning during phase 1; LRAD never uses its logits at
inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + identity shortcut."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    layers: list[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Stem variants
# ---------------------------------------------------------------------------

def _build_stem(in_channels: int, stem_out: int, kind: str) -> tuple[nn.Module, int]:
    """Return (stem_module, downsampling_factor)."""
    if kind == "large":
        # 7x7 stride2 conv + 3x3 stride2 maxpool  → /4 total
        stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, 7, 2, 3, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        return stem, 4
    if kind == "small":
        # 3x3 stride2 conv only  → /2 total
        stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )
        return stem, 2
    raise ValueError(f"unknown stem kind {kind!r}")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DeepCNNClassifier(nn.Module):
    """Deep convolutional classifier with per-stage activation extraction.

    Architecture::

        Input (B, C, H, W)
          └─ Stem            → (B, c0, H/s, W/s)        s = 4 (large) or 2 (small)
          └─ Stage 1 (k1×B)  → (B, c1, .,. )            stride 1 (no further down)
          └─ Stage 2 (k2×B)  → (B, c2, ./2, ./2)        stride 2 at first block
          └─ Stage 3 (k3×B)  → (B, c3, ./4, ./4)
          └─ Stage 4 (k4×B)  → (B, c4, ./8, ./8)
          └─ GAP + Linear    → logits

    Args:
        in_channels:        input channels (3 for RGB).
        num_classes:        size of the pretext head (4 for rotation, etc).
        input_size:         spatial side of the (square) input — used to
                            precompute spatial sizes per stage.
        stage_channels:     channel dim of each stage's output. Default
                            [64, 128, 256, 512].
        blocks_per_stage:   number of BasicBlocks per stage. Default
                            [2, 2, 2, 2] (ResNet-18 layout).
        stem:               "large" (7x7 + maxpool, ResNet default) or
                            "small" (3x3 stride2 only — for low-res inputs).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        input_size: int = 224,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        stem: str = "large",
    ):
        super().__init__()
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must align")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.stage_channels = list(stage_channels)
        self.blocks_per_stage = list(blocks_per_stage)
        self.stem_kind = stem

        # Stem
        self.stem, stem_factor = _build_stem(in_channels, stage_channels[0], stem)

        # Stages: stage 0 keeps resolution, stages 1..N-1 downsample by 2.
        self.stages = nn.ModuleList()
        prev_ch = stage_channels[0]
        for i, (ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            stride = 1 if i == 0 else 2
            self.stages.append(_make_stage(prev_ch, ch, n_blocks, stride))
            prev_ch = ch

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[-1], num_classes)

        # Precompute spatial sizes [input, stage0, stage1, ...]
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            _, acts = self._forward_with_activations(dummy)
            self.spatial_sizes = [input_size] + [a.shape[-1] for a in acts]

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _forward_with_activations(self, x: torch.Tensor):
        x = self.stem(x)
        activations: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            activations.append(x)
        pooled = self.pool(x).flatten(1)
        logits = self.fc(pooled)
        return logits, activations

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self._forward_with_activations(x)

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return (
            f"DeepCNNClassifier(stages={self.stage_channels}, "
            f"blocks={self.blocks_per_stage}, "
            f"input={self.input_size}, stem={self.stem_kind!r}, "
            f"spatial_sizes={self.spatial_sizes})"
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    "resnet10":      dict(stage_channels=(64, 128, 256, 512), blocks_per_stage=(1, 1, 1, 1)),
    "resnet18":      dict(stage_channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2)),
    "resnet34":      dict(stage_channels=(64, 128, 256, 512), blocks_per_stage=(3, 4, 6, 3)),
    "resnet18_slim": dict(stage_channels=(32, 64, 128, 256),  blocks_per_stage=(2, 2, 2, 2)),
}


def build_deep_cnn(
    preset: str | None = "resnet18",
    in_channels: int = 3,
    num_classes: int = 4,
    input_size: int = 224,
    stem: str = "large",
    stage_channels: Sequence[int] | None = None,
    blocks_per_stage: Sequence[int] | None = None,
) -> DeepCNNClassifier:
    """Factory: build a DeepCNNClassifier from a preset name or explicit dims."""
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose one of {list(_PRESETS)}")
        params = dict(_PRESETS[preset])
    else:
        if stage_channels is None or blocks_per_stage is None:
            raise ValueError("either preset or (stage_channels & blocks_per_stage) is required")
        params = dict(stage_channels=stage_channels, blocks_per_stage=blocks_per_stage)

    return DeepCNNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        input_size=input_size,
        stem=stem,
        **params,
    )
