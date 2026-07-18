"""CutPaste-style synthetic occlusions for self-supervised OOD training.

The post-hoc scoring stack tops out around AUROC 0.81 on this task — the
models were never taught what "something covers the face" looks like.
CutPaste (Li et al., CVPR 2021) closes that gap without any real OOD data:
during training, a rectangular patch cut from a *donor* face is pasted at
a random location of some images, and an extra binary head learns to tell
intact faces from altered ones. At test time ``P(altered | x)`` fires on
real occlusions the model never saw — eyeglasses included.

Two patch flavours, mixed by ``scar_prob``:

  * **patch** — a rectangle covering ``area_range`` of the image with
    aspect ratio in ``aspect_range`` (the classic CutPaste box);
  * **scar** — a long thin sliver (2–8 px wide, 10–45 % of the image
    long, either orientation), closer in shape to a glasses frame.

The donor is another image of the same batch (a roll), so pasted content
is always real face texture — harder negatives than noise or black boxes.
Labels: 1 = altered, 0 = intact. All randomness flows through the given
``torch.Generator`` so an epoch is reproducible given the seed.
"""

from __future__ import annotations

import torch

__all__ = ["cutpaste_batch"]


def _rand(lo: float, hi: float, gen: torch.Generator | None) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=gen).item()


def _patch_hw(
    H: int,
    W: int,
    area_range: tuple[float, float],
    aspect_range: tuple[float, float],
    gen: torch.Generator | None,
) -> tuple[int, int]:
    """Height/width of a classic CutPaste rectangle."""
    area = _rand(*area_range, gen) * H * W
    aspect = _rand(*aspect_range, gen)
    h = int(round((area * aspect) ** 0.5))
    w = int(round((area / aspect) ** 0.5))
    return max(2, min(h, H - 1)), max(2, min(w, W - 1))


def _scar_hw(H: int, W: int, gen: torch.Generator | None) -> tuple[int, int]:
    """Height/width of a thin scar sliver (either orientation)."""
    thick = int(round(_rand(2.0, 8.0, gen)))
    length = int(round(_rand(0.10, 0.45, gen) * max(H, W)))
    length = max(thick + 1, min(length, max(H, W) - 1))
    if torch.rand(1, generator=gen).item() < 0.5:
        return min(thick, H - 1), min(length, W - 1)     # horizontal
    return min(length, H - 1), min(thick, W - 1)         # vertical


def cutpaste_batch(
    images: torch.Tensor,
    *,
    prob: float = 0.5,
    area_range: tuple[float, float] = (0.02, 0.12),
    aspect_range: tuple[float, float] = (0.3, 3.3),
    scar_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste donor patches onto a random subset of the batch.

    Returns ``(augmented, labels)`` — a copy of ``images`` where each image
    was altered with probability ``prob`` (label 1) or left intact
    (label 0). ``scar_prob`` picks the scar flavour over the box flavour
    per altered image. The donor of image ``i`` is image ``i+1`` (mod B),
    so no image donates to itself.
    """
    if images.dim() != 4:
        raise ValueError(
            f"expected (B, C, H, W) images, got shape {tuple(images.shape)}"
        )
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    lo, hi = area_range
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"bad area_range {area_range!r}")
    lo, hi = aspect_range
    if not 0.0 < lo <= hi:
        raise ValueError(f"bad aspect_range {aspect_range!r}")
    if not 0.0 <= scar_prob <= 1.0:
        raise ValueError(f"scar_prob must be in [0, 1], got {scar_prob}")

    B, _, H, W = images.shape
    out = images.clone()
    donors = images.roll(shifts=1, dims=0)
    labels = (
        torch.rand(B, generator=generator) < prob
    ).long()

    for i in range(B):
        if labels[i] == 0:
            continue
        if torch.rand(1, generator=generator).item() < scar_prob:
            ph, pw = _scar_hw(H, W, generator)
        else:
            ph, pw = _patch_hw(H, W, area_range, aspect_range, generator)
        # Source window in the donor and destination window in the target
        # are drawn independently — content moves, not just copies over.
        sy = int(torch.randint(0, H - ph + 1, (1,), generator=generator))
        sx = int(torch.randint(0, W - pw + 1, (1,), generator=generator))
        dy = int(torch.randint(0, H - ph + 1, (1,), generator=generator))
        dx = int(torch.randint(0, W - pw + 1, (1,), generator=generator))
        out[i, :, dy:dy + ph, dx:dx + pw] = \
            donors[i, :, sy:sy + ph, sx:sx + pw]

    return out, labels.to(images.device)
