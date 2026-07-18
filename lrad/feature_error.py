"""Localized feature-space reconstruction error — the ``locfre`` signal.

Pixel-space bias tops out around AUROC 0.64–0.68 on the eyeglasses task
because the decoders are blurry: clean faces already carry a large,
spatially structured error floor that a small occlusion barely moves.
Feature space sidesteps the blur. The signal:

1. Reconstruct the input at the **deepest** block and average over the
   ensemble (the consensus reconstruction ``f̄`` — glasses and other
   out-of-distribution structure are *absent* from it, because no decoder
   trained on clean faces knows how to draw them).
2. Re-encode ``f̄`` with each member's trunk and compare, **per spatial
   position**, the channel-normalized activations of the input against
   those of ``f̄`` (squared L2 of the difference of unit vectors), averaged
   over members. OOD structure survives in the input's features but not in
   ``f̄``'s, so the error map lights up exactly where the anomaly sits.
3. Score like ``lrad.localized``: z-score each map against per-position
   in-distribution reference stats, then reduce with the multi-scale
   patch-max.

On the 20260707 ensemble run (subsampled test splits) the block-3 locfre
score reaches AUROC ≈ 0.78 alone, against 0.64 for the pixel-space p95
bias baseline. Blocks 1 (16x16, fine texture) and 3 (4x4, semantic) are
the strongest and complementary depths, hence the default.

Everything here runs under ``@torch.no_grad()``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .localized import STD_FLOOR, patch_max_reduce
from .model import FacialCNN

logger = logging.getLogger("celeba_ood")

# Trunk blocks whose activations are compared (input vs reconstruction).
DEFAULT_FEATURE_BLOCKS: tuple[int, ...] = (1, 3)


def _feature_patch_sizes(h: int, w: int) -> tuple[int, ...]:
    """Patch-max window sizes adapted to a feature map's resolution."""
    m = min(h, w)
    return tuple(s for s in (2, 4, 8) if s <= m) or (1,)


@torch.no_grad()
def feature_error_maps(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
) -> dict[int, torch.Tensor]:
    """Member-averaged per-position feature error maps for one batch.

    Returns ``{j: (B, h_j, w_j)}`` for each ``j`` in ``blocks``: the
    squared L2 distance between the channel-normalized block-``j``
    activations of the input and of the re-encoded ensemble-mean deepest
    reconstruction, averaged over the ensemble. Values live in ``[0, 4]``
    (distance between two unit vectors, squared).
    """
    if not models:
        raise ValueError("need at least one model in the ensemble")
    n_blocks = len(decoders_list[0])
    kd = n_blocks - 1
    blocks = list(blocks)
    for j in blocks:
        if not 0 <= j < n_blocks:
            raise ValueError(f"block {j} out of range [0, {n_blocks - 1}]")

    images = images.to(device, non_blocking=True)
    M = len(models)

    # Pass 1 — encode the input once per member; keep the activations the
    # comparison needs and build the ensemble-mean deepest reconstruction.
    acts_x: list[dict[int, torch.Tensor]] = []
    recon_sum: torch.Tensor | None = None
    for m in range(M):
        models[m].eval()
        decoders_list[m].eval()
        _, acts = models[m].forward_features(images)
        acts_x.append({j: F.normalize(acts[j], dim=1) for j in blocks})
        recon = decoders_list[m][kd](acts[kd])
        recon_sum = recon if recon_sum is None else recon_sum + recon
    mean_recon = recon_sum / M

    # Pass 2 — re-encode the consensus reconstruction with each member and
    # accumulate the per-position normalized-feature error.
    err: dict[int, torch.Tensor] = {}
    for m in range(M):
        _, acts_r = models[m].forward_features(mean_recon)
        for j in blocks:
            d = acts_x[m][j] - F.normalize(acts_r[j], dim=1)
            e = (d ** 2).sum(dim=1)  # (B, h, w)
            err[j] = e if j not in err else err[j] + e
    return {j: err[j] / M for j in blocks}


@torch.no_grad()
def fit_feature_error_stats(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
    max_batches: int | None = None,
    std_floor: float = STD_FLOOR,
) -> dict:
    """Per-position mean/std of the feature error maps on in-dist data.

    Same contract as ``lrad.localized.fit_reference_stats`` (float64
    streaming accumulation, float32 outputs, std clamped to
    ``std_floor``), but per feature block::

        {'n_images': int, 'std_floor': float,
         'stats': {j: {'mean': (h_j, w_j), 'std': (h_j, w_j)}}}
    """
    if std_floor <= 0:
        raise ValueError(f"std_floor must be > 0, got {std_floor}")
    s: dict[int, torch.Tensor] = {}
    ss: dict[int, torch.Tensor] = {}
    n_images = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        maps = feature_error_maps(
            models, decoders_list, batch[0], device, blocks,
        )
        n_images += batch[0].shape[0]
        for j, mp in maps.items():
            mp = mp.double()
            add, add2 = mp.sum(dim=0).cpu(), (mp ** 2).sum(dim=0).cpu()
            if j not in s:
                s[j], ss[j] = add, add2
            else:
                s[j] += add
                ss[j] += add2
    if n_images == 0:
        raise ValueError("reference loader yielded no images")

    stats = {}
    for j in s:
        mean = s[j] / n_images
        var = (ss[j] / n_images - mean ** 2).clamp_min(0.0)
        stats[j] = {
            "mean": mean.float(),
            "std": var.sqrt().float().clamp_min(std_floor),
        }
    logger.info(
        "locfre reference stats fitted on %d in-dist images (blocks %s)",
        n_images, sorted(stats),
    )
    return {"n_images": n_images, "std_floor": std_floor, "stats": stats}


@torch.no_grad()
def locfre_scores_from_maps(
    maps: dict[int, torch.Tensor],
    ref: dict,
) -> dict[int, torch.Tensor]:
    """Per-image locfre score for one batch's maps: z-score against
    ``ref`` (from :func:`fit_feature_error_stats`) then patch-max with
    windows adapted to each block's resolution. Returns ``{j: (B,)}``."""
    out = {}
    for j, mp in maps.items():
        st = ref["stats"][j]
        z = (mp - st["mean"].to(mp.device)) / st["std"].to(mp.device)
        out[j] = patch_max_reduce(z, _feature_patch_sizes(*z.shape[-2:]))
    return out


@torch.no_grad()
def collect_locfre_scores(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    ref: dict,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
    max_batches: int | None = None,
) -> dict[int, np.ndarray]:
    """Per-image locfre scores over a whole loader: ``{j: (N,)}``."""
    chunks: dict[int, list[np.ndarray]] = {}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        maps = feature_error_maps(
            models, decoders_list, batch[0], device, blocks,
        )
        for j, sc in locfre_scores_from_maps(maps, ref).items():
            chunks.setdefault(j, []).append(sc.cpu().numpy())
    return {
        j: (np.concatenate(v) if v else np.zeros(0))
        for j, v in chunks.items()
    }
