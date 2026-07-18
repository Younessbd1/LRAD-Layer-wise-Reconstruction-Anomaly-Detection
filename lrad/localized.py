"""Localized, in-dist-normalized OOD scoring: per-pixel z-score + patch-max.

The global ``p95`` reduction in ``lrad.anomaly_score`` dilutes small,
localized anomalies: a pair of eyeglasses covers a few hundred pixels of a
64x64 face, while hair and background — structurally hard to reconstruct —
supply thousands of high-error pixels on *every* clean face. Two standard
fixes, composed, without ever hard-coding where the anomaly is:

1. **Per-pixel z-score.** Fit the per-pixel mean/std of each decomposition
   map (Risk / Bias / Variance, per block) on in-distribution reference
   images. CelebA faces are aligned, so pixel position is semantically
   stable: the z-map ``(map − μ) / σ`` cancels zones that are *always*
   badly reconstructed (large μ) and amplifies deviations in zones that are
   normally easy (small σ, e.g. the aligned face itself).
2. **Multi-scale patch-max.** Average-pool the z-map at a few window sizes
   and keep the maximum patch response. A localized anomaly *anywhere* in
   the frame lights up exactly one window; single-pixel noise is averaged
   away inside the window. This replaces the global p95, which only fires
   when ~5 % of *all* pixels move.

Reference stats must come from in-distribution data only (val split, or a
slice of train when ``val_ratio=0``) — never from ``test_in`` (that would
leak the evaluation split) and never from OOD data.

Everything here runs under ``@torch.no_grad()``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anomaly_score import aggregate_anomaly_score
from .ensemble import TERMS, block_reconstructions, decomposition_maps
from .evaluate import _auroc_entry
from .model import FacialCNN

logger = logging.getLogger("celeba_ood")

# Window sizes (pixels at image resolution) for the patch-max reduction.
# 4/8/16 on a 64x64 face bracket "a lens", "a frame half", "the whole
# glasses" without assuming which anomaly will show up.
DEFAULT_PATCH_SIZES: tuple[int, ...] = (4, 8, 16)

# Floor for the per-pixel reference std. Bias pixels live in [0, 3]; a
# pixel whose reference std collapses toward 0 (e.g. saturated background)
# would otherwise turn float noise into an unbounded z-score.
STD_FLOOR = 1e-3


@torch.no_grad()
def fit_reference_stats(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    max_batches: int | None = None,
    std_floor: float = STD_FLOOR,
) -> dict:
    """Per-pixel mean/std of each term map over an in-distribution loader.

    Streams the decomposition maps batch by batch and accumulates float64
    sum / sum-of-squares per pixel, so memory stays ``O(H·W)`` per block.
    ``max_batches`` caps the pass (a few thousand images give stable
    per-pixel stats). Returns::

        {
          'n_images': int,
          'std_floor': float,
          'stats': {term: {k: {'mean': (H, W), 'std': (H, W)}}},
        }

    with float32 CPU tensors, ``std`` already clamped to ``std_floor``.
    """
    if std_floor <= 0:
        raise ValueError(f"std_floor must be > 0, got {std_floor}")
    s: dict[str, dict[int, torch.Tensor]] = {t: {} for t in TERMS}
    ss: dict[str, dict[int, torch.Tensor]] = {t: {} for t in TERMS}
    n_images = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        maps = decomposition_maps(img, recons_per_model)
        n_images += img.shape[0]
        for t in TERMS:
            for k in maps:
                mp = maps[k][t].double()
                add = mp.sum(dim=0).cpu()
                add2 = (mp ** 2).sum(dim=0).cpu()
                if k not in s[t]:
                    s[t][k] = add
                    ss[t][k] = add2
                else:
                    s[t][k] += add
                    ss[t][k] += add2

    if n_images == 0:
        raise ValueError("reference loader yielded no images")

    stats: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for t in TERMS:
        stats[t] = {}
        for k, tot in s[t].items():
            mean = tot / n_images
            var = (ss[t][k] / n_images - mean ** 2).clamp_min(0.0)
            stats[t][k] = {
                "mean": mean.float(),
                "std": var.sqrt().float().clamp_min(std_floor),
            }
    logger.info(
        "Reference stats fitted on %d in-dist images (%d blocks, "
        "std_floor=%g)", n_images, len(stats[TERMS[0]]), std_floor,
    )
    return {"n_images": n_images, "std_floor": std_floor, "stats": stats}


def patch_max_reduce(
    a: torch.Tensor,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
) -> torch.Tensor:
    """Reduce a ``(B, H, W)`` map to ``(B,)``: max mean-response window.

    For each window size ``s`` the map is average-pooled (stride ``s // 2``,
    overlapping, so an anomaly straddling a window boundary is still seen
    whole) and the spatial max is taken; the final score is the max across
    window sizes. Fires on a localized hot region *anywhere* in the frame
    while averaging single-pixel noise away inside the window.
    """
    if a.dim() != 3:
        raise ValueError(f"expected a (B, H, W) map, got shape {tuple(a.shape)}")
    sizes = list(patch_sizes)
    if not sizes:
        raise ValueError("patch_sizes is empty — give at least one size")
    H, W = a.shape[-2:]
    for sz in sizes:
        if not 1 <= sz <= min(H, W):
            raise ValueError(
                f"patch size {sz} out of range [1, {min(H, W)}] for a "
                f"{H}x{W} map"
            )
    x = a.unsqueeze(1).float()  # (B, 1, H, W)
    per_scale = [
        F.avg_pool2d(x, kernel_size=sz, stride=max(1, sz // 2))
        .amax(dim=(1, 2, 3))
        for sz in sizes
    ]
    return torch.stack(per_scale, dim=0).max(dim=0).values  # (B,)


@torch.no_grad()
def collect_localized_scores(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    ref: dict,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
    max_batches: int | None = None,
) -> dict:
    """Per-image localized scores over a loader, plus the p95 baseline.

    Every term map is z-scored against ``ref`` (from
    :func:`fit_reference_stats`) and reduced with :func:`patch_max_reduce`;
    z-scores are block-comparable, so the aggregated score is the uniform
    block mean. ``baseline_p95`` is the existing raw global-p95 aggregate
    computed on the *same* maps, so old and new scores compare on exactly
    the same images. Returns::

        {
          'per_block':    {term: {k: (N,)}},
          'aggregated':   {term: (N,)},
          'baseline_p95': {term: (N,)},
        }
    """
    stats = ref["stats"]
    per_block: dict[str, dict[int, list]] = {t: {} for t in TERMS}
    aggregated: dict[str, list] = {t: [] for t in TERMS}
    baseline: dict[str, list] = {t: [] for t in TERMS}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        maps = decomposition_maps(img, recons_per_model)
        for t in TERMS:
            block_scores = []
            for k in sorted(maps.keys()):
                st = stats[t][k]
                z = (maps[k][t] - st["mean"].to(img.device)) \
                    / st["std"].to(img.device)
                score = patch_max_reduce(z, patch_sizes)
                per_block[t].setdefault(k, []).append(score.cpu().numpy())
                block_scores.append(score)
            aggregated[t].append(
                torch.stack(block_scores, dim=0).mean(dim=0).cpu().numpy()
            )
            baseline[t].append(
                aggregate_anomaly_score(
                    {k: maps[k][t] for k in maps}, "p95",
                ).cpu().numpy()
            )

    def _cat(chunks: list) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.zeros(0)

    return {
        "per_block": {
            t: {k: _cat(v) for k, v in per_block[t].items()} for t in TERMS
        },
        "aggregated": {t: _cat(aggregated[t]) for t in TERMS},
        "baseline_p95": {t: _cat(baseline[t]) for t in TERMS},
    }


@torch.no_grad()
def evaluate_localized_ood(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    ref_loader,
    test_in_loader,
    test_ood_loader,
    device: torch.device,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
    std_floor: float = STD_FLOOR,
    max_ref_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict:
    """Full localized evaluation: fit reference stats, score both test
    splits, and report AUROC for the z-score+patch-max scores next to the
    global-p95 baseline computed on the same images.

    ``ref_loader`` must be in-distribution data disjoint from both test
    splits. The ``auroc`` dict uses ``zscore_{term}_per_block_{k}``,
    ``zscore_{term}_aggregated`` and ``baseline_p95_{term}_aggregated``
    keys; ``anomaly_auroc`` surfaces the headline Bias term.
    """
    ref = fit_reference_stats(
        models, decoders_list, ref_loader, device,
        max_batches=max_ref_batches, std_floor=std_floor,
    )
    scores_in = collect_localized_scores(
        models, decoders_list, test_in_loader, device, ref,
        patch_sizes, max_eval_batches,
    )
    scores_ood = collect_localized_scores(
        models, decoders_list, test_ood_loader, device, ref,
        patch_sizes, max_eval_batches,
    )

    ks = sorted(scores_in["per_block"][TERMS[0]].keys())
    auroc: dict = {}
    per_block_auroc: dict[str, list[float]] = {t: [] for t in TERMS}
    for t in TERMS:
        for k in ks:
            entry = _auroc_entry(
                scores_in["per_block"][t][k],
                scores_ood["per_block"][t][k],
            )
            auroc[f"zscore_{t}_per_block_{k}"] = entry
            per_block_auroc[t].append(entry.get("auroc", float("nan")))
        auroc[f"zscore_{t}_aggregated"] = _auroc_entry(
            scores_in["aggregated"][t], scores_ood["aggregated"][t],
        )
        auroc[f"baseline_p95_{t}_aggregated"] = _auroc_entry(
            scores_in["baseline_p95"][t], scores_ood["baseline_p95"][t],
        )

    return {
        "patch_sizes": list(patch_sizes),
        "std_floor": std_floor,
        "n_ref_images": ref["n_images"],
        "blocks": ks,
        "n_models": len(models),
        "scores_in": scores_in,
        "scores_ood": scores_ood,
        "auroc": auroc,
        "per_block_auroc": per_block_auroc,
        "anomaly_auroc": {
            "aggregated": auroc["zscore_bias_aggregated"].get("auroc"),
            "baseline": auroc["baseline_p95_bias_aggregated"].get("auroc"),
            "per_block": list(per_block_auroc["bias"]),
        },
    }
