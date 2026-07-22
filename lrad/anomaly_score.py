"""Per-pixel reconstruction error and pixel→scalar reduction utilities.

The OOD anomaly in this project is the **bias** term of the deep-ensemble
decomposition (see ``lrad.ensemble``)::

    anomaly(x)[p]  =  bias_k(x)[p]  =  risk_k(x)[p] − variance_k(x)[p]
                   =  ( x[p] − f̄_k(x)[p] )^2

i.e. the squared error of the *ensemble-mean* reconstruction ``f̄``. There
is **no sigma and no division**: subtracting the variance (the model
disagreement / epistemic term) from the risk leaves exactly the consensus
model's irreducible error, which is what genuinely fails to reconstruct on
OOD inputs (e.g. eyeglasses occluding the face).

This module no longer performs any sigma-whitening. What remains is the
pixel → scalar reduction machinery shared across the codebase:
``_reduce_over_pixels`` collapses a per-pixel map ``(B, H, W)`` to one
scalar per image, and ``aggregate_anomaly_score`` combines the per-block
scalars into a single anomaly score. Both back the ensemble decomposition
scorer (``lrad.ensemble``) and the localized variants.
"""

from __future__ import annotations

from typing import Sequence

import torch

_AGGS = ("mean", "max", "p95")


def _reduce_over_pixels(a: torch.Tensor, agg: str) -> torch.Tensor:
    """Reduce a ``(B, H, W)`` map to a scalar ``(B,)`` per image.

    ``mean`` is sensitive to the whole image, ``max`` to the single most
    surprising pixel, and ``p95`` (95th percentile) is a robust
    compromise that ignores lone hot pixels but still fires on a
    localized anomaly such as a pair of glasses.
    """
    flat = a.flatten(1).float()
    if agg == "mean":
        return flat.mean(dim=1)
    if agg == "max":
        return flat.max(dim=1).values
    if agg == "p95":
        return torch.quantile(flat, 0.95, dim=1)
    raise ValueError(f"agg must be one of {_AGGS}, got {agg!r}")


@torch.no_grad()
def aggregate_anomaly_score(
    maps: dict[int, torch.Tensor],
    agg: str = "p95",
    block_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Collapse per-block anomaly maps into one scalar score / image.

    Each block map is first reduced over pixels with ``agg`` (``mean`` /
    ``max`` / ``p95``), then the per-block scores are combined as a
    weighted average. ``block_weights`` (one weight per block, in block
    order) is renormalized to sum to 1; ``None`` means a uniform average.

    Returns a ``(B,)`` tensor of per-image anomaly scores.
    """
    ks = sorted(maps.keys())
    per_block = torch.stack(
        [_reduce_over_pixels(maps[k], agg) for k in ks], dim=0,
    )  # (n_blocks, B)

    if block_weights is None:
        w = torch.full(
            (len(ks),), 1.0 / len(ks),
            dtype=per_block.dtype, device=per_block.device,
        )
    else:
        w = torch.as_tensor(
            list(block_weights), dtype=per_block.dtype,
            device=per_block.device,
        )
        if w.numel() != len(ks):
            raise ValueError(
                f"block_weights has {w.numel()} entries, expected {len(ks)}"
            )
        w = w / w.sum().clamp_min(1e-12)
    return (per_block * w.view(-1, 1)).sum(dim=0)  # (B,)
