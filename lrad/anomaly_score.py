"""Bias/variance debiasing of per-block reconstruction error maps.

The raw per-pixel reconstruction error of a block decoder mixes three
additive components::

    raw_error(x)[p]  =  bias_k(p)  +  variance_k(p)  +  anomaly_k(x)[p]

  * ``bias_k(p)``     — a *systematic* offset present on **every**
                        in-distribution image. It comes from the block's
                        spatial compression, from under-represented
                        regions of the CelebA distribution, and from the
                        decoder's limited capacity at high frequencies.
                        Estimated by the per-pixel **mean** ``mu_k`` of
                        the error over a clean reference set.

  * ``variance_k(p)`` — the residual training fluctuation of the error
                        at that pixel. Estimated by the per-pixel
                        **standard deviation** ``sigma_k`` over the same
                        clean set.

  * ``anomaly_k``     — the only term that carries OOD signal (e.g.
                        eyeglasses occluding the face). It is what we
                        want to keep.

Subtracting ``mu_k`` removes the bias and dividing by ``sigma_k``
whitens the residual into *sigma units*, so a value of ``a_k(x)[p] = 5``
means "this pixel reconstructs 5 standard deviations worse than a
typical clean face does at that location". A final ``max(0, ·)`` keeps
only pixels that are *worse* than expected (better-than-average
reconstruction is never anomalous)::

    a_k(x)[p]  =  max( 0,  (raw_error(x)[p] - mu_k(p)) / (sigma_k(p) + eps) )

These cleaned, whitened maps are then reduced to a scalar per image and
combined across blocks to obtain the final anomaly score.

Everything here runs under ``@torch.no_grad()``. ``mu_k`` / ``sigma_k``
are estimated **once** on a clean reference loader (use ``test_in`` — not
``train`` — to avoid leaking the very training bias we are trying to
remove).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .model import FacialCNN

_AGGS = ("mean", "max", "p95")


@torch.no_grad()
def _per_pixel_errors(
    model: FacialCNN,
    decoders: nn.ModuleList,
    images: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Raw per-pixel reconstruction error for every conv block.

    ``error_k = mean_RGB |x - decoder_k(activation_k)|`` upsampled to the
    input resolution. Returns ``{k: (B, H, W)}`` where ``H, W`` is the
    *image* size (64x64), not the activation size.
    """
    model.eval()
    decoders.eval()
    _, acts = model.forward_features(images)
    errs: dict[int, torch.Tensor] = {}
    for k, dec in enumerate(decoders):
        recon = dec(acts[k])
        errs[k] = (images - recon).abs().mean(dim=1)  # (B, H, W)
    return errs


def _clean_errors(
    errs: dict[int, torch.Tensor],
    baseline_stats: dict,
    eps: float = 1e-6,
) -> dict[int, torch.Tensor]:
    """Debias + whiten precomputed raw error maps.

    ``a_k = max(0, (error_k - mu_k) / (sigma_k + eps))``. ``mu_k`` and
    ``sigma_k`` are moved onto the device of the error tensors.
    """
    cleaned: dict[int, torch.Tensor] = {}
    for k, e in errs.items():
        mu = baseline_stats[k]["mu"].to(e.device, e.dtype)
        sigma = baseline_stats[k]["sigma"].to(e.device, e.dtype)
        cleaned[k] = ((e - mu) / (sigma + eps)).clamp_min(0.0)
    return cleaned


@torch.no_grad()
def estimate_baseline_stats(
    model: FacialCNN,
    decoders: nn.ModuleList,
    loader,
    device: torch.device,
) -> dict:
    """Estimate the per-pixel reconstruction-error bias and variance.

    Iterates **once** over a clean reference loader (use ``test_in``) and
    accumulates, for every conv block ``k`` and every image pixel ``p``:

      * ``mu_k(p)``    — mean raw error  = the systematic **bias** of the
                         decoder/dataset at that pixel.
      * ``sigma_k(p)`` — std of the raw error = the residual training
                         **variance** at that pixel.

    A batched Welford (Chan parallel) update keeps a single ``(H, W)``
    mean/M2 accumulator per block, so memory is independent of the
    dataset size. Accumulation is done in float64 for numerical safety.

    Returns ``{k: {'mu': (H, W) tensor, 'sigma': (H, W) tensor}}`` on CPU
    (callers move them onto the batch device as needed).
    """
    model.eval()
    decoders.eval()
    acc: dict[int, dict] = {}

    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        errs = _per_pixel_errors(model, decoders, img)
        for k, e in errs.items():
            e = e.double()  # (b, H, W)
            b = e.shape[0]
            if k not in acc:
                hw = e.shape[1:]
                acc[k] = {
                    "n": 0,
                    "mean": torch.zeros(hw, dtype=torch.float64, device=device),
                    "m2": torch.zeros(hw, dtype=torch.float64, device=device),
                }
            a = acc[k]
            n = a["n"]
            batch_mean = e.mean(dim=0)
            batch_m2 = ((e - batch_mean) ** 2).sum(dim=0)
            delta = batch_mean - a["mean"]
            new_n = n + b
            a["mean"] += delta * (b / new_n)
            a["m2"] += batch_m2 + delta**2 * (n * b / new_n)
            a["n"] = new_n

    stats: dict[int, dict] = {}
    for k, a in acc.items():
        n = max(a["n"], 1)
        mu = a["mean"]
        var = (a["m2"] / n).clamp_min(0.0)
        stats[k] = {
            "mu": mu.float().cpu(),
            "sigma": var.sqrt().float().cpu(),
        }
    return stats


@torch.no_grad()
def cleaned_anomaly_maps(
    model: FacialCNN,
    decoders: nn.ModuleList,
    images: torch.Tensor,
    baseline_stats: dict,
    device: torch.device,
    eps: float = 1e-6,
) -> dict[int, torch.Tensor]:
    """Debiased, variance-whitened anomaly map per block for a batch.

    For each block ``k``::

        a_k(x)[p] = max(0, (|x - recon_k|[p] - mu_k(p)) / (sigma_k(p) + eps))

    where ``mu_k`` is the estimated **bias** and ``sigma_k`` the
    estimated **variance** (std) at pixel ``p``. The result is expressed
    in sigma units, so on clean (in-distribution) faces the map is close
    to 0 everywhere while genuine anomalies (eyeglasses) light up — and
    values of 10-30 sigma are normal and expected for strong anomalies.

    Returns ``{k: (B, H, W)}``.
    """
    images = images.to(device, non_blocking=True)
    errs = _per_pixel_errors(model, decoders, images)
    return _clean_errors(errs, baseline_stats, eps=eps)


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
    cleaned_maps: dict[int, torch.Tensor],
    agg: str = "p95",
    block_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Collapse the per-block cleaned maps into one scalar score / image.

    Each block map is first reduced over pixels with ``agg`` (``mean`` /
    ``max`` / ``p95``), then the per-block scores are combined as a
    weighted average. ``block_weights`` (one weight per block, in block
    order) is renormalized to sum to 1; ``None`` means a uniform average.

    Returns a ``(B,)`` tensor of per-image anomaly scores.
    """
    ks = sorted(cleaned_maps.keys())
    per_block = torch.stack(
        [_reduce_over_pixels(cleaned_maps[k], agg) for k in ks], dim=0,
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
