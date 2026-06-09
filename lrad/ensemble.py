"""Ensemble bias/variance decomposition of the per-block reconstruction.

This implements the decomposition from the LRAD note. Train ``M`` models
independently — a *deep ensemble*: diversity comes purely from the random
weight init and the SGD shuffle order, **not** from MC-Dropout (the
Dropout layer was removed for exactly this reason). Each model owns a
full classifier + per-block decoders, so each yields its own
reconstruction ``f_hat^m`` of the input image.

For a conv block ``k``, an image ``x`` and a pixel ``i`` (the per-pixel
error is the L2 squared error summed over the RGB channels,
``sum_c ( x_c - f_c )^2`` — no mean, no sqrt, so a pixel lives in
``[0, 3]``), with the ``M`` reconstructions ``f_hat^m`` and their ensemble
mean ``f_bar = (1/M) * sum_m f_hat^m``::

    Risk_k(x)[i]      = (1/M) * sum_m ( x[i] - f_hat^m(x)[i] )^2
    Bias_k(x)[i]      = ( x[i] - f_bar(x)[i] )^2
    Variance_k(x)[i]  = (1/M) * sum_m ( f_hat^m(x)[i] - f_bar(x)[i] )^2

These obey the exact algebraic identity, pixel by pixel::

    Risk = Bias + Variance

So ``Risk`` is what a single (average) model costs, ``Bias`` is the error
that survives even after ensembling — scoring the *mean* model isolates
the bias term — and ``Variance`` is how much the ``M`` models disagree.
The variance term doubles as an epistemic-uncertainty OOD signal: on
out-of-distribution inputs the models extrapolate differently, so they
disagree more.

Everything runs under ``torch.no_grad()``. Maps are reduced to per-image
scalars with the same ``agg`` reduction (``mean`` / ``max`` / ``p95``)
used elsewhere in the project, so the three terms stay comparable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .anomaly_score import _reduce_over_pixels, aggregate_anomaly_score
from .evaluate import _auroc_entry
from .model import FacialCNN

TERMS: tuple[str, ...] = ("risk", "bias", "variance")


@torch.no_grad()
def block_reconstructions(
    model: FacialCNN,
    decoders: nn.ModuleList,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-block decoder reconstructions ``(B, 3, H, W)`` for one model."""
    model.eval()
    decoders.eval()
    _, acts = model.forward_features(images)
    return [dec(acts[k]) for k, dec in enumerate(decoders)]


@torch.no_grad()
def decomposition_maps(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Per-pixel Risk / Bias / Variance maps for one batch.

    ``recons_per_model`` is indexed ``[model][block]`` and every entry is
    a ``(B, 3, H, W)`` reconstruction. Returns ``{k: {...}}`` where each
    block ``k`` holds the ``risk``, ``bias`` and ``variance`` maps of
    shape ``(B, H, W)`` (summed over RGB) plus the ensemble-mean
    reconstruction ``mean_recon`` ``(B, 3, H, W)``. The three maps
    satisfy ``risk == bias + variance`` per pixel up to float error.
    """
    n_models = len(recons_per_model)
    if n_models == 0:
        raise ValueError("need at least one model in the ensemble")
    n_blocks = len(recons_per_model[0])
    out: dict[int, dict[str, torch.Tensor]] = {}
    for k in range(n_blocks):
        recons = torch.stack(
            [recons_per_model[m][k] for m in range(n_models)], dim=0,
        )  # (M, B, 3, H, W)
        mean_recon = recons.mean(dim=0)  # (B, 3, H, W)
        # Per-model squared error, summed over RGB -> (M, B, H, W). The L2
        # squared error per pixel is sum_c (x_c - f_c)^2 (no mean, no sqrt),
        # so a pixel lives in [0, 3] for 3 channels. Summing (not averaging)
        # over RGB keeps the exact Risk = Bias + Variance identity, which
        # holds channel by channel and therefore survives the channel sum.
        se_per_model = ((images.unsqueeze(0) - recons) ** 2).sum(dim=2)
        risk = se_per_model.mean(dim=0)  # (B, H, W)
        bias = ((images - mean_recon) ** 2).sum(dim=1)  # (B, H, W)
        variance = (
            ((recons - mean_recon.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=0)
        )  # (B, H, W)
        out[k] = {
            "risk": risk,
            "bias": bias,
            "variance": variance,
            "mean_recon": mean_recon,
        }
    return out


@torch.no_grad()
def _per_model_error_stack(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
    k: int,
) -> torch.Tensor:
    """Per-model squared error stack ``(M, B, H, W)`` for block ``k``.

    The per-pixel error is the RGB-summed squared error
    ``sum_c (x_c − f̂^m_c)²`` (no mean, no sqrt — a pixel lives in
    ``[0, 3]``), consistent with :func:`decomposition_maps`. Shared
    backend for the mean / min / quantile-min error maps.
    """
    n_models = len(recons_per_model)
    if n_models == 0:
        raise ValueError("need at least one model in the ensemble")
    recons = torch.stack(
        [recons_per_model[m][k] for m in range(n_models)], dim=0,
    )  # (M, B, 3, H, W)
    return ((images.unsqueeze(0) - recons) ** 2).sum(dim=2)


@torch.no_grad()
def _ensemble_recons(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Move ``images`` to ``device`` and reconstruct with every model."""
    images = images.to(device, non_blocking=True)
    recons_per_model = [
        block_reconstructions(models[m], decoders_list[m], images)
        for m in range(len(models))
    ]
    return images, recons_per_model


@torch.no_grad()
def mean_error_maps(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Per-block ensemble-averaged squared error ``mean_m (x − f̂^m_k)²``.

    ``recons_per_model`` is indexed ``[model][block]`` (each entry a
    ``(B, 3, H, W)`` reconstruction). Returns ``{k: (B, H, W)}`` where each
    pixel is the mean over the ``M`` models of the RGB-summed squared
    reconstruction error ``sum_c ( x_c − f̂^m_c )²`` (a pixel lives in
    ``[0, 3]``).

    This is exactly the ``Risk`` term of the decomposition: the *average of
    the per-model error maps*, as opposed to the ``Bias`` term ``(x − f̄)²``
    which is the error of the *averaged* reconstruction. The squared error
    (rather than the absolute error) is what makes ``Risk = Bias + Variance``
    hold pixel by pixel, so the gap between this map and the bias map is the
    variance — the model disagreement.
    """
    n_models = len(recons_per_model)
    if n_models == 0:
        raise ValueError("need at least one model in the ensemble")
    n_blocks = len(recons_per_model[0])
    return {
        # Per-model squared error, summed over RGB -> (M, B, H, W),
        # then averaged over the M models -> (B, H, W).
        k: _per_model_error_stack(images, recons_per_model, k).mean(dim=0)
        for k in range(n_blocks)
    }


@torch.no_grad()
def sample_mean_error_maps(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Ensemble-averaged squared error maps for a small set of samples."""
    images, recons_per_model = _ensemble_recons(
        models, decoders_list, images, device,
    )
    return mean_error_maps(images, recons_per_model)


@torch.no_grad()
def min_error_maps(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Per-block ensemble *minimum* squared error ``min_m (x − f̂^m_k)²``.

    Same inputs/shape contract as :func:`mean_error_maps`, but at every
    pixel it keeps the *smallest* per-model error instead of the average.
    This answers "how well does the **best** member reconstruct this pixel":
    a region stays bright only if *no* model in the ensemble can reconstruct
    it (a stronger OOD signal than the mean, which a single bad member can
    inflate).
    """
    n_models = len(recons_per_model)
    if n_models == 0:
        raise ValueError("need at least one model in the ensemble")
    n_blocks = len(recons_per_model[0])
    return {
        # Per-model squared error, summed over RGB -> (M, B, H, W),
        # then the per-pixel minimum over the M models -> (B, H, W).
        k: _per_model_error_stack(images, recons_per_model, k)
        .min(dim=0).values
        for k in range(n_blocks)
    }


@torch.no_grad()
def sample_min_error_maps(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Ensemble per-pixel *minimum* squared error maps for a few samples."""
    images, recons_per_model = _ensemble_recons(
        models, decoders_list, images, device,
    )
    return min_error_maps(images, recons_per_model)


@torch.no_grad()
def quantile_min_error_maps(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
    k: int = 3,
) -> dict[int, torch.Tensor]:
    """Per-block *robust minimum*: the ``k``-th smallest per-model error.

    Same inputs/shape contract as :func:`min_error_maps`, but at every
    pixel the ``M`` per-model squared errors are sorted in increasing
    order and the ``k``-th smallest is kept (``torch.kthvalue``).
    ``k=1`` reproduces :func:`min_error_maps` exactly; ``k>1`` makes the
    minimum robust to a few lucky members — a pixel only stays dark if at
    least ``k`` models reconstruct it well.
    """
    n_models = len(recons_per_model)
    if n_models == 0:
        raise ValueError("need at least one model in the ensemble")
    if not 1 <= k <= n_models:
        raise ValueError(
            f"k must be in [1, {n_models}] (ensemble size), got {k}"
        )
    n_blocks = len(recons_per_model[0])
    return {
        b: _per_model_error_stack(images, recons_per_model, b)
        .kthvalue(k, dim=0).values
        for b in range(n_blocks)
    }


@torch.no_grad()
def sample_quantile_min_error_maps(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
    k: int = 3,
) -> dict[int, torch.Tensor]:
    """Robust-minimum (k-th smallest) error maps for a few samples."""
    images, recons_per_model = _ensemble_recons(
        models, decoders_list, images, device,
    )
    return quantile_min_error_maps(images, recons_per_model, k=k)


def identity_residual(maps: dict[int, dict[str, torch.Tensor]]) -> float:
    """Largest absolute deviation from ``Risk = Bias + Variance``.

    A correct decomposition returns a value at float32 noise level
    (~1e-6); anything larger signals a bug in the maps.
    """
    worst = 0.0
    for m in maps.values():
        dev = (m["risk"] - m["bias"] - m["variance"]).abs().max().item()
        worst = max(worst, dev)
    return worst


@torch.no_grad()
def sample_decomposition(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    """Decomposition maps for a small set of sample images (for plots)."""
    images, recons_per_model = _ensemble_recons(
        models, decoders_list, images, device,
    )
    return decomposition_maps(images, recons_per_model)


@torch.no_grad()
def collect_decomposition_scores(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    agg: str = "p95",
    block_weights: Sequence[float] | None = None,
) -> dict:
    """Per-image Risk / Bias / Variance scalar scores over a whole loader.

    For each batch every model reconstructs the input, the per-pixel
    decomposition maps are formed, and each map is reduced to one scalar
    per image with ``agg`` (per block) and also aggregated across blocks
    with the same reduction. Memory stays ``O(N)`` — only the scalar
    scores are kept, never the full maps.

    Returns::

        {
          'per_block':  {term: {k: (N,)}},
          'aggregated': {term: (N,)},
        }

    for ``term`` in ``('risk', 'bias', 'variance')``.
    """
    per_block: dict[str, dict[int, list]] = {t: {} for t in TERMS}
    aggregated: dict[str, list] = {t: [] for t in TERMS}

    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        maps = decomposition_maps(img, recons_per_model)
        for t in TERMS:
            term_maps = {k: maps[k][t] for k in maps}
            for k, mp in term_maps.items():
                per_block[t].setdefault(k, []).append(
                    _reduce_over_pixels(mp, agg).cpu().numpy()
                )
            aggregated[t].append(
                aggregate_anomaly_score(term_maps, agg, block_weights)
                .cpu().numpy()
            )

    return {
        "per_block": {
            t: {k: np.concatenate(v) for k, v in per_block[t].items()}
            for t in TERMS
        },
        "aggregated": {
            t: (np.concatenate(aggregated[t]) if aggregated[t]
                else np.zeros(0))
            for t in TERMS
        },
    }


@torch.no_grad()
def evaluate_ensemble_decomposition(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loaders: dict,
    device: torch.device,
    agg: str = "p95",
    block_weights: Sequence[float] | None = None,
) -> dict:
    """Full ensemble decomposition: per-block and aggregated OOD AUROC for
    the Risk, Bias and Variance scores on ``test_in`` vs ``test_ood``.

    The ``auroc`` dict uses the keys ``score_{term}_per_block_{k}`` and
    ``score_{term}_aggregated`` so it merges straight into a results
    ``auroc`` mapping. ``per_block_auroc`` holds one AUROC list per term,
    in block order, ready for the bar plot. ``anomaly_auroc`` surfaces the
    headline OOD score — the Bias term ``bias = risk − variance`` — as a
    convenience for callers.
    """
    scores_in = collect_decomposition_scores(
        models, decoders_list, loaders["test_in"], device, agg, block_weights,
    )
    scores_ood = collect_decomposition_scores(
        models, decoders_list, loaders["test_ood"], device, agg, block_weights,
    )

    ks = sorted(scores_in["per_block"]["risk"].keys())
    auroc: dict = {}
    per_block_auroc: dict[str, list[float]] = {t: [] for t in TERMS}
    for t in TERMS:
        for k in ks:
            entry = _auroc_entry(
                scores_in["per_block"][t][k],
                scores_ood["per_block"][t][k],
            )
            auroc[f"score_{t}_per_block_{k}"] = entry
            per_block_auroc[t].append(entry.get("auroc", float("nan")))
        auroc[f"score_{t}_aggregated"] = _auroc_entry(
            scores_in["aggregated"][t], scores_ood["aggregated"][t],
        )

    return {
        "agg": agg,
        "blocks": ks,
        "n_models": len(models),
        "scores_in": scores_in,
        "scores_ood": scores_ood,
        "auroc": auroc,
        "per_block_auroc": per_block_auroc,
        # The OOD anomaly is the bias term itself: bias = risk − variance
        # = (x − f̄)², with no sigma and no division. Surface it directly so
        # callers don't have to know it lives under the "bias" key.
        "anomaly_auroc": {
            "aggregated": auroc["score_bias_aggregated"].get("auroc"),
            "per_block": list(per_block_auroc["bias"]),
        },
    }
