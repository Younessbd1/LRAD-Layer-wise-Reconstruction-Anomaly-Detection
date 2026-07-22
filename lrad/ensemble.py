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

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from .anomaly_score import _reduce_over_pixels, aggregate_anomaly_score
from .decoder import build_decoders
from .evaluate import _auroc_entry, collect_predictions
from .model import FacialCNN, build_model

logger = logging.getLogger("celeba_ood")

TERMS: tuple[str, ...] = ("risk", "bias", "variance")


def load_ensemble_members(
    output_dir: Path, device: torch.device,
) -> tuple[list[FacialCNN], list[nn.ModuleList], int]:
    """Load every ``model_<i>/`` member written by ``scripts/run_ensemble.py``.

    Each member directory holds its own resolved architecture
    (``config.resolved.yaml``) and weights (``weights/model.pt`` +
    ``weights/decoders.pt``). Returns ``(models, decoders_list,
    image_size)`` with everything on ``device`` in eval mode. All members
    must share the same input ``image_size``.
    """
    model_dirs = sorted(
        (p for p in Path(output_dir).iterdir()
         if p.is_dir() and p.name.startswith("model_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not model_dirs:
        raise FileNotFoundError(f"no model_* dirs found under {output_dir}")

    models, decoders_list, image_size = [], [], None
    for mdir in model_dirs:
        with open(mdir / "config.resolved.yaml") as f:
            mcfg = yaml.safe_load(f)
        size = mcfg.get("dataset", {}).get("image_size", 64)
        image_size = image_size or size
        if size != image_size:
            raise ValueError(
                f"{mdir} was trained at image_size={size}, expected "
                f"{image_size} — every member must share the input size."
            )
        model = build_model(mcfg).to(device).eval()
        model.load_state_dict(
            torch.load(mdir / "weights" / "model.pt", map_location=device,
                       weights_only=True),
        )
        decoders = build_decoders(model, image_size=image_size)
        decoders = decoders.to(device).eval()
        decoders.load_state_dict(
            torch.load(mdir / "weights" / "decoders.pt",
                       map_location=device, weights_only=True),
        )
        models.append(model)
        decoders_list.append(decoders)
        logger.info(
            f"loaded {mdir.name}  channels={model.channels}  "
            f"kernel_size={model.kernel_size}"
        )
    return models, decoders_list, image_size

# Predictive-uncertainty decomposition terms (classifier-head, not pixel).
UNC_TERMS: tuple[str, ...] = ("total", "aleatoric", "epistemic")
_UNC_HEADS: tuple[str, ...] = ("gender", "attrs", "combined")
_PROB_EPS = 1e-12  # float64 entropy floor (numpy path, safe at this scale)


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
    keep_per_model_errors: bool = False,
) -> dict[int, dict[str, torch.Tensor]]:
    """Per-pixel Risk / Bias / Variance maps for one batch.

    ``recons_per_model`` is indexed ``[model][block]`` and every entry is
    a ``(B, 3, H, W)`` reconstruction. Returns ``{k: {...}}`` where each
    block ``k`` holds the ``risk``, ``bias`` and ``variance`` maps of
    shape ``(B, H, W)`` (summed over RGB) plus the ensemble-mean
    reconstruction ``mean_recon`` ``(B, 3, H, W)``. The three maps
    satisfy ``risk == bias + variance`` per pixel up to float error.

    ``keep_per_model_errors`` additionally stores the raw per-model error
    stack under ``per_model_error`` ``(M, B, H, W)`` — it is a free
    by-product of the Risk term, and the per-member scorer needs it, so
    keeping it here avoids a second full ``(x − f̂)²`` pass. Off by
    default because it multiplies the batch's map memory by ``M``.
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
        if keep_per_model_errors:
            out[k]["per_model_error"] = se_per_model
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
def sample_block_recons(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    images: torch.Tensor,
    device: torch.device,
    block: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Each member's reconstruction at ONE block depth, for a few images.

    Returns ``(images_on_device, recons)`` where ``recons`` is a length-``M``
    list of ``(B, 3, H, W)`` tensors — member ``m``'s reconstruction of the
    batch at ``block``. Unlike the pooled map helpers above, this keeps the
    members separate, which is what the per-instance figure draws (one tile
    per model). The images are returned too so callers can render against the
    exact tensor that was reconstructed.
    """
    images, recons_per_model = _ensemble_recons(
        models, decoders_list, images, device,
    )
    n_blocks = len(recons_per_model[0])
    if not 0 <= block < n_blocks:
        raise ValueError(f"block must be in [0, {n_blocks - 1}], got {block}")
    return images, [recons_per_model[m][block] for m in range(len(models))]


# Eye-region window as fractions of the image height/width. CelebA aligned
# faces put the eyes slightly above the vertical centre; after the square
# resize the eyeglasses (frame + lenses) span roughly rows 42–62 % and
# columns 15–85 % of the image. Display-independent: used only to *rank*
# OOD faces by how much bias falls on the glasses region.
EYE_REGION: tuple[float, float, float, float] = (0.42, 0.62, 0.15, 0.85)


@torch.no_grad()
def collect_eye_region_bias(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    block: int,
    region: tuple[float, float, float, float] = EYE_REGION,
) -> dict[str, np.ndarray]:
    """Per-image bias statistics at ONE block over a whole loader.

    For every image the ensemble bias map ``(x − f̄)²`` is reduced to two
    scalars: its mean over the whole image and its mean over the eye-region
    window ``region`` (fractional ``(row0, row1, col0, col1)``). The
    returned ``score`` is::

        score = eye_mean * (eye_mean / global_mean)

    which is large only when the bias is *strong* in the eye region AND
    *concentrated* there — exactly "the glasses are well detected as the
    anomaly". A face with equally high bias everywhere (e.g. a globally bad
    reconstruction) gets no concentration boost. Memory stays ``O(N)``:
    only the scalars are kept, never the maps.

    Returns ``{'global_mean': (N,), 'eye_mean': (N,), 'score': (N,)}`` in
    loader order.
    """
    r0f, r1f, c0f, c1f = region
    if not (0.0 <= r0f < r1f <= 1.0 and 0.0 <= c0f < c1f <= 1.0):
        raise ValueError(f"bad eye region {region!r}")

    global_means: list[np.ndarray] = []
    eye_means: list[np.ndarray] = []
    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        recons = torch.stack(
            [block_reconstructions(models[m], decoders_list[m], img)[block]
             for m in range(len(models))],
            dim=0,
        )  # (M, B, 3, H, W)
        bias = ((img - recons.mean(dim=0)) ** 2).sum(dim=1)  # (B, H, W)
        H, W = bias.shape[-2:]
        r0, r1 = int(round(r0f * H)), max(int(round(r1f * H)), 1)
        c0, c1 = int(round(c0f * W)), max(int(round(c1f * W)), 1)
        global_means.append(bias.flatten(1).mean(dim=1).cpu().numpy())
        eye_means.append(
            bias[:, r0:r1, c0:c1].flatten(1).mean(dim=1).cpu().numpy()
        )

    g = (np.concatenate(global_means) if global_means else np.zeros(0))
    e = (np.concatenate(eye_means) if eye_means else np.zeros(0))
    return {
        "global_mean": g,
        "eye_mean": e,
        "score": e * (e / np.maximum(g, 1e-12)),
    }


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
          'per_member': {'per_block': [{k: (N,)}], 'aggregated': [(N,)]},
        }

    for ``term`` in ``('risk', 'bias', 'variance')``. ``per_member`` holds
    each member's OWN reconstruction-error scores (``sum_c (x − f̂^m)²``
    reduced with the same ``agg``), indexed by member — the raw material
    for the architecture-effect analysis (how channel widths / kernel size
    affect each member's bias detection).
    """
    n_models = len(models)
    per_block: dict[str, dict[int, list]] = {t: {} for t in TERMS}
    aggregated: dict[str, list] = {t: [] for t in TERMS}
    member_per_block: list[dict[int, list]] = [{} for _ in range(n_models)]
    member_aggregated: list[list] = [[] for _ in range(n_models)]

    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        maps = decomposition_maps(
            img, recons_per_model, keep_per_model_errors=True,
        )
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
        # Member scores reuse the error stack the Risk term was built
        # from — no second (x − f̂)² pass over the batch.
        for m in range(n_models):
            err_maps = {k: maps[k]["per_model_error"][m] for k in maps}
            for k, mp in err_maps.items():
                member_per_block[m].setdefault(k, []).append(
                    _reduce_over_pixels(mp, agg).cpu().numpy()
                )
            member_aggregated[m].append(
                aggregate_anomaly_score(err_maps, agg, block_weights)
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
        "per_member": {
            "per_block": [
                {k: np.concatenate(v) for k, v in member_per_block[m].items()}
                for m in range(n_models)
            ],
            "aggregated": [
                (np.concatenate(member_aggregated[m]) if member_aggregated[m]
                 else np.zeros(0))
                for m in range(n_models)
            ],
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
    convenience for callers. ``member_auroc`` scores each member ALONE
    (its own reconstruction error, same ``agg``), one dict per member with
    ``aggregated`` and ``per_block`` AUROCs — the y-axis of the
    architecture-effect plots.
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

    member_auroc: list[dict] = []
    for m in range(len(models)):
        mem_in = scores_in["per_member"]
        mem_ood = scores_ood["per_member"]
        member_auroc.append({
            "aggregated": _auroc_entry(
                mem_in["aggregated"][m], mem_ood["aggregated"][m],
            ).get("auroc", float("nan")),
            "per_block": [
                _auroc_entry(
                    mem_in["per_block"][m][k], mem_ood["per_block"][m][k],
                ).get("auroc", float("nan"))
                for k in ks
            ],
        })

    return {
        "agg": agg,
        "blocks": ks,
        "n_models": len(models),
        "scores_in": scores_in,
        "scores_ood": scores_ood,
        "auroc": auroc,
        "per_block_auroc": per_block_auroc,
        "member_auroc": member_auroc,
        # The OOD anomaly is the bias term itself: bias = risk − variance
        # = (x − f̄)², with no sigma and no division. Surface it directly so
        # callers don't have to know it lives under the "bias" key.
        "anomaly_auroc": {
            "aggregated": auroc["score_bias_aggregated"].get("auroc"),
            "per_block": list(per_block_auroc["bias"]),
        },
    }


# ---------------------------------------------------------------------------
# Predictive-uncertainty decomposition (classifier heads, not reconstruction)
# ---------------------------------------------------------------------------
#
# The pixel-space Variance term is a weak OOD signal for an *occlusion* task:
# when eyeglasses hide the face, every member fails to reconstruct the same
# region, so the error is large but *correlated* — high Bias, low Variance
# (model disagreement). The disagreement that actually grows out-of-
# distribution lives in the classifier's *predictions*, not its pixels.
#
# For a deep ensemble the standard decomposition of predictive uncertainty is
#
#     Total       = H( mean_m p_m )                  (entropy of mean pred)
#     Aleatoric   = mean_m H( p_m )                  (expected member entropy)
#     Epistemic   = Total − Aleatoric  =  MI         (member disagreement)
#
# where the Epistemic term is the mutual information between the label and the
# model parameters (BALD). It is exactly the *variance* of the ensemble's
# predictive distribution, and — unlike pixel variance — it rises on OOD
# inputs because off-manifold the independently-trained members extrapolate to
# *different* predictions. This is the variance-based OOD score that works.


def _categorical_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy (nats) over the last axis of a categorical pmf ``(..., C)``."""
    p = np.clip(p, _PROB_EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _bernoulli_entropy_np(p: np.ndarray) -> np.ndarray:
    """Per-element binary entropy (nats) for Bernoulli probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _uncertainty_scores(
    gender_probs: np.ndarray,  # (M, N, C)
    attr_probs: np.ndarray,    # (M, N, A)
) -> dict[str, dict[str, np.ndarray]]:
    """Total / aleatoric / epistemic uncertainty for each head + combined.

    Inputs are per-member predictive probabilities stacked on axis 0. Returns
    ``{head: {term: (N,)}}`` for ``head`` in ``gender / attrs / combined`` and
    ``term`` in ``total / aleatoric / epistemic``. The combined head sums the
    gender and (attr-averaged) terms — the summed two-head entropy.
    """
    out: dict[str, dict[str, np.ndarray]] = {}

    # --- gender (categorical) ---
    mean_g = gender_probs.mean(axis=0)                       # (N, C)
    total_g = _categorical_entropy(mean_g)                   # (N,)
    alea_g = _categorical_entropy(gender_probs).mean(axis=0)  # (N,)
    out["gender"] = {
        "total": total_g,
        "aleatoric": alea_g,
        "epistemic": total_g - alea_g,
    }

    # --- attributes (per-attr Bernoulli, averaged over attrs) ---
    mean_a = attr_probs.mean(axis=0)                          # (N, A)
    total_a = _bernoulli_entropy_np(mean_a).mean(axis=-1)     # (N,)
    alea_a = _bernoulli_entropy_np(attr_probs).mean(axis=0).mean(axis=-1)
    out["attrs"] = {
        "total": total_a,
        "aleatoric": alea_a,
        "epistemic": total_a - alea_a,
    }

    # --- combined (gender + attrs): summed two-head entropy terms ---
    out["combined"] = {
        t: out["gender"][t] + out["attrs"][t] for t in UNC_TERMS
    }
    return out


@torch.no_grad()
def _collect_member_probs(
    models: Sequence[FacialCNN],
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack every member's predictive probabilities over a loader.

    Returns ``(gender_probs, attr_probs)`` with shapes ``(M, N, C)`` and
    ``(M, N, A)``. Reuses :func:`collect_predictions` so the per-member
    forward pass is identical to the single-model evaluation.
    """
    gender, attrs = [], []
    for m in models:
        preds = collect_predictions(m, loader, device)
        gender.append(preds["gender_probs"])
        attrs.append(preds["attr_probs"])
    return np.stack(gender, axis=0), np.stack(attrs, axis=0)


@torch.no_grad()
def evaluate_ensemble_uncertainty(
    models: Sequence[FacialCNN],
    loaders: dict,
    device: torch.device,
) -> dict:
    """OOD AUROC for the predictive total / aleatoric / epistemic uncertainty.

    For every classifier head (``gender``, ``attrs``, ``combined``) and every
    uncertainty term, scores ``test_in`` (label 0) vs ``test_ood`` (label 1).
    The headline OOD signal is ``combined / epistemic`` — the ensemble's
    predictive mutual information, i.e. the *variance* of its predictions —
    surfaced under ``epistemic_auroc`` for callers.
    """
    g_in, a_in = _collect_member_probs(models, loaders["test_in"], device)
    g_ood, a_ood = _collect_member_probs(models, loaders["test_ood"], device)

    scores_in = _uncertainty_scores(g_in, a_in)
    scores_ood = _uncertainty_scores(g_ood, a_ood)

    auroc: dict[str, dict[str, dict]] = {}
    for head in _UNC_HEADS:
        auroc[head] = {
            t: _auroc_entry(scores_in[head][t], scores_ood[head][t])
            for t in UNC_TERMS
        }

    return {
        "n_models": len(models),
        "heads": list(_UNC_HEADS),
        "terms": list(UNC_TERMS),
        "auroc": auroc,
        # Headline: epistemic (mutual information) = the predictive-variance
        # OOD score. Combined head is the most informative.
        "epistemic_auroc": {
            head: auroc[head]["epistemic"].get("auroc") for head in _UNC_HEADS
        },
    }
