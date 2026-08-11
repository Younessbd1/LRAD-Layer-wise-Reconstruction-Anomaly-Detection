"""Pixel-level anomaly localization metrics: pixel AUROC and PRO.

PatchCore reports three numbers on MVTec AD, and image-level AUROC is only
the first of them. The other two score the *localization* — where the model
says the defect is — against MVTec's ground-truth masks:

* **pixel AUROC** — AUROC over every pixel of every test image, pooled.
  Simple, but dominated by large defects: one big region contributes
  thousands of positive pixels, a hairline scratch a few dozen.
* **PRO** (per-region overlap) [Bergmann et al., CVPR 2020] — for each
  *connected component* of the ground truth, the fraction of that region
  recovered at a given threshold, averaged over regions with **equal weight
  per region regardless of size**, then integrated over the false-positive
  rate up to ``fpr_limit`` (0.30 by convention) and normalized by it. This
  is the metric that refuses to let a method win by finding only the big
  defects, which is exactly why the MVTec literature reports it.

Reference numbers to beat, from the PatchCore paper (Tables 2 and 3, mean
over the 15 categories): pixel AUROC 98.1, PRO 93.5 for PatchCore-10%;
PaDiM 97.5 / 92.1; SPADE 96.0 / 91.7.

The anomaly map fed to both metrics is the ensemble **Bias** term
``(x − f̄_k)²`` — the same quantity the image-level score reduces, just kept
at full resolution instead of collapsed to a scalar. See
:func:`collect_anomaly_maps`.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .ensemble import block_reconstructions, decomposition_maps
from .model import FacialCNN

logger = logging.getLogger("celeba_ood")

# PatchCore smooths its segmentation map with a Gaussian of width sigma=4
# ("but did not optimize this parameter", §3.3). We keep the same default
# so the localization comparison is like for like.
DEFAULT_SMOOTH_SIGMA = 4.0

# The conventional integration limit for PRO. Bergmann et al. integrate the
# per-region overlap curve over FPR in [0, 0.3] and divide by 0.3, so a
# perfect localizer scores 1.0 and the number stays comparable across papers.
DEFAULT_FPR_LIMIT = 0.30

# Below this "tracking ratio" — how much a reconstruction varies between
# images, relative to how much the inputs do — the decoders are treated as
# collapsed to the dataset mean (see collect_anomaly_maps). This is a coarse
# safety net for the unambiguous case, not a calibrated pass/fail line:
# a randomly initialized decoder sits near 0.2 and a 420-step MVTec run
# measured 0.15, so the threshold is set below both to avoid crying wolf,
# and the ratio itself is logged and returned on every run.
COLLAPSE_RATIO_WARN = 0.10

# numpy 2.0 renamed ``trapz`` to ``trapezoid``; requirements.txt still pins
# numpy<2.0 for the CUDA 11.8 environment, so bind whichever exists rather
# than making the metric depend on which numpy the node happens to have.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@torch.no_grad()
def collect_anomaly_maps(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    *,
    term: str = "bias",
    blocks: Sequence[int] | None = None,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
) -> np.ndarray:
    """Full-resolution per-image anomaly maps over a loader → ``(N, H, W)``.

    The per-block decomposition maps are averaged over ``blocks`` (all of
    them by default) after each block is normalized to a comparable scale,
    then optionally Gaussian-smoothed.

    **Why the per-block normalization.** Reconstruction error grows sharply
    with tap depth — the deepest decoder starts from an 8×8 map and can only
    paint a blur, so its raw error is an order of magnitude above block 0's.
    Averaging the raw maps would therefore be a disguised "use the deepest
    block only". Each block map is divided by its own mean over the batch
    axis before averaging, which puts the blocks on one scale and lets the
    shallow, sharp blocks actually contribute to the localization.
    """
    if term not in ("risk", "bias", "variance"):
        raise ValueError(f"term must be risk/bias/variance, got {term!r}")

    chunks: list[np.ndarray] = []
    # Running per-pixel sums for the collapse check below, over the whole
    # loader: how much the reconstructions vary BETWEEN images, against how
    # much the inputs do.
    n_seen = 0
    img_sum = img_sq = None
    rec_sum: list | None = None
    rec_sq: list | None = None

    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        maps = decomposition_maps(img, recons_per_model)
        ks = sorted(maps.keys()) if blocks is None else list(blocks)
        stack = torch.stack([maps[k][term] for k in ks], dim=0)  # (K, B, H, W)
        # Per-block scale normalization (see docstring). clamp_min keeps a
        # block whose error is genuinely ~0 from exploding the ratio.
        scale = stack.flatten(1).mean(dim=1).clamp_min(1e-12)
        stack = stack / scale.view(-1, 1, 1, 1)
        chunks.append(stack.mean(dim=0).cpu().numpy())

        n_seen += img.shape[0]
        f = img.double().flatten(1)
        img_sum = f.sum(0) if img_sum is None else img_sum + f.sum(0)
        img_sq = (f * f).sum(0) if img_sq is None else img_sq + (f * f).sum(0)
        recs = [maps[k]["mean_recon"].double().flatten(1) for k in ks]
        if rec_sum is None:
            rec_sum = [r.sum(0) for r in recs]
            rec_sq = [(r * r).sum(0) for r in recs]
        else:
            for j, r in enumerate(recs):
                rec_sum[j] += r.sum(0)
                rec_sq[j] += (r * r).sum(0)

    if not chunks:
        return np.zeros((0, 0, 0), dtype=np.float32)
    out = np.concatenate(chunks, axis=0).astype(np.float32)

    # Collapse check. Undertrained decoders converge to the dataset mean and
    # emit near-identical output whatever the input; the "anomaly" map then
    # degenerates into |x − mean image|, which tracks image brightness and
    # can score BELOW 0.5 pixel AUROC (MVTec defects are often dark, so they
    # reconstruct MORE easily than nominal texture). Otherwise this failure
    # is completely silent — the run finishes and the figures render.
    #
    # The test is whether a reconstruction VARIES BETWEEN IMAGES as much as
    # the input does. Comparing the map's between-image to its within-image
    # spread would not work: on a dataset whose images all look alike (every
    # MVTec category) those are legitimately close, and on random inputs
    # they are equal by construction, so that form both false-alarms and
    # misses. Ratio against the input's own spread has a fixed meaning —
    # 1.0 is a decoder tracking its input, 0.0 one ignoring it.
    if n_seen > 1 and img_sum is not None:
        def _std(s, sq):
            var = (sq / n_seen) - (s / n_seen) ** 2
            return float(var.clamp_min(0).sqrt().mean())

        img_std = _std(img_sum, img_sq)
        # Deep blocks are legitimately blurry, so judge on the best-tracking
        # block: if even that one has lost the input, all of them have.
        best = max(_std(rec_sum[j], rec_sq[j]) for j in range(len(rec_sum)))
        ratio = best / img_std if img_std > 1e-8 else float("nan")
        collect_anomaly_maps.last_tracking_ratio = ratio
        # Always report it — the number is the useful part, and it lands in
        # the run's summary. The warning threshold is deliberately low: it
        # is a coarse safety net for the unambiguous case (a decoder that
        # essentially ignores its input), NOT a calibrated pass/fail line.
        # A well-trained decoder should sit far above it.
        logger.info(
            "decoder tracking ratio = %.2f (reconstruction vs input "
            "between-image variation; 1.0 = tracks its input fully)", ratio,
        )
        if ratio == ratio and ratio < COLLAPSE_RATIO_WARN:
            logger.warning(
                "reconstructions vary only %.0f%% as much between images as "
                "the inputs do (%.4g vs %.4g) — the decoders look COLLAPSED "
                "to the dataset mean, so this map tracks image brightness "
                "rather than anomaly and its pixel AUROC may fall below 0.5. "
                "Train longer (training.decoders.steps).",
                100 * ratio, best, img_std,
            )

    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Smooth each image independently — sigma=0 on the batch axis would
        # otherwise blend neighbouring test images into each other.
        out = gaussian_filter(out, sigma=(0, smooth_sigma, smooth_sigma))
    return out


def load_masks(dataset, indices: Sequence[int] | None = None) -> np.ndarray:
    """Stack a dataset's ground-truth masks into ``(N, H, W)`` in {0, 1}.

    Works on any dataset exposing ``load_mask(i)`` — i.e.
    :class:`lrad.mvtec.MVTecCategory`, which synthesizes an all-zero mask
    for nominal images so they still count as true negatives.
    """
    idx = range(len(dataset)) if indices is None else indices
    return np.stack([dataset.load_mask(i).numpy() for i in idx]).astype(
        np.float32
    )


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """Pooled per-pixel AUROC of ``maps`` against binary ``masks``."""
    if maps.shape != masks.shape:
        raise ValueError(
            f"maps {maps.shape} and masks {masks.shape} must match"
        )
    y = masks.ravel().astype(np.int8)
    s = maps.ravel().astype(np.float64)
    if y.min() == y.max():
        return float("nan")  # single-class ground truth — AUROC undefined
    return float(roc_auc_score(y, s))


def _region_slices(masks: np.ndarray) -> list[np.ndarray]:
    """Flat score-index arrays, one per connected ground-truth component.

    Regions are found per image with 8-connectivity (the MVTec convention),
    so two defects touching only at a corner count as one region — matching
    the reference PRO implementations.
    """
    from scipy.ndimage import label

    structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
    regions: list[np.ndarray] = []
    for i, m in enumerate(masks):
        if m.max() <= 0:
            continue  # nominal image: no region contributes to PRO
        lab, n = label(m > 0.5, structure=structure)
        flat = lab.ravel()
        offset = i * m.size
        for r in range(1, n + 1):
            regions.append(np.flatnonzero(flat == r) + offset)
    return regions


def pro_score(
    maps: np.ndarray,
    masks: np.ndarray,
    *,
    fpr_limit: float = DEFAULT_FPR_LIMIT,
    n_thresholds: int = 300,
) -> dict:
    """Per-region overlap (PRO), integrated over FPR in ``[0, fpr_limit]``.

    Returns ``{"pro": float, "fpr": [...], "pro_curve": [...],
    "n_regions": int}`` — the scalar plus the curve, so callers can plot it
    next to the ROC.

    Implementation note: the naive form thresholds the whole ``(N, H, W)``
    stack once per threshold, which is ~10^9 element visits for one MVTec
    category. Instead each region's scores are **sorted once** and every
    threshold answered with a single ``searchsorted`` — the count of pixels
    at or above ``t`` is the length of the sorted tail. Same numbers, and it
    turns an O(T·N·H·W) sweep into O(N·H·W·log) work done once.
    """
    if maps.shape != masks.shape:
        raise ValueError(
            f"maps {maps.shape} and masks {masks.shape} must match"
        )
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")

    flat_scores = maps.ravel().astype(np.float64)
    flat_masks = masks.ravel() > 0.5

    regions = _region_slices(masks)
    if not regions:
        return {"pro": float("nan"), "fpr": [], "pro_curve": [],
                "n_regions": 0}

    normal = flat_scores[~flat_masks]
    if normal.size == 0:
        return {"pro": float("nan"), "fpr": [], "pro_curve": [],
                "n_regions": len(regions)}

    normal_sorted = np.sort(normal)

    # Thresholds are picked as QUANTILES of the normal-pixel scores, at FPR
    # levels spread evenly over the integration range — not as a linspace
    # over the score range. Anomaly maps are heavily skewed (a few defect
    # pixels sit far above a dense nominal bulk), so a linear threshold grid
    # spends nearly all its points in the empty gap above the bulk, where
    # FPR is already 0, and samples the [0, fpr_limit] window that actually
    # gets integrated with only a handful. Sampling the FPR axis directly
    # puts every threshold where the integral needs one.
    levels = np.linspace(0.0, fpr_limit, n_thresholds)
    quantile_t = np.quantile(normal_sorted, np.clip(1.0 - levels, 0.0, 1.0))
    # Quantile sampling alone collapses when the normal scores are heavily
    # tied (large exactly-equal regions — a saturated background, a map with
    # wide zero areas): every quantile returns the same value and the curve
    # is left with a single usable point. Union with a linear sweep over the
    # score range, which degrades gracefully in exactly that case. Between
    # them one of the two always samples the integration window properly.
    linear_t = np.linspace(flat_scores.min(), flat_scores.max(), n_thresholds)
    # The +inf-side point pins the curve's (FPR=0, PRO=0) end so the integral
    # starts at the origin however the map is distributed.
    thresholds = np.unique(np.concatenate([
        quantile_t, linear_t, [np.nextafter(flat_scores.max(), np.inf)],
    ]))

    # FPR(t) = fraction of NORMAL pixels scored >= t. Recomputed from the
    # sorted array rather than assumed equal to ``levels``: ties in the
    # score map (saturated or quantized regions) make the realized FPR of a
    # quantile differ from its nominal level, and the integral must use the
    # realized one.
    fpr = 1.0 - np.searchsorted(
        normal_sorted, thresholds, side="left",
    ) / normal_sorted.size

    # PRO(t) = mean over regions of the fraction of that region scored >= t.
    # Every region contributes with weight 1, whatever its area — that is
    # the whole point of the metric.
    overlap = np.zeros_like(thresholds)
    for idx in regions:
        rs = np.sort(flat_scores[idx])
        covered = rs.size - np.searchsorted(rs, thresholds, side="left")
        overlap += covered / rs.size
    overlap /= len(regions)

    # Both curves are decreasing in t; flip to ascending FPR for the integral.
    order = np.argsort(fpr)
    fpr_a, pro_a = fpr[order], overlap[order]

    keep = fpr_a <= fpr_limit
    if keep.sum() < 2:
        return {"pro": float("nan"), "fpr": fpr_a.tolist(),
                "pro_curve": pro_a.tolist(), "n_regions": len(regions)}
    x, y = fpr_a[keep], pro_a[keep]
    # Close the interval at exactly fpr_limit by interpolating the first
    # point past it, so the normalization by fpr_limit stays exact instead
    # of quietly integrating over a slightly shorter interval.
    if x[-1] < fpr_limit and keep.sum() < fpr_a.size:
        nxt = np.flatnonzero(~keep)[0]
        y_at = np.interp(fpr_limit, [x[-1], fpr_a[nxt]], [y[-1], pro_a[nxt]])
        x = np.append(x, fpr_limit)
        y = np.append(y, y_at)

    return {
        "pro": float(_trapezoid(y, x) / fpr_limit),
        "fpr": fpr_a.tolist(),
        "pro_curve": pro_a.tolist(),
        "n_regions": len(regions),
    }


def evaluate_pixel_metrics(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loaders: dict,
    device: torch.device,
    *,
    term: str = "bias",
    blocks: Sequence[int] | None = None,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    fpr_limit: float = DEFAULT_FPR_LIMIT,
    keep_maps: bool = False,
) -> dict:
    """Pixel AUROC + PRO over the full test set (nominal AND anomalous).

    Both splits are scored: ``test/good`` images carry an all-zero ground
    truth and contribute true negatives. Dropping them would inflate pixel
    AUROC, since the easiest negatives — whole clean images — would vanish
    from the pool.

    ``keep_maps=True`` additionally returns the per-split anomaly maps and
    masks under ``maps_in`` / ``maps_ood`` / ``masks_ood``, so the caller
    can draw qualitative segmentation panels without recomputing them.
    They are held out of the returned dict by default because a category's
    maps run to tens of MB and would otherwise be carried into every
    ``summary.json``.
    """
    maps, masks = [], []
    for key in ("test_in", "test_ood"):
        ds = loaders.get(f"{key}_ds")
        if ds is None:
            raise KeyError(
                f"loaders['{key}_ds'] is missing — pixel metrics need the "
                "dataset object to read ground-truth masks (lrad.mvtec "
                "provides it; the CelebA loaders do not)"
            )
        maps.append(collect_anomaly_maps(
            models, decoders_list, loaders[key], device,
            term=term, blocks=blocks, smooth_sigma=smooth_sigma,
        ))
        masks.append(load_masks(ds))

    all_maps = np.concatenate(maps, axis=0)
    all_masks = np.concatenate(masks, axis=0)
    logger.info(
        "pixel metrics on %d images (%d nominal + %d anomalous), %dx%d",
        all_maps.shape[0], maps[0].shape[0], maps[1].shape[0],
        all_maps.shape[1], all_maps.shape[2],
    )

    pro = pro_score(all_maps, all_masks, fpr_limit=fpr_limit)
    out = {
        "term": term,
        "smooth_sigma": smooth_sigma,
        "n_images": int(all_maps.shape[0]),
        # Health of the decoders behind these maps — see COLLAPSE_RATIO_WARN.
        # Recorded per run so a suspicious metric can be checked against it
        # after the fact instead of being re-derived.
        "decoder_tracking_ratio": getattr(
            collect_anomaly_maps, "last_tracking_ratio", float("nan"),
        ),
        "pixel_auroc": pixel_auroc(all_maps, all_masks),
        "pro": pro["pro"],
        "pro_fpr_limit": fpr_limit,
        "n_regions": pro["n_regions"],
        "pro_curve": {"fpr": pro["fpr"], "pro": pro["pro_curve"]},
    }
    if keep_maps:
        out["maps_in"], out["maps_ood"] = maps[0], maps[1]
        out["masks_ood"] = masks[1]
    return out
