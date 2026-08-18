"""Rank fusion of complementary OOD signals — the headline detector.

No single signal on this task clears AUROC 0.80 alone: the localized
feature error (``lrad.feature_error``) sees *where* the reconstruction
fails, the ensemble's epistemic uncertainty sees the members *disagree*,
and the gender-head energy sees the classifier *hesitate*. They err on
different images, so fusing them helps. The default recipe, validated on
the 20260707 ensemble run (subsampled test splits):

    locfre_b1 + locfre_b3 + epistemic(combined) + energy(gender)
    →  AUROC ≈ 0.81   (pixel p95 bias baseline: 0.64)

**Rank fusion** replaces each signal by its normalized rank within the
pooled evaluation scores, then averages the ranks. It is label-free (a
monotone per-signal transform — individual AUROCs are unchanged), needs
no calibration data, and is immune to the signals' wildly different
scales. A deployed detector would freeze the transform by ranking against
a reference-score bank instead of the evaluation pool.

Everything runs in ONE pass over each loader: per member the input is
encoded once (trunk + heads), the ensemble-mean deepest reconstruction is
decoded and re-encoded, and every signal is read off those activations.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ensemble import _uncertainty_scores
from .evaluate import _auroc_entry
from .feature_error import (
    DEFAULT_FEATURE_BLOCKS,
    fit_feature_error_stats,
    locfre_scores_from_maps,
)
from .localized import STD_FLOOR
from .model import FacialCNN

logger = logging.getLogger("celeba_ood")


def rank_fusion(
    signals: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted average of normalized ranks. ``signals`` are same-length
    per-image score arrays (higher = more OOD); returns one fused array."""
    signals = [np.asarray(s, dtype=np.float64).ravel() for s in signals]
    if not signals:
        raise ValueError("need at least one signal to fuse")
    n = signals[0].shape[0]
    if any(s.shape[0] != n for s in signals):
        raise ValueError("all signals must score the same images")
    w = np.ones(len(signals)) if weights is None else np.asarray(
        list(weights), dtype=np.float64,
    )
    if w.shape[0] != len(signals):
        raise ValueError(
            f"{w.shape[0]} weights for {len(signals)} signals"
        )
    fused = np.zeros(n)
    for s, wk in zip(signals, w):
        fused += wk * (s.argsort().argsort() / max(n - 1, 1))
    return fused / w.sum()


@torch.no_grad()
def collect_fusion_signals(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    loader,
    device: torch.device,
    ref: dict,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """Every fusion signal over a loader, one pass. Returns per-image
    arrays keyed ``locfre_b{j}``, ``unc_epistemic_combined``,
    ``ens_energy_gender`` (all: higher = more OOD)."""
    M = len(models)
    if M == 0:
        raise ValueError("need at least one model in the ensemble")
    kd = len(decoders_list[0]) - 1
    blocks = list(blocks)

    locfre_chunks: dict[int, list[np.ndarray]] = {j: [] for j in blocks}
    g_chunks: list[np.ndarray] = []
    a_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        img = batch[0].to(device, non_blocking=True)

        acts_x: list[dict[int, torch.Tensor]] = []
        g_probs, a_probs, g_logits = [], [], []
        recon_sum: torch.Tensor | None = None
        for m in range(M):
            models[m].eval()
            decoders_list[m].eval()
            out, acts = models[m].forward_with_activations(img)
            acts_x.append(
                {j: F.normalize(acts[j], dim=1) for j in blocks}
            )
            recon = decoders_list[m][kd](acts[kd])
            recon_sum = recon if recon_sum is None else recon_sum + recon
            g_probs.append(F.softmax(out["gender_logits"], dim=-1).cpu())
            a_probs.append(torch.sigmoid(out["attr_logits"]).cpu())
            g_logits.append(out["gender_logits"].cpu())
        mean_recon = recon_sum / M

        err: dict[int, torch.Tensor] = {}
        for m in range(M):
            _, acts_r = models[m].forward_features(mean_recon)
            for j in blocks:
                d = acts_x[m][j] - F.normalize(acts_r[j], dim=1)
                e = (d ** 2).sum(dim=1)
                err[j] = e if j not in err else err[j] + e
        maps = {j: err[j] / M for j in blocks}
        for j, sc in locfre_scores_from_maps(maps, ref).items():
            locfre_chunks[j].append(sc.cpu().numpy())

        g = torch.stack(g_probs).numpy()   # (M, B, 2)
        a = torch.stack(a_probs).numpy()   # (M, B, A)
        lg = torch.stack(g_logits).numpy()
        g_chunks.append(g)
        a_chunks.append(a)
        # Energy of the gender head, averaged over members: OOD inputs get
        # smaller logits, hence higher (less negative) energy.
        energy_chunks.append(
            -np.logaddexp(lg[..., 0], lg[..., 1]).mean(axis=0)
        )

    if not g_chunks:
        raise ValueError("loader yielded no images")
    g = np.concatenate(g_chunks, axis=1)
    unc = _uncertainty_scores(g, np.concatenate(a_chunks, axis=1))
    out: dict[str, np.ndarray] = {
        f"locfre_b{j}": np.concatenate(locfre_chunks[j]) for j in blocks
    }
    out["unc_epistemic_combined"] = unc["combined"]["epistemic"]
    out["ens_energy_gender"] = np.concatenate(energy_chunks)
    return out


@torch.no_grad()
def evaluate_fused_ood(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    ref_loader,
    test_in_loader,
    test_ood_loader,
    device: torch.device,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
    std_floor: float = STD_FLOOR,
    weights: Sequence[float] | None = None,
    max_ref_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict:
    """Full fused evaluation: fit locfre reference stats on ``ref_loader``
    (in-dist only, disjoint from both test splits), collect every signal
    on both test splits, report per-signal AUROC and the fused AUROC.

    ``weights`` optionally weight the fusion in signal order
    (``locfre_b{blocks...}``, epistemic, energy); default uniform.
    """
    ref = fit_feature_error_stats(
        models, decoders_list, ref_loader, device, blocks,
        max_batches=max_ref_batches, std_floor=std_floor,
    )
    sig_in = collect_fusion_signals(
        models, decoders_list, test_in_loader, device, ref, blocks,
        max_eval_batches,
    )
    sig_ood = collect_fusion_signals(
        models, decoders_list, test_ood_loader, device, ref, blocks,
        max_eval_batches,
    )

    # The fused score uses only the validated recipe (locfre blocks +
    # epistemic + energy); the extra head scores are reported individually.
    names = [f"locfre_b{j}" for j in blocks] + [
        "unc_epistemic_combined", "ens_energy_gender",
    ]
    n_in = sig_in[names[0]].shape[0]
    fused = rank_fusion(
        [np.concatenate([sig_in[k], sig_ood[k]]) for k in names], weights,
    )
    auroc = {k: _auroc_entry(sig_in[k], sig_ood[k]) for k in sig_in}
    auroc["fused"] = _auroc_entry(fused[:n_in], fused[n_in:])

    return {
        "signals": names,
        "blocks": list(blocks),
        "weights": (list(weights) if weights is not None
                    else [1.0] * len(names)),
        "std_floor": std_floor,
        "n_ref_images": ref["n_images"],
        "n_models": len(models),
        "signals_in": sig_in,
        "signals_ood": sig_ood,
        "fused_in": fused[:n_in],
        "fused_ood": fused[n_in:],
        "auroc": auroc,
    }


# ---------------------------------------------------------------------------
# Supervised fusion — logistic calibration on labelled calibration splits
# ---------------------------------------------------------------------------
#
# Rank fusion weighs every signal equally; a logistic regression learns the
# weights (and sign corrections) instead, at the price of needing labelled
# calibration data. The protocol keeps the evaluation honest:
#   * negatives come from the TRAIN split (never test_in),
#   * positives are a dedicated OOD calibration half, split from the OOD
#     pool deterministically and DISJOINT from the evaluated OOD images.
# On the 20260707 run (subsampled) this reached AUROC ≈ 0.83 vs 0.80 for
# the unsupervised rank fusion.

def fit_signal_fusion(
    sig_in: dict[str, np.ndarray],
    sig_ood: dict[str, np.ndarray],
    signals: Sequence[str] | None = None,
) -> dict:
    """Fit a logistic fusion of the signals on labelled calibration data.

    ``sig_in`` / ``sig_ood`` map signal names to per-image scores of the
    calibration negatives / positives. Features are z-normalized with the
    calibration stats. Returns a JSON-serializable dict (signal names,
    normalization, coefficients) for :func:`apply_signal_fusion`.
    """
    from sklearn.linear_model import LogisticRegression

    names = list(signals) if signals is not None else sorted(sig_in)
    missing = [k for k in names if k not in sig_in or k not in sig_ood]
    if missing:
        raise ValueError(f"signals missing from calibration data: {missing}")
    Xi = np.stack([np.asarray(sig_in[k], dtype=np.float64) for k in names], 1)
    Xo = np.stack([np.asarray(sig_ood[k], dtype=np.float64) for k in names], 1)
    if len(Xi) == 0 or len(Xo) == 0:
        raise ValueError("calibration needs both in-dist and OOD examples")
    X = np.concatenate([Xi, Xo])
    y = np.concatenate([np.zeros(len(Xi)), np.ones(len(Xo))])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=1000).fit((X - mu) / sd, y)
    logger.info(
        "supervised fusion fitted on %d in / %d ood: %s",
        len(Xi), len(Xo),
        {k: round(c, 3) for k, c in zip(names, clf.coef_[0])},
    )
    return {
        "signals": names,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def apply_signal_fusion(
    calib: dict, sig: dict[str, np.ndarray],
) -> np.ndarray:
    """Fused score (logit of P(OOD)) for per-image signals ``sig`` using a
    calibration from :func:`fit_signal_fusion`. Higher = more OOD."""
    names = calib["signals"]
    X = np.stack([np.asarray(sig[k], dtype=np.float64) for k in names], 1)
    Xn = (X - np.asarray(calib["mu"])) / np.asarray(calib["sd"])
    return Xn @ np.asarray(calib["coef"]) + calib["intercept"]


@torch.no_grad()
def evaluate_supervised_fusion(
    models: Sequence[FacialCNN],
    decoders_list: Sequence[nn.ModuleList],
    ref_loader,
    cal_in_loader,
    cal_ood_loader,
    test_in_loader,
    test_ood_loader,
    device: torch.device,
    blocks: Sequence[int] = DEFAULT_FEATURE_BLOCKS,
    std_floor: float = STD_FLOOR,
    max_ref_batches: int | None = None,
    max_cal_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict:
    """Supervised fused evaluation. The caller guarantees the splits:
    ``ref_loader`` and ``cal_in_loader`` are in-dist and disjoint from
    ``test_in_loader``; ``cal_ood_loader`` is disjoint from
    ``test_ood_loader``. Reports per-signal AUROC, the unsupervised rank
    fusion, and the supervised logistic fusion — all on the test loaders
    only.
    """
    ref = fit_feature_error_stats(
        models, decoders_list, ref_loader, device, blocks,
        max_batches=max_ref_batches, std_floor=std_floor,
    )

    def _collect(loader, cap):
        return collect_fusion_signals(
            models, decoders_list, loader, device, ref, blocks, cap,
        )

    calib = fit_signal_fusion(
        _collect(cal_in_loader, max_cal_batches),
        _collect(cal_ood_loader, max_cal_batches),
    )
    sig_in = _collect(test_in_loader, max_eval_batches)
    sig_ood = _collect(test_ood_loader, max_eval_batches)

    rank_names = [f"locfre_b{j}" for j in blocks] + [
        "unc_epistemic_combined", "ens_energy_gender",
    ]
    n_in = sig_in[rank_names[0]].shape[0]
    fused_rank = rank_fusion(
        [np.concatenate([sig_in[k], sig_ood[k]]) for k in rank_names],
    )
    sup_in = apply_signal_fusion(calib, sig_in)
    sup_ood = apply_signal_fusion(calib, sig_ood)

    auroc = {k: _auroc_entry(sig_in[k], sig_ood[k]) for k in sig_in}
    auroc["fused_rank"] = _auroc_entry(
        fused_rank[:n_in], fused_rank[n_in:],
    )
    auroc["fused_supervised"] = _auroc_entry(sup_in, sup_ood)

    return {
        "signals": sorted(sig_in),
        "rank_signals": rank_names,
        "blocks": list(blocks),
        "std_floor": std_floor,
        "n_ref_images": ref["n_images"],
        "n_models": len(models),
        "calibration": calib,
        "signals_in": sig_in,
        "signals_ood": sig_ood,
        "fused_rank_in": fused_rank[:n_in],
        "fused_rank_ood": fused_rank[n_in:],
        "fused_supervised_in": sup_in,
        "fused_supervised_ood": sup_ood,
        "auroc": auroc,
    }
