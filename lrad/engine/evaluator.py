"""Real-IAD evaluation: image- and pixel-level metrics + map collection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..models import LRADModel

logger = logging.getLogger("lrad")


def _unpack(batch) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if len(batch) == 2:
        return batch[0], batch[1], {}
    return batch[0], batch[1], batch[2] if len(batch) > 2 else {}


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_anomaly_scores(
    model: LRADModel,
    loader: DataLoader,
    device: torch.device,
    fusion: str = "mean",
    max_batches: int | None = None,
    keep_masks: bool = False,
) -> dict:
    """Run the model over ``loader`` and stack image- + pixel-level outputs.

    Returns a dict with::

        scores          : (N,)   image-level anomaly scores
        heatmaps        : (N, H, W) fused per-pixel error
        images          : (N, C, H, W) inputs (CPU, float32)
        labels          : (N,)   0/1 ground truth
        per_layer_maps  : list[(N, H, W)] one per decoder
        reconstructions : list[(N, C, H, W)] one per decoder
        masks           : (N, H, W) ground-truth masks (only if keep_masks=True
                          and the loader returned them)
    """
    model.to(device)
    model.eval()
    n_dec = len(model.decoders)

    s_buf: list[np.ndarray] = []
    h_buf: list[np.ndarray] = []
    i_buf: list[np.ndarray] = []
    l_buf: list[np.ndarray] = []
    p_buf: list[list[np.ndarray]] = [[] for _ in range(n_dec)]
    r_buf: list[list[np.ndarray]] = [[] for _ in range(n_dec)]
    m_buf: list[np.ndarray] = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images, labels, extras = _unpack(batch)
        images = images.to(device, non_blocking=True)

        out_fwd = model.forward(images)
        out = model.compute_anomaly_maps(images, fusion=fusion)

        s_buf.append(out["scores"].cpu().numpy())
        h_buf.append(out["fused"].squeeze(1).cpu().numpy())
        i_buf.append(images.cpu().numpy())
        l_buf.append(labels.numpy())

        for k, plm in enumerate(out["per_layer"]):
            p_buf[k].append(plm.squeeze(1).cpu().numpy())
        for k, recon in enumerate(out_fwd["reconstructions"]):
            r_buf[k].append(recon.cpu().numpy())

        if keep_masks and "mask" in extras:
            m_buf.append(extras["mask"].squeeze(1).cpu().numpy())

    out: dict[str, Any] = {
        "scores": np.concatenate(s_buf),
        "heatmaps": np.concatenate(h_buf),
        "images": np.concatenate(i_buf),
        "labels": np.concatenate(l_buf),
        "per_layer_maps": [np.concatenate(p) for p in p_buf],
        "reconstructions": [np.concatenate(r) for r in r_buf],
    }
    if m_buf:
        out["masks"] = np.concatenate(m_buf)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def image_metrics(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """Image-level AUROC / PR-AUC / best-F1 + the corresponding threshold."""
    labels = np.concatenate([
        np.zeros(len(normal_scores), dtype=np.int64),
        np.ones(len(anomaly_scores), dtype=np.int64),
    ])
    scores = np.concatenate([normal_scores, anomaly_scores])

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    pr_auc = float(average_precision_score(labels, scores))
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best = int(np.nanargmax(f1s))
    if best >= len(thresholds):
        best = len(thresholds) - 1
    threshold = float(thresholds[best])
    f1 = float(f1s[best])
    precision = float(precisions[best])
    recall = float(recalls[best])

    pred = (scores >= threshold).astype(np.int64)
    f1_check = float(f1_score(labels, pred))

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1": f1,
        "f1_check": f1_check,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "precisions": precisions,
        "recalls": recalls,
    }


def pixel_metrics(
    heatmaps_normal: np.ndarray | None,
    heatmaps_anomaly: np.ndarray,
    masks_anomaly: np.ndarray,
) -> dict:
    """Pixel-level AUROC / PR-AUC over all anomaly pixels (and optionally
    a sample of normal pixels for negative class)."""
    if heatmaps_anomaly.shape != masks_anomaly.shape:
        raise ValueError(
            f"heatmaps {heatmaps_anomaly.shape} and masks {masks_anomaly.shape}"
            " must align"
        )
    # Flatten across all NG samples
    h_a = heatmaps_anomaly.reshape(-1)
    m_a = masks_anomaly.reshape(-1).astype(np.int64)

    if heatmaps_normal is not None:
        h_n = heatmaps_normal.reshape(-1)
        m_n = np.zeros_like(h_n, dtype=np.int64)
        labels = np.concatenate([m_n, m_a])
        scores = np.concatenate([h_n, h_a])
    else:
        labels = m_a
        scores = h_a

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "pr_auc": float("nan")}

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate_realiad(
    model: LRADModel,
    loaders: dict,
    device: torch.device,
    fusion: str = "mean",
    max_batches: int | None = None,
    keep_masks: bool = True,
) -> dict:
    """Run image- and pixel-level evaluation on the Real-IAD test splits.

    Returns dict with::

        normal   : score-collection bundle on test_normal
        anomaly  : score-collection bundle on test_anomaly
        image    : image-level metrics (image_metrics(...))
        pixel    : pixel-level metrics or None if no masks
    """
    logger.info("evaluating on test_normal...")
    normal = collect_anomaly_scores(
        model, loaders["test_normal"], device,
        fusion=fusion, max_batches=max_batches, keep_masks=False,
    )

    logger.info("evaluating on test_anomaly...")
    anomaly = collect_anomaly_scores(
        model, loaders["test_anomaly"], device,
        fusion=fusion, max_batches=max_batches, keep_masks=keep_masks,
    )

    image = image_metrics(normal["scores"], anomaly["scores"])
    logger.info(
        f"  image AUROC={image['auroc']:.4f}  PR-AUC={image['pr_auc']:.4f}  "
        f"F1={image['f1']:.4f}  τ={image['threshold']:.4f}"
    )

    pixel = None
    if keep_masks and "masks" in anomaly:
        pixel = pixel_metrics(
            heatmaps_normal=normal["heatmaps"],
            heatmaps_anomaly=anomaly["heatmaps"],
            masks_anomaly=anomaly["masks"],
        )
        logger.info(
            f"  pixel AUROC={pixel['auroc']:.4f}  PR-AUC={pixel['pr_auc']:.4f}"
        )

    return {"normal": normal, "anomaly": anomaly, "image": image, "pixel": pixel}
