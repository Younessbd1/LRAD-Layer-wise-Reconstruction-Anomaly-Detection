"""Training and evaluation routines for the CelebA one-class pipeline.

Phase 1 used to be the rotation pretext of Gidaris et al. (2018) — a 4-way
classification of randomly rotated training images. That pretext doesn't
need to attend to specific facial regions to predict rotation, so the
features it learns are weak at localizing local anomalies (e.g. sunglasses).

This module replaces it with a **multi-label attribute pretext**: the
classifier predicts ``len(pretext_attrs)`` independent binary attributes
(e.g. Male / Young / Smiling) jointly via sigmoid + BCE. All the chosen
attributes are present (with both polarities) inside the eyeglasses-free
training pool, so the encoder is forced to attend to the actual face
(eyes, mouth, jaw, hair). The features it learns localize sunglasses much
more sharply at test time.

Pipeline:
    Phase 1 — train_classifier_attributes : multi-label BCE pretext, with
                                            val tracking + early stopping
    Phase 2 — train_decoders              : MSE reconstruction with val
                                            tracking
    Eval   — evaluate_celeba              : image-level AUROC, PR-AUC, F1
                                            (no pixel masks on CelebA)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..models.classifier import CNNClassifier
from ..models.lrad_model import LRADModel

logger = logging.getLogger("lrad")


# ---------------------------------------------------------------------------
# Phase 1 — multi-label attribute pretext
# ---------------------------------------------------------------------------

def _classifier_epoch(
    classifier: CNNClassifier,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
) -> tuple[float, np.ndarray, float]:
    """Run one multi-label BCE epoch.

    Returns:
        avg_loss : float — mean BCE per sample
        per_attr_acc : (P,) array — accuracy of the sigmoid > 0.5 prediction
                       for each of the ``P`` pretext attributes
        mean_acc : float — mean of ``per_attr_acc``

    If ``optimizer`` is None, runs in eval mode (no grad updates).
    """
    train_mode = optimizer is not None
    classifier.train(train_mode)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0
    correct_per_attr: Optional[np.ndarray] = None

    for batch in loader:
        # Loaders yield (images, anomaly_label, pretext_labels). Phase 1
        # uses only images + pretext_labels.
        images, _anom, attrs = batch
        images = images.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True).float()

        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits, _ = classifier(images)
            loss = criterion(logits, attrs)
            if train_mode:
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total += bs

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == attrs).float().sum(dim=0).cpu().numpy()
        correct_per_attr = (correct if correct_per_attr is None
                            else correct_per_attr + correct)

    n = max(total, 1)
    per_attr_acc = (correct_per_attr / n) if correct_per_attr is not None \
        else np.zeros(0, dtype=np.float32)
    mean_acc = float(per_attr_acc.mean()) if per_attr_acc.size else 0.0
    return total_loss / n, per_attr_acc.astype(np.float32), mean_acc


def train_classifier_attributes(
    classifier: CNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pretext_attrs: list[str],
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    log_every: int = 5,
    early_stop_patience: Optional[int] = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 5,
) -> dict:
    """Train ``classifier`` on a multi-label attribute pretext.

    Tracks train + val loss/accuracy each epoch (mean and per-attribute) and
    supports early stopping on val loss plateau. Freezes the classifier in
    place once training is done (so callers can hand it straight to
    ``LRADModel``).
    """
    classifier.to(device)
    # Defensive: ensure parameters are trainable. ``LRADModel.__init__``
    # auto-freezes its classifier; if a future caller built the LRADModel
    # before Phase 1 ran, every parameter would silently have
    # ``requires_grad=False`` and ``loss.backward()`` would fail.
    for p in classifier.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(
        (p for p in classifier.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )

    history: dict = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "train_acc_per_attr": [], "val_acc_per_attr": [],
        "pretext_attrs": list(pretext_attrs),
    }

    best_val = float("inf")
    bad_epochs = 0
    best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}

    for epoch in range(epochs):
        train_loss, train_per_attr, train_acc = _classifier_epoch(
            classifier, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss, val_per_attr, val_acc = _classifier_epoch(
                classifier, val_loader, optimizer=None, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_acc_per_attr"].append(train_per_attr.tolist())
        history["val_acc_per_attr"].append(val_per_attr.tolist())

        if (epoch + 1) % log_every == 0 or epoch == 0:
            per_attr_str = ", ".join(
                f"{name}={a:.3f}" for name, a in zip(pretext_attrs, val_per_attr)
            )
            logger.info(
                f"  [Phase 1] epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                f"[per-attr val: {per_attr_str}]"
            )

        improved = val_loss < best_val - early_stop_min_delta
        if improved:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
        else:
            bad_epochs += 1
            if (early_stop_patience is not None
                and epoch + 1 >= early_stop_min_epochs
                and bad_epochs >= early_stop_patience):
                logger.info(
                    f"  [Phase 1] early stop at epoch {epoch+1} "
                    f"(no val_loss improvement for {bad_epochs} epochs)"
                )
                break

    classifier.load_state_dict(best_state)
    classifier.freeze()
    logger.info(f"  [Phase 1] best val_loss={best_val:.4f} — classifier frozen")
    return history


# ---------------------------------------------------------------------------
# Phase 2 — decoder training with val tracking
# ---------------------------------------------------------------------------

def _decoder_epoch(
    model: LRADModel,
    loader: DataLoader,
    optimizers: Optional[list[optim.Optimizer]],
    device: torch.device,
) -> list[float]:
    """Run one decoder-training epoch (or eval if optimizers is None).
    Returns a list of average MSE per decoder."""
    train_mode = optimizers is not None
    for dec in model.decoders:
        dec.train(train_mode)
    criterion = nn.MSELoss()
    n_dec = len(model.decoders)

    sums = [0.0] * n_dec
    total = 0
    for batch in loader:
        images = batch[0]
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            _, all_acts = model.classifier(images)

        bs = images.size(0)
        total += bs
        for i, dec in enumerate(model.decoders):
            act = all_acts[model.decoder_layers[i]].detach()
            if train_mode:
                optimizers[i].zero_grad()
            with torch.set_grad_enabled(train_mode):
                recon = dec(act)
                loss = criterion(recon, images)
                if train_mode:
                    loss.backward()
                    optimizers[i].step()
            sums[i] += loss.item() * bs

    return [s / max(total, 1) for s in sums]


def train_decoders(
    model: LRADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    log_every: int = 5,
) -> dict:
    """Train every decoder independently to reconstruct the input image
    from its assigned classifier activation. MSE loss, Adam optimizer.

    Tracks train + val MSE per decoder per epoch.
    """
    model.to(device)
    optimizers = [optim.Adam(d.parameters(), lr=lr, weight_decay=weight_decay)
                  for d in model.decoders]
    n_dec = len(model.decoders)
    history = {
        "train_loss": [[] for _ in range(n_dec)],
        "val_loss": [[] for _ in range(n_dec)],
    }

    for epoch in range(epochs):
        train_losses = _decoder_epoch(model, train_loader, optimizers, device)
        with torch.no_grad():
            val_losses = _decoder_epoch(model, val_loader, None, device)
        for i in range(n_dec):
            history["train_loss"][i].append(train_losses[i])
            history["val_loss"][i].append(val_losses[i])

        if (epoch + 1) % log_every == 0 or epoch == 0:
            tr = ", ".join(f"D{i}={train_losses[i]:.4f}" for i in range(n_dec))
            va = ", ".join(f"D{i}={val_losses[i]:.4f}" for i in range(n_dec))
            logger.info(
                f"  [Phase 2] epoch {epoch+1}/{epochs} train[{tr}] val[{va}]"
            )

    return history


# ---------------------------------------------------------------------------
# Evaluation — image-level only (CelebA has no pixel masks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_scores(
    model: LRADModel,
    loader: DataLoader,
    device: torch.device,
    fusion: str,
    max_batches: Optional[int],
    n_viz_samples: int = 8,
    smooth_sigma: float = 1.5,
    normalize_per_layer: bool = True,
    score_top_k_pct: float = 1.0,
) -> dict:
    """Run inference, return image-level scores, fused heatmaps, and originals.

    The first ``n_viz_samples`` items are also retained with their
    per-decoder reconstructions and per-layer error heatmaps, so the
    plotting layer can show side-by-side comparisons across decoder
    blocks without having to re-run the model.
    """
    model.eval()
    model.to(device)

    scores_list = []
    heatmaps_list = []
    images_list = []

    viz_images: list[np.ndarray] = []
    viz_fused: list[np.ndarray] = []
    viz_scores: list[float] = []
    viz_recons: Optional[list[list[np.ndarray]]] = None  # per-decoder × per-sample
    viz_per_layer: Optional[list[list[np.ndarray]]] = None
    viz_collected = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = batch[0]
        images = images.to(device, non_blocking=True)
        result = model.compute_anomaly_maps(
            images,
            fusion=fusion,
            smooth_sigma=smooth_sigma,
            normalize_per_layer=normalize_per_layer,
            score_top_k_pct=score_top_k_pct,
        )
        scores_list.append(result["scores"].cpu().numpy())
        heatmaps_list.append(result["fused"].squeeze(1).cpu().numpy())
        images_list.append(images.cpu().numpy())

        if viz_collected < n_viz_samples:
            need = n_viz_samples - viz_collected
            take = min(need, images.size(0))
            full = model.forward(images[:take])
            recons = [r.cpu().numpy() for r in full["reconstructions"]]
            per_layer = [m[:take].cpu().numpy() for m in result["per_layer"]]
            n_dec = len(recons)
            if viz_recons is None:
                viz_recons = [[] for _ in range(n_dec)]
                viz_per_layer = [[] for _ in range(n_dec)]
            viz_images.append(images[:take].cpu().numpy())
            viz_fused.append(result["fused"][:take].squeeze(1).cpu().numpy())
            viz_scores.extend(result["scores"][:take].cpu().numpy().tolist())
            for d in range(n_dec):
                viz_recons[d].append(recons[d])
                viz_per_layer[d].append(per_layer[d])
            viz_collected += take

    out = {
        "scores": np.concatenate(scores_list) if scores_list else np.zeros(0),
        "heatmaps": np.concatenate(heatmaps_list) if heatmaps_list else np.zeros((0, 1, 1)),
        "images": np.concatenate(images_list) if images_list else np.zeros((0, 3, 1, 1)),
    }

    if viz_collected > 0 and viz_recons is not None:
        out["viz"] = {
            "images": np.concatenate(viz_images, axis=0),
            "fused": np.concatenate(viz_fused, axis=0),
            "scores": np.asarray(viz_scores, dtype=np.float32),
            "reconstructions": [np.concatenate(rs, axis=0) for rs in viz_recons],
            "per_layer": [np.concatenate(ms, axis=0) for ms in viz_per_layer],
        }
    else:
        out["viz"] = None
    return out


def _image_level_metrics(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """Compute image-level AUROC, PR-AUC, best-F1 threshold, precision,
    recall, and the underlying ROC/PR curves."""
    y = np.concatenate([np.zeros(len(normal_scores)),
                        np.ones(len(anomaly_scores))])
    s = np.concatenate([normal_scores, anomaly_scores])

    auroc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    pr_auc = float(average_precision_score(y, s))
    precisions, recalls, thresholds = precision_recall_curve(y, s)

    eps = 1e-12
    f1s = 2 * precisions * recalls / (precisions + recalls + eps)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else float("nan")
    best_f1 = float(f1s[best_idx]) if len(thresholds) else float("nan")
    preds = (s >= best_threshold).astype(int)
    best_f1_check = float(f1_score(y, preds)) if len(thresholds) else float("nan")

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "f1": best_f1_check,
        "best_f1_at_threshold": best_f1,
        "best_threshold": best_threshold,
        "fpr": fpr,
        "tpr": tpr,
        "precisions": precisions,
        "recalls": recalls,
    }


def evaluate_celeba(
    model: LRADModel,
    loaders: dict,
    device: torch.device,
    fusion: str = "mean",
    max_batches: Optional[int] = None,
    smooth_sigma: float = 1.5,
    normalize_per_layer: bool = True,
    score_top_k_pct: float = 1.0,
) -> dict:
    """Image-level evaluation on the test_normal and test_anomaly splits
    of a CelebA loader bundle. Returns scores, heatmaps, originals, and
    the headline metrics dict.

    CelebA has no pixel masks, so the pixel-level slot is left empty —
    callers can check ``loaders["has_masks"]`` if they want to assert this.
    """
    if loaders.get("has_masks"):
        logger.warning(
            "evaluate_celeba: loaders['has_masks'] is True, but this routine "
            "computes image-level metrics only. Pixel-AUROC will be skipped."
        )

    score_kwargs = dict(
        smooth_sigma=smooth_sigma,
        normalize_per_layer=normalize_per_layer,
        score_top_k_pct=score_top_k_pct,
    )
    logger.info(
        f"Scoring config: smooth_sigma={smooth_sigma}  "
        f"normalize_per_layer={normalize_per_layer}  "
        f"top_k_pct={score_top_k_pct}"
    )

    logger.info("Collecting scores on test_normal...")
    normal = _collect_scores(model, loaders["test_normal"], device,
                             fusion, max_batches, **score_kwargs)
    logger.info("Collecting scores on test_anomaly...")
    anomaly = _collect_scores(model, loaders["test_anomaly"], device,
                              fusion, max_batches, **score_kwargs)

    image_metrics = _image_level_metrics(normal["scores"], anomaly["scores"])
    logger.info(
        f"  AUROC={image_metrics['auroc']:.4f}  "
        f"PR-AUC={image_metrics['pr_auc']:.4f}  "
        f"F1={image_metrics['f1']:.4f}  "
        f"thr={image_metrics['best_threshold']:.4f}"
    )

    return {
        "normal": normal,
        "anomaly": anomaly,
        "image": image_metrics,
        "pixel": None,
    }
