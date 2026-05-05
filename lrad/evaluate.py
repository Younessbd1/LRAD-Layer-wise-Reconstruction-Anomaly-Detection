"""Evaluation: per-attribute accuracy on in-dist + OOD AUROC.

Two flavors of OOD score, computed from the trained classifier's outputs
on each test image:

  * ``msp``     — 1 - max softmax probability of the gender head.
                 Confident predictions (close to 0/1) get a low OOD score;
                 confused predictions (close to 0.5) get a high OOD score.

  * ``entropy`` — combined predictive entropy across both heads:
                 H(gender_softmax) + mean over attrs of Bernoulli entropy
                 H(sigmoid(z_i)). High entropy means the model is uncertain
                 about *every* facial trait — exactly the signal we want
                 when sunglasses occlude the face.

AUROC is computed by combining in-distribution (label=0) and OOD
(label=1) scores into a single (score, label) array and feeding it to
``sklearn.metrics.roc_auc_score``.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .model import FacialCNN

logger = logging.getLogger("celeba_ood")

_EPS = 1e-12


def _bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy in nats. Stable for p ∈ [0, 1]."""
    p = p.clamp(_EPS, 1.0 - _EPS)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy over the last dim, in nats. Shape: (B,)."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


@torch.no_grad()
def collect_predictions(
    model: FacialCNN,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run the model over a loader and return numpy arrays of predictions
    along with several OOD-score variants."""
    model.eval()
    gender_logits_chunks = []
    attr_logits_chunks = []
    gender_targets_chunks = []
    attr_targets_chunks = []
    is_ood_chunks = []

    for img, gender, attrs, is_ood in loader:
        img = img.to(device, non_blocking=True)
        out = model(img)
        gender_logits_chunks.append(out["gender_logits"].cpu())
        attr_logits_chunks.append(out["attr_logits"].cpu())
        gender_targets_chunks.append(gender.cpu())
        attr_targets_chunks.append(attrs.cpu())
        is_ood_chunks.append(torch.as_tensor(is_ood).cpu())

    gender_logits = torch.cat(gender_logits_chunks, dim=0)
    attr_logits = torch.cat(attr_logits_chunks, dim=0)
    gender_targets = torch.cat(gender_targets_chunks, dim=0)
    attr_targets = torch.cat(attr_targets_chunks, dim=0)
    is_ood = torch.cat(is_ood_chunks, dim=0)

    gender_probs = F.softmax(gender_logits, dim=-1)
    attr_probs = torch.sigmoid(attr_logits)

    msp = gender_probs.max(dim=-1).values
    score_msp = (1.0 - msp).numpy()

    h_gender = _softmax_entropy(gender_logits)
    h_attrs = _bernoulli_entropy(attr_probs).mean(dim=-1)
    score_entropy_combined = (h_gender + h_attrs).numpy()
    score_entropy_gender = h_gender.numpy()
    score_entropy_attrs = h_attrs.numpy()

    return {
        "gender_logits": gender_logits.numpy(),
        "attr_logits": attr_logits.numpy(),
        "gender_probs": gender_probs.numpy(),
        "attr_probs": attr_probs.numpy(),
        "gender_targets": gender_targets.numpy(),
        "attr_targets": attr_targets.numpy(),
        "is_ood": is_ood.numpy().astype(np.int64),
        "score_msp": score_msp,
        "score_entropy_gender": score_entropy_gender,
        "score_entropy_attrs": score_entropy_attrs,
        "score_entropy_combined": score_entropy_combined,
    }


def per_attribute_accuracy(
    preds: dict,
    attr_names: Iterable[str],
) -> dict:
    """Per-attribute accuracy on the in-distribution subset only."""
    in_mask = preds["is_ood"] == 0
    if not in_mask.any():
        return {"gender": float("nan"), "attrs": {}}

    gender_pred = preds["gender_probs"][in_mask].argmax(axis=1)
    gender_target = preds["gender_targets"][in_mask]
    gender_acc = float((gender_pred == gender_target).mean())

    attr_pred = (preds["attr_probs"][in_mask] >= 0.5).astype(np.float32)
    attr_target = preds["attr_targets"][in_mask].astype(np.float32)
    attr_acc = (attr_pred == attr_target).mean(axis=0).tolist()
    return {
        "gender": gender_acc,
        "attrs": dict(zip(attr_names, [float(a) for a in attr_acc])),
    }


def ood_auroc(preds_in: dict, preds_ood: dict) -> dict:
    """AUROC for several OOD-score definitions, using preds_in as
    label=0 and preds_ood as label=1."""
    score_keys = (
        "score_msp",
        "score_entropy_gender",
        "score_entropy_attrs",
        "score_entropy_combined",
    )
    n_in = preds_in["is_ood"].shape[0]
    n_ood = preds_ood["is_ood"].shape[0]
    labels = np.concatenate([np.zeros(n_in), np.ones(n_ood)])
    out: dict = {"n_in": int(n_in), "n_ood": int(n_ood)}
    for k in score_keys:
        scores = np.concatenate([preds_in[k], preds_ood[k]])
        if np.isfinite(scores).all() and len(np.unique(labels)) == 2:
            auroc = float(roc_auc_score(labels, scores))
            fpr, tpr, _ = roc_curve(labels, scores)
            out[k] = {
                "auroc": auroc,
                "in_mean": float(preds_in[k].mean()),
                "ood_mean": float(preds_ood[k].mean()),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }
        else:
            out[k] = {"auroc": float("nan")}
    return out


def evaluate(
    model: FacialCNN,
    loaders: dict,
    device: torch.device,
) -> dict:
    """Run the full evaluation: per-attribute accuracy on the held-out
    in-distribution test split and OOD AUROC against the Eyeglasses split."""
    logger.info("Collecting predictions on test_in...")
    preds_in = collect_predictions(model, loaders["test_in"], device)
    logger.info(
        f"  test_in size = {preds_in['is_ood'].shape[0]}, "
        f"OOD score (combined entropy) mean = "
        f"{preds_in['score_entropy_combined'].mean():.4f}"
    )

    logger.info("Collecting predictions on test_ood (Eyeglasses)...")
    preds_ood = collect_predictions(model, loaders["test_ood"], device)
    logger.info(
        f"  test_ood size = {preds_ood['is_ood'].shape[0]}, "
        f"OOD score (combined entropy) mean = "
        f"{preds_ood['score_entropy_combined'].mean():.4f}"
    )

    accuracy = per_attribute_accuracy(preds_in, loaders["attr_targets"])
    auroc = ood_auroc(preds_in, preds_ood)

    logger.info("Per-attribute accuracy (in-distribution test set):")
    logger.info(f"  Male/Female : {accuracy['gender']*100:.2f}%")
    for name, acc in accuracy["attrs"].items():
        logger.info(f"  {name:<22}: {acc*100:.2f}%")

    logger.info("OOD AUROC (in-dist vs Eyeglasses):")
    for k in (
        "score_msp",
        "score_entropy_gender",
        "score_entropy_attrs",
        "score_entropy_combined",
    ):
        v = auroc[k]
        if "auroc" in v and v["auroc"] == v["auroc"]:  # not NaN
            logger.info(
                f"  {k:<24} AUROC = {v['auroc']:.4f}  "
                f"(in_mean={v['in_mean']:.3f}, ood_mean={v['ood_mean']:.3f})"
            )

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "preds_in": preds_in,
        "preds_ood": preds_ood,
    }
