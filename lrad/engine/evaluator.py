"""Evaluation: anomaly scoring, AUROC computation, and summary statistics."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Optional
import logging

from ..models.lrad_model import LRADModel

logger = logging.getLogger("lrad")


@torch.no_grad()
def collect_anomaly_scores(
    model: LRADModel,
    dataloader: DataLoader,
    device: torch.device,
    fusion: str = "mean",
    max_batches: Optional[int] = None,
) -> dict:
    """Run model on a dataloader and collect anomaly scores + heatmaps.

    Args:
        model: The LRAD model.
        dataloader: Any dataloader (normal or anomaly).
        device: Compute device.
        fusion: Heatmap fusion strategy.
        max_batches: Limit number of batches (for quick evaluation).

    Returns:
        dict with:
            'scores': np.array of image-level anomaly scores
            'heatmaps': np.array (N, H, W) of fused anomaly maps
            'images': np.array (N, C, H, W) of input images
            'per_layer_maps': list of np.arrays per decoder level
    """
    model.eval()
    model.to(device)

    n_decoders = len(model.decoders)
    all_scores = []
    all_heatmaps = []
    all_images = []
    all_per_layer = [[] for _ in range(n_decoders)]
    all_recons = [[] for _ in range(n_decoders)]

    for batch_idx, (images, _) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        fwd = model.forward(images)
        result = model.compute_anomaly_maps(images, fusion=fusion)

        all_scores.append(result["scores"].cpu().numpy())
        all_heatmaps.append(result["fused"].squeeze(1).cpu().numpy())
        all_images.append(images.cpu().numpy())

        for i, plm in enumerate(result["per_layer"]):
            all_per_layer[i].append(plm.squeeze(1).cpu().numpy())

        for i, recon in enumerate(fwd["reconstructions"]):
            all_recons[i].append(recon.cpu().numpy())

    return {
        "scores": np.concatenate(all_scores),
        "heatmaps": np.concatenate(all_heatmaps),
        "images": np.concatenate(all_images),
        "per_layer_maps": [np.concatenate(p) for p in all_per_layer],
        "reconstructions": [np.concatenate(r) for r in all_recons],
    }


def compute_auroc(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """Compute image-level AUROC between normal and anomaly score distributions.

    Returns:
        dict with 'auroc', 'fpr', 'tpr', 'thresholds'.
    """
    labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores)),
    ])
    scores = np.concatenate([normal_scores, anomaly_scores])

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def evaluate_full(
    model: LRADModel,
    loaders: dict,
    device: torch.device,
    fusion: str = "mean",
    max_batches: Optional[int] = None,
) -> dict:
    """Run full evaluation across all test splits.

    Args:
        model: Trained LRAD model.
        loaders: Dict from get_mnist_loaders or get_cifar10_loaders.
        device: Compute device.
        fusion: Heatmap fusion strategy.

    Returns:
        dict with per-split results and AUROC scores.
    """
    logger.info("Evaluating on normal test set...")
    normal_result = collect_anomaly_scores(
        model, loaders["test_normal"], device, fusion, max_batches
    )

    results = {"normal": normal_result, "aurocs": {}}

    # Evaluate anomaly splits
    anomaly_keys = [k for k in loaders if k.startswith("test_anomaly")]
    for key in anomaly_keys:
        split_name = key.replace("test_anomaly_", "").replace("test_anomaly", "ood")
        logger.info(f"Evaluating on {split_name}...")

        anomaly_result = collect_anomaly_scores(
            model, loaders[key], device, fusion, max_batches
        )

        roc = compute_auroc(normal_result["scores"], anomaly_result["scores"])
        results[split_name] = anomaly_result
        results["aurocs"][split_name] = roc["auroc"]

        logger.info(f"  AUROC ({split_name}): {roc['auroc']:.4f}")

    return results
