"""Training and evaluation for LRAD-UQ models.

Handles the specifics of MC Dropout and Ensemble decoder training:
  - MC Dropout: Train normally (dropout active), evaluate with T passes
  - Ensemble: Train each member independently with different random seeds
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Optional
import logging

from ..models.lrad_uq import LRADModelUQ
from ..models.uq_decoders import EnsembleDecoder

logger = logging.getLogger("lrad")


def train_uq_decoders(
    model: LRADModelUQ,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train UQ decoders (MC Dropout or Ensemble members).

    For MC Dropout decoders: standard training with dropout active.
    For Ensemble decoders: train each member independently.
    """
    model.to(device)
    criterion = nn.MSELoss()
    history = {"losses": [[] for _ in range(len(model.decoders))]}

    # Build optimizers
    if model.uq_method == "ensemble":
        # One optimizer per member per decoder
        all_optimizers = []
        for dec in model.decoders:
            all_optimizers.append(dec.get_optimizers(lr=lr))
    else:
        all_optimizers = [
            [optim.Adam(dec.parameters(), lr=lr)]
            for dec in model.decoders
        ]

    for epoch in range(epochs):
        epoch_losses = [0.0] * len(model.decoders)
        total = 0

        for images, _ in train_loader:
            images = images.to(device)
            total += images.size(0)

            with torch.no_grad():
                _, all_acts = model.classifier(images)

            for i, dec in enumerate(model.decoders):
                act = all_acts[model.decoder_layers[i]].detach()

                if model.uq_method == "ensemble":
                    # Train each ensemble member independently
                    member_loss = 0.0
                    for member, opt in zip(dec.members, all_optimizers[i]):
                        opt.zero_grad()
                        recon = member(act)
                        loss = criterion(recon, images)
                        loss.backward()
                        opt.step()
                        member_loss += loss.item()
                    epoch_losses[i] += (member_loss / len(dec.members)) * images.size(0)
                else:
                    # MC Dropout: standard training
                    dec.train()
                    all_optimizers[i][0].zero_grad()
                    recon = dec(act)
                    loss = criterion(recon, images)
                    loss.backward()
                    all_optimizers[i][0].step()
                    epoch_losses[i] += loss.item() * images.size(0)

        for i in range(len(model.decoders)):
            history["losses"][i].append(epoch_losses[i] / total)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss_str = ", ".join(
                f"D{i}={history['losses'][i][-1]:.4f}"
                for i in range(len(model.decoders))
            )
            logger.info(f"  [UQ Decoders] Epoch {epoch+1}/{epochs} - {loss_str}")

    return history


@torch.no_grad()
def evaluate_uq(
    model: LRADModelUQ,
    dataloader: DataLoader,
    device: torch.device,
    fusion: str = "mean",
    max_batches: Optional[int] = None,
) -> dict:
    """Collect UQ-enhanced anomaly scores from a dataloader.

    Returns:
        dict with 'scores_error', 'scores_uncertainty', 'scores_combined',
        'error_maps', 'uncertainty_maps', 'combined_maps', 'images'.
    """
    model.to(device)
    results = {
        "scores_error": [], "scores_uncertainty": [], "scores_combined": [],
        "error_maps": [], "uncertainty_maps": [], "combined_maps": [],
        "images": [],
    }

    for batch_idx, (images, _) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        out = model.compute_uq_anomaly_maps(images, fusion=fusion)

        results["scores_error"].append(out["scores_error"].cpu().numpy())
        results["scores_uncertainty"].append(out["scores_uncertainty"].cpu().numpy())
        results["scores_combined"].append(out["scores_combined"].cpu().numpy())
        results["error_maps"].append(out["fused_error"].squeeze(1).cpu().numpy())
        results["uncertainty_maps"].append(out["fused_uncertainty"].squeeze(1).cpu().numpy())
        results["combined_maps"].append(out["combined"].squeeze(1).cpu().numpy())
        results["images"].append(images.cpu().numpy())

    for key in results:
        results[key] = np.concatenate(results[key])

    return results


def full_uq_evaluation(
    model: LRADModelUQ,
    loaders: dict,
    device: torch.device,
    fusion: str = "mean",
    max_batches: Optional[int] = None,
) -> dict:
    """Full UQ evaluation across all splits.

    Returns per-split results + AUROC for each scoring method.
    """
    logger.info("UQ Evaluation - normal test set...")
    normal = evaluate_uq(model, loaders["test_normal"], device, fusion, max_batches)

    output = {"normal": normal, "aurocs": {}}

    anomaly_keys = [k for k in loaders if k.startswith("test_anomaly")]
    for key in anomaly_keys:
        split = key.replace("test_anomaly_", "").replace("test_anomaly", "ood")
        logger.info(f"UQ Evaluation - {split}...")
        anomaly = evaluate_uq(model, loaders[key], device, fusion, max_batches)
        output[split] = anomaly

        # Compute AUROC for each scoring method
        for score_type in ["error", "uncertainty", "combined"]:
            score_key = f"scores_{score_type}"
            labels = np.concatenate([
                np.zeros(len(normal[score_key])),
                np.ones(len(anomaly[score_key])),
            ])
            scores = np.concatenate([normal[score_key], anomaly[score_key]])
            auroc = roc_auc_score(labels, scores)
            output["aurocs"][f"{split}_{score_type}"] = auroc
            logger.info(f"  AUROC ({split}, {score_type}): {auroc:.4f}")

    return output
