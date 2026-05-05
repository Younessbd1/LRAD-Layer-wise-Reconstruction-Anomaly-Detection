"""Training loop: combined cross-entropy (gender) + BCE (attrs) loss."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import FacialCNN

logger = logging.getLogger("celeba_ood")


def _accuracy_gender(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return the fraction of correctly classified Male/Female samples."""
    return (logits.argmax(dim=1) == target).float().mean().item()


def _accuracy_attrs(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-attribute binary accuracy (vector of length n_attrs)."""
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == target).float().mean(dim=0)


@torch.no_grad()
def evaluate_one_epoch(
    model: FacialCNN,
    loader: DataLoader,
    device: torch.device,
    attr_loss_weight: float = 1.0,
) -> dict:
    model.eval()
    n = 0
    loss_sum = 0.0
    gender_correct = 0
    attr_correct = torch.zeros(model.n_attrs, device=device)
    for img, gender, attrs, _ in loader:
        img = img.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)

        out = model(img)
        loss_g = F.cross_entropy(out["gender_logits"], gender)
        loss_a = F.binary_cross_entropy_with_logits(out["attr_logits"], attrs)
        loss = loss_g + attr_loss_weight * loss_a

        bs = img.size(0)
        n += bs
        loss_sum += loss.item() * bs
        gender_correct += (out["gender_logits"].argmax(dim=1) == gender).sum().item()
        attr_pred = (torch.sigmoid(out["attr_logits"]) >= 0.5).float()
        attr_correct += (attr_pred == attrs).float().sum(dim=0)

    return {
        "loss": loss_sum / max(n, 1),
        "gender_acc": gender_correct / max(n, 1),
        "attr_acc": (attr_correct / max(n, 1)).cpu().tolist(),
    }


def train_model(
    model: FacialCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    attr_loss_weight: float = 1.0,
    device: torch.device,
    log_every: int = 5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 5,
) -> dict:
    """Train the multi-head classifier with combined CE + BCE loss.

    Returns a history dict with per-epoch train/val loss, gender accuracy,
    and per-attribute accuracy. Early stops on val loss once it plateaus.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_gender_acc": [],
        "val_gender_acc": [],
        "train_attr_acc": [],
        "val_attr_acc": [],
    }

    best_val = float("inf")
    best_state: Optional[dict] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        n = 0
        loss_sum = 0.0
        gender_correct = 0
        attr_correct = torch.zeros(model.n_attrs, device=device)

        for img, gender, attrs, _ in train_loader:
            img = img.to(device, non_blocking=True)
            gender = gender.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)

            out = model(img)
            loss_g = F.cross_entropy(out["gender_logits"], gender)
            loss_a = F.binary_cross_entropy_with_logits(
                out["attr_logits"], attrs,
            )
            loss = loss_g + attr_loss_weight * loss_a

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = img.size(0)
            n += bs
            loss_sum += loss.item() * bs
            gender_correct += (
                out["gender_logits"].argmax(dim=1) == gender
            ).sum().item()
            attr_pred = (torch.sigmoid(out["attr_logits"]) >= 0.5).float()
            attr_correct += (attr_pred == attrs).float().sum(dim=0)

        train_metrics = {
            "loss": loss_sum / max(n, 1),
            "gender_acc": gender_correct / max(n, 1),
            "attr_acc": (attr_correct / max(n, 1)).cpu().tolist(),
        }
        val_metrics = evaluate_one_epoch(
            model, val_loader, device, attr_loss_weight=attr_loss_weight,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_gender_acc"].append(train_metrics["gender_acc"])
        history["val_gender_acc"].append(val_metrics["gender_acc"])
        history["train_attr_acc"].append(train_metrics["attr_acc"])
        history["val_attr_acc"].append(val_metrics["attr_acc"])

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            attr_acc_str = " ".join(
                f"{a*100:.1f}" for a in val_metrics["attr_acc"]
            )
            logger.info(
                f"epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_gender_acc={val_metrics['gender_acc']*100:.1f}%  "
                f"val_attr_acc=[{attr_acc_str}]%  "
                f"({time.time() - t0:.1f}s)"
            )

        if val_metrics["loss"] < best_val - early_stop_min_delta:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if (
                epochs_no_improve >= early_stop_patience
                and epoch >= early_stop_min_epochs
            ):
                logger.info(
                    f"early stop at epoch {epoch} "
                    f"(no val improvement for {epochs_no_improve} epochs)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"restored best weights (val_loss={best_val:.4f})")

    history["best_val_loss"] = best_val
    return history
