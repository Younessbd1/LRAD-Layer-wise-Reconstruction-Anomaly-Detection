"""Training loops.

Two stages live here:

1. ``train_model`` — supervised training of the multi-head classifier
   on combined CE(gender) + BCE(attrs) loss.
2. ``train_decoders`` — freezes the classifier and trains one
   ``BlockDecoder`` per conv block to reconstruct the input from that
   block's activations (MSE loss). The reconstructions drive the ensemble
   bias/variance decomposition (the project's anomaly score) and the
   per-block plots.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
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
    for img, gender, attrs, _ in (loader or []):
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


def _has_samples(loader: Optional[DataLoader]) -> bool:
    """True iff ``loader`` is a real loader holding at least one sample.

    A ``val_ratio=0`` config yields ``val=None``; a misconfigured split can
    yield a loader over an empty dataset. Both mean "no validation" — never
    validate against them, because an empty loader's loss is a constant 0
    that would silently trigger early stopping on the first epoch.
    """
    if loader is None:
        return False
    ds = getattr(loader, "dataset", None)
    try:
        return ds is not None and len(ds) > 0
    except TypeError:
        return True


def train_model(
    model: FacialCNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    attr_loss_weight: float = 1.0,
    device: torch.device,
    log_every: int = 5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 5,
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """Train the multi-head classifier with combined CE + BCE loss.

    Returns a history dict with per-epoch train (and, when a validation set
    is given, val) loss, gender accuracy, and per-attribute accuracy.

    When ``checkpoint_dir`` is given (config flag
    ``training.save_every_epoch``), the model state is saved at the end of
    EVERY epoch as ``model_ep{e}.pt`` — these per-epoch checkpoints feed
    ``scripts/epoch_variability_study.py`` (inter-model variability
    ``sigma(e)`` as a function of training epochs).

    When ``val_loader`` is ``None`` or empty (e.g. ``val_ratio=0``), there is
    no validation loop and no early stopping: the model runs the full
    ``epochs`` schedule and the final-epoch weights are kept. With a non-empty
    val set, training early-stops on val loss once it plateaus and the
    best-val weights are restored.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    has_val = _has_samples(val_loader)
    if not has_val:
        logger.info(
            "No validation set — running the full %d-epoch schedule with no "
            "early stopping; final-epoch weights are kept.", epochs,
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_gender_acc": [],
        "val_gender_acc": [],
        "train_attr_acc": [],
        "val_attr_acc": [],
        "batch_gender_acc": [],
        "batch_attr_acc_mean": [],
        "epoch_ends": [],
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

            history["batch_gender_acc"].append(
                (out["gender_logits"].argmax(dim=1) == gender).float().mean().item()
            )
            history["batch_attr_acc_mean"].append(
                (attr_pred == attrs).float().mean().item()
            )

        history["epoch_ends"].append(len(history["batch_gender_acc"]) - 1)

        train_metrics = {
            "loss": loss_sum / max(n, 1),
            "gender_acc": gender_correct / max(n, 1),
            "attr_acc": (attr_correct / max(n, 1)).cpu().tolist(),
        }
        history["train_loss"].append(train_metrics["loss"])
        history["train_gender_acc"].append(train_metrics["gender_acc"])
        history["train_attr_acc"].append(train_metrics["attr_acc"])

        val_metrics = None
        if has_val:
            val_metrics = evaluate_one_epoch(
                model, val_loader, device, attr_loss_weight=attr_loss_weight,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_gender_acc"].append(val_metrics["gender_acc"])
            history["val_attr_acc"].append(val_metrics["attr_acc"])

        if checkpoint_dir is not None:
            torch.save(
                model.state_dict(),
                Path(checkpoint_dir) / f"model_ep{epoch}.pt",
            )

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            # With a val set, report val metrics; without one, report the
            # train metrics (and drop the redundant duplicate train_loss).
            report = val_metrics if has_val else train_metrics
            tag = "val" if has_val else "train"
            attr_acc_str = " ".join(f"{a*100:.1f}" for a in report["attr_acc"])
            prefix = (f"train_loss={train_metrics['loss']:.4f}  "
                      if has_val else "")
            logger.info(
                f"epoch {epoch:>3}/{epochs}  "
                f"{prefix}"
                f"{tag}_loss={report['loss']:.4f}  "
                f"{tag}_gender_acc={report['gender_acc']*100:.1f}%  "
                f"{tag}_attr_acc=[{attr_acc_str}]%  "
                f"({time.time() - t0:.1f}s)"
            )

        # Early stopping only makes sense with a real validation signal.
        if has_val:
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

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"restored best weights (val_loss={best_val:.4f})")

    history["best_val_loss"] = best_val if has_val else None
    return history


# ---------------------------------------------------------------------------
# Decoder training (per-block reconstruction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _decoder_eval(
    model: FacialCNN,
    decoders: nn.ModuleList,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Return (combined_val_loss, per_block_val_loss)."""
    model.eval()
    for d in decoders:
        d.eval()
    n = 0
    total_sum = 0.0
    per_block_sum = [0.0 for _ in decoders]
    for img, _, _, _ in (loader or []):
        img = img.to(device, non_blocking=True)
        _, acts = model.forward_features(img)
        bs = img.size(0)
        n += bs
        for k, d in enumerate(decoders):
            mse = F.mse_loss(d(acts[k]), img)
            per_block_sum[k] += mse.item() * bs
            total_sum += mse.item() * bs
    return total_sum / max(n, 1), [s / max(n, 1) for s in per_block_sum]


def train_decoders(
    model: FacialCNN,
    decoders: nn.ModuleList,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device,
    log_every: int = 1,
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """Train per-block decoders to reconstruct the input from frozen
    classifier activations. Sum-of-MSE objective across all blocks.

    Per-block reconstruction loss is tracked on the validation set when one
    is given; with ``val_loader=None`` or empty (``val_ratio=0``) the
    per-block loss is reported on the training set instead, so the log stays
    meaningful and never collapses to a constant 0.

    When ``checkpoint_dir`` is given (config flag
    ``training.save_every_epoch``), the decoder states are saved at the end
    of EVERY epoch as ``decoders_ep{e}.pt`` — the per-epoch checkpoints
    used by ``scripts/epoch_variability_study.py``.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    decoders.to(device)

    params = [p for d in decoders for p in d.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    has_val = _has_samples(val_loader)

    history = {
        "train_loss": [],
        "train_loss_per_block": [],
        "val_loss": [],
        "val_loss_per_block": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        for d in decoders:
            d.train()

        n = 0
        loss_sum = 0.0
        per_block_sum = [0.0 for _ in decoders]
        for img, _, _, _ in train_loader:
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                _, acts = model.forward_features(img)

            optimizer.zero_grad(set_to_none=True)
            total = img.new_zeros(())
            block_losses = []
            for k, d in enumerate(decoders):
                mse = F.mse_loss(d(acts[k]), img)
                block_losses.append(mse)
                total = total + mse
            total.backward()
            optimizer.step()

            bs = img.size(0)
            n += bs
            loss_sum += total.item() * bs
            for k, mse in enumerate(block_losses):
                per_block_sum[k] += mse.item() * bs

        train_loss = loss_sum / max(n, 1)
        train_per_block = [s / max(n, 1) for s in per_block_sum]
        history["train_loss"].append(train_loss)
        history["train_loss_per_block"].append(train_per_block)

        if checkpoint_dir is not None:
            torch.save(
                decoders.state_dict(),
                Path(checkpoint_dir) / f"decoders_ep{epoch}.pt",
            )

        if has_val:
            val_loss, val_per_block = _decoder_eval(
                model, decoders, val_loader, device,
            )
            history["val_loss"].append(val_loss)
            history["val_loss_per_block"].append(val_per_block)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                blocks_str = " ".join(f"{v:.4f}" for v in val_per_block)
                logger.info(
                    f"[decoders] epoch {epoch:>3}/{epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"per_block(val)=[{blocks_str}]  "
                    f"({time.time() - t0:.1f}s)"
                )
            else:
                blocks_str = " ".join(f"{v:.4f}" for v in train_per_block)
                logger.info(
                    f"[decoders] epoch {epoch:>3}/{epochs}  "
                    f"train={train_loss:.4f}  "
                    f"per_block(train)=[{blocks_str}]  "
                    f"({time.time() - t0:.1f}s)"
                )

    return history
