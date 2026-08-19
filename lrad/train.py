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

from .cutpaste import cutpaste_batch
from .model import FacialCNN

logger = logging.getLogger("celeba_ood")


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
        # Heads the model was built without contribute nothing. The
        # CutPaste pretext CE is tracked per batch in ``train_model``
        # instead, since it needs the CutPaste labels.
        loss = img.new_zeros(())
        bs = img.size(0)
        if "gender_logits" in out:
            loss = loss + F.cross_entropy(out["gender_logits"], gender)
            gender_correct += (
                out["gender_logits"].argmax(dim=1) == gender
            ).sum().item()
        if "attr_logits" in out:
            loss = loss + attr_loss_weight * F.binary_cross_entropy_with_logits(
                out["attr_logits"], attrs,
            )
            attr_pred = (torch.sigmoid(out["attr_logits"]) >= 0.5).float()
            attr_correct += (attr_pred == attrs).float().sum(dim=0)

        n += bs
        loss_sum += loss.item() * bs

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
    attr_loss_weight: float = 1.0,
    device: torch.device,
    log_every: int = 5,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 5,
    checkpoint_dir: Optional[Path] = None,
    cutpaste: Optional[dict] = None,
) -> dict:
    """Train the multi-head classifier with combined CE + BCE loss.

    ``cutpaste`` (config section ``training.cutpaste``) enables the
    CutPaste pretext task: each batch is augmented with
    :func:`lrad.cutpaste.cutpaste_batch` (keys ``prob`` / ``area_range`` /
    ``aspect_range`` / ``scar_prob`` are forwarded), the model's
    ``cutpaste_logits`` head is trained with CE against the intact/altered
    labels (weighted by ``loss_weight``, default 1.0), and the supervised
    gender/attr losses are computed on the INTACT images only — a pasted
    patch can occlude the very evidence those labels describe. Requires a
    model built with ``model.cutpaste_head: true``.

    Returns a history dict with per-epoch train (and, when a validation set
    is given, val) loss, gender accuracy, and per-attribute accuracy, plus
    the per-batch series (``batch_loss``, ``batch_*_acc``) and the
    ``epoch_ends`` markers the batch-level figures draw from.

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cutpaste is not None and getattr(model, "head_cutpaste", None) is None:
        raise ValueError(
            "training.cutpaste is set but the model has no cutpaste head — "
            "set model.cutpaste_head: true"
        )
    cp_weight = float(cutpaste.get("loss_weight", 1.0)) if cutpaste else 0.0
    cp_kwargs = (
        {
            "prob": float(cutpaste.get("prob", 0.5)),
            "area_range": tuple(cutpaste.get("area_range", (0.02, 0.12))),
            "aspect_range": tuple(cutpaste.get("aspect_range", (0.3, 3.3))),
            "scar_prob": float(cutpaste.get("scar_prob", 0.5)),
            # Derived from the head the model actually carries rather than
            # read from the config: a 3-way head fed binary labels would
            # train class 2 on nothing and never error, so the two must not
            # be able to disagree.
            "three_way": getattr(model, "cutpaste_classes", 2) == 3,
        }
        if cutpaste is not None else None
    )
    if cp_kwargs is not None:
        logger.info("CutPaste pretext on: %s  loss_weight=%g",
                    cp_kwargs, cp_weight)

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
        "batch_loss": [],
        "batch_gender_acc": [],
        "batch_attr_acc_mean": [],
        "batch_cutpaste_acc": [],
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

            if cp_kwargs is not None:
                # Augment on CPU-side tensor already moved: cutpaste_batch
                # is pure tensor indexing, cheap on either device.
                img, cp_labels = cutpaste_batch(img, **cp_kwargs)
                intact = cp_labels == 0
            else:
                cp_labels = None
                intact = torch.ones(
                    img.size(0), dtype=torch.bool, device=img.device,
                )

            out = model(img)
            # Supervised losses on intact faces only — a pasted patch can
            # occlude the very evidence the gender/attr labels describe.
            loss = img.new_zeros(())
            if "gender_logits" in out:
                loss = loss + (
                    F.cross_entropy(out["gender_logits"][intact],
                                    gender[intact])
                    if intact.any()
                    # vanishingly rare with prob <= 0.5, but keep the graph
                    # connected so .backward() never sees a detached loss
                    else out["gender_logits"].sum() * 0.0
                )
            if "attr_logits" in out:
                loss = loss + attr_loss_weight * (
                    F.binary_cross_entropy_with_logits(
                        out["attr_logits"][intact], attrs[intact])
                    if intact.any()
                    else out["attr_logits"].sum() * 0.0
                )
            if cp_labels is not None:
                loss_cp = F.cross_entropy(
                    out["cutpaste_logits"], cp_labels,
                )
                loss = loss + cp_weight * loss_cp
                cp_match = out["cutpaste_logits"].argmax(dim=1) == cp_labels
                history["batch_cutpaste_acc"].append(
                    cp_match.float().mean().item()
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Epoch metrics stay intact-only so they mean the same thing
            # with and without the pretext task.
            bs = int(intact.sum().item())
            n += bs
            loss_sum += loss.item() * bs
            if "gender_logits" in out:
                g_out = out["gender_logits"][intact]
                gender_match = g_out.argmax(dim=1) == gender[intact]
                gender_correct += gender_match.sum().item()
                batch_gender_acc = (
                    gender_match.float().mean().item() if bs else 0.0
                )
            else:
                batch_gender_acc = 0.0
            if "attr_logits" in out:
                attr_pred = (
                    torch.sigmoid(out["attr_logits"][intact]) >= 0.5
                ).float()
                attr_match = (attr_pred == attrs[intact]).float()
                attr_correct += attr_match.sum(dim=0)
                batch_attr_acc = attr_match.mean().item() if bs else 0.0
            else:
                batch_attr_acc = 0.0

            history["batch_loss"].append(loss.item())
            history["batch_gender_acc"].append(batch_gender_acc)
            history["batch_attr_acc_mean"].append(batch_attr_acc)

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
            prefix = (f"train_loss={train_metrics['loss']:.4f}  "
                      if has_val else "")
            # Only report the heads the model has. A pretext-only trunk
            # would otherwise log a constant "gender_acc=0.0% attr_acc=[]",
            # which reads like a broken run rather than an absent head.
            parts = [f"{tag}_loss={report['loss']:.4f}"]
            if model.head_gender is not None:
                parts.append(f"{tag}_gender_acc={report['gender_acc']*100:.1f}%")
            if model.head_attrs is not None:
                attr_acc_str = " ".join(
                    f"{a*100:.1f}" for a in report["attr_acc"]
                )
                parts.append(f"{tag}_attr_acc=[{attr_acc_str}]%")
            if history["batch_cutpaste_acc"]:
                # Mean over the epoch's batches: the number that says
                # whether the pretext head is learning at all.
                start = (history["epoch_ends"][-2] + 1
                         if len(history["epoch_ends"]) > 1 else 0)
                ep_cp = history["batch_cutpaste_acc"][start:]
                if ep_cp:
                    parts.append(
                        f"train_cutpaste_acc={sum(ep_cp)/len(ep_cp)*100:.1f}%"
                    )
            logger.info(
                f"epoch {epoch:>3}/{epochs}  {prefix}"
                + "  ".join(parts)
                + f"  ({time.time() - t0:.1f}s)"
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

    No regularizer is applied: the ensemble is a plain deep ensemble whose
    diversity comes only from the random init and the SGD shuffle order.

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
    optimizer = torch.optim.Adam(params, lr=lr)

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
