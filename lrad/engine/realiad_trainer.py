"""Two-phase LRAD training driven by Real-IAD dataloaders.

Phase 1 (classifier).
    Self-supervised rotation prediction (4-way) by default; supports a
    standard category-classification pretext too. Tracks train + val loss
    and accuracy each epoch.

Phase 2 (decoders).
    Frozen classifier, MSE reconstruction. One optimizer per decoder so
    decoders never share gradients. Tracks per-decoder train + val loss
    each epoch and optionally writes the lowest-val checkpoint to disk.

Both training functions consume the (image, label, extras) triples that
``lrad.data.realiad.get_realiad_loaders`` produces; ``label`` is ignored
for rotation pretext, used directly for category pretext.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models import DeepCNNClassifier, LRADModel

logger = logging.getLogger("lrad")


# ---------------------------------------------------------------------------
# Rotation pretext
# ---------------------------------------------------------------------------

def _rotate_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate the batch four times, each rotation as its own example.

    Returns:
        rotated (4B, C, H, W) — concatenation of [rot0, rot90, rot180, rot270]
        labels  (4B,)         — [0,..,0, 1,..,1, 2,..,2, 3,..,3]
    """
    B = x.size(0)
    r0 = x
    r1 = torch.rot90(x, 1, dims=[-2, -1])
    r2 = torch.rot90(x, 2, dims=[-2, -1])
    r3 = torch.rot90(x, 3, dims=[-2, -1])
    rotated = torch.cat([r0, r1, r2, r3], dim=0)
    labels = torch.arange(4, device=x.device).repeat_interleave(B)
    return rotated, labels


def _unpack(batch) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Accept either (img, lbl) or (img, lbl, extras)."""
    if len(batch) == 2:
        return batch[0], batch[1], {}
    return batch[0], batch[1], batch[2] if len(batch) > 2 else {}


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def train_classifier_pretext(
    classifier: DeepCNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pretext: str = "rotation",
    device: torch.device = torch.device("cpu"),
    log_every: int = 5,
    early_stop_patience: int | None = 8,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 5,
) -> dict:
    """Train the classifier on a self-supervised pretext.

    Pretexts:
        'rotation' — predict 0°/90°/180°/270° rotation of each image.
        'category' — supervised class prediction using the 'category' extra.

    Early stopping (val_loader required, otherwise disabled):
        Stop when the val loss has not improved by >= ``early_stop_min_delta``
        for ``early_stop_patience`` consecutive epochs. Set patience to None
        or 0 to disable. Always runs at least ``early_stop_min_epochs`` epochs
        so a noisy first epoch can't kill the run.

    Returns history with epoch-level train/val loss and accuracy.
    """
    if pretext not in {"rotation", "category"}:
        raise ValueError(f"unknown pretext {pretext!r}")

    classifier.to(device)
    # Undo any prior freeze(): when the classifier is wrapped in LRADModel
    # before phase 1, LRADModel.__init__ disables grads and switches it to
    # eval mode. Re-enable both so this training pass can learn.
    for p in classifier.parameters():
        p.requires_grad = True
    classifier.train()

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    def _run_epoch(loader: DataLoader, train: bool) -> tuple[float, float]:
        classifier.train(train)
        loss_sum = 0.0
        correct = 0
        total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                images, labels, extras = _unpack(batch)
                images = images.to(device, non_blocking=True)

                if pretext == "rotation":
                    images, targets = _rotate_batch(images)
                else:
                    cat = extras.get("category")
                    if cat is None:
                        raise RuntimeError(
                            "category pretext requires 'category' in extras — "
                            "set return_category=True on the loader."
                        )
                    targets = cat.to(device, non_blocking=True)

                logits, _ = classifier(images)
                loss = criterion(logits, targets)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item() * targets.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                total += targets.size(0)

        return loss_sum / max(total, 1), correct / max(total, 1)

    best_val = math.inf
    bad_epochs = 0
    use_early_stop = (
        val_loader is not None
        and early_stop_patience is not None
        and early_stop_patience > 0
    )

    for epoch in range(epochs):
        tr_loss, tr_acc = _run_epoch(train_loader, train=True)
        va_loss, va_acc = (
            _run_epoch(val_loader, train=False)
            if val_loader is not None else (math.nan, math.nan)
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if (epoch + 1) % log_every == 0 or epoch in (0, epochs - 1):
            logger.info(
                f"  [classifier] {epoch + 1:3d}/{epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}"
            )

        if use_early_stop:
            if va_loss < best_val - early_stop_min_delta:
                best_val = va_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            if (epoch + 1 >= early_stop_min_epochs
                    and bad_epochs >= early_stop_patience):
                logger.info(
                    f"  [classifier] early stop at epoch {epoch + 1}: "
                    f"val_loss has not improved for {bad_epochs} epochs "
                    f"(best={best_val:.4f})"
                )
                break

    classifier.freeze()
    return history


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def train_decoders_with_val(
    model: LRADModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    log_every: int = 5,
    save_best_path: str | None = None,
) -> dict:
    """Train every decoder independently with MSE on the input image.

    Returns history::

        {
          'train_loss': [[per-epoch] for each decoder],
          'val_loss':   [[per-epoch] for each decoder],
          'best_val':   float — sum of per-decoder val losses at best epoch,
        }
    """
    model.to(device)
    n = len(model.decoders)
    optimizers = [
        optim.AdamW(d.parameters(), lr=lr, weight_decay=weight_decay)
        for d in model.decoders
    ]
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs) for o in optimizers
    ]
    criterion = nn.MSELoss()

    history: dict[str, Any] = {
        "train_loss": [[] for _ in range(n)],
        "val_loss":   [[] for _ in range(n)],
        "best_val":   float("inf"),
    }
    best_state: list[dict] | None = None

    for epoch in range(epochs):
        # ---- train ----
        for d in model.decoders:
            d.train()
        sums = [0.0] * n
        total = 0
        for batch in train_loader:
            images, _, _ = _unpack(batch)
            images = images.to(device, non_blocking=True)
            B = images.size(0)
            total += B

            with torch.no_grad():
                _, all_acts = model.classifier(images)

            for k in range(n):
                act = all_acts[model.decoder_layers[k]].detach()
                optimizers[k].zero_grad()
                pred = model.decoders[k](act)
                loss = criterion(pred, images)
                loss.backward()
                optimizers[k].step()
                sums[k] += loss.item() * B

        for k in range(n):
            history["train_loss"][k].append(sums[k] / max(total, 1))

        # ---- val ----
        if val_loader is not None:
            for d in model.decoders:
                d.eval()
            v_sums = [0.0] * n
            v_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, _, _ = _unpack(batch)
                    images = images.to(device, non_blocking=True)
                    B = images.size(0)
                    v_total += B
                    _, all_acts = model.classifier(images)
                    for k in range(n):
                        act = all_acts[model.decoder_layers[k]]
                        pred = model.decoders[k](act)
                        loss = criterion(pred, images)
                        v_sums[k] += loss.item() * B
            for k in range(n):
                history["val_loss"][k].append(v_sums[k] / max(v_total, 1))
            v_total_loss = sum(h[-1] for h in history["val_loss"])
            if v_total_loss < history["best_val"]:
                history["best_val"] = v_total_loss
                best_state = [
                    {k: v.detach().cpu().clone() for k, v in d.state_dict().items()}
                    for d in model.decoders
                ]
        else:
            for k in range(n):
                history["val_loss"][k].append(math.nan)

        for s in schedulers:
            s.step()

        if (epoch + 1) % log_every == 0 or epoch in (0, epochs - 1):
            tr = ", ".join(f"D{k}={history['train_loss'][k][-1]:.4f}" for k in range(n))
            va = ", ".join(f"D{k}={history['val_loss'][k][-1]:.4f}" for k in range(n))
            logger.info(
                f"  [decoders]   {epoch + 1:3d}/{epochs}  "
                f"train [{tr}]  val [{va}]"
            )

    if save_best_path is not None and best_state is not None:
        Path(save_best_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_best_path)
        logger.info(f"  best decoders saved to {save_best_path}")
        # Restore best weights into the live model for downstream evaluation.
        for d, st in zip(model.decoders, best_state):
            d.load_state_dict(st)

    return history


# ---------------------------------------------------------------------------
# Convenience: feature collection (used by feature-space scatter plot)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_features(
    model: LRADModel,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int = -1,
    max_samples: int | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return globally-pooled activations from ``layer_idx`` (default last stage)
    plus the per-image labels.
    """
    import numpy as np

    model.to(device)
    model.eval()

    classifier = model.classifier
    feats: list[torch.Tensor] = []
    labs: list[int] = []
    n = 0
    for batch in loader:
        images, labels, _ = _unpack(batch)
        images = images.to(device, non_blocking=True)
        _, acts = classifier(images)
        a = acts[layer_idx]
        f = nn.functional.adaptive_avg_pool2d(a, 1).flatten(1).cpu()
        feats.append(f)
        labs.extend(labels.tolist())
        n += images.size(0)
        if max_samples is not None and n >= max_samples:
            break

    feats_np = torch.cat(feats, dim=0).numpy()
    if max_samples is not None:
        feats_np = feats_np[:max_samples]
        labs = labs[:max_samples]
    return feats_np, np.asarray(labs, dtype=np.int64)
