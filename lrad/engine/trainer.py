"""Training loops for the LRAD pipeline.

Two-phase training:
  Phase 1: Train the classifier on normal data (standard supervised).
  Phase 2: Freeze classifier, train each decoder independently to
           reconstruct input images from its assigned activation layer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
import logging

from ..models.classifier import CNNClassifier, MLPClassifier
from ..models.lrad_model import LRADModel
from ..models.lrad_nested import LRADModelNested

logger = logging.getLogger("lrad")


def train_classifier(
    classifier: CNNClassifier | MLPClassifier,
    train_loader: DataLoader,
    class_map: dict[int, int],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Phase 1: Train the classifier on normal-class data.

    Args:
        classifier: The classifier model.
        train_loader: DataLoader with normal-class samples.
        class_map: Mapping from original labels to 0..N-1.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device.

    Returns:
        dict with 'losses' (list of epoch-average losses) and
        'accuracies' (list of epoch-average accuracies).
    """
    classifier.to(device)
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    history = {"losses": [], "accuracies": []}

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            # Remap labels
            labels = torch.tensor(
                [class_map[int(l)] for l in labels], device=device
            )

            optimizer.zero_grad()
            logits, _ = classifier(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        history["losses"].append(avg_loss)
        history["accuracies"].append(accuracy)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  [Classifier] Epoch {epoch+1}/{epochs} — "
                f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}"
            )

    # Freeze after training
    classifier.freeze()
    logger.info("  Classifier frozen.")

    return history


def train_decoders(
    model: LRADModel,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Phase 2: Train decoders to reconstruct images from frozen activations.

    Each decoder is trained independently with MSE loss against the original
    input image.

    Args:
        model: The LRADModel (classifier already frozen, decoders trainable).
        train_loader: DataLoader with normal-class samples.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device.

    Returns:
        dict with 'losses': list of lists (per-decoder, per-epoch average MSE).
    """
    model.to(device)
    criterion = nn.MSELoss()

    # One optimizer per decoder
    optimizers = [optim.Adam(dec.parameters(), lr=lr) for dec in model.decoders]

    n_decoders = len(model.decoders)
    history = {"losses": [[] for _ in range(n_decoders)]}

    for epoch in range(epochs):
        epoch_losses = [0.0] * n_decoders
        total = 0

        for images, _ in train_loader:
            images = images.to(device)
            total += images.size(0)

            # Get frozen activations
            with torch.no_grad():
                _, all_activations = model.classifier(images)

            # Train each decoder
            for i, (decoder, optimizer) in enumerate(zip(model.decoders, optimizers)):
                act = all_activations[model.decoder_layers[i]].detach()

                optimizer.zero_grad()
                reconstruction = decoder(act)
                loss = criterion(reconstruction, images)
                loss.backward()
                optimizer.step()

                epoch_losses[i] += loss.item() * images.size(0)

        for i in range(n_decoders):
            avg = epoch_losses[i] / total
            history["losses"][i].append(avg)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss_str = ", ".join(f"D{i}={history['losses'][i][-1]:.4f}"
                                for i in range(n_decoders))
            logger.info(
                f"  [Decoders] Epoch {epoch+1}/{epochs} — {loss_str}"
            )

    return history


def train_decoders_nested(
    model: LRADModelNested,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train nested decoders with local targets.

    Each decoder D_k is trained with:
      D_0:  loss = MSE(D_0(a_0), x)            ← image as target
      D_k:  loss = MSE(D_k(a_k), a_{k-1})      ← previous activation as target

    All decoders are updated jointly each batch (one optimizer per decoder),
    but losses are isolated — no gradients flow between decoders. This matches
    the diagram's "stage k" semantics while staying training-efficient.

    Returns:
        dict with 'losses': list of lists (per-decoder, per-epoch average MSE).
    """
    model.to(device)
    criterion = nn.MSELoss()

    n_decoders = len(model.decoders)
    optimizers = [optim.Adam(d.parameters(), lr=lr) for d in model.decoders]
    history = {"losses": [[] for _ in range(n_decoders)]}

    for epoch in range(epochs):
        epoch_losses = [0.0] * n_decoders
        total = 0

        for images, _ in train_loader:
            images = images.to(device)
            B = images.size(0)
            total += B

            with torch.no_grad():
                _, all_acts = model.classifier(images)

            for k in range(n_decoders):
                input_act = all_acts[k].detach()
                target = images if k == 0 else all_acts[k - 1].detach()

                optimizers[k].zero_grad()
                pred = model.decoders[k](input_act)
                loss = criterion(pred, target)
                loss.backward()
                optimizers[k].step()

                epoch_losses[k] += loss.item() * B

        for k in range(n_decoders):
            history["losses"][k].append(epoch_losses[k] / total)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss_str = ", ".join(
                f"D{k}={history['losses'][k][-1]:.4f}" for k in range(n_decoders)
            )
            logger.info(
                f"  [Nested Decoders] Epoch {epoch+1}/{epochs} — {loss_str}"
            )

    return history
