#!/usr/bin/env python3
"""Offline demo — uses synthetic data to verify the full LRAD pipeline.

Generates simple geometric shapes as 'normal' data and noise/different
shapes as 'anomaly' data. No network downloads required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lrad.utils.helpers import get_device, seed_everything, setup_logging, count_parameters
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_model import LRADModel
from lrad.visualization.heatmaps import plot_heatmap_grid, plot_score_distributions

seed_everything(42)
device = get_device()
out = Path("outputs/synthetic_demo")
out.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(out / "logs"))


# ═══════════════════════════════════════════════
#  Generate synthetic data
# ═══════════════════════════════════════════════

def make_circles(n: int, size: int = 28) -> torch.Tensor:
    """Normal class: centered circles with slight variation."""
    images = torch.zeros(n, 1, size, size)
    for i in range(n):
        cx, cy = size // 2 + np.random.randint(-2, 3), size // 2 + np.random.randint(-2, 3)
        r = np.random.randint(6, 10)
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
        images[i, 0, mask] = 0.8 + np.random.rand() * 0.2
    return images

def make_squares(n: int, size: int = 28) -> torch.Tensor:
    """Normal class: centered squares."""
    images = torch.zeros(n, 1, size, size)
    for i in range(n):
        cx, cy = size // 2 + np.random.randint(-2, 3), size // 2 + np.random.randint(-2, 3)
        half = np.random.randint(4, 8)
        x0, x1 = max(cx - half, 0), min(cx + half, size)
        y0, y1 = max(cy - half, 0), min(cy + half, size)
        images[i, 0, y0:y1, x0:x1] = 0.8 + np.random.rand() * 0.2
    return images

def make_anomalies(n: int, size: int = 28) -> torch.Tensor:
    """Anomaly: random noise blobs in unexpected locations."""
    images = torch.zeros(n, 1, size, size)
    for i in range(n):
        # Background shape (circle)
        cx, cy = size // 2, size // 2
        r = np.random.randint(6, 10)
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
        images[i, 0, mask] = 0.8

        # Add anomalous region (bright patch in corner)
        ax, ay = np.random.choice([3, size - 6]), np.random.choice([3, size - 6])
        aw = np.random.randint(3, 6)
        images[i, 0, ay:ay+aw, ax:ax+aw] = 1.0
    return images


logger.info("Generating synthetic data...")
n_train, n_test = 500, 100

# Two normal classes
train_circles = make_circles(n_train // 2)
train_squares = make_squares(n_train // 2)
train_images = torch.cat([train_circles, train_squares])
train_labels = torch.cat([torch.zeros(n_train // 2), torch.ones(n_train // 2)]).long()

test_normal = torch.cat([make_circles(n_test // 2), make_squares(n_test // 2)])
test_anomaly = make_anomalies(n_test)

# Simple dataloaders
from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=32, shuffle=True)
test_normal_loader = DataLoader(TensorDataset(test_normal, torch.zeros(n_test).long()), batch_size=32)
test_anomaly_loader = DataLoader(TensorDataset(test_anomaly, torch.ones(n_test).long()), batch_size=32)


# ═══════════════════════════════════════════════
#  CNN Pipeline
# ═══════════════════════════════════════════════
logger.info("=" * 50)
logger.info("CNN Pipeline on synthetic data")
logger.info("=" * 50)

channels = [1, 8, 16, 32]
classifier = CNNClassifier(channels, num_classes=2, input_size=28).to(device)
logger.info(f"Spatial sizes: {classifier.spatial_sizes}")
logger.info(f"Classifier params: {count_parameters(classifier):,}")

# Train classifier
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

classifier.train()
for epoch in range(10):
    total_loss = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits, _ = classifier(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        logger.info(f"  Classifier epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

classifier.freeze()

# Build LRAD model + train decoders
model = LRADModel(classifier)
logger.info(f"Decoders: {len(model.decoders)}")

dec_criterion = nn.MSELoss()
dec_optimizers = [torch.optim.Adam(d.parameters(), lr=1e-3) for d in model.decoders]

for epoch in range(15):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _, acts = model.classifier(imgs)
        for i, (dec, opt) in enumerate(zip(model.decoders, dec_optimizers)):
            act = acts[model.decoder_layers[i]].detach()
            opt.zero_grad()
            recon = dec(act)
            loss = dec_criterion(recon, imgs)
            loss.backward()
            opt.step()
    if (epoch + 1) % 5 == 0:
        logger.info(f"  Decoder epoch {epoch+1}")

# Evaluate
model.eval()
normal_scores = []
anomaly_scores = []
normal_heatmaps = []
anomaly_heatmaps = []
normal_images = []
anomaly_images = []
normal_per_layer = [[] for _ in range(len(model.decoders))]
anomaly_per_layer = [[] for _ in range(len(model.decoders))]

with torch.no_grad():
    for imgs, _ in test_normal_loader:
        imgs = imgs.to(device)
        result = model.compute_anomaly_maps(imgs, fusion="mean")
        normal_scores.append(result["scores"].cpu().numpy())
        normal_heatmaps.append(result["fused"].squeeze(1).cpu().numpy())
        normal_images.append(imgs.cpu().numpy())
        for j, plm in enumerate(result["per_layer"]):
            normal_per_layer[j].append(plm.squeeze(1).cpu().numpy())

    for imgs, _ in test_anomaly_loader:
        imgs = imgs.to(device)
        result = model.compute_anomaly_maps(imgs, fusion="mean")
        anomaly_scores.append(result["scores"].cpu().numpy())
        anomaly_heatmaps.append(result["fused"].squeeze(1).cpu().numpy())
        anomaly_images.append(imgs.cpu().numpy())
        for j, plm in enumerate(result["per_layer"]):
            anomaly_per_layer[j].append(plm.squeeze(1).cpu().numpy())

normal_scores = np.concatenate(normal_scores)
anomaly_scores = np.concatenate(anomaly_scores)
normal_heatmaps = np.concatenate(normal_heatmaps)
anomaly_heatmaps = np.concatenate(anomaly_heatmaps)
normal_images = np.concatenate(normal_images)
anomaly_images = np.concatenate(anomaly_images)
normal_per_layer = [np.concatenate(p) for p in normal_per_layer]
anomaly_per_layer = [np.concatenate(p) for p in anomaly_per_layer]

# AUROC
from sklearn.metrics import roc_auc_score
labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
all_scores = np.concatenate([normal_scores, anomaly_scores])
auroc = roc_auc_score(labels, all_scores)
logger.info(f"IMAGE-LEVEL AUROC: {auroc:.4f}")

# Visualizations
n = 8
plot_heatmap_grid(
    normal_images[:n], normal_heatmaps[:n],
    [p[:n] for p in normal_per_layer],
    scores=normal_scores[:n],
    title=f"Normal samples (AUROC={auroc:.3f})",
    save_path=str(out / "cnn_normal.png"),
)

plot_heatmap_grid(
    anomaly_images[:n], anomaly_heatmaps[:n],
    [p[:n] for p in anomaly_per_layer],
    scores=anomaly_scores[:n],
    title=f"Anomaly samples (AUROC={auroc:.3f})",
    save_path=str(out / "cnn_anomaly.png"),
)

plot_score_distributions(
    normal_scores, {"anomaly": anomaly_scores},
    title=f"Score Distribution (AUROC={auroc:.3f})",
    save_path=str(out / "score_dist.png"),
)

logger.info(f"Normal mean score:  {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
logger.info(f"Anomaly mean score: {anomaly_scores.mean():.4f} ± {anomaly_scores.std():.4f}")
logger.info(f"All outputs saved to {out}/")
logger.info("Demo complete.")
