#!/usr/bin/env python3
"""Quick smoke test - runs a mini experiment to verify the full pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")

from lrad.utils.helpers import get_device, seed_everything, setup_logging
from lrad.data.datasets import get_mnist_loaders
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_model import LRADModel
from lrad.engine.trainer import train_classifier, train_decoders
from lrad.engine.evaluator import evaluate_full
from lrad.visualization.heatmaps import plot_heatmap_grid, plot_score_distributions

seed_everything(42)
device = get_device()
logger = setup_logging("outputs/smoke_test/logs")
out = Path("outputs/smoke_test")
out.mkdir(parents=True, exist_ok=True)

logger.info(f"Device: {device}")

# --- Load data ---
loaders = get_mnist_loaders(normal_classes=[0, 1, 2, 3], batch_size=128)

# ═══════════════════════════════════════════
#  TEST 1: CNN pipeline
# ═══════════════════════════════════════════
logger.info("=" * 50)
logger.info("CNN PIPELINE")
logger.info("=" * 50)

cnn = CNNClassifier([1, 8, 16, 32], num_classes=4, input_size=28)
logger.info(f"CNN spatial sizes: {cnn.spatial_sizes}")

train_classifier(cnn, loaders["train"], loaders["class_map"],
                 epochs=3, lr=1e-3, device=device)

cnn_model = LRADModel(cnn)
train_decoders(cnn_model, loaders["train"], epochs=3, lr=1e-3, device=device)

cnn_results = evaluate_full(cnn_model, loaders, device, fusion="mean", max_batches=5)

for name, auroc in cnn_results["aurocs"].items():
    logger.info(f"  CNN AUROC ({name}): {auroc:.4f}")

# Heatmaps
for split in ["normal", "mnist", "fashion"]:
    data = cnn_results.get(split, cnn_results.get("normal"))
    if data is None:
        continue
    n = min(8, len(data["images"]))
    plot_heatmap_grid(
        data["images"][:n], data["heatmaps"][:n],
        [p[:n] for p in data["per_layer_maps"]],
        scores=data["scores"][:n],
        title=f"CNN - {split}",
        save_path=str(out / f"cnn_{split}.png"),
    )

# ═══════════════════════════════════════════
#  TEST 2: MLP pipeline
# ═══════════════════════════════════════════
logger.info("=" * 50)
logger.info("MLP PIPELINE")
logger.info("=" * 50)

mlp = MLPClassifier(784, [512, 256, 128], num_classes=4, input_size=28)

train_classifier(mlp, loaders["train"], loaders["class_map"],
                 epochs=3, lr=1e-3, device=device)

mlp_model = LRADModel(mlp)
train_decoders(mlp_model, loaders["train"], epochs=3, lr=1e-3, device=device)

mlp_results = evaluate_full(mlp_model, loaders, device, fusion="mean", max_batches=5)

for name, auroc in mlp_results["aurocs"].items():
    logger.info(f"  MLP AUROC ({name}): {auroc:.4f}")

# Heatmaps
for split in ["normal", "mnist", "fashion"]:
    data = mlp_results.get(split, mlp_results.get("normal"))
    if data is None:
        continue
    n = min(8, len(data["images"]))
    plot_heatmap_grid(
        data["images"][:n], data["heatmaps"][:n],
        [p[:n] for p in data["per_layer_maps"]],
        scores=data["scores"][:n],
        title=f"MLP - {split}",
        save_path=str(out / f"mlp_{split}.png"),
    )

# Score distributions
for label, res in [("CNN", cnn_results), ("MLP", mlp_results)]:
    anom = {}
    for k in res:
        if k not in ("normal", "aurocs"):
            anom[k] = res[k]["scores"]
    if anom:
        plot_score_distributions(
            res["normal"]["scores"], anom,
            title=f"{label} Score Distributions",
            save_path=str(out / f"{label.lower()}_scores.png"),
        )

logger.info("Smoke test complete. Check outputs/smoke_test/")
