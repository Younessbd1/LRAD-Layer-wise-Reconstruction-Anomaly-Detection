#!/usr/bin/env python3
"""Synthetic UQ demo - verifies MC Dropout and Ensemble pipelines."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")

from lrad.utils.helpers import get_device, seed_everything, setup_logging, count_parameters
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_uq import LRADModelUQ
from lrad.engine.uq_engine import evaluate_uq
from lrad.visualization.uq_plots import (
    plot_uq_heatmap_grid,
    plot_uq_score_comparison,
    plot_uncertainty_calibration,
)
from sklearn.metrics import roc_auc_score

seed_everything(42)
device = get_device()
out = Path("outputs/synthetic_uq")
out.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(out / "logs"))

# ═══════════ Synthetic data ═══════════
def make_shapes(n, size=28, shape="circle"):
    images = torch.zeros(n, 1, size, size)
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for i in range(n):
        cx, cy = size//2 + np.random.randint(-3, 4), size//2 + np.random.randint(-3, 4)
        if shape == "circle":
            r = np.random.randint(5, 9)
            mask = ((x - cx)**2 + (y - cy)**2) < r**2
        else:
            h = np.random.randint(4, 8)
            mask = (x >= cx-h) & (x < cx+h) & (y >= cy-h) & (y < cy+h)
        images[i, 0, mask] = 0.7 + np.random.rand() * 0.3
    return images

def make_anomalies(n, size=28):
    images = make_shapes(n, size, "circle")
    for i in range(n):
        ax, ay = np.random.choice([2, size-6]), np.random.choice([2, size-6])
        s = np.random.randint(3, 6)
        images[i, 0, ay:ay+s, ax:ax+s] = 1.0
    return images

train_imgs = torch.cat([make_shapes(300, shape="circle"), make_shapes(300, shape="square")])
train_lbls = torch.cat([torch.zeros(300), torch.ones(300)]).long()
test_normal = torch.cat([make_shapes(80, shape="circle"), make_shapes(80, shape="square")])
test_anomaly = make_anomalies(120)

from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=32, shuffle=True)
test_n_loader = DataLoader(TensorDataset(test_normal, torch.zeros(160).long()), batch_size=32)
test_a_loader = DataLoader(TensorDataset(test_anomaly, torch.ones(120).long()), batch_size=32)

# ═══════════ Train classifier ═══════════
logger.info("Training classifier...")
channels = [1, 8, 16, 32]
classifier = CNNClassifier(channels, num_classes=2, input_size=28).to(device)
opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

classifier.train()
for ep in range(10):
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        logits, _ = classifier(imgs)
        crit(logits, lbls).backward()
        opt.step()
classifier.freeze()

# ═══════════════════════════════════════════
#  Test 1: MC Dropout
# ═══════════════════════════════════════════
logger.info("=" * 50)
logger.info("MC DROPOUT (T=30)")
logger.info("=" * 50)

mc_model = LRADModelUQ(classifier, uq_method="mc_dropout", mc_samples=30, dropout_rate=0.25)
logger.info(f"Decoders: {len(mc_model.decoders)}")

# Train decoders
dec_crit = nn.MSELoss()
dec_opts = [torch.optim.Adam(d.parameters(), lr=1e-3) for d in mc_model.decoders]

for ep in range(15):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _, acts = mc_model.classifier(imgs)
        for i, (dec, dopt) in enumerate(zip(mc_model.decoders, dec_opts)):
            act = acts[mc_model.decoder_layers[i]].detach()
            dec.train()
            dopt.zero_grad()
            loss = dec_crit(dec(act), imgs)
            loss.backward()
            dopt.step()

# Evaluate
mc_normal = evaluate_uq(mc_model, test_n_loader, device, fusion="mean")
mc_anomaly = evaluate_uq(mc_model, test_a_loader, device, fusion="mean")

for score_type in ["error", "uncertainty", "combined"]:
    key = f"scores_{score_type}"
    labels = np.concatenate([np.zeros(len(mc_normal[key])), np.ones(len(mc_anomaly[key]))])
    scores = np.concatenate([mc_normal[key], mc_anomaly[key]])
    auroc = roc_auc_score(labels, scores)
    logger.info(f"  MC Dropout AUROC ({score_type}): {auroc:.4f}")

# Visualize
n = 8
plot_uq_heatmap_grid(
    mc_normal["images"][:n], mc_normal["error_maps"][:n],
    mc_normal["uncertainty_maps"][:n], mc_normal["combined_maps"][:n],
    scores_combined=mc_normal["scores_combined"][:n],
    title="MC Dropout - Normal",
    save_path=str(out / "mc_normal.png"),
)
plot_uq_heatmap_grid(
    mc_anomaly["images"][:n], mc_anomaly["error_maps"][:n],
    mc_anomaly["uncertainty_maps"][:n], mc_anomaly["combined_maps"][:n],
    scores_combined=mc_anomaly["scores_combined"][:n],
    title="MC Dropout - Anomaly",
    save_path=str(out / "mc_anomaly.png"),
)
plot_uq_score_comparison(
    mc_normal, mc_anomaly, split_name="MC Dropout",
    save_path=str(out / "mc_score_comparison.png"),
)
plot_uncertainty_calibration(
    mc_normal, mc_anomaly,
    save_path=str(out / "mc_calibration.png"),
)

# ═══════════════════════════════════════════
#  Test 2: Deep Ensemble
# ═══════════════════════════════════════════
logger.info("=" * 50)
logger.info("DEEP ENSEMBLE (M=5)")
logger.info("=" * 50)

ens_model = LRADModelUQ(classifier, uq_method="ensemble", n_ensemble=5)
logger.info(f"Decoders: {len(ens_model.decoders)}")

# Train each ensemble member independently
for ep in range(15):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            _, acts = ens_model.classifier(imgs)
        for i, dec in enumerate(ens_model.decoders):
            act = acts[ens_model.decoder_layers[i]].detach()
            for member in dec.members:
                member.train()
                member_opt = torch.optim.Adam(member.parameters(), lr=1e-3)
                member_opt.zero_grad()
                loss = dec_crit(member(act), imgs)
                loss.backward()
                member_opt.step()

# Evaluate
ens_normal = evaluate_uq(ens_model, test_n_loader, device, fusion="mean")
ens_anomaly = evaluate_uq(ens_model, test_a_loader, device, fusion="mean")

for score_type in ["error", "uncertainty", "combined"]:
    key = f"scores_{score_type}"
    labels = np.concatenate([np.zeros(len(ens_normal[key])), np.ones(len(ens_anomaly[key]))])
    scores = np.concatenate([ens_normal[key], ens_anomaly[key]])
    auroc = roc_auc_score(labels, scores)
    logger.info(f"  Ensemble AUROC ({score_type}): {auroc:.4f}")

plot_uq_heatmap_grid(
    ens_anomaly["images"][:n], ens_anomaly["error_maps"][:n],
    ens_anomaly["uncertainty_maps"][:n], ens_anomaly["combined_maps"][:n],
    scores_combined=ens_anomaly["scores_combined"][:n],
    title="Deep Ensemble - Anomaly",
    save_path=str(out / "ens_anomaly.png"),
)
plot_uncertainty_calibration(
    ens_normal, ens_anomaly,
    save_path=str(out / "ens_calibration.png"),
)

logger.info(f"All UQ outputs saved to {out}/")
logger.info("UQ demo complete.")
