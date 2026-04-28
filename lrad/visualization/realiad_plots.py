"""Real-IAD specific plots: training curves, PR curves, feature scatter,
reconstruction gallery, and class distribution.

All share the same look-and-feel as ``heatmaps.py`` and ``activations.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from ._style import (
    apply_style,
    clean_image_axis,
    ensure_dir,
    img_to_display,
    is_grayscale,
    soft_grid,
    PALETTE, CYCLE, HEAT_CMAP, IMG_CMAP,
)


# ---------------------------------------------------------------------------
# Class distribution (per category, OK vs NG bars)
# ---------------------------------------------------------------------------

def plot_class_distribution(
    ok_per_cat: dict[str, int],
    ng_per_cat: dict[str, int],
    title: str = "Class distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    apply_style()
    cats = sorted(set(ok_per_cat) | set(ng_per_cat))
    ok = np.asarray([ok_per_cat.get(c, 0) for c in cats], dtype=float)
    ng = np.asarray([ng_per_cat.get(c, 0) for c in cats], dtype=float)
    x = np.arange(len(cats))
    w = 0.4

    fig_h = max(4.0, len(cats) * 0.25 + 2.0)
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.4), fig_h),
                           constrained_layout=True)
    ax.bar(x - w / 2, ok, w, color=PALETTE["normal"], label="OK", zorder=3)
    ax.bar(x + w / 2, ng, w, color=PALETTE["anomaly"], label="NG", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Sample count")
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["text"])
    ax.legend(loc="upper right", frameon=False)
    soft_grid(ax, axis="y")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Training curves (classifier loss/acc + per-decoder loss)
# ---------------------------------------------------------------------------

def plot_training_curves(
    cls_history: Optional[dict],
    dec_history: Optional[dict],
    title: str = "Training curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    apply_style()
    has_cls = cls_history is not None
    has_dec = dec_history is not None
    if not (has_cls or has_dec):
        raise ValueError("nothing to plot — both histories are None")

    n_panels = (2 if has_cls else 0) + (2 if has_dec else 0)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.0 * n_panels, 4.2),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    axes = list(np.atleast_1d(axes))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=PALETTE["text"])
    cur = 0

    if has_cls:
        ax = axes[cur]; cur += 1
        ep = np.arange(1, len(cls_history["train_loss"]) + 1)
        ax.plot(ep, cls_history["train_loss"], color=PALETTE["normal"],
                linewidth=1.6, label="train")
        if cls_history.get("val_loss"):
            ax.plot(ep, cls_history["val_loss"], color=PALETTE["anomaly"],
                    linewidth=1.6, label="val")
        ax.set_title("Classifier loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy")
        ax.legend(loc="upper right", frameon=False)
        soft_grid(ax)

        ax = axes[cur]; cur += 1
        ax.plot(ep, cls_history["train_acc"], color=PALETTE["normal"],
                linewidth=1.6, label="train")
        if cls_history.get("val_acc"):
            ax.plot(ep, cls_history["val_acc"], color=PALETTE["anomaly"],
                    linewidth=1.6, label="val")
        ax.set_title("Classifier accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", frameon=False)
        soft_grid(ax)

    if has_dec:
        for split, name in [("train_loss", "Decoder train loss"),
                            ("val_loss",   "Decoder val loss")]:
            ax = axes[cur]; cur += 1
            n_dec = len(dec_history[split])
            ep = np.arange(1, len(dec_history[split][0]) + 1)
            for k in range(n_dec):
                ax.plot(ep, dec_history[split][k],
                        color=CYCLE[k % len(CYCLE)],
                        linewidth=1.6, label=f"D{k}")
            ax.set_title(name)
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
            soft_grid(ax)

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# PR curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    pr_results: dict[str, dict],
    title: str = "Precision-Recall",
    save_path: Optional[str] = None,
) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)

    for i, (name, pr) in enumerate(pr_results.items()):
        precisions = np.asarray(pr["precisions"])
        recalls = np.asarray(pr["recalls"])
        color = CYCLE[i % len(CYCLE)]
        ax.plot(recalls, precisions,
                label=f"{name}  (AP={pr['pr_auc']:.3f}, F1*={pr.get('f1', float('nan')):.3f})",
                color=color, linewidth=2.0)
        ax.fill_between(recalls, precisions, 0, color=color, alpha=0.06)

    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True,
              facecolor="white", edgecolor=PALETTE["grid"])
    soft_grid(ax, axis="both")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Reconstruction gallery
# ---------------------------------------------------------------------------

def plot_reconstruction_gallery(
    images: np.ndarray,
    reconstructions: list[np.ndarray],
    heatmaps: Optional[np.ndarray] = None,
    titles: Optional[list[str]] = None,
    n: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compact gallery: input | best reconstruction | overlay, repeated."""
    apply_style()
    n = min(n, len(images))
    has_heat = heatmaps is not None

    # "Best" recon = mean across decoders (visually closer to ground truth).
    recon_stack = np.stack(reconstructions, axis=0)
    best_recon = recon_stack.mean(axis=0)

    rows = 3 if has_heat else 2
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.7, rows * 1.7),
                             gridspec_kw=dict(hspace=0.18, wspace=0.06))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Reconstruction gallery", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])

    for col in range(n):
        gray = is_grayscale(images[col])
        ax = axes[0, col]
        ax.imshow(img_to_display(images[col]),
                  cmap=IMG_CMAP if gray else None)
        clean_image_axis(ax)
        if titles and col < len(titles):
            ax.set_title(titles[col], fontsize=8, color=PALETTE["text_muted"])

        ax = axes[1, col]
        ax.imshow(img_to_display(best_recon[col]), vmin=0, vmax=1,
                  cmap=IMG_CMAP if gray else None)
        clean_image_axis(ax)

        if has_heat:
            ax = axes[2, col]
            ax.imshow(img_to_display(images[col]),
                      cmap=IMG_CMAP if gray else None)
            ax.imshow(heatmaps[col], cmap=HEAT_CMAP, alpha=0.55,
                      vmin=0, vmax=max(1e-6, heatmaps.max()))
            clean_image_axis(ax)

    for r, lbl in enumerate(["Input", "Mean recon", "Overlay"][: rows]):
        axes[r, 0].set_ylabel(lbl, fontsize=9)

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Feature-space scatter (PCA-2D)
# ---------------------------------------------------------------------------

def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    # SVD on centered data — robust on tall (N >> D) and wide (D >> N).
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return (u[:, :2] * s[:2])


def plot_feature_scatter(
    feats_normal: np.ndarray,
    feats_anomaly: np.ndarray,
    title: str = "Feature scatter (PCA)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    apply_style()
    all_feat = np.vstack([feats_normal, feats_anomaly])
    proj = _pca_2d(all_feat)
    n_n = len(feats_normal)
    p_n = proj[:n_n]
    p_a = proj[n_n:]

    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    ax.scatter(p_n[:, 0], p_n[:, 1], s=14, color=PALETTE["normal"],
               alpha=0.65, edgecolor="white", linewidth=0.3,
               label=f"normal  (n={n_n})", zorder=3)
    ax.scatter(p_a[:, 0], p_a[:, 1], s=14, color=PALETTE["anomaly"],
               alpha=0.65, edgecolor="white", linewidth=0.3,
               label=f"anomaly  (n={len(feats_anomaly)})", zorder=4)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["text"])
    ax.legend(loc="upper right", frameon=False)
    soft_grid(ax, axis="both")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig
