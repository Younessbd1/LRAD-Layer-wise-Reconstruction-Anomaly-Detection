"""Visualization utilities for LRAD anomaly heatmaps.

Produces publication-quality figures showing:
  - Original images
  - Per-layer reconstructions
  - Per-layer error heatmaps
  - Fused anomaly heatmaps with overlay
  - Score distribution histograms
  - ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional


def plot_heatmap_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    per_layer_maps: list[np.ndarray],
    reconstructions: Optional[list[np.ndarray]] = None,
    scores: Optional[np.ndarray] = None,
    n_samples: int = 8,
    title: str = "Anomaly Heatmaps",
    save_path: Optional[str] = None,
    cmap_image: str = "gray",
    cmap_heat: str = "inferno",
) -> plt.Figure:
    """Plot a grid of images, per-layer heatmaps, and fused heatmaps.

    Rows: [Original | Layer_0 heatmap | ... | Layer_N heatmap | Fused | Overlay]
    Columns: individual samples.
    """
    n = min(n_samples, len(images))
    n_layers = len(per_layer_maps)
    n_rows = 2 + n_layers + 1  # original + per-layer + fused + overlay

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 1.8, n_rows * 1.8))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if n == 1:
        axes = axes[:, np.newaxis]

    for col in range(n):
        img = images[col]
        if img.shape[0] == 1:
            img_show = img[0]
        elif img.shape[0] == 3:
            img_show = np.transpose(img, (1, 2, 0))
            img_show = np.clip(img_show, 0, 1)
        else:
            img_show = img[0]

        # Row 0: Original
        axes[0, col].imshow(img_show, cmap=cmap_image if img.shape[0] == 1 else None)
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=9)
        if scores is not None:
            axes[0, col].set_title(f"s={scores[col]:.3f}", fontsize=7)

        # Rows 1..N: Per-layer heatmaps
        for r, layer_maps in enumerate(per_layer_maps):
            hm = layer_maps[col]
            axes[r + 1, col].imshow(hm, cmap=cmap_heat, vmin=0)
            if col == 0:
                axes[r + 1, col].set_ylabel(f"Layer {r}", fontsize=9)

        # Row N+1: Fused heatmap
        fused = heatmaps[col]
        axes[n_layers + 1, col].imshow(fused, cmap=cmap_heat, vmin=0)
        if col == 0:
            axes[n_layers + 1, col].set_ylabel("Fused", fontsize=9)

        # Row N+2: Overlay
        axes[n_layers + 2, col].imshow(img_show, cmap=cmap_image if img.shape[0] == 1 else None)
        axes[n_layers + 2, col].imshow(fused, cmap=cmap_heat, alpha=0.5, vmin=0)
        if col == 0:
            axes[n_layers + 2, col].set_ylabel("Overlay", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_score_distributions(
    normal_scores: np.ndarray,
    anomaly_scores_dict: dict[str, np.ndarray],
    title: str = "Anomaly Score Distributions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot histogram of anomaly scores for normal vs each anomaly split."""
    n_splits = len(anomaly_scores_dict)
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 4))
    if n_splits == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (name, anomaly_scores) in zip(axes, anomaly_scores_dict.items()):
        ax.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="#0F6E56", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label=f"Anomaly ({name})", color="#D85A30", density=True)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_roc_curves(
    auroc_results: dict[str, dict],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curves for each anomaly split.

    Args:
        auroc_results: dict mapping split_name → {'fpr': ..., 'tpr': ..., 'auroc': ...}
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    colors = ["#0F6E56", "#D85A30", "#534AB7", "#BA7517"]

    for i, (name, roc) in enumerate(auroc_results.items()):
        color = colors[i % len(colors)]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUROC={roc['auroc']:.3f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_reconstruction_comparison(
    images: np.ndarray,
    normal_recons: list[np.ndarray],
    anomaly_recons: list[np.ndarray],
    n_samples: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison: normal vs anomaly reconstruction quality."""
    n = min(n_samples, len(images))
    n_decoders = len(normal_recons)
    n_rows = 1 + n_decoders

    fig, axes = plt.subplots(n_rows, 2 * n, figsize=(2 * n * 1.5, n_rows * 1.5))
    fig.suptitle("Normal (left) vs Anomaly (right) Reconstructions", fontsize=12)

    for col in range(n):
        for r in range(n_decoders):
            # Normal reconstruction
            nr = normal_recons[r][col]
            if nr.ndim == 3 and nr.shape[0] == 1:
                nr = nr[0]
            axes[r + 1, col].imshow(nr, cmap="gray", vmin=0, vmax=1)
            if col == 0:
                axes[r + 1, col].set_ylabel(f"Dec. {r}", fontsize=8)

            # Anomaly reconstruction
            ar = anomaly_recons[r][col]
            if ar.ndim == 3 and ar.shape[0] == 1:
                ar = ar[0]
            axes[r + 1, n + col].imshow(ar, cmap="gray", vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
