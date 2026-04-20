"""Visualization utilities for LRAD-UQ - error, uncertainty, and combined maps."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_uq_heatmap_grid(
    images: np.ndarray,
    error_maps: np.ndarray,
    uncertainty_maps: np.ndarray,
    combined_maps: np.ndarray,
    scores_error: Optional[np.ndarray] = None,
    scores_uncertainty: Optional[np.ndarray] = None,
    scores_combined: Optional[np.ndarray] = None,
    n_samples: int = 8,
    title: str = "LRAD-UQ Anomaly Maps",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot: Original | Error map | Uncertainty map | Combined | Overlay.

    This is the key visualization for your internship presentation -
    it shows the three outputs side by side, demonstrating how uncertainty
    adds information beyond raw reconstruction error.
    """
    n = min(n_samples, len(images))
    n_rows = 5  # original, error, uncertainty, combined, overlay

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2, n_rows * 2))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Original", "Recon. error", "Epistemic unc.", "Combined", "Overlay"]

    for col in range(n):
        img = images[col]
        img_show = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0)).clip(0, 1)
        is_gray = img.shape[0] == 1

        # Row 0: Original
        axes[0, col].imshow(img_show, cmap="gray" if is_gray else None)
        if scores_combined is not None:
            axes[0, col].set_title(f"{scores_combined[col]:.3f}", fontsize=7)

        # Row 1: Reconstruction error
        axes[1, col].imshow(error_maps[col], cmap="inferno", vmin=0)

        # Row 2: Epistemic uncertainty
        axes[2, col].imshow(uncertainty_maps[col], cmap="viridis", vmin=0)

        # Row 3: Combined
        axes[3, col].imshow(combined_maps[col], cmap="magma", vmin=0)

        # Row 4: Overlay (combined on image)
        axes[4, col].imshow(img_show, cmap="gray" if is_gray else None)
        axes[4, col].imshow(combined_maps[col], cmap="magma", alpha=0.55, vmin=0)

        for r in range(n_rows):
            if col == 0:
                axes[r, col].set_ylabel(row_labels[r], fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_uq_score_comparison(
    normal_results: dict,
    anomaly_results: dict,
    split_name: str = "anomaly",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare AUROC of error-only vs uncertainty vs combined scoring.

    Three side-by-side histograms showing how each scoring strategy
    separates normal from anomaly distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Score Comparison - {split_name}", fontsize=13, fontweight="bold")

    score_types = [
        ("scores_error", "Reconstruction Error", "#0F6E56"),
        ("scores_uncertainty", "Epistemic Uncertainty", "#534AB7"),
        ("scores_combined", "Combined (E × U)", "#D85A30"),
    ]

    for ax, (key, label, color) in zip(axes, score_types):
        n_scores = normal_results[key]
        a_scores = anomaly_results[key]

        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(n_scores)), np.ones(len(a_scores))])
        all_s = np.concatenate([n_scores, a_scores])
        auroc = roc_auc_score(labels, all_s)

        ax.hist(n_scores, bins=40, alpha=0.6, label="Normal", color="#0F6E56", density=True)
        ax.hist(a_scores, bins=40, alpha=0.6, label="Anomaly", color=color, density=True)
        ax.set_title(f"{label}\nAUROC = {auroc:.3f}", fontsize=10)
        ax.set_xlabel("Score")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_uncertainty_calibration(
    normal_results: dict,
    anomaly_results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter: reconstruction error vs epistemic uncertainty.

    Normal samples should cluster at (low error, low uncertainty).
    Anomalies should appear at (high error, high/variable uncertainty).
    This plot directly demonstrates the value of UQ.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.scatter(
        normal_results["scores_error"],
        normal_results["scores_uncertainty"],
        alpha=0.4, s=15, c="#0F6E56", label="Normal", edgecolors="none",
    )
    ax.scatter(
        anomaly_results["scores_error"],
        anomaly_results["scores_uncertainty"],
        alpha=0.4, s=15, c="#D85A30", label="Anomaly", edgecolors="none",
    )

    ax.set_xlabel("Reconstruction Error (max pixel)", fontsize=11)
    ax.set_ylabel("Epistemic Uncertainty (max pixel)", fontsize=11)
    ax.set_title("Error vs Uncertainty - Normal vs Anomaly", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
