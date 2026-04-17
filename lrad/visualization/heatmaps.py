"""Visualization utilities for LRAD anomaly heatmaps.

Produces publication-quality figures showing:
  - Original images
  - Per-decoder reconstructions with per-layer scores
  - Per-layer error heatmaps
  - Fused anomaly heatmaps with overlay and reference colorbar
  - Score distribution histograms with KDE overlay
  - ROC curves with AUROC annotations
  - Normal vs. anomaly reconstruction comparisons
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional

from ._style import (
    apply_style,
    attach_reference_colorbar,
    clean_image_axis,
    annotate_score,
    row_label,
    soft_grid,
    img_to_display,
    is_grayscale,
    ensure_dir,
    PALETTE, CYCLE, HEAT_CMAP, IMG_CMAP,
)


def plot_heatmap_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    per_layer_maps: list[np.ndarray],
    reconstructions: Optional[list[np.ndarray]] = None,
    scores: Optional[np.ndarray] = None,
    n_samples: int = 8,
    title: str = "Anomaly Heatmaps",
    save_path: Optional[str] = None,
    cmap_image: str = IMG_CMAP,
    cmap_heat: str = HEAT_CMAP,
) -> plt.Figure:
    """Grid of originals, per-decoder reconstructions + errors, fused map, overlay.

    Layout (rows, top → bottom):
        Original                       (score = fused max)
        Recon_0  → Error_0             (score = layer 0 max)
        Recon_1  → Error_1
        ...
        Recon_N  → Error_N
        Fused                          (= image-level anomaly score)
        Overlay (fused on original)

    A reference colorbar on the right shows the full error gradient [0, 1].
    Each tile is auto-scaled so low-error tiles remain readable; the
    colorbar is a palette reference, not a per-tile scale.
    """
    apply_style()
    n = min(n_samples, len(images))
    n_layers = len(per_layer_maps)
    has_recons = reconstructions is not None

    rows_per_layer = 2 if has_recons else 1
    n_rows = 1 + n_layers * rows_per_layer + 2

    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * 1.9, n_rows * 1.85),
        gridspec_kw=dict(hspace=0.18, wspace=0.06),
    )
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995,
                 color=PALETTE["text"])

    for col in range(n):
        img = images[col]
        img_show = img_to_display(img)
        gray = is_grayscale(img)

        # ---- Row 0: Original ----
        ax = axes[0, col]
        ax.imshow(img_show, cmap=cmap_image if gray else None)
        clean_image_axis(ax)
        if col == 0:
            row_label(ax, "Original")
        if scores is not None:
            annotate_score(ax, float(scores[col]))

        # ---- Recon + Error per decoder ----
        for l_idx in range(n_layers):
            hm = per_layer_maps[l_idx][col]
            if hm.ndim == 3:
                hm = hm[0]
            layer_score = float(hm.max())

            if has_recons:
                recon = img_to_display(reconstructions[l_idx][col])
                r_recon = 1 + l_idx * 2
                r_err = r_recon + 1
                ax_r = axes[r_recon, col]
                ax_r.imshow(recon, cmap=cmap_image if gray else None, vmin=0, vmax=1)
                clean_image_axis(ax_r)
                if col == 0:
                    row_label(ax_r, f"Recon L{l_idx}")
                annotate_score(ax_r, layer_score)
            else:
                r_err = 1 + l_idx

            ax_e = axes[r_err, col]
            ax_e.imshow(hm, cmap=cmap_heat)
            clean_image_axis(ax_e)
            if col == 0:
                row_label(ax_e, f"Error L{l_idx}")
            annotate_score(ax_e, layer_score)

        # ---- Fused ----
        r_fused = 1 + n_layers * rows_per_layer
        fused = heatmaps[col]
        ax_f = axes[r_fused, col]
        ax_f.imshow(fused, cmap=cmap_heat)
        clean_image_axis(ax_f)
        if col == 0:
            row_label(ax_f, "Fused")
        annotate_score(ax_f, float(fused.max()))

        # ---- Overlay ----
        r_overlay = r_fused + 1
        ax_o = axes[r_overlay, col]
        ax_o.imshow(img_show, cmap=cmap_image if gray else None)
        ax_o.imshow(fused, cmap=cmap_heat, alpha=0.55)
        clean_image_axis(ax_o)
        if col == 0:
            row_label(ax_o, "Overlay")

    # Reference colorbar (shared gradient legend)
    attach_reference_colorbar(
        fig, axes, cmap=cmap_heat, vmax=1.0,
        label="Squared reconstruction error  (per-tile auto-scaled; reference 0–1)",
    )

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)

    return fig


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def _auroc_numpy(normal: np.ndarray, anomaly: np.ndarray) -> float:
    """Compute AUROC via rank statistic (Mann–Whitney U). Pure numpy."""
    n_n, n_a = len(normal), len(anomaly)
    if n_n == 0 or n_a == 0:
        return float("nan")
    combined = np.concatenate([normal, anomaly])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks for ties
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_anom = ranks[n_n:].sum()
    u = r_anom - n_a * (n_a + 1) / 2
    return float(u / (n_n * n_a))


def plot_score_distributions(
    normal_scores: np.ndarray,
    anomaly_scores_dict: dict[str, np.ndarray],
    title: str = "Anomaly Score Distributions",
    save_path: Optional[str] = None,
    threshold_percentile: float = 95.0,
) -> plt.Figure:
    """Per-split histogram + KDE of anomaly scores with separation stats, AUROC,
    and an operating threshold line at the given percentile of the normal set.
    """
    apply_style()
    n_splits = len(anomaly_scores_dict)
    fig, axes = plt.subplots(
        1, n_splits,
        figsize=(5.6 * n_splits, 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]
    fig.suptitle(title, fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Shared operating threshold derived from the normal distribution only
    threshold = float(np.percentile(normal_scores, threshold_percentile))

    def _kde(x, grid, bw=None):
        bw = bw or (1.06 * x.std() * len(x) ** (-1 / 5))
        bw = max(bw, 1e-6)
        diff = (grid[:, None] - x[None, :]) / bw
        k = np.exp(-0.5 * diff ** 2) / np.sqrt(2 * np.pi)
        return k.sum(axis=1) / (len(x) * bw)

    for ax, (name, anomaly_scores) in zip(axes, anomaly_scores_dict.items()):
        all_vals = np.concatenate([normal_scores, anomaly_scores])
        bins = np.linspace(all_vals.min(), all_vals.max(), 55)

        ax.hist(normal_scores, bins=bins, alpha=0.55,
                color=PALETTE["normal"], edgecolor="white", linewidth=0.3,
                density=True, label=f"Normal  (n = {len(normal_scores)})", zorder=2)
        ax.hist(anomaly_scores, bins=bins, alpha=0.55,
                color=PALETTE["anomaly"], edgecolor="white", linewidth=0.3,
                density=True, label=f"Anomaly  (n = {len(anomaly_scores)})", zorder=2)

        grid = np.linspace(all_vals.min(), all_vals.max(), 400)
        ax.plot(grid, _kde(normal_scores, grid),
                color=PALETTE["normal"], linewidth=1.8, zorder=3)
        ax.plot(grid, _kde(anomaly_scores, grid),
                color=PALETTE["anomaly"], linewidth=1.8, zorder=3)

        # Operating threshold marker
        ax.axvline(threshold, color=PALETTE["text_muted"], linestyle="--",
                   linewidth=1.0, zorder=4,
                   label=f"τ at p{int(threshold_percentile)} normal = {threshold:.3f}")

        # Stats: separation, AUROC, detection rate at threshold
        sep = abs(anomaly_scores.mean() - normal_scores.mean()) / (
            0.5 * (normal_scores.std() + anomaly_scores.std() + 1e-9)
        )
        auroc = _auroc_numpy(normal_scores, anomaly_scores)
        tpr_at_t = float((anomaly_scores >= threshold).mean())
        fpr_at_t = float((normal_scores >= threshold).mean())

        stats_txt = (
            f"AUROC = {auroc:.3f}\n"
            f"separation  Δμ/σ̄ = {sep:.2f}\n"
            f"TPR @ τ = {tpr_at_t * 100:.1f}%\n"
            f"FPR @ τ = {fpr_at_t * 100:.1f}%"
        )
        ax.text(
            0.02, 0.97, stats_txt,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=PALETTE["text"], linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=PALETTE["grid"], linewidth=0.6, alpha=0.95),
        )

        ax.set_title(name, fontsize=11, color=PALETTE["text"])
        ax.set_xlabel("Anomaly score  (image-level, fused)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
                  facecolor="white", edgecolor=PALETTE["grid"])
        soft_grid(ax, axis="y")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    auroc_results: dict[str, dict],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """ROC per anomaly split with AUROC, chance diagonal, and the Youden-J
    operating point (max TPR − FPR) highlighted on each curve.
    """
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.0), constrained_layout=True)

    ax.plot([0, 1], [0, 1], color=PALETTE["text_muted"],
            linestyle="--", linewidth=0.9, alpha=0.6, zorder=1,
            label="Chance  (AUROC = 0.50)")

    for i, (name, roc) in enumerate(auroc_results.items()):
        color = CYCLE[i % len(CYCLE)]
        fpr = np.asarray(roc["fpr"])
        tpr = np.asarray(roc["tpr"])
        ax.plot(fpr, tpr,
                label=f"{name}  (AUROC = {roc['auroc']:.3f})",
                color=color, linewidth=2.0, zorder=3)
        ax.fill_between(fpr, tpr, 0, color=color, alpha=0.06, zorder=2)

        j_idx = int(np.argmax(tpr - fpr))
        ax.scatter([fpr[j_idx]], [tpr[j_idx]], color=color,
                   s=38, edgecolor="white", linewidth=1.0, zorder=5)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9, frameon=True,
              facecolor="white", edgecolor=PALETTE["grid"])
    ax.text(
        0.02, 0.02,
        "● Youden-J operating point  (max TPR − FPR)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.5, color=PALETTE["text_muted"],
    )
    soft_grid(ax, axis="both")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Reconstruction comparison (normal vs anomaly)
# ---------------------------------------------------------------------------

def plot_reconstruction_comparison(
    images: np.ndarray,
    normal_recons: list[np.ndarray],
    anomaly_recons: list[np.ndarray],
    n_samples: int = 6,
    title: str = "Reconstruction Comparison — Normal (left) vs Anomaly (right)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side per-decoder reconstructions for normal vs anomaly samples."""
    apply_style()
    n = min(n_samples, len(images))
    n_decoders = len(normal_recons)
    n_rows = n_decoders

    fig, axes = plt.subplots(
        n_rows, 2 * n,
        figsize=(2 * n * 1.6, n_rows * 1.6),
        gridspec_kw=dict(wspace=0.05, hspace=0.18),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=13, fontweight="bold", color=PALETTE["text"])

    for col in range(n):
        for r in range(n_decoders):
            nr = normal_recons[r][col]
            if nr.ndim == 3 and nr.shape[0] == 1:
                nr = nr[0]
            ax_n = axes[r, col]
            ax_n.imshow(nr, cmap=IMG_CMAP, vmin=0, vmax=1)
            clean_image_axis(ax_n, frame_color=PALETTE["normal"])

            ar = anomaly_recons[r][col]
            if ar.ndim == 3 and ar.shape[0] == 1:
                ar = ar[0]
            ax_a = axes[r, n + col]
            ax_a.imshow(ar, cmap=IMG_CMAP, vmin=0, vmax=1)
            clean_image_axis(ax_a, frame_color=PALETTE["anomaly"])

            if col == 0:
                row_label(ax_n, f"Dec {r}")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig
