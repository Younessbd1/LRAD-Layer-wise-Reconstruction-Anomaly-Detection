"""Inspection utilities: visualize what happens *before* heatmap fusion.

These plots answer questions like:
  - What features does the classifier learn at each depth?
  - How sparse is each layer? How many channels are dead?
  - What does each individual decoder reconstruct, before they're combined?
  - Which layer contributes most to catching a given anomaly?
  - How are activation values distributed?

All plots share a consistent publication-quality style defined in ``_style``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Feature maps (Zeiler/Fergus-style grid)
# ---------------------------------------------------------------------------

def plot_feature_maps(
    activations: list,
    sample_idx: int = 0,
    top_k: int = 16,
    layer_labels: Optional[list[str]] = None,
    title: str = "Feature Activations Across Layers",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Top-k most-active feature channels per layer, for a single sample.

    CNN activations (B, C, H, W) -> channel maps as 2D heatmaps.
    MLP activations (B, D)        -> vector reshaped to near-square grid.

    A viridis reference colorbar on the right indicates the low-to-high
    activation gradient. Channels are sorted left-to-right by mean activation
    magnitude, so column 0 is the strongest for this specific input.
    """
    apply_style()
    n_layers = len(activations)
    is_cnn = activations[0].ndim == 4

    if is_cnn:
        fig, axes = plt.subplots(
            n_layers, top_k,
            figsize=(top_k * 1.05 + 1.2, n_layers * 1.32 + 0.8),
            gridspec_kw=dict(wspace=0.10, hspace=0.42),
        )
        if n_layers == 1:
            axes = axes[np.newaxis, :]

        for l_idx, act in enumerate(activations):
            a = _to_numpy(act[sample_idx])
            channel_strength = a.reshape(a.shape[0], -1).mean(axis=1)
            order = np.argsort(-np.abs(channel_strength))[:top_k]

            base = layer_labels[l_idx] if layer_labels else f"Layer {l_idx}"
            spatial = f"{a.shape[1]}×{a.shape[2]}"
            label = f"{base}\n{spatial} · {a.shape[0]}ch"
            for col, ch in enumerate(order):
                ax = axes[l_idx, col]
                ax.imshow(a[ch], cmap=cmap)
                clean_image_axis(ax)
                if col == 0:
                    row_label(ax, label, fontsize=8.5)
                ax.set_title(f"ch{ch}  ({channel_strength[ch]:.2f})",
                             fontsize=6.5, color=PALETTE["text_muted"], pad=3)

            for col in range(len(order), top_k):
                axes[l_idx, col].axis("off")

        attach_reference_colorbar(
            fig, axes, cmap=cmap, vmax=1.0,
            label="Activation intensity",
            endpoint_labels=("silent", "firing"),
        )

    else:
        fig, axes = plt.subplots(
            1, n_layers,
            figsize=(n_layers * 3.0 + 1.2, 3.0),
            gridspec_kw=dict(wspace=0.22),
        )
        if n_layers == 1:
            axes = [axes]

        for l_idx, act in enumerate(activations):
            a = _to_numpy(act[sample_idx])
            side = int(np.ceil(np.sqrt(a.shape[0])))
            padded = np.zeros(side * side)
            padded[: a.shape[0]] = a
            grid = padded.reshape(side, side)

            label = layer_labels[l_idx] if layer_labels else f"Layer {l_idx}"
            ax = axes[l_idx]
            ax.imshow(grid, cmap=cmap)
            clean_image_axis(ax)
            ax.set_title(f"{label}  (dim = {a.shape[0]})", fontsize=10)

        attach_reference_colorbar(
            fig, axes, cmap=cmap, vmax=1.0,
            label="Activation intensity",
            endpoint_labels=("silent", "firing"),
        )

    fig.suptitle(
        f"{title}\nSample #{sample_idx}  ·  channels sorted by mean activation (strongest -> weakest)",
        fontsize=12, fontweight="bold", color=PALETTE["text"], y=1.00,
    )

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Activation statistics across layers
# ---------------------------------------------------------------------------

def compute_activation_stats(activations: list) -> dict:
    """Per-layer summary statistics."""
    stats = {"mean": [], "std": [], "sparsity": [], "l2_norm": [],
             "dead_channels": [], "max": [], "min": []}

    for act in activations:
        a = _to_numpy(act)
        stats["mean"].append(float(a.mean()))
        stats["std"].append(float(a.std()))
        stats["sparsity"].append(float((a == 0).mean()))
        stats["l2_norm"].append(float(np.linalg.norm(a.reshape(a.shape[0], -1), axis=1).mean()))
        stats["max"].append(float(a.max()))
        stats["min"].append(float(a.min()))
        if a.ndim == 4:
            per_channel = a.reshape(a.shape[0], a.shape[1], -1).sum(axis=(0, 2))
            dead = (per_channel == 0).mean()
        else:
            per_unit = a.sum(axis=0)
            dead = (per_unit == 0).mean()
        stats["dead_channels"].append(float(dead))

    return stats


def plot_activation_stats(
    activations: list,
    title: str = "Per-Layer Activation Statistics",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """6-panel bar chart: mean, std, L2, sparsity, dead channels, max."""
    apply_style()
    stats = compute_activation_stats(activations)
    n_layers = len(activations)
    x = np.arange(n_layers)
    labels = [f"L{i}" for i in range(n_layers)]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 7),
        gridspec_kw=dict(hspace=0.38, wspace=0.28),
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", color=PALETTE["text"])

    panels = [
        ("mean",          "Mean activation",         axes[0, 0], PALETTE["normal"],       ""),
        ("std",           "Std. deviation",          axes[0, 1], PALETTE["accent"],       ""),
        ("l2_norm",       "Mean L2 norm",            axes[0, 2], PALETTE["neutral"],      ""),
        ("sparsity",      "Sparsity  (% zeros)",     axes[1, 0], PALETTE["anomaly"],      "%"),
        ("dead_channels", "Dead channels  (%)",      axes[1, 1], PALETTE["anomaly_2"],    "%"),
        ("max",           "Max activation",          axes[1, 2], PALETTE["accent_2"],     ""),
    ]

    for key, title_txt, ax, color, suffix in panels:
        values = stats[key]
        if key in ("sparsity", "dead_channels"):
            values = [v * 100 for v in values]

        bars = ax.bar(x, values, color=color, width=0.7,
                      edgecolor="white", linewidth=0.6, zorder=3)

        # value labels above bars
        vmax = max(values) if max(values) > 0 else 1
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.02,
                    f"{v:.2f}{suffix}", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title_txt, fontsize=10.5)
        soft_grid(ax, axis="y")
        ax.set_ylim(0 if min(values) >= 0 else None,
                    vmax * 1.18 if max(values) > 0 else 1)

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Per-layer reconstructions
# ---------------------------------------------------------------------------

def plot_per_layer_reconstructions(
    images: np.ndarray,
    reconstructions: list,
    n_samples: int = 6,
    per_layer_maps: Optional[list] = None,
    title: str = "Per-Decoder Reconstructions (before fusion)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Input vs. each decoder's reconstruction. Scores overlaid when maps given."""
    apply_style()
    images = _to_numpy(images)
    recons = [_to_numpy(r) for r in reconstructions]
    maps = [_to_numpy(m) for m in per_layer_maps] if per_layer_maps is not None else None

    n = min(n_samples, len(images))
    n_decoders = len(recons)
    n_rows = 1 + n_decoders

    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * 1.9, n_rows * 1.9),
        gridspec_kw=dict(hspace=0.18, wspace=0.06),
    )
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02, color=PALETTE["text"])

    for col in range(n):
        img = img_to_display(images[col])
        gray = is_grayscale(images[col])
        ax0 = axes[0, col]
        ax0.imshow(img, cmap=IMG_CMAP if gray else None)
        clean_image_axis(ax0)
        if col == 0:
            row_label(ax0, "Original")

        for d_idx, r in enumerate(recons):
            ri = img_to_display(r[col])
            ax = axes[d_idx + 1, col]
            ax.imshow(ri, cmap=IMG_CMAP if gray else None, vmin=0, vmax=1)
            clean_image_axis(ax)
            if col == 0:
                row_label(ax, f"Dec {d_idx}")
            if maps is not None:
                hm = maps[d_idx][col]
                if hm.ndim == 3:
                    hm = hm[0]
                annotate_score(ax, float(hm.max()))

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Per-layer error heatmaps (with optional reconstruction alongside)
# ---------------------------------------------------------------------------

def plot_per_layer_errors(
    images: np.ndarray,
    per_layer_maps: list,
    reconstructions: Optional[list] = None,
    n_samples: int = 6,
    title: str = "Per-Layer Reconstructions & Error Heatmaps  (before fusion)",
    save_path: Optional[str] = None,
    cmap: str = HEAT_CMAP,
) -> plt.Figure:
    """Show each decoder's reconstruction and its error heatmap side-by-side.

    Reference colorbar on the right shows the full inferno gradient [0, 1]
    (theoretical squared-error range for inputs in [0, 1]).
    Each tile is auto-scaled for readability; the score badge `s=` shows the
    true per-layer image-level anomaly score.
    """
    apply_style()
    images = _to_numpy(images)
    maps = [_to_numpy(m) for m in per_layer_maps]
    recons = [_to_numpy(r) for r in reconstructions] if reconstructions is not None else None

    n = min(n_samples, len(images))
    n_layers = len(maps)
    has_recons = recons is not None
    rows_per_layer = 2 if has_recons else 1
    n_rows = 1 + n_layers * rows_per_layer

    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(n * 1.9, n_rows * 1.9),
        gridspec_kw=dict(hspace=0.18, wspace=0.06),
    )
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995, color=PALETTE["text"])

    for col in range(n):
        img = img_to_display(images[col])
        gray = is_grayscale(images[col])
        ax0 = axes[0, col]
        ax0.imshow(img, cmap=IMG_CMAP if gray else None)
        clean_image_axis(ax0)
        if col == 0:
            row_label(ax0, "Original")

        for l_idx, layer_maps in enumerate(maps):
            hm = layer_maps[col]
            if hm.ndim == 3:
                hm = hm[0]
            layer_score = float(hm.max())

            if has_recons:
                recon = img_to_display(recons[l_idx][col])
                r_recon = 1 + l_idx * 2
                r_err = r_recon + 1
                ax_r = axes[r_recon, col]
                ax_r.imshow(recon, cmap=IMG_CMAP if gray else None, vmin=0, vmax=1)
                clean_image_axis(ax_r)
                if col == 0:
                    row_label(ax_r, f"Recon L{l_idx}")
                annotate_score(ax_r, layer_score)
            else:
                r_err = 1 + l_idx

            ax_e = axes[r_err, col]
            ax_e.imshow(hm, vmin=0, vmax=1, cmap=cmap)
            clean_image_axis(ax_e)
            if col == 0:
                row_label(ax_e, f"Error L{l_idx}")
            annotate_score(ax_e, layer_score)

    attach_reference_colorbar(
        fig, axes, cmap=cmap, vmax=1.0,
        label="Squared reconstruction error  (absolute scale 0-1)",
    )

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Activation value distributions
# ---------------------------------------------------------------------------

def plot_activation_distributions(
    activations: list,
    bins: int = 60,
    log_y: bool = True,
    title: str = "Activation Value Distributions per Layer",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of activation values per layer with a sparsity / moments callout.

    On a log-y axis the zero bin is typically the tallest (ReLU sparsity), so
    the stat callout is anchored to the upper-right where the distribution
    has already decayed and cannot collide with the bars.
    """
    apply_style()
    n_layers = len(activations)
    n_cols = min(n_layers, 4)
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.0, n_rows * 3.2),
        constrained_layout=True,
    )
    suptitle = title + ("  ·  y-axis: log scale" if log_y else "")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=PALETTE["text"])
    axes_flat = np.atleast_1d(axes).flatten()

    for i, act in enumerate(activations):
        a = _to_numpy(act).flatten()
        ax = axes_flat[i]
        ax.hist(a, bins=bins, color=PALETTE["accent_2"],
                edgecolor="white", linewidth=0.3, zorder=3)

        sparsity = float((a == 0).mean())
        mean = float(a.mean())
        std = float(a.std())
        amax = float(a.max())

        ax.axvline(0, color=PALETTE["text_muted"], linestyle=":",
                   linewidth=0.8, zorder=2, label="zero")
        ax.axvline(mean, color=PALETTE["anomaly"], linestyle="--",
                   linewidth=0.9, zorder=4, label=f"mean = {mean:.2f}")

        stats_txt = (
            f"sparsity  {sparsity * 100:5.1f}%\n"
            f"mean      {mean:+.3f}\n"
            f"std       {std:.3f}\n"
            f"max       {amax:.3f}"
        )
        ax.text(
            0.98, 0.97, stats_txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=PALETTE["text"], family="monospace",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=PALETTE["grid"], linewidth=0.6, alpha=0.95),
        )
        ax.set_title(f"Layer {i}   (n = {a.size:,})", fontsize=10.5)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count")
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
        soft_grid(ax, axis="y")

    for j in range(len(activations), len(axes_flat)):
        axes_flat[j].axis("off")

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def format_stats_report(
    activations: list,
    per_layer_maps: Optional[list] = None,
) -> str:
    """Human-readable per-layer summary."""
    stats = compute_activation_stats(activations)
    lines = ["Per-layer activation report", "=" * 50]

    for i in range(len(activations)):
        a = _to_numpy(activations[i])
        shape = tuple(a.shape)
        lines.append(f"Layer {i}  shape={shape}")
        lines.append(f"  mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}  "
                     f"min={stats['min'][i]:+.4f}  max={stats['max'][i]:+.4f}")
        lines.append(f"  sparsity={stats['sparsity'][i] * 100:.1f}%  "
                     f"dead_channels={stats['dead_channels'][i] * 100:.1f}%  "
                     f"mean_L2={stats['l2_norm'][i]:.3f}")

        if per_layer_maps is not None and i < len(per_layer_maps):
            m = _to_numpy(per_layer_maps[i])
            mse_mean = float(m.mean())
            score_mean = float(m.reshape(m.shape[0], -1).max(axis=1).mean())
            lines.append(f"  recon MSE (mean pixel)={mse_mean:.5f}  "
                         f"mean image score={score_mean:.5f}")
        lines.append("")

    return "\n".join(lines)
