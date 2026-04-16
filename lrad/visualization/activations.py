"""Inspection utilities: visualize what happens *before* heatmap fusion.

These plots answer questions like:
  - What features does the classifier learn at each depth?
  - How much of the activation is zero (ReLU sparsity)?
  - What does each individual decoder reconstruct, before they're combined?
  - Which layer contributes most to catching a given anomaly?
  - How are activation values distributed?
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _img_for_display(img: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) to a matplotlib-friendly array."""
    if img.ndim == 2:
        return img
    if img.shape[0] == 1:
        return img[0]
    if img.shape[0] == 3:
        rgb = np.transpose(img, (1, 2, 0))
        return np.clip(rgb, 0, 1)
    return img[0]


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


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
) -> plt.Figure:
    """Plot top-k most-active feature channels per layer for a single sample.

    For CNN activations (B, C, H, W): shows channel maps as 2D heatmaps.
    For MLP activations (B, D): reshapes vector to near-square grid and plots as 1 heatmap per layer.

    The visual mimics the classic low/mid/high-level feature diagram.
    """
    n_layers = len(activations)

    is_cnn = activations[0].ndim == 4

    if is_cnn:
        fig, axes = plt.subplots(n_layers, top_k, figsize=(top_k * 1.0, n_layers * 1.2))
        if n_layers == 1:
            axes = axes[np.newaxis, :]

        for l_idx, act in enumerate(activations):
            a = _to_numpy(act[sample_idx])  # (C, H, W)
            # Pick channels by mean activation magnitude (most "active")
            channel_strength = a.reshape(a.shape[0], -1).mean(axis=1)
            order = np.argsort(-np.abs(channel_strength))[:top_k]

            label = layer_labels[l_idx] if layer_labels else f"Layer {l_idx}"
            for col, ch in enumerate(order):
                ax = axes[l_idx, col]
                ax.imshow(a[ch], cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.set_title(f"ch{ch}", fontsize=6)

            # Pad remaining cells if fewer channels than top_k
            for col in range(len(order), top_k):
                axes[l_idx, col].axis("off")

    else:
        # MLP: one reshaped heatmap per layer
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 2.5, 2.5))
        if n_layers == 1:
            axes = [axes]

        for l_idx, act in enumerate(activations):
            a = _to_numpy(act[sample_idx])  # (D,)
            side = int(np.ceil(np.sqrt(a.shape[0])))
            padded = np.zeros(side * side)
            padded[: a.shape[0]] = a
            grid = padded.reshape(side, side)

            label = layer_labels[l_idx] if layer_labels else f"Layer {l_idx}"
            axes[l_idx].imshow(grid, cmap="viridis")
            axes[l_idx].set_title(f"{label}\n(dim={a.shape[0]})", fontsize=9)
            axes[l_idx].set_xticks([])
            axes[l_idx].set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 2. Activation statistics across layers
# ---------------------------------------------------------------------------

def compute_activation_stats(activations: list) -> dict:
    """Per-layer summary statistics.

    Returns dict with lists indexed by layer:
        mean, std, sparsity (% exact zeros), l2_norm,
        dead_channels (% channels with all-zero activations across the batch),
        max, min.
    """
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

        if a.ndim == 4:  # (B, C, H, W)
            per_channel = a.reshape(a.shape[0], a.shape[1], -1).sum(axis=(0, 2))
            dead = (per_channel == 0).mean()
        else:  # (B, D)
            per_unit = a.sum(axis=0)
            dead = (per_unit == 0).mean()
        stats["dead_channels"].append(float(dead))

    return stats


def plot_activation_stats(
    activations: list,
    title: str = "Per-Layer Activation Statistics",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-layer stats: mean, std, sparsity, L2 norm, dead channels."""
    stats = compute_activation_stats(activations)
    n_layers = len(activations)
    x = np.arange(n_layers)
    labels = [f"L{i}" for i in range(n_layers)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    panels = [
        ("mean", "Mean activation", axes[0, 0], "#0F6E56"),
        ("std", "Std. dev.", axes[0, 1], "#534AB7"),
        ("l2_norm", "Mean L2 norm", axes[0, 2], "#BA7517"),
        ("sparsity", "Sparsity (% zeros)", axes[1, 0], "#D85A30"),
        ("dead_channels", "Dead channels (%)", axes[1, 1], "#8B2E1F"),
        ("max", "Max activation", axes[1, 2], "#1F6FB4"),
    ]

    for key, title_txt, ax, color in panels:
        values = stats[key]
        if key in ("sparsity", "dead_channels"):
            values = [v * 100 for v in values]
        ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title_txt, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. Per-layer reconstructions (what each decoder produces, before fusion)
# ---------------------------------------------------------------------------

def plot_per_layer_reconstructions(
    images: np.ndarray,
    reconstructions: list,
    n_samples: int = 6,
    title: str = "Per-Decoder Reconstructions (before fusion)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare input image with each decoder's reconstruction.

    Rows: [Original | Decoder 0 recon | Decoder 1 recon | ... | Decoder N recon]
    Columns: samples.
    """
    images = _to_numpy(images)
    recons = [_to_numpy(r) for r in reconstructions]

    n = min(n_samples, len(images))
    n_decoders = len(recons)
    n_rows = 1 + n_decoders

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 1.8, n_rows * 1.8))
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for col in range(n):
        img = _img_for_display(images[col])
        axes[0, col].imshow(img, cmap="gray" if img.ndim == 2 else None)
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=9)

        for d_idx, r in enumerate(recons):
            ri = _img_for_display(r[col])
            axes[d_idx + 1, col].imshow(ri, cmap="gray" if ri.ndim == 2 else None, vmin=0, vmax=1)
            if col == 0:
                axes[d_idx + 1, col].set_ylabel(f"Dec. {d_idx}", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Per-layer error maps + per-layer image-level scores
# ---------------------------------------------------------------------------

def plot_per_layer_errors(
    images: np.ndarray,
    per_layer_maps: list,
    n_samples: int = 6,
    title: str = "Per-Layer Error Heatmaps (before fusion)",
    save_path: Optional[str] = None,
    cmap: str = "inferno",
) -> plt.Figure:
    """Show each decoder's error heatmap with its image-level score (max pixel).

    Lets you see *which layer catches which anomaly* before fusion collapses them.
    """
    images = _to_numpy(images)
    maps = [_to_numpy(m) for m in per_layer_maps]

    n = min(n_samples, len(images))
    n_layers = len(maps)
    n_rows = 1 + n_layers

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 1.8, n_rows * 1.8))
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for col in range(n):
        img = _img_for_display(images[col])
        axes[0, col].imshow(img, cmap="gray" if img.ndim == 2 else None)
        if col == 0:
            axes[0, col].set_ylabel("Original", fontsize=9)

        for l_idx, layer_maps in enumerate(maps):
            hm = layer_maps[col]
            if hm.ndim == 3:
                hm = hm[0]
            axes[l_idx + 1, col].imshow(hm, cmap=cmap, vmin=0)
            score = float(hm.max())
            axes[l_idx + 1, col].set_title(f"s={score:.3f}", fontsize=7)
            if col == 0:
                axes[l_idx + 1, col].set_ylabel(f"Layer {l_idx}", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 5. Activation value distributions (histograms)
# ---------------------------------------------------------------------------

def plot_activation_distributions(
    activations: list,
    bins: int = 60,
    log_y: bool = True,
    title: str = "Activation Value Distributions per Layer",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of activation values for each layer.

    With ReLU, most values pile up at 0 — the spike tells you how sparse the layer is.
    """
    n_layers = len(activations)
    n_cols = min(n_layers, 4)
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes_flat = np.atleast_1d(axes).flatten()

    for i, act in enumerate(activations):
        a = _to_numpy(act).flatten()
        ax = axes_flat[i]
        ax.hist(a, bins=bins, color="#0F6E56", alpha=0.8)
        ax.set_title(f"Layer {i}  (n={a.size:,})", fontsize=9)
        ax.set_xlabel("activation", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for j in range(len(activations), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

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
