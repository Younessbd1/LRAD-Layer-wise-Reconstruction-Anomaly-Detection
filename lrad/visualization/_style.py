"""Shared visual style: a single consistent aesthetic across every LRAD plot.

Typography, spacing, palette, and reference colorbars are defined here so
every figure in the project looks like it came from the same report.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Optional


# ---------------------------------------------------------------------------
# Palette — muted, print-safe, colorblind-friendly
# ---------------------------------------------------------------------------

PALETTE = {
    "normal":       "#0F6E56",   # deep teal
    "anomaly":      "#D85A30",   # burnt orange
    "anomaly_2":    "#8B2E1F",   # dark brick
    "accent":       "#534AB7",   # violet
    "accent_2":     "#1F6FB4",   # steel blue
    "neutral":      "#BA7517",   # ochre
    "bar_primary":  "#2D5F7A",
    "bar_secondary":"#7F3C48",
    "bar_tertiary": "#5B7F3C",
    "grid":         "#D8D8D8",
    "text":         "#1A1A1A",
    "text_muted":   "#5A5A5A",
}

CYCLE = [
    PALETTE["normal"], PALETTE["anomaly"], PALETTE["accent"],
    PALETTE["accent_2"], PALETTE["neutral"], PALETTE["anomaly_2"],
]

HEAT_CMAP = "inferno"
IMG_CMAP = "gray"


# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------

_APPLIED = False


def apply_style() -> None:
    """Install consistent matplotlib rcParams. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    plt.rcParams.update({
        # Figure
        "figure.facecolor":   "white",
        "figure.dpi":         110,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "white",

        # Axes
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#333333",
        "axes.linewidth":     0.8,
        "axes.titlesize":     11,
        "axes.titleweight":   "semibold",
        "axes.titlepad":      6,
        "axes.labelsize":     10,
        "axes.labelcolor":    PALETTE["text"],
        "axes.labelweight":   "regular",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,      # opt-in per plot

        # Ticks
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "xtick.color":        PALETTE["text_muted"],
        "ytick.color":        PALETTE["text_muted"],
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3,
        "ytick.major.size":   3,

        # Legend
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "legend.title_fontsize": 9,

        # Grid
        "grid.color":         PALETTE["grid"],
        "grid.linewidth":     0.5,
        "grid.alpha":         0.6,

        # Font
        "font.family":        "DejaVu Sans",
        "font.size":          10,

        # Color cycle
        "axes.prop_cycle":    plt.cycler(color=CYCLE),
    })
    _APPLIED = True


# ---------------------------------------------------------------------------
# Helpers — colorbar, axis polish, tile frames
# ---------------------------------------------------------------------------

def attach_reference_colorbar(
    fig: plt.Figure,
    axes,
    cmap: str = HEAT_CMAP,
    vmax: float = 1.0,
    label: str = "Squared reconstruction error",
    endpoint_labels: Optional[tuple[str, str]] = None,
) -> plt.colorbar.Colorbar:
    """Attach a reference colorbar showing the full heatmap gradient.

    Each tile is auto-scaled to its own range; the reference bar shows the
    cmap mapped onto the theoretical [0, vmax] range so the gradient itself
    is visible. `endpoint_labels` overrides the numeric ticks with a (low, high)
    text pair — useful when the gradient is purely qualitative.
    """
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
    sm.set_array([])
    if hasattr(axes, "ravel"):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    cbar = fig.colorbar(
        sm, ax=ax_list,
        fraction=0.020, pad=0.025, shrink=0.85, aspect=30,
    )
    cbar.set_label(label, fontsize=9, color=PALETTE["text"], labelpad=6)
    cbar.ax.tick_params(labelsize=8, color=PALETTE["text_muted"])
    cbar.outline.set_edgecolor("#555555")
    cbar.outline.set_linewidth(0.7)

    if endpoint_labels is not None:
        low, high = endpoint_labels
        cbar.set_ticks([0.0, vmax])
        cbar.set_ticklabels([low, high])
    return cbar


def clean_image_axis(ax: plt.Axes, frame_color: str = "#444444") -> None:
    """Axis suitable for an image tile: no ticks, thin border frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(0.6)


def annotate_score(ax: plt.Axes, score: float, fontsize: int = 7,
                   color: str = "white", bg: str = "#1A1A1A") -> None:
    """Place a score badge inside the top-left of an image tile."""
    ax.text(
        0.04, 0.96, f"s={score:.3f}",
        transform=ax.transAxes,
        fontsize=fontsize, color=color, weight="semibold",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg, edgecolor="none", alpha=0.75),
    )


def row_label(ax: plt.Axes, text: str, fontsize: int = 9) -> None:
    """Left-side row label, styled consistently."""
    ax.set_ylabel(text, fontsize=fontsize, color=PALETTE["text"],
                  rotation=90, labelpad=8)


def soft_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Subtle background grid for bar/line charts."""
    ax.grid(axis=axis, alpha=0.3, linewidth=0.6, color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)


def img_to_display(img: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) to a display-ready array for imshow."""
    if img.ndim == 2:
        return img
    if img.shape[0] == 1:
        return img[0]
    if img.shape[0] == 3:
        return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
    return img[0]


def is_grayscale(img: np.ndarray) -> bool:
    return img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1)


def ensure_dir(path: Optional[str]) -> None:
    if path:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
