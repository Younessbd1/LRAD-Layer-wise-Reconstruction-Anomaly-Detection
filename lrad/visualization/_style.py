"""Shared visual style: a single consistent aesthetic across every LRAD plot.

Typography, spacing, palette, and reference colorbars are defined here so
every figure in the project looks like it came from the same report.

Design choices:
  - Modern, slightly desaturated palette with one strong accent for anomaly
    highlights so the eye is drawn to the right tile.
  - "magma" for anomaly-error heatmaps. magma is perceptually uniform,
    print-safe, and its top values are bright yellow — the exact signal we
    want for sunglasses to "pop" at test time.
  - Compact default figure sizes (small + visible) with savefig DPI bumped
    to 300 so even small figures are crisp at full size.
  - Legends always keep a thin frame so the "key" is unambiguous when the
    figure is dropped into a slide.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional


# ---------------------------------------------------------------------------
# Palette — modern, print-safe, colorblind-friendly
# ---------------------------------------------------------------------------

PALETTE = {
    "normal":        "#1B7F65",   # rich teal — "good" / normal class
    "anomaly":       "#E5552B",   # vivid orange — anomaly highlight
    "anomaly_2":     "#9C2C1A",   # dark brick
    "accent":        "#5B4FCF",   # modern indigo
    "accent_2":      "#2D8FD6",   # bright cobalt
    "neutral":       "#C98A1F",   # warm ochre
    "bar_primary":   "#2F6F8E",
    "bar_secondary": "#8C3F4F",
    "bar_tertiary":  "#5F8F46",
    "grid":          "#E2E2E2",
    "text":          "#15171A",
    "text_muted":    "#5C6166",
    "panel":         "#FAFAFB",
}

# A 6-color cycle that stays distinct on white and on slide backgrounds.
CYCLE = [
    PALETTE["normal"], PALETTE["anomaly"], PALETTE["accent"],
    PALETTE["accent_2"], PALETTE["neutral"], PALETTE["anomaly_2"],
]

# Colormaps. inferno is perceptually uniform with a bright-yellow top end —
# anomalous regions (e.g. sunglasses) "pop" without saturating mid-range.
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
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "white",

        # Axes — thinner spines and reference lines
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#2A2D31",
        "axes.linewidth":     0.5,
        "axes.titlesize":     9,
        "axes.titleweight":   "regular",
        "axes.titlepad":      4,
        "axes.labelsize":     8.5,
        "axes.labelcolor":    PALETTE["text"],
        "axes.labelweight":   "regular",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,      # opt-in per plot

        # Ticks
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "xtick.color":        PALETTE["text_muted"],
        "ytick.color":        PALETTE["text_muted"],
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,

        # Lines — narrower default for plot lines
        "lines.linewidth":    1.0,
        "lines.markersize":   4.0,
        "patch.linewidth":    0.5,

        # Legend — thin frame keeps the "key" unambiguous
        "legend.fontsize":    7.5,
        "legend.frameon":     True,
        "legend.facecolor":   "white",
        "legend.edgecolor":   PALETTE["grid"],
        "legend.framealpha":  0.95,
        "legend.borderpad":   0.4,
        "legend.title_fontsize": 8,

        # Grid
        "grid.color":         PALETTE["grid"],
        "grid.linewidth":     0.4,
        "grid.alpha":         0.55,

        # Font — prefer Calibri then Times New Roman; fall through to the
        # closest free metric-compatible fonts (Carlito mimics Calibri,
        # Liberation Serif mimics Times) when those proprietary fonts are
        # not installed.
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Calibri", "Carlito", "Times New Roman",
                               "Liberation Serif", "DejaVu Sans"],
        "font.serif":         ["Times New Roman", "Liberation Serif",
                               "DejaVu Serif"],
        "font.size":          8.5,

        # Color cycle
        "axes.prop_cycle":    plt.cycler(color=CYCLE),
    })
    _APPLIED = True


# ---------------------------------------------------------------------------
# Helpers - colorbar, axis polish, tile frames
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
    is visible. ``endpoint_labels`` overrides the numeric ticks with a
    (low, high) text pair — useful when the gradient is purely qualitative.
    """
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
    sm.set_array([])
    if hasattr(axes, "ravel"):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    cbar = fig.colorbar(
        sm, ax=ax_list,
        fraction=0.022, pad=0.022, shrink=0.88, aspect=32,
    )
    cbar.set_label(label, fontsize=9, color=PALETTE["text"], labelpad=6)
    cbar.ax.tick_params(labelsize=8, color=PALETTE["text_muted"])
    cbar.outline.set_edgecolor("#3A3D42")
    cbar.outline.set_linewidth(0.6)

    if endpoint_labels is not None:
        low, high = endpoint_labels
        cbar.set_ticks([0.0, vmax])
        cbar.set_ticklabels([low, high])
    return cbar


def clean_image_axis(ax: plt.Axes, frame_color: str = "#3A3D42") -> None:
    """Axis suitable for an image tile: no ticks, thin border frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(0.5)


def annotate_score(ax: plt.Axes, score: float, fontsize: int = 7,
                   color: str = "white", bg: str = "#15171A") -> None:
    """Place a score badge inside the top-left of an image tile."""
    ax.text(
        0.04, 0.96, f"s={score:.3f}",
        transform=ax.transAxes,
        fontsize=fontsize, color=color, weight="semibold",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor="none", alpha=0.78),
    )


def row_label(ax: plt.Axes, text: str, fontsize: int = 9) -> None:
    """Left-side row label, styled consistently."""
    ax.set_ylabel(text, fontsize=fontsize, color=PALETTE["text"],
                  rotation=90, labelpad=8)


def soft_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Subtle background grid for bar/line charts."""
    ax.grid(axis=axis, alpha=0.35, linewidth=0.55, color=PALETTE["grid"], zorder=0)
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
