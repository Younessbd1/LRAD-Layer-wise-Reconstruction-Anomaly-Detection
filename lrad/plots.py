"""Per-block reconstruction visualizations.

Two figures are produced for a small set of sample images, mixing
in-distribution and OOD samples with row labels:

  * ``plot_per_block_breakdown`` — for every conv block ``k``, three
    columns side by side: the original image, the squared
    reconstruction error ``(x − recon_k)²`` (viridis), and the
    reconstruction itself. Layout::

        Original  Err L0  Recon L0  Err L1  Recon L1  ...  Err Ln  Recon Ln

  * ``plot_recons_only`` — same rows, just original + per-block
    reconstructions::

        Original  Recon L0  Recon L1  ...  Recon Ln

A second family of figures visualizes the **ensemble bias/variance
decomposition** (see ``lrad.ensemble``). The OOD anomaly is the *bias*
term itself — ``bias = risk − variance = (x − f̄)²`` per pixel — computed
directly from the ensemble, with **no sigma and no division**:

  * ``plot_mean_abs_bias`` — per-block ``(x − f̄_k)²`` heatmaps for each
    sample (the bias term — squared error of the ensemble-mean recon).
  * ``plot_variance_heatmaps`` — per-block model-disagreement (variance)
    heatmaps, used to show the epistemic signal on OOD inputs.
  * ``plot_score_comparison`` — Bias / Risk / min / quantile-min error
    maps side by side at a single block depth, each column with its own
    colour scale.
  * ``plot_bias_variance_vs_block`` — how the mean bias and variance
    evolve with conv-block depth, ID vs OOD.
  * ``plot_bias_variance_vs_percentile`` — how the per-image bias and
    variance scores separate ID from OOD across aggregation percentiles.
  * ``plot_ensemble_decomposition`` / ``plot_decomposition_auroc_bars`` /
    ``plot_ensemble_score_hists`` — the full Risk | Bias | Variance maps,
    per-block AUROC bars, and aggregated score histograms.
  * ``plot_member_instance`` — ONE member's reconstruction + error for one
    test image (one figure per ensemble member).
  * ``plot_instance_summary`` — the consensus view for one test image:
    Bias / Mean-error / Min-error maps and the smoothed bias painted onto
    the face (``smooth_cam``).
  * ``plot_top_ood_glasses`` — the top-N OOD eyeglasses faces ranked by
    how strongly the bias lights up in the eye region.

Styling targets a conference paper: serif (Times-like) typeface with STIX
math, the Okabe–Ito colorblind-safe palette with FIXED role assignments
(ID is always blue, OOD always vermillion, variance always orange),
recessive axes (no top/right spines, light grid), viridis for error maps
with fixed colour scales so figures stay comparable, hidden spines/ticks
on image axes, ``layout='constrained'``, and 300-dpi PNG export. Error
tiles carry no per-tile numeric annotations — the maps speak through the
shared colour bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .arch_diagram import KERNEL_COLORS


_TITLE_FS = 13
_LABEL_FS = 11
_TICK_FS = 9

# High-resolution export for every figure in this module (requirement:
# dpi >= 200, ideally 300).
_SAVE_DPI = 300

# Okabe–Ito colorblind-safe palette (Wong, Nature Methods 2011), with fixed
# role assignments used across every figure in the project — a role never
# changes colour between figures:
_C_ID = "#0072B2"        # in-distribution           (blue)
_C_OOD = "#D55E00"       # out-of-distribution       (vermillion)
_C_RISK = "#56B4E9"      # Risk term                 (sky blue)
_C_BIAS = "#0072B2"      # Bias term (the anomaly)   (blue)
_C_VARIANCE = "#E69F00"  # Variance term             (orange)
_C_ACCENT = "#009E73"    # secondary series          (bluish green)

# Conference-paper styling, module-wide so callers that build their own
# figures (e.g. run_celeba's history plots) share the look: serif
# (Times-like) text with STIX math, recessive axes and unframed legends.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Times", "Nimbus Roman No9 L",
        "STIXGeneral", "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "savefig.dpi": _SAVE_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _causal_rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean (no look-ahead) over a 1-D series."""
    n = arr.size
    out = np.empty_like(arr)
    cs = np.cumsum(arr)
    out[:w] = cs[:w] / np.arange(1, min(w, n) + 1)
    out[w:] = (cs[w:] - cs[:-w]) / w
    return out


def _epoch_boundaries(ax: plt.Axes, epoch_ends, n: int, y: float) -> None:
    """Vertical dashed lines at epoch ends + centred ``E<i>`` labels."""
    ends = np.asarray(epoch_ends, dtype=float)
    starts = np.concatenate([[0], ends[:-1] + 1])
    for i, end in enumerate(ends):
        if end < n - 1:
            ax.axvline(end + 0.5, color="gray", linestyle="--",
                       linewidth=0.75, alpha=0.55)
        mid = (starts[i] + end) / 2
        ax.text(mid, y, f"E{i + 1}", ha="center", va="bottom",
                fontsize=7.5, color="#555", clip_on=True)


def plot_batch_accuracy(history: dict, save_path: str | Path) -> None:
    """Per-batch gender and attr accuracy across all training batches.

    Draws the raw per-batch series (faint) and a rolling mean (thick),
    with vertical dashed lines at epoch boundaries and epoch number labels.
    """
    gender = np.asarray(history.get("batch_gender_acc", []))
    attrs = np.asarray(history.get("batch_attr_acc_mean", []))
    epoch_ends = history.get("epoch_ends", [])

    if gender.size == 0:
        return

    n = gender.size
    x = np.arange(n)

    # rolling mean — ~1/80 of the full run, at least 10 batches
    w = max(10, n // 80)
    gender_smooth = _causal_rolling_mean(gender, w)
    attrs_smooth = _causal_rolling_mean(attrs, w)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    fig.suptitle("Per-batch accuracy during training", fontsize=_TITLE_FS)

    pairs = [
        (axes[0], gender, gender_smooth, "Gender accuracy", _C_ID),
        (axes[1], attrs,  attrs_smooth,  "Mean attribute accuracy", _C_ACCENT),
    ]
    for ax, raw, smooth, title, colour in pairs:
        ax.plot(x, raw,    alpha=0.18, linewidth=0.55, color=colour)
        ax.plot(x, smooth, linewidth=1.8, color=colour,
                label=f"rolling mean  (w={w})")
        _epoch_boundaries(ax, epoch_ends, n, y=0.475)

        ax.set_xlabel("Batch (global index)", fontsize=_LABEL_FS)
        ax.set_ylabel("Accuracy", fontsize=_LABEL_FS)
        ax.set_title(title, fontsize=_LABEL_FS)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_batch_loss(history: dict, save_path: str | Path) -> None:
    """Per-batch classifier loss across ALL training batches.

    Raw per-batch combined loss (CE gender + weighted BCE attrs, + the
    CutPaste CE when the pretext task is on) drawn faint, with a causal
    rolling mean on top, epoch boundaries marked and a log y-scale so the
    within- and between-epoch decrease stays visible over the whole run.
    """
    loss = np.asarray(history.get("batch_loss", []), dtype=float)
    epoch_ends = history.get("epoch_ends", [])
    if loss.size == 0:
        return

    n = loss.size
    x = np.arange(n)
    w = max(10, n // 80)
    smooth = _causal_rolling_mean(loss, w)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(x, loss, alpha=0.18, linewidth=0.55, color=_C_ID)
    ax.plot(x, smooth, linewidth=1.8, color=_C_ID,
            label=f"rolling mean  (w={w})")
    ax.set_yscale("log")
    _epoch_boundaries(ax, epoch_ends, n,
                      y=float(np.percentile(loss, 1)))
    ax.set_xlabel("Batch (global index)", fontsize=_LABEL_FS)
    ax.set_ylabel("Classifier loss (log scale)", fontsize=_LABEL_FS)
    ax.set_title("Per-batch classifier loss during training",
                 fontsize=_TITLE_FS)
    ax.set_xlim(0, n - 1)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)

    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_history(
    history: dict,
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """How the per-block decoders learn: reconstruction MSE per epoch.

    One line per conv block (depth-ordered viridis shades) plus the summed
    total (grey dashed), log y-scale so the epoch-to-epoch decrease of the
    reconstruction error stays readable even after the initial drop. Each
    block line carries a direct end label with its final MSE. Validation
    curves (when a val split exists) are drawn dotted in the same colours.
    ``history`` comes from :func:`lrad.train.train_decoders`.
    """
    per_block = np.asarray(history.get("train_loss_per_block", []),
                           dtype=float)  # (E, n_blocks)
    if per_block.size == 0:
        return
    total = np.asarray(history.get("train_loss", []), dtype=float)
    val_per_block = np.asarray(history.get("val_loss_per_block", []),
                               dtype=float)
    n_epochs, n_blocks = per_block.shape
    epochs = np.arange(1, n_epochs + 1)
    colours = plt.cm.viridis(np.linspace(0.10, 0.85, n_blocks))

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for k in range(n_blocks):
        ax.plot(epochs, per_block[:, k], color=colours[k], linewidth=1.8,
                label=f"block L{k}")
        if val_per_block.size:
            ax.plot(epochs, val_per_block[:, k], color=colours[k],
                    linewidth=1.2, linestyle=":")
        ax.annotate(
            f"{per_block[-1, k]:.4f}",
            xy=(n_epochs, per_block[-1, k]),
            xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=colours[k],
        )
    if total.size:
        ax.plot(epochs, total, color="#666666", linewidth=1.2,
                linestyle="--", label="total (sum)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=_LABEL_FS)
    ax.set_ylabel("Reconstruction MSE (log scale)", fontsize=_LABEL_FS)
    ax.set_title(title or "Per-block decoder reconstruction error",
                 fontsize=_TITLE_FS)
    ax.set_xlim(1, n_epochs)
    ax.set_xticks(epochs[:: max(1, n_epochs // 10)])
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, ncols=2)

    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _to_image_grid(t: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) tensor in [0, 1] → (B, H, W, 3) numpy in [0, 1]."""
    return t.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()


def _sq_error(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """(H, W) squared error summed across RGB channels.

    Squared (not absolute) so every error tile is on the same L2 footing as
    the ensemble bias/variance decomposition, where ``Risk = Bias + Variance``
    only holds for squared error. The per-channel squared errors are
    **summed** (not averaged, no sqrt), so a pixel lives in ``[0, 3]``.
    """
    return ((orig - recon) ** 2).sum(axis=-1)


def _bare(ax: plt.Axes) -> None:
    """Hide ticks and spines on an image axis but keep title/ylabel usable."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _row_label(ax: plt.Axes, label: str) -> None:
    ax.set_ylabel(label, fontsize=_LABEL_FS, rotation=90,
                  labelpad=6, va="center")


def plot_per_block_breakdown(
    images: torch.Tensor,
    recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Side-by-side (Original | Err Lk | Recon Lk) for every conv block.

    Error tiles use the fixed ``[0, 3]`` colour scale (images live in
    ``[0, 1]`` so the RGB-summed squared error lives in ``[0, 3]``), never a
    per-figure max, so error tiles are directly comparable across different
    figures. A single colour bar is drawn on the right.
    """
    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + 2 * n_blocks

    err_maps: list[list[np.ndarray]] = []
    for r in range(n_rows):
        row = [_sq_error(images_np[r], recons_np[k][r])
               for k in range(n_blocks)]
        err_maps.append(row)
    # cap at 0.5 — errors live in [0, 3] (RGB-summed) but typical recon
    # error is well below 1; 0.5 gives a readable colour range without
    # washing out structure. fixed (not data-dependent) so figures compare.
    vmax = 0.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows

    err_im = None

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for k in range(n_blocks):
            err = err_maps[r][k]
            ax_err = axes[r, 1 + 2 * k]
            im = ax_err.imshow(err, cmap="viridis", vmin=0.0, vmax=vmax)
            err_im = im
            _bare(ax_err)
            if r == 0:
                ax_err.set_title(f"Err L{k}", fontsize=_LABEL_FS)

            ax_rec = axes[r, 2 + 2 * k]
            ax_rec.imshow(recons_np[k][r])
            _bare(ax_rec)
            if r == 0:
                ax_rec.set_title(f"Recon L{k}", fontsize=_LABEL_FS)

    if err_im is not None:
        cbar = fig.colorbar(
            err_im, ax=axes[:, 1::2].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("(x − recon)²  (sum over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_activations(
    images: torch.Tensor,
    activations: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block activation maps overlaid on each input image.

    For every conv block ``k`` and every sample, the channel-mean of the
    block activations is upsampled to image size (nearest-neighbour via
    ``imshow``'s ``extent``) and rendered as a translucent ``inferno``
    heat map on top of the original image. This shows *where* each block
    is putting its energy — i.e. what the classifier is actually
    learning to attend to at each depth.

    Layout::

        Original  Act L0  Act L1  Act L2  ...  Act Ln
    """
    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    n_blocks = len(activations)
    n_cols = 1 + n_blocks
    H, W = images_np.shape[1], images_np.shape[2]

    # Channel-mean activation per block, on cpu numpy.
    acts_np: list[np.ndarray] = []
    for a in activations:
        acts_np.append(a.detach().cpu().mean(dim=1).numpy())  # (B, h, w)

    # Per-block global vmax so colours are comparable across rows but
    # not across blocks (deeper blocks have very different magnitudes).
    block_vmax = [float(a.max()) if a.max() > 0 else 1.0 for a in acts_np]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for k in range(n_blocks):
            ax_a = axes[r, 1 + k]
            # darkened original underneath for context
            ax_a.imshow(images_np[r] * 0.45)
            ax_a.imshow(
                acts_np[k][r],
                cmap="inferno",
                alpha=0.65,
                extent=(0, W, H, 0),
                vmin=0.0,
                vmax=block_vmax[k],
                interpolation="bilinear",
            )
            _bare(ax_a)
            if r == 0:
                ax_a.set_title(f"Act L{k}", fontsize=_LABEL_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_overlay(
    images: torch.Tensor,
    recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Multi-scale fusion of per-block error maps + overlay on the input.

    For every sample row, draws::

        Original  Err L0  Err L1  ...  Err Ln  Fused (max)  Overlay

    where ``err_map_k = (x − recon_k)²`` summed over RGB, the fused map
    is the per-pixel max across all blocks, and the overlay renders the
    fused heatmap on top of the original image.

    Error tiles, the fused tile, and the overlay all share the fixed
    ``[0, 3]`` colour scale (no data-dependent normalization) and a single
    colour bar on the right, so figures stay comparable to one another.
    """
    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    H, W = images_np.shape[1], images_np.shape[2]
    n_cols = 1 + n_blocks + 2  # Original | Err Lk × n | Fused | Overlay

    err_maps: list[list[np.ndarray]] = []
    fused_maps: list[np.ndarray] = []
    for r in range(n_rows):
        row = [_sq_error(images_np[r], recons_np[k][r])
               for k in range(n_blocks)]
        err_maps.append(row)
        fused_maps.append(np.maximum.reduce(row))  # per-pixel max over blocks
    # same fixed cap as plot_per_block_breakdown — see comment there
    vmax = 0.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows

    err_im = None
    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for k in range(n_blocks):
            err = err_maps[r][k]
            ax_err = axes[r, 1 + k]
            im = ax_err.imshow(err, cmap="viridis", vmin=0.0, vmax=vmax)
            err_im = im
            _bare(ax_err)
            if r == 0:
                ax_err.set_title(f"Err L{k}", fontsize=_LABEL_FS)

        fused = fused_maps[r]
        ax_fused = axes[r, 1 + n_blocks]
        im = ax_fused.imshow(fused, cmap="viridis", vmin=0.0, vmax=vmax)
        err_im = im
        _bare(ax_fused)
        if r == 0:
            ax_fused.set_title("Fused (max)", fontsize=_LABEL_FS)

        ax_ov = axes[r, 2 + n_blocks]
        ax_ov.imshow(images_np[r] * 0.45)
        ax_ov.imshow(
            fused, cmap="viridis", alpha=0.65,
            extent=(0, W, H, 0),
            vmin=0.0, vmax=vmax,
            interpolation="bilinear",
        )
        _bare(ax_ov)
        if r == 0:
            ax_ov.set_title("Overlay", fontsize=_LABEL_FS)

    if err_im is not None:
        heat_axes = np.concatenate([
            axes[:, 1:1 + n_blocks].ravel(),
            axes[:, 1 + n_blocks:1 + n_blocks + 1].ravel(),
            axes[:, 2 + n_blocks:3 + n_blocks].ravel(),
        ]).tolist()
        cbar = fig.colorbar(
            err_im, ax=heat_axes,
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("(x − recon)²  (sum over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_recons_only(
    images: torch.Tensor,
    recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """(Original | Recon L0 | Recon L1 | ... | Recon Ln) for every block."""
    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + n_blocks

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for k in range(n_blocks):
            ax_rec = axes[r, 1 + k]
            ax_rec.imshow(recons_np[k][r])
            _bare(ax_rec)
            if r == 0:
                ax_rec.set_title(f"Recon L{k}", fontsize=_LABEL_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ensemble bias/variance decomposition (see lrad.ensemble)
#
# The anomaly is the bias term itself — bias = risk − variance = (x − f̄)²
# per pixel — with no sigma and no division. The helpers below render the
# per-block bias/variance maps, their evolution with depth, and how the
# per-image scores separate ID from OOD.
# ---------------------------------------------------------------------------

def _stat_np(t) -> np.ndarray:
    """(H, W) torch tensor (or array) → float32 numpy on CPU."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _block_heatmap_grid(
    images: torch.Tensor,
    block_maps: Sequence,
    save_path: str | Path,
    *,
    col_titles: Sequence[str],
    cbar_label: str,
    cmap: str = "viridis",
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Original image + one per-block heatmap column, one row per sample.

    ``block_maps`` is one ``(B, H, W)`` map per displayed block (already
    sliced to the sample rows). Every tile uses a fixed display cap of
    ``vmax=0.5`` — errors live in ``[0, 3]`` (RGB-summed, images in
    ``[0, 1]``) but typical recon errors sit well below 1, so 0.5 gives a
    readable range without washing out structure. The cap is fixed, not
    data-dependent, so any two figures are comparable. One colour bar sits on
    the right. Shared backend for the bias / mean-error / min-error /
    variance heatmap figures.
    """
    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    n_blocks = len(block_maps)
    n_cols = 1 + n_blocks
    maps_np = [_stat_np(m) for m in block_maps]
    vmax = 0.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    im = None

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for bi in range(n_blocks):
            ax_e = axes[r, 1 + bi]
            e = maps_np[bi][r]
            im = ax_e.imshow(e, cmap=cmap, vmin=0.0, vmax=vmax)
            _bare(ax_e)
            if r == 0:
                ax_e.set_title(col_titles[bi], fontsize=_LABEL_FS)

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label(cbar_label, fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_abs_bias(
    images: torch.Tensor,
    mean_recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block ``(x − f̄_k)²`` heatmaps of the ensemble-mean reconstruction.

    ``mean_recons`` is the list of per-block ensemble-mean reconstructions
    ``f̄_k`` (each ``(B, 3, H, W)``). For every sample one row is drawn::

        Original | (x − f̄_L0)² | (x − f̄_L1)² | ... | (x − f̄_Ln)²

    This is the Bias term of the decomposition — the squared error of the
    consensus model. The colour scale is fixed to ``[0, 3]`` (no per-figure
    normalization) so figures are comparable, with a single colour bar.
    ID rows should be dim everywhere while OOD rows light up on the
    anomaly (e.g. eyeglasses).
    """
    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in mean_recons]
    # bias term per block: squared error of the ensemble-mean recon
    block_maps = [_sq_error(images_np, recons_np[j])
                  for j in range(len(recons_np))]
    _block_heatmap_grid(
        images, block_maps, save_path,
        col_titles=[f"(x − f̄)² L{j}" for j in range(len(recons_np))],
        cbar_label="(x − f̄)²  (sum over RGB)",
        row_labels=row_labels, title=title,
    )


def plot_mean_error_maps(
    images: torch.Tensor,
    error_maps: dict,
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block ensemble-averaged squared error heatmaps.

    ``error_maps`` is ``{k: (B, H, W)}`` — the per-pixel mean over the
    ``M`` models of ``(x − f̂^m_k)²`` (see ``lrad.ensemble.mean_error_maps``),
    already sliced to the displayed samples. For every sample one row is
    drawn::

        Original | Err L0 | Err L1 | ... | Err Ln

    This is the average of the per-model error maps, i.e. exactly the Risk
    term of the decomposition — not the error of the averaged
    reconstruction. The colour scale is fixed to ``[0, 3]`` (no per-figure
    normalization) so figures are comparable, with a single colour bar.
    """
    blocks = sorted(error_maps.keys())
    _block_heatmap_grid(
        images, [error_maps[k] for k in blocks], save_path,
        col_titles=[f"Err L{k}" for k in blocks],
        cbar_label="mean_m (x − f̂^m)²  (sum over RGB)",
        row_labels=row_labels, title=title,
    )


def plot_min_error_maps(
    images: torch.Tensor,
    error_maps: dict,
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block ensemble per-pixel *minimum* squared error heatmaps.

    ``error_maps`` is ``{k: (B, H, W)}`` — the per-pixel minimum over the
    ``M`` models of ``(x − f̂^m_k)²`` (see ``lrad.ensemble.min_error_maps``),
    already sliced to the displayed samples. For every sample one row is
    drawn::

        Original | Err L0 | Err L1 | ... | Err Ln

    Unlike ``plot_mean_error_maps`` (average of the per-model error maps),
    each pixel keeps the error of the *best* member, so a tile stays bright
    only where no model in the ensemble reconstructs well. The colour scale
    is fixed to ``[0, 3]`` (no per-figure normalization) so figures are
    comparable, with a single colour bar.
    """
    blocks = sorted(error_maps.keys())
    _block_heatmap_grid(
        images, [error_maps[k] for k in blocks], save_path,
        col_titles=[f"Err L{k}" for k in blocks],
        cbar_label="min_m (x − f̂^m)²  (sum over RGB)",
        row_labels=row_labels, title=title,
    )


# ---------------------------------------------------------------------------
# Per-instance view: one test image at a time
#
# The grid figures above stack many samples as rows, which is great for
# spotting trends but hides what each individual model does. The functions
# below zoom into a single image. Per instance the runner writes one figure
# PER ENSEMBLE MEMBER (that member's reconstruction + error map,
# ``plot_member_instance``) plus one consensus summary figure (Bias / Mean /
# Min error maps and the smoothed bias painted back onto the face,
# ``plot_instance_summary``).
# ---------------------------------------------------------------------------

def smooth_cam(cam: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Blur and peak-normalize an error/saliency map for overlaying.

    Rectifies to non-negative, normalizes by the peak, Gaussian-blurs by
    ``sigma``, then normalizes again so the result lives in ``[0, 1]`` and can
    drive both the colour and the per-pixel alpha of an overlay. A bigger
    ``sigma`` spreads the highlight into smoother blobs; it is a display knob,
    not part of the score.
    """
    from scipy.ndimage import gaussian_filter

    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    peak = float(cam.max())
    if peak > 1e-8:
        cam = cam / peak
    cam = gaussian_filter(cam, sigma=sigma)
    peak = float(cam.max())
    if peak > 1e-8:
        cam = cam / peak
    return cam


def _single_image(t: torch.Tensor) -> np.ndarray:
    """A single ``(3, H, W)`` tensor → ``(H, W, 3)`` numpy in [0, 1]."""
    return _to_image_grid(t.unsqueeze(0))[0]


def plot_member_instance(
    image: torch.Tensor,
    recon: torch.Tensor,
    save_path: str | Path,
    *,
    member: int = 1,
    vmax: float = 0.5,
    title: str | None = None,
) -> None:
    """ONE member's view of ONE image: Original | Reconstruction | Error.

    ``image`` and ``recon`` are single ``(3, H, W)`` tensors — the input and
    member ``member``'s reconstruction of it at one block depth. The error
    tile is the RGB-summed squared error ``sum_c (x_c − f̂_c)²`` on the fixed
    ``[0, vmax]`` colour scale shared by every error figure in the project,
    with its own colour bar, so the ``M`` member figures of one instance are
    directly comparable to each other and across instances.
    """
    img_np = _single_image(image)
    recon_np = _single_image(recon)
    err = _sq_error(img_np, recon_np)

    fig, axes = plt.subplots(
        1, 3, figsize=(5.4, 2.0), layout="constrained",
    )
    axes[0].imshow(img_np)
    _bare(axes[0])
    axes[0].set_title("Original", fontsize=_LABEL_FS)

    axes[1].imshow(recon_np)
    _bare(axes[1])
    axes[1].set_title(f"Recon $M_{{{member}}}$", fontsize=_LABEL_FS)

    im = axes[2].imshow(err, cmap="viridis", vmin=0.0, vmax=vmax)
    _bare(axes[2])
    axes[2].set_title(r"$(x - \hat{f}^{m})^2$", fontsize=_LABEL_FS)
    cbar = fig.colorbar(im, ax=[axes[2]], location="right",
                        shrink=0.85, pad=0.03, aspect=14)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _instance_summary_maps(
    image: torch.Tensor,
    recons: Sequence[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(img_np, bias, mean_err, min_err) for one image and its M recons.

    Everything is derived in one place so the summary panels are guaranteed
    to agree with one another::

        e_m   = sum_RGB (x − f̂^m)²      per-model squared error  (M maps)
        f̄     = mean_m f̂^m              ensemble-mean reconstruction
        bias  = sum_RGB (x − f̄)²        error of the consensus model
        mean  = mean_m e_m               Risk — the average of the errors
        min   = min_m e_m                error of the best member, per pixel
    """
    img_np = _single_image(image)                       # (H, W, 3)
    recon_np = [_single_image(r) for r in recons]       # M × (H, W, 3)
    if not recon_np:
        raise ValueError("need at least one reconstruction to plot")
    err_stack = np.stack(
        [_sq_error(img_np, rc) for rc in recon_np], axis=0,
    )                                                    # (M, H, W)
    mean_recon = np.mean(np.stack(recon_np, axis=0), axis=0)
    bias = _sq_error(img_np, mean_recon)                 # (H, W)
    return img_np, bias, err_stack.mean(axis=0), err_stack.min(axis=0)


def plot_instance_summary(
    image: torch.Tensor,
    recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    sigma: float = 3.0,
    overlay_power: float = 0.8,
    vmax: float = 0.5,
    title: str | None = None,
) -> None:
    """Consensus summary for ONE image: bias / mean / min errors + overlay.

    ``image`` is a single ``(3, H, W)`` tensor and ``recons`` holds the ``M``
    ensemble members' reconstructions of it, each ``(3, H, W)``, taken at one
    block depth (see :func:`_instance_summary_maps` for the derived maps).
    One row of five tiles::

        Original | Bias (x − f̄)² | Mean error | Min error | Bias overlay

    Bias and Mean share the fixed ``[0, vmax]`` colour scale; the Min-error
    tile gets its OWN colour scale — the per-pixel minimum lives on a much
    smaller magnitude, so the shared cap would render it almost flat. The
    overlay is the Gaussian-smoothed bias (see :func:`smooth_cam`) painted
    over the face with ``inferno``; ``sigma`` sets the blur and
    ``overlay_power`` the alpha falloff (alpha = cam ** overlay_power) —
    both are purely cosmetic, never part of the score.
    """
    img_np, bias, mean_err, min_err = _instance_summary_maps(image, recons)
    cam = smooth_cam(bias, sigma=sigma)

    fig, axes = plt.subplots(
        1, 5, figsize=(9.2, 2.35), layout="constrained",
    )
    axes[0].imshow(img_np)
    _bare(axes[0])
    axes[0].set_title("Original", fontsize=_LABEL_FS)

    fixed = [(r"Bias  $(x - \bar{f})^2$", bias), ("Mean error", mean_err)]
    for ax, (name, mp) in zip(axes[1:3], fixed):
        im = ax.imshow(mp, cmap="viridis", vmin=0.0, vmax=vmax)
        _bare(ax)
        ax.set_title(name, fontsize=_LABEL_FS)
        cb = fig.colorbar(im, ax=[ax], location="bottom",
                          shrink=0.9, pad=0.02, aspect=12)
        cb.ax.tick_params(labelsize=7)

    # Min error on the raw scale: no vmin/vmax at all, matplotlib autoscales
    # to the data range (the per-pixel minimum sits far below bias/mean, so
    # any shared cap would render it flat).
    im_min = axes[3].imshow(min_err, cmap="viridis")
    _bare(axes[3])
    axes[3].set_title("Min error", fontsize=_LABEL_FS)
    cb = fig.colorbar(im_min, ax=[axes[3]], location="bottom",
                      shrink=0.9, pad=0.02, aspect=12)
    cb.ax.tick_params(labelsize=7)

    # Smoothed bias painted back onto the face.
    axes[4].imshow(img_np)
    axes[4].imshow(cam, cmap="inferno",
                   alpha=np.clip(cam, 0.0, 1.0) ** overlay_power)
    _bare(axes[4])
    axes[4].set_title(f"Bias overlay ($\\sigma$={sigma:g})",
                      fontsize=_LABEL_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_ood_glasses(
    images: torch.Tensor,
    recons_per_model: Sequence[Sequence[torch.Tensor]],
    save_path: str | Path,
    *,
    sigma: float = 3.0,
    overlay_power: float = 0.8,
    vmax: float = 0.5,
    col_labels: Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    """Top-N OOD eyeglasses faces, one column per face (rank order).

    ``images`` is ``(N, 3, H, W)`` — the top-ranked OOD faces, best first —
    and ``recons_per_model`` is indexed ``[model][image]`` with ``(3, H, W)``
    reconstructions at one block depth. Three rows per face::

        Original
        Bias (x − f̄)²          (fixed [0, vmax] scale, one colour bar)
        Bias overlay            (smoothed bias painted over the face)

    Columns are titled by rank (``#1 … #N``) unless ``col_labels`` gives one
    title per image (e.g. ``"Glasses 1"``, ``"Clean 2"`` for a generalization
    probe on custom photos — see ``scripts/run_generalization.py``), in
    which case those are used instead. The ranking itself — how strongly and
    how locally the bias lights up in the eye region — is computed by the
    caller (see ``lrad.ensemble.collect_eye_region_bias``).
    """
    n = images.size(0)
    if n == 0:
        raise ValueError("need at least one image to plot")
    if col_labels is not None and len(col_labels) != n:
        raise ValueError(
            f"col_labels has {len(col_labels)} entries, expected {n}"
        )

    fig, axes = plt.subplots(
        3, n, figsize=(1.4 * n + 0.8, 4.6),
        layout="constrained", squeeze=False,
    )
    bias_im = None
    for i in range(n):
        recons_i = [recons_per_model[m][i]
                    for m in range(len(recons_per_model))]
        img_np, bias, _, _ = _instance_summary_maps(images[i], recons_i)
        cam = smooth_cam(bias, sigma=sigma)

        axes[0, i].imshow(img_np)
        _bare(axes[0, i])
        label = col_labels[i] if col_labels is not None else f"#{i + 1}"
        axes[0, i].set_title(label, fontsize=_LABEL_FS)

        bias_im = axes[1, i].imshow(bias, cmap="viridis",
                                    vmin=0.0, vmax=vmax)
        _bare(axes[1, i])

        axes[2, i].imshow(img_np)
        axes[2, i].imshow(cam, cmap="inferno",
                          alpha=np.clip(cam, 0.0, 1.0) ** overlay_power)
        _bare(axes[2, i])

    _row_label(axes[0, 0], "Original")
    _row_label(axes[1, 0], r"Bias  $(x - \bar{f})^2$")
    _row_label(axes[2, 0], "Overlay")

    if bias_im is not None:
        cbar = fig.colorbar(bias_im, ax=axes[1, :].ravel().tolist(),
                            location="right", shrink=0.9, pad=0.01,
                            aspect=14)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        title or "Top OOD eyeglasses faces — bias concentrated on the "
                 "eye region",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_score_comparison(
    images: torch.Tensor,
    score_maps: Sequence[tuple[str, torch.Tensor]],
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Side-by-side comparison of per-pixel score maps at ONE block depth.

    ``score_maps`` is an ordered sequence of ``(column_title, (B, H, W))``
    pairs, e.g. the Bias ``(x − f̄)²``, the Risk ``mean_m (x − f̂^m)²``,
    the minimum ``min_m (x − f̂^m)²`` and the robust quantile-minimum
    (k-th smallest error). Layout: one row per sample::

        Original | map 1 | map 2 | ... | map n

    IMPORTANT: the maps are deliberately NOT normalized across columns —
    the scores are not identically distributed (the minimum lives on a far
    smaller scale than the risk), so a shared vmin/vmax would crush the
    small-magnitude columns. Each column gets its own colour scale
    (``vmin=0``, ``vmax`` = that column's max) and its own colour bar
    below the column.
    """
    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    score_maps = list(score_maps)
    n_score = len(score_maps)
    n_cols = 1 + n_score
    maps_np = [_stat_np(m) for _, m in score_maps]
    col_titles = [name for name, _ in score_maps]
    # Per-column scale: each score has its own dynamic range.
    col_vmax = [max(float(m.max()), 1e-12) for m in maps_np]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.9),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    col_ims: list = [None] * n_score

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for c in range(n_score):
            ax_e = axes[r, 1 + c]
            e = maps_np[c][r]
            col_ims[c] = ax_e.imshow(
                e, cmap="viridis", vmin=0.0, vmax=col_vmax[c],
            )
            _bare(ax_e)
            if r == 0:
                ax_e.set_title(col_titles[c], fontsize=_LABEL_FS)

    # One horizontal colour bar per score column, under that column only.
    for c, im in enumerate(col_ims):
        if im is None:
            continue
        cbar = fig.colorbar(
            im, ax=axes[:, 1 + c].ravel().tolist(),
            location="bottom", shrink=0.92, pad=0.015, aspect=12,
        )
        cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_variance_heatmaps(
    images: torch.Tensor,
    maps: dict,
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block ensemble-variance (model-disagreement) heatmaps.

    ``maps`` is ``{k: {'variance': (B, H, W)}}`` aligned with ``images``
    (one variance map per block, already sliced to the displayed samples).
    For every sample one row is drawn::

        Original | Var L0 | Var L1 | ... | Var Ln

    Variance is the epistemic-uncertainty signal: on OOD inputs the
    ensemble members extrapolate differently, so they disagree more and
    the maps brighten. The colour scale is fixed to ``[0, 3]`` (no
    per-figure normalization) so figures are comparable, with a single
    colour bar. Variance is bounded above by the risk so it shares the
    ``[0, 3]`` image-error scale, though in practice it stays small.
    """
    blocks = sorted(maps.keys())
    _block_heatmap_grid(
        images, [maps[k]["variance"] for k in blocks], save_path,
        col_titles=[f"Var L{k}" for k in blocks],
        cbar_label="ensemble variance  (sum over RGB)",
        cmap="magma",
        row_labels=row_labels, title=title,
    )


def _mean_std_over_blocks(
    per_block: dict, term: str, blocks: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of a per-image score across blocks for one term.

    ``per_block`` is ``{term: {k: (N,)}}``; returns two ``(n_blocks,)``
    arrays (mean and std over the N images, per block).
    """
    means, stds = [], []
    for k in blocks:
        arr = np.asarray(per_block[term][k]).ravel()
        means.append(float(arr.mean()) if arr.size else float("nan"))
        stds.append(float(arr.std()) if arr.size else float("nan"))
    return np.asarray(means), np.asarray(stds)


def plot_bias_variance_vs_block(
    scores_in_per_block: dict,
    scores_ood_per_block: dict,
    save_path: str | Path,
    *,
    blocks: Sequence[int],
    title: str | None = None,
) -> None:
    """Evolution of the mean Bias and Variance scores with conv-block depth.

    ``scores_*_per_block`` is ``{term: {k: (N,)}}`` (the per-image,
    per-block scores from ``lrad.ensemble.collect_decomposition_scores``).
    Two panels — Bias (left) and Variance (right) — each plotting the mean
    score per block for ID vs OOD with a ±1 std band, so the depth at
    which OOD separates from ID is visible for both terms.
    """
    blocks = list(blocks)
    x = np.asarray(blocks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             constrained_layout=True)
    panels = [(axes[0], "bias", "Bias"), (axes[1], "variance", "Variance")]
    for ax, term, name in panels:
        in_m, in_s = _mean_std_over_blocks(scores_in_per_block, term, blocks)
        ood_m, ood_s = _mean_std_over_blocks(
            scores_ood_per_block, term, blocks,
        )
        ax.plot(x, in_m, "-o", color=_C_ID, markersize=4.5,
                label="in-dist")
        ax.fill_between(x, in_m - in_s, in_m + in_s, color=_C_ID,
                        alpha=0.15, linewidth=0)
        ax.plot(x, ood_m, "--s", color=_C_OOD, markersize=4.5,
                label="OOD")
        ax.fill_between(x, ood_m - ood_s, ood_m + ood_s, color=_C_OOD,
                        alpha=0.15, linewidth=0)
        ax.set_xlabel("Conv block", fontsize=_LABEL_FS)
        ax.set_ylabel(f"{name} score (mean ± std)", fontsize=_LABEL_FS)
        ax.set_title(name, fontsize=_LABEL_FS)
        ax.set_xticks(x)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        title or "Bias & Variance evolution with block depth (ID vs OOD)",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bias_variance_vs_percentile(
    agg_in: dict,
    agg_ood: dict,
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Bias & Variance score separation as a function of percentile.

    ``agg_in`` / ``agg_ood`` are ``{term: (N,)}`` aggregated per-image
    scores. For each term, sweep the percentile ``q ∈ [1, 99]`` and plot
    the ``q``-th percentile of the ID and OOD score distributions. The
    vertical gap between the two curves is the separability of that term;
    a term whose OOD curve sits well above the ID curve discriminates OOD
    well across the whole range.
    """
    qs = np.linspace(1.0, 99.0, 99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             constrained_layout=True)
    panels = [(axes[0], "bias", "Bias"), (axes[1], "variance", "Variance")]
    for ax, term, name in panels:
        in_s = np.asarray(agg_in[term]).ravel()
        ood_s = np.asarray(agg_ood[term]).ravel()
        in_q = np.percentile(in_s, qs) if in_s.size else np.full_like(qs, np.nan)
        ood_q = (np.percentile(ood_s, qs) if ood_s.size
                 else np.full_like(qs, np.nan))
        ax.plot(qs, in_q, "-", color=_C_ID, linewidth=1.8, label="in-dist")
        ax.plot(qs, ood_q, "--", color=_C_OOD, linewidth=1.8, label="OOD")
        ax.set_xlabel("Percentile of score distribution", fontsize=_LABEL_FS)
        ax.set_ylabel(f"{name} score value", fontsize=_LABEL_FS)
        ax.set_title(name, fontsize=_LABEL_FS)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        title or "Bias & Variance score vs percentile (ID vs OOD)",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full Risk | Bias | Variance maps and AUROC summaries
# ---------------------------------------------------------------------------

def plot_ensemble_decomposition(
    images: torch.Tensor,
    maps: dict,
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Per-block Risk / Bias / Variance maps for each sample image.

    ``maps`` is the ``{k: {'risk','bias','variance', ...}}`` dict produced
    by ``lrad.ensemble.decomposition_maps``. For every conv block ``k``
    three tiles are drawn: the Risk map (mean per-model squared error),
    the Bias map (squared error of the ensemble-mean reconstruction) and
    the Variance map (model disagreement). All three share one colour
    scale — because ``Risk = Bias + Variance`` pixel by pixel, the Bias
    and Variance tiles visibly sum to the Risk tile.
    """
    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    blocks = sorted(maps.keys())
    n_blocks = len(blocks)
    n_cols = 1 + 3 * n_blocks

    term_order = ("risk", "bias", "variance")
    term_title = {"risk": "Risk", "bias": "Bias", "variance": "Var"}

    np_maps = {
        k: {t: maps[k][t].detach().cpu().numpy() for t in term_order}
        for k in blocks
    }
    # Fixed [0, 3] colour scale (images live in [0, 1] so the RGB-summed
    # squared-error terms live in [0, 3]): no data-dependent normalization,
    # so any two figures are directly comparable.
    vmax = 0.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    im = None

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for bi, k in enumerate(blocks):
            for ti, t in enumerate(term_order):
                ax_m = axes[r, 1 + 3 * bi + ti]
                mp = np_maps[k][t][r]
                im = ax_m.imshow(mp, cmap="viridis", vmin=0.0, vmax=vmax)
                _bare(ax_m)
                if r == 0:
                    ax_m.set_title(f"{term_title[t]} L{k}",
                                   fontsize=_LABEL_FS)

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("squared error  (sum over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_decomposition_auroc_bars(
    per_block_auroc: dict,
    save_path: str | Path,
    *,
    block_labels: Sequence[str] | None = None,
    aggregated: dict | None = None,
    title: str | None = None,
) -> None:
    """Per-block OOD AUROC for the Risk / Bias / Variance scores.

    ``per_block_auroc`` maps each term to its per-block AUROC list (block
    order). Three horizontal bars per block. ``aggregated`` optionally
    maps each term to its all-blocks AUROC, appended as a final group.
    """
    terms = ("risk", "bias", "variance")
    colors = {"risk": _C_RISK, "bias": _C_BIAS, "variance": _C_VARIANCE}
    series = {t: list(per_block_auroc[t]) for t in terms}
    n = len(series["risk"])
    names = (
        list(block_labels) if block_labels is not None
        else [f"Block {k}" for k in range(n)]
    )
    if aggregated is not None:
        for t in terms:
            series[t] = series[t] + [aggregated[t]]
        names = names + ["Aggregated"]

    y = np.arange(len(names))[::-1]
    h = 0.26
    offsets = {"risk": h, "bias": 0.0, "variance": -h}

    fig, ax = plt.subplots(
        figsize=(7.8, 0.85 * len(names) + 1.4), constrained_layout=True,
    )
    for t in terms:
        bars = ax.barh(y + offsets[t], series[t], height=h,
                       color=colors[t], label=t)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)

    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=_LABEL_FS)
    ax.set_xlabel("OOD AUROC", fontsize=_LABEL_FS)
    ax.set_xlim(0.0, 1.05)
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        title or "Per-block OOD AUROC — Risk / Bias / Variance",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_score_hists(
    scores_in: dict,
    scores_ood: dict,
    save_path: str | Path,
    *,
    auroc: dict | None = None,
    title: str | None = None,
) -> None:
    """ID vs OOD histograms for the aggregated Risk / Bias / Variance scores.

    ``scores_in`` / ``scores_ood`` map each term name to a 1-D array of
    per-image scores. ``auroc`` optionally maps each term to its AUROC,
    annotated in the panel title.
    """
    terms = ("risk", "bias", "variance")
    label = {
        "risk": "Risk score", "bias": "Bias score",
        "variance": "Variance score",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             constrained_layout=True)
    for ax, t in zip(axes, terms):
        in_s = np.asarray(scores_in[t]).ravel()
        ood_s = np.asarray(scores_ood[t]).ravel()
        bins = 50
        ax.hist(in_s, bins=bins, alpha=0.6, density=True,
                color=_C_ID, label=f"in-dist (n={in_s.size})")
        ax.hist(ood_s, bins=bins, alpha=0.6, density=True,
                color=_C_OOD, label=f"OOD (n={ood_s.size})")
        ax.set_xlabel(label[t], fontsize=_LABEL_FS)
        ax.set_ylabel("Density", fontsize=_LABEL_FS)
        ttl = label[t]
        if auroc is not None and t in auroc:
            ttl += f"\nAUROC = {auroc[t]:.3f}"
        ax.set_title(ttl, fontsize=_LABEL_FS)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        title or "Ensemble decomposition — score distributions",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# Kernel size is a categorical identity with a FIXED colour/marker per
# value, shared with the architecture SVG so both views paint a kernel
# the same way (never cycled).
_KERNEL_STYLE = {3: (KERNEL_COLORS[3], "o"), 5: (KERNEL_COLORS[5], "s")}
_KERNEL_FALLBACK = ("#888888", "D")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (ties broken by order — fine for the
    handful of distinct architectures this annotates)."""
    if x.size < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    cx, cy = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((cx ** 2).sum() * (cy ** 2).sum())
    return float((cx * cy).sum() / denom) if denom > 0 else float("nan")


def plot_architecture_effect(
    records: Sequence[dict],
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """How each member's architecture shapes its OOD bias detection.

    ``records`` — one dict per ensemble member::

        {'member': 1-based index, 'channels': [c0..c4], 'kernel_size': int,
         'n_params': int, 'auroc': float, 'auroc_per_block': [float]}

    where ``auroc`` is the member's OWN reconstruction-error OOD AUROC
    (aggregated over blocks, same ``agg`` as the decomposition). Four
    panels: (a) AUROC vs parameter count, (b) AUROC vs total channel
    width, (c) AUROC grouped by conv kernel size, (d) the member × block
    AUROC heatmap. Kernel size keeps a fixed colour/marker everywhere
    (3 = blue circle, 5 = green square); members are direct-labelled
    ``M<i>`` so points can be traced back to ``model_<i>/``.
    """
    records = list(records)
    if not records:
        return
    members = [int(r["member"]) for r in records]
    kernels = [int(r["kernel_size"]) for r in records]
    params = np.asarray([float(r["n_params"]) for r in records])
    widths = np.asarray([float(sum(r["channels"])) for r in records])
    auroc = np.asarray([float(r["auroc"]) for r in records])
    per_block = np.asarray(
        [list(r["auroc_per_block"]) for r in records], dtype=float,
    )  # (M, n_blocks)
    n_blocks = per_block.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0),
                             constrained_layout=True)
    fig.suptitle(
        title or "Architecture vs OOD bias detection — per-member "
                 "reconstruction-error AUROC",
        fontsize=_TITLE_FS,
    )

    def _scatter(ax: plt.Axes, x: np.ndarray, xlabel: str,
                 log_x: bool = False) -> None:
        seen: set[int] = set()
        for i, k in enumerate(kernels):
            colour, marker = _KERNEL_STYLE.get(k, _KERNEL_FALLBACK)
            ax.scatter(x[i], auroc[i], s=52, color=colour, marker=marker,
                       zorder=3,
                       label=f"kernel {k}×{k}" if k not in seen else None)
            seen.add(k)
            ax.annotate(f"M{members[i]}", (x[i], auroc[i]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color="#444")
        # A log axis needs strictly positive data; fall back to linear
        # rather than let matplotlib warn and blank the panel.
        if log_x and np.all(x > 0):
            ax.set_xscale("log")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax.set_ylabel("OOD AUROC (member recon error)", fontsize=_LABEL_FS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_title(
            f"Spearman ρ = {_spearman(x, auroc):.2f}",
            fontsize=_LABEL_FS, loc="right", color="#555",
        )

    _scatter(axes[0, 0], params, "Trainable parameters", log_x=True)
    _scatter(axes[0, 1], widths, "Total channel width  Σ channels")

    # (c) grouped by kernel size: one jittered column per kernel value,
    # a horizontal bar at each group mean.
    ax = axes[1, 0]
    kvals = sorted(set(kernels))
    rng = np.random.default_rng(0)
    for gi, k in enumerate(kvals):
        colour, marker = _KERNEL_STYLE.get(k, _KERNEL_FALLBACK)
        ys = auroc[np.asarray(kernels) == k]
        xs = gi + rng.uniform(-0.10, 0.10, size=ys.size)
        ax.scatter(xs, ys, s=48, color=colour, marker=marker, zorder=3)
        ax.hlines(ys.mean(), gi - 0.22, gi + 0.22, color=colour,
                  linewidth=2.2)
        ax.annotate(f"mean {ys.mean():.3f}", (gi + 0.24, ys.mean()),
                    va="center", fontsize=8.5, color=colour)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(kvals)))
    ax.set_xticklabels([f"{k}×{k}" for k in kvals])
    ax.set_xlim(-0.6, len(kvals) - 0.4 + 0.5)
    ax.set_xlabel("Conv kernel size", fontsize=_LABEL_FS)
    ax.set_ylabel("OOD AUROC (member recon error)", fontsize=_LABEL_FS)
    ax.grid(alpha=0.25, axis="y")

    # (d) member × block AUROC heatmap.
    ax = axes[1, 1]
    im = ax.imshow(per_block, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(per_block), vmax=np.nanmax(per_block))
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels([f"L{k}" for k in range(n_blocks)])
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([
        f"M{m}  {'-'.join(str(c) for c in r['channels'])}  k{k}"
        for m, k, r in zip(members, kernels, records)
    ], fontsize=7.5)
    mid = (np.nanmin(per_block) + np.nanmax(per_block)) / 2
    for i in range(len(records)):
        for j in range(n_blocks):
            v = per_block[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7,
                    color="black" if v > mid else "white")
    ax.set_xlabel("Conv block", fontsize=_LABEL_FS)
    ax.set_title("Per-block AUROC by member", fontsize=_LABEL_FS)
    fig.colorbar(im, ax=ax, shrink=0.85, label="OOD AUROC")

    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Raw per-instance image exports (no axes, no titles, no colorbars)
# ---------------------------------------------------------------------------

def _overlay_composite(
    img_np: np.ndarray,
    cam: np.ndarray,
    overlay_power: float = 0.8,
    cmap: str = "inferno",
) -> np.ndarray:
    """Alpha-composite a smoothed cam onto an image, exactly as the overlay
    tiles render it (colour = cmap(cam), alpha = cam ** overlay_power), but
    as a plain ``(H, W, 3)`` array suitable for ``plt.imsave``."""
    rgba = plt.get_cmap(cmap)(np.clip(cam, 0.0, 1.0))[..., :3]
    alpha = np.clip(cam, 0.0, 1.0) ** overlay_power
    out = img_np * (1.0 - alpha[..., None]) + rgba * alpha[..., None]
    return np.clip(out, 0.0, 1.0)


def save_instance_raw_images(
    image: torch.Tensor,
    recons: Sequence[torch.Tensor],
    out_dir: str | Path,
    *,
    sigma: float = 1.5,
    overlay_power: float = 0.8,
    vmax: float = 0.5,
) -> None:
    """Bare image files for ONE instance — no legend, axes or colorbars.

    Writes into ``out_dir`` (created if needed), one standalone PNG each::

        original.png       the input face
        bias_overlay.png   smoothed-bias overlay composited on the face
        bias.png           raw bias map, viridis on the shared [0, vmax]
        mean_error.png     raw mean-error map, same scale
        min_error.png      raw min-error map, autoscaled to its own range

    Same derivations as :func:`plot_instance_summary` (via
    :func:`_instance_summary_maps`), so these are pixel-identical to the
    summary tiles — just without the figure chrome, ready for slides or a
    paper's own layout.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_np, bias, mean_err, min_err = _instance_summary_maps(image, recons)
    cam = smooth_cam(bias, sigma=sigma)

    plt.imsave(out_dir / "original.png", np.clip(img_np, 0.0, 1.0))
    plt.imsave(
        out_dir / "bias_overlay.png",
        _overlay_composite(img_np, cam, overlay_power),
    )
    viridis = plt.get_cmap("viridis")
    plt.imsave(out_dir / "bias.png",
               viridis(np.clip(bias / vmax, 0.0, 1.0)))
    plt.imsave(out_dir / "mean_error.png",
               viridis(np.clip(mean_err / vmax, 0.0, 1.0)))
    peak = float(min_err.max())
    plt.imsave(out_dir / "min_error.png",
               viridis(min_err / peak if peak > 1e-8 else min_err))


def plot_instance_all_models(
    image: torch.Tensor,
    recons: Sequence[torch.Tensor],
    save_path: str | Path,
    *,
    sigma: float = 1.5,
    overlay_power: float = 0.8,
    title: str | None = None,
) -> None:
    """Every member side by side for ONE image, as overlays.

    Tiles: Original | one error overlay PER MEMBER (that member's squared
    error, smoothed and composited like the bias overlay) | Bias overlay |
    Mean-error overlay | Min-error overlay. The per-member tiles make the
    disagreement visible (which members fail where); the last three tiles
    are the ensemble summaries of the same maps.
    """
    img_np, bias, mean_err, min_err = _instance_summary_maps(image, recons)
    recon_np = [_single_image(r) for r in recons]
    member_maps = [_sq_error(img_np, rc) for rc in recon_np]

    tiles: list[tuple[str, np.ndarray | None]] = [("Original", None)]
    tiles += [
        (f"$M_{{{m + 1}}}$", member_maps[m]) for m in range(len(recons))
    ]
    tiles += [
        (r"Bias  $(x - \bar{f})^2$", bias),
        ("Mean error", mean_err),
        ("Min error", min_err),
    ]

    n = len(tiles)
    ncols = int(np.ceil(n / 2)) if n > 7 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.55 * ncols, 1.75 * nrows),
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, mp) in zip(axes, tiles):
        ax.imshow(img_np)
        if mp is not None:
            cam = smooth_cam(mp, sigma=sigma)
            ax.imshow(cam, cmap="inferno",
                      alpha=np.clip(cam, 0.0, 1.0) ** overlay_power)
        _bare(ax)
        ax.set_title(name, fontsize=_LABEL_FS - 2)
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fused-evaluation panel (per-signal AUROC bars + ROC + fused histograms)
# ---------------------------------------------------------------------------

def plot_fused_auroc_panel(
    auroc: dict,
    save_path: str | Path,
    *,
    fused_in: np.ndarray | None = None,
    fused_ood: np.ndarray | None = None,
    fused_key: str = "fused",
    title: str | None = None,
) -> None:
    """Visual summary of a fused OOD evaluation (lrad.fusion).

    ``auroc`` is the ``out["auroc"]`` mapping of ``evaluate_fused_ood`` /
    ``evaluate_supervised_fusion`` — each entry holds at least ``auroc``
    and, when present, ``fpr``/``tpr``. Three panels:

      * left — horizontal AUROC bars, one per signal, fused entries in the
        OOD accent colour, chance line at 0.5;
      * middle — ROC curves of the fused entries plus the best single
        signal, with the y = x chance diagonal;
      * right — score histograms of ``fused_in`` vs ``fused_ood`` for
        ``fused_key`` (panel omitted when the arrays are not given).
    """
    entries = {
        k: v for k, v in auroc.items()
        if isinstance(v, dict) and np.isfinite(v.get("auroc", float("nan")))
    }
    if not entries:
        raise ValueError("no finite AUROC entries to plot")
    fused_names = [k for k in entries if k.startswith("fused")]
    signal_names = [k for k in entries if not k.startswith("fused")]

    n_panels = 3 if fused_in is not None and fused_ood is not None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.4 * n_panels, 3.6), layout="constrained",
    )

    # --- AUROC bars ------------------------------------------------------
    order = sorted(entries, key=lambda k: entries[k]["auroc"])
    ys = np.arange(len(order))
    vals = [entries[k]["auroc"] for k in order]
    cols = [_C_OOD if k.startswith("fused") else _C_ID for k in order]
    ax = axes[0]
    ax.barh(ys, vals, color=cols, height=0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=_TICK_FS)
    ax.axvline(0.5, color="#888888", lw=0.8, ls="--")
    for y, v in zip(ys, vals):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0.45, min(1.0, max(vals) + 0.06))
    ax.set_xlabel("OOD AUROC", fontsize=_LABEL_FS)
    ax.set_title("Per-signal AUROC", fontsize=_LABEL_FS)
    ax.tick_params(labelsize=_TICK_FS)

    # --- ROC curves ------------------------------------------------------
    ax = axes[1]
    best_signal = (max(signal_names, key=lambda k: entries[k]["auroc"])
                   if signal_names else None)
    show = ([best_signal] if best_signal else []) + fused_names
    palette = [_C_ACCENT, _C_OOD, _C_BIAS, _C_VARIANCE]
    for k, c in zip(show, palette):
        e = entries[k]
        if "fpr" not in e or "tpr" not in e:
            continue
        ax.plot(e["fpr"], e["tpr"], color=c, lw=1.6,
                label=f"{k}  ({e['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--")
    ax.set_xlabel("False positive rate", fontsize=_LABEL_FS)
    ax.set_ylabel("True positive rate", fontsize=_LABEL_FS)
    ax.set_title("ROC", fontsize=_LABEL_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.legend(fontsize=8, loc="lower right")

    # --- fused histograms ------------------------------------------------
    if n_panels == 3:
        ax = axes[2]
        both = np.concatenate([fused_in, fused_ood])
        bins = np.linspace(both.min(), both.max(), 40)
        ax.hist(fused_in, bins=bins, alpha=0.6, density=True,
                color=_C_ID, label=f"in-dist (n={fused_in.size})")
        ax.hist(fused_ood, bins=bins, alpha=0.6, density=True,
                color=_C_OOD, label=f"OOD (n={fused_ood.size})")
        ax.set_xlabel(f"{fused_key} score", fontsize=_LABEL_FS)
        ax.set_ylabel("Density", fontsize=_LABEL_FS)
        au = entries.get(fused_key, {}).get("auroc")
        ax.set_title(
            f"{fused_key}" + (f"  AUROC = {au:.3f}" if au else ""),
            fontsize=_LABEL_FS,
        )
        ax.tick_params(labelsize=_TICK_FS)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation study — arm-comparison figures (scripts/compare_ablation.py)
# ---------------------------------------------------------------------------
#
# Four arms, two ablated factors (architecture diversity x CutPaste), one
# common control. Every figure keys the arms to FIXED colours — an arm never
# changes colour between figures, and the baseline is deliberately the
# neutral gray so the treated arms carry the hue. The palette is the
# project's Okabe-Ito set (colorblind-safe); identity is never colour
# alone: every figure carries a legend and/or per-arm axis labels.

ABLATION_ARM_COLORS = {
    "baseline": "#888888",       # control: same arch, no CutPaste (gray)
    "arch": _C_BIAS,             # architecture diversity     (blue)
    "cutpaste": _C_VARIANCE,     # CutPaste pretext           (orange)
    "arch_cutpaste": _C_ACCENT,  # both factors               (green)
}


def _arm_color(arm: dict) -> str:
    return arm.get("color") or ABLATION_ARM_COLORS.get(arm["arm"], "#444444")


def plot_ablation_auroc_bars(
    arms: Sequence[dict],
    metrics: Sequence[tuple[str, str]],
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Grouped AUROC bars — one group per metric, one bar per arm.

    ``arms`` is a list of ``{arm, label, metrics: {name: auroc}}`` dicts;
    ``metrics`` lists ``(key, display_label)`` pairs in plot order. A metric
    an arm does not have (e.g. ``cutpaste_prob`` on a no-pretext arm) simply
    leaves a gap in that group.

    The y-axis starts just below the smallest plotted value (never above
    the 0.5 chance line) instead of at 0 — every method here scores in the
    0.5-0.9 band and a zero-based axis would compress exactly the
    differences the ablation exists to show. The truncation is explicit in
    the axis label, and the chance line is drawn so the anchor stays
    visible.
    """
    keys = [k for k, _ in metrics]
    vals = np.array([
        [
            (arm["metrics"].get(k) if arm["metrics"].get(k) is not None
             else np.nan)
            for k in keys
        ]
        for arm in arms
    ], dtype=float)
    if not np.isfinite(vals).any():
        raise ValueError("no finite metric values to plot")

    n_arms = len(arms)
    x = np.arange(len(keys))
    width = 0.8 / n_arms
    lo = min(0.45, np.nanmin(vals) - 0.03)
    lo = max(0.0, lo)

    fig, ax = plt.subplots(
        figsize=(1.6 + 1.05 * len(keys), 4.4), layout="constrained",
    )
    for i, arm in enumerate(arms):
        ax.bar(
            x + (i - (n_arms - 1) / 2) * width, vals[i] - lo, width * 0.92,
            bottom=lo, color=_arm_color(arm), label=arm["label"],
        )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    ax.text(
        len(keys) - 0.42, 0.5, "chance", fontsize=8, color="gray",
        ha="right", va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=_TICK_FS,
                       rotation=20, ha="right")
    ylabel = "OOD AUROC"
    if lo > 0:
        ylabel += f"  (axis starts at {lo:.2f})"
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
    ax.set_ylim(lo, min(1.0, np.nanmax(vals) + 0.06))
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(labelsize=_TICK_FS)
    ax.legend(loc="upper right", fontsize=9, ncols=min(n_arms, 2))
    fig.suptitle(title or "Ablation — OOD AUROC by arm", fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_deltas(
    cases: Sequence[dict],
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """The three case studies: signed AUROC deltas against the baseline.

    ``cases`` is a list of ``{label, deltas: {metric_label: delta}}`` — one
    horizontal-bar panel per case study, all panels on a SHARED x-scale so
    the eye can compare effect sizes across cases. The encoding is
    diverging (this is polarity, not identity): improvements point right in
    the project's accent green, regressions left in the OOD vermillion,
    with a zero line. Each bar carries its value — the delta IS the
    result, so here every label is load-bearing rather than clutter.
    """
    cases = [c for c in cases if c.get("deltas")]
    if not cases:
        raise ValueError("no case studies with deltas to plot")

    all_vals = [v for c in cases for v in c["deltas"].values()]
    span = max(0.02, max(abs(v) for v in all_vals) * 1.25)
    n_rows = max(len(c["deltas"]) for c in cases)

    fig, axes = plt.subplots(
        1, len(cases),
        figsize=(3.6 * len(cases), 0.42 * n_rows + 1.8),
        layout="constrained", sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        names = list(case["deltas"].keys())
        vals = np.array([case["deltas"][n] for n in names], dtype=float)
        y = np.arange(len(names))[::-1]
        colors = [_C_ACCENT if v >= 0 else _C_OOD for v in vals]
        ax.barh(y, vals, height=0.62, color=colors)
        for yi, v in zip(y, vals):
            ax.annotate(
                f"{v:+.3f}", (v, yi), fontsize=8,
                xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if v >= 0 else "right", va="center",
            )
        ax.axvline(0.0, color="#444444", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=_TICK_FS)
        ax.set_xlim(-span, span)
        ax.set_xlabel("Δ AUROC vs baseline", fontsize=_LABEL_FS)
        ax.set_title(case["label"], fontsize=_LABEL_FS)
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(labelsize=_TICK_FS)
    fig.suptitle(
        title or "Ablation case studies — what each ingredient buys",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_per_block(
    arms: Sequence[dict],
    save_path: str | Path,
    *,
    terms: Sequence[str] = ("risk", "bias", "variance"),
    title: str | None = None,
) -> None:
    """Per-block decomposition AUROC, one line per arm, one panel per term.

    ``arms`` is a list of ``{arm, label, per_block: {term: [auroc, ...]}}``
    dicts. Shows WHERE in the trunk each ingredient helps: architecture
    diversity should move the deep-block variance signal, the pretext the
    mid-block bias signal.
    """
    fig, axes = plt.subplots(
        1, len(terms), figsize=(3.6 * len(terms), 3.6),
        layout="constrained", sharey=True,
    )
    axes = np.atleast_1d(axes)
    drew = False
    for ax, term in zip(axes, terms):
        for arm in arms:
            series = arm.get("per_block", {}).get(term)
            if not series:
                continue
            blocks = np.arange(len(series))
            ax.plot(
                blocks, series, color=_arm_color(arm), linewidth=1.8,
                marker="o", markersize=4.5, label=arm["label"],
            )
            drew = True
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        n_blocks = max(
            (len(a.get("per_block", {}).get(term) or ()) for a in arms),
            default=0,
        )
        ax.set_xticks(np.arange(n_blocks))
        ax.set_xticklabels(
            [f"L{k}" for k in range(n_blocks)], fontsize=_TICK_FS,
        )
        ax.set_xlabel("conv block", fontsize=_LABEL_FS)
        ax.set_title(term.capitalize(), fontsize=_LABEL_FS)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=_TICK_FS)
    if not drew:
        plt.close(fig)
        raise ValueError("no per-block series to plot")
    axes[0].set_ylabel("OOD AUROC", fontsize=_LABEL_FS)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        title or "Ablation — per-block decomposition AUROC",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_members(
    arms: Sequence[dict],
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Per-member spread by arm: recon AUROC (left), gender accuracy (right).

    ``arms`` is a list of ``{arm, label, members: {auroc_recon: [...],
    gender_acc: [...]}}`` dicts. Each member is one dot (deterministic
    horizontal jitter by member index — same member, same offset in both
    panels), the arm mean a horizontal dash. The spread is the point: a
    diverse ensemble should show a WIDER member distribution while the
    ensemble score improves — evidence the gain comes from decorrelation,
    not from stronger individual members.
    """
    panels = (("auroc_recon", "per-member recon AUROC"),
              ("gender_acc", "per-member gender accuracy"))
    fig, axes = plt.subplots(
        1, 2, figsize=(2.1 * max(len(arms), 2) + 3.2, 3.8),
        layout="constrained",
    )
    drew = False
    for ax, (key, label) in zip(axes, panels):
        for i, arm in enumerate(arms):
            vals = [v for v in arm.get("members", {}).get(key, [])
                    if v is not None]
            if not vals:
                continue
            vals = np.asarray(vals, dtype=float)
            jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2)
            jitter = jitter / max(len(vals) - 1, 1) * 0.42
            ax.scatter(
                i + jitter, vals, s=26, color=_arm_color(arm),
                alpha=0.75, linewidths=0, zorder=3,
            )
            ax.hlines(
                float(vals.mean()), i - 0.30, i + 0.30,
                color=_arm_color(arm), linewidth=2.2, zorder=4,
            )
            drew = True
        if key == "auroc_recon":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(
            [a["label"] for a in arms], fontsize=_TICK_FS,
            rotation=15, ha="right",
        )
        ax.set_ylabel(label, fontsize=_LABEL_FS)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=_TICK_FS)
    if not drew:
        plt.close(fig)
        raise ValueError("no member series to plot")
    fig.suptitle(
        title or "Ablation — member spread by arm (dash = arm mean)",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
