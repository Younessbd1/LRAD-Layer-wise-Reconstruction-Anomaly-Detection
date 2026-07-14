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

    def _roll(arr: np.ndarray) -> np.ndarray:
        """Causal rolling mean (no look-ahead)."""
        out = np.empty_like(arr)
        cs = np.cumsum(arr)
        out[:w] = cs[:w] / np.arange(1, min(w, n) + 1)
        out[w:] = (cs[w:] - cs[:-w]) / w
        return out

    gender_smooth = _roll(gender)
    attrs_smooth = _roll(attrs)

    # epoch boundary x-positions and start indices
    epoch_ends_arr = np.asarray(epoch_ends, dtype=float)
    epoch_starts = np.concatenate([[0], epoch_ends_arr[:-1] + 1])

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

        # epoch boundary lines + epoch number label
        for i, end in enumerate(epoch_ends_arr):
            if end < n - 1:
                ax.axvline(end + 0.5, color="gray", linestyle="--",
                           linewidth=0.75, alpha=0.55)
            mid = (epoch_starts[i] + end) / 2
            ax.text(mid, 0.475, f"E{i + 1}", ha="center", va="bottom",
                    fontsize=7.5, color="#555", clip_on=True)

        ax.set_xlabel("Batch (global index)", fontsize=_LABEL_FS)
        ax.set_ylabel("Accuracy", fontsize=_LABEL_FS)
        ax.set_title(title, fontsize=_LABEL_FS)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

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


def plot_fusion_auroc(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
    save_path: str | Path,
    *,
    title: str | None = None,
) -> float:
    """ROC curve for the per-image fusion-based anomaly score.

    ``in_scores`` and ``ood_scores`` are scalar per-image scores
    (typically ``fused.max()``). Returns the AUROC value.

    The figure has two panels:

      * left  — score histogram for in-dist vs OOD (density-normalized).
      * right — ROC curve with AUROC annotated and the ``y = x`` chance
                line for reference.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    in_scores = np.asarray(in_scores).ravel()
    ood_scores = np.asarray(ood_scores).ravel()
    labels = np.concatenate([
        np.zeros(in_scores.shape[0]),
        np.ones(ood_scores.shape[0]),
    ])
    scores = np.concatenate([in_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             constrained_layout=True)

    ax_h = axes[0]
    bins = 50
    ax_h.hist(in_scores, bins=bins, alpha=0.6, density=True,
              color=_C_ID, label=f"normal  (n={in_scores.size})")
    ax_h.hist(ood_scores, bins=bins, alpha=0.6, density=True,
              color=_C_OOD, label=f"anomaly (n={ood_scores.size})")
    ax_h.set_xlabel("Fused anomaly score  (max of per-pixel fused map)",
                    fontsize=_LABEL_FS)
    ax_h.set_ylabel("Density", fontsize=_LABEL_FS)
    ax_h.set_title("Score distribution", fontsize=_LABEL_FS)
    ax_h.tick_params(labelsize=_TICK_FS)
    ax_h.grid(alpha=0.3)
    ax_h.legend(fontsize=9)

    ax_r = axes[1]
    ax_r.plot(fpr, tpr, color=_C_BIAS, linewidth=1.6,
              label=f"fused (max)  AUROC = {auroc:.3f}")
    ax_r.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.7)
    ax_r.set_xlabel("False Positive Rate", fontsize=_LABEL_FS)
    ax_r.set_ylabel("True Positive Rate", fontsize=_LABEL_FS)
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.set_title("ROC — anomaly vs normal", fontsize=_LABEL_FS)
    ax_r.tick_params(labelsize=_TICK_FS)
    ax_r.grid(alpha=0.3)
    ax_r.legend(loc="lower right", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return auroc


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
