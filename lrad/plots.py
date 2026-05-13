"""Per-block reconstruction visualizations.

Two figures are produced for a small set of sample images, mixing
in-distribution and OOD samples with row labels:

  * ``plot_per_block_breakdown`` — for every conv block ``k``, three
    columns side by side: the original image, the absolute
    reconstruction error ``|x − recon_k|`` (viridis), and the
    reconstruction itself. Layout::

        Original  Err L0  Recon L0  Err L1  Recon L1  ...  Err Ln  Recon Ln

  * ``plot_recons_only`` — same rows, just original + per-block
    reconstructions::

        Original  Recon L0  Recon L1  ...  Recon Ln

Implementation follows current matplotlib best practices:
``layout='constrained'``, viridis colormap for error maps, hidden
spines/ticks on image axes, dpi=150 PNG export with tight bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


_TITLE_FS = 14
_LABEL_FS = 12
_TICK_FS = 10


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
        (axes[0], gender, gender_smooth, "Gender accuracy", "#1f4e79"),
        (axes[1], attrs,  attrs_smooth,  "Mean attribute accuracy", "#5a1f6e"),
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

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_image_grid(t: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) tensor in [0, 1] → (B, H, W, 3) numpy in [0, 1]."""
    return t.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()


def _abs_error(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """(H, W) mean absolute error across RGB channels."""
    return np.abs(orig - recon).mean(axis=-1)


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

    Error tiles share a single global colour scale and a colour bar is
    drawn on the right of the figure. Each error tile is annotated with
    its mean absolute error score (white text with a black outline).
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + 2 * n_blocks

    # Pre-compute every error map and a shared global vmax so the colour
    # bar is meaningful across the whole figure.
    err_maps: list[list[np.ndarray]] = []
    global_max = 0.0
    for r in range(n_rows):
        row = []
        for k in range(n_blocks):
            err = _abs_error(images_np[r], recons_np[k][r])
            row.append(err)
            global_max = max(global_max, float(err.max()))
        err_maps.append(row)
    vmax = global_max if global_max > 0 else 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows

    err_im = None
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]

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
            score = float(err.mean())
            ax_err.text(
                0.5, 0.04, f"{score:.3f}",
                transform=ax_err.transAxes,
                ha="center", va="bottom",
                fontsize=8.5, color="white",
                path_effects=text_pe,
            )

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
        cbar.set_label("|x − recon|  (mean over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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

    where ``err_map_k = |x − recon_k|`` averaged over RGB, the fused map
    is the per-pixel max across all blocks, and the overlay renders the
    fused heatmap on top of the original image. The scalar anomaly score
    annotated under each fused tile is ``fused.max()`` — the most
    surprising pixel for that sample.

    Error tiles, the fused tile, and the overlay all share a single
    global colour scale and a single colour bar on the right.
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    H, W = images_np.shape[1], images_np.shape[2]
    n_cols = 1 + n_blocks + 2  # Original | Err Lk × n | Fused | Overlay

    err_maps: list[list[np.ndarray]] = []
    fused_maps: list[np.ndarray] = []
    anomaly_scores: list[float] = []
    global_max = 0.0
    for r in range(n_rows):
        row = []
        for k in range(n_blocks):
            err = _abs_error(images_np[r], recons_np[k][r])
            row.append(err)
            global_max = max(global_max, float(err.max()))
        err_maps.append(row)
        fused = np.maximum.reduce(row)  # per-pixel max across blocks
        fused_maps.append(fused)
        anomaly_scores.append(float(fused.max()))
        global_max = max(global_max, float(fused.max()))
    vmax = global_max if global_max > 0 else 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]

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
            ax_err.text(
                0.5, 0.04, f"{float(err.mean()):.3f}",
                transform=ax_err.transAxes,
                ha="center", va="bottom",
                fontsize=8.5, color="white",
                path_effects=text_pe,
            )

        fused = fused_maps[r]
        ax_fused = axes[r, 1 + n_blocks]
        im = ax_fused.imshow(fused, cmap="viridis", vmin=0.0, vmax=vmax)
        err_im = im
        _bare(ax_fused)
        if r == 0:
            ax_fused.set_title("Fused (max)", fontsize=_LABEL_FS)
        ax_fused.text(
            0.5, 0.04, f"score={anomaly_scores[r]:.3f}",
            transform=ax_fused.transAxes,
            ha="center", va="bottom",
            fontsize=8.5, color="white",
            path_effects=text_pe,
        )

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
        cbar.set_label("|x − recon|  (mean over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    ax_h.hist(in_scores, bins=bins, alpha=0.55, density=True,
              color="#377eb8", label=f"normal  (n={in_scores.size})")
    ax_h.hist(ood_scores, bins=bins, alpha=0.55, density=True,
              color="#e41a1c", label=f"anomaly (n={ood_scores.size})")
    ax_h.set_xlabel("Fused anomaly score  (max of per-pixel fused map)",
                    fontsize=_LABEL_FS)
    ax_h.set_ylabel("Density", fontsize=_LABEL_FS)
    ax_h.set_title("Score distribution", fontsize=_LABEL_FS)
    ax_h.tick_params(labelsize=_TICK_FS)
    ax_h.grid(alpha=0.3)
    ax_h.legend(fontsize=9)

    ax_r = axes[1]
    ax_r.plot(fpr, tpr, color="#1f4e79", linewidth=1.6,
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
