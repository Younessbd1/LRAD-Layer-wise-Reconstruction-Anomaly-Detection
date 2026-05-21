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

A second family of figures visualizes the bias/variance **debiasing**
of the error maps (see ``lrad.anomaly_score``):

  * ``plot_bias_variance_maps`` — the estimated per-pixel bias ``mu_k``
    and variance ``sigma_k`` for every block.
  * ``plot_anomaly_cleaning_comparison`` — raw vs cleaned error maps
    side by side, showing that clean ID faces lose their error while
    OOD anomalies survive the debiasing.
  * ``plot_score_distribution_comparison`` — ID vs OOD score histograms
    for the raw vs the debiased score, AUROC annotated.
  * ``plot_per_block_auroc_bars`` — per-block AUROC before/after
    debiasing, to see which block benefits most.

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


# ---------------------------------------------------------------------------
# Bias / variance debiasing visualizations
# ---------------------------------------------------------------------------

def _stat_np(t) -> np.ndarray:
    """(H, W) torch tensor (or array) → float32 numpy on CPU."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _clean_np(
    err: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy mirror of the cleaned map: max(0, (err - mu) / (sigma+eps))."""
    return np.maximum((err - mu) / (sigma + eps), 0.0)


def plot_bias_variance_maps(
    baseline_stats: dict,
    save_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Per-pixel reconstruction-error **bias** (μ) and **variance** (σ).

    One column per conv block, two rows::

                     Block 0   Block 1   Block 2   ...
        μ (bias)     [heatmap] [heatmap] [heatmap]
        σ (variance) [heatmap] [heatmap] [heatmap]

    The colour scale is shared **per column** (μ and σ of the same block
    use one vmax and one colour bar underneath), so within a block you
    can read directly whether the systematic bias dominates the residual
    variance, while deeper blocks — whose errors are intrinsically larger
    — are not washed out by a single global scale.
    """
    ks = sorted(baseline_stats.keys())
    n_blocks = len(ks)
    mus = [_stat_np(baseline_stats[k]["mu"]) for k in ks]
    sigmas = [_stat_np(baseline_stats[k]["sigma"]) for k in ks]

    fig, axes = plt.subplots(
        2, n_blocks,
        figsize=(1.9 * n_blocks + 0.6, 4.4),
        layout="constrained",
        squeeze=False,
    )

    for j, k in enumerate(ks):
        mu, sigma = mus[j], sigmas[j]
        vmax = max(float(mu.max()), float(sigma.max()))
        vmax = vmax if vmax > 0 else 1.0

        axes[0, j].imshow(mu, cmap="viridis", vmin=0.0, vmax=vmax)
        im = axes[1, j].imshow(sigma, cmap="viridis", vmin=0.0, vmax=vmax)
        for ax in (axes[0, j], axes[1, j]):
            _bare(ax)
        axes[0, j].set_title(f"Block {k}", fontsize=_LABEL_FS)
        axes[0, j].text(
            0.5, 0.03, f"μ̄={mu.mean():.3f}", transform=axes[0, j].transAxes,
            ha="center", va="bottom", fontsize=8, color="white",
        )
        axes[1, j].text(
            0.5, 0.03, f"σ̄={sigma.mean():.3f}",
            transform=axes[1, j].transAxes,
            ha="center", va="bottom", fontsize=8, color="white",
        )
        cbar = fig.colorbar(
            im, ax=[axes[0, j], axes[1, j]], location="bottom",
            shrink=0.9, pad=0.02, aspect=22,
        )
        cbar.ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("μ  (bias)", fontsize=_LABEL_FS)
    axes[1, 0].set_ylabel("σ  (variance)", fontsize=_LABEL_FS)

    fig.suptitle(
        title or "Per-pixel reconstruction-error bias & variance",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_cleaning_comparison(
    images: torch.Tensor,
    recons: Sequence[torch.Tensor],
    baseline_stats: dict,
    save_path: str | Path,
    *,
    row_labels: Iterable[str] | None = None,
    title: str | None = None,
    eps: float = 1e-6,
) -> None:
    """Raw vs debiased error maps, side by side, for a few images.

    For every block ``k`` two columns are drawn — the raw error
    ``|x − recon_k|`` and the cleaned map
    ``max(0, (err − μ_k) / (σ_k + eps))`` (in σ units)::

        Original | Err raw L0 | Err clean L0 | Err raw L1 | Err clean L1 | …

    All raw tiles share one colour scale/bar; all cleaned tiles share
    another (the cleaned maps live in σ-units and are typically much
    larger). The point to read off: clean **ID** rows have substantial
    *raw* error that all but vanishes once cleaned, whereas **OOD** rows
    keep a bright localized blob (the glasses) after cleaning — that
    surviving signal is the true anomaly.
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + 2 * n_blocks

    ks = sorted(baseline_stats.keys())
    mus = [_stat_np(baseline_stats[k]["mu"]) for k in ks]
    sigmas = [_stat_np(baseline_stats[k]["sigma"]) for k in ks]

    raw_maps: list[list[np.ndarray]] = []
    clean_maps: list[list[np.ndarray]] = []
    raw_vmax = clean_vmax = 0.0
    for r in range(n_rows):
        raw_row, clean_row = [], []
        for j in range(n_blocks):
            err = _abs_error(images_np[r], recons_np[j][r])
            cln = _clean_np(err, mus[j], sigmas[j], eps)
            raw_row.append(err)
            clean_row.append(cln)
            raw_vmax = max(raw_vmax, float(err.max()))
            clean_vmax = max(clean_vmax, float(cln.max()))
        raw_maps.append(raw_row)
        clean_maps.append(clean_row)
    raw_vmax = raw_vmax if raw_vmax > 0 else 1.0
    clean_vmax = clean_vmax if clean_vmax > 0 else 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.45 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]

    raw_im = clean_im = None
    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for j in range(n_blocks):
            err = raw_maps[r][j]
            cln = clean_maps[r][j]
            ax_raw = axes[r, 1 + 2 * j]
            ax_cln = axes[r, 2 + 2 * j]
            raw_im = ax_raw.imshow(err, cmap="viridis",
                                   vmin=0.0, vmax=raw_vmax)
            clean_im = ax_cln.imshow(cln, cmap="viridis",
                                     vmin=0.0, vmax=clean_vmax)
            _bare(ax_raw)
            _bare(ax_cln)
            if r == 0:
                ax_raw.set_title(f"Raw L{j}", fontsize=_LABEL_FS)
                ax_cln.set_title(f"Clean L{j}", fontsize=_LABEL_FS)
            ax_raw.text(
                0.5, 0.04,
                f"max={err.max():.2f}\nμ={err.mean():.3f}",
                transform=ax_raw.transAxes, ha="center", va="bottom",
                fontsize=7.5, color="white", path_effects=text_pe,
            )
            ax_cln.text(
                0.5, 0.04,
                f"max={cln.max():.1f}σ\nμ={cln.mean():.2f}σ",
                transform=ax_cln.transAxes, ha="center", va="bottom",
                fontsize=7.5, color="white", path_effects=text_pe,
            )

    if raw_im is not None:
        cb_raw = fig.colorbar(
            raw_im, ax=axes[:, 1::2].ravel().tolist(),
            location="right", shrink=0.85, pad=0.012, aspect=30,
        )
        cb_raw.set_label("raw |x − recon|", fontsize=_LABEL_FS)
        cb_raw.ax.tick_params(labelsize=_TICK_FS)
    if clean_im is not None:
        cb_cln = fig.colorbar(
            clean_im, ax=axes[:, 2::2].ravel().tolist(),
            location="right", shrink=0.85, pad=0.012, aspect=30,
        )
        cb_cln.set_label("cleaned anomaly  (σ units)", fontsize=_LABEL_FS)
        cb_cln.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution_comparison(
    raw_in: np.ndarray,
    raw_ood: np.ndarray,
    clean_in: np.ndarray,
    clean_ood: np.ndarray,
    save_path: str | Path,
    *,
    raw_auroc: float | None = None,
    clean_auroc: float | None = None,
    title: str | None = None,
) -> None:
    """ID vs OOD score histograms — raw error vs debiased, side by side.

    Left panel: the raw aggregated error score. Right panel: the
    debiased/whitened score (same aggregation, σ units). The AUROC is
    annotated on each panel; a working debiasing widens the ID/OOD
    separation, so the right panel should show much less overlap and a
    higher AUROC than the left.
    """
    from sklearn.metrics import roc_auc_score

    def _auroc(in_s, ood_s):
        in_s = np.asarray(in_s).ravel()
        ood_s = np.asarray(ood_s).ravel()
        y = np.concatenate([np.zeros(in_s.size), np.ones(ood_s.size)])
        s = np.concatenate([in_s, ood_s])
        if not np.isfinite(s).all() or np.unique(y).size != 2:
            return float("nan")
        return float(roc_auc_score(y, s))

    if raw_auroc is None:
        raw_auroc = _auroc(raw_in, raw_ood)
    if clean_auroc is None:
        clean_auroc = _auroc(clean_in, clean_ood)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             constrained_layout=True)
    panels = [
        (axes[0], np.asarray(raw_in).ravel(), np.asarray(raw_ood).ravel(),
         "Raw error score", raw_auroc),
        (axes[1], np.asarray(clean_in).ravel(),
         np.asarray(clean_ood).ravel(),
         "Debiased score  (σ units)", clean_auroc),
    ]
    for ax, in_s, ood_s, name, auroc in panels:
        bins = 50
        ax.hist(in_s, bins=bins, alpha=0.55, density=True,
                color="#377eb8", label=f"in-dist (n={in_s.size})")
        ax.hist(ood_s, bins=bins, alpha=0.55, density=True,
                color="#e41a1c", label=f"OOD (n={ood_s.size})")
        ax.set_xlabel(name, fontsize=_LABEL_FS)
        ax.set_ylabel("Density", fontsize=_LABEL_FS)
        ax.set_title(f"{name}\nAUROC = {auroc:.3f}", fontsize=_LABEL_FS)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        title or "Score distribution — raw vs debiased",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_block_auroc_bars(
    auroc_raw: Sequence[float],
    auroc_debiased: Sequence[float],
    save_path: str | Path,
    *,
    block_labels: Sequence[str] | None = None,
    aggregated: tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    """Per-block OOD AUROC before vs after debiasing (grouped bars).

    Two horizontal bars per block — raw error AUROC and debiased AUROC —
    so it is immediately visible which block carries the most
    debiasing-recoverable anomaly signal (typically the early, high-
    resolution blocks, whose raw error is dominated by decoder bias).
    Optionally appends an "aggregated" group combining all blocks.
    """
    raw = list(auroc_raw)
    deb = list(auroc_debiased)
    n = len(raw)
    names = (
        list(block_labels) if block_labels is not None
        else [f"Block {k}" for k in range(n)]
    )
    if aggregated is not None:
        raw = raw + [aggregated[0]]
        deb = deb + [aggregated[1]]
        names = names + ["Aggregated"]

    y = np.arange(len(names))[::-1]  # block 0 on top
    h = 0.38

    fig, ax = plt.subplots(
        figsize=(7.5, 0.7 * len(names) + 1.4), constrained_layout=True,
    )
    b_raw = ax.barh(y + h / 2, raw, height=h, color="#9bbcd6",
                    label="raw error")
    b_deb = ax.barh(y - h / 2, deb, height=h, color="#1f4e79",
                    label="debiased")
    ax.bar_label(b_raw, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b_deb, fmt="%.3f", fontsize=8, padding=2)

    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=_LABEL_FS)
    ax.set_xlabel("OOD AUROC", fontsize=_LABEL_FS)
    ax.set_xlim(0.0, 1.05)
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        title or "Per-block OOD AUROC — raw vs debiased",
        fontsize=_TITLE_FS,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
