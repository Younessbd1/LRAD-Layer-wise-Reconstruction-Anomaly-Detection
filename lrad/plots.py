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
  * ``plot_bias_variance_vs_block`` — how the mean bias and variance
    evolve with conv-block depth, ID vs OOD.
  * ``plot_bias_variance_vs_percentile`` — how the per-image bias and
    variance scores separate ID from OOD across aggregation percentiles.
  * ``plot_ensemble_decomposition`` / ``plot_decomposition_auroc_bars`` /
    ``plot_ensemble_score_hists`` — the full Risk | Bias | Variance maps,
    per-block AUROC bars, and aggregated score histograms.

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


def _sq_error(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """(H, W) mean squared error across RGB channels.

    Squared (not absolute) so every error tile is on the same L2 footing as
    the ensemble bias/variance decomposition, where ``Risk = Bias + Variance``
    only holds for squared error.
    """
    return ((orig - recon) ** 2).mean(axis=-1)


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

    Error tiles use the fixed ``[0, 1]`` colour scale (images live in
    ``[0, 1]`` so the squared error does too), never a per-figure max, so
    error tiles are directly comparable across different figures. A single
    colour bar is drawn on the right. Each error tile is annotated with its
    mean squared error score (white text with a black outline).
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + 2 * n_blocks

    # Pre-compute every error map. The colour scale is fixed to [0, 1]
    # (no data-dependent normalization) so any two figures are comparable.
    err_maps: list[list[np.ndarray]] = []
    for r in range(n_rows):
        row = [_sq_error(images_np[r], recons_np[k][r])
               for k in range(n_blocks)]
        err_maps.append(row)
    vmax = 1.0

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
        cbar.set_label("(x − recon)²  (mean over RGB)", fontsize=_LABEL_FS)
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

    where ``err_map_k = (x − recon_k)²`` averaged over RGB, the fused map
    is the per-pixel max across all blocks, and the overlay renders the
    fused heatmap on top of the original image. The scalar anomaly score
    annotated under each fused tile is ``fused.max()`` — the most
    surprising pixel for that sample.

    Error tiles, the fused tile, and the overlay all share the fixed
    ``[0, 1]`` colour scale (no data-dependent normalization) and a single
    colour bar on the right, so figures stay comparable to one another.
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
    for r in range(n_rows):
        row = [_sq_error(images_np[r], recons_np[k][r])
               for k in range(n_blocks)]
        err_maps.append(row)
        fused = np.maximum.reduce(row)  # per-pixel max across blocks
        fused_maps.append(fused)
        anomaly_scores.append(float(fused.max()))
    vmax = 1.0

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
        cbar.set_label("(x − recon)²  (mean over RGB)", fontsize=_LABEL_FS)
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
    consensus model. The colour scale is fixed to ``[0, 1]`` (no per-figure
    normalization) so figures are comparable, with a single colour bar; each
    tile is annotated with its mean. ID rows should be dim everywhere while
    OOD rows light up on the anomaly (e.g. eyeglasses).
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    recons_np = [_to_image_grid(r) for r in mean_recons]
    n_rows = images_np.shape[0]
    n_blocks = len(recons_np)
    n_cols = 1 + n_blocks

    err_maps: list[list[np.ndarray]] = []
    for r in range(n_rows):
        row = [_sq_error(images_np[r], recons_np[j][r])
               for j in range(n_blocks)]
        err_maps.append(row)
    vmax = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]
    im = None

    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(images_np[r])
        _bare(ax)
        if r == 0:
            ax.set_title("Original", fontsize=_LABEL_FS)
        if labels[r] is not None:
            _row_label(ax, labels[r])

        for j in range(n_blocks):
            ax_e = axes[r, 1 + j]
            e = err_maps[r][j]
            im = ax_e.imshow(e, cmap="viridis", vmin=0.0, vmax=vmax)
            _bare(ax_e)
            if r == 0:
                ax_e.set_title(f"(x − f̄)² L{j}", fontsize=_LABEL_FS)
            ax_e.text(
                0.5, 0.04, f"{float(e.mean()):.3f}",
                transform=ax_e.transAxes, ha="center", va="bottom",
                fontsize=8.0, color="white", path_effects=text_pe,
            )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("(x − f̄)²  (mean over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    reconstruction. The colour scale is fixed to ``[0, 1]`` (no per-figure
    normalization) so figures are comparable, with a single colour bar; each
    tile is annotated with its mean.
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    blocks = sorted(error_maps.keys())
    n_blocks = len(blocks)
    n_cols = 1 + n_blocks

    np_err = {k: _stat_np(error_maps[k]) for k in blocks}
    vmax = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]
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
            ax_e = axes[r, 1 + bi]
            e = np_err[k][r]
            im = ax_e.imshow(e, cmap="viridis", vmin=0.0, vmax=vmax)
            _bare(ax_e)
            if r == 0:
                ax_e.set_title(f"Err L{k}", fontsize=_LABEL_FS)
            ax_e.text(
                0.5, 0.04, f"{float(e.mean()):.3f}",
                transform=ax_e.transAxes, ha="center", va="bottom",
                fontsize=8.0, color="white", path_effects=text_pe,
            )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("mean_m (x − f̂^m)²  (mean over RGB)",
                       fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    is fixed to ``[0, 1]`` (no per-figure normalization) so figures are
    comparable, with a single colour bar; each is annotated with its mean.
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    blocks = sorted(error_maps.keys())
    n_blocks = len(blocks)
    n_cols = 1 + n_blocks

    np_err = {k: _stat_np(error_maps[k]) for k in blocks}
    vmax = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]
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
            ax_e = axes[r, 1 + bi]
            e = np_err[k][r]
            im = ax_e.imshow(e, cmap="viridis", vmin=0.0, vmax=vmax)
            _bare(ax_e)
            if r == 0:
                ax_e.set_title(f"Err L{k}", fontsize=_LABEL_FS)
            ax_e.text(
                0.5, 0.04, f"{float(e.mean()):.3f}",
                transform=ax_e.transAxes, ha="center", va="bottom",
                fontsize=8.0, color="white", path_effects=text_pe,
            )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("min_m (x − f̂^m)²  (mean over RGB)",
                       fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    the maps brighten. The colour scale is fixed to ``[0, 1]`` (no
    per-figure normalization) so figures are comparable, with a single
    colour bar; each is annotated with its mean. Variance is bounded above
    by the risk so it shares the ``[0, 1]`` image-error scale, though in
    practice it stays small.
    """
    import matplotlib.patheffects as pe

    images_np = _to_image_grid(images)
    n_rows = images_np.shape[0]
    blocks = sorted(maps.keys())
    n_blocks = len(blocks)
    n_cols = 1 + n_blocks

    np_var = {k: _stat_np(maps[k]["variance"]) for k in blocks}
    vmax = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.8, 1.55 * n_rows + 0.3),
        layout="constrained",
        squeeze=False,
    )
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]
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
            ax_v = axes[r, 1 + bi]
            v = np_var[k][r]
            im = ax_v.imshow(v, cmap="magma", vmin=0.0, vmax=vmax)
            _bare(ax_v)
            if r == 0:
                ax_v.set_title(f"Var L{k}", fontsize=_LABEL_FS)
            ax_v.text(
                0.5, 0.04, f"{float(v.mean()):.4f}",
                transform=ax_v.transAxes, ha="center", va="bottom",
                fontsize=8.0, color="white", path_effects=text_pe,
            )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("ensemble variance  (mean over RGB)",
                       fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        ax.plot(x, in_m, "-o", color="#377eb8", label="in-dist")
        ax.fill_between(x, in_m - in_s, in_m + in_s, color="#377eb8",
                        alpha=0.18)
        ax.plot(x, ood_m, "--s", color="#e41a1c", label="OOD")
        ax.fill_between(x, ood_m - ood_s, ood_m + ood_s, color="#e41a1c",
                        alpha=0.18)
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
        ax.plot(qs, in_q, "-", color="#377eb8", label="in-dist")
        ax.plot(qs, ood_q, "--", color="#e41a1c", label="OOD")
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    and Variance tiles visibly sum to the Risk tile. Each tile is
    annotated with its mean value.
    """
    import matplotlib.patheffects as pe

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
    # Fixed [0, 1] colour scale (images live in [0, 1] so the squared-error
    # terms do too): no data-dependent normalization, so any two figures are
    # directly comparable.
    vmax = 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.55 * n_rows + 0.3),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    labels = list(row_labels) if row_labels is not None else [None] * n_rows
    text_pe = [pe.withStroke(linewidth=1.6, foreground="black")]
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
                ax_m.text(
                    0.5, 0.04, f"{float(mp.mean()):.4f}",
                    transform=ax_m.transAxes, ha="center", va="bottom",
                    fontsize=8.0, color="white", path_effects=text_pe,
                )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes[:, 1:].ravel().tolist(),
            location="right", shrink=0.85, pad=0.015, aspect=30,
        )
        cbar.set_label("squared error  (mean over RGB)", fontsize=_LABEL_FS)
        cbar.ax.tick_params(labelsize=_TICK_FS)

    if title:
        fig.suptitle(title, fontsize=_TITLE_FS)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    colors = {"risk": "#9bbcd6", "bias": "#1f4e79", "variance": "#e08214"}
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
        ax.hist(in_s, bins=bins, alpha=0.55, density=True,
                color="#377eb8", label=f"in-dist (n={in_s.size})")
        ax.hist(ood_s, bins=bins, alpha=0.55, density=True,
                color="#e41a1c", label=f"OOD (n={ood_s.size})")
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
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
