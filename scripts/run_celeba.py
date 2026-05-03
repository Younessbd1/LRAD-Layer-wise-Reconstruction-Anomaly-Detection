#!/usr/bin/env python3
"""End-to-end CelebA experiment runner for LRAD.

Pipeline (driven from a single YAML config):

  0. load config, seed everything, create output tree
  1. build CelebA dataloaders (train / val / test_normal / test_anomaly)
  2. build CNNClassifier + LRADModel (one decoder per block)
  3. Phase 1 — train classifier on a multi-label attribute pretext
     (sigmoid + BCE over ``dataset.pretext_attrs``), with val tracking
     and early stopping
  4. Phase 2 — train each decoder with MSE reconstruction loss
  5. Image-level evaluation on test_normal + test_anomaly:
        AUROC, PR-AUC, F1, precision, recall, best-F1 threshold
     (CelebA has no pixel masks, so pixel-level metrics are skipped.)
  6. Emit minimal plots: training curves, score histogram, ROC, PR.
  7. Persist weights + a summary.json with every metric + config snapshot.

Usage::

    python scripts/run_celeba.py --config configs/celeba_eyeglasses.yaml
    python scripts/run_celeba.py --config configs/celeba_eyeglasses.yaml --eval-only
    python scripts/run_celeba.py --config configs/celeba_eyeglasses.yaml \\
        --override training.decoders.epochs=80 dataset.batch_size=128
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Force unbuffered stdio so OAR's `tail -f oar.<jobid>.stdout` shows progress
# in real time, even when the process is launched without `python -u`.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe backend

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.data.celeba import get_celeba_loaders
from lrad.engine.celeba_engine import (
    evaluate_celeba,
    train_classifier_attributes,
    train_decoders,
)
from lrad.models.classifier import CNNClassifier
from lrad.models.lrad_model import LRADModel
from lrad.utils.helpers import count_parameters, get_device, seed_everything, setup_logging
from lrad.visualization import (
    apply_style,
    plot_score_distributions,
    plot_roc_curves,
    plot_per_layer_errors,
    plot_per_layer_reconstructions,
    plot_heatmap_grid,
    plot_reconstruction_comparison,
    PALETTE,
)

apply_style()

logger = logging.getLogger("lrad")


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_override(cfg: dict, key_path: str, value: str) -> None:
    """Apply a dotted-key override like ``training.decoders.epochs=80``."""
    node = cfg
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    for cast in (int, float):
        try:
            node[keys[-1]] = cast(value)
            return
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        node[keys[-1]] = value.lower() == "true"
        return
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
    return cfg


# ---------------------------------------------------------------------------
# Plotting — uses the shared `lrad.visualization` style (white bg, palette,
# typography, no text/image overlap). Inline plots only cover training-curve
# panels that aren't part of the reusable visualization package.
# ---------------------------------------------------------------------------

def _styled_history_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4,
            color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, facecolor="white",
              edgecolor=PALETTE["grid"])


def _plot_classifier_history(history: dict, save_path: Path) -> None:
    """Phase 1 curves: BCE loss, mean accuracy, and per-attribute val accuracy."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    pretext_attrs = history.get("pretext_attrs", [])
    val_per_attr = np.asarray(history.get("val_acc_per_attr") or [])  # (E, P)
    show_per_attr = val_per_attr.size > 0 and len(pretext_attrs) > 0

    n_panels = 3 if show_per_attr else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.0 * n_panels, 3.4), constrained_layout=True,
    )

    axes[0].plot(epochs, history["train_loss"],
                 color=PALETTE["accent_2"], linewidth=1.0, label="train", zorder=3)
    axes[0].plot(epochs, history["val_loss"],
                 color=PALETTE["anomaly"], linewidth=1.0,
                 linestyle="--", label="val", zorder=3)
    _styled_history_axes(axes[0], "BCE loss")

    axes[1].plot(epochs, history["train_acc"],
                 color=PALETTE["accent_2"], linewidth=1.0, label="train", zorder=3)
    axes[1].plot(epochs, history["val_acc"],
                 color=PALETTE["anomaly"], linewidth=1.0,
                 linestyle="--", label="val", zorder=3)
    axes[1].set_ylim(0.45, 1.02)
    _styled_history_axes(axes[1], "Mean attribute accuracy")

    if show_per_attr:
        cycle = [PALETTE["normal"], PALETTE["accent"], PALETTE["neutral"],
                 PALETTE["accent_2"], PALETTE["anomaly_2"], PALETTE["bar_secondary"]]
        for j, name in enumerate(pretext_attrs):
            color = cycle[j % len(cycle)]
            axes[2].plot(epochs, val_per_attr[:, j], color=color,
                         linewidth=1.0, label=name, zorder=3)
        axes[2].set_ylim(0.45, 1.02)
        _styled_history_axes(axes[2], "Per-attribute val accuracy")

    fig.savefig(save_path)
    plt.close(fig)


def _plot_decoder_history(history: dict, save_path: Path) -> None:
    n = len(history["train_loss"])
    epochs = range(1, len(history["train_loss"][0]) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)

    cycle = [PALETTE["normal"], PALETTE["accent"], PALETTE["neutral"],
             PALETTE["accent_2"], PALETTE["anomaly_2"], PALETTE["bar_secondary"]]
    for i in range(n):
        c = cycle[i % len(cycle)]
        ax.plot(epochs, history["train_loss"][i], color=c, linewidth=1.0,
                label=f"D{i} train", zorder=3)
        ax.plot(epochs, history["val_loss"][i], color=c, linewidth=0.9,
                linestyle="--", alpha=0.85, label=f"D{i} val", zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.4,
            color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(n, 3), fontsize=7.5, frameon=True,
              facecolor="white", edgecolor=PALETTE["grid"],
              loc="upper right")
    fig.savefig(save_path)
    plt.close(fig)


def _plot_pr_curve(metrics: dict, save_path: Path) -> None:
    """Standalone PR curve, styled like the rest of the report."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)

    recalls = np.asarray(metrics["recalls"])
    precisions = np.asarray(metrics["precisions"])
    ax.plot(recalls, precisions, color=PALETTE["accent"], linewidth=1.1,
            zorder=3, label=f"PR curve  (PR-AUC = {metrics['pr_auc']:.3f})")
    ax.fill_between(recalls, precisions, 0,
                    color=PALETTE["accent"], alpha=0.08, zorder=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3, linewidth=0.4, color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=PALETTE["grid"])
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run LRAD on CelebA.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Override experiment.output_dir from the config.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Load weights from the output dir and skip training.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dotted-key overrides, e.g. training.decoders.epochs=80.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip visualisation steps (faster smoke tests).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    exp = cfg["experiment"]
    output_dir = args.output_dir or Path(exp["output_dir"])
    output_dir = Path(output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)

    seed_everything(exp.get("seed", 42))
    job_id = os.environ.get("OAR_JOB_ID", "")
    run_tag = f"{exp['name']}{('_oar' + job_id) if job_id else ''}"
    setup_logging(str(output_dir / "logs"), run_tag=run_tag)
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"Experiment  : {exp['name']}")
    logger.info(f"Config      : {args.config}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Seed        : {exp.get('seed', 42)}")
    if job_id:
        logger.info(f"OAR job     : {job_id}  host={os.environ.get('HOSTNAME', 'unknown')}")
    if device.type == "cuda":
        try:
            logger.info(
                f"GPU         : {torch.cuda.get_device_name(0)}  "
                f"(torch {torch.__version__}, cuda {torch.version.cuda})"
            )
        except Exception:
            pass
    logger.info("=" * 70)

    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Data ---
    loaders = get_celeba_loaders(cfg)
    logger.info(
        f"Dataloaders  train={len(loaders['train'].dataset)}  "
        f"val={len(loaders['val'].dataset)}  "
        f"test_normal={len(loaders['test_normal'].dataset)}  "
        f"test_anomaly={len(loaders['test_anomaly'].dataset)}"
    )
    logger.info(f"Anomaly attr : {loaders['attr_name']}  has_masks={loaders['has_masks']}")

    # --- Model ---
    # NOTE: build the LRADModel *after* Phase 1. ``LRADModel.__init__``
    # freezes the classifier (sets ``requires_grad=False`` on every param),
    # which silently kills Phase 1's gradient flow if done up front.
    mcfg = cfg["model"]
    pretext_attrs = loaders["pretext_attrs"]
    classifier = CNNClassifier(
        channels=mcfg["channels"],
        num_classes=len(pretext_attrs),     # multi-label attribute pretext
        input_size=mcfg.get("input_size", 64),
    )
    logger.info(f"Pretext attrs: {pretext_attrs}  ({len(pretext_attrs)}-way multi-label)")
    logger.info(f"Classifier   : {classifier}")
    logger.info(f"  parameters : {count_parameters(classifier):,}")

    cls_history: dict | None = None
    dec_history: dict | None = None

    # --- Phase 1 ---
    if args.eval_only:
        cls_path = output_dir / "weights" / "classifier.pt"
        if not cls_path.exists():
            raise FileNotFoundError(f"missing classifier weights at {cls_path}")
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        classifier.freeze()
        logger.info(f"Loaded classifier weights from {cls_path}")
    else:
        logger.info("=" * 70)
        logger.info("PHASE 1 — multi-label attribute pretext training")
        logger.info("=" * 70)
        tcls = cfg["training"]["classifier"]
        t0 = time.time()
        cls_history = train_classifier_attributes(
            classifier,
            loaders["train"],
            loaders["val"],
            pretext_attrs=pretext_attrs,
            epochs=tcls.get("epochs", 30),
            lr=tcls.get("lr", 1e-3),
            weight_decay=tcls.get("weight_decay", 1e-4),
            device=device,
            log_every=tcls.get("log_every", 5),
            early_stop_patience=tcls.get("early_stop_patience", 8),
            early_stop_min_delta=tcls.get("early_stop_min_delta", 1e-4),
            early_stop_min_epochs=tcls.get("early_stop_min_epochs", 5),
        )
        logger.info(f"Phase 1 done in {time.time() - t0:.1f}s")
        torch.save(classifier.state_dict(), output_dir / "weights" / "classifier.pt")
        with open(output_dir / "cls_history.json", "w") as f:
            json.dump(cls_history, f, indent=2)

    # Now that the classifier is trained (or loaded) and frozen, build the
    # LRADModel. Decoders are sized from classifier.spatial_sizes.
    model = LRADModel(classifier)
    for i, d in enumerate(model.decoders):
        logger.info(f"  Decoder {i} : {count_parameters(d):,} params")
    logger.info(f"  total dec  : {sum(count_parameters(d) for d in model.decoders):,}")

    # --- Phase 2 ---
    baselines_path = output_dir / "weights" / "lrad_buffers.pt"
    if args.eval_only:
        for i, dec in enumerate(model.decoders):
            p = output_dir / "weights" / f"decoder_{i}.pt"
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")
            dec.load_state_dict(torch.load(p, map_location=device))
            logger.info(f"Loaded decoder {i} weights from {p}")
        if baselines_path.exists():
            buffers = torch.load(baselines_path, map_location=device)
            for name, tensor in buffers.items():
                model.register_buffer(name, tensor.to(device))
            logger.info(f"Loaded calibration buffers from {baselines_path}")
    else:
        logger.info("=" * 70)
        logger.info("PHASE 2 — decoder training")
        logger.info("=" * 70)
        tdec = cfg["training"]["decoders"]
        t0 = time.time()
        dec_history = train_decoders(
            model,
            loaders["train"],
            loaders["val"],
            epochs=tdec.get("epochs", 30),
            lr=tdec.get("lr", 1e-3),
            weight_decay=tdec.get("weight_decay", 1e-5),
            device=device,
            log_every=tdec.get("log_every", 5),
        )
        logger.info(f"Phase 2 done in {time.time() - t0:.1f}s")
        for i, dec in enumerate(model.decoders):
            torch.save(dec.state_dict(),
                       output_dir / "weights" / f"decoder_{i}.pt")
        with open(output_dir / "dec_history.json", "w") as f:
            json.dump(dec_history, f, indent=2)

    # --- Calibration ---
    # PaDiM-style per-pixel "background" subtraction: fit on normal val data,
    # subtract at test time so consistently hard-to-reconstruct regions
    # (face/hair edges, lighting hotspots) stop dominating the heatmap.
    ecfg = cfg.get("evaluation", {})
    if not args.eval_only and ecfg.get("calibrate", True):
        logger.info("=" * 70)
        logger.info("CALIBRATION — fitting per-pixel baseline on val (normals)")
        logger.info("=" * 70)
        t0 = time.time()
        model.calibrate(loaders["val"], device=device)
        logger.info(f"Calibration done in {time.time() - t0:.1f}s")
        buffers = {f"baseline_{i}": getattr(model, f"baseline_{i}").detach().cpu()
                   for i in range(len(model.decoders))}
        buffers["calibrated"] = model.calibrated.detach().cpu()
        torch.save(buffers, baselines_path)
        logger.info(f"Saved calibration buffers to {baselines_path}")

    # --- Evaluation ---
    logger.info("=" * 70)
    logger.info("EVALUATION (image-level only — CelebA has no pixel masks)")
    logger.info("=" * 70)
    results = evaluate_celeba(
        model,
        loaders,
        device=device,
        fusion=ecfg.get("fusion", "mean"),
        max_batches=ecfg.get("max_eval_batches"),
        smooth_sigma=ecfg.get("smooth_sigma", 1.5),
        normalize_per_layer=ecfg.get("normalize_per_layer", True),
        score_top_k_pct=ecfg.get("score_top_k_pct", 1.0),
    )

    # --- Plots ---
    if not args.no_plots:
        logger.info("Saving plots...")
        plot_dir = output_dir / "plots"
        if cls_history is not None:
            _plot_classifier_history(cls_history, plot_dir / "phase1_history.png")
        if dec_history is not None:
            _plot_decoder_history(dec_history, plot_dir / "phase2_history.png")

        # 1. Anomaly score distributions — histogram + KDE + AUROC + threshold
        fig = plot_score_distributions(
            normal_scores=results["normal"]["scores"],
            anomaly_scores_dict={loaders["attr_name"]: results["anomaly"]["scores"]},
            save_path=str(plot_dir / "score_distribution.png"),
        )
        plt.close(fig)

        # 2. ROC curves with Youden-J operating point
        fig = plot_roc_curves(
            auroc_results={loaders["attr_name"]: results["image"]},
            save_path=str(plot_dir / "roc.png"),
        )
        plt.close(fig)

        # 2b. Standalone PR curve (kept separate for readability)
        _plot_pr_curve(results["image"], plot_dir / "pr.png")

        # 3. Per-decoder block plots — only available if the engine retained
        # the visualization subset. Show a few normal and a few anomaly samples.
        viz_n = results["normal"].get("viz")
        viz_a = results["anomaly"].get("viz")

        if viz_n is not None and viz_a is not None:
            n_show = min(6, len(viz_n["images"]), len(viz_a["images"]))

            # 3a. Original | reconstruction Lk | error heatmap Lk for each
            # decoder block, side by side per sample (anomaly samples)
            fig = plot_per_layer_errors(
                images=viz_a["images"][:n_show],
                per_layer_maps=[m[:n_show] for m in viz_a["per_layer"]],
                reconstructions=[r[:n_show] for r in viz_a["reconstructions"]],
                n_samples=n_show,
                save_path=str(plot_dir / "per_decoder_anomaly.png"),
            )
            plt.close(fig)

            fig = plot_per_layer_errors(
                images=viz_n["images"][:n_show],
                per_layer_maps=[m[:n_show] for m in viz_n["per_layer"]],
                reconstructions=[r[:n_show] for r in viz_n["reconstructions"]],
                n_samples=n_show,
                save_path=str(plot_dir / "per_decoder_normal.png"),
            )
            plt.close(fig)

            # 3b. Heatmap-only stack: error heatmaps from every decoder
            # stacked one below the other, for direct comparison.
            fig = plot_per_layer_errors(
                images=viz_a["images"][:n_show],
                per_layer_maps=[m[:n_show] for m in viz_a["per_layer"]],
                reconstructions=None,
                n_samples=n_show,
                save_path=str(plot_dir / "heatmaps_stacked_anomaly.png"),
            )
            plt.close(fig)

            fig = plot_per_layer_errors(
                images=viz_n["images"][:n_show],
                per_layer_maps=[m[:n_show] for m in viz_n["per_layer"]],
                reconstructions=None,
                n_samples=n_show,
                save_path=str(plot_dir / "heatmaps_stacked_normal.png"),
            )
            plt.close(fig)

            # 3c. Full breakdown grid: original + per-decoder recon/error
            # + fused map + overlay (anomaly samples).
            fig = plot_heatmap_grid(
                images=viz_a["images"][:n_show],
                heatmaps=viz_a["fused"][:n_show],
                per_layer_maps=[m[:n_show] for m in viz_a["per_layer"]],
                reconstructions=[r[:n_show] for r in viz_a["reconstructions"]],
                scores=viz_a["scores"][:n_show],
                n_samples=n_show,
                save_path=str(plot_dir / "heatmap_grid_anomaly.png"),
            )
            plt.close(fig)

            # 3d. Side-by-side normal vs. anomaly per-decoder reconstructions
            n_pair = min(n_show, len(viz_n["images"]), len(viz_a["images"]))
            fig = plot_reconstruction_comparison(
                images=viz_n["images"][:n_pair],
                normal_recons=[r[:n_pair] for r in viz_n["reconstructions"]],
                anomaly_recons=[r[:n_pair] for r in viz_a["reconstructions"]],
                n_samples=n_pair,
                save_path=str(plot_dir / "recon_normal_vs_anomaly.png"),
            )
            plt.close(fig)
        else:
            logger.info("Skipping per-decoder plots (no viz subset captured).")
    else:
        logger.info("--no-plots set, skipping plots")

    # --- Summary ---
    summary = {
        "experiment": exp["name"],
        "seed": exp.get("seed", 42),
        "device": str(device),
        "anomaly_attr": loaders["attr_name"],
        "pretext_attrs": loaders["pretext_attrs"],
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test_normal": len(loaders["test_normal"].dataset),
        "n_test_anomaly": len(loaders["test_anomaly"].dataset),
        "classifier": {
            "architecture": "CNNClassifier",
            "channels": classifier.channels,
            "input_size": mcfg.get("input_size", 64),
            "spatial_sizes": classifier.spatial_sizes,
            "parameters": count_parameters(classifier),
        },
        "decoders": {
            "count": len(model.decoders),
            "parameters_total": sum(count_parameters(d) for d in model.decoders),
        },
        "metrics": {
            "image": _to_jsonable({k: v for k, v in results["image"].items()
                                   if not isinstance(v, np.ndarray)}),
            "pixel": None,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {output_dir / 'summary.json'}")
    logger.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Make sure crashes always land in the file logger and the OAR
        # stderr capture, even if setup_logging hasn't run yet.
        if logger.handlers:
            logger.exception("run_celeba crashed")
        else:
            traceback.print_exc()
        sys.exit(1)
