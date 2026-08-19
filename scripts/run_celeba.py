#!/usr/bin/env python3
"""End-to-end runner for the CelebA OOD experiment.

Pipeline:
    1. load YAML config, seed, set up output tree and logging
    2. build the four CelebA dataloaders (train / val / test_in / test_ood)
    3. build the from-scratch FacialCNN
    4. train on combined CrossEntropy(gender) + BCE(attrs) loss
    5. evaluate per-attribute accuracy on test_in
    6. compute AUROC for OOD detection (test_in vs test_ood = Eyeglasses)
    7. save weights, history, summary, and a small set of plots

Usage:
    python scripts/run_celeba.py --config configs/celeba_ood.yaml
    python scripts/run_celeba.py --config configs/celeba_ood.yaml --eval-only
    python scripts/run_celeba.py --config configs/celeba_ood.yaml \\
        --override training.epochs=50 dataset.batch_size=128
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.config import apply_overrides, load_config, to_jsonable
from lrad.dataset import gather_samples, get_celeba_loaders
from lrad.decoder import build_decoders
from lrad.evaluate import evaluate
from lrad.model import build_model, count_parameters
from lrad.plots import (
    plot_activations,
    plot_batch_accuracy,
    plot_batch_loss,
    plot_decoder_history,
    plot_fusion_overlay,
    plot_per_block_breakdown,
    plot_recons_only,
)
from lrad.train import train_decoders, train_model
from lrad.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger("celeba_ood")


# ---------------------------------------------------------------------------
# Script-local figures (epoch curves, score histograms, ROC)
# ---------------------------------------------------------------------------

def _plot_history(history: dict, save_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    has_val = bool(history.get("val_loss"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], label="train")
    if has_val:
        axes[0].plot(epochs, history["val_loss"], label="val", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined CE + BCE loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_gender_acc"], label="train gender")
    if has_val:
        axes[1].plot(epochs, history["val_gender_acc"], label="val gender",
                     linestyle="--")
        val_attr = np.asarray(history["val_attr_acc"])  # (E, n_attrs)
        if val_attr.size:
            axes[1].plot(epochs, val_attr.mean(axis=1),
                         label="val attrs (mean)", linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.savefig(save_path)
    plt.close(fig)


def _plot_score_distribution(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
    score_name: str,
    save_path: Path,
) -> None:
    from lrad.plots import _C_ID, _C_OOD

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    bins = 60
    ax.hist(in_scores, bins=bins, alpha=0.6, label="in-distribution",
            color=_C_ID, density=True)
    ax.hist(ood_scores, bins=bins, alpha=0.6, label="OOD",
            color=_C_OOD, density=True)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_roc(auroc: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    for k in ("score_entropy_gender", "score_entropy_attrs"):
        v = auroc.get(k, {})
        if "fpr" not in v or "tpr" not in v:
            continue
        ax.plot(v["fpr"], v["tpr"],
                label=f"{k} (AUROC={v['auroc']:.3f})", linewidth=1.0)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(save_path)
    plt.close(fig)


def _write_model_plots(
    cfg: dict,
    model,
    decoders,
    loaders: dict,
    results: dict,
    history: dict | None,
    dec_history: dict | None,
    device: torch.device,
    plot_dir: Path,
) -> None:
    """All figures of a single-model run: training curves, score
    distributions, ROC, and — when decoders exist — the per-block
    reconstruction figures on a small ID + OOD sample."""
    if history is not None:
        _plot_history(history, plot_dir / "training_history.png")
        plot_batch_accuracy(history, plot_dir / "batch_accuracy.png")
        plot_batch_loss(history, plot_dir / "batch_loss.png")
    if dec_history is not None:
        plot_decoder_history(dec_history, plot_dir / "decoder_history.png")

    for key, label in [
        ("score_entropy_gender", "Predictive entropy (gender head)"),
        ("score_entropy_attrs", "Predictive entropy (attr heads)"),
    ]:
        _plot_score_distribution(
            results["preds_in"][key],
            results["preds_ood"][key],
            label,
            plot_dir / f"score_dist_{key}.png",
        )
    _plot_roc(results["auroc"], plot_dir / "roc_ood.png")

    if decoders is None:
        return

    ecfg = cfg.get("evaluation", {})
    n_viz_in = int(ecfg.get("n_viz_in_samples", ecfg.get("n_viz_samples", 4)))
    n_viz_ood = int(ecfg.get("n_viz_ood_samples",
                             ecfg.get("n_viz_samples", 4)))
    viz_seed = ecfg.get("viz_seed")
    samples_in = gather_samples(
        loaders["test_in"], n_viz_in, seed=viz_seed,
    ).to(device)
    samples_ood = gather_samples(
        loaders["test_ood"], n_viz_ood,
        seed=(viz_seed + 1) if viz_seed is not None else None,
    ).to(device)

    decoders.eval()
    with torch.no_grad():
        _, acts_in = model.forward_features(samples_in)
        _, acts_ood = model.forward_features(samples_ood)
        recons_in = [d(acts_in[k]) for k, d in enumerate(decoders)]
        recons_ood = [d(acts_ood[k]) for k, d in enumerate(decoders)]

    all_images = torch.cat([samples_in, samples_ood], dim=0)
    all_recons = [
        torch.cat([ri, ro], dim=0)
        for ri, ro in zip(recons_in, recons_ood)
    ]
    row_labels = (
        [f"ID  {i+1}" for i in range(samples_in.size(0))]
        + [f"OOD {i+1}" for i in range(samples_ood.size(0))]
    )

    plot_per_block_breakdown(
        all_images, all_recons,
        plot_dir / "per_block_breakdown.png",
        row_labels=row_labels,
        title="Original | Err Lk | Recon Lk  per conv block",
    )
    plot_recons_only(
        all_images, all_recons,
        plot_dir / "recons_only.png",
        row_labels=row_labels,
        title="Per-block reconstructions",
    )

    all_acts = [
        torch.cat([ai, ao], dim=0) for ai, ao in zip(acts_in, acts_ood)
    ]
    plot_activations(
        all_images, all_acts,
        plot_dir / "activations.png",
        row_labels=row_labels,
        title="Per-block classifier activations (channel-mean)",
    )

    plot_fusion_overlay(
        all_images, all_recons,
        plot_dir / "fusion_overlay.png",
        row_labels=row_labels,
        title="Multi-scale fusion (per-pixel max) + overlay",
    )
    logger.info(
        "Wrote per-block plots: per_block_breakdown.png, "
        "recons_only.png, activations.png, fusion_overlay.png"
    )


# ---------------------------------------------------------------------------
# Single-model pipeline (build -> train -> evaluate -> plot)
# ---------------------------------------------------------------------------

def build_and_run(
    cfg: dict,
    loaders: dict,
    device: torch.device,
    output_dir: Path,
    *,
    seed: int = 42,
    eval_only: bool = False,
    no_plots: bool = False,
) -> tuple:
    """Build a FacialCNN, train classifier + decoders, evaluate, and write
    plots/summary into ``output_dir``.

    The RNGs are seeded with ``seed`` right before model construction, so
    the weight init and SGD shuffle order are reproducible — and, when
    this is called repeatedly with different seeds, the runs form an
    independent deep ensemble (the basis of the bias/variance
    decomposition in ``scripts/run_ensemble.py``).

    Returns ``(model, decoders, results)``. ``decoders`` is ``None`` when
    no decoder section is configured, or when ``--eval-only`` finds no
    decoder weights.
    """
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed_everything(seed)
    exp = cfg["experiment"]

    # --- Model ---
    model = build_model(cfg).to(device)
    # Captured now: train_decoders later freezes the classifier
    # (requires_grad=False), which would make this count read 0.
    n_params = count_parameters(model)
    logger.info(f"Model        : {model.__class__.__name__}  "
                f"({n_params:,} params)")

    history: dict | None = None
    weights_path = output_dir / "weights" / "model.pt"

    # --- Train ---
    if eval_only:
        if not weights_path.exists():
            raise FileNotFoundError(f"missing weights at {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True),
        )
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.info("=" * 70)
        logger.info("TRAINING — combined CE(gender) + BCE(attrs) loss")
        logger.info("=" * 70)
        tcfg = cfg["training"]
        save_every_epoch = bool(tcfg.get("save_every_epoch", False))
        t0 = time.time()
        history = train_model(
            model,
            loaders["train"],
            loaders.get("val"),
            epochs=tcfg.get("epochs", 30),
            lr=tcfg.get("lr", 1e-3),
            attr_loss_weight=tcfg.get("attr_loss_weight", 1.0),
            device=device,
            log_every=tcfg.get("log_every", 1),
            early_stop_patience=tcfg.get("early_stop_patience", 6),
            early_stop_min_delta=tcfg.get("early_stop_min_delta", 1e-4),
            early_stop_min_epochs=tcfg.get("early_stop_min_epochs", 5),
            checkpoint_dir=(output_dir / "weights") if save_every_epoch
                           else None,
            cutpaste=tcfg.get("cutpaste"),
        )
        logger.info(f"Training done in {time.time() - t0:.1f}s")
        torch.save(model.state_dict(), weights_path)
        with open(output_dir / "history.json", "w") as f:
            json.dump(to_jsonable(history), f, indent=2)

    # --- Per-block decoders ---
    image_size = cfg.get("dataset", {}).get("image_size", 64)
    dec_cfg = cfg.get("training", {}).get("decoders")
    decoders = build_decoders(model, image_size=image_size).to(device)
    dec_history: dict | None = None
    decoders_path = output_dir / "weights" / "decoders.pt"

    if eval_only:
        if decoders_path.exists():
            decoders.load_state_dict(
                torch.load(decoders_path, map_location=device,
                           weights_only=True),
            )
            logger.info(f"Loaded decoder weights from {decoders_path}")
        else:
            logger.info(
                "No decoder weights found; skipping per-block plots."
            )
            decoders = None  # type: ignore[assignment]
    elif dec_cfg:
        logger.info("=" * 70)
        logger.info("PHASE 2 — training per-block reconstruction decoders")
        logger.info("=" * 70)
        for k, d in enumerate(decoders):
            logger.info(
                f"  decoder L{k}: in=(C={model.block_out_channels[k]}, "
                f"H={model.block_spatial_sizes[k]}) → "
                f"out=(3, {image_size}, {image_size})  "
                f"params={sum(p.numel() for p in d.parameters()):,}"
            )
        save_every_epoch = bool(
            cfg.get("training", {}).get("save_every_epoch", False),
        )
        t0 = time.time()
        dec_history = train_decoders(
            model,
            decoders,
            loaders["train"],
            loaders.get("val"),
            epochs=dec_cfg.get("epochs", 20),
            lr=dec_cfg.get("lr", 1e-3),
            device=device,
            log_every=dec_cfg.get("log_every", 1),
            checkpoint_dir=(output_dir / "weights") if save_every_epoch
                           else None,
        )
        logger.info(f"Decoder training done in {time.time() - t0:.1f}s")
        torch.save(decoders.state_dict(), decoders_path)
        with open(output_dir / "decoders_history.json", "w") as f:
            json.dump(to_jsonable(dec_history), f, indent=2)
    else:
        decoders = None  # type: ignore[assignment]

    # --- Evaluate ---
    # Classifier-head scores only: the headline anomaly score (the bias
    # term) needs the spread across independently trained models and
    # therefore lives in run_ensemble.py.
    logger.info("=" * 70)
    logger.info("EVALUATION — per-attribute accuracy + OOD AUROC")
    logger.info("=" * 70)
    results = evaluate(model, loaders, device)

    if not no_plots:
        _write_model_plots(
            cfg, model, decoders, loaders, results, history, dec_history,
            device, output_dir / "plots",
        )

    # --- Summary ---
    summary = {
        "experiment": exp["name"],
        "seed": seed,
        "device": str(device),
        "n_train": len(loaders["train"].dataset),
        "n_val": (len(loaders["val"].dataset)
                  if loaders.get("val") is not None else 0),
        "n_test_in": len(loaders["test_in"].dataset),
        "n_test_ood": len(loaders["test_ood"].dataset),
        "gender_attr": loaders["gender_attr"],
        "attr_targets": loaders["attr_targets"],
        "ood_attr": loaders["ood_attr"],
        "model": {
            "channels": list(getattr(model, "channels", [])),
            "kernel_size": int(getattr(model, "kernel_size", 3)),
            "parameters": n_params,
        },
        "accuracy": to_jsonable(results["accuracy"]),
        "auroc": to_jsonable({
            k: {kk: vv for kk, vv in v.items()
                if kk in ("auroc", "in_mean", "ood_mean")}
            if isinstance(v, dict) else v
            for k, v in results["auroc"].items()
        }),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {output_dir / 'summary.json'}")

    return model, decoders, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CelebA OOD experiment.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Override experiment.output_dir from the config.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Load weights from output_dir and skip training.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dotted overrides, e.g. training.epochs=50.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    exp = cfg["experiment"]
    output_dir = Path(args.output_dir or exp["output_dir"])
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    seed = int(exp.get("seed", 42))
    seed_everything(seed)
    job_id = os.environ.get("OAR_JOB_ID", "")
    run_tag = f"{exp['name']}{('_oar' + job_id) if job_id else ''}"
    setup_logging(str(output_dir / "logs"), run_tag=run_tag)
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"Experiment  : {exp['name']}")
    logger.info(f"Config      : {args.config}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    if device.type == "cuda":
        try:
            logger.info(
                f"GPU         : {torch.cuda.get_device_name(0)}  "
                f"(torch {torch.__version__}, cuda {torch.version.cuda})"
            )
        except Exception:
            pass
    logger.info("=" * 70)

    # --- Data ---
    loaders = get_celeba_loaders(cfg)
    n_val = (len(loaders['val'].dataset)
             if loaders.get('val') is not None else 0)
    logger.info(
        f"Dataloaders  train={len(loaders['train'].dataset)}  "
        f"val={n_val}  "
        f"test_in={len(loaders['test_in'].dataset)}  "
        f"test_ood={len(loaders['test_ood'].dataset)}"
    )
    logger.info(f"Targets      : gender={loaders['gender_attr']}, "
                f"attrs={loaders['attr_targets']}, "
                f"ood={loaders['ood_attr']}")

    build_and_run(
        cfg, loaders, device, output_dir,
        seed=seed, eval_only=args.eval_only, no_plots=args.no_plots,
    )
    logger.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if logger.handlers:
            logger.exception("run_celeba crashed")
        else:
            traceback.print_exc()
        sys.exit(1)
