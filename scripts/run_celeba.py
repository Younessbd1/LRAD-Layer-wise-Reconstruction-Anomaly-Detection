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
import copy
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

from lrad.dataset import get_celeba_loaders
from lrad.decoder import build_decoders
from lrad.evaluate import evaluate
from lrad.model import build_model, count_parameters
from lrad.plots import plot_batch_accuracy, plot_per_block_breakdown, plot_recons_only
from lrad.train import train_decoders, train_model
from lrad.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger("celeba_ood")


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_override(cfg: dict, key_path: str, value: str) -> None:
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
# Plots
# ---------------------------------------------------------------------------

def _plot_history(history: dict, save_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined CE + BCE loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_gender_acc"], label="train gender")
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
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    bins = 60
    ax.hist(in_scores, bins=bins, alpha=0.55, label="in-dist (no glasses)",
            color="#377eb8", density=True)
    ax.hist(ood_scores, bins=bins, alpha=0.55, label="OOD (Eyeglasses)",
            color="#e41a1c", density=True)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def _gather_samples(loader, n: int) -> torch.Tensor:
    """Take the first ``n`` images from a (non-shuffled) loader."""
    chunks: list[torch.Tensor] = []
    have = 0
    for batch in loader:
        chunks.append(batch[0])
        have += batch[0].size(0)
        if have >= n:
            break
    return torch.cat(chunks, dim=0)[:n]


def _plot_roc(auroc: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    for k in ("score_msp", "score_entropy_gender",
              "score_entropy_attrs", "score_entropy_combined"):
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


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


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
        f"test_in={len(loaders['test_in'].dataset)}  "
        f"test_ood={len(loaders['test_ood'].dataset)}"
    )
    logger.info(f"Targets      : gender={loaders['gender_attr']}, "
                f"attrs={loaders['attr_targets']}, "
                f"ood={loaders['ood_attr']}")

    # --- Model ---
    model = build_model(cfg).to(device)
    logger.info(f"Model        : {model.__class__.__name__}  "
                f"({count_parameters(model):,} params)")

    history: dict | None = None
    weights_path = output_dir / "weights" / "model.pt"

    # --- Train ---
    if args.eval_only:
        if not weights_path.exists():
            raise FileNotFoundError(f"missing weights at {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.info("=" * 70)
        logger.info("TRAINING — combined CE(gender) + BCE(attrs) loss")
        logger.info("=" * 70)
        tcfg = cfg["training"]
        t0 = time.time()
        history = train_model(
            model,
            loaders["train"],
            loaders["val"],
            epochs=tcfg.get("epochs", 30),
            lr=tcfg.get("lr", 1e-3),
            weight_decay=tcfg.get("weight_decay", 1e-4),
            attr_loss_weight=tcfg.get("attr_loss_weight", 1.0),
            device=device,
            log_every=tcfg.get("log_every", 1),
            early_stop_patience=tcfg.get("early_stop_patience", 6),
            early_stop_min_delta=tcfg.get("early_stop_min_delta", 1e-4),
            early_stop_min_epochs=tcfg.get("early_stop_min_epochs", 5),
        )
        logger.info(f"Training done in {time.time() - t0:.1f}s")
        torch.save(model.state_dict(), weights_path)
        with open(output_dir / "history.json", "w") as f:
            json.dump(_to_jsonable(history), f, indent=2)

    # --- Decoders (visualization only) ---
    image_size = cfg.get("dataset", {}).get("image_size", 64)
    decoders = build_decoders(model, image_size=image_size).to(device)
    dec_history: dict | None = None
    decoders_path = output_dir / "weights" / "decoders.pt"
    dec_cfg = cfg.get("training", {}).get("decoders")

    if args.eval_only:
        if decoders_path.exists():
            decoders.load_state_dict(
                torch.load(decoders_path, map_location=device),
            )
            logger.info(f"Loaded decoder weights from {decoders_path}")
        else:
            logger.info(
                "No decoder weights found; skipping per-block plots."
            )
            decoders = None  # type: ignore[assignment]
    elif dec_cfg:
        logger.info("=" * 70)
        logger.info("PHASE 2 — training per-block decoders (visualization)")
        logger.info("=" * 70)
        for k, d in enumerate(decoders):
            logger.info(
                f"  decoder L{k}: in=(C={model.block_out_channels[k]}, "
                f"H={model.block_spatial_sizes[k]}) → "
                f"out=(3, {image_size}, {image_size})  "
                f"params={sum(p.numel() for p in d.parameters()):,}"
            )
        t0 = time.time()
        dec_history = train_decoders(
            model,
            decoders,
            loaders["train"],
            loaders["val"],
            epochs=dec_cfg.get("epochs", 20),
            lr=dec_cfg.get("lr", 1e-3),
            weight_decay=dec_cfg.get("weight_decay", 1e-5),
            device=device,
            log_every=dec_cfg.get("log_every", 1),
        )
        logger.info(f"Decoder training done in {time.time() - t0:.1f}s")
        torch.save(decoders.state_dict(), decoders_path)
        with open(output_dir / "decoders_history.json", "w") as f:
            json.dump(_to_jsonable(dec_history), f, indent=2)
    else:
        decoders = None  # type: ignore[assignment]

    # --- Evaluate ---
    logger.info("=" * 70)
    logger.info("EVALUATION — per-attribute accuracy + OOD AUROC")
    logger.info("=" * 70)
    results = evaluate(model, loaders, device)

    # --- Plots ---
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        if history is not None:
            _plot_history(history, plot_dir / "training_history.png")
            plot_batch_accuracy(history, plot_dir / "batch_accuracy.png")
        for key, label in [
            ("score_msp", "1 - max softmax prob (gender)"),
            ("score_entropy_combined", "Combined predictive entropy"),
        ]:
            _plot_score_distribution(
                results["preds_in"][key],
                results["preds_ood"][key],
                label,
                plot_dir / f"score_dist_{key}.png",
            )
        _plot_roc(results["auroc"], plot_dir / "roc_ood.png")

        # --- Per-block reconstruction plots (visualization only) ----------
        if decoders is not None:
            n_viz = int(cfg.get("evaluation", {}).get("n_viz_samples", 4))
            samples_in = _gather_samples(loaders["test_in"], n_viz).to(device)
            samples_ood = _gather_samples(loaders["test_ood"], n_viz).to(device)

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
            logger.info(
                "Wrote per-block plots: per_block_breakdown.png, recons_only.png"
            )

    # --- Summary ---
    summary = {
        "experiment": exp["name"],
        "seed": exp.get("seed", 42),
        "device": str(device),
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test_in": len(loaders["test_in"].dataset),
        "n_test_ood": len(loaders["test_ood"].dataset),
        "gender_attr": loaders["gender_attr"],
        "attr_targets": loaders["attr_targets"],
        "ood_attr": loaders["ood_attr"],
        "model": {
            "channels": list(getattr(model, "channels", [])),
            "parameters": count_parameters(model),
        },
        "accuracy": _to_jsonable(results["accuracy"]),
        "auroc": _to_jsonable({
            k: {kk: vv for kk, vv in v.items() if kk in ("auroc", "in_mean", "ood_mean")}
            if isinstance(v, dict) else v
            for k, v in results["auroc"].items()
        }),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {output_dir / 'summary.json'}")
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
