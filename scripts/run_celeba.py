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

from lrad.anomaly_score import _per_pixel_errors
from lrad.dataset import get_celeba_loaders
from lrad.decoder import build_decoders
from lrad.evaluate import evaluate
from lrad.model import build_model, count_parameters
from lrad.plots import (
    plot_activations,
    plot_batch_accuracy,
    plot_fusion_auroc,
    plot_fusion_overlay,
    plot_per_block_breakdown,
    plot_recons_only,
)
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
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    bins = 60
    ax.hist(in_scores, bins=bins, alpha=0.55, label="in-distribution",
            color="#377eb8", density=True)
    ax.hist(ood_scores, bins=bins, alpha=0.55, label="OOD",
            color="#e41a1c", density=True)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def _compute_fusion_scores(model, decoders, loader, device) -> np.ndarray:
    """Per-image fused anomaly score = max of per-pixel max over per-block
    error maps. Returns shape (N,) over the full loader."""
    scores: list[np.ndarray] = []
    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        errs_by_block = _per_pixel_errors(model, decoders, img)  # {k: (B,H,W)}
        errs = torch.stack(
            [errs_by_block[k] for k in sorted(errs_by_block)], dim=0,
        )  # (n_blocks, B, H, W)
        fused = errs.max(dim=0).values          # (B, H, W) — per-pixel max
        s = fused.flatten(1).max(dim=1).values  # (B,) — most surprising pixel
        scores.append(s.cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def _gather_samples(
    loader,
    n: int,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Return ``n`` images from a (non-shuffled) loader.

    With ``seed=None`` the call is backwards compatible: the first ``n``
    images are returned. With a seed, ``n`` images are drawn uniformly at
    random from the entire underlying dataset (reproducible), which gives
    visually-varied picks and lets us show samples beyond the loader's
    first batch.
    """
    if seed is None:
        chunks: list[torch.Tensor] = []
        have = 0
        for batch in loader:
            chunks.append(batch[0])
            have += batch[0].size(0)
            if have >= n:
                break
        return torch.cat(chunks, dim=0)[:n]

    ds = loader.dataset
    total = len(ds)
    n = min(n, total)
    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(total, generator=gen)[:n].tolist()
    imgs = [ds[i][0] for i in idx]
    return torch.stack(imgs, dim=0)


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
        return None  # fpr/tpr/score arrays are too large for the summary JSON
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


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
    logger.info(f"Model        : {model.__class__.__name__}  "
                f"({count_parameters(model):,} params)")

    history: dict | None = None
    weights_path = output_dir / "weights" / "model.pt"

    # --- Train ---
    if eval_only:
        if not weights_path.exists():
            raise FileNotFoundError(f"missing weights at {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
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
            weight_decay=tcfg.get("weight_decay", 0.0),
            attr_loss_weight=tcfg.get("attr_loss_weight", 1.0),
            device=device,
            log_every=tcfg.get("log_every", 1),
            early_stop_patience=tcfg.get("early_stop_patience", 6),
            early_stop_min_delta=tcfg.get("early_stop_min_delta", 1e-4),
            early_stop_min_epochs=tcfg.get("early_stop_min_epochs", 5),
            checkpoint_dir=(output_dir / "weights") if save_every_epoch
                           else None,
        )
        logger.info(f"Training done in {time.time() - t0:.1f}s")
        torch.save(model.state_dict(), weights_path)
        with open(output_dir / "history.json", "w") as f:
            json.dump(_to_jsonable(history), f, indent=2)

    # --- Per-block decoders ---
    image_size = cfg.get("dataset", {}).get("image_size", 64)
    dec_cfg = cfg.get("training", {}).get("decoders")
    # Upsampling stage of every decoder (bilinear resize-then-conv, or a
    # transposed conv). Read before building so eval-only loads the same
    # architecture the weights were trained with.
    upsample = str((dec_cfg or {}).get("upsample", "bilinear"))
    decoders = build_decoders(
        model, image_size=image_size, upsample=upsample,
    ).to(device)
    dec_history: dict | None = None
    decoders_path = output_dir / "weights" / "decoders.pt"

    if eval_only:
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
        logger.info("PHASE 2 — training per-block reconstruction decoders "
                    f"(upsample={upsample})")
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
            weight_decay=dec_cfg.get("weight_decay", 0.0),
            anchor_lambda=dec_cfg.get("anchor_lambda", 0.0),
            device=device,
            log_every=dec_cfg.get("log_every", 1),
            checkpoint_dir=(output_dir / "weights") if save_every_epoch
                           else None,
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

    # The headline bias/variance anomaly score needs the spread across
    # independently trained models, so it lives in run_ensemble.py. A single
    # model here reports the classifier OOD scores (msp / entropy) plus a
    # fused per-block reconstruction-error score (computed with the plots
    # below).

    # --- Plots ---
    if not no_plots:
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

        # --- Per-block reconstruction plots + fused error score -----------
        if decoders is not None:
            ecfg = cfg.get("evaluation", {})
            n_viz_in = int(ecfg.get(
                "n_viz_in_samples", ecfg.get("n_viz_samples", 4),
            ))
            n_viz_ood = int(ecfg.get(
                "n_viz_ood_samples", ecfg.get("n_viz_samples", 4),
            ))
            viz_seed = ecfg.get("viz_seed")
            samples_in = _gather_samples(
                loaders["test_in"], n_viz_in, seed=viz_seed,
            ).to(device)
            samples_ood = _gather_samples(
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
                torch.cat([ai, ao], dim=0)
                for ai, ao in zip(acts_in, acts_ood)
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

            in_fusion = _compute_fusion_scores(
                model, decoders, loaders["test_in"], device,
            )
            ood_fusion = _compute_fusion_scores(
                model, decoders, loaders["test_ood"], device,
            )
            fusion_auroc = plot_fusion_auroc(
                in_fusion, ood_fusion,
                plot_dir / "fusion_auroc.png",
                title="Fusion-based anomaly detection",
            )
            logger.info(
                f"Fusion-based AUROC (max-pixel) = {fusion_auroc:.4f}  "
                f"(in_mean={in_fusion.mean():.4f}, "
                f"ood_mean={ood_fusion.mean():.4f})"
            )
            results["fusion"] = {
                "auroc": fusion_auroc,
                "in_mean": float(in_fusion.mean()),
                "ood_mean": float(ood_fusion.mean()),
                "n_in": int(in_fusion.size),
                "n_ood": int(ood_fusion.size),
            }
            logger.info(
                "Wrote per-block plots: per_block_breakdown.png, "
                "recons_only.png, activations.png, "
                "fusion_overlay.png, fusion_auroc.png"
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
            "parameters": count_parameters(model),
        },
        "accuracy": _to_jsonable(results["accuracy"]),
        "auroc": _to_jsonable({
            k: {kk: vv for kk, vv in v.items() if kk in ("auroc", "in_mean", "ood_mean")}
            if isinstance(v, dict) else v
            for k, v in results["auroc"].items()
        }),
        "fusion": _to_jsonable(results.get("fusion", {})),
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
