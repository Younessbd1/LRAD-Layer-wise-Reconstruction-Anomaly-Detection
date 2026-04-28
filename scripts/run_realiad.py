#!/usr/bin/env python3
"""End-to-end Real-IAD experiment runner for LRAD.

Pipeline stages (all driven from a single YAML config):

  0. load config, seed everything, create output tree
  1. build Real-IAD dataloaders (train / val / test_normal / test_anomaly)
  2. build DeepCNNClassifier + LRADModel (decoder per stage)
  3. Phase 1 — train the classifier on a pretext task (rotation or category)
     while tracking train+val loss
  4. Phase 2 — train the stage-wise decoders with train+val MSE tracking
  5. Evaluate on test_normal + test_anomaly:
        image-level: AUROC, PR-AUC, F1, precision, recall, threshold
        pixel-level: AUROC, PR-AUC (when masks are available)
  6. Emit all visualisations (training curves, score distributions, ROC/PR,
     heatmap gallery, feature-space scatter, class distribution).
  7. Persist weights + a summary.json with every metric + config snapshot.

Usage::

    python scripts/run_realiad.py --config configs/realiad.yaml
    python scripts/run_realiad.py --config configs/realiad.yaml --eval-only
    python scripts/run_realiad.py --config configs/realiad.yaml \\
        --override model.preset=resnet34 training.decoders.epochs=80
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe backend

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Keep script runnable both via `python scripts/run_realiad.py` and after
# `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.data.realiad import get_realiad_loaders, load_manifest
from lrad.engine import (
    collect_features,
    evaluate_realiad,
    train_classifier_pretext,
    train_decoders_with_val,
)
from lrad.models import LRADModel, build_deep_cnn
from lrad.utils.helpers import count_parameters, get_device, seed_everything, setup_logging
from lrad.visualization import (
    plot_class_distribution,
    plot_feature_scatter,
    plot_heatmap_grid,
    plot_pr_curves,
    plot_reconstruction_gallery,
    plot_roc_curves,
    plot_score_distributions,
    plot_training_curves,
)

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
    # Try to coerce to the most useful type.
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
# Model building
# ---------------------------------------------------------------------------

def build_classifier(cfg: dict, n_categories: int):
    mcfg = cfg["model"]
    kwargs = {
        "in_channels": mcfg.get("in_channels", 3),
        "input_size": mcfg.get("input_size", 224),
    }

    pretext = cfg["training"]["classifier"].get("pretext", "rotation")
    # Head size depends on the pretext: 4 for rotation, n_categories for
    # the category-classification variant.
    if pretext == "rotation":
        kwargs["num_classes"] = 4
    elif pretext == "category":
        kwargs["num_classes"] = max(1, n_categories)
    else:
        raise ValueError(f"unknown pretext {pretext!r}")

    preset = mcfg.get("preset")
    if preset:
        cls = build_deep_cnn(preset=preset, **kwargs)
    else:
        # Explicit structure from config.
        kwargs["stage_channels"] = mcfg.get("stage_channels", [64, 128, 256, 512])
        kwargs["blocks_per_stage"] = mcfg.get("blocks_per_stage", [2, 2, 2, 2])
        if "stem" in mcfg:
            kwargs["stem"] = mcfg["stem"]
        cls = build_deep_cnn(preset=None, **kwargs)
    return cls, pretext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_distribution_from_manifest(manifest_path: str, categories: list[str] | None
                                       ) -> tuple[dict[str, int], dict[str, int]]:
    samples = load_manifest(manifest_path)
    allow = set(categories) if categories else None
    ok: dict[str, int] = {}
    ng: dict[str, int] = {}
    for s in samples:
        if allow is not None and s.category not in allow:
            continue
        bucket = ok if s.label == 0 else ng
        bucket[s.category] = bucket.get(s.category, 0) + 1
    return ok, ng


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        # Full arrays (e.g. ROC/PR curves) would bloat summary.json — drop.
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _pick_gallery_samples(scores: np.ndarray, n: int) -> np.ndarray:
    """Return indices that bracket the score distribution: a few most
    anomalous, a few near median, a few most normal."""
    order = np.argsort(scores)
    if len(order) <= n:
        return order
    picks = [order[-i - 1] for i in range(n // 2)]         # top anomalies
    mid = len(order) // 2
    picks += [order[mid + i - n // 4] for i in range(n - len(picks))]
    return np.unique(np.array(picks))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run LRAD on Real-IAD.")
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
    setup_logging(str(output_dir / "logs"))
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"Experiment  : {exp['name']}")
    logger.info(f"Config      : {args.config}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Seed        : {exp.get('seed', 42)}")
    logger.info("=" * 70)

    # Snapshot the resolved config so reruns are reproducible.
    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Data ---
    dcfg = cfg["dataset"]
    if dcfg["name"] != "realiad":
        raise ValueError(f"this script only handles realiad, got {dcfg['name']!r}")

    pretext = cfg["training"]["classifier"].get("pretext", "rotation")
    return_cat = (pretext == "category")

    aug_cfg = dcfg.get("augmentation", {}) or {}
    loaders = get_realiad_loaders(
        manifest=dcfg["manifest"],
        categories=dcfg.get("categories"),
        image_size=dcfg.get("image_size", 224),
        batch_size=dcfg.get("batch_size", 32),
        num_workers=dcfg.get("num_workers", 4),
        train_ratio=dcfg.get("train_ratio", 0.8),
        val_ratio=dcfg.get("val_ratio", 0.1),
        load_masks=dcfg.get("load_masks", True),
        return_category=return_cat,
        pin_memory=True,
        seed=exp.get("seed", 42),
        augment_train=aug_cfg.get("enabled", True),
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.85, 1.0])),
        color_jitter=aug_cfg.get("color_jitter", 0.2),
        blur_p=aug_cfg.get("blur_p", 0.1),
    )
    logger.info(
        f"Dataloaders  train={len(loaders['train'].dataset)}  "
        f"val={len(loaders['val'].dataset)}  "
        f"test_normal={len(loaders['test_normal'].dataset)}  "
        f"test_anomaly={len(loaders['test_anomaly'].dataset)}"
    )
    logger.info(f"Categories   : {loaders['categories']}")

    # --- Model ---
    classifier, pretext = build_classifier(cfg, loaders["n_categories"])
    logger.info(f"Classifier   : {classifier}")
    logger.info(f"  parameters : {count_parameters(classifier):,}")
    model = LRADModel(classifier)
    for i, d in enumerate(model.decoders):
        logger.info(f"  Decoder {i} : {d}  ({count_parameters(d):,} params)")
    logger.info(f"  total dec  : {sum(count_parameters(d) for d in model.decoders):,}")

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
        logger.info(f"PHASE 1 — classifier pretext training ({pretext})")
        logger.info("=" * 70)
        tcls = cfg["training"]["classifier"]
        t0 = time.time()
        cls_history = train_classifier_pretext(
            classifier,
            loaders["train"],
            loaders["val"],
            epochs=tcls.get("epochs", 30),
            lr=tcls.get("lr", 1e-3),
            weight_decay=tcls.get("weight_decay", 1e-4),
            pretext=pretext,
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

    # --- Phase 2 ---
    if args.eval_only:
        for i, dec in enumerate(model.decoders):
            p = output_dir / "weights" / f"decoder_{i}.pt"
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")
            dec.load_state_dict(torch.load(p, map_location=device))
            logger.info(f"Loaded decoder {i} weights from {p}")
    else:
        logger.info("=" * 70)
        logger.info("PHASE 2 — decoder training")
        logger.info("=" * 70)
        tdec = cfg["training"]["decoders"]
        t0 = time.time()
        dec_history = train_decoders_with_val(
            model,
            loaders["train"],
            loaders["val"],
            epochs=tdec.get("epochs", 30),
            lr=tdec.get("lr", 1e-3),
            weight_decay=tdec.get("weight_decay", 1e-5),
            device=device,
            log_every=tdec.get("log_every", 5),
            save_best_path=str(output_dir / "weights" / "decoders_best.pt"),
        )
        logger.info(f"Phase 2 done in {time.time() - t0:.1f}s")
        for i, dec in enumerate(model.decoders):
            torch.save(dec.state_dict(),
                       output_dir / "weights" / f"decoder_{i}.pt")
        with open(output_dir / "dec_history.json", "w") as f:
            json.dump(dec_history, f, indent=2)

    # --- Evaluation ---
    logger.info("=" * 70)
    logger.info("EVALUATION")
    logger.info("=" * 70)
    ecfg = cfg.get("evaluation", {})
    results = evaluate_realiad(
        model,
        loaders,
        device=device,
        fusion=ecfg.get("fusion", "mean"),
        max_batches=ecfg.get("max_eval_batches"),
        keep_masks=dcfg.get("load_masks", True),
    )

    # --- Visualisations ---
    if not args.no_plots:
        logger.info("=" * 70)
        logger.info("VISUALISATIONS")
        logger.info("=" * 70)
        plot_dir = output_dir / "plots"

        # Class distribution (no image IO — manifest-only)
        try:
            ok_per, ng_per = _class_distribution_from_manifest(
                dcfg["manifest"], dcfg.get("categories"))
            plot_class_distribution(
                ok_per, ng_per,
                title="Real-IAD — class distribution",
                save_path=str(plot_dir / "class_distribution.png"))
        except Exception as e:
            logger.warning(f"class distribution plot failed: {e}")

        # Training curves
        if cls_history is not None or dec_history is not None:
            plot_training_curves(
                cls_history, dec_history,
                title=f"{exp['name']} — training curves",
                save_path=str(plot_dir / "training_curves.png"))

        # Score distribution + ROC + PR
        plot_score_distributions(
            normal_scores=results["normal"]["scores"],
            anomaly_scores_dict={"anomaly": results["anomaly"]["scores"]},
            title=f"{exp['name']} — score distribution",
            save_path=str(plot_dir / "score_distribution.png"))

        roc_pack = {"anomaly": {
            "fpr": results["image"]["fpr"],
            "tpr": results["image"]["tpr"],
            "auroc": results["image"]["auroc"],
        }}
        plot_roc_curves(roc_pack, title=f"{exp['name']} — ROC",
                        save_path=str(plot_dir / "roc_curves.png"))

        pr_pack = {"anomaly": {
            "precisions": results["image"]["precisions"],
            "recalls": results["image"]["recalls"],
            "pr_auc": results["image"]["pr_auc"],
            "f1": results["image"]["f1"],
        }}
        plot_pr_curves(pr_pack, title=f"{exp['name']} — Precision-Recall",
                       save_path=str(plot_dir / "pr_curves.png"))

        # Heatmap gallery: normal samples
        n_disp = ecfg.get("n_display", 8)
        plot_heatmap_grid(
            images=results["normal"]["images"][:n_disp],
            heatmaps=results["normal"]["heatmaps"][:n_disp],
            per_layer_maps=[p[:n_disp] for p in results["normal"]["per_layer_maps"]],
            reconstructions=[r[:n_disp] for r in results["normal"]["reconstructions"]],
            scores=results["normal"]["scores"][:n_disp],
            title=f"{exp['name']} — normal samples",
            save_path=str(plot_dir / "heatmaps_normal.png"))

        # Heatmap gallery: anomaly samples (pick a bracketed selection)
        idx = _pick_gallery_samples(results["anomaly"]["scores"], n_disp)
        plot_heatmap_grid(
            images=results["anomaly"]["images"][idx],
            heatmaps=results["anomaly"]["heatmaps"][idx],
            per_layer_maps=[p[idx] for p in results["anomaly"]["per_layer_maps"]],
            reconstructions=[r[idx] for r in results["anomaly"]["reconstructions"]],
            scores=results["anomaly"]["scores"][idx],
            title=f"{exp['name']} — anomaly samples",
            save_path=str(plot_dir / "heatmaps_anomaly.png"))

        # Reconstruction gallery (more compact format, great for slides)
        plot_reconstruction_gallery(
            images=results["anomaly"]["images"][idx],
            reconstructions=[r[idx] for r in results["anomaly"]["reconstructions"]],
            heatmaps=results["anomaly"]["heatmaps"][idx],
            titles=[f"s={s:.3f}" for s in results["anomaly"]["scores"][idx]],
            n=len(idx),
            save_path=str(plot_dir / "reconstruction_gallery.png"))

        # Feature-space scatter
        try:
            feat_n, _ = collect_features(
                model, loaders["test_normal"], device=device,
                layer_idx=-1, max_samples=ecfg.get("feat_max", 1000))
            feat_a, _ = collect_features(
                model, loaders["test_anomaly"], device=device,
                layer_idx=-1, max_samples=ecfg.get("feat_max", 1000))
            plot_feature_scatter(
                feat_n, feat_a,
                title=f"{exp['name']} — PCA of last-stage activations",
                save_path=str(plot_dir / "feature_scatter.png"))
        except Exception as e:
            logger.warning(f"feature scatter failed: {e}")

        plt.close("all")
    else:
        logger.info("--no-plots set, skipping visualisations")

    # --- Summary ---
    summary = {
        "experiment": exp["name"],
        "seed": exp.get("seed", 42),
        "device": str(device),
        "categories": loaders["categories"],
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test_normal": len(loaders["test_normal"].dataset),
        "n_test_anomaly": len(loaders["test_anomaly"].dataset),
        "classifier": {
            "architecture": "DeepCNNClassifier",
            "pretext": pretext,
            "stages": classifier.stage_channels,
            "blocks_per_stage": classifier.blocks_per_stage,
            "input_size": classifier.input_size,
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
            "pixel": _to_jsonable(results["pixel"]) if results["pixel"] else None,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {output_dir / 'summary.json'}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
