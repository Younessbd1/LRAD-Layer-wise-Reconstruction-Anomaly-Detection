#!/usr/bin/env python3
"""Compare parallel-decoder vs nested-decoder LRAD architectures.

Trains a single classifier, then trains both decoder families on top of it and
compares them on:
  - Total decoder parameters
  - Per-layer training MSE curves
  - Per-layer reconstruction MSE on the test set
  - Per-split AUROC
  - Side-by-side anomaly heatmaps

Usage:
    python scripts/run_comparison.py --config configs/mnist_cnn.yaml
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lrad.utils.helpers import (
    get_device, seed_everything, setup_logging, count_parameters,
)
from lrad.data.datasets import get_mnist_loaders, get_cifar10_loaders
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_model import LRADModel
from lrad.models.lrad_nested import LRADModelNested
from lrad.engine.trainer import (
    train_classifier, train_decoders, train_decoders_nested,
)
from lrad.engine.evaluator import evaluate_full, compute_auroc
from lrad.visualization.heatmaps import plot_heatmap_grid, plot_heatmap_grid_sidebyside


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_classifier(cfg: dict):
    mcfg = cfg["model"]
    if mcfg["architecture"] == "cnn":
        return CNNClassifier(
            channels=mcfg["channels"],
            num_classes=mcfg["num_classes"],
            input_size=mcfg["input_size"],
        )
    return MLPClassifier(
        input_dim=mcfg["input_dim"],
        hidden_dims=mcfg["hidden_dims"],
        num_classes=mcfg["num_classes"],
        input_size=mcfg["input_size"],
    )


def get_loaders(cfg: dict):
    dcfg = cfg["dataset"]
    if dcfg["name"] == "mnist":
        return get_mnist_loaders(
            normal_classes=dcfg["normal_classes"],
            batch_size=dcfg["batch_size"],
            data_root=dcfg["data_root"],
        )
    return get_cifar10_loaders(
        normal_classes=dcfg["normal_classes"],
        batch_size=dcfg["batch_size"],
        data_root=dcfg["data_root"],
    )


def plot_training_curves(parallel_hist, nested_hist, save_path):
    n = len(parallel_hist["losses"])
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    axes = axes[0]

    for k in range(n):
        ax = axes[k]
        epochs = range(1, len(parallel_hist["losses"][k]) + 1)
        ax.plot(epochs, parallel_hist["losses"][k], label="parallel", color="#1e40af", linewidth=2)
        ax.plot(epochs, nested_hist["losses"][k], label="nested (local target)", color="#059669", linewidth=2)
        ax.set_title(f"Decoder {k}", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training MSE per decoder - parallel vs nested\n"
                 "(targets differ: parallel=image; nested D₀=image, Dₖ=a_{k-1})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_recon_mse_bars(parallel_mse, nested_mse, save_path):
    n = len(parallel_mse)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width/2, parallel_mse, width, label="parallel", color="#1e40af")
    ax.bar(x + width/2, nested_mse, width, label="nested cascade", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels([f"x̂_{k}" for k in range(n)])
    ax.set_ylabel("Mean reconstruction MSE  (test_normal)")
    ax.set_title("Per-level reconstruction error vs original image", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (p, ne) in enumerate(zip(parallel_mse, nested_mse)):
        ax.text(i - width/2, p, f"{p:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, ne, f"{ne:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_params_bars(parallel_params, nested_params, save_path):
    n = max(len(parallel_params), len(nested_params))
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width/2, parallel_params, width, label=f"parallel (total: {sum(parallel_params):,})", color="#1e40af")
    ax.bar(x + width/2, nested_params, width, label=f"nested (total: {sum(nested_params):,})", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{k}" for k in range(n)])
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Decoder parameter count - parallel vs nested", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_auroc_bars(parallel_aurocs, nested_aurocs, save_path):
    splits = list(parallel_aurocs.keys())
    n = len(splits)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    p_vals = [parallel_aurocs[s] for s in splits]
    n_vals = [nested_aurocs[s] for s in splits]
    ax.bar(x - width/2, p_vals, width, label="parallel", color="#1e40af")
    ax.bar(x + width/2, n_vals, width, label="nested", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_title("Per-split AUROC - parallel vs nested", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (p, ne) in enumerate(zip(p_vals, n_vals)):
        ax.text(i - width/2, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, ne, f"{ne:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def per_level_recon_mse(eval_result):
    """Mean per-level reconstruction MSE across all images in the split."""
    images = eval_result["images"]  # (N, C, H, W)
    recons = eval_result["reconstructions"]  # list of (N, C, H, W)
    out = []
    for r in recons:
        diff = (r - images) ** 2
        out.append(float(diff.mean()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output_dir", default=None,
        help="Override output dir (default: config.experiment.output_dir + '_compare')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    output_dir = Path(args.output_dir or (exp["output_dir"].rstrip("/") + "_compare"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(exp.get("seed", 42))
    logger = setup_logging(str(output_dir / "logs"))
    device = get_device()

    logger.info(f"Comparison experiment: {exp['name']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    # ---- Data ----
    loaders = get_loaders(cfg)
    logger.info(f"Normal classes: {cfg['dataset']['normal_classes']}")

    # ---- Train classifier ONCE ----
    classifier = build_classifier(cfg)
    logger.info(f"Classifier: {classifier}")
    logger.info(f"  params: {count_parameters(classifier):,}")

    logger.info("=" * 60)
    logger.info("PHASE 1: Train classifier")
    logger.info("=" * 60)
    tcfg = cfg["training"]["classifier"]
    train_classifier(
        classifier=classifier,
        train_loader=loaders["train"],
        class_map=loaders["class_map"],
        epochs=tcfg["epochs"],
        lr=tcfg["lr"],
        device=device,
    )
    torch.save(classifier.state_dict(), output_dir / "classifier.pt")

    # Make a deep copy so the two LRAD models are guaranteed to share identical
    # frozen weights and not interfere with each other.
    classifier_for_parallel = classifier
    classifier_for_nested = copy.deepcopy(classifier).to(device)
    classifier_for_nested.freeze()

    # ---- Build both LRAD wrappers ----
    parallel_model = LRADModel(classifier_for_parallel)
    nested_model = LRADModelNested(classifier_for_nested)

    parallel_params = [count_parameters(d) for d in parallel_model.decoders]
    nested_params = [count_parameters(d) for d in nested_model.decoders]

    logger.info("--- Parallel decoders ---")
    for i, d in enumerate(parallel_model.decoders):
        logger.info(f"  D{i}: {d}  ({parallel_params[i]:,} params)")
    logger.info(f"  total: {sum(parallel_params):,}")

    logger.info("--- Nested decoders ---")
    for i, d in enumerate(nested_model.decoders):
        logger.info(f"  D{i}: {d}  ({nested_params[i]:,} params)")
    logger.info(f"  total: {sum(nested_params):,}")

    # ---- Train both ----
    dcfg = cfg["training"]["decoders"]

    logger.info("=" * 60)
    logger.info("PHASE 2a: Train PARALLEL decoders")
    logger.info("=" * 60)
    parallel_hist = train_decoders(
        model=parallel_model, train_loader=loaders["train"],
        epochs=dcfg["epochs"], lr=dcfg["lr"], device=device,
    )

    logger.info("=" * 60)
    logger.info("PHASE 2b: Train NESTED decoders")
    logger.info("=" * 60)
    nested_hist = train_decoders_nested(
        model=nested_model, train_loader=loaders["train"],
        epochs=dcfg["epochs"], lr=dcfg["lr"], device=device,
    )

    # ---- Evaluate both ----
    ecfg = cfg["evaluation"]
    logger.info("=" * 60)
    logger.info("EVAL: parallel")
    logger.info("=" * 60)
    parallel_results = evaluate_full(
        model=parallel_model, loaders=loaders, device=device,
        fusion=ecfg["fusion"], max_batches=ecfg.get("max_eval_batches"),
    )

    logger.info("=" * 60)
    logger.info("EVAL: nested")
    logger.info("=" * 60)
    nested_results = evaluate_full(
        model=nested_model, loaders=loaders, device=device,
        fusion=ecfg["fusion"], max_batches=ecfg.get("max_eval_batches"),
    )

    # ---- Per-level reconstruction MSE on test_normal ----
    parallel_recon_mse = per_level_recon_mse(parallel_results["normal"])
    nested_recon_mse = per_level_recon_mse(nested_results["normal"])

    logger.info("--- Reconstruction MSE on test_normal ---")
    for k in range(len(parallel_recon_mse)):
        logger.info(
            f"  level {k}:  parallel={parallel_recon_mse[k]:.6f}  "
            f"nested={nested_recon_mse[k]:.6f}"
        )

    # ---- Plots ----
    logger.info("Generating comparison plots...")

    plot_training_curves(parallel_hist, nested_hist, output_dir / "training_curves.png")
    plot_recon_mse_bars(parallel_recon_mse, nested_recon_mse, output_dir / "recon_mse_bars.png")
    plot_params_bars(parallel_params, nested_params, output_dir / "params_bars.png")
    plot_auroc_bars(
        parallel_results["aurocs"], nested_results["aurocs"],
        output_dir / "auroc_bars.png",
    )

    # Side-by-side heatmap comparison per split (both architectures in one figure)
    n_disp = ecfg.get("n_display", 8)
    splits = ["normal"] + [k for k in parallel_results if k not in ("normal", "aurocs")]
    for split in splits:
        p_data = parallel_results[split]
        n_data = nested_results[split]
        plot_heatmap_grid_sidebyside(
            images=p_data["images"][:n_disp],
            parallel_data={
                "heatmaps": p_data["heatmaps"][:n_disp],
                "scores": p_data["scores"][:n_disp],
            },
            nested_data={
                "heatmaps": n_data["heatmaps"][:n_disp],
                "scores": n_data["scores"][:n_disp],
            },
            n_samples=n_disp,
            title=f"{exp['name']} - parallel vs nested - {split}",
            save_path=str(output_dir / f"heatmaps_{split}_compare.png"),
        )
        plt.close("all")

    # ---- Summary ----
    summary = {
        "experiment": exp["name"],
        "architecture": cfg["model"]["architecture"],
        "normal_classes": cfg["dataset"]["normal_classes"],
        "decoder_params": {
            "parallel": {f"D{i}": p for i, p in enumerate(parallel_params)},
            "parallel_total": int(sum(parallel_params)),
            "nested": {f"D{i}": p for i, p in enumerate(nested_params)},
            "nested_total": int(sum(nested_params)),
            "ratio_nested_over_parallel": float(sum(nested_params) / max(sum(parallel_params), 1)),
        },
        "recon_mse_test_normal": {
            "parallel": parallel_recon_mse,
            "nested": nested_recon_mse,
        },
        "aurocs": {
            "parallel": {k: float(v) for k, v in parallel_results["aurocs"].items()},
            "nested": {k: float(v) for k, v in nested_results["aurocs"].items()},
        },
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Outputs in: {output_dir}/")
    logger.info(f"Summary: {output_dir}/comparison_summary.json")


if __name__ == "__main__":
    main()
