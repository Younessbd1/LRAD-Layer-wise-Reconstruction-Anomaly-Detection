#!/usr/bin/env python3
"""LRAD-UQ Experiment Runner.

Runs the full uncertainty-quantified anomaly detection pipeline.

Usage:
    python scripts/run_uq_experiment.py --config configs/mnist_cnn_uq.yaml
"""

import argparse
import sys
import json
from pathlib import Path

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lrad.utils.helpers import get_device, seed_everything, setup_logging, count_parameters
from lrad.data.datasets import get_mnist_loaders, get_cifar10_loaders
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_uq import LRADModelUQ
from lrad.engine.trainer import train_classifier
from lrad.engine.uq_engine import train_uq_decoders, full_uq_evaluation
from lrad.visualization.uq_plots import (
    plot_uq_heatmap_grid,
    plot_uq_score_comparison,
    plot_uncertainty_calibration,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run LRAD-UQ experiment")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    out = Path(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(exp.get("seed", 42))
    logger = setup_logging(str(out / "logs"))
    device = get_device()

    logger.info(f"LRAD-UQ Experiment: {exp['name']}")
    logger.info(f"Device: {device}")
    logger.info(f"UQ Method: {cfg['uq']['method']}")

    # --- Data ---
    dcfg = cfg["dataset"]
    if dcfg["name"] == "mnist":
        loaders = get_mnist_loaders(
            normal_classes=dcfg["normal_classes"],
            batch_size=dcfg["batch_size"],
            data_root=dcfg["data_root"],
        )
    elif dcfg["name"] == "cifar10":
        loaders = get_cifar10_loaders(
            normal_classes=dcfg["normal_classes"],
            batch_size=dcfg["batch_size"],
            data_root=dcfg["data_root"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dcfg['name']}")

    # --- Build classifier ---
    mcfg = cfg["model"]
    if mcfg["architecture"] == "cnn":
        classifier = CNNClassifier(
            mcfg["channels"], mcfg["num_classes"], mcfg["input_size"]
        )
    else:
        classifier = MLPClassifier(
            mcfg["input_dim"], mcfg["hidden_dims"],
            mcfg["num_classes"], mcfg["input_size"],
        )

    logger.info(f"Classifier: {classifier}")

    # --- Phase 1: Train classifier ---
    logger.info("=" * 50)
    logger.info("PHASE 1: Training classifier")
    cls_history = train_classifier(
        classifier, loaders["train"], loaders["class_map"],
        epochs=cfg["training"]["classifier"]["epochs"],
        lr=cfg["training"]["classifier"]["lr"],
        device=device,
    )
    torch.save(classifier.state_dict(), out / "classifier.pt")

    # --- Build UQ model ---
    uq_cfg = cfg["uq"]
    model = LRADModelUQ(
        classifier,
        uq_method=uq_cfg["method"],
        mc_samples=uq_cfg.get("mc_samples", 20),
        n_ensemble=uq_cfg.get("n_ensemble", 5),
        dropout_rate=uq_cfg.get("dropout_rate", 0.2),
    )
    logger.info(f"UQ Model: {len(model.decoders)} decoders, method={uq_cfg['method']}")
    for i, dec in enumerate(model.decoders):
        logger.info(f"  Decoder {i}: {count_parameters(dec):,} params")

    # --- Phase 2: Train UQ decoders ---
    logger.info("=" * 50)
    logger.info("PHASE 2: Training UQ decoders")
    train_uq_decoders(
        model, loaders["train"],
        epochs=cfg["training"]["decoders"]["epochs"],
        lr=cfg["training"]["decoders"]["lr"],
        device=device,
    )

    # Save decoders
    for i, dec in enumerate(model.decoders):
        torch.save(dec.state_dict(), out / f"uq_decoder_{i}.pt")

    # --- Evaluation ---
    logger.info("=" * 50)
    logger.info("UQ EVALUATION")
    ecfg = cfg["evaluation"]
    results = full_uq_evaluation(
        model, loaders, device,
        fusion=ecfg["fusion"],
        max_batches=ecfg.get("max_eval_batches"),
    )

    # --- Visualizations ---
    logger.info("Generating UQ visualizations...")
    n = ecfg.get("n_display", 8)

    # Normal UQ heatmaps
    plot_uq_heatmap_grid(
        results["normal"]["images"][:n],
        results["normal"]["error_maps"][:n],
        results["normal"]["uncertainty_maps"][:n],
        results["normal"]["combined_maps"][:n],
        scores_combined=results["normal"]["scores_combined"][:n],
        title=f"{exp['name']} — Normal (UQ)",
        save_path=str(out / "uq_normal.png"),
    )

    # Anomaly UQ heatmaps + score comparison + calibration
    for split_name in [k for k in results if k not in ("normal", "aurocs")]:
        data = results[split_name]

        plot_uq_heatmap_grid(
            data["images"][:n],
            data["error_maps"][:n],
            data["uncertainty_maps"][:n],
            data["combined_maps"][:n],
            scores_combined=data["scores_combined"][:n],
            title=f"{exp['name']} — Anomaly: {split_name} (UQ)",
            save_path=str(out / f"uq_{split_name}.png"),
        )

        plot_uq_score_comparison(
            results["normal"], data, split_name=split_name,
            save_path=str(out / f"uq_scores_{split_name}.png"),
        )

        plot_uncertainty_calibration(
            results["normal"], data,
            save_path=str(out / f"uq_calibration_{split_name}.png"),
        )

    plt.close("all")

    # --- Summary ---
    summary = {
        "experiment": exp["name"],
        "uq_method": uq_cfg["method"],
        "aurocs": {k: float(v) for k, v in results["aurocs"].items()},
        "classifier_final_acc": cls_history["accuracies"][-1],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("-" * 40)
    for k, v in results["aurocs"].items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("-" * 40)
    logger.info(f"Outputs saved to {out}/")


if __name__ == "__main__":
    main()
