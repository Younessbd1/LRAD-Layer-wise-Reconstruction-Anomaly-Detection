#!/usr/bin/env python3
"""LRAD Experiment Runner.

Usage:
    python scripts/run_experiment.py --config configs/mnist_mlp.yaml
    python scripts/run_experiment.py --config configs/mnist_cnn.yaml
    python scripts/run_experiment.py --config configs/cifar10_cnn.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lrad.utils.helpers import get_device, seed_everything, setup_logging, count_parameters
from lrad.data.datasets import get_mnist_loaders, get_cifar10_loaders
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_model import LRADModel
from lrad.models.lrad_nested import LRADModelNested
from lrad.engine.trainer import train_classifier, train_decoders, train_decoders_nested
from lrad.engine.evaluator import evaluate_full, compute_auroc, collect_anomaly_scores
from lrad.visualization.heatmaps import (
    plot_heatmap_grid,
    plot_score_distributions,
    plot_roc_curves,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_classifier(cfg: dict) -> CNNClassifier | MLPClassifier:
    """Instantiate classifier from config."""
    mcfg = cfg["model"]
    if mcfg["architecture"] == "cnn":
        return CNNClassifier(
            channels=mcfg["channels"],
            num_classes=mcfg["num_classes"],
            input_size=mcfg["input_size"],
        )
    elif mcfg["architecture"] == "mlp":
        return MLPClassifier(
            input_dim=mcfg["input_dim"],
            hidden_dims=mcfg["hidden_dims"],
            num_classes=mcfg["num_classes"],
            input_size=mcfg["input_size"],
        )
    else:
        raise ValueError(f"Unknown architecture: {mcfg['architecture']}")


def main():
    parser = argparse.ArgumentParser(description="Run LRAD experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--decoder_type", choices=["parallel", "nested"], default=None,
        help="Override decoder architecture. Falls back to model.decoder_type "
             "in the config, then to 'parallel'.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and load existing weights from the output directory.",
    )
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Path to directory containing classifier.pt and decoder_*.pt. "
             "Defaults to the experiment output directory.",
    )
    args = parser.parse_args()

    # --- Setup ---
    cfg = load_config(args.config)
    exp = cfg["experiment"]

    decoder_type = (
        args.decoder_type
        or cfg["model"].get("decoder_type", "parallel")
    ).lower()
    if decoder_type not in ("parallel", "nested"):
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    output_dir = Path(exp["output_dir"].rstrip("/") + f"_{decoder_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(exp.get("seed", 42))
    logger = setup_logging(str(output_dir / "logs"))
    device = get_device()

    logger.info(f"Experiment: {exp['name']}")
    logger.info(f"Decoder type: {decoder_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

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

    logger.info(f"Normal classes: {dcfg['normal_classes']}")

    # --- Build classifier ---
    classifier = build_classifier(cfg)
    logger.info(f"Classifier: {classifier}")
    logger.info(f"  Parameters: {count_parameters(classifier):,}")

    # --- Build LRAD model ---
    if decoder_type == "nested":
        model = LRADModelNested(classifier)
        train_fn = train_decoders_nested
    else:
        model = LRADModel(classifier)
        train_fn = train_decoders

    if args.eval_only:
        # Load saved weights, skip training entirely
        weights_dir = Path(args.weights_dir) if args.weights_dir else output_dir
        logger.info(f"--eval-only: loading weights from {weights_dir}")

        cls_path = weights_dir / "classifier.pt"
        if not cls_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {cls_path}")
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        logger.info(f"  Loaded classifier from {cls_path}")

        for i, dec in enumerate(model.decoders):
            dec_path = weights_dir / f"decoder_{i}.pt"
            if not dec_path.exists():
                raise FileNotFoundError(f"Decoder weights not found: {dec_path}")
            dec.load_state_dict(torch.load(dec_path, map_location=device))
            logger.info(f"  Loaded decoder {i} from {dec_path}")

    else:
        # --- Phase 1: Train classifier ---
        logger.info("=" * 50)
        logger.info("PHASE 1: Training classifier")
        logger.info("=" * 50)

        tcfg = cfg["training"]["classifier"]
        cls_history = train_classifier(
            classifier=classifier,
            train_loader=loaders["train"],
            class_map=loaders["class_map"],
            epochs=tcfg["epochs"],
            lr=tcfg["lr"],
            device=device,
        )

        torch.save(classifier.state_dict(), output_dir / "classifier.pt")

        logger.info(f"LRAD Model ({decoder_type}) built with {len(model.decoders)} decoders")
        for i, dec in enumerate(model.decoders):
            logger.info(f"  Decoder {i}: {dec} ({count_parameters(dec):,} params)")
        logger.info(f"  total decoder params: {sum(count_parameters(d) for d in model.decoders):,}")

        # --- Phase 2: Train decoders ---
        logger.info("=" * 50)
        logger.info(f"PHASE 2: Training decoders ({decoder_type})")
        logger.info("=" * 50)

        dcfg_train = cfg["training"]["decoders"]
        dec_history = train_fn(
            model=model,
            train_loader=loaders["train"],
            epochs=dcfg_train["epochs"],
            lr=dcfg_train["lr"],
            device=device,
        )

        for i, dec in enumerate(model.decoders):
            torch.save(dec.state_dict(), output_dir / f"decoder_{i}.pt")

    # --- Evaluation ---
    logger.info("=" * 50)
    logger.info("EVALUATION")
    logger.info("=" * 50)

    ecfg = cfg["evaluation"]
    results = evaluate_full(
        model=model,
        loaders=loaders,
        device=device,
        fusion=ecfg["fusion"],
        max_batches=ecfg.get("max_eval_batches"),
    )

    # Log AUROC results
    logger.info("-" * 40)
    for split, auroc in results["aurocs"].items():
        logger.info(f"  AUROC ({split}): {auroc:.4f}")
    logger.info("-" * 40)

    # --- Visualizations ---
    logger.info("Generating visualizations...")

    n_display = ecfg.get("n_display", 8)

    # 1. Normal heatmaps
    plot_heatmap_grid(
        images=results["normal"]["images"][:n_display],
        heatmaps=results["normal"]["heatmaps"][:n_display],
        per_layer_maps=[p[:n_display] for p in results["normal"]["per_layer_maps"]],
        reconstructions=[r[:n_display] for r in results["normal"].get("reconstructions", [])] or None,
        scores=results["normal"]["scores"][:n_display],
        title=f"{exp['name']} - Normal Samples",
        save_path=str(output_dir / "heatmaps_normal.png"),
    )

    # 2. Anomaly heatmaps per split
    for split_name in [k for k in results if k not in ("normal", "aurocs")]:
        split_data = results[split_name]
        plot_heatmap_grid(
            images=split_data["images"][:n_display],
            heatmaps=split_data["heatmaps"][:n_display],
            per_layer_maps=[p[:n_display] for p in split_data["per_layer_maps"]],
            reconstructions=[r[:n_display] for r in split_data.get("reconstructions", [])] or None,
            scores=split_data["scores"][:n_display],
            title=f"{exp['name']} - Anomaly: {split_name}",
            save_path=str(output_dir / f"heatmaps_{split_name}.png"),
        )

    # 3. Score distributions
    anomaly_scores = {}
    for split_name in [k for k in results if k not in ("normal", "aurocs")]:
        anomaly_scores[split_name] = results[split_name]["scores"]

    plot_score_distributions(
        normal_scores=results["normal"]["scores"],
        anomaly_scores_dict=anomaly_scores,
        title=f"{exp['name']} - Score Distributions",
        save_path=str(output_dir / "score_distributions.png"),
    )

    # 4. ROC curves
    roc_data = {}
    for split_name in [k for k in results if k not in ("normal", "aurocs")]:
        roc = compute_auroc(results["normal"]["scores"], results[split_name]["scores"])
        roc_data[split_name] = roc

    plot_roc_curves(
        auroc_results=roc_data,
        title=f"{exp['name']} - ROC Curves",
        save_path=str(output_dir / "roc_curves.png"),
    )

    plt.close("all")

    # --- Save summary ---
    summary = {
        "experiment": exp["name"],
        "normal_classes": dcfg["normal_classes"],
        "architecture": cfg["model"]["architecture"],
        "decoder_type": decoder_type,
        "aurocs": {k: float(v) for k, v in results["aurocs"].items()},
        **(
            {}
            if args.eval_only
            else {
                "classifier_final_acc": cls_history["accuracies"][-1],
                "classifier_final_loss": cls_history["losses"][-1],
            }
        ),
        "decoder_params_total": int(sum(count_parameters(d) for d in model.decoders)),
    }

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All outputs saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
