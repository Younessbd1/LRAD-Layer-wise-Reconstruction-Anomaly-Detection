#!/usr/bin/env python3
"""Inspect a trained LRAD model: visualize activations, reconstructions, and per-layer errors.

Loads classifier.pt + decoder_*.pt from an experiment's output dir and produces
diagnostic plots showing what's happening *before* heatmap fusion.

Usage:
    python scripts/inspect_model.py --config configs/mnist_cnn.yaml
    python scripts/inspect_model.py --config configs/mnist_cnn.yaml --n-samples 4
    python scripts/inspect_model.py --config configs/mnist_cnn.yaml --split normal
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lrad.utils.helpers import get_device, seed_everything, setup_logging
from lrad.data.datasets import get_mnist_loaders, get_cifar10_loaders
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.lrad_model import LRADModel
from lrad.visualization.activations import (
    plot_feature_maps,
    plot_activation_stats,
    plot_per_layer_reconstructions,
    plot_per_layer_errors,
    plot_activation_distributions,
    format_stats_report,
)


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
    if dcfg["name"] == "cifar10":
        return get_cifar10_loaders(
            normal_classes=dcfg["normal_classes"],
            batch_size=dcfg["batch_size"],
            data_root=dcfg["data_root"],
        )
    raise ValueError(f"Unknown dataset: {dcfg['name']}")


def pick_samples(loader, n: int, device):
    """Grab the first n samples from a loader."""
    images = []
    for batch, _ in loader:
        images.append(batch)
        if sum(b.shape[0] for b in images) >= n:
            break
    x = torch.cat(images, dim=0)[:n].to(device)
    return x


def inspect_split(
    model: LRADModel,
    x: torch.Tensor,
    split_name: str,
    out_dir: Path,
    n_samples: int,
    logger,
) -> str:
    """Run model on a batch and produce all inspection plots."""
    logger.info(f"Inspecting split: {split_name} ({x.shape[0]} samples)")
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        fwd = model.forward(x)
        maps = model.compute_anomaly_maps(x, fusion="mean")

    activations = fwd["activations"]
    reconstructions = fwd["reconstructions"]
    per_layer_maps = maps["per_layer"]

    # 1. Feature maps per sample
    for i in range(min(n_samples, x.shape[0])):
        plot_feature_maps(
            activations=activations,
            sample_idx=i,
            top_k=16,
            title=f"{split_name} — sample {i}: feature maps",
            save_path=str(split_dir / f"feature_maps_sample_{i}.png"),
        )
        plt.close("all")

    # 2. Activation stats (single plot, uses full batch)
    plot_activation_stats(
        activations=activations,
        title=f"{split_name} — activation statistics",
        save_path=str(split_dir / "activation_stats.png"),
    )
    plt.close("all")

    # 3. Per-layer reconstructions
    plot_per_layer_reconstructions(
        images=x.cpu().numpy(),
        reconstructions=[r.cpu().numpy() for r in reconstructions],
        n_samples=n_samples,
        title=f"{split_name} — per-decoder reconstructions",
        save_path=str(split_dir / "per_layer_reconstructions.png"),
    )
    plt.close("all")

    # 4. Per-layer error heatmaps (with reconstructions alongside, shared vmax)
    plot_per_layer_errors(
        images=x.cpu().numpy(),
        per_layer_maps=[m.cpu().numpy() for m in per_layer_maps],
        reconstructions=[r.cpu().numpy() for r in reconstructions],
        n_samples=n_samples,
        title=f"{split_name} — per-layer reconstructions and error heatmaps",
        save_path=str(split_dir / "per_layer_errors.png"),
    )
    plt.close("all")

    # 5. Activation histograms
    plot_activation_distributions(
        activations=activations,
        title=f"{split_name} — activation distributions",
        save_path=str(split_dir / "activation_distributions.png"),
    )
    plt.close("all")

    # Text report
    report = format_stats_report(activations, per_layer_maps)
    report_path = split_dir / "stats_report.txt"
    report_path.write_text(report)

    logger.info(f"  Saved plots + stats to {split_dir}/")
    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect a trained LRAD model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--n-samples", type=int, default=6, help="Samples to display per split")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "normal", "anomaly"],
                        help="Which split(s) to inspect")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    base_out = Path(exp["output_dir"])
    inspect_out = base_out / "inspection"
    inspect_out.mkdir(parents=True, exist_ok=True)

    seed_everything(exp.get("seed", 42))
    logger = setup_logging(str(inspect_out / "logs"))
    device = get_device()

    logger.info(f"Inspecting experiment: {exp['name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {inspect_out}")

    # Check required checkpoint files exist
    classifier_ckpt = base_out / "classifier.pt"
    if not classifier_ckpt.exists():
        logger.error(f"Missing {classifier_ckpt}. Run scripts/run_experiment.py first.")
        sys.exit(1)

    # Load classifier
    classifier = build_classifier(cfg)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    classifier.to(device)
    classifier.freeze()
    logger.info(f"Loaded classifier from {classifier_ckpt}")

    # Build LRAD model (this also builds decoder architectures)
    model = LRADModel(classifier).to(device)

    # Load each decoder checkpoint
    for i, dec in enumerate(model.decoders):
        dec_path = base_out / f"decoder_{i}.pt"
        if not dec_path.exists():
            logger.error(f"Missing {dec_path}")
            sys.exit(1)
        dec.load_state_dict(torch.load(dec_path, map_location=device))
        logger.info(f"Loaded decoder {i} from {dec_path}")

    model.eval()

    # Load data
    loaders = get_loaders(cfg)

    # Pick which splits to inspect
    splits_to_run = []
    if args.split in ("all", "normal"):
        splits_to_run.append(("normal", loaders["test_normal"]))
    if args.split in ("all", "anomaly"):
        for key in loaders:
            if key.startswith("test_anomaly"):
                name = key.replace("test_anomaly_", "").replace("test_anomaly", "anomaly")
                splits_to_run.append((name, loaders[key]))

    full_report = []
    for split_name, loader in splits_to_run:
        x = pick_samples(loader, n=args.n_samples, device=device)
        report = inspect_split(model, x, split_name, inspect_out, args.n_samples, logger)
        full_report.append(f"\n### {split_name}\n{report}")

    (inspect_out / "combined_report.txt").write_text("\n".join(full_report))
    logger.info(f"Done. All inspection outputs in {inspect_out}/")


if __name__ == "__main__":
    main()
