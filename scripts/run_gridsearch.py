#!/usr/bin/env python3
"""Grid search over the CutPaste hyperparameters — multi-metric.

Each grid point trains ONE short-schedule model (classifier + per-block
decoders, 64 px) and is scored on the FULL scoring stack, not just the
pretext head:

  * ``auroc_cutpaste``  — P(altered | x) from the pretext head
  * ``auroc_bias_p95``  — reconstruction bias, global p95 (the project's
                          historical anomaly score; single model, so
                          bias == that model's squared recon error)
  * ``auroc_locfre_b3`` — localized feature error at block 3 (the best
                          single signal of the fused stack)
  * ``auroc_energy``    — gender-head energy
  * ``auroc_fused``     — rank fusion of the four signals above
                          (the SELECTION metric: best config = best fused)

A per-epoch cutpaste AUROC curve is also recorded from the epoch
checkpoints, so the run tells you how many classifier epochs the pretext
head actually needs.

Grid: scar_prob x area_range x prob x loss_weight, with the redundant
combinations pruned (scar-only configs ignore area_range).

Writes to --output-dir:
  * ``gridsearch_results.json``  — every config + all metrics, sorted by
                                   the fused AUROC
  * ``gridsearch_auroc.png``     — grouped bars, every metric per config
  * ``gridsearch_epochs.png``    — per-epoch cutpaste AUROC curves
and logs the winning config as a ready-to-paste YAML block.

Usage (local smoke test):
    python scripts/run_gridsearch.py --output-dir /tmp/grid \\
        --epochs 1 --decoder-epochs 1 --max-train-batches 4 \\
        --max-eval-batches 2
Cluster: scripts/oar_run_gridsearch.sh (full grid, ~20 min/config).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from lrad.dataset import get_celeba_loaders  # noqa: E402
from lrad.decoder import build_decoders  # noqa: E402
from lrad.ensemble import collect_decomposition_scores  # noqa: E402
from lrad.evaluate import _auroc_entry  # noqa: E402
from lrad.feature_error import fit_feature_error_stats  # noqa: E402
from lrad.fusion import collect_fusion_signals, rank_fusion  # noqa: E402
from lrad.model import build_model  # noqa: E402
from lrad.train import train_decoders, train_model  # noqa: E402
from lrad.utils import get_device, seed_everything  # noqa: E402

logger = logging.getLogger("celeba_ood")

GRID = {
    "scar_prob": [0.0, 0.5, 1.0],
    "area_range": [(0.02, 0.08), (0.05, 0.15)],
    "prob": [0.3, 0.5],
    "loss_weight": [0.5, 1.0, 2.0],
}
LOCFRE_BLOCKS = (1, 3)
# Signals entering the selection metric (rank fusion). Epistemic is absent
# on purpose: with a single model it is identically zero.
FUSED_SIGNALS = ("cutpaste", "bias_p95", "locfre_b3", "energy")


def _combos() -> list[dict]:
    """Grid points with the redundant ones pruned: a scar-only config
    (scar_prob == 1.0) never reads area_range, so only keep one."""
    out = []
    for values in itertools.product(*GRID.values()):
        combo = dict(zip(GRID.keys(), values))
        if (combo["scar_prob"] == 1.0
                and combo["area_range"] != GRID["area_range"][0]):
            continue
        out.append(combo)
    return out


class _CappedLoader:
    """Wrap a DataLoader, yielding at most ``max_batches`` batches."""

    def __init__(self, loader, max_batches: int | None):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            yield batch


@torch.no_grad()
def _cutpaste_scores(model, loader, device, max_batches) -> np.ndarray:
    model.eval()
    chunks = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        img = batch[0].to(device, non_blocking=True)
        out = model(img)
        chunks.append(
            F.softmax(out["cutpaste_logits"], dim=-1)[:, 1].cpu().numpy()
        )
    return np.concatenate(chunks)


def _auroc(s_in: np.ndarray, s_ood: np.ndarray) -> float:
    return _auroc_entry(s_in, s_ood).get("auroc", float("nan"))


def _evaluate_config(model, decoders, loaders, device, args) -> dict:
    """All per-config metrics on capped test splits, single-member stack."""
    mb = args.max_eval_batches
    sig = {}

    # locfre + cutpaste + energy in one pass (M=1 ensemble).
    ref = fit_feature_error_stats(
        [model], [decoders], _CappedLoader(loaders["train"], args.max_ref_batches),
        device, LOCFRE_BLOCKS,
    )
    f_in = collect_fusion_signals(
        [model], [decoders], loaders["test_in"], device, ref,
        LOCFRE_BLOCKS, mb,
    )
    f_ood = collect_fusion_signals(
        [model], [decoders], loaders["test_ood"], device, ref,
        LOCFRE_BLOCKS, mb,
    )
    sig["cutpaste"] = (f_in["cutpaste_prob"], f_ood["cutpaste_prob"])
    sig["locfre_b3"] = (f_in["locfre_b3"], f_ood["locfre_b3"])
    sig["energy"] = (f_in["ens_energy_gender"], f_ood["ens_energy_gender"])

    # reconstruction bias, global p95 (historical anomaly score).
    d_in = collect_decomposition_scores(
        [model], [decoders], _CappedLoader(loaders["test_in"], mb),
        device, "p95",
    )
    d_ood = collect_decomposition_scores(
        [model], [decoders], _CappedLoader(loaders["test_ood"], mb),
        device, "p95",
    )
    sig["bias_p95"] = (
        d_in["aggregated"]["bias"], d_ood["aggregated"]["bias"],
    )

    metrics = {f"auroc_{k}": _auroc(*sig[k]) for k in FUSED_SIGNALS}
    n_in = len(sig[FUSED_SIGNALS[0]][0])
    fused = rank_fusion([
        np.concatenate([sig[k][0], sig[k][1]]) for k in FUSED_SIGNALS
    ])
    metrics["auroc_fused"] = _auroc(fused[:n_in], fused[n_in:])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-metric grid search over CutPaste hyperparams.",
    )
    ap.add_argument("--config", type=Path,
                    default=_ROOT / "configs" / "celeba_ood.yaml")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6,
                    help="Classifier epochs per config.")
    ap.add_argument("--decoder-epochs", type=int, default=8,
                    help="Decoder epochs per config (full runs use 25; 8 "
                         "is enough to RANK configs).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="Cap train batches per epoch (smoke tests only).")
    ap.add_argument("--max-eval-batches", type=int, default=40,
                    help="Cap on test batches per split for the AUROCs.")
    ap.add_argument("--max-ref-batches", type=int, default=8,
                    help="Cap on locfre reference-stat batches.")
    ap.add_argument("--per-epoch-eval-batches", type=int, default=10,
                    help="Caps for the per-epoch cutpaste AUROC curve.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["cutpaste_head"] = True
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["dataset"]["num_workers"] = args.num_workers

    device = get_device()
    loaders = get_celeba_loaders(cfg)
    train_loader = _CappedLoader(loaders["train"], args.max_train_batches)

    combos = _combos()
    logger.info(
        "Grid search: %d configs x (%d cls + %d dec epochs) on %s",
        len(combos), args.epochs, args.decoder_epochs, device,
    )

    results = []
    epoch_curves = {}
    for i, combo in enumerate(combos):
        tag = (f"scar={combo['scar_prob']:g} area={combo['area_range']} "
               f"p={combo['prob']:g} w={combo['loss_weight']:g}")
        logger.info("=" * 64)
        logger.info("[%d/%d] %s", i + 1, len(combos), tag)
        t0 = time.time()
        seed_everything(args.seed)
        model = build_model(cfg).to(device)
        ckpt_dir = args.output_dir / f"_ckpt_{i:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cp_cfg = {k: v for k, v in combo.items() if k != "loss_weight"}
        history = train_model(
            model, train_loader, None,
            epochs=args.epochs,
            lr=cfg["training"].get("lr", 3e-4),
            attr_loss_weight=cfg["training"].get("attr_loss_weight", 1.0),
            device=device,
            log_every=max(args.epochs // 2, 1),
            checkpoint_dir=ckpt_dir,
            cutpaste={**cp_cfg, "loss_weight": combo["loss_weight"]},
        )

        # Per-epoch pretext AUROC from the epoch checkpoints (small caps —
        # a curve to pick the epoch budget, not a precise measurement).
        curve = []
        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        for e in range(1, args.epochs + 1):
            ck = ckpt_dir / f"model_ep{e}.pt"
            if not ck.exists():
                continue
            model.load_state_dict(
                torch.load(ck, map_location=device, weights_only=True),
            )
            curve.append(_auroc(
                _cutpaste_scores(model, loaders["test_in"], device,
                                 args.per_epoch_eval_batches),
                _cutpaste_scores(model, loaders["test_ood"], device,
                                 args.per_epoch_eval_batches),
            ))
            ck.unlink()  # checkpoints are per-config scratch, not results
        ckpt_dir.rmdir()
        model.load_state_dict(final_state)
        epoch_curves[tag] = curve

        decoders = build_decoders(
            model, image_size=cfg["dataset"].get("image_size", 64),
        ).to(device)
        train_decoders(
            model, decoders, train_loader, None,
            epochs=args.decoder_epochs,
            lr=cfg["training"].get("decoders", {}).get("lr", 1e-3),
            device=device,
            log_every=max(args.decoder_epochs, 1),
        )

        metrics = _evaluate_config(model, decoders, loaders, device, args)
        logger.info(
            "-> fused=%.4f  cutpaste=%.4f  bias_p95=%.4f  locfre_b3=%.4f  "
            "energy=%.4f  gender_acc=%.3f  (%.0fs)",
            metrics["auroc_fused"], metrics["auroc_cutpaste"],
            metrics["auroc_bias_p95"], metrics["auroc_locfre_b3"],
            metrics["auroc_energy"], history["train_gender_acc"][-1],
            time.time() - t0,
        )
        results.append({
            **{k: list(v) if isinstance(v, tuple) else v
               for k, v in combo.items()},
            **metrics,
            "cutpaste_auroc_per_epoch": curve,
            "gender_acc": history["train_gender_acc"][-1],
        })
        del model, decoders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda r: -(r["auroc_fused"]
                                 if np.isfinite(r["auroc_fused"])
                                 else -1.0))
    out_json = args.output_dir / "gridsearch_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "decoder_epochs": args.decoder_epochs,
            "seed": args.seed,
            "selection_metric": "auroc_fused",
            "results": results,
        }, f, indent=2)

    # --- grouped bar plot: every metric, every config ---------------------
    metric_cols = ["auroc_fused"] + [f"auroc_{k}" for k in FUSED_SIGNALS]
    colors = ["#D55E00", "#0072B2", "#56B4E9", "#009E73", "#E69F00"]
    labels = [
        f"scar={r['scar_prob']:g} area={tuple(r['area_range'])} "
        f"p={r['prob']:g} w={r['loss_weight']:g}"
        for r in results
    ]
    ys = np.arange(len(results))[::-1]
    h = 0.8 / len(metric_cols)
    fig, ax = plt.subplots(
        figsize=(8.4, 0.72 * len(results) + 1.8), layout="constrained",
    )
    for j, (mc, c) in enumerate(zip(metric_cols, colors)):
        vals = [r[mc] for r in results]
        ax.barh(ys + 0.4 - (j + 0.5) * h, vals, height=h, color=c,
                label=mc.replace("auroc_", ""))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0.5, color="#888888", lw=0.8, ls="--")
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("OOD AUROC")
    ax.set_title("CutPaste grid search — every metric per config "
                 "(sorted by fused)")
    ax.legend(fontsize=8, ncols=len(metric_cols), loc="lower right")
    fig.savefig(args.output_dir / "gridsearch_auroc.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- per-epoch curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    for tag, curve in epoch_curves.items():
        if curve:
            ax.plot(range(1, len(curve) + 1), curve, lw=1.0, alpha=0.65)
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
    ax.set_xlabel("classifier epoch")
    ax.set_ylabel("cutpaste-head OOD AUROC")
    ax.set_title(
        f"Pretext AUROC vs epochs — one line per config "
        f"({len(epoch_curves)} configs)",
    )
    fig.savefig(args.output_dir / "gridsearch_epochs.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    best = results[0]
    logger.info("=" * 64)
    logger.info(
        "BEST (by fused AUROC=%.4f): paste into training.cutpaste:",
        best["auroc_fused"],
    )
    logger.info(
        "  cutpaste:\n    prob: %g\n    area_range: [%g, %g]\n"
        "    aspect_range: [0.3, 3.3]\n    scar_prob: %g\n"
        "    loss_weight: %g",
        best["prob"], *best["area_range"], best["scar_prob"],
        best["loss_weight"],
    )
    logger.info("Wrote %s, gridsearch_auroc.png, gridsearch_epochs.png",
                out_json)


if __name__ == "__main__":
    main()
