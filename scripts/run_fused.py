#!/usr/bin/env python3
"""Fused OOD scoring on an already-trained ensemble — no retraining.

Loads every ``model_<i>/`` member written by ``scripts/run_ensemble.py``,
rebuilds the CelebA loaders from member 0's resolved config, fits the
locfre reference stats on in-distribution data (val split when it exists,
otherwise a capped slice of train), and reports per-signal + fused AUROC
(see ``lrad.fusion``: locfre_b1 + locfre_b3 + epistemic + energy).

Usage:
    python scripts/run_fused.py \\
        --output-dir outputs/celeba_ood/ensemble_.../ \\
        [--max-eval-batches 12 --max-ref-batches 8]  # quick CPU estimate

Writes ``<output-dir>/ensemble/fused_auroc.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.dataset import get_celeba_loaders, split_loader  # noqa: E402
from lrad.ensemble import load_ensemble_members  # noqa: E402
from lrad.feature_error import DEFAULT_FEATURE_BLOCKS  # noqa: E402
from lrad.fusion import (  # noqa: E402
    evaluate_fused_ood,
    evaluate_supervised_fusion,
)
from lrad.localized import STD_FLOOR  # noqa: E402
from lrad.plots import plot_fused_auroc_panel  # noqa: E402
from lrad.utils import get_device  # noqa: E402

logger = logging.getLogger("celeba_ood")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fused (locfre + epistemic + energy) OOD AUROC on a "
                    "trained ensemble.",
    )
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Ensemble root holding model_0/ .. model_<M-1>/.")
    ap.add_argument("--blocks", type=int, nargs="+",
                    default=list(DEFAULT_FEATURE_BLOCKS),
                    help="Feature blocks for the locfre signals "
                         "(default: %(default)s).")
    ap.add_argument("--std-floor", type=float, default=STD_FLOOR)
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Fusion weights in signal order (locfre blocks, "
                         "epistemic, energy). Default: uniform.")
    ap.add_argument("--max-ref-batches", type=int, default=40,
                    help="Cap on reference-stat batches (default "
                         "%(default)s).")
    ap.add_argument("--max-eval-batches", type=int, default=None,
                    help="Cap on test batches per split (quick CPU "
                         "estimate). Default: full test.")
    ap.add_argument("--supervised", action="store_true",
                    help="Also fit the supervised logistic fusion: "
                         "calibrates on train negatives + a dedicated "
                         "OOD calibration half (disjoint from the "
                         "evaluated OOD images).")
    ap.add_argument("--ood-cal-frac", type=float, default=0.5,
                    help="Fraction of the OOD pool used for supervised "
                         "calibration (default %(default)s).")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed of the deterministic calibration splits "
                         "(default %(default)s).")
    ap.add_argument("--max-cal-batches", type=int, default=None,
                    help="Cap on calibration batches per side "
                         "(quick CPU estimate). Default: all.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path. Default: "
                         "<output-dir>/ensemble/fused_auroc.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = get_device()
    models, decoders_list, _ = load_ensemble_members(args.output_dir, device)

    with open(args.output_dir / "model_0" / "config.resolved.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["dataset"]["num_workers"] = args.num_workers
    loaders = get_celeba_loaders(cfg)

    if loaders["val"] is not None:
        ref_loader, ref_name = loaders["val"], "val"
    else:
        ref_loader, ref_name = loaders["train"], "train"

    record = {
        "output_dir": str(args.output_dir),
        "blocks": list(args.blocks),
        "std_floor": args.std_floor,
        "n_models": len(models),
    }

    if args.supervised:
        # Calibration splits, all disjoint from what is evaluated:
        #   positives — a dedicated fraction of the OOD pool;
        #   negatives — a same-sized slice of train (never test_in);
        #   locfre reference stats — the REMAINING train slice.
        n_ood_pool = len(loaders["test_ood"].dataset)
        n_cal = int(round(args.ood_cal_frac * n_ood_pool))
        cal_ood_loader, test_ood_loader = split_loader(
            loaders["test_ood"], n_cal, seed=args.split_seed,
        )
        cal_in_loader, ref_loader = split_loader(
            loaders["train"], n_cal, seed=args.split_seed,
        )
        ref_name = "train (minus calibration slice)"
        logger.info(
            "Supervised fusion: %d OOD calibration / %d OOD eval images, "
            "%d train negatives, seed=%d",
            n_cal, n_ood_pool - n_cal, n_cal, args.split_seed,
        )
        out = evaluate_supervised_fusion(
            models, decoders_list, ref_loader,
            cal_in_loader, cal_ood_loader,
            loaders["test_in"], test_ood_loader, device,
            blocks=args.blocks, std_floor=args.std_floor,
            max_ref_batches=args.max_ref_batches,
            max_cal_batches=args.max_cal_batches,
            max_eval_batches=args.max_eval_batches,
        )
        n_in = out["signals_in"][out["signals"][0]].shape[0]
        n_ood = out["signals_ood"][out["signals"][0]].shape[0]
        fused_keys = ("fused_rank", "fused_supervised")
        record.update({
            "supervised": True,
            "ood_cal_frac": args.ood_cal_frac,
            "split_seed": args.split_seed,
            "n_cal": n_cal,
            "calibration": out["calibration"],
            "rank_signals": out["rank_signals"],
        })
    else:
        logger.info(
            "Reference stats from '%s' (max %s batches), locfre blocks %s",
            ref_name, args.max_ref_batches, args.blocks,
        )
        out = evaluate_fused_ood(
            models, decoders_list, ref_loader,
            loaders["test_in"], loaders["test_ood"], device,
            blocks=args.blocks, std_floor=args.std_floor,
            weights=args.weights,
            max_ref_batches=args.max_ref_batches,
            max_eval_batches=args.max_eval_batches,
        )
        n_in = out["fused_in"].shape[0]
        n_ood = out["fused_ood"].shape[0]
        fused_keys = ("fused",)
        record.update({"supervised": False, "weights": out["weights"]})

    logger.info(
        "Evaluated %d in-dist vs %d OOD images (%d ref images)",
        n_in, n_ood, out["n_ref_images"],
    )
    logger.info("OOD AUROC per signal:")
    for k in sorted(out["signals_in"]):
        logger.info(f"  {k:<26} {out['auroc'][k].get('auroc'):.4f}")
    for k in fused_keys:
        logger.info(
            f"  {k.upper():<26} {out['auroc'][k].get('auroc'):.4f}"
        )

    record.update({
        "ref_split": ref_name,
        "n_ref_images": out["n_ref_images"],
        "n_test_in": int(n_in),
        "n_test_ood": int(n_ood),
        "auroc": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
            for k, v in out["auroc"].items()
        },
    })
    out_path = args.out or (
        args.output_dir / "ensemble" / "fused_auroc.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Wrote {out_path}")

    # Visual summary next to the JSON: AUROC bars + ROC + fused histograms.
    plot_dir = out_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if args.supervised:
        fused_in = out["fused_supervised_in"]
        fused_ood = out["fused_supervised_ood"]
        fused_key = "fused_supervised"
    else:
        fused_in, fused_ood, fused_key = (
            out["fused_in"], out["fused_ood"], "fused",
        )
    panel_path = plot_dir / "fused_auroc.png"
    plot_fused_auroc_panel(
        out["auroc"], panel_path,
        fused_in=fused_in, fused_ood=fused_ood, fused_key=fused_key,
        title=f"Fused OOD evaluation — {n_in} in-dist vs {n_ood} OOD  "
              f"({out['n_models']}-model ensemble)",
    )
    logger.info(f"Wrote {panel_path}")


if __name__ == "__main__":
    main()
