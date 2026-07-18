#!/usr/bin/env python3
"""Localized OOD scoring on an already-trained ensemble — no retraining.

Loads every ``model_<i>/`` member written by ``scripts/run_ensemble.py``,
rebuilds the CelebA loaders from member 0's resolved config (same split
seed → identical test_in / test_ood partitions), fits the per-pixel
reference stats on in-distribution data (the val split when it exists,
otherwise a capped slice of train — the run this targets used
``val_ratio=0``), and reports the z-score+patch-max AUROC next to the
global-p95 baseline computed on exactly the same images
(see ``lrad.localized``).

Usage:
    python scripts/run_localized.py \\
        --output-dir outputs/celeba_ood/ensemble_.../ \\
        [--max-eval-batches 8 --max-ref-batches 8]   # quick CPU estimate

Writes ``<output-dir>/ensemble/localized_auroc.json``.
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

from lrad.dataset import get_celeba_loaders  # noqa: E402
from lrad.ensemble import TERMS, load_ensemble_members  # noqa: E402
from lrad.localized import (  # noqa: E402
    DEFAULT_PATCH_SIZES,
    STD_FLOOR,
    evaluate_localized_ood,
)
from lrad.utils import get_device  # noqa: E402

logger = logging.getLogger("celeba_ood")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Localized (z-score + patch-max) OOD AUROC on a "
                    "trained ensemble.",
    )
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Ensemble root holding model_0/ .. model_<M-1>/ "
                         "(same tree scripts/run_ensemble.py writes).")
    ap.add_argument("--patch-sizes", type=int, nargs="+",
                    default=list(DEFAULT_PATCH_SIZES),
                    help="Window sizes (pixels) for the patch-max "
                         "reduction (default: %(default)s).")
    ap.add_argument("--std-floor", type=float, default=STD_FLOOR,
                    help="Per-pixel reference-std floor "
                         "(default: %(default)s).")
    ap.add_argument("--max-ref-batches", type=int, default=40,
                    help="Cap on reference-stat batches (default: "
                         "%(default)s ≈ 10k images at batch 256; per-pixel "
                         "stats stabilize well before that).")
    ap.add_argument("--max-eval-batches", type=int, default=None,
                    help="Cap on test batches per split — for a quick "
                         "subsampled estimate on CPU. Default: full test.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override the training-config batch size.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override the training-config loader workers.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path. Default: "
                         "<output-dir>/ensemble/localized_auroc.json")
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

    # Reference stats need in-dist data disjoint from test_in/test_ood.
    # The val split is ideal; the 20260707 run trained with val_ratio=0,
    # so fall back to a capped slice of train (models saw it, which is the
    # standard PaDiM-style compromise — still zero OOD information).
    if loaders["val"] is not None:
        ref_loader, ref_name = loaders["val"], "val"
    else:
        ref_loader, ref_name = loaders["train"], "train"
    logger.info(
        "Reference stats from '%s' (max %s batches), patch sizes %s",
        ref_name, args.max_ref_batches, args.patch_sizes,
    )

    out = evaluate_localized_ood(
        models, decoders_list, ref_loader,
        loaders["test_in"], loaders["test_ood"], device,
        patch_sizes=args.patch_sizes, std_floor=args.std_floor,
        max_ref_batches=args.max_ref_batches,
        max_eval_batches=args.max_eval_batches,
    )

    n_in = out["scores_in"]["aggregated"]["bias"].shape[0]
    n_ood = out["scores_ood"]["aggregated"]["bias"].shape[0]
    logger.info(
        "Evaluated %d in-dist vs %d OOD images "
        "(%d ref images for the stats)", n_in, n_ood, out["n_ref_images"],
    )
    logger.info("OOD AUROC — global-p95 baseline vs z-score+patch-max:")
    for t in TERMS:
        base = out["auroc"][f"baseline_p95_{t}_aggregated"].get("auroc")
        new = out["auroc"][f"zscore_{t}_aggregated"].get("auroc")
        logger.info(
            f"  {t:<9} baseline={base:.4f}   localized={new:.4f}   "
            f"(Δ {new - base:+.4f})"
        )
    for t in TERMS:
        blocks = "  ".join(
            f"L{k}={a:.4f}"
            for k, a in zip(out["blocks"], out["per_block_auroc"][t])
        )
        logger.info(f"  per-block {t:<9} {blocks}")

    record = {
        "output_dir": str(args.output_dir),
        "ref_split": ref_name,
        "n_ref_images": out["n_ref_images"],
        "n_test_in": int(n_in),
        "n_test_ood": int(n_ood),
        "patch_sizes": out["patch_sizes"],
        "std_floor": out["std_floor"],
        "blocks": out["blocks"],
        "n_models": out["n_models"],
        "auroc": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
            for k, v in out["auroc"].items()
        },
        "per_block_auroc": out["per_block_auroc"],
        "anomaly_auroc": out["anomaly_auroc"],
    }
    out_path = args.out or (
        args.output_dir / "ensemble" / "localized_auroc.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
