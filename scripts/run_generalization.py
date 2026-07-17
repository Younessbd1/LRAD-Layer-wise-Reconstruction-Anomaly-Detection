#!/usr/bin/env python3
"""Generalization probe — feed the trained ensemble arbitrary photos (not
CelebA), with and without eyeglasses, and render the bias overlay exactly
like the top-OOD-glasses figure (see ``lrad.plots.plot_top_ood_glasses``).

This does not touch the CelebA test split or retrain anything: it is a
sanity check that the reconstruction bias — the signal the whole OOD score
is built on — still lights up on the eye region for glasses photos taken
outside the CelebA distribution altogether, and stays quiet for clean ones.

Loads every ``model_<i>/`` member written by ``scripts/run_ensemble.py``
(its own resolved architecture + weights), reconstructs the given photos at
one block depth with the whole ensemble, and writes:

  * ``<out>.png``  — Original | Bias | Overlay columns, one per photo
                     (``lrad.plots.plot_top_ood_glasses``, labelled by
                     group instead of rank)
  * ``<out>.json`` — per-image eye-region bias score
                     (``lrad.ensemble.collect_eye_region_bias``)

Usage:
    python scripts/run_generalization.py \\
        --output-dir outputs/celeba_ood/ensemble_.../ \\
        --glasses-dir my_photos/glasses \\
        --no-glasses-dir my_photos/clean
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.dataset import load_generalization_images  # noqa: E402
from lrad.ensemble import (  # noqa: E402
    collect_eye_region_bias,
    load_ensemble_members,
    sample_block_recons,
)
from lrad.plots import plot_top_ood_glasses  # noqa: E402
from lrad.utils import get_device  # noqa: E402

logger = logging.getLogger("celeba_ood")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(IMG_EXTS)}) in {d}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe ensemble generalization on custom photos, with "
                    "and without eyeglasses.",
    )
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Ensemble root holding model_0/ .. model_<M-1>/ "
                         "(same tree scripts/run_ensemble.py writes).")
    ap.add_argument("--glasses-dir", type=Path, default=None,
                    help="Folder of photos WITH eyeglasses.")
    ap.add_argument("--no-glasses-dir", type=Path, default=None,
                    help="Folder of photos WITHOUT eyeglasses.")
    ap.add_argument("--block", type=int, default=None,
                    help="Conv block to reconstruct at. Default: the "
                         "deepest block (same default run_ensemble.py uses "
                         "for its instance figures).")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="Gaussian blur sigma for the bias overlay.")
    ap.add_argument("--overlay-power", type=float, default=0.8,
                    help="Alpha falloff for the bias overlay.")
    ap.add_argument("--out", type=Path, default=None,
                    help="PNG/JSON basename. Default: "
                         "<output-dir>/generalization.png")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.glasses_dir is None and args.no_glasses_dir is None:
        raise SystemExit(
            "give at least one of --glasses-dir / --no-glasses-dir"
        )

    device = get_device()
    models, decoders_list, image_size = load_ensemble_members(
        args.output_dir, device,
    )
    n_blocks = len(models[0].channels)
    block = n_blocks - 1 if args.block is None else args.block
    if not 0 <= block < n_blocks:
        raise ValueError(
            f"--block must be in [0, {n_blocks - 1}], got {block}"
        )

    groups: list[tuple[str, Path]] = []
    if args.no_glasses_dir is not None:
        groups += [("Clean", p) for p in _list_images(args.no_glasses_dir)]
    if args.glasses_dir is not None:
        groups += [("Glasses", p) for p in _list_images(args.glasses_dir)]

    paths = [p for _, p in groups]
    images = load_generalization_images(paths, image_size=image_size)

    counters: dict[str, int] = {}
    col_labels = []
    for label, _ in groups:
        counters[label] = counters.get(label, 0) + 1
        col_labels.append(f"{label} {counters[label]}")

    logger.info(
        f"Probing {len(groups)} custom photos "
        f"({counters.get('Clean', 0)} clean, {counters.get('Glasses', 0)} "
        f"glasses) through the {len(models)}-model ensemble at block "
        f"L{block}, image_size={image_size}"
    )

    images_dev, recons = sample_block_recons(
        models, decoders_list, images, device, block,
    )
    eye_scores = collect_eye_region_bias(
        models, decoders_list, [(images,)], device, block,
    )

    out_path = args.out or (args.output_dir / "generalization.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_top_ood_glasses(
        images_dev, recons, out_path,
        sigma=args.sigma, overlay_power=args.overlay_power,
        col_labels=col_labels,
        title=f"Generalization probe — bias overlay on custom photos  "
              f"(block L{block}, {len(models)}-model ensemble)",
    )

    record = [
        {
            "label": col_labels[i],
            "group": groups[i][0],
            "file": str(groups[i][1]),
            "global_mean_bias": float(eye_scores["global_mean"][i]),
            "eye_mean_bias": float(eye_scores["eye_mean"][i]),
            "score": float(eye_scores["score"][i]),
        }
        for i in range(len(groups))
    ]
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    for group_name in ("Clean", "Glasses"):
        scores = [r["score"] for r in record if r["group"] == group_name]
        if scores:
            logger.info(
                f"{group_name:>7}: n={len(scores)}  "
                f"mean eye-region score={sum(scores) / len(scores):.4f}  "
                f"range=[{min(scores):.4f}, {max(scores):.4f}]"
            )

    logger.info(f"Wrote {out_path} and {json_path}")


if __name__ == "__main__":
    main()
