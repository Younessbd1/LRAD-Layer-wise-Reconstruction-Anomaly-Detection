#!/usr/bin/env python3
"""MVTec AD runner — per-category LRAD ensembles, benchmarked against PatchCore.

Mirrors the PatchCore protocol so the numbers are directly comparable: one
independent model per category, trained on that category's nominal images
only, evaluated on its own test split, then averaged over categories.

For each category in ``dataset.categories``:

  1. Train ``ensemble.size`` members. Each member is a conv trunk trained
     **purely self-supervised** on the 3-way CutPaste pretext task (MVTec
     has no labels), then frozen, then fitted with one ``BlockDecoder`` per
     conv block. Members differ by seed and by architecture
     (``ensemble.member_variants``), as on CelebA.
  2. Decompose the per-block reconstruction risk into Bias + Variance and
     score image-level detection AUROC from the Bias term.
  3. Score localization — pixel AUROC and PRO — against MVTec's
     ground-truth masks (``lrad.pixel_metrics``).

Then write a ``summary.json`` plus a ``results.md`` table with the
per-category and mean scores next to PatchCore's published values.

Usage:
    python scripts/run_mvtec.py --config configs/mvtec.yaml
    python scripts/run_mvtec.py --config configs/mvtec.yaml --categories all
    python scripts/run_mvtec.py --config configs/mvtec.yaml \\
        --categories bottle screw --output-dir outputs/mvtec/pilot
    python scripts/run_mvtec.py --config configs/mvtec.yaml \\
        --output-dir <dir> --eval-only     # re-score saved members
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.arch_diagram import resolve_member_configs  # noqa: E402
from lrad.config import apply_overrides, load_config, to_jsonable  # noqa: E402
from lrad.decoder import build_decoders  # noqa: E402
from lrad.ensemble import TERMS, evaluate_ensemble_decomposition  # noqa: E402
from lrad.model import build_model, count_parameters  # noqa: E402
from lrad.mvtec import MVTEC_CATEGORIES, get_mvtec_loaders  # noqa: E402
from lrad.pixel_metrics import evaluate_pixel_metrics  # noqa: E402
from lrad.plots import (  # noqa: E402
    plot_architecture_effect,
    plot_batch_loss,
    plot_decoder_history,
    plot_decomposition_auroc_bars,
    plot_ensemble_score_hists,
    plot_mvtec_curves,
    plot_mvtec_segmentation,
    plot_mvtec_vs_patchcore,
)
from lrad.train import train_decoders, train_model  # noqa: E402
from lrad.utils import get_device, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger("celeba_ood")

# PatchCore-10% per-category scores, from Tables S1/S2/S3 of "Towards Total
# Recall in Industrial Anomaly Detection" (Roth et al., CVPR 2022), as
# (image AUROC, pixel AUROC, PRO) in percent. Printed beside our own
# numbers so a run is self-contained — no flipping back to the paper to
# find out whether a result is good.
PATCHCORE_10: dict[str, tuple[float, float, float]] = {
    "bottle":     (100.0, 98.6, 96.1),
    "cable":      (99.4, 98.5, 92.6),
    "capsule":    (97.8, 98.9, 95.5),
    "carpet":     (98.7, 99.1, 96.6),
    "grid":       (97.9, 98.7, 95.9),
    "hazelnut":   (100.0, 98.7, 93.9),
    "leather":    (100.0, 99.3, 98.9),
    "metal_nut":  (100.0, 98.4, 91.3),
    "pill":       (96.0, 97.6, 94.1),
    "screw":      (97.0, 99.4, 87.4),
    "tile":       (98.9, 98.7, 91.4),
    "toothbrush": (100.0, 98.7, 83.5),
    "transistor": (100.0, 96.4, 83.5),
    "wood":       (99.0, 95.1, 89.6),
    "zipper":     (99.5, 98.9, 97.1),
}
PATCHCORE_10_MEAN = (99.0, 98.1, 93.5)


def _epochs_for(steps, n_batches: int, fallback: int) -> int:
    """Convert a target number of optimizer steps into an epoch count.

    **Epochs are the wrong unit on MVTec.** A category holds 60
    (toothbrush) to 391 (hazelnut) training images, so at batch 32 a fixed
    epoch count buys 2 to 13 optimizer steps per epoch — a 6.5x spread in
    how much training each category actually receives, which alone would
    make the per-category table incomparable. Worse, the absolute counts
    are tiny: 60 epochs on bottle is 420 steps, against ~31,650 for the
    CelebA decoder run. At that scale the decoders never individuate; they
    converge to the dataset mean and the "anomaly map" degenerates into
    |x − mean image|, which correlates ~0.85 with plain image brightness
    and scores BELOW 0.5 pixel AUROC because MVTec defects are often
    darker — and therefore easier to reconstruct — than nominal texture.

    Setting ``training.steps`` / ``training.decoders.steps`` fixes the
    step budget instead, so every category is trained equally. ``epochs``
    stays as the fallback when no step budget is given.
    """
    if not steps:
        return int(fallback)
    return max(1, -(-int(steps) // max(1, n_batches)))   # ceil division


def _train_member(
    mcfg: dict,
    loaders: dict,
    device: torch.device,
    model_dir: Path,
    *,
    seed: int,
    eval_only: bool,
    no_plots: bool = False,
):
    """Train (or load) one ensemble member: trunk + per-block decoders."""
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "weights"
    weights.mkdir(exist_ok=True)

    seed_everything(seed)
    model = build_model(mcfg).to(device)
    image_size = mcfg["dataset"].get("crop_size") or mcfg["dataset"]["image_size"]
    decoders = build_decoders(model, image_size=image_size).to(device)

    if eval_only:
        model.load_state_dict(
            torch.load(weights / "model.pt", map_location=device)
        )
        decoders.load_state_dict(
            torch.load(weights / "decoders.pt", map_location=device)
        )
        return model.eval(), decoders.eval(), {}

    with open(model_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(mcfg, f, sort_keys=False)

    tcfg = mcfg.get("training", {})
    n_batches = len(loaders["train"])
    clf_epochs = _epochs_for(tcfg.get("steps"), n_batches,
                             tcfg.get("epochs", 40))
    dcfg = tcfg.get("decoders", {})
    dec_epochs = _epochs_for(dcfg.get("steps"), n_batches,
                             dcfg.get("epochs", 60))
    logger.info(
        "schedule: %d batches/epoch -> classifier %d epochs (%d steps), "
        "decoders %d epochs (%d steps)",
        n_batches, clf_epochs, clf_epochs * n_batches,
        dec_epochs, dec_epochs * n_batches,
    )

    hist = train_model(
        model, loaders["train"], loaders.get("val"),
        epochs=clf_epochs,
        lr=float(tcfg.get("lr", 3e-4)),
        attr_loss_weight=float(tcfg.get("attr_loss_weight", 0.0)),
        device=device,
        log_every=int(tcfg.get("log_every", 5)),
        cutpaste=tcfg.get("cutpaste"),
    )
    torch.save(model.state_dict(), weights / "model.pt")

    dhist = train_decoders(
        model, decoders, loaders["train"], loaders.get("val"),
        epochs=dec_epochs,
        lr=float(dcfg.get("lr", 1e-3)),
        device=device,
        log_every=int(dcfg.get("log_every", 10)),
    )
    torch.save(decoders.state_dict(), weights / "decoders.pt")

    history = {"classifier": hist, "decoders": dhist}
    with open(model_dir / "history.json", "w") as f:
        json.dump(to_jsonable(history), f, indent=2)

    if not no_plots:
        plots = model_dir / "plots"
        plots.mkdir(exist_ok=True)
        # Batch-level loss and per-block decoder loss are the two curves
        # that say whether a member trained at all — the only training
        # diagnostics available here, since a pretext-only trunk has no
        # supervised accuracy to watch.
        plot_batch_loss(hist, plots / "batch_loss.png")
        plot_decoder_history(dhist, plots / "decoder_loss.png")

    return model.eval(), decoders.eval(), history


def _plot_category(
    result: dict,
    decomp: dict,
    pix: dict | None,
    loaders: dict,
    plots: Path,
    *,
    n_instances: int,
    fpr_limit: float,
    overlay_sigma: float,
) -> None:
    """All per-category figures. Never fatal — a plotting failure must not
    throw away a category's numbers, which are already computed by here."""
    cat = result["category"]

    plot_decomposition_auroc_bars(
        decomp["per_block_auroc"], plots / "auroc_per_block.png",
        aggregated={t: decomp["auroc"][f"score_{t}_aggregated"].get("auroc")
                    for t in TERMS},
        title=f"{cat} — per-block detection AUROC",
    )
    plot_ensemble_score_hists(
        decomp["scores_in"]["aggregated"], decomp["scores_ood"]["aggregated"],
        plots / "score_hists.png",
        auroc={t: decomp["auroc"][f"score_{t}_aggregated"].get("auroc")
               for t in TERMS},
        title=f"{cat} — nominal vs defective score distributions",
    )
    # Needs >2 members to say anything: the panel regresses AUROC on
    # parameter count and width, and its Spearman term is undefined below
    # three points.
    if len(result["member_auroc"]) > 2:
        plot_architecture_effect(
            [{
                "member": i + 1,
                "channels": v["channels"],
                "kernel_size": v["kernel_size"],
                "n_params": p,
                "auroc": m["aggregated"],
                "auroc_per_block": m["per_block"],
            } for i, (v, m, p) in enumerate(zip(
                result["member_variants"], result["member_auroc"],
                result["member_params"],
            ))],
            plots / "architecture_effect.png",
            title=f"{cat} — architecture vs detection AUROC",
        )

    plot_mvtec_curves(
        plots / "curves.png",
        roc=decomp["auroc"]["score_bias_aggregated"],
        pro_curve=pix["pro_curve"] if pix else None,
        pixel_auroc=pix["pixel_auroc"] if pix else None,
        pro=pix["pro"] if pix else None,
        image_auroc=result["image_auroc"],
        fpr_limit=fpr_limit,
        title=f"{cat} — detection and localization",
    )

    if pix is None or "maps_ood" not in pix:
        return

    # Qualitative panels. Rank the defective test images by their own
    # anomaly score and show BOTH ends: the top detections and the
    # lowest-scoring ones. Showing only the winners would make a category
    # look solved when its failures are exactly what the PRO number is
    # reacting to.
    maps = pix["maps_ood"]
    masks = pix["masks_ood"]
    ds = loaders["test_ood_ds"]
    scores = np.asarray(decomp["scores_ood"]["aggregated"]["bias"])
    if scores.shape[0] != maps.shape[0]:
        logger.warning(
            "%s: %d scores vs %d maps — skipping instance panels",
            cat, scores.shape[0], maps.shape[0],
        )
        return

    order = np.argsort(scores)[::-1]
    for name, idx in (
        ("instances_best.png", order[:n_instances]),
        ("instances_worst.png", order[-n_instances:][::-1]),
    ):
        if len(idx) == 0:
            continue
        imgs = torch.stack([ds[int(i)][0] for i in idx])
        plot_mvtec_segmentation(
            imgs, maps[idx], masks[idx], plots / name,
            scores=[float(scores[i]) for i in idx],
            labels=[f"{ds.defects[int(i)]}\n{ds.paths[int(i)].stem}"
                    for i in idx],
            overlay_sigma=overlay_sigma,
            title=(f"{cat} — "
                   + ("highest" if "best" in name else "lowest")
                   + " scoring defective images"),
        )


def run_category(
    cfg: dict,
    category: str,
    device: torch.device,
    out_dir: Path,
    *,
    eval_only: bool,
    no_plots: bool = False,
) -> dict:
    """Train and evaluate one category's ensemble. Returns its result dict."""
    ccfg = copy.deepcopy(cfg)
    ccfg["dataset"]["category"] = category
    out_dir.mkdir(parents=True, exist_ok=True)

    ecfg = ccfg.get("ensemble", {})
    size = int(ecfg.get("size", 4))
    base_seed = int(ecfg.get("base_seed", ccfg["experiment"].get("seed", 42)))
    agg = str(ecfg.get("agg", "p95"))

    loaders = get_mvtec_loaders(ccfg)
    member_cfgs = resolve_member_configs(ccfg, size)

    models, decoders_list, member_params = [], [], []
    t0 = time.time()
    for i in range(size):
        seed = base_seed + i
        mv = member_cfgs[i]["model"]
        logger.info(
            "--- %s: member %d/%d (seed=%d, channels=%s, k=%d) ---",
            category, i + 1, size, seed, mv.get("channels"),
            mv.get("kernel_size", 3),
        )
        model, decoders, _ = _train_member(
            member_cfgs[i], loaders, device, out_dir / f"model_{i}",
            seed=seed, eval_only=eval_only, no_plots=no_plots,
        )
        member_params.append(count_parameters(model))
        # Park finished members on CPU so GPU memory stays free while the
        # rest train; they come back as a group for the decomposition.
        models.append(model.to("cpu").eval())
        decoders_list.append(decoders.to("cpu").eval())
    logger.info("%s: %d members ready in %.1fs", category, size,
                time.time() - t0)

    models = [m.to(device).eval() for m in models]
    decoders_list = [d.to(device).eval() for d in decoders_list]

    decomp = evaluate_ensemble_decomposition(
        models, decoders_list, loaders, device, agg=agg,
    )
    image_auroc = decomp["auroc"]["score_bias_aggregated"].get("auroc")
    logger.info(
        "%s: image AUROC (%s)  Risk=%.4f  Bias=%.4f  Variance=%.4f",
        category, agg,
        *(decomp["auroc"][f"score_{t}_aggregated"].get("auroc", float("nan"))
          for t in TERMS),
    )

    result = {
        "category": category,
        "n_members": size,
        "member_params": member_params,
        "member_variants": [
            {"channels": list(mc["model"]["channels"]),
             "kernel_size": int(mc["model"].get("kernel_size", 3))}
            for mc in member_cfgs
        ],
        "agg": agg,
        "image_auroc": image_auroc,
        "image_auroc_per_term": {
            t: decomp["auroc"][f"score_{t}_aggregated"].get("auroc")
            for t in TERMS
        },
        "per_block_auroc": decomp["per_block_auroc"],
        "member_auroc": decomp["member_auroc"],
        "n_test_in": len(loaders["test_in_ds"]),
        "n_test_ood": len(loaders["test_ood_ds"]),
        "defect_types": loaders["defect_types"],
    }

    evcfg = ccfg.get("evaluation", {})
    fpr_limit = float(evcfg.get("pro_fpr_limit", 0.30))
    pix = None
    if evcfg.get("pixel_metrics", True):
        pix = evaluate_pixel_metrics(
            models, decoders_list, loaders, device,
            smooth_sigma=float(evcfg.get("smooth_sigma", 4.0)),
            fpr_limit=fpr_limit,
            keep_maps=not no_plots,
        )
        result["pixel_auroc"] = pix["pixel_auroc"]
        result["pro"] = pix["pro"]
        # Scalars only — the map arrays and the PRO curve stay out of the
        # summary, which is meant to be read by eye.
        result["pixel_detail"] = {
            k: v for k, v in pix.items()
            if k not in ("pro_curve", "maps_in", "maps_ood", "masks_ood")
        }
        logger.info(
            "%s: pixel AUROC=%.4f  PRO=%.4f  (%d regions)",
            category, pix["pixel_auroc"], pix["pro"], pix["n_regions"],
        )

    if not no_plots:
        plots = out_dir / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        try:
            _plot_category(
                result, decomp, pix, loaders, plots,
                n_instances=int(evcfg.get("n_instances", 6)),
                fpr_limit=fpr_limit,
                overlay_sigma=float(evcfg.get("overlay_sigma", 3.0)),
            )
            logger.info("%s: figures written to %s", category, plots)
        except Exception:
            # The numbers are already computed and about to be written; a
            # matplotlib failure must not cost the category its results.
            logger.exception("%s: plotting failed — numbers kept", category)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(to_jsonable(result), f, indent=2)

    # Free the member weights before the next category allocates its own.
    del models, decoders_list, pix
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _pct(x) -> str:
    return "  n/a" if x is None or x != x else f"{x * 100:.1f}"


def write_results_table(results: list[dict], path: Path) -> str:
    """Markdown table of our scores against PatchCore's, per category."""
    lines = [
        "# LRAD on MVTec AD — versus PatchCore",
        "",
        "Ours = ensemble Bias term. PatchCore = PatchCore-10% as published "
        "(Roth et al., CVPR 2022, Tables S1/S2/S3). All values in percent.",
        "",
        "| Category | Image AUROC | | Pixel AUROC | | PRO | |",
        "|---|---|---|---|---|---|---|",
        "| | ours | PatchCore | ours | PatchCore | ours | PatchCore |",
    ]
    got = {r["category"]: r for r in results}
    sums = {"image_auroc": [], "pixel_auroc": [], "pro": []}
    for cat in MVTEC_CATEGORIES:
        if cat not in got:
            continue
        r = got[cat]
        ref = PATCHCORE_10.get(cat, (float("nan"),) * 3)
        for i, k in enumerate(("image_auroc", "pixel_auroc", "pro")):
            v = r.get(k)
            if v is not None and v == v:
                sums[k].append(v)
        lines.append(
            f"| {cat} | {_pct(r.get('image_auroc'))} | {ref[0]:.1f} "
            f"| {_pct(r.get('pixel_auroc'))} | {ref[1]:.1f} "
            f"| {_pct(r.get('pro'))} | {ref[2]:.1f} |"
        )

    means = {k: (sum(v) / len(v) if v else float("nan"))
             for k, v in sums.items()}
    # The PatchCore mean quoted here is its 15-category mean. When only a
    # subset of categories has been run, our column averages fewer of them,
    # so the two means are NOT directly comparable — say so rather than let
    # the table imply otherwise.
    n_run = len(sums["image_auroc"])
    lines += [
        f"| **mean ({n_run} cat.)** | **{_pct(means['image_auroc'])}** "
        f"| {PATCHCORE_10_MEAN[0]:.1f} | **{_pct(means['pixel_auroc'])}** "
        f"| {PATCHCORE_10_MEAN[1]:.1f} | **{_pct(means['pro'])}** "
        f"| {PATCHCORE_10_MEAN[2]:.1f} |",
        "",
    ]
    if n_run < len(MVTEC_CATEGORIES):
        lines.append(
            f"> Only {n_run} of 15 categories were run. The PatchCore mean "
            f"column is its **15-category** mean, so the mean row compares "
            f"different category sets — read the per-category rows instead."
        )
        lines.append("")
        subset = [PATCHCORE_10[c] for c in got if c in PATCHCORE_10]
        if subset:
            lines.append(
                "> PatchCore on these same categories: image "
                f"{sum(s[0] for s in subset) / len(subset):.1f}, pixel "
                f"{sum(s[1] for s in subset) / len(subset):.1f}, PRO "
                f"{sum(s[2] for s in subset) / len(subset):.1f}."
            )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train per-category LRAD ensembles on MVTec AD and "
                    "compare to PatchCore.",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Categories to run; 'all' for the full 15. "
                         "Default: dataset.categories from the config.")
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; reload saved member weights.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip every figure (numbers and tables only).")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dotted overrides, e.g. training.epochs=10.")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    exp = cfg["experiment"]

    cats = args.categories or cfg["dataset"].get("categories")
    if cats and len(cats) == 1 and cats[0] == "all":
        cats = list(MVTEC_CATEGORIES)
    if not cats:
        raise SystemExit(
            "no categories given — set dataset.categories or pass --categories"
        )
    if args.ensemble_size:
        cfg.setdefault("ensemble", {})["size"] = int(args.ensemble_size)

    output_dir = Path(args.output_dir or exp["output_dir"])
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    seed_everything(int(exp.get("seed", 42)))
    job_id = os.environ.get("OAR_JOB_ID", "")
    setup_logging(
        str(output_dir / "logs"),
        run_tag=f"{exp['name']}{('_oar' + job_id) if job_id else ''}",
    )
    device = get_device()

    logger.info("=" * 70)
    logger.info("MVTec AD experiment : %s", exp["name"])
    logger.info("Config              : %s", args.config)
    logger.info("Output dir          : %s", output_dir)
    logger.info("Device              : %s", device)
    logger.info("Categories (%d)      : %s", len(cats), ", ".join(cats))
    logger.info("Ensemble size       : %s",
                cfg.get("ensemble", {}).get("size", 4))
    logger.info("=" * 70)

    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    results: list[dict] = []
    t_all = time.time()
    for n, cat in enumerate(cats, 1):
        logger.info("#" * 70)
        logger.info("# CATEGORY %d/%d — %s", n, len(cats), cat)
        logger.info("#" * 70)
        t0 = time.time()
        try:
            results.append(run_category(
                cfg, cat, device, output_dir / cat,
                eval_only=args.eval_only, no_plots=args.no_plots,
            ))
        except Exception:
            # One bad category (missing download, OOM) must not throw away
            # the categories already finished — log it, keep going, and let
            # the table show which ones are missing.
            logger.exception("category %s FAILED — continuing", cat)
            continue
        logger.info("category %s done in %.1fs", cat, time.time() - t0)

        # Rewrite the table after every category so a job killed at the
        # walltime still leaves a readable, up-to-date result file.
        write_results_table(results, output_dir / "results.md")
        with open(output_dir / "summary.json", "w") as f:
            json.dump(to_jsonable({
                "experiment": exp["name"],
                "categories": [r["category"] for r in results],
                "results": results,
            }), f, indent=2)

    if not results:
        raise SystemExit("every category failed — see the log above")

    if not args.no_plots and len(results) > 1:
        try:
            plot_mvtec_vs_patchcore(
                results, PATCHCORE_10, output_dir / "vs_patchcore.png",
            )
            logger.info("Wrote %s", output_dir / "vs_patchcore.png")
        except Exception:
            logger.exception("cross-category figure failed — numbers kept")

    logger.info("=" * 70)
    logger.info("ALL CATEGORIES DONE in %.1fs", time.time() - t_all)
    logger.info("\n%s", write_results_table(results, output_dir / "results.md"))
    logger.info("Wrote %s", output_dir / "results.md")


if __name__ == "__main__":
    main()
