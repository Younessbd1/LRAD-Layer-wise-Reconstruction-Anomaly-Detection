#!/usr/bin/env python3
"""Deep-ensemble runner — bias/variance decomposition for the CelebA OOD task.

Implements the LRAD note end to end:

  1. Train ``size`` independent models (classifier + per-block decoders),
     each seeded ``base_seed + i`` — and, when ``ensemble.member_variants``
     is set, each with its OWN architecture (channel widths + conv kernel
     size). Diversity therefore comes from the architecture, the random
     init and the SGD shuffle order — an architecturally diverse deep
     ensemble, not MC-Dropout.
  2. Write the full per-model results into ``model_<i>/`` (the usual
     plots + summary, one set per model — "the results x5").
  3. Form the ensemble-mean reconstruction ``E_D[f_hat]`` and decompose
     the per-block reconstruction risk into Bias + Variance, with the
     exact per-pixel identity ``Risk = Bias + Variance``.
  4. Write the decomposition plots + summary into ``ensemble/``. Scoring
     the *mean* reconstruction isolates the Bias term (item 5 of the
     note); the Variance term doubles as an epistemic-uncertainty OOD
     score. The runner also ranks the OOD eyeglasses faces by eye-region
     bias and writes the top-N to ``plots/top_ood_glasses.png``.

Usage:
    python scripts/run_ensemble.py --config configs/celeba_ood.yaml
    python scripts/run_ensemble.py --config configs/celeba_ood.yaml \\
        --output-dir outputs/celeba_ood/ensemble_run --ensemble-size 5
    python scripts/run_ensemble.py --config configs/celeba_ood.yaml \\
        --output-dir <dir> --eval-only      # re-decompose saved models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_celeba import (  # noqa: E402  (path set up just above)
    _gather_samples,
    _to_jsonable,
    apply_overrides,
    build_and_run,
    load_config,
)

from lrad.dataset import CELEBA_ATTRS, get_celeba_loaders  # noqa: E402
from lrad.ensemble import (  # noqa: E402
    collect_eye_region_bias,
    evaluate_ensemble_decomposition,
    evaluate_ensemble_uncertainty,
    identity_residual,
    sample_block_recons,
    sample_decomposition,
    sample_mean_error_maps,
    sample_min_error_maps,
    sample_quantile_min_error_maps,
)
from lrad.plots import (  # noqa: E402
    plot_bias_variance_vs_block,
    plot_bias_variance_vs_percentile,
    plot_decomposition_auroc_bars,
    plot_ensemble_decomposition,
    plot_ensemble_score_hists,
    plot_instance_summary,
    plot_mean_abs_bias,
    plot_mean_error_maps,
    plot_member_instance,
    plot_min_error_maps,
    plot_per_block_breakdown,
    plot_recons_only,
    plot_score_comparison,
    plot_top_ood_glasses,
    plot_variance_heatmaps,
)
from lrad.utils import get_device, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger("celeba_ood")

TERMS = ("risk", "bias", "variance")


def _per_model_record(seed: int, results: dict, variant: dict) -> dict:
    """Compact per-model summary: architecture, gender accuracy + the
    classifier OOD AUROCs."""
    auroc = results.get("auroc", {})

    def _au(key: str):
        v = auroc.get(key, {})
        return v.get("auroc") if isinstance(v, dict) else None

    return {
        "seed": seed,
        "channels": list(variant.get("channels", [])),
        "kernel_size": variant.get("kernel_size"),
        "gender_acc": results.get("accuracy", {}).get("gender"),
        "auroc_msp": _au("score_msp"),
        "auroc_entropy_gender": _au("score_entropy_gender"),
        "auroc_entropy_combined": _au("score_entropy_combined"),
    }


def _member_configs(cfg: dict, size: int) -> list[dict]:
    """One resolved config per ensemble member.

    ``ensemble.member_variants`` is a list of model overrides
    (``{channels, kernel_size}``); member ``i`` gets variant ``i`` (cycled
    when the ensemble outgrows the list). Every variant must keep the same
    number of conv blocks so the per-block decomposition stays aligned
    across members. Without variants every member shares ``cfg['model']``
    (the historical behaviour).
    """
    import copy

    variants = cfg.get("ensemble", {}).get("member_variants") or []
    n_blocks = len(cfg.get("model", {}).get("channels",
                                            (32, 64, 128, 256, 256)))
    member_cfgs: list[dict] = []
    for i in range(size):
        mc = copy.deepcopy(cfg)
        if variants:
            var = variants[i % len(variants)]
            channels = list(var.get("channels",
                                    mc["model"].get("channels")))
            if len(channels) != n_blocks:
                raise ValueError(
                    f"member_variants[{i % len(variants)}] has "
                    f"{len(channels)} blocks, expected {n_blocks} — the "
                    "per-block decomposition needs the same block count "
                    "in every member."
                )
            mc["model"]["channels"] = channels
            mc["model"]["kernel_size"] = int(var.get("kernel_size", 3))
        member_cfgs.append(mc)
    return member_cfgs


def _write_instance_figures(
    models,
    decoders_list,
    images,
    device,
    out_dir: Path,
    *,
    block: int,
    prefix: str,
    sigma: float,
    overlay_power: float,
) -> int:
    """Save the per-instance figures for every image in ``images``.

    Reconstructs the whole batch once with every member (at ``block``), then
    slices it image by image. Each instance gets its own folder
    ``{prefix}_XX/`` holding one figure PER ENSEMBLE MEMBER —
    ``model_01.png`` … ``model_M.png``, that member's reconstruction +
    error map — plus ``summary.png`` with the Bias / Mean-error / Min-error
    maps and the smoothed-bias overlay. Returns the number of instances
    written.
    """
    images_dev, recons = sample_block_recons(
        models, decoders_list, images, device, block,
    )
    n = images_dev.size(0)
    n_models = len(recons)
    for i in range(n):
        inst_dir = out_dir / f"{prefix}_{i + 1:02d}"
        inst_dir.mkdir(parents=True, exist_ok=True)
        recon_i = [recons[m][i] for m in range(n_models)]
        for m in range(n_models):
            plot_member_instance(
                images_dev[i], recon_i[m],
                inst_dir / f"model_{m + 1:02d}.png",
                member=m + 1,
                title=f"{prefix} sample {i + 1} — member {m + 1}/{n_models}"
                      f"  (block L{block})",
            )
        plot_instance_summary(
            images_dev[i], recon_i,
            inst_dir / "summary.png",
            sigma=sigma,
            overlay_power=overlay_power,
            title=f"{prefix} sample {i + 1} — block L{block}  "
                  f"({n_models}-model ensemble)",
        )
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a deep ensemble and run the bias/variance "
                    "decomposition for the CelebA OOD task.",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Ensemble output root. Default: a sibling "
                         "'ensemble' folder next to experiment.output_dir.")
    ap.add_argument("--ensemble-size", type=int, default=None,
                    help="Number of models. Default: ensemble.size in cfg.")
    ap.add_argument("--base-seed", type=int, default=None,
                    help="Model i is seeded base_seed + i. "
                         "Default: ensemble.base_seed in cfg.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; load each model_<i>/weights/ and "
                         "re-run the decomposition.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dotted overrides, e.g. training.epochs=50.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip per-model and ensemble plots.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    exp = cfg["experiment"]
    ecfg = cfg.get("ensemble", {})
    size = int(args.ensemble_size or ecfg.get("size", 5))
    base_seed = int(args.base_seed if args.base_seed is not None
                    else ecfg.get("base_seed", exp.get("seed", 42)))
    agg = str(ecfg.get("agg", cfg.get("evaluation", {})
                        .get("anomaly_agg", "p95")))
    seeds = [base_seed + i for i in range(size)]

    output_dir = Path(
        args.output_dir
        or (Path(exp["output_dir"]).parent / "ensemble")
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    ens_dir = output_dir / "ensemble"
    (ens_dir / "plots").mkdir(parents=True, exist_ok=True)

    seed_everything(base_seed)
    job_id = os.environ.get("OAR_JOB_ID", "")
    run_tag = f"{exp['name']}_ensemble{('_oar' + job_id) if job_id else ''}"
    setup_logging(str(output_dir / "logs"), run_tag=run_tag)
    device = get_device()

    logger.info("=" * 70)
    logger.info(f"Ensemble experiment : {exp['name']}")
    logger.info(f"Config              : {args.config}")
    logger.info(f"Output dir          : {output_dir}")
    logger.info(f"Device              : {device}")
    logger.info(f"Ensemble size       : {size}   seeds = {seeds}")
    logger.info(f"Decomposition agg   : {agg}")
    if device.type == "cuda":
        try:
            logger.info(
                f"GPU                 : {torch.cuda.get_device_name(0)}  "
                f"(torch {torch.__version__}, cuda {torch.version.cuda})"
            )
        except Exception:
            pass
    logger.info("=" * 70)

    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Data (built once; identical split for every model) --------------
    loaders = get_celeba_loaders(cfg)
    n_val = (len(loaders['val'].dataset)
             if loaders.get('val') is not None else 0)
    logger.info(
        f"Dataloaders  train={len(loaders['train'].dataset)}  "
        f"val={n_val}  "
        f"test_in={len(loaders['test_in'].dataset)}  "
        f"test_ood={len(loaders['test_ood'].dataset)}"
    )

    # --- Train / load the ensemble ---------------------------------------
    # Each member gets its own architecture from ensemble.member_variants
    # (cycled if size exceeds the list) on top of its own seed.
    member_cfgs = _member_configs(cfg, size)

    models: list = []
    decoders_list: list = []
    per_model: list[dict] = []
    t_ensemble = time.time()

    for i, seed in enumerate(seeds):
        mcfg = member_cfgs[i]["model"]
        logger.info("#" * 70)
        logger.info(
            f"# MODEL {i + 1}/{size}   (seed={seed}, "
            f"channels={mcfg.get('channels')}, "
            f"kernel={mcfg.get('kernel_size', 3)})"
        )
        logger.info("#" * 70)
        model_dir = output_dir / f"model_{i}"
        t0 = time.time()
        model, decoders, results = build_and_run(
            member_cfgs[i], loaders, device, model_dir,
            seed=seed, eval_only=args.eval_only, no_plots=args.no_plots,
        )
        if decoders is None:
            raise RuntimeError(
                f"model {i} has no decoders — the ensemble decomposition "
                "needs per-block decoders; check the training.decoders "
                "config section (or the saved decoder weights)."
            )
        logger.info(
            f"model {i + 1}/{size} done in {time.time() - t0:.1f}s"
        )
        # Park the finished model on CPU so GPU memory stays free while the
        # remaining models train; moved back as a group for decomposition.
        models.append(model.to("cpu").eval())
        decoders_list.append(decoders.to("cpu").eval())
        per_model.append(_per_model_record(seed, results, mcfg))

    logger.info(
        f"All {size} models ready in {time.time() - t_ensemble:.1f}s"
    )

    # --- Ensemble bias/variance decomposition ----------------------------
    logger.info("=" * 70)
    logger.info("ENSEMBLE DECOMPOSITION — Risk = Bias + Variance")
    logger.info("=" * 70)
    models = [m.to(device).eval() for m in models]
    decoders_list = [d.to(device).eval() for d in decoders_list]

    deb = evaluate_ensemble_decomposition(
        models, decoders_list, loaders, device, agg=agg,
    )
    blocks = deb["blocks"]
    auroc = deb["auroc"]

    agg_au = {t: auroc[f"score_{t}_aggregated"].get("auroc") for t in TERMS}
    logger.info(
        f"Aggregated OOD AUROC ({agg}):  "
        f"Risk={agg_au['risk']:.4f}  Bias={agg_au['bias']:.4f}  "
        f"Variance={agg_au['variance']:.4f}"
    )
    logger.info(
        f"  -> anomaly score = Bias = |Risk − Variance| (no sigma, no "
        f"division):  AUROC={agg_au['bias']:.4f}"
    )
    for k in blocks:
        row = "  ".join(
            f"{t}={auroc[f'score_{t}_per_block_{k}'].get('auroc', float('nan')):.4f}"
            for t in TERMS
        )
        logger.info(f"  block {k}:  {row}")

    # --- Predictive-uncertainty decomposition (classifier heads) ----------
    # The variance-based OOD signal that actually works on this occlusion
    # task: epistemic uncertainty = ensemble predictive mutual information.
    logger.info("=" * 70)
    logger.info("PREDICTIVE UNCERTAINTY — Total = Aleatoric + Epistemic (MI)")
    logger.info("=" * 70)
    unc = evaluate_ensemble_uncertainty(models, loaders, device)
    for head in unc["heads"]:
        row = "  ".join(
            f"{t}={unc['auroc'][head][t].get('auroc', float('nan')):.4f}"
            for t in unc["terms"]
        )
        logger.info(f"  {head:<9}: {row}")
    logger.info(
        "  -> OOD score = epistemic (combined predictive MI):  "
        f"AUROC={unc['epistemic_auroc']['combined']:.4f}"
    )

    # --- Plots ------------------------------------------------------------
    if not args.no_plots:
        plot_dir = ens_dir / "plots"
        ecfg_v = cfg.get("evaluation", {})
        n_viz_in = int(ecfg_v.get(
            "n_viz_in_samples", ecfg_v.get("n_viz_samples", 4),
        ))
        n_viz_ood = int(ecfg_v.get(
            "n_viz_ood_samples", ecfg_v.get("n_viz_samples", 4),
        ))
        viz_seed = ecfg_v.get("viz_seed")
        samples_in = _gather_samples(
            loaders["test_in"], n_viz_in, seed=viz_seed,
        )
        samples_ood = _gather_samples(
            loaders["test_ood"], n_viz_ood,
            seed=(viz_seed + 1) if viz_seed is not None else None,
        )
        all_images = torch.cat([samples_in, samples_ood], dim=0)
        row_labels = (
            [f"ID  {i + 1}" for i in range(samples_in.size(0))]
            + [f"OOD {i + 1}" for i in range(samples_ood.size(0))]
        )

        maps = sample_decomposition(models, decoders_list, all_images, device)
        residual = identity_residual(maps)
        logger.info(
            f"Risk = Bias + Variance identity — max abs residual on "
            f"{all_images.size(0)} samples ({n_viz_in} ID + {n_viz_ood} "
            f"OOD): {residual:.2e}"
        )

        plot_ensemble_decomposition(
            all_images, maps, plot_dir / "ensemble_decomposition.png",
            row_labels=row_labels,
            title=f"Per-block Risk | Bias | Variance  ({size}-model ensemble)",
        )
        plot_decomposition_auroc_bars(
            deb["per_block_auroc"], plot_dir / "decomposition_auroc.png",
            block_labels=[f"Block {k}" for k in blocks],
            aggregated=agg_au,
            title="Per-block OOD AUROC — Risk / Bias / Variance",
        )
        plot_ensemble_score_hists(
            deb["scores_in"]["aggregated"], deb["scores_ood"]["aggregated"],
            plot_dir / "ensemble_score_hists.png",
            auroc=agg_au,
            title=f"Ensemble decomposition scores (agg='{agg}')",
        )

        # Item 5 of the note: results on the ensemble-mean reconstruction.
        # (x - E_D[f_hat])² is exactly the Bias view of the error.
        mean_recons = [maps[k]["mean_recon"] for k in blocks]
        plot_per_block_breakdown(
            all_images, mean_recons, plot_dir / "mean_recon_breakdown.png",
            row_labels=row_labels,
            title="Ensemble-mean reconstruction — Original | Err Lk | Recon Lk "
                  "(error = Bias term, (x − f̄)²)",
        )
        plot_recons_only(
            all_images, mean_recons, plot_dir / "mean_recons_only.png",
            row_labels=row_labels,
            title="Per-block ensemble-mean reconstructions  E_D[f_hat]",
        )

        # --- New plots requested by the user ---------------------------
        # (x − f̄)² per block (bias-only view).
        plot_mean_abs_bias(
            all_images, mean_recons, plot_dir / "mean_abs_bias.png",
            row_labels=row_labels,
            title="Ensemble bias  (x − f̄)²  per conv block  "
                  f"({size}-model ensemble)",
        )
        # Variance heatmaps on OOD (and ID for reference), overlaid.
        plot_variance_heatmaps(
            samples_ood,
            {k: {"variance": maps[k]["variance"]
                 [n_viz_in:]} for k in blocks},
            plot_dir / "variance_heatmaps_ood.png",
            row_labels=[f"OOD {i + 1}" for i in range(samples_ood.size(0))],
            title="Ensemble variance heatmap — OOD samples",
        )
        plot_variance_heatmaps(
            all_images, maps,
            plot_dir / "variance_heatmaps_all.png",
            row_labels=row_labels,
            title="Ensemble variance heatmap — ID + OOD samples",
        )

        # Per-block mean over models of the squared error map
        # (mean_m (x − f̂^m)²): the average of the per-model error maps,
        # pixel by pixel — exactly the Risk term, displayed per block.
        err_maps = sample_mean_error_maps(
            models, decoders_list, all_images, device,
        )
        plot_mean_error_maps(
            all_images, err_maps, plot_dir / "mean_error_maps.png",
            row_labels=row_labels,
            title="Ensemble-averaged error  mean_m (x − f̂^m)²  per conv block  "
                  f"({size}-model ensemble)",
        )

        # Same view but the per-pixel *minimum* over models
        # (min_m (x − f̂^m)²): error of the best member, pixel by pixel.
        min_err_maps = sample_min_error_maps(
            models, decoders_list, all_images, device,
        )
        plot_min_error_maps(
            all_images, min_err_maps, plot_dir / "min_error_maps.png",
            row_labels=row_labels,
            title="Ensemble best-member error  min_m (x − f̂^m)²  per conv block  "
                  f"({size}-model ensemble)",
        )

        # Comparative score maps at ONE reconstruction depth (the deepest
        # block by default; evaluation.score_comparison_block overrides):
        # Bias (error of the mean prediction) | Risk (mean of the errors) |
        # min of the errors | robust quantile-min (k-th smallest error,
        # k = evaluation.score_comparison_k, default 3). Columns keep their
        # OWN colour scales — the scores are not identically distributed.
        cmp_block = int(ecfg_v.get("score_comparison_block", blocks[-1]))
        cmp_k = int(ecfg_v.get("score_comparison_k", 3))
        qmin_err_maps = sample_quantile_min_error_maps(
            models, decoders_list, all_images, device, k=cmp_k,
        )
        plot_score_comparison(
            all_images,
            [
                ("Bias  (x − f̄)²", maps[cmp_block]["bias"]),
                ("Risk  mean_m", maps[cmp_block]["risk"]),
                ("min_m", min_err_maps[cmp_block]),
                (f"{cmp_k}-th smallest", qmin_err_maps[cmp_block]),
            ],
            plot_dir / "score_comparison.png",
            row_labels=row_labels,
            title=f"Score comparison at block {cmp_block}  "
                  f"(per-column colour scales, {size}-model ensemble)",
        )

        # --- Per-instance figures: one face at a time -------------------
        # For a couple of dozen ID and OOD faces, write a standalone figure
        # showing every member's reconstruction and error, the bias / mean /
        # min summary, and the smoothed bias overlaid on the face. Same
        # reconstruction depth as the score comparison; the overlay's sigma
        # and alpha falloff are display-only knobs.
        n_inst_in = int(ecfg_v.get("n_instances_in",
                                   ecfg_v.get("n_instances", 20)))
        n_inst_ood = int(ecfg_v.get("n_instances_ood",
                                    ecfg_v.get("n_instances", 20)))
        inst_block = int(ecfg_v.get("instance_block", cmp_block))
        overlay_sigma = float(ecfg_v.get("overlay_sigma", 3.0))
        overlay_power = float(ecfg_v.get("overlay_power", 0.8))
        # A fresh, larger draw of faces; the seed offset keeps them distinct
        # from the small grid-figure samples gathered above.
        inst_in = _gather_samples(
            loaders["test_in"], n_inst_in,
            seed=(viz_seed + 100) if viz_seed is not None else None,
        )
        inst_ood = _gather_samples(
            loaders["test_ood"], n_inst_ood,
            seed=(viz_seed + 101) if viz_seed is not None else None,
        )
        n_in_done = _write_instance_figures(
            models, decoders_list, inst_in, device,
            plot_dir / "instances_in",
            block=inst_block, prefix="ID",
            sigma=overlay_sigma, overlay_power=overlay_power,
        )
        n_ood_done = _write_instance_figures(
            models, decoders_list, inst_ood, device,
            plot_dir / "instances_ood",
            block=inst_block, prefix="OOD",
            sigma=overlay_sigma, overlay_power=overlay_power,
        )
        logger.info(
            f"Wrote per-instance figures: {n_in_done} ID → instances_in/, "
            f"{n_ood_done} OOD → instances_ood/  (one folder per instance: "
            f"{size} member figures + summary.png; block L{inst_block}, "
            f"sigma={overlay_sigma:g}, overlay_power={overlay_power:g})"
        )

        # --- Top OOD faces where the glasses are best detected ----------
        # Rank the FULL OOD test set, restricted to faces that actually
        # carry the Eyeglasses attribute, by how strongly AND how locally
        # the bias (x − f̄)² lights up in the eye region (see
        # collect_eye_region_bias). The winners are rendered as one figure
        # (Original / Bias / Overlay per column, rank order) and their
        # ranking is recorded in top_ood_glasses.json.
        n_top = int(ecfg_v.get("n_top_ood", 10))
        ood_ds = loaders["test_ood"].dataset
        eye_idx = CELEBA_ATTRS.index("Eyeglasses")
        glasses_mask = (
            ood_ds.base.attr[list(ood_ds.indices), eye_idx]
            .to(torch.bool).numpy()
        )
        if not glasses_mask.any():
            logger.info(
                "No Eyeglasses faces in the OOD split — skipping the "
                "top-OOD-glasses figure."
            )
        else:
            eye_scores = collect_eye_region_bias(
                models, decoders_list, loaders["test_ood"], device,
                block=inst_block,
            )
            score = np.where(glasses_mask, eye_scores["score"], -np.inf)
            n_top = min(n_top, int(glasses_mask.sum()))
            top_idx = np.argsort(score)[::-1][:n_top]
            top_images = torch.stack(
                [ood_ds[int(j)][0] for j in top_idx], dim=0,
            )
            top_dev, top_recons = sample_block_recons(
                models, decoders_list, top_images, device, inst_block,
            )
            plot_top_ood_glasses(
                top_dev, top_recons,
                plot_dir / "top_ood_glasses.png",
                sigma=overlay_sigma, overlay_power=overlay_power,
                title=f"Top {n_top} OOD eyeglasses faces — eye-region bias "
                      f"(block L{inst_block}, {size}-model ensemble)",
            )
            top_record = [
                {
                    "rank": r + 1,
                    "ood_dataset_index": int(j),
                    "celeba_index": int(ood_ds.indices[int(j)]),
                    "score": float(eye_scores["score"][int(j)]),
                    "eye_mean_bias": float(eye_scores["eye_mean"][int(j)]),
                    "global_mean_bias": float(
                        eye_scores["global_mean"][int(j)]
                    ),
                }
                for r, j in enumerate(top_idx)
            ]
            with open(ens_dir / "top_ood_glasses.json", "w") as f:
                json.dump(top_record, f, indent=2)
            logger.info(
                f"Top {n_top} OOD eyeglasses faces (eye-region bias, block "
                f"L{inst_block}) → top_ood_glasses.png; ranking in "
                "top_ood_glasses.json"
            )

        # Bias/Variance evolution curves (full test loaders).
        plot_bias_variance_vs_block(
            deb["scores_in"]["per_block"],
            deb["scores_ood"]["per_block"],
            plot_dir / "bias_variance_vs_block.png",
            blocks=blocks,
        )
        plot_bias_variance_vs_percentile(
            {t: deb["scores_in"]["aggregated"][t] for t in TERMS},
            {t: deb["scores_ood"]["aggregated"][t] for t in TERMS},
            plot_dir / "bias_variance_vs_percentile.png",
        )

        logger.info(
            "Wrote ensemble plots: ensemble_decomposition.png, "
            "decomposition_auroc.png, ensemble_score_hists.png, "
            "mean_recon_breakdown.png, mean_recons_only.png, "
            "mean_abs_bias.png, variance_heatmaps_ood.png, "
            "variance_heatmaps_all.png, mean_error_maps.png, "
            "min_error_maps.png, score_comparison.png, "
            "bias_variance_vs_block.png, bias_variance_vs_percentile.png"
        )
    else:
        residual = None

    # --- Summary ----------------------------------------------------------
    summary = {
        "experiment": exp["name"],
        "ensemble_size": size,
        "seeds": seeds,
        "agg": agg,
        "device": str(device),
        "identity_max_residual": residual,
        "per_model": per_model,
        "decomposition_auroc": {
            "aggregated": agg_au,
            "per_block": {
                t: _to_jsonable(deb["per_block_auroc"][t]) for t in TERMS
            },
        },
        # Headline OOD score: the Bias term (bias = risk − variance).
        "anomaly_auroc": _to_jsonable(deb["anomaly_auroc"]),
        # Predictive-uncertainty decomposition (classifier heads). The
        # epistemic term is the ensemble's predictive variance (mutual
        # information) — the OOD signal that works on this occlusion task.
        "uncertainty_auroc": {
            "epistemic": _to_jsonable(unc["epistemic_auroc"]),
            "by_head": {
                head: {
                    t: unc["auroc"][head][t].get("auroc")
                    for t in unc["terms"]
                }
                for head in unc["heads"]
            },
        },
    }
    with open(ens_dir / "summary.json", "w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    logger.info(f"Ensemble summary written to {ens_dir / 'summary.json'}")
    logger.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if logger.handlers:
            logger.exception("run_ensemble crashed")
        else:
            traceback.print_exc()
        sys.exit(1)
