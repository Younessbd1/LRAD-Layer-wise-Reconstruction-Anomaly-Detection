#!/usr/bin/env python3
"""Merge the ablation arms into one results table + comparison figures.

The ablation (see configs/ablation_baseline.yaml for the design) trains
four ensembles that differ only in two factors — architecture diversity
and the CutPaste pretext — and answers three case studies, all against
the same control:

  1. arch          vs baseline   effect of architecture diversity alone
  2. cutpaste      vs baseline   effect of the CutPaste pretext alone
  3. arch_cutpaste vs baseline   joint effect of both ingredients

Each arm is a full run directory written by scripts/oar_run_ablation.sh
(same structure as outputs/celeba_ood/LASTOF_RESULTS): this script only
READS each arm's ensemble/{summary,fused_auroc,localized_auroc}.json, so
it is cheap and idempotent — rerun it whenever another arm lands.

Arm discovery (per arm, first hit wins):
  1. an explicit ``--arm name=dir`` override;
  2. the newest ``<root>/<name>_<ts>_<jobid>/`` (or plain ``<root>/<name>/``)
     that contains ensemble/summary.json;
  3. for arch_cutpaste only: outputs/celeba_ood/LASTOF_RESULTS — the
     archived run of exactly that recipe (128 px, 10 diverse members,
     CutPaste winner, seeds 42..51).

Outputs (under ``<root>/comparison/``):
  ablation_results.json          every metric of every arm + the deltas
  ablation_table.md / .csv       the results table (also printed to stdout)
  plots/auroc_bars.png           grouped AUROC bars, one group per metric
  plots/case_studies.png         signed deltas vs baseline, per case study
  plots/per_block.png            per-block Risk/Bias/Variance AUROC lines
  plots/members.png              per-member recon AUROC + gender-acc spread

Usage:
    python scripts/compare_ablation.py
    python scripts/compare_ablation.py --root outputs/celeba_ood/ablation \\
        --arm arch_cutpaste=outputs/celeba_ood/LASTOF_RESULTS
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.plots import (  # noqa: E402
    plot_ablation_auroc_bars,
    plot_ablation_deltas,
    plot_ablation_members,
    plot_ablation_per_block,
)

logger = logging.getLogger("celeba_ood")

BASELINE = "baseline"
ARMS = (BASELINE, "arch", "cutpaste", "arch_cutpaste")
ARM_LABELS = {
    "baseline": "Baseline (same arch)",
    "arch": "Arch diversity",
    "cutpaste": "CutPaste",
    "arch_cutpaste": "Arch + CutPaste",
}
CASE_STUDIES = (
    ("arch", "1. Arch diversity vs baseline"),
    ("cutpaste", "2. CutPaste vs baseline"),
    ("arch_cutpaste", "3. Arch + CutPaste vs baseline"),
)

# (metric key, display label) in table/figure order. Keys come from
# ensemble/summary.json (decomposition + uncertainty), fused_auroc.json
# (per-signal + fused) and localized_auroc.json (aggregated entries).
METRICS = (
    ("fused_supervised", "fused (supervised)"),
    ("fused_rank", "fused (rank)"),
    ("locfre_b1", "locfre b1"),
    ("locfre_b2", "locfre b2"),
    ("locfre_b3", "locfre b3"),
    ("cutpaste_prob", "cutpaste P(altered)"),
    ("ens_energy_gender", "gender energy"),
    ("unc_epistemic_combined", "epistemic MI"),
    ("bias", "decomp bias (p95)"),
    ("risk", "decomp risk (p95)"),
    ("variance", "decomp variance (p95)"),
    ("zscore_risk_aggregated", "localized z-score (risk)"),
)
# The subset worth a bar in the figures (the full list stays in the table).
FIGURE_METRICS = tuple(
    (k, lbl) for k, lbl in METRICS
    if k not in ("locfre_b1", "locfre_b2", "risk")
)
# Deltas are only meaningful for signals both arms carry; cutpaste_prob
# does not exist on the baseline, so it never yields a delta.
DELTA_METRICS = FIGURE_METRICS

_RUN_DIR_RE = r"^{arm}(_\d{{8}}_\d{{6}}(_\w+)?)?$"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "ensemble" / "summary.json"


def discover_arm(root: Path, arm: str) -> Path | None:
    """Newest finished run dir for ``arm`` under ``root`` (None if none)."""
    pat = re.compile(_RUN_DIR_RE.format(arm=re.escape(arm)))
    candidates = [
        d for d in root.glob(f"{arm}*")
        if d.is_dir() and pat.match(d.name) and _summary_path(d).is_file()
    ]
    if candidates:
        # Newest first; the dir name (which embeds the submission
        # timestamp) breaks mtime ties deterministically.
        return max(
            candidates,
            key=lambda d: (_summary_path(d).stat().st_mtime, d.name),
        )
    if arm == "arch_cutpaste":
        legacy = _ROOT / "outputs" / "celeba_ood" / "LASTOF_RESULTS"
        if _summary_path(legacy).is_file():
            return legacy
    return None


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def load_arm(arm: str, run_dir: Path) -> dict:
    """Every metric this arm has, plus the per-block and per-member series.

    ``fused_auroc.json`` / ``localized_auroc.json`` are optional (the
    post-evaluations may still be running); the metrics they would
    contribute are then simply absent.
    """
    summary = _load_json(_summary_path(run_dir))
    if summary is None:
        raise FileNotFoundError(f"{_summary_path(run_dir)} not found")

    metrics: dict[str, float] = dict(
        summary.get("decomposition_auroc", {}).get("aggregated", {}),
    )
    epi = summary.get("uncertainty_auroc", {}).get("epistemic", {})
    if epi.get("combined") is not None:
        metrics["unc_epistemic_combined"] = epi["combined"]

    fused = _load_json(run_dir / "ensemble" / "fused_auroc.json")
    if fused:
        for k, v in fused.get("auroc", {}).items():
            if isinstance(v, dict) and v.get("auroc") is not None:
                metrics[k] = v["auroc"]

    localized = _load_json(run_dir / "ensemble" / "localized_auroc.json")
    if localized:
        for k, v in localized.get("auroc", {}).items():
            if (k.endswith("_aggregated") and isinstance(v, dict)
                    and v.get("auroc") is not None):
                metrics[k] = v["auroc"]

    per_model = summary.get("per_model", [])
    return {
        "arm": arm,
        "label": ARM_LABELS.get(arm, arm),
        "run_dir": str(run_dir),
        "ensemble_size": summary.get("ensemble_size"),
        "seeds": summary.get("seeds"),
        "metrics": {k: float(v) for k, v in metrics.items()
                    if v is not None},
        "per_block": {
            t: list(v) for t, v in
            summary.get("decomposition_auroc", {}).get("per_block", {}).items()
        },
        "members": {
            "auroc_recon": [m.get("auroc_recon") for m in per_model],
            "gender_acc": [m.get("gender_acc") for m in per_model],
            "n_params": [m.get("n_params") for m in per_model],
        },
        "has_fused": fused is not None,
        "has_localized": localized is not None,
    }


def _mean_std(vals: list) -> tuple[float, float] | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def case_deltas(arm: dict, baseline: dict) -> dict[str, float]:
    """Signed deltas vs the baseline over the shared headline metrics."""
    return {
        lbl: arm["metrics"][k] - baseline["metrics"][k]
        for k, lbl in DELTA_METRICS
        if k in arm["metrics"] and k in baseline["metrics"]
    }


def build_table_md(arms: list[dict], cases: list[dict]) -> str:
    """The whole comparison as one Markdown document."""

    def fmt(v, delta=False):
        if v is None:
            return "—"
        return f"{v:+.4f}" if delta else f"{v:.4f}"

    lines = ["# CelebA OOD ablation — results", ""]
    lines.append("Arms (10-member ensembles, 128 px, seeds 42..51):")
    lines.append("")
    for a in arms:
        extra = "" if a["has_fused"] else "  *(fused scoring pending)*"
        lines.append(f"- **{a['label']}** — `{a['run_dir']}`{extra}")
    lines.append("")

    header = ["Metric (OOD AUROC)"] + [a["label"] for a in arms]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for k, lbl in METRICS:
        row = [lbl] + [fmt(a["metrics"].get(k)) for a in arms]
        lines.append("| " + " | ".join(row) + " |")
    for key, lbl in (("auroc_recon", "member recon AUROC (mean ± sd)"),
                     ("gender_acc", "member gender acc (mean ± sd)")):
        row = [lbl]
        for a in arms:
            ms = _mean_std(a["members"][key])
            row.append("—" if ms is None else f"{ms[0]:.4f} ± {ms[1]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Case studies — Δ AUROC vs baseline")
    lines.append("")
    for case in cases:
        lines.append(f"### {case['label']}")
        lines.append("")
        if not case["deltas"]:
            lines.append("*arm not available yet*")
            lines.append("")
            continue
        lines.append("| Metric | Δ vs baseline |")
        lines.append("|---|---|")
        for lbl, v in case["deltas"].items():
            lines.append(f"| {lbl} | {fmt(v, delta=True)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_table_csv(arms: list[dict]) -> str:
    lines = ["metric,arm,auroc"]
    for k, _lbl in METRICS:
        for a in arms:
            v = a["metrics"].get(k)
            if v is not None:
                lines.append(f"{k},{a['arm']},{v:.6f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge the ablation arms into one results table + "
                    "comparison figures.",
    )
    ap.add_argument("--root", type=Path,
                    default=_ROOT / "outputs" / "celeba_ood" / "ablation",
                    help="Directory holding the per-arm run dirs.")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=DIR",
                    help="Explicit run dir for an arm, e.g. "
                         "arch_cutpaste=outputs/celeba_ood/LASTOF_RESULTS. "
                         "Repeatable; overrides discovery.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write tables + figures "
                         "(default: <root>/comparison).")
    ap.add_argument("--no-plots", action="store_true",
                    help="Tables only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    overrides: dict[str, Path] = {}
    for spec in args.arm:
        name, _, d = spec.partition("=")
        if name not in ARMS or not d:
            ap.error(f"--arm expects NAME=DIR with NAME in {ARMS}, "
                     f"got {spec!r}")
        overrides[name] = Path(d)

    arms: list[dict] = []
    missing: list[str] = []
    for arm in ARMS:
        run_dir = overrides.get(arm) or discover_arm(args.root, arm)
        if run_dir is None or not _summary_path(run_dir).is_file():
            missing.append(arm)
            continue
        arms.append(load_arm(arm, run_dir))
        logger.info(f"arm {arm:<14} <- {arms[-1]['run_dir']}")
    if missing:
        logger.info(f"arms not found (yet): {', '.join(missing)}")

    by_arm = {a["arm"]: a for a in arms}
    if BASELINE not in by_arm:
        logger.info(
            f"cannot compare: no finished '{BASELINE}' arm under "
            f"{args.root} — nothing written."
        )
        return 2
    if len(arms) < 2:
        logger.info("cannot compare: only the baseline has finished — "
                    "nothing written.")
        return 2

    baseline = by_arm[BASELINE]
    cases = [
        {
            "arm": arm,
            "label": label,
            "deltas": (case_deltas(by_arm[arm], baseline)
                       if arm in by_arm else {}),
        }
        for arm, label in CASE_STUDIES
    ]

    out_dir = args.output_dir or (args.root / "comparison")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump({"arms": arms, "case_studies": cases}, f, indent=2)
    table_md = build_table_md(arms, cases)
    (out_dir / "ablation_table.md").write_text(table_md)
    (out_dir / "ablation_table.csv").write_text(build_table_csv(arms))

    if not args.no_plots:
        plot_ablation_auroc_bars(
            arms, FIGURE_METRICS, plot_dir / "auroc_bars.png",
            title="Ablation — OOD AUROC by arm (CelebA eyeglasses, 128 px)",
        )
        drawn_cases = [c for c in cases if c["deltas"]]
        if drawn_cases:
            plot_ablation_deltas(
                drawn_cases, plot_dir / "case_studies.png",
            )
        plot_ablation_per_block(arms, plot_dir / "per_block.png")
        plot_ablation_members(arms, plot_dir / "members.png")
        logger.info(
            f"figures -> {plot_dir}/auroc_bars.png, case_studies.png, "
            f"per_block.png, members.png"
        )

    logger.info(f"tables  -> {out_dir}/ablation_table.md, "
                f"ablation_table.csv, ablation_results.json\n")
    print(table_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
