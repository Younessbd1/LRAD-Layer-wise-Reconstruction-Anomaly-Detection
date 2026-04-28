#!/usr/bin/env python3
"""Depth-ablation sweep for the Real-IAD LRAD pipeline.

Runs the full train+eval pipeline once per preset (resnet10 / resnet18 /
resnet34) sharing the same config, the same seed and the same data
split, then collates the resulting ``summary.json`` files into a
single comparison table + plot.

Usage::

    python scripts/ablation_depth.py --config configs/realiad.yaml \\
        --presets resnet10 resnet18 resnet34 \\
        --out-dir outputs/realiad/ablation_depth

Each preset gets its own sub-directory (``<out-dir>/<preset>``) so that
outputs don't collide and reruns are cheap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run_one(config: Path, preset: str, out_dir: Path,
             extra_overrides: list[str], dry_run: bool) -> Path:
    """Kick off scripts/run_realiad.py for one preset. Returns the sub-dir
    containing the summary.json."""
    sub = out_dir / preset
    sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_realiad.py"),
        "--config", str(config),
        "--output-dir", str(sub),
        "--override", f"model.preset={preset}",
        f"experiment.name=ablation_{preset}",
    ]
    for ov in extra_overrides:
        cmd.extend(["--override", ov])

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return sub

    print("=" * 70)
    print(f"Running preset {preset} ->  {sub}")
    print("=" * 70)
    subprocess.run(cmd, check=True)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--presets", nargs="+",
                    default=["resnet10", "resnet18", "resnet34"])
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/realiad/ablation_depth"))
    ap.add_argument("--override", nargs="*", default=[],
                    help="Extra overrides forwarded to run_realiad.py.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands but don't run.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for p in args.presets:
        sub = _run_one(args.config, p, args.out_dir, args.override, args.dry_run)
        sfile = sub / "summary.json"
        if sfile.exists():
            with open(sfile) as f:
                summaries.append(json.load(f))

    if args.dry_run or not summaries:
        return

    # --- Collate ---
    rows = []
    for s in summaries:
        rows.append({
            "preset": s["experiment"].replace("ablation_", ""),
            "params": s["classifier"]["parameters"] + s["decoders"]["parameters_total"],
            "auroc": s["metrics"]["image"]["auroc"],
            "pr_auc": s["metrics"]["image"]["pr_auc"],
            "f1": s["metrics"]["image"]["f1"],
            "pixel_auroc": (s["metrics"].get("pixel") or {}).get("auroc"),
        })

    table_path = args.out_dir / "depth_comparison.csv"
    with open(table_path, "w") as f:
        f.write("preset,params,auroc,pr_auc,f1,pixel_auroc\n")
        for r in rows:
            f.write(
                f"{r['preset']},{r['params']},{r['auroc']:.4f},"
                f"{r['pr_auc']:.4f},{r['f1']:.4f},"
                f"{r['pixel_auroc'] if r['pixel_auroc'] is not None else ''}\n"
            )
    print(f"Wrote comparison table to {table_path}")

    # --- Plot ---
    names = [r["preset"] for r in rows]
    auroc = [r["auroc"] for r in rows]
    prauc = [r["pr_auc"] for r in rows]
    f1 = [r["f1"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4.5),
                           constrained_layout=True)
    w = 0.27
    ax.bar(x - w, auroc, w, label="Image AUROC")
    ax.bar(x,     prauc, w, label="PR-AUC")
    ax.bar(x + w, f1,    w, label="F1 (best)")
    for i, v in enumerate(auroc):
        ax.text(i - w, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("LRAD depth ablation on Real-IAD")
    ax.legend()
    fig.savefig(args.out_dir / "depth_comparison.png")
    print(f"Wrote plot to {args.out_dir / 'depth_comparison.png'}")


if __name__ == "__main__":
    main()
