#!/usr/bin/env python3
"""Merge the per-category run directories of a parallel MVTec sweep.

scripts/oar_run_mvtec.sh trains one category per OAR job (see the header of
that script for why: ~10.2 h per category on a T4, so the five-category pilot
is ~51 h sequentially but ~10 h when the categories run side by side). Each
job therefore writes its OWN run directory holding a single-category
``summary.json`` and a single-row ``results.md``.

This rebuilds the one combined table those five jobs would have produced had
they run sequentially, using the same writer the runner itself uses, so the
merged table is identical in format to a single-job one.

Usage:
    python scripts/merge_mvtec_results.py outputs/mvtec/run_*_68492*
    python scripts/merge_mvtec_results.py outputs/mvtec/run_* -o outputs/mvtec/pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_mvtec import MVTEC_CATEGORIES, write_results_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Run directories to merge (each holding a summary.json).")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Where to write the merged results (default: alongside "
                         "the first run dir, as <parent>/merged).")
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    skipped: list[str] = []

    for d in args.run_dirs:
        summary = d / "summary.json"
        if not summary.is_file():
            # A job still running, or one killed before its first category
            # finished, simply has nothing to contribute yet.
            skipped.append(f"{d.name}: no summary.json")
            continue
        try:
            with open(summary) as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            # summary.json is rewritten in place after every category, so a
            # read that races the writer can land mid-write.
            skipped.append(f"{d.name}: unreadable ({exc})")
            continue

        for r in payload.get("results", []):
            cat = r.get("category")
            if cat is None:
                continue
            if cat in merged:
                skipped.append(f"{d.name}: duplicate {cat}, keeping the first")
                continue
            merged[cat] = r

    if not merged:
        print("No category results found — nothing to merge.", file=sys.stderr)
        return 1

    # Canonical MVTec order, so the merged table matches a sequential run's
    # regardless of which job happened to finish first.
    results = [merged[c] for c in MVTEC_CATEGORIES if c in merged]

    out_dir = args.output_dir or (args.run_dirs[0].parent / "merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = write_results_table(results, out_dir / "results.md")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "experiment": "mvtec_lrad",
            "categories": [r["category"] for r in results],
            "results": results,
            "merged_from": [str(d) for d in args.run_dirs],
        }, f, indent=2)

    for note in skipped:
        print(f"  note: {note}", file=sys.stderr)
    print(f"Merged {len(results)}/{len(MVTEC_CATEGORIES)} categories "
          f"({', '.join(r['category'] for r in results)}) -> {out_dir}")
    print()
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
