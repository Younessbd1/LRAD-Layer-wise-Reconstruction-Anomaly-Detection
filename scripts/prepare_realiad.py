"""Build a Real-IAD manifest CSV without unpacking the zips.

The manifest is the single source of truth for the dataset loader and all
downstream analysis. Every row describes one (category, sample_id, view)
triple and its paired mask (if any).

Usage:
    python scripts/prepare_realiad.py \\
        --data-root /home/ybahaddo/Real-IAD/realiad_1024 \\
        --out-dir outputs/realiad/manifest

Outputs:
    manifest.csv       one row per image with zip path + internal member
    metadata.json      per-category counts and the defect-type vocabulary
    warnings.txt       any detected inconsistencies (missing views, orphan masks)
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lrad.data.realiad import CATEGORIES, VIEWS, _parse_image_member


def _scan_zip(zip_path: Path) -> tuple[list[dict], list[str]]:
    """Return (rows, warnings) for one category zip."""
    rows: list[dict] = []
    warnings: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]

    # First pass: index PNG masks by stem (<dir>/<basename-without-ext>)
    mask_index: dict[str, str] = {}
    for m in members:
        if m.lower().endswith(".png"):
            mask_index[m[:-4]] = m

    # Second pass: JPG images → rows, pairing masks by shared stem.
    seen_masks: set[str] = set()
    samples_by_key: dict[str, set[str]] = defaultdict(set)

    for m in members:
        if not m.lower().endswith(".jpg"):
            continue
        parsed = _parse_image_member(m)
        if parsed is None:
            warnings.append(f"unparseable member: {m}")
            continue
        sample_id, view, label, defect_type, sample_key = parsed
        stem = m[:-4]
        mask_member = mask_index.get(stem)
        if mask_member is not None:
            seen_masks.add(mask_member)
        category = m.split("/")[0]
        samples_by_key[sample_key].add(view)
        rows.append({
            "category": category,
            "sample_id": sample_id,
            "view": view,
            "label": label,
            "defect_type": defect_type,
            "zip_path": str(zip_path),
            "image_member": m,
            "mask_member": mask_member or "",
        })

    # Flag samples missing any of the five standard views.
    for key, views in samples_by_key.items():
        missing = set(VIEWS) - views
        if missing:
            warnings.append(f"{key}: missing views {sorted(missing)}")

    # Flag masks with no matching JPG (shouldn't happen, but check).
    for mask in mask_index.values():
        if mask not in seen_masks:
            warnings.append(f"orphan mask: {mask}")

    return rows, warnings


def build_manifest(data_root: Path, out_dir: Path,
                   categories: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    warnings_path = out_dir / "warnings.txt"
    meta_path = out_dir / "metadata.json"

    cats = categories or list(CATEGORIES)
    all_rows: list[dict] = []
    all_warnings: list[str] = []

    print(f"Scanning {len(cats)} zips in {data_root}")
    for cat in cats:
        zip_path = data_root / f"{cat}.zip"
        if not zip_path.exists():
            all_warnings.append(f"missing zip: {zip_path}")
            print(f"  [skip] {cat}: zip not found")
            continue
        rows, warnings = _scan_zip(zip_path)
        all_rows.extend(rows)
        all_warnings.extend(warnings)
        ok = sum(1 for r in rows if r["label"] == 0)
        ng = len(rows) - ok
        masks = sum(1 for r in rows if r["mask_member"])
        print(f"  {cat:20s}  total={len(rows):6d}  OK={ok:5d}  NG={ng:5d}  masks={masks:5d}")

    fields = ["category", "sample_id", "view", "label", "defect_type",
              "zip_path", "image_member", "mask_member"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Metadata summary
    per_cat = defaultdict(lambda: {"ok": 0, "ng": 0, "views": Counter(),
                                    "defects": Counter(), "masks": 0})
    for r in all_rows:
        s = per_cat[r["category"]]
        if r["label"] == 0:
            s["ok"] += 1
        else:
            s["ng"] += 1
            s["defects"][r["defect_type"]] += 1
        s["views"][r["view"]] += 1
        if r["mask_member"]:
            s["masks"] += 1

    meta = {
        "data_root": str(data_root),
        "total_rows": len(all_rows),
        "categories": {
            cat: {
                "ok": v["ok"],
                "ng": v["ng"],
                "masks": v["masks"],
                "views": dict(v["views"]),
                "defect_types": dict(v["defects"]),
            }
            for cat, v in sorted(per_cat.items())
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    with open(warnings_path, "w") as f:
        f.write("\n".join(all_warnings))

    print()
    print(f"manifest : {manifest_path}  ({len(all_rows)} rows)")
    print(f"metadata : {meta_path}")
    print(f"warnings : {warnings_path}  ({len(all_warnings)} entries)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/ybahaddo/Real-IAD/realiad_1024"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/realiad/manifest"))
    p.add_argument("--categories", nargs="*", default=None,
                   help="Restrict to these category names (default: all 30).")
    args = p.parse_args()
    build_manifest(args.data_root, args.out_dir, args.categories)


if __name__ == "__main__":
    main()
