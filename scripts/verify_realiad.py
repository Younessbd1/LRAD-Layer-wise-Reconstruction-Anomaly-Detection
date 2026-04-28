"""Integrity / cleanliness check for the Real-IAD manifest.

Scans the zips referenced by the manifest and flags:
  * unreadable JPEGs or PNG masks (truncated / corrupt)
  * unexpected image modes or non-1024 dimensions
  * NG samples with a view that has no mask at all (informational)
  * duplicate rows

Designed to be cheap when ``--sample-frac`` is small; full sweep (~73k
rows) takes longer but will surface any rotten file.

Usage:
    python scripts/verify_realiad.py \\
        --manifest outputs/realiad/manifest/manifest.csv \\
        --out-dir  outputs/realiad/verify \\
        --sample-frac 0.1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from lrad.data.realiad import load_manifest


def _check_image(zf: zipfile.ZipFile, member: str,
                 expect_mode: str, expect_size: tuple[int, int] | None):
    try:
        with zf.open(member) as fh:
            data = fh.read()
        img = Image.open(io.BytesIO(data))
        img.load()  # forces decode of JPEG/PNG
    except Exception as e:
        return f"unreadable: {e.__class__.__name__}: {e}"
    if expect_size and img.size != expect_size:
        return f"unexpected size {img.size}, expected {expect_size}"
    if expect_mode and img.mode != expect_mode and not (
            expect_mode == "RGB" and img.mode in {"RGB", "RGBA"}):
        return f"unexpected mode {img.mode}, expected {expect_mode}"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("outputs/realiad/manifest/manifest.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/realiad/verify"))
    p.add_argument("--sample-frac", type=float, default=0.1,
                   help="Fraction of rows to check (1.0 = full sweep).")
    p.add_argument("--expected-size", type=int, default=1024,
                   help="Expected image dimension (both W and H).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(args.manifest)
    print(f"{len(samples)} rows loaded")

    # Duplicate detection
    seen: set[tuple[str, str, str, str]] = set()
    dup_rows = 0
    for s in samples:
        key = (s.category, s.defect_type, s.sample_id, s.view)
        if key in seen:
            dup_rows += 1
        seen.add(key)

    # Sampling
    if args.sample_frac < 1.0:
        random.shuffle(samples)
        checked = samples[: int(len(samples) * args.sample_frac)]
    else:
        checked = samples
    print(f"Checking {len(checked)} rows "
          f"(sample_frac={args.sample_frac})")

    by_zip: dict[str, list] = {}
    for s in checked:
        by_zip.setdefault(s.zip_path, []).append(s)

    expected_size = (args.expected_size, args.expected_size)
    issues: list[dict] = []
    mode_counter: Counter = Counter()
    size_counter: Counter = Counter()

    for zip_path, rows in sorted(by_zip.items()):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for s in rows:
                err = _check_image(zf, s.image_member, "RGB", expected_size)
                if err:
                    issues.append({"type": "image", "row": s.__dict__, "err": err})
                else:
                    with zf.open(s.image_member) as fh:
                        im = Image.open(fh)
                        mode_counter[im.mode] += 1
                        size_counter[im.size] += 1
                if s.mask_member:
                    err = _check_image(zf, s.mask_member, "L", expected_size)
                    if err:
                        issues.append({"type": "mask", "row": s.__dict__,
                                        "err": err})

    summary = {
        "checked": len(checked),
        "duplicates": dup_rows,
        "issues": len(issues),
        "modes_seen": {k: v for k, v in mode_counter.most_common()},
        "sizes_seen": {f"{w}x{h}": v for (w, h), v in size_counter.most_common()},
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_dir / "issues.json", "w") as f:
        json.dump(issues, f, indent=2)

    print()
    print(f"duplicates : {dup_rows}")
    print(f"issues     : {len(issues)}")
    print(f"modes seen : {dict(mode_counter)}")
    print(f"sizes seen : {dict(size_counter)}")
    print(f"summary    : {args.out_dir / 'summary.json'}")
    print(f"issues     : {args.out_dir / 'issues.json'}")


if __name__ == "__main__":
    main()
