#!/usr/bin/env python3
"""Fetch and extract the MVTec AD dataset.

MVTec AD is released by MVTec Software GmbH under CC BY-NC-SA 4.0 —
**non-commercial research use only**. By running this script you accept
those terms; see https://www.mvtec.com/company/research/datasets/mvtec-ad.

The archive is ~4.9 GB and expands to ~5.3 GB, so the two together need
roughly 10 GB free while extracting.

Usage:
    python scripts/download_mvtec.py                    # into ./data
    python scripts/download_mvtec.py --root /path/to/data
    python scripts/download_mvtec.py --verify-only      # check an existing copy
    python scripts/download_mvtec.py --url https://...  # if the link rotated

An interrupted transfer leaves a .tar.xz.part that the next run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.mvtec import (  # noqa: E402
    DEFAULT_DIRNAME,
    MVTEC_CATEGORIES,
    category_root,
)

# MVTec rotates its mydrive.ch share links without notice — the previous one
# started answering 404. Override with --url (or $LRAD_MVTEC_URL) if this one
# dies too; the current link can be read off anomalib's MVTecAD datamodule.
MVTEC_URL = (
    "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
    "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
)
MVTEC_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
ARCHIVE_NAME = "mvtec_anomaly_detection.tar.xz"

# Image counts of the official release, per category, as
# (train/good, test total). Used by --verify to catch a truncated download
# or a partial extraction — both of which otherwise surface much later as
# a mysteriously easy AUROC.
EXPECTED: dict[str, tuple[int, int]] = {
    "bottle": (209, 83), "cable": (224, 150), "capsule": (219, 132),
    "carpet": (280, 117), "grid": (264, 78), "hazelnut": (391, 110),
    "leather": (245, 124), "metal_nut": (220, 115), "pill": (267, 167),
    "screw": (320, 160), "tile": (230, 117), "toothbrush": (60, 42),
    "transistor": (213, 100), "wood": (247, 79), "zipper": (240, 151),
}


def _progress(done: int, total: int) -> None:
    if total <= 0:
        return
    done = min(done, total)
    pct = 100.0 * done / total
    sys.stdout.write(
        f"\r  {done / 2**30:.2f} / {total / 2**30:.2f} GiB  ({pct:5.1f}%)"
    )
    sys.stdout.flush()


def download(url: str, part: Path) -> str:
    """Stream `url` into `part`, resuming it if it already exists.

    Returns the sha256 of the bytes written. Five GiB over a flaky link is
    worth resuming rather than restarting, so an existing .part is re-hashed
    and continued with a Range request; a server that ignores the range
    (answering 200 instead of 206) restarts the file from scratch.
    """
    sha = hashlib.sha256()
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if have:
        print(f"  resuming at {have / 2**30:.2f} GiB")
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req) as resp:
        resumed = resp.status == 206
        if have and not resumed:
            print("  server ignored the range request — restarting")
            have = 0
        if have:
            with part.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    sha.update(chunk)
        total = int(resp.headers.get("Content-Length", 0)) + have
        done = have
        with part.open("ab" if have else "wb") as fh:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                fh.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                _progress(done, total)
    print()
    return sha.hexdigest()


def verify(root: Path) -> bool:
    """Check every category is present with the expected image counts."""
    ok = True
    for cat, (n_train, n_test) in EXPECTED.items():
        try:
            d = category_root(root, cat)
        except FileNotFoundError:
            print(f"  MISSING  {cat}")
            ok = False
            continue
        train = list((d / "train" / "good").glob("*.png"))
        test = [p for p in (d / "test").rglob("*.png")]
        masks = list((d / "ground_truth").rglob("*.png"))
        bad = len(train) != n_train or len(test) != n_test
        ok &= not bad
        print(
            f"  {'BAD  ' if bad else 'ok   '} {cat:<11} "
            f"train={len(train):>3}/{n_train:<3} "
            f"test={len(test):>3}/{n_test:<3} masks={len(masks):>3}"
        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=_ROOT / "data",
                    help="Parent directory; the dataset lands in "
                         f"<root>/{DEFAULT_DIRNAME}/ (default: ./data)")
    ap.add_argument("--keep-archive", action="store_true",
                    help="Do not delete the .tar.xz after extracting.")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only check an existing copy; download nothing.")
    ap.add_argument("--url", default=os.environ.get("LRAD_MVTEC_URL", MVTEC_URL),
                    help="Archive URL, for when MVTec rotates the share link "
                         "again (env: LRAD_MVTEC_URL).")
    ap.add_argument("--skip-hash-check", action="store_true",
                    help="Do not compare the download against the sha256 of "
                         "the official release (implied by --url).")
    args = ap.parse_args()

    root: Path = args.root
    target = root / DEFAULT_DIRNAME

    if args.verify_only:
        print(f"Verifying MVTec AD under {root} ...")
        raise SystemExit(0 if verify(root) else 1)

    root.mkdir(parents=True, exist_ok=True)

    if target.is_dir() and all(
        (target / c).is_dir() for c in MVTEC_CATEGORIES
    ):
        print(f"{target} already holds all 15 categories — verifying.")
        raise SystemExit(0 if verify(root) else 1)

    archive = root / ARCHIVE_NAME
    if not archive.exists():
        free = shutil.disk_usage(root).free
        if free < 11 * 2**30:
            print(
                f"WARNING: only {free / 2**30:.1f} GiB free under {root}; "
                "the archive plus its extraction need ~10 GiB.",
                file=sys.stderr,
            )
        print("MVTec AD is CC BY-NC-SA 4.0 — non-commercial research only.")
        print(f"Downloading ~4.9 GiB to {archive} ...")
        # Download to a .part file and rename only on success, so an
        # interrupted transfer can never be mistaken for a complete archive
        # on the next run.
        part = archive.with_suffix(archive.suffix + ".part")
        digest = download(args.url, part)
        if not args.skip_hash_check and args.url == MVTEC_URL:
            if digest != MVTEC_SHA256:
                print(
                    f"sha256 mismatch: got {digest}, expected {MVTEC_SHA256}.\n"
                    f"The partial download is kept at {part}; delete it before "
                    "retrying.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            print("  sha256 ok")
        part.rename(archive)
    else:
        print(f"Using existing archive {archive}")

    print(f"Extracting into {target} ...")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tf:
        # Python 3.12+ warns without an extraction filter; "data" rejects
        # absolute paths and symlinks escaping the destination.
        try:
            tf.extractall(target, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(target)

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed {archive}")

    print("Verifying ...")
    ok = verify(root)
    print("OK" if ok else "VERIFICATION FAILED — see the lines marked BAD")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
