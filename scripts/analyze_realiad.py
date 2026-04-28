"""Exploratory analysis of the Real-IAD dataset.

Reproduces the statistical views published in the Real-IAD paper (Wang et
al., 2024) from the local manifest produced by ``prepare_realiad.py``.
Every figure from the paper that the data itself can drive is emitted:

* Table 1 — cross-dataset summary written to ``dataset_card.md``.
* Fig 2(a) — anomaly/normal volume vs. BTAD / SSGD / MVTec / VisA.
* Fig 2(b) — defect area fraction histogram over 0..3%.
* Fig 2(c) — defect minimum-bounding-rectangle aspect-ratio histogram.
* Fig 2(d) — per-category normal/anomaly counts (grouped bars).
* Fig 2(e) — per-category defect breakdown stacked by semantic name.
* Fig 3   — multi-view showcase (5 views per row, several categories).
* Fig 4   — per-defect gallery (one row per semantic defect class).
* Fig 5(a/b) — object catalogs for complex and simple products.

Additional views that aren't in the paper but are useful for LRAD:
per-category grids (``samples/<cat>.png``), mask-overlay mosaic,
image-size scatter, per-category RGB pixel stats, view coverage.

Usage::

    python scripts/analyze_realiad.py \\
        --manifest outputs/realiad/manifest/manifest.csv \\
        --out-dir  outputs/realiad/analysis

Flags worth knowing:
    --max-mask-scan   cap on masks read for area/aspect-ratio stats
                      (default 4000; pass 0 to read all)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lrad.data.realiad import VIEWS, load_manifest, RealIADSample


# ---------------------------------------------------------------------------
# Constants drawn from the Real-IAD paper.
# ---------------------------------------------------------------------------

# Defect-code → semantic name mapping (paper Fig 2e legend).
DEFECT_NAMES: dict[str, str] = {
    "AK": "Pit",
    "BX": "Deformation",
    "CH": "Abrasion",
    "HS": "Scratch",
    "PS": "Damage",
    "QS": "Missing",
    "YW": "Foreign Objects",
    "ZW": "Contamination",
    "OK": "Normal",
}

DEFECT_ORDER: list[str] = [
    "Pit", "Deformation", "Abrasion", "Scratch",
    "Damage", "Missing", "Foreign Objects", "Contamination",
]

DEFECT_COLOURS: dict[str, str] = {
    "Pit":              "#4c78a8",
    "Deformation":      "#f58518",
    "Abrasion":         "#54a24b",
    "Scratch":          "#e45756",
    "Damage":           "#72b7b2",
    "Missing":          "#b279a2",
    "Foreign Objects":  "#9d755d",
    "Contamination":    "#bab0ac",
}

# Paper Fig 5: 12 "complex" (mixed material / complex geometry) objects.
COMPLEX_CATEGORIES: tuple[str, ...] = (
    "audiojack", "pcb", "phone_battery", "sim_card_set", "switch",
    "terminalblock", "toothbrush", "toy", "transistor1", "usb",
    "usb_adaptor", "zipper",
)

# Reference numbers from Table 1 of the paper (for Fig 2a).
OTHER_DATASETS: dict[str, dict] = {
    "BTAD":     {"normal": 952,  "anomaly": 392,  "classes": 3},
    "SSGD":     {"normal": 0,    "anomaly": 2504, "classes": 1},
    "MVTec AD": {"normal": 4096, "anomaly": 1258, "classes": 15},
    "VisA":     {"normal": 9621, "anomaly": 1200, "classes": 12},
}


# ---------------------------------------------------------------------------
# Zip / image IO helpers.
# ---------------------------------------------------------------------------

class _ZipPool:
    def __init__(self) -> None:
        self._cache: dict[str, zipfile.ZipFile] = {}

    def get(self, path: str) -> zipfile.ZipFile:
        zf = self._cache.get(path)
        if zf is None:
            zf = zipfile.ZipFile(path, "r")
            self._cache[path] = zf
        return zf

    def close(self) -> None:
        for zf in self._cache.values():
            zf.close()
        self._cache.clear()


def _load_image(pool: _ZipPool, zip_path: str, member: str,
                resize: int | None = None) -> np.ndarray:
    with pool.get(zip_path).open(member) as fh:
        img = Image.open(io.BytesIO(fh.read())).convert("RGB")
    if resize:
        img = img.resize((resize, resize))
    return np.asarray(img)


def _load_mask(pool: _ZipPool, zip_path: str, member: str,
               resize: int | None = None) -> np.ndarray:
    with pool.get(zip_path).open(member) as fh:
        m = Image.open(io.BytesIO(fh.read())).convert("L")
    if resize:
        m = m.resize((resize, resize), Image.NEAREST)
    return np.asarray(m)


def _mask_bbox_stats(mask: np.ndarray) -> tuple[float, float] | None:
    """Return (area_fraction, bbox_aspect_ratio) for a binary mask."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area_frac = float(ys.size) / float(mask.size)
    ar = max(h, w) / max(min(h, w), 1)
    return area_frac, ar


# ---------------------------------------------------------------------------
# Aggregations over the manifest.
# ---------------------------------------------------------------------------

def summarise_manifest(samples: list[RealIADSample]) -> dict:
    per_cat_ok: Counter = Counter()
    per_cat_ng: Counter = Counter()
    # Paper-style per-view counts (Table 1 / Fig 2a):
    # * visible anomaly = NG view that has a mask (defect actually visible)
    # * visible normal  = OK view + NG view without a mask
    per_cat_visible_normal: Counter = Counter()
    per_cat_visible_anomaly: Counter = Counter()

    per_cat_defect: defaultdict[str, Counter] = defaultdict(Counter)
    per_cat_defect_named: defaultdict[str, Counter] = defaultdict(Counter)
    per_cat_views: defaultdict[str, Counter] = defaultdict(Counter)
    per_cat_masks: Counter = Counter()
    sample_keys: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    ng_view_has_mask: defaultdict[str, Counter] = defaultdict(Counter)

    for s in samples:
        if s.label == 0:
            per_cat_ok[s.category] += 1
            per_cat_visible_normal[s.category] += 1
        else:
            per_cat_ng[s.category] += 1
            per_cat_defect[s.category][s.defect_type] += 1
            per_cat_defect_named[s.category][DEFECT_NAMES.get(s.defect_type,
                                                              s.defect_type)] += 1
            ng_view_has_mask[s.view]["mask" if s.mask_member else "no_mask"] += 1
            if s.mask_member:
                per_cat_visible_anomaly[s.category] += 1
            else:
                per_cat_visible_normal[s.category] += 1
        per_cat_views[s.category][s.view] += 1
        if s.mask_member:
            per_cat_masks[s.category] += 1
        sample_keys[(s.category, s.defect_type, s.sample_id)].add(s.view)

    view_counts = Counter(len(v) for v in sample_keys.values())
    defect_vocab = sorted({s.defect_type for s in samples if s.label == 1})

    return {
        "n_rows": len(samples),
        "categories": sorted({s.category for s in samples}),
        "defect_vocab": defect_vocab,
        "defect_vocab_named": [DEFECT_NAMES.get(d, d) for d in defect_vocab],
        "per_cat_ok": dict(per_cat_ok),
        "per_cat_ng": dict(per_cat_ng),
        "per_cat_visible_normal": dict(per_cat_visible_normal),
        "per_cat_visible_anomaly": dict(per_cat_visible_anomaly),
        "per_cat_defect": {c: dict(v) for c, v in per_cat_defect.items()},
        "per_cat_defect_named": {c: dict(v) for c, v in per_cat_defect_named.items()},
        "per_cat_views": {c: dict(v) for c, v in per_cat_views.items()},
        "per_cat_masks": dict(per_cat_masks),
        "views_per_sample_hist": dict(view_counts),
        "ng_view_mask_rate": {v: dict(c) for v, c in ng_view_has_mask.items()},
        "n_unique_samples": len(sample_keys),
    }


def collect_mask_stats(samples: list[RealIADSample], pool: _ZipPool,
                       max_masks: int, mask_resize: int = 256) -> list[dict]:
    """For each (subsampled) NG sample with a mask, compute area/aspect ratio."""
    ng = [s for s in samples if s.label == 1 and s.mask_member]
    if max_masks and len(ng) > max_masks:
        ng = random.sample(ng, max_masks)
    out: list[dict] = []
    for s in ng:
        mask = _load_mask(pool, s.zip_path, s.mask_member, resize=mask_resize)
        stats = _mask_bbox_stats(mask)
        if stats is None:
            continue
        area, ar = stats
        out.append({
            "category": s.category,
            "defect_code": s.defect_type,
            "defect_name": DEFECT_NAMES.get(s.defect_type, s.defect_type),
            "area_frac": area,
            "aspect_ratio": ar,
        })
    return out


# ---------------------------------------------------------------------------
# Figure builders (existing analyses).
# ---------------------------------------------------------------------------

def plot_counts_stacked(stats: dict, out_path: Path) -> None:
    cats = sorted(stats["categories"])
    ok = [stats["per_cat_ok"].get(c, 0) for c in cats]
    ng = [stats["per_cat_ng"].get(c, 0) for c in cats]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(cats, ok, label="OK (normal)", color="#4C9AFF")
    ax.bar(cats, ng, bottom=ok, label="NG (defective)", color="#E55934")
    ax.set_ylabel("image count")
    ax.set_title("Real-IAD: OK vs NG image counts per category (stacked)")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_defect_matrix(stats: dict, out_path: Path) -> None:
    cats = sorted(stats["categories"])
    defects = stats["defect_vocab"]
    mat = np.zeros((len(cats), len(defects)), dtype=int)
    for i, c in enumerate(cats):
        row = stats["per_cat_defect"].get(c, {})
        for j, d in enumerate(defects):
            mat[i, j] = row.get(d, 0)
    labels = [f"{d}\n({DEFECT_NAMES.get(d, d)})" for d in defects]
    fig, ax = plt.subplots(figsize=(1.1 * max(6, len(defects)) + 2, 0.35 * len(cats) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(defects)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_title("NG image counts per (category, defect code)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, mat[i, j], ha="center", va="center",
                        color="white" if mat[i, j] < mat.max() / 2 else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_views_coverage(stats: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    hist = stats["views_per_sample_hist"]
    xs = sorted(hist)
    ys = [hist[x] for x in xs]
    axes[0].bar([str(x) for x in xs], ys, color="#4C9AFF")
    axes[0].set_xlabel("views per physical sample")
    axes[0].set_ylabel("# unique samples")
    axes[0].set_title("View completeness")

    mask_rate = []
    for v in VIEWS:
        row = stats["ng_view_mask_rate"].get(v, {})
        total = row.get("mask", 0) + row.get("no_mask", 0)
        mask_rate.append(row.get("mask", 0) / total if total else 0.0)
    axes[1].bar(list(VIEWS), mask_rate, color="#E55934")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction with mask")
    axes[1].set_title("NG samples: mask availability per view")
    for i, v in enumerate(mask_rate):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_image_size_hist(samples: list[RealIADSample], pool: _ZipPool,
                         out_path: Path, n: int = 400) -> dict:
    picks = random.sample(samples, min(n, len(samples)))
    sizes = []
    for s in picks:
        with pool.get(s.zip_path).open(s.image_member) as fh:
            img = Image.open(io.BytesIO(fh.read()))
            sizes.append(img.size)
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(widths, heights, alpha=0.4)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title(f"Image dimensions (sampled {len(picks)} images)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"widths": Counter(widths), "heights": Counter(heights)}


def plot_pixel_stats(samples: list[RealIADSample], pool: _ZipPool,
                     out_path: Path, n_per_cat: int = 40) -> dict:
    by_cat: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        if s.label == 0:
            by_cat[s.category].append(s)
    stats: dict[str, dict[str, list[float]]] = {}
    for cat, bucket in by_cat.items():
        picks = random.sample(bucket, min(n_per_cat, len(bucket)))
        means, stds = [], []
        for s in picks:
            arr = _load_image(pool, s.zip_path, s.image_member, resize=128)
            arr = arr.astype(np.float32) / 255.0
            means.append(arr.reshape(-1, 3).mean(axis=0))
            stds.append(arr.reshape(-1, 3).std(axis=0))
        mean = np.stack(means).mean(axis=0)
        std = np.stack(stds).mean(axis=0)
        stats[cat] = {"mean": mean.tolist(), "std": std.tolist()}

    cats = sorted(stats)
    if not cats:
        print("[pixel_stats] no OK samples found — skipping pixel stats plot")
        return {}
    means = np.array([stats[c]["mean"] for c in cats])
    stds = np.array([stats[c]["std"] for c in cats])
    if means.ndim == 1:
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, data, title in [(axes[0], means, "per-channel mean"),
                             (axes[1], stds, "per-channel std")]:
        x = np.arange(len(cats))
        ax.bar(x - 0.25, data[:, 0], 0.25, color="#e45756", label="R")
        ax.bar(x, data[:, 1], 0.25, color="#54a24b", label="G")
        ax.bar(x + 0.25, data[:, 2], 0.25, color="#4c78a8", label="B")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=75)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Per-category RGB pixel statistics (OK samples, 128×128 downscale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return stats


def plot_category_grids(samples: list[RealIADSample], pool: _ZipPool,
                         out_dir: Path, thumb: int = 192) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cat: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    for cat, bucket in sorted(by_cat.items()):
        by_group: defaultdict[tuple[int, str], defaultdict[str, list[RealIADSample]]] = defaultdict(lambda: defaultdict(list))
        for s in bucket:
            by_group[(s.label, s.defect_type)][s.sample_id].append(s)

        rows: list[tuple[str, dict[str, RealIADSample]]] = []
        ok_groups = by_group.get((0, "OK"), {})
        if ok_groups:
            sid = max(ok_groups, key=lambda k: len(ok_groups[k]))
            row = {s.view: s for s in ok_groups[sid]}
            rows.append((f"OK / {sid}", row))

        for (lbl, defect), groups in sorted(by_group.items()):
            if lbl == 0:
                continue
            sid = max(groups, key=lambda k: sum(1 for s in groups[k] if s.mask_member)
                                      or len(groups[k]))
            row = {s.view: s for s in groups[sid]}
            name = DEFECT_NAMES.get(defect, defect)
            rows.append((f"NG / {defect} ({name}) / {sid}", row))

        if not rows:
            continue

        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, len(VIEWS),
                                  figsize=(len(VIEWS) * 2.3, n_rows * 2.3))
        if n_rows == 1:
            axes = np.array([axes])
        fig.suptitle(f"{cat} — one sample per row, 5 camera views per column",
                     fontsize=14)
        for i, (label, row) in enumerate(rows):
            for j, v in enumerate(VIEWS):
                ax = axes[i, j]
                s = row.get(v)
                if s is None:
                    ax.set_facecolor("#dddddd")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
                ax.imshow(arr)
                if s.label == 1 and s.mask_member:
                    mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
                    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                    overlay[..., 0] = 1.0
                    overlay[..., 3] = (mask > 0).astype(np.float32) * 0.45
                    ax.imshow(overlay)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0,
                                  ha="right", va="center")
                if i == 0:
                    ax.set_title(v, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"{cat}.png", dpi=100)
        plt.close(fig)


def plot_mask_overlay_mosaic(samples: list[RealIADSample], pool: _ZipPool,
                              out_path: Path, n: int = 24, thumb: int = 224) -> None:
    ng = [s for s in samples if s.label == 1 and s.mask_member]
    if not ng:
        return
    picks = random.sample(ng, min(n, len(ng)))
    cols = 6
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_2d(axes)
    for idx, s in enumerate(picks):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
        mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
        ax.imshow(arr)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = (mask > 0).astype(np.float32) * 0.5
        ax.imshow(overlay)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s.category}/{DEFECT_NAMES.get(s.defect_type, s.defect_type)}/{s.view}",
                     fontsize=8)
    for idx in range(len(picks), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Random NG samples with defect-mask overlay", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paper-style figures (Fig 2, Fig 3, Fig 4, Fig 5).
# ---------------------------------------------------------------------------

def plot_dataset_comparison(stats: dict, out_path: Path) -> None:
    """Paper Fig 2(a): anomaly vs. normal image counts across public IAD datasets.

    Uses paper semantics for Real-IAD: anomaly = NG views with visible mask,
    normal = OK views + NG views where the defect isn't visible from that angle.
    """
    names = list(OTHER_DATASETS.keys()) + ["Real-IAD"]
    normal = [OTHER_DATASETS[n]["normal"] for n in OTHER_DATASETS] + \
             [sum(stats["per_cat_visible_normal"].values())]
    anomaly = [OTHER_DATASETS[n]["anomaly"] for n in OTHER_DATASETS] + \
              [sum(stats["per_cat_visible_anomaly"].values())]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    x = np.arange(len(names))
    b1 = ax.bar(x - width / 2, anomaly, width, label="Anomaly", color="#4c78a8")
    b2 = ax.bar(x + width / 2, normal, width, label="Normal", color="#f58518")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("image count (paper definition: anomaly = view w/ visible mask)")
    ax.set_title("Cross-dataset volume comparison (paper Fig 2a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_counts_grouped(stats: dict, out_path: Path) -> None:
    """Paper Fig 2(d): per-category normal vs anomaly grouped bars (paper semantics)."""
    cats = sorted(stats["categories"],
                  key=lambda c: -(stats["per_cat_visible_normal"].get(c, 0)
                                   + stats["per_cat_visible_anomaly"].get(c, 0)))
    normal = [stats["per_cat_visible_normal"].get(c, 0) for c in cats]
    anomaly = [stats["per_cat_visible_anomaly"].get(c, 0) for c in cats]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(cats))
    width = 0.4
    ax.bar(x - width / 2, normal, width, label="normal", color="#4c78a8")
    ax.bar(x + width / 2, anomaly, width, label="anomaly", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=75)
    ax.set_ylabel("image count (paper definition)")
    ax.set_title("Per-category normal vs anomaly counts (paper Fig 2d)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_defect_types_stacked_named(stats: dict, out_path: Path) -> None:
    """Paper Fig 2(e): per-category stacked defect counts with semantic names."""
    cats = sorted(stats["categories"])
    defects = DEFECT_ORDER

    mat = np.zeros((len(defects), len(cats)), dtype=int)
    for j, c in enumerate(cats):
        row = stats["per_cat_defect_named"].get(c, {})
        for i, d in enumerate(defects):
            mat[i, j] = row.get(d, 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(cats), dtype=int)
    x = np.arange(len(cats))
    for i, d in enumerate(defects):
        ax.bar(x, mat[i], bottom=bottom, color=DEFECT_COLOURS[d], label=d)
        bottom = bottom + mat[i]
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=75)
    ax.set_ylabel("anomaly image count")
    ax.set_title("Defect type breakdown per category (paper Fig 2e)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_defect_area_paper_style(mask_stats: list[dict], out_path: Path) -> dict:
    """Paper Fig 2(b): defect area fraction histogram (0..~3%)."""
    if not mask_stats:
        return {}
    areas = np.array([m["area_frac"] * 100 for m in mask_stats])  # %
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # cap at 3% like the paper, note overflow count.
    bins = np.linspace(0, 3.0, 30)
    in_range = areas[areas <= 3.0]
    ax.hist(in_range, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("defect area as % of image")
    ax.set_ylabel("# masks")
    over = int((areas > 3.0).sum())
    ax.set_title(
        f"Defect area distribution (paper Fig 2b) — "
        f"n={len(areas)}, {over} masks > 3% clipped from view"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {
        "mean_pct": float(areas.mean()),
        "median_pct": float(np.median(areas)),
        "p95_pct": float(np.percentile(areas, 95)),
        "max_pct": float(areas.max()),
        "n": len(areas),
    }


def plot_defect_aspect_ratio(mask_stats: list[dict], out_path: Path) -> dict:
    """Paper Fig 2(c): histogram of bbox max/min aspect ratio of defects."""
    if not mask_stats:
        return {}
    ar = np.array([m["aspect_ratio"] for m in mask_stats])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    bins_lin = np.linspace(1, 20, 40)
    axes[0].hist(np.clip(ar, 1, 20), bins=bins_lin, color="#4c78a8", edgecolor="white")
    axes[0].set_xlabel("bbox aspect ratio (max/min)")
    axes[0].set_ylabel("# masks")
    over = int((ar > 20).sum())
    axes[0].set_title(f"Linear scale (ar ≤ 20); {over} clipped")

    log_ar = np.log10(ar)
    axes[1].hist(log_ar, bins=40, color="#f58518", edgecolor="white")
    axes[1].set_xlabel("log10(aspect ratio)")
    axes[1].set_ylabel("# masks")
    axes[1].set_title("Log scale — highlights the long tail (paper Fig 2c)")

    fig.suptitle(f"Defect minimum-bounding-rect aspect ratio (n={len(ar)})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {
        "mean": float(ar.mean()),
        "median": float(np.median(ar)),
        "p95": float(np.percentile(ar, 95)),
        "max": float(ar.max()),
        "n": len(ar),
    }


def plot_multiview_showcase(samples: list[RealIADSample], pool: _ZipPool,
                             out_path: Path, categories: list[str] | None = None,
                             thumb: int = 192) -> None:
    """Paper Fig 3: multi-view collection. Shows the 5 camera views per sample,
    one OK row and one NG row per chosen category, with mask overlay on NG.
    """
    if categories is None:
        categories = ["button_battery", "porcelain_doll", "audiojack", "tape"]

    by_cat: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    rows: list[tuple[str, dict[str, RealIADSample]]] = []
    for cat in categories:
        bucket = by_cat.get(cat, [])
        if not bucket:
            continue

        ok_ids: defaultdict[str, list[RealIADSample]] = defaultdict(list)
        ng_ids: defaultdict[tuple[str, str], list[RealIADSample]] = defaultdict(list)
        for s in bucket:
            if s.label == 0:
                ok_ids[s.sample_id].append(s)
            else:
                ng_ids[(s.defect_type, s.sample_id)].append(s)

        if ok_ids:
            sid = max(ok_ids, key=lambda k: len(ok_ids[k]))
            rows.append((f"{cat}\nOK", {s.view: s for s in ok_ids[sid]}))
        if ng_ids:
            key = max(ng_ids, key=lambda k: sum(1 for s in ng_ids[k] if s.mask_member)
                                            or len(ng_ids[k]))
            defect = DEFECT_NAMES.get(key[0], key[0])
            rows.append((f"{cat}\nNG ({defect})", {s.view: s for s in ng_ids[key]}))

    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, len(VIEWS),
                              figsize=(len(VIEWS) * 2.4, n_rows * 2.4))
    axes = np.atleast_2d(axes)
    fig.suptitle("Multi-view showcase: 1 top + 4 × 45° cameras (paper Fig 3)",
                 fontsize=13)
    for i, (label, row) in enumerate(rows):
        for j, v in enumerate(VIEWS):
            ax = axes[i, j]
            s = row.get(v)
            if s is None:
                ax.set_facecolor("#dddddd")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
            ax.imshow(arr)
            if s.label == 1 and s.mask_member:
                mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 3] = (mask > 0).astype(np.float32) * 0.5
                ax.imshow(overlay)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title(v + (" (top)" if v == "C1" else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_defect_gallery(samples: list[RealIADSample], pool: _ZipPool,
                         out_path: Path, per_defect: int = 8,
                         thumb: int = 192) -> None:
    """Paper Fig 4: one row per semantic defect class, drawing from mixed categories."""
    by_name: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        if s.label == 1 and s.mask_member:
            by_name[DEFECT_NAMES.get(s.defect_type, s.defect_type)].append(s)

    rows_data: list[tuple[str, list[RealIADSample]]] = []
    for name in DEFECT_ORDER:
        bucket = by_name.get(name, [])
        if not bucket:
            continue
        # try to spread over categories
        random.shuffle(bucket)
        seen: set[str] = set()
        chosen: list[RealIADSample] = []
        for s in bucket:
            if s.category in seen:
                continue
            chosen.append(s); seen.add(s.category)
            if len(chosen) >= per_defect:
                break
        while len(chosen) < per_defect and len(chosen) < len(bucket):
            chosen.append(bucket[len(chosen)])
        rows_data.append((name, chosen))

    if not rows_data:
        return

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, per_defect,
                              figsize=(per_defect * 2.0, n_rows * 2.0))
    axes = np.atleast_2d(axes)
    fig.suptitle("Per-defect gallery — rows = semantic defect class (paper Fig 4)",
                 fontsize=13)
    for i, (name, chosen) in enumerate(rows_data):
        for j in range(per_defect):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(chosen):
                ax.axis("off")
                continue
            s = chosen[j]
            arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
            mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
            ax.imshow(arr)
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = (mask > 0).astype(np.float32) * 0.5
            ax.imshow(overlay)
            ax.set_xlabel(s.category, fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=11, rotation=0, ha="right", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _pick_sample_views(bucket: list[RealIADSample], n_views: int) -> list[RealIADSample]:
    """Return n_views image entries from a single physical sample, preferring
    samples that have all 5 views."""
    by_sample: defaultdict[tuple[int, str, str], list[RealIADSample]] = defaultdict(list)
    for s in bucket:
        by_sample[(s.label, s.defect_type, s.sample_id)].append(s)
    sid = max(by_sample, key=lambda k: len(by_sample[k]))
    rows = sorted(by_sample[sid], key=lambda s: VIEWS.index(s.view) if s.view in VIEWS else 99)
    return rows[:n_views]


def plot_object_catalog_complex(samples: list[RealIADSample], pool: _ZipPool,
                                  out_path: Path, thumb: int = 160) -> None:
    """Paper Fig 5(a): 12 complex objects, 5 views each."""
    by_cat: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    cats = [c for c in COMPLEX_CATEGORIES if c in by_cat]
    if not cats:
        return
    n_rows = len(cats)
    fig, axes = plt.subplots(n_rows, 5, figsize=(5 * 2.0, n_rows * 2.0))
    axes = np.atleast_2d(axes)
    fig.suptitle("Complex products — all 5 views per object (paper Fig 5a)", fontsize=13)
    for i, cat in enumerate(cats):
        row = _pick_sample_views(by_cat[cat], 5)
        for j in range(5):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(row):
                ax.axis("off")
                continue
            s = row[j]
            arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
            ax.imshow(arr)
            if s.label == 1 and s.mask_member:
                mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 3] = (mask > 0).astype(np.float32) * 0.5
                ax.imshow(overlay)
            if j == 0:
                ax.set_ylabel(cat, fontsize=10, rotation=0, ha="right", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_object_catalog_simple(samples: list[RealIADSample], pool: _ZipPool,
                                 out_path: Path, thumb: int = 160) -> None:
    """Paper Fig 5(b): 18 simple objects, 2 views each."""
    by_cat: defaultdict[str, list[RealIADSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)
    simple = [c for c in sorted(by_cat) if c not in set(COMPLEX_CATEGORIES)]
    if not simple:
        return
    n_rows = len(simple)
    fig, axes = plt.subplots(n_rows, 2, figsize=(2 * 2.2, n_rows * 2.2))
    axes = np.atleast_2d(axes)
    fig.suptitle("Simple products — 2 representative views per object (paper Fig 5b)",
                 fontsize=13)
    for i, cat in enumerate(simple):
        row = _pick_sample_views(by_cat[cat], 2)
        for j in range(2):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(row):
                ax.axis("off")
                continue
            s = row[j]
            arr = _load_image(pool, s.zip_path, s.image_member, resize=thumb)
            ax.imshow(arr)
            if s.label == 1 and s.mask_member:
                mask = _load_mask(pool, s.zip_path, s.mask_member, resize=thumb)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 3] = (mask > 0).astype(np.float32) * 0.5
                ax.imshow(overlay)
            if j == 0:
                ax.set_ylabel(cat, fontsize=10, rotation=0, ha="right", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_dataset_card(stats: dict, mask_area_stats: dict, aspect_stats: dict,
                        out_path: Path) -> None:
    """Table 1-style markdown summary."""
    total_ok = sum(stats["per_cat_ok"].values())
    total_ng = sum(stats["per_cat_ng"].values())
    total = total_ok + total_ng
    visible_normal = sum(stats["per_cat_visible_normal"].values())
    visible_anomaly = sum(stats["per_cat_visible_anomaly"].values())
    lines: list[str] = []
    lines.append("# Real-IAD — dataset card\n")
    lines.append("Generated from the local manifest. Cross-dataset rows come from Table 1 of Wang et al., 2024.\n")
    lines.append("## Volume comparison (paper Table 1 / Fig 2a semantics)\n")
    lines.append("*Paper-style split: an image is 'Anomaly' only when a visible mask exists for that view; "
                  "NG images captured from an angle where the defect isn't visible count as 'Normal'.*\n")
    lines.append("| Dataset | Classes | Normal | Anomaly | Total |")
    lines.append("|---|---|---|---|---|")
    for name, info in OTHER_DATASETS.items():
        lines.append(f"| {name} | {info['classes']} | {info['normal']:,} | "
                      f"{info['anomaly']:,} | {info['normal'] + info['anomaly']:,} |")
    lines.append(f"| **Real-IAD (local, paper-style)** | **{len(stats['categories'])}** | "
                  f"**{visible_normal:,}** | **{visible_anomaly:,}** | **{total:,}** |\n")
    lines.append(f"Unique physical samples: **{stats['n_unique_samples']:,}**, "
                  f"each with {len(VIEWS)} views (C1..C5).\n")
    lines.append("## Folder-based split (used for LRAD training labels)\n")
    lines.append("*Every view of a defective physical sample keeps label=1, regardless of mask availability.*\n")
    lines.append(f"- OK views (folder = `<cat>/OK/`) : **{total_ok:,}**")
    lines.append(f"- NG views (folder = `<cat>/NG/`) : **{total_ng:,}**")
    lines.append(f"- Of those NG views, **{visible_anomaly:,}** have a segmentation mask "
                  f"(defect visible from that angle).\n")
    lines.append("## Defect vocabulary\n")
    lines.append("| Code | Semantic name |")
    lines.append("|---|---|")
    for code, name in DEFECT_NAMES.items():
        if code == "OK":
            continue
        lines.append(f"| {code} | {name} |")
    lines.append("")
    if mask_area_stats:
        lines.append("## Defect area (% of image)\n")
        lines.append(f"- mean   : {mask_area_stats['mean_pct']:.3f}%")
        lines.append(f"- median : {mask_area_stats['median_pct']:.3f}%")
        lines.append(f"- p95    : {mask_area_stats['p95_pct']:.3f}%")
        lines.append(f"- max    : {mask_area_stats['max_pct']:.3f}%")
        lines.append(f"- n masks: {mask_area_stats['n']:,}\n")
    if aspect_stats:
        lines.append("## Defect bbox aspect ratio (max/min)\n")
        lines.append(f"- mean   : {aspect_stats['mean']:.2f}")
        lines.append(f"- median : {aspect_stats['median']:.2f}")
        lines.append(f"- p95    : {aspect_stats['p95']:.2f}")
        lines.append(f"- max    : {aspect_stats['max']:.2f}\n")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("outputs/realiad/manifest/manifest.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/realiad/analysis"))
    p.add_argument("--samples-per-category", type=int, default=20)
    p.add_argument("--max-pixel-samples", type=int, default=300)
    p.add_argument("--max-mask-scan", type=int, default=4000,
                   help="Max # masks to read for area/aspect stats. 0 = all.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.manifest}")
    samples = load_manifest(args.manifest)
    print(f"  {len(samples)} rows")

    stats = summarise_manifest(samples)

    pool = _ZipPool()
    try:
        print("Figures over manifest only (no image decode)")
        plot_counts_stacked(stats, args.out_dir / "counts_ok_ng.png")
        plot_counts_grouped(stats, args.out_dir / "counts_grouped.png")
        plot_defect_matrix(stats, args.out_dir / "defect_type_matrix.png")
        plot_defect_types_stacked_named(stats, args.out_dir / "defect_types_stacked_named.png")
        plot_views_coverage(stats, args.out_dir / "views_coverage.png")
        plot_dataset_comparison(stats, args.out_dir / "dataset_comparison.png")

        print("Sampling images for size / pixel-stats")
        sizes = plot_image_size_hist(samples, pool,
                                      args.out_dir / "image_size_hist.png",
                                      n=args.max_pixel_samples)
        stats["image_sizes"] = {
            "widths": {str(k): v for k, v in sizes["widths"].items()},
            "heights": {str(k): v for k, v in sizes["heights"].items()},
        }
        pix = plot_pixel_stats(samples, pool,
                               args.out_dir / "pixel_stats.png",
                               n_per_cat=args.samples_per_category)
        stats["pixel_stats"] = pix

        print(f"Scanning masks (up to {args.max_mask_scan or 'all'}) for area + aspect ratio")
        mask_stats = collect_mask_stats(samples, pool,
                                         max_masks=args.max_mask_scan)
        stats["mask_scan_n"] = len(mask_stats)
        print(f"  collected {len(mask_stats)} mask measurements")

        area_stats = plot_defect_area_paper_style(
            mask_stats, args.out_dir / "defect_area_hist.png")
        aspect_stats = plot_defect_aspect_ratio(
            mask_stats, args.out_dir / "defect_aspect_ratio_hist.png")
        stats["defect_area_stats"] = area_stats
        stats["defect_aspect_ratio_stats"] = aspect_stats

        print("Paper-style multi-view / defect / object figures")
        plot_multiview_showcase(samples, pool,
                                 args.out_dir / "multiview_showcase.png")
        plot_defect_gallery(samples, pool,
                             args.out_dir / "defect_gallery.png")
        plot_object_catalog_complex(samples, pool,
                                     args.out_dir / "object_catalog_complex.png")
        plot_object_catalog_simple(samples, pool,
                                    args.out_dir / "object_catalog_simple.png")

        print("Per-category sample grids + mask-overlay mosaic")
        plot_category_grids(samples, pool, args.out_dir / "samples")
        plot_mask_overlay_mosaic(samples, pool,
                                  args.out_dir / "mask_overlays.png")
    finally:
        pool.close()

    with open(args.out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    write_dataset_card(stats,
                       stats.get("defect_area_stats", {}),
                       stats.get("defect_aspect_ratio_stats", {}),
                       args.out_dir / "dataset_card.md")

    print()
    print(f"Artifacts written under {args.out_dir}")


if __name__ == "__main__":
    main()
