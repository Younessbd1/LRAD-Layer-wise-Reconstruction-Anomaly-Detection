"""Tests for the MVTec runner's schedule logic and the MVTec figures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

from lrad.mvtec import MVTEC_CATEGORIES  # noqa: E402
from lrad.plots import (  # noqa: E402
    plot_mvtec_curves,
    plot_mvtec_segmentation,
    plot_mvtec_vs_patchcore,
)
from run_mvtec import (  # noqa: E402
    PATCHCORE_10,
    _epochs_for,
    write_results_table,
)


# ---------------------------------------------------------------------------
# step-based schedule
# ---------------------------------------------------------------------------

def test_step_budget_is_equal_across_category_sizes():
    """The whole point: toothbrush (2 batches/ep) trains as much as
    hazelnut (13), which a fixed epoch count would not deliver."""
    steps = 8000
    got = {n: _epochs_for(steps, n, 60) * n for n in (2, 7, 9, 13)}
    for total in got.values():
        # ceil division overshoots by at most one epoch's worth of batches
        assert steps <= total < steps + 13


def test_fixed_epochs_would_not_be_equal():
    """Pins the bug the step schedule exists to prevent."""
    assert (60 * 13) / (60 * 2) == pytest.approx(6.5)


def test_epochs_fallback_when_no_step_budget():
    assert _epochs_for(None, 7, 60) == 60
    assert _epochs_for(0, 7, 60) == 60


def test_epochs_is_at_least_one():
    assert _epochs_for(1, 1000, 60) == 1


# ---------------------------------------------------------------------------
# results table
# ---------------------------------------------------------------------------

def _result(cat, img=0.99, pix=0.98, pro=0.93):
    return {"category": cat, "image_auroc": img, "pixel_auroc": pix,
            "pro": pro}


def test_results_table_warns_on_partial_sweeps(tmp_path):
    """A 2-category mean must not be silently compared to a 15-category one."""
    txt = write_results_table(
        [_result("bottle"), _result("screw")], tmp_path / "r.md",
    )
    assert "Only 2 of 15 categories" in txt
    assert "PatchCore on these same categories" in txt
    assert "bottle" in txt and "screw" in txt


def test_results_table_omits_the_warning_when_complete(tmp_path):
    txt = write_results_table(
        [_result(c) for c in MVTEC_CATEGORIES], tmp_path / "r.md",
    )
    assert "Only" not in txt
    assert "mean (15 cat.)" in txt


def test_results_table_renders_missing_metrics(tmp_path):
    """--no pixel metrics leaves those columns absent, not crashing."""
    txt = write_results_table(
        [{"category": "bottle", "image_auroc": 0.9}], tmp_path / "r.md",
    )
    assert "n/a" in txt


def test_patchcore_reference_covers_every_category():
    assert set(PATCHCORE_10) == set(MVTEC_CATEGORIES)
    for cat, vals in PATCHCORE_10.items():
        assert len(vals) == 3
        assert all(80.0 <= v <= 100.0 for v in vals), cat


# ---------------------------------------------------------------------------
# figures — smoke tests (they must produce a file, not a traceback)
# ---------------------------------------------------------------------------

def test_segmentation_panel_writes_a_figure(tmp_path):
    n, s = 3, 32
    imgs = torch.rand(n, 3, s, s)
    maps = np.random.default_rng(0).random((n, s, s)).astype(np.float32)
    masks = np.zeros((n, s, s), np.float32)
    masks[0, 4:12, 4:12] = 1
    out = tmp_path / "seg.png"
    plot_mvtec_segmentation(
        imgs, maps, masks, out,
        scores=[1.0, 2.0, 3.0], labels=["broken", "scratch", "good"],
    )
    assert out.exists() and out.stat().st_size > 0


def test_segmentation_panel_handles_an_all_nominal_batch(tmp_path):
    """No mask to contour must not break the overlay column."""
    out = tmp_path / "seg.png"
    plot_mvtec_segmentation(
        torch.rand(2, 3, 16, 16),
        np.random.default_rng(1).random((2, 16, 16)).astype(np.float32),
        np.zeros((2, 16, 16), np.float32),
        out,
    )
    assert out.exists()


def test_segmentation_panel_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot_mvtec_segmentation(
            torch.rand(0, 3, 8, 8), np.zeros((0, 8, 8)),
            np.zeros((0, 8, 8)), tmp_path / "x.png",
        )


def test_curves_figure_writes_with_and_without_pixel_data(tmp_path):
    roc = {"fpr": [0, 0.2, 1.0], "tpr": [0, 0.8, 1.0], "auroc": 0.9}
    pro_curve = {"fpr": list(np.linspace(0, 1, 20)),
                 "pro": list(np.linspace(0, 1, 20))}
    a = tmp_path / "full.png"
    plot_mvtec_curves(a, roc=roc, pro_curve=pro_curve, pixel_auroc=0.98,
                      pro=0.93, image_auroc=0.9)
    assert a.exists()

    b = tmp_path / "detection_only.png"
    plot_mvtec_curves(b, roc=roc, image_auroc=0.9)   # no pixel metrics
    assert b.exists()


def test_vs_patchcore_figure(tmp_path):
    out = tmp_path / "vs.png"
    plot_mvtec_vs_patchcore(
        [_result("bottle"), _result("screw", 0.90, 0.95, 0.80)],
        PATCHCORE_10, out,
    )
    assert out.exists() and out.stat().st_size > 0


def test_vs_patchcore_needs_a_shared_category(tmp_path):
    with pytest.raises(ValueError, match="no categories in common"):
        plot_mvtec_vs_patchcore(
            [_result("not_a_category")], PATCHCORE_10, tmp_path / "x.png",
        )
