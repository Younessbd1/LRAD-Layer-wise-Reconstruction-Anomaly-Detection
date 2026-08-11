"""Tests for pixel AUROC and the PRO metric.

The PRO cases pin down behaviour that has a known analytic answer, which is
what makes the metric trustworthy: a perfect localizer scores 1.0, a random
one scores ``fpr_limit / 2`` (0.15 at the conventional 0.30 limit, NOT 0.5
— PRO integrates the diagonal, it does not average it), and a localizer
that finds only the large regions is punished hard even though its pixel
AUROC stays high. That last case is the entire reason MVTec papers report
PRO alongside pixel AUROC.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.decoder import build_decoders
from lrad.model import build_model
from lrad.pixel_metrics import (
    COLLAPSE_RATIO_WARN,
    DEFAULT_FPR_LIMIT,
    _region_slices,
    collect_anomaly_maps,
    load_masks,
    pixel_auroc,
    pro_score,
)

H = W = 32


@pytest.fixture
def masks():
    """Three regions across four images: big, thin-wide, tiny, plus a nominal."""
    m = np.zeros((4, H, W), np.float32)
    m[0, 4:12, 4:12] = 1     # 64 px
    m[1, 20:24, 2:28] = 1    # 104 px
    m[2, 0:2, 0:2] = 1       # 4 px
    return m


# ---------------------------------------------------------------------------
# connected components
# ---------------------------------------------------------------------------

def test_regions_use_eight_connectivity():
    m = np.zeros((1, 8, 8), np.float32)
    m[0, 1:3, 1:3] = 1
    m[0, 3:5, 3:5] = 1       # touches the first only at a corner
    assert len(_region_slices(m)) == 1


def test_disjoint_blobs_are_separate_regions():
    m = np.zeros((1, 8, 8), np.float32)
    m[0, 1:3, 1:3] = 1
    m[0, 5:7, 5:7] = 1
    assert len(_region_slices(m)) == 2


def test_nominal_images_contribute_no_regions(masks):
    assert len(_region_slices(masks)) == 3   # image 3 is nominal


# ---------------------------------------------------------------------------
# pixel AUROC
# ---------------------------------------------------------------------------

def test_pixel_auroc_perfect_and_inverted(masks):
    assert pixel_auroc(masks * 10.0, masks) == pytest.approx(1.0)
    assert pixel_auroc(-masks * 10.0, masks) == pytest.approx(0.0)


def test_pixel_auroc_is_nan_without_both_classes():
    m = np.zeros((2, 8, 8), np.float32)
    assert np.isnan(pixel_auroc(np.random.rand(2, 8, 8), m))


def test_pixel_auroc_rejects_shape_mismatch(masks):
    with pytest.raises(ValueError, match="must match"):
        pixel_auroc(np.zeros((4, 8, 8)), masks)


# ---------------------------------------------------------------------------
# PRO
# ---------------------------------------------------------------------------

def test_pro_perfect_localizer(masks):
    rng = np.random.default_rng(0)
    maps = masks * 10.0 + rng.random(masks.shape) * 0.1
    r = pro_score(maps, masks)
    assert r["n_regions"] == 3
    assert r["pro"] == pytest.approx(1.0, abs=0.01)


def test_pro_random_localizer_scores_half_the_fpr_limit(masks):
    """PRO integrates y=x over [0, L] and divides by L -> L/2, not 0.5."""
    rng = np.random.default_rng(1)
    maps = rng.random(masks.shape).astype(np.float32)
    r = pro_score(maps, masks)
    assert r["pro"] == pytest.approx(DEFAULT_FPR_LIMIT / 2, abs=0.05)


def test_pro_punishes_finding_only_the_big_region(masks):
    """The size bias PRO exists to expose: pixel AUROC holds up, PRO does not.

    Region 1 (104 px) is 60 % of all defective pixels, so recovering it and
    nothing else still buys a respectable pixel AUROC. PRO sees one region
    of three and lands near the 1/3 + chance floor.
    """
    rng = np.random.default_rng(2)
    maps = rng.random(masks.shape).astype(np.float32) * 0.1
    maps[1, 20:24, 2:28] = 10.0          # only the largest region found
    au = pixel_auroc(maps, masks)
    pro = pro_score(maps, masks)["pro"]
    assert au > 0.75, "pixel AUROC stays up — the big region dominates pixels"
    assert pro < 0.5, "PRO must collapse: two of three regions were missed"
    assert pro < au


def test_pro_weights_every_region_equally(masks):
    """Finding ONLY the 4 px region scores the same as only the 104 px one.

    Deterministic on purpose: the unfound regions score a flat 0 so their
    coverage is exactly 0 at every threshold. With noise there instead, the
    4 px region's coverage curve is a 4-step staircase whose realized value
    swings by ~0.25 per pixel — sampling noise that has nothing to do with
    the equal-weighting property being tested here.
    """
    # A deterministic background gradient gives the curve real resolution
    # between FPR 0 and 1, so the two runs are compared on a well-sampled
    # curve rather than on a two-valued one.
    ramp = np.linspace(0, 0.5, masks.size).reshape(masks.shape).astype(np.float32)
    tiny = ramp.copy()
    tiny[2, 0:2, 0:2] = 10.0
    big = ramp.copy()
    big[1, 20:24, 2:28] = 10.0

    pro_tiny = pro_score(tiny, masks)["pro"]
    pro_big = pro_score(big, masks)["pro"]
    # THE property: the recovered region's area does not change its weight.
    # 104 px and 4 px each count for exactly one region out of three.
    assert pro_tiny == pytest.approx(pro_big, abs=0.01)
    # And recovering one region of three is far from a perfect score.
    assert pro_tiny < pro_score(masks * 10.0, masks)["pro"] - 0.3


def test_pro_survives_heavily_tied_scores(masks):
    """A binary map (all normal pixels exactly equal) must still score.

    Quantile-picked thresholds all collapse to one value here; the metric
    has to fall back on its linear sweep rather than return nan.
    """
    maps = masks.copy()          # perfect, and perfectly tied
    r = pro_score(maps, masks)
    assert np.isfinite(r["pro"])
    assert r["pro"] == pytest.approx(1.0, abs=0.02)


def test_pro_is_nan_without_any_region():
    m = np.zeros((2, 8, 8), np.float32)
    r = pro_score(np.random.rand(2, 8, 8), m)
    assert np.isnan(r["pro"]) and r["n_regions"] == 0


def test_pro_fpr_limit_changes_the_score(masks):
    rng = np.random.default_rng(4)
    maps = rng.random(masks.shape).astype(np.float32)
    lo = pro_score(maps, masks, fpr_limit=0.1)["pro"]
    hi = pro_score(maps, masks, fpr_limit=0.6)["pro"]
    # A random localizer's PRO is fpr_limit/2, so a wider limit scores higher.
    assert lo < hi


def test_pro_rejects_bad_fpr_limit(masks):
    with pytest.raises(ValueError, match="fpr_limit"):
        pro_score(np.zeros_like(masks), masks, fpr_limit=0.0)


# ---------------------------------------------------------------------------
# anomaly maps from a real (tiny) ensemble
# ---------------------------------------------------------------------------

def test_collect_anomaly_maps_shape_and_smoothing():
    cfg = {
        "model": {"channels": [8, 16], "gender_head": False,
                  "attrs_head": False, "cutpaste_head": True},
        "dataset": {"image_size": 16, "crop_size": 16},
    }
    models = [build_model(cfg).eval() for _ in range(2)]
    decs = [build_decoders(m, image_size=16).eval() for m in models]
    imgs = torch.rand(5, 3, 16, 16)
    loader = [(imgs[:3], 0, torch.zeros(3, 0), 0),
              (imgs[3:], 0, torch.zeros(2, 0), 0)]

    maps = collect_anomaly_maps(models, decs, loader, torch.device("cpu"),
                                smooth_sigma=0.0)
    assert maps.shape == (5, 16, 16)
    assert np.isfinite(maps).all()
    assert (maps >= 0).all(), "the bias term is a squared error"

    smoothed = collect_anomaly_maps(models, decs, loader, torch.device("cpu"),
                                     smooth_sigma=2.0)
    assert smoothed.shape == maps.shape
    # Smoothing must reduce spatial variance without touching image count.
    assert smoothed.std(axis=(1, 2)).mean() < maps.std(axis=(1, 2)).mean()


def test_collect_anomaly_maps_rejects_unknown_term():
    with pytest.raises(ValueError, match="risk/bias/variance"):
        collect_anomaly_maps([], [], [], torch.device("cpu"), term="nope")


class _ConstantDecoder(torch.nn.Module):
    """A decoder that ignores its input — the collapsed-to-the-mean failure."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer("out", torch.rand(1, 3, size, size))

    def forward(self, x):
        return self.out.expand(x.shape[0], -1, -1, -1)


def test_collapsed_decoders_are_warned_about(caplog):
    """Undertrained decoders emit |x - mean|, which tracks brightness.

    This is silent otherwise: the run finishes and the figures render, and
    only the pixel AUROC looks odd (it can land BELOW 0.5). The warning is
    the one place that failure is named.
    """
    cfg = {
        "model": {"channels": [8, 16], "gender_head": False,
                  "attrs_head": False, "cutpaste_head": True},
        "dataset": {"image_size": 16, "crop_size": 16},
    }
    models = [build_model(cfg).eval()]
    decs = [torch.nn.ModuleList([_ConstantDecoder(16), _ConstantDecoder(16)])]
    # Images that differ a lot from each other; a healthy decoder would
    # track them, a collapsed one cannot.
    imgs = torch.rand(6, 3, 16, 16)
    loader = [(imgs, 0, torch.zeros(6, 0), 0)]

    with caplog.at_level("WARNING", logger="celeba_ood"):
        collect_anomaly_maps(models, decs, loader, torch.device("cpu"),
                             smooth_sigma=0.0)
    assert any("COLLAPSED" in r.message for r in caplog.records), (
        "a decoder that ignores its input must be flagged"
    )


class _TrackingDecoder(torch.nn.Module):
    """A decoder whose output is a function of its input activation.

    Stands in for a *converged* decoder. A freshly built ``BlockDecoder``
    will NOT do: at random init its output barely responds to its input,
    which is genuinely the collapsed regime the guard is meant to catch —
    using one here would assert that the guard stays quiet on exactly the
    case it exists to flag.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        out = torch.nn.functional.interpolate(
            m, size=(self.size, self.size), mode="nearest",
        )
        return out.expand(-1, 3, -1, -1).clamp(0, 1)


def test_healthy_decoders_are_not_warned_about(caplog):
    cfg = {
        "model": {"channels": [8, 16], "gender_head": False,
                  "attrs_head": False, "cutpaste_head": True},
        "dataset": {"image_size": 16, "crop_size": 16},
    }
    models = [build_model(cfg).eval()]
    decs = [torch.nn.ModuleList([_TrackingDecoder(16), _TrackingDecoder(16)])]
    imgs = torch.rand(6, 3, 16, 16)
    loader = [(imgs, 0, torch.zeros(6, 0), 0)]

    with caplog.at_level("WARNING", logger="celeba_ood"):
        collect_anomaly_maps(models, decs, loader, torch.device("cpu"),
                             smooth_sigma=0.0)
    assert not any("COLLAPSED" in r.message for r in caplog.records)


def test_tracking_ratio_is_always_reported():
    """The ratio is the useful output; the warning is only a coarse net.

    It is recorded on every run so a suspicious pixel AUROC can be checked
    against decoder health after the fact.
    """
    cfg = {
        "model": {"channels": [8, 16], "gender_head": False,
                  "attrs_head": False, "cutpaste_head": True},
        "dataset": {"image_size": 16, "crop_size": 16},
    }
    models = [build_model(cfg).eval()]
    imgs = torch.rand(6, 3, 16, 16)
    loader = [(imgs, 0, torch.zeros(6, 0), 0)]

    collect_anomaly_maps(
        models, [torch.nn.ModuleList([_TrackingDecoder(16)] * 2)],
        loader, torch.device("cpu"), smooth_sigma=0.0,
    )
    tracking = collect_anomaly_maps.last_tracking_ratio

    collect_anomaly_maps(
        models, [torch.nn.ModuleList([_ConstantDecoder(16)] * 2)],
        loader, torch.device("cpu"), smooth_sigma=0.0,
    )
    constant = collect_anomaly_maps.last_tracking_ratio

    assert constant == pytest.approx(0.0, abs=1e-6)
    assert tracking > constant
    assert constant < COLLAPSE_RATIO_WARN <= tracking


def test_load_masks_stacks_dataset_masks(tmp_path):
    from PIL import Image
    from lrad.mvtec import MVTecCategory

    paths = []
    for i in range(3):
        p = tmp_path / f"{i:03d}.png"
        Image.fromarray(np.full((16, 16, 3), 128, np.uint8)).save(p)
        paths.append(p)
    gt = tmp_path / "gt" / "d"
    gt.mkdir(parents=True)
    arr = np.zeros((16, 16), np.uint8)
    arr[2:6, 2:6] = 255
    for i in range(3):
        Image.fromarray(arr, mode="L").save(gt / f"{i:03d}_mask.png")

    ds = MVTecCategory(paths, ["d"] * 3, is_ood=1, image_size=16,
                       crop_size=16, mask_root=tmp_path / "gt")
    m = load_masks(ds)
    assert m.shape == (3, 16, 16)
    assert set(np.unique(m).tolist()) <= {0.0, 1.0}
    assert m.sum() == 3 * 16
