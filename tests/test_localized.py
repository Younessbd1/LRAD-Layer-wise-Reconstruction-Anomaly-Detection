"""Tests for the localized (per-pixel z-score + patch-max) OOD scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.decoder import build_decoders
from lrad.ensemble import TERMS
from lrad.localized import (
    collect_localized_scores,
    evaluate_localized_ood,
    fit_reference_stats,
    patch_max_reduce,
)
from lrad.model import FacialCNN

IMG = 16
N_BLOCKS = 2
N_MODELS = 3
PATCHES = (2, 4)


def _make_model(seed: int) -> tuple[FacialCNN, torch.nn.ModuleList]:
    torch.manual_seed(seed)
    model = FacialCNN(channels=(4, 8), input_size=IMG).eval()
    decoders = build_decoders(model, image_size=IMG).eval()
    return model, decoders


@pytest.fixture(scope="module")
def ensemble():
    pairs = [_make_model(s) for s in range(N_MODELS)]
    return [m for m, _ in pairs], [d for _, d in pairs]


def _loader(images: torch.Tensor, batch: int = 4):
    return [(images[i:i + batch],) for i in range(0, len(images), batch)]


# ---------------------------------------------------------------------------
# patch_max_reduce
# ---------------------------------------------------------------------------

def test_patch_max_uniform_map_is_identity():
    """On a constant map every window mean equals the constant."""
    a = torch.full((3, IMG, IMG), 0.7)
    out = patch_max_reduce(a, PATCHES)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.full((3,), 0.7), atol=1e-6)


def test_patch_max_fires_on_local_hot_patch_where_p95_does_not():
    """A small hot square must dominate patch-max but stay invisible to a
    global p95 — the exact failure mode this scoring fixes."""
    clean = torch.zeros(1, IMG, IMG)
    hot = clean.clone()
    hot[0, 4:7, 4:7] = 1.0  # 9 of 256 pixels ≈ 3.5 %, below p95's 5 % reach

    p95_clean = torch.quantile(clean.flatten(1), 0.95, dim=1)
    p95_hot = torch.quantile(hot.flatten(1), 0.95, dim=1)
    assert torch.allclose(p95_clean, p95_hot)  # p95 is blind to it

    pm_clean = patch_max_reduce(clean, PATCHES)
    pm_hot = patch_max_reduce(hot, PATCHES)
    assert pm_hot.item() > pm_clean.item() + 0.5  # patch-max fires


def test_patch_max_anomaly_position_invariance():
    """The same hot patch anywhere in the frame gives the same score."""
    a = torch.zeros(2, IMG, IMG)
    a[0, 0:4, 0:4] = 1.0     # top-left corner
    a[1, 10:14, 6:10] = 1.0  # elsewhere
    out = patch_max_reduce(a, PATCHES)
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_patch_max_validates_inputs():
    with pytest.raises(ValueError):
        patch_max_reduce(torch.zeros(2, IMG, IMG), ())
    with pytest.raises(ValueError):
        patch_max_reduce(torch.zeros(2, IMG, IMG), (IMG + 1,))
    with pytest.raises(ValueError):
        patch_max_reduce(torch.zeros(2, 3, IMG, IMG), PATCHES)


# ---------------------------------------------------------------------------
# fit_reference_stats
# ---------------------------------------------------------------------------

def test_reference_stats_shapes_and_floor(ensemble):
    models, decoders_list = ensemble
    images = torch.rand(8, 3, IMG, IMG)
    ref = fit_reference_stats(
        models, decoders_list, _loader(images), torch.device("cpu"),
    )
    assert ref["n_images"] == 8
    for t in TERMS:
        assert set(ref["stats"][t].keys()) == set(range(N_BLOCKS))
        for k in range(N_BLOCKS):
            st = ref["stats"][t][k]
            assert st["mean"].shape == (IMG, IMG)
            assert st["std"].shape == (IMG, IMG)
            assert (st["std"] >= ref["std_floor"]).all()
            assert torch.isfinite(st["mean"]).all()


def test_reference_stats_constant_input_has_floored_std(ensemble):
    """The same image repeated leaves zero spread → std == floor exactly."""
    models, decoders_list = ensemble
    one = torch.rand(1, 3, IMG, IMG).expand(6, -1, -1, -1).contiguous()
    ref = fit_reference_stats(
        models, decoders_list, _loader(one), torch.device("cpu"),
    )
    for t in TERMS:
        for k in range(N_BLOCKS):
            std = ref["stats"][t][k]["std"]
            assert torch.allclose(
                std, torch.full_like(std, ref["std_floor"]), atol=1e-6,
            )


def test_reference_stats_rejects_empty_and_bad_floor(ensemble):
    models, decoders_list = ensemble
    with pytest.raises(ValueError):
        fit_reference_stats(models, decoders_list, [], torch.device("cpu"))
    with pytest.raises(ValueError):
        fit_reference_stats(
            models, decoders_list, _loader(torch.rand(2, 3, IMG, IMG)),
            torch.device("cpu"), std_floor=0.0,
        )


# ---------------------------------------------------------------------------
# collect_localized_scores / evaluate_localized_ood
# ---------------------------------------------------------------------------

def test_scoring_same_images_as_reference_is_zero(ensemble):
    """Scoring the exact reference batch: map == μ pixelwise, so every
    z-score — and hence every patch-max — is exactly 0."""
    models, decoders_list = ensemble
    one = torch.rand(1, 3, IMG, IMG).expand(6, -1, -1, -1).contiguous()
    device = torch.device("cpu")
    ref = fit_reference_stats(models, decoders_list, _loader(one), device)
    out = collect_localized_scores(
        models, decoders_list, _loader(one[:2]), device, ref, PATCHES,
    )
    for t in TERMS:
        assert np.allclose(out["aggregated"][t], 0.0, atol=1e-4)
        for k in range(N_BLOCKS):
            assert np.allclose(out["per_block"][t][k], 0.0, atol=1e-4)


def test_collect_scores_shapes_and_baseline(ensemble):
    models, decoders_list = ensemble
    device = torch.device("cpu")
    ref = fit_reference_stats(
        models, decoders_list, _loader(torch.rand(8, 3, IMG, IMG)), device,
    )
    n = 6
    out = collect_localized_scores(
        models, decoders_list, _loader(torch.rand(n, 3, IMG, IMG)),
        device, ref, PATCHES,
    )
    for t in TERMS:
        assert out["aggregated"][t].shape == (n,)
        assert out["baseline_p95"][t].shape == (n,)
        assert np.isfinite(out["aggregated"][t]).all()
        assert np.isfinite(out["baseline_p95"][t]).all()
        for k in range(N_BLOCKS):
            assert out["per_block"][t][k].shape == (n,)


def test_evaluate_localized_ood_end_to_end(ensemble):
    """AUROC keys present, finite scores, baseline reported alongside."""
    models, decoders_list = ensemble
    device = torch.device("cpu")
    ref = _loader(torch.rand(8, 3, IMG, IMG))
    test_in = _loader(torch.rand(6, 3, IMG, IMG))
    # Give OOD images a bright square occlusion so the split is nontrivial.
    ood_imgs = torch.rand(6, 3, IMG, IMG)
    ood_imgs[:, :, 5:9, 5:9] = 1.0
    out = evaluate_localized_ood(
        models, decoders_list, ref, test_in, _loader(ood_imgs), device,
        patch_sizes=PATCHES,
    )
    assert out["blocks"] == list(range(N_BLOCKS))
    for t in TERMS:
        assert f"zscore_{t}_aggregated" in out["auroc"]
        assert f"baseline_p95_{t}_aggregated" in out["auroc"]
        assert len(out["per_block_auroc"][t]) == N_BLOCKS
        for k in range(N_BLOCKS):
            assert f"zscore_{t}_per_block_{k}" in out["auroc"]
        a = out["auroc"][f"zscore_{t}_aggregated"]["auroc"]
        assert 0.0 <= a <= 1.0
    assert out["anomaly_auroc"]["aggregated"] == \
        out["auroc"]["zscore_bias_aggregated"]["auroc"]
    assert out["anomaly_auroc"]["baseline"] == \
        out["auroc"]["baseline_p95_bias_aggregated"]["auroc"]


def test_max_batches_caps_the_pass(ensemble):
    models, decoders_list = ensemble
    device = torch.device("cpu")
    images = torch.rand(8, 3, IMG, IMG)
    ref = fit_reference_stats(
        models, decoders_list, _loader(images, batch=4), device,
        max_batches=1,
    )
    assert ref["n_images"] == 4
    out = collect_localized_scores(
        models, decoders_list, _loader(images, batch=4), device, ref,
        PATCHES, max_batches=1,
    )
    assert out["aggregated"]["bias"].shape == (4,)
