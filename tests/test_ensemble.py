"""Shape/contract tests for the ensemble bias/variance decomposition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.decoder import build_decoders
from lrad.ensemble import (
    block_reconstructions,
    decomposition_maps,
    evaluate_ensemble_decomposition,
    identity_residual,
    mean_error_maps,
    min_error_maps,
    quantile_min_error_maps,
)
from lrad.model import FacialCNN

IMG = 16
N_BLOCKS = 2
N_MODELS = 3
TERMS = ("risk", "bias", "variance")


def _make_model(seed: int) -> tuple[FacialCNN, torch.nn.ModuleList]:
    torch.manual_seed(seed)
    model = FacialCNN(channels=(4, 8), input_size=IMG).eval()
    decoders = build_decoders(model, image_size=IMG).eval()
    return model, decoders


@pytest.fixture(scope="module")
def setup():
    pairs = [_make_model(s) for s in range(N_MODELS)]
    models = [m for m, _ in pairs]
    decoders_list = [d for _, d in pairs]
    images = torch.rand(8, 3, IMG, IMG)
    recons = [
        block_reconstructions(models[i], decoders_list[i], images)
        for i in range(N_MODELS)
    ]
    maps = decomposition_maps(images, recons)
    return models, decoders_list, images, maps


def test_decomposition_map_shapes(setup):
    _, _, images, maps = setup
    assert set(maps.keys()) == set(range(N_BLOCKS))
    for k in range(N_BLOCKS):
        for t in TERMS:
            m = maps[k][t]
            assert m.shape == (images.shape[0], IMG, IMG)
            assert torch.isfinite(m).all()
        assert maps[k]["mean_recon"].shape == (images.shape[0], 3, IMG, IMG)


def test_risk_bias_variance_identity(setup):
    """Risk = Bias + Variance must hold pixelwise up to float32 noise."""
    _, _, _, maps = setup
    for k in range(N_BLOCKS):
        recombined = maps[k]["bias"] + maps[k]["variance"]
        assert torch.allclose(maps[k]["risk"], recombined, atol=1e-5)
    assert identity_residual(maps) < 1e-5


def test_terms_are_non_negative(setup):
    """Squared-error and spread terms are non-negative by construction."""
    _, _, _, maps = setup
    for k in range(N_BLOCKS):
        for t in TERMS:
            assert (maps[k][t] >= 0).all()


def test_single_model_has_zero_variance():
    """A 1-model 'ensemble' has no spread: Variance == 0 and Risk == Bias."""
    model, decoders = _make_model(0)
    images = torch.rand(5, 3, IMG, IMG)
    recons = [block_reconstructions(model, decoders, images)]
    maps = decomposition_maps(images, recons)
    for k in range(N_BLOCKS):
        assert torch.allclose(maps[k]["variance"],
                              torch.zeros_like(maps[k]["variance"]),
                              atol=1e-6)
        assert torch.allclose(maps[k]["risk"], maps[k]["bias"], atol=1e-6)


def test_identical_models_have_zero_variance():
    """If every member is the same model, the members never disagree."""
    model, decoders = _make_model(1)
    images = torch.rand(5, 3, IMG, IMG)
    recons = [
        block_reconstructions(model, decoders, images) for _ in range(4)
    ]
    maps = decomposition_maps(images, recons)
    for k in range(N_BLOCKS):
        assert maps[k]["variance"].max().item() < 1e-6


def test_decomposition_maps_rejects_empty():
    with pytest.raises(ValueError):
        decomposition_maps(torch.rand(2, 3, IMG, IMG), [])


def _recons(setup):
    models, decoders_list, images, _ = setup
    return images, [
        block_reconstructions(models[i], decoders_list[i], images)
        for i in range(N_MODELS)
    ]


def test_quantile_min_k1_equals_min(setup):
    """k=1 must reproduce min_error_maps exactly (bit-for-bit)."""
    images, recons = _recons(setup)
    qmin = quantile_min_error_maps(images, recons, k=1)
    mn = min_error_maps(images, recons)
    assert set(qmin.keys()) == set(mn.keys())
    for k in qmin:
        assert torch.equal(qmin[k], mn[k])


def test_quantile_min_is_monotone_in_k(setup):
    """The k-th smallest error is non-decreasing in k, and bounded by
    min (k=1) below and the per-pixel max (k=M) above."""
    images, recons = _recons(setup)
    prev = None
    for k in range(1, N_MODELS + 1):
        cur = quantile_min_error_maps(images, recons, k=k)
        if prev is not None:
            for b in cur:
                assert (cur[b] >= prev[b]).all()
        prev = cur


def test_quantile_min_shapes_and_finiteness(setup):
    images, recons = _recons(setup)
    qmin = quantile_min_error_maps(images, recons, k=2)
    mean = mean_error_maps(images, recons)
    assert set(qmin.keys()) == set(range(N_BLOCKS))
    for b in range(N_BLOCKS):
        assert qmin[b].shape == (images.shape[0], IMG, IMG)
        assert torch.isfinite(qmin[b]).all()
        assert (qmin[b] >= 0).all()
        # The k-th smallest of M values never exceeds M times their mean.
        assert (qmin[b] <= mean[b] * N_MODELS + 1e-6).all()


def test_quantile_min_rejects_bad_k(setup):
    images, recons = _recons(setup)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, recons, k=0)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, recons, k=N_MODELS + 1)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, [], k=1)


def test_evaluate_ensemble_decomposition(setup):
    """End-to-end: AUROC keys present and score arrays well-shaped."""
    models, decoders_list, images, _ = setup
    device = torch.device("cpu")
    in_imgs = torch.rand(10, 3, IMG, IMG)
    ood_imgs = torch.rand(7, 3, IMG, IMG)
    loaders = {
        "test_in": [(in_imgs[:6], None, None, None),
                    (in_imgs[6:], None, None, None)],
        "test_ood": [(ood_imgs, None, None, None)],
    }
    out = evaluate_ensemble_decomposition(
        models, decoders_list, loaders, device, agg="p95",
    )
    assert out["blocks"] == list(range(N_BLOCKS))
    for t in TERMS:
        assert len(out["per_block_auroc"][t]) == N_BLOCKS
        assert f"score_{t}_aggregated" in out["auroc"]
        for k in range(N_BLOCKS):
            assert f"score_{t}_per_block_{k}" in out["auroc"]
        assert out["scores_in"]["aggregated"][t].shape == (10,)
        assert out["scores_ood"]["aggregated"][t].shape == (7,)
        assert np.isfinite(out["scores_in"]["aggregated"][t]).all()
