"""Minimal shape/contract tests for the bias/variance debiasing module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.anomaly_score import (
    aggregate_anomaly_score,
    cleaned_anomaly_maps,
    estimate_baseline_stats,
)
from lrad.decoder import build_decoders
from lrad.model import FacialCNN

IMG = 16
N_BLOCKS = 2


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = FacialCNN(channels=(4, 8), input_size=IMG).eval()
    decoders = build_decoders(model, image_size=IMG).eval()
    device = torch.device("cpu")
    images = torch.rand(10, 3, IMG, IMG)
    # A loader is just an iterable of (img, gender, attrs, is_ood) tuples.
    loader = [
        (images[:6], None, None, None),
        (images[6:], None, None, None),
    ]
    stats = estimate_baseline_stats(model, decoders, loader, device)
    return model, decoders, device, images, stats


def test_baseline_stats_shapes(setup):
    _, _, _, _, stats = setup
    assert set(stats.keys()) == set(range(N_BLOCKS))
    for k in range(N_BLOCKS):
        mu, sigma = stats[k]["mu"], stats[k]["sigma"]
        assert mu.shape == (IMG, IMG)
        assert sigma.shape == (IMG, IMG)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()
        assert (sigma >= 0).all()  # std is non-negative


def test_cleaned_maps_nonneg_and_shaped(setup):
    model, decoders, device, images, stats = setup
    maps = cleaned_anomaly_maps(model, decoders, images, stats, device)
    assert set(maps.keys()) == set(range(N_BLOCKS))
    for k in range(N_BLOCKS):
        m = maps[k]
        assert m.shape == (images.shape[0], IMG, IMG)
        assert torch.isfinite(m).all()
        assert (m >= 0).all()  # max(0, ·) keeps only worse-than-expected


@pytest.mark.parametrize("agg", ["mean", "max", "p95"])
def test_aggregate_score(setup, agg):
    model, decoders, device, images, stats = setup
    maps = cleaned_anomaly_maps(model, decoders, images, stats, device)
    score = aggregate_anomaly_score(maps, agg=agg)
    assert score.shape == (images.shape[0],)
    assert torch.isfinite(score).all()
    assert (score >= 0).all()


def test_block_weights_validation(setup):
    model, decoders, device, images, stats = setup
    maps = cleaned_anomaly_maps(model, decoders, images, stats, device)
    weighted = aggregate_anomaly_score(maps, block_weights=[0.0, 1.0])
    uniform = aggregate_anomaly_score(maps, block_weights=[1.0, 1.0])
    assert weighted.shape == uniform.shape == (images.shape[0],)
    with pytest.raises(ValueError):
        aggregate_anomaly_score(maps, block_weights=[1.0])


def test_in_distribution_score_is_small(setup):
    """Scoring the very data used to estimate the baseline should give a
    near-zero debiased score (no anomaly present)."""
    model, decoders, device, images, stats = setup
    maps = cleaned_anomaly_maps(model, decoders, images, stats, device)
    score = aggregate_anomaly_score(maps, agg="mean")
    # mean over pixels of max(0, z) for z ~ zero-mean noise is O(sigma),
    # which in whitened units stays well below a few sigma.
    assert float(score.mean()) < 3.0
    assert np.isfinite(float(score.mean()))
