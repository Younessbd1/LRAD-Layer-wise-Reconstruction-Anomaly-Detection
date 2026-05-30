"""Shape/contract tests for the raw error + reduction utilities.

The OOD anomaly itself (bias = risk − variance) is an ensemble quantity
and is covered by ``tests/test_ensemble.py``. This file only checks the
low-level helpers that survived the removal of the sigma-whitening path:
``_per_pixel_errors``, ``_reduce_over_pixels`` and
``aggregate_anomaly_score``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.anomaly_score import (
    _per_pixel_errors,
    _reduce_over_pixels,
    aggregate_anomaly_score,
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
    images = torch.rand(10, 3, IMG, IMG)
    errs = _per_pixel_errors(model, decoders, images)
    return model, decoders, images, errs


def test_per_pixel_error_shapes(setup):
    _, _, images, errs = setup
    assert set(errs.keys()) == set(range(N_BLOCKS))
    for k in range(N_BLOCKS):
        e = errs[k]
        assert e.shape == (images.shape[0], IMG, IMG)
        assert torch.isfinite(e).all()
        assert (e >= 0).all()  # |x - recon| is non-negative


@pytest.mark.parametrize("agg", ["mean", "max", "p95"])
def test_reduce_over_pixels_shape(setup, agg):
    _, _, images, errs = setup
    s = _reduce_over_pixels(errs[0], agg)
    assert s.shape == (images.shape[0],)
    assert torch.isfinite(s).all()


def test_reduce_over_pixels_ordering(setup):
    """For any map, mean <= p95 <= max, per image."""
    _, _, _, errs = setup
    e = errs[0]
    mean = _reduce_over_pixels(e, "mean")
    p95 = _reduce_over_pixels(e, "p95")
    mx = _reduce_over_pixels(e, "max")
    assert torch.all(mean <= p95 + 1e-6)
    assert torch.all(p95 <= mx + 1e-6)


def test_reduce_over_pixels_rejects_unknown_agg(setup):
    _, _, _, errs = setup
    with pytest.raises(ValueError):
        _reduce_over_pixels(errs[0], "median")


@pytest.mark.parametrize("agg", ["mean", "max", "p95"])
def test_aggregate_score(setup, agg):
    _, _, images, errs = setup
    score = aggregate_anomaly_score(errs, agg=agg)
    assert score.shape == (images.shape[0],)
    assert torch.isfinite(score).all()
    assert (score >= 0).all()


def test_block_weights_validation(setup):
    _, _, images, errs = setup
    weighted = aggregate_anomaly_score(errs, block_weights=[0.0, 1.0])
    uniform = aggregate_anomaly_score(errs, block_weights=[1.0, 1.0])
    assert weighted.shape == uniform.shape == (images.shape[0],)
    # Putting all weight on block 1 must equal block 1's reduced score.
    only_b1 = _reduce_over_pixels(errs[1], "p95")
    assert torch.allclose(weighted, only_b1, atol=1e-5)
    with pytest.raises(ValueError):
        aggregate_anomaly_score(errs, block_weights=[1.0])


def test_aggregate_score_is_finite_scalar(setup):
    _, _, _, errs = setup
    score = aggregate_anomaly_score(errs, agg="mean")
    assert np.isfinite(float(score.mean()))
