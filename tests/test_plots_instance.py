"""Tests for smooth_cam and the per-instance decomposition figure.

The figure code is mostly matplotlib plumbing, so these checks stay light:
smooth_cam's numeric contract (rectify, blur, peak-normalize into [0, 1]) and
that plot_instance_decomposition actually writes a non-empty PNG for a small
synthetic ensemble. matplotlib runs headless via the Agg backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from lrad.plots import plot_instance_decomposition, smooth_cam


def test_smooth_cam_normalized_into_unit_range():
    rng = np.random.default_rng(0)
    cam = rng.normal(size=(32, 32)).astype(np.float32)
    out = smooth_cam(cam, sigma=5.0)
    assert out.shape == cam.shape
    assert out.min() >= 0.0
    # Peak-normalized: the brightest pixel sits at exactly 1.
    assert float(out.max()) == pytest.approx(1.0, abs=1e-5)


def test_smooth_cam_all_zero_input_stays_zero():
    out = smooth_cam(np.zeros((16, 16), dtype=np.float32), sigma=3.0)
    assert np.all(out == 0.0)


def test_smooth_cam_rectifies_negative_input():
    # A fully negative map carries no signal once rectified.
    out = smooth_cam(-np.ones((8, 8), dtype=np.float32))
    assert np.all(out == 0.0)


def test_smooth_cam_larger_sigma_spreads_energy():
    cam = np.zeros((33, 33), dtype=np.float32)
    cam[16, 16] = 1.0  # single hot pixel
    tight = smooth_cam(cam, sigma=1.0)
    wide = smooth_cam(cam, sigma=6.0)
    # More blur spreads the peak out, so a pixel away from the centre picks
    # up more of the (re-normalized) mass.
    assert wide[16, 24] > tight[16, 24]


def test_plot_instance_decomposition_writes_png(tmp_path):
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16)
    recons = [torch.rand(3, 16, 16) for _ in range(4)]
    out = tmp_path / "instance.png"
    plot_instance_decomposition(
        img, recons, out, label="ID 1", sigma=3.0, overlay_power=0.8,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_instance_decomposition_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        plot_instance_decomposition(
            torch.rand(3, 16, 16), [], tmp_path / "empty.png",
        )
