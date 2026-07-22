"""Smoke tests for the training-curve and architecture-effect figures.

Same spirit as test_plots_instance.py: the code is matplotlib plumbing, so
the checks stay light — each figure writes a non-empty PNG for synthetic
inputs, and empty histories are a silent no-op.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from lrad.plots import (
    plot_architecture_effect,
    plot_batch_loss,
    plot_decoder_history,
)


def _history(n_epochs: int = 3, bpe: int = 20) -> dict:
    rng = np.random.default_rng(0)
    return {
        "batch_loss": rng.uniform(0.1, 2.0, n_epochs * bpe).tolist(),
        "epoch_ends": [bpe * (e + 1) - 1 for e in range(n_epochs)],
    }


def test_plot_batch_loss_writes_png(tmp_path):
    out = tmp_path / "batch_loss.png"
    plot_batch_loss(_history(), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_batch_loss_empty_history_is_noop(tmp_path):
    out = tmp_path / "batch_loss.png"
    plot_batch_loss({"batch_loss": [], "epoch_ends": []}, out)
    assert not out.exists()


def test_plot_decoder_history_writes_png(tmp_path):
    E, B = 6, 4
    per_block = np.linspace(0.05, 0.01, E)[:, None] * (
        1.0 + np.arange(B) / B
    )
    hist = {
        "train_loss_per_block": per_block.tolist(),
        "train_loss": per_block.sum(axis=1).tolist(),
        "val_loss_per_block": (per_block * 1.1).tolist(),
    }
    out = tmp_path / "decoder_history.png"
    plot_decoder_history(hist, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_decoder_history_empty_is_noop(tmp_path):
    out = tmp_path / "decoder_history.png"
    plot_decoder_history({"train_loss_per_block": []}, out)
    assert not out.exists()


def test_plot_architecture_effect_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    records = [
        {
            "member": i + 1,
            "channels": [8 * (i + 1), 16, 32, 32, 32],
            "kernel_size": 3 if i % 2 == 0 else 5,
            "n_params": 10_000 * (i + 1),
            "auroc": float(rng.uniform(0.5, 0.7)),
            "auroc_per_block": rng.uniform(0.5, 0.7, 5).tolist(),
        }
        for i in range(4)
    ]
    out = tmp_path / "architecture_effect.png"
    plot_architecture_effect(records, out, title="arch effect")
    assert out.exists() and out.stat().st_size > 0


def test_plot_architecture_effect_empty_is_noop(tmp_path):
    out = tmp_path / "arch.png"
    plot_architecture_effect([], out)
    assert not out.exists()
