"""Tests for the CutPaste augmentation, pretext head, and training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.cutpaste import cutpaste_batch
from lrad.decoder import build_decoders
from lrad.ensemble import load_ensemble_members  # noqa: F401  (import check)
from lrad.fusion import collect_fusion_signals
from lrad.localized import STD_FLOOR  # noqa: F401
from lrad.feature_error import fit_feature_error_stats
from lrad.model import FacialCNN, build_model
from lrad.train import train_model

IMG = 16
BLOCKS = (0, 1)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# cutpaste_batch
# ---------------------------------------------------------------------------

def test_cutpaste_labels_match_alterations():
    """Every label-1 image differs from the input; label-0 are untouched."""
    imgs = torch.rand(16, 3, IMG, IMG)
    aug, labels = cutpaste_batch(imgs, prob=0.5, generator=_gen())
    assert aug.shape == imgs.shape
    assert labels.shape == (16,)
    for i in range(16):
        changed = not torch.equal(aug[i], imgs[i])
        assert changed == bool(labels[i])


def test_cutpaste_prob_extremes():
    imgs = torch.rand(8, 3, IMG, IMG)
    aug, labels = cutpaste_batch(imgs, prob=0.0, generator=_gen())
    assert labels.sum() == 0 and torch.equal(aug, imgs)
    _, labels = cutpaste_batch(imgs, prob=1.0, generator=_gen())
    assert labels.sum() == 8


def test_cutpaste_patch_content_comes_from_donor():
    """With constant-colour images, pasted pixels carry the donor's colour."""
    imgs = torch.zeros(2, 3, IMG, IMG)
    imgs[0] += 0.25
    imgs[1] += 0.75
    aug, labels = cutpaste_batch(
        imgs, prob=1.0, scar_prob=0.0, generator=_gen(),
    )
    for i, donor in ((0, 1), (1, 0)):
        changed = aug[i] != imgs[i]
        assert changed.any()
        assert torch.allclose(
            aug[i][changed],
            torch.full_like(aug[i][changed], float(imgs[donor, 0, 0, 0])),
        )


def test_cutpaste_is_deterministic_given_generator():
    imgs = torch.rand(6, 3, IMG, IMG)
    a1, l1 = cutpaste_batch(imgs, generator=_gen(7))
    a2, l2 = cutpaste_batch(imgs, generator=_gen(7))
    assert torch.equal(a1, a2) and torch.equal(l1, l2)


def test_cutpaste_validates_inputs():
    imgs = torch.rand(2, 3, IMG, IMG)
    with pytest.raises(ValueError):
        cutpaste_batch(torch.rand(3, IMG, IMG))
    with pytest.raises(ValueError):
        cutpaste_batch(imgs, prob=1.5)
    with pytest.raises(ValueError):
        cutpaste_batch(imgs, area_range=(0.5, 0.2))
    with pytest.raises(ValueError):
        cutpaste_batch(imgs, scar_prob=-0.1)


# ---------------------------------------------------------------------------
# model head + build_model
# ---------------------------------------------------------------------------

def test_model_cutpaste_head_optional():
    plain = FacialCNN(channels=(4, 8), input_size=IMG)
    assert plain.head_cutpaste is None
    assert "cutpaste_logits" not in plain(torch.rand(2, 3, IMG, IMG))

    with_head = FacialCNN(channels=(4, 8), input_size=IMG,
                          cutpaste_head=True)
    out = with_head(torch.rand(2, 3, IMG, IMG))
    assert out["cutpaste_logits"].shape == (2, 2)


def test_build_model_reads_cutpaste_flag():
    cfg = {"model": {"channels": [4, 8], "cutpaste_head": True},
           "dataset": {"image_size": IMG}}
    assert build_model(cfg).head_cutpaste is not None
    cfg["model"]["cutpaste_head"] = False
    assert build_model(cfg).head_cutpaste is None


def test_old_checkpoints_load_into_headless_model():
    """A state dict saved without the head loads into a headless model."""
    old = FacialCNN(channels=(4, 8), input_size=IMG)
    fresh = FacialCNN(channels=(4, 8), input_size=IMG)
    fresh.load_state_dict(old.state_dict())  # must not raise


# ---------------------------------------------------------------------------
# training loop with the pretext task
# ---------------------------------------------------------------------------

def _tiny_loader(n: int = 12, batch: int = 4):
    imgs = torch.rand(n, 3, IMG, IMG)
    gender = torch.randint(0, 2, (n,))
    attrs = torch.randint(0, 2, (n, 6)).float()
    is_ood = torch.zeros(n, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(imgs, gender, attrs, is_ood)
    return torch.utils.data.DataLoader(ds, batch_size=batch)


def test_train_model_with_cutpaste_runs_and_tracks_acc():
    model = FacialCNN(channels=(4, 8), input_size=IMG, cutpaste_head=True)
    history = train_model(
        model, _tiny_loader(), None,
        epochs=1, lr=1e-3, device=torch.device("cpu"), log_every=1,
        cutpaste={"prob": 0.5, "loss_weight": 1.0},
    )
    assert len(history["batch_cutpaste_acc"]) == 3  # one per batch
    assert np.isfinite(history["train_loss"][0])


def test_train_model_cutpaste_requires_head():
    model = FacialCNN(channels=(4, 8), input_size=IMG)
    with pytest.raises(ValueError):
        train_model(
            model, _tiny_loader(), None,
            epochs=1, device=torch.device("cpu"),
            cutpaste={"prob": 0.5},
        )


def test_train_model_without_cutpaste_unchanged():
    model = FacialCNN(channels=(4, 8), input_size=IMG)
    history = train_model(
        model, _tiny_loader(), None,
        epochs=1, lr=1e-3, device=torch.device("cpu"), log_every=1,
    )
    assert history["batch_cutpaste_acc"] == []


# ---------------------------------------------------------------------------
# fusion signal
# ---------------------------------------------------------------------------

def test_collect_fusion_signals_includes_cutpaste_prob():
    torch.manual_seed(0)
    models, decoders_list = [], []
    for s in range(2):
        torch.manual_seed(s)
        m = FacialCNN(channels=(4, 8), input_size=IMG,
                      cutpaste_head=True).eval()
        models.append(m)
        decoders_list.append(build_decoders(m, image_size=IMG).eval())
    loader = [(torch.rand(6, 3, IMG, IMG),)]
    ref = fit_feature_error_stats(
        models, decoders_list, loader, torch.device("cpu"), BLOCKS,
    )
    sig = collect_fusion_signals(
        models, decoders_list, loader, torch.device("cpu"), ref, BLOCKS,
    )
    assert "cutpaste_prob" in sig
    assert sig["cutpaste_prob"].shape == (6,)
    assert ((sig["cutpaste_prob"] >= 0) & (sig["cutpaste_prob"] <= 1)).all()
