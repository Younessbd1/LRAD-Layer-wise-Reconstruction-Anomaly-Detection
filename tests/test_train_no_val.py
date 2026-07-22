"""Regression tests for training without a validation set (val_ratio=0).

An empty/absent validation loader must not silently sabotage training:
its loss is a constant 0, which previously (a) made every val metric log as
0.0000 and (b) tripped early stopping on epoch 1, restoring the
barely-trained epoch-1 weights. These tests pin the fixed behavior — full
schedule, no early stop, final weights kept.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from lrad.model import FacialCNN
from lrad.decoder import build_decoders
from lrad.train import _has_samples, train_decoders, train_model

IMG = 16
N_ATTRS = 6


def _loader(n: int, batch_size: int = 4) -> DataLoader:
    """A loader yielding (img, gender, attrs, is_ood) tuples like the real one."""
    imgs = torch.rand(n, 3, IMG, IMG)
    gender = torch.randint(0, 2, (n,))
    attrs = torch.randint(0, 2, (n, N_ATTRS)).float()
    is_ood = torch.zeros(n, dtype=torch.long)
    return DataLoader(
        TensorDataset(imgs, gender, attrs, is_ood), batch_size=batch_size,
    )


def _empty_loader() -> DataLoader:
    ds = TensorDataset(
        torch.empty(0, 3, IMG, IMG),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, N_ATTRS),
        torch.empty(0, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=4)


def _model() -> FacialCNN:
    torch.manual_seed(0)
    return FacialCNN(channels=(4, 8), n_attrs=N_ATTRS, input_size=IMG)


def test_has_samples_distinguishes_none_empty_and_real():
    assert _has_samples(None) is False
    assert _has_samples(_empty_loader()) is False
    assert _has_samples(_loader(8)) is True


def test_train_runs_full_schedule_without_val():
    """val_loader=None → no early stop, full schedule, no val history."""
    epochs = 5
    history = train_model(
        _model(), _loader(16), None,
        epochs=epochs, lr=1e-3, device=torch.device("cpu"), log_every=10,
    )
    assert len(history["train_loss"]) == epochs   # ran every epoch
    assert history["val_loss"] == []              # no fake val curve
    assert history["best_val_loss"] is None


def test_empty_val_loader_does_not_early_stop():
    """An empty val loader must be treated exactly like None — no early stop
    on a constant-zero val loss."""
    epochs = 5
    history = train_model(
        _model(), _loader(16), _empty_loader(),
        epochs=epochs, lr=1e-3, device=torch.device("cpu"),
        log_every=10, early_stop_patience=2, early_stop_min_epochs=1,
    )
    assert len(history["train_loss"]) == epochs
    assert history["val_loss"] == []


def test_final_weights_kept_without_val():
    """Without val there is no 'restore best' — the returned weights are the
    final-epoch weights, i.e. they keep training (differ from the init)."""
    model = _model()
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_model(
        model, _loader(16), None,
        epochs=3, lr=1e-2, device=torch.device("cpu"), log_every=10,
    )
    changed = any(
        not torch.equal(init[k], v) for k, v in model.state_dict().items()
    )
    assert changed


def test_train_decoders_without_val_reports_per_block_train():
    """Decoders run the full schedule and log per-block *train* loss when
    no val set is given."""
    model = _model()
    decoders = build_decoders(model, image_size=IMG)
    epochs = 3
    history = train_decoders(
        model, decoders, _loader(16), None,
        epochs=epochs, lr=1e-3, device=torch.device("cpu"), log_every=10,
    )
    assert len(history["train_loss"]) == epochs
    assert len(history["train_loss_per_block"]) == epochs
    assert len(history["train_loss_per_block"][0]) == len(decoders)
    assert history["val_loss"] == []


def test_train_model_saves_per_epoch_checkpoints(tmp_path):
    """With checkpoint_dir set, model_ep{e}.pt is written at every epoch."""
    epochs = 3
    train_model(
        _model(), _loader(16), None,
        epochs=epochs, lr=1e-3, device=torch.device("cpu"), log_every=10,
        checkpoint_dir=tmp_path,
    )
    for e in range(1, epochs + 1):
        assert (tmp_path / f"model_ep{e}.pt").exists()


def test_train_decoders_saves_per_epoch_checkpoints(tmp_path):
    """With checkpoint_dir set, decoders_ep{e}.pt is written at every epoch
    and loads back into a freshly built decoder stack."""
    model = _model()
    decoders = build_decoders(model, image_size=IMG)
    epochs = 2
    train_decoders(
        model, decoders, _loader(16), None,
        epochs=epochs, lr=1e-3, device=torch.device("cpu"), log_every=10,
        checkpoint_dir=tmp_path,
    )
    for e in range(1, epochs + 1):
        assert (tmp_path / f"decoders_ep{e}.pt").exists()
    fresh = build_decoders(model, image_size=IMG)
    fresh.load_state_dict(
        torch.load(tmp_path / f"decoders_ep{epochs}.pt", weights_only=True),
    )


def test_train_with_val_still_records_val_curve():
    """With a real val set the val history is populated (back-compat)."""
    history = train_model(
        _model(), _loader(16), _loader(8),
        epochs=3, lr=1e-3, device=torch.device("cpu"), log_every=10,
        early_stop_patience=99, early_stop_min_epochs=99,
    )
    assert len(history["val_loss"]) == 3
    assert history["best_val_loss"] is not None
