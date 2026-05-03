"""Tests for the CelebA loader bundle.

Every test mocks ``torchvision.datasets.CelebA`` so the suite never touches
the 1.4 GB CelebA download. The mock returns a controllable in-memory
``attr`` matrix and dummy 3x64x64 image tensors.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock CelebA
# ---------------------------------------------------------------------------

def _mock_celeba_class(
    n_total: int,
    attr_idx: int,
    n_anomaly: int,
    pretext_idx_with_pos: dict[int, int] | None = None,
):
    """Factory for a torchvision.datasets.CelebA stand-in.

    The first ``n_anomaly`` rows have ``attr[:, attr_idx] == 1`` (anomalies),
    every other row is a normal. ``pretext_idx_with_pos`` optionally seeds
    a positive count for any pretext attribute column so the per-attribute
    balance log line in the loader has something to report. All images are
    zero tensors (no I/O).
    """

    pretext_idx_with_pos = pretext_idx_with_pos or {}

    class _MockCelebA(torch.utils.data.Dataset):
        def __init__(self, root, split, target_type, transform, download):
            self.root = root
            self.split = split
            self.target_type = target_type
            self.transform = transform
            self.download = download
            attr = torch.zeros((n_total, 40), dtype=torch.long)
            attr[:n_anomaly, attr_idx] = 1
            for col, n_pos in pretext_idx_with_pos.items():
                attr[:n_pos, col] = 1
            self.attr = attr

        def __len__(self):
            return n_total

        def __getitem__(self, i):
            return torch.zeros(3, 64, 64), self.attr[i]

    return _MockCelebA


def _base_cfg(**dataset_overrides) -> dict:
    cfg = {
        "experiment": {"seed": 42},
        "dataset": {
            "anomaly_attr": "Eyeglasses",
            "pretext_attrs": ["Male", "Young", "Smiling"],
            "image_size": 64,
            "batch_size": 8,
            "num_workers": 0,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "pin_memory": False,
            "root": "/tmp/fake-celeba",
            "download": False,
        },
    }
    cfg["dataset"].update(dataset_overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1. Attribute resolution — known name
# ---------------------------------------------------------------------------

def test_resolve_known_attribute_returns_official_index():
    """Eyeglasses is the 16th attribute in the official CelebA list,
    i.e. column index 15 (0-based)."""
    from lrad.data.celeba import _resolve_attr_index, CELEBA_ATTRS

    assert _resolve_attr_index("Eyeglasses") == 15
    assert CELEBA_ATTRS[15] == "Eyeglasses"
    assert len(CELEBA_ATTRS) == 40


# ---------------------------------------------------------------------------
# 2. Attribute resolution — unknown name raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_attribute_raises_value_error():
    from lrad.data.celeba import _resolve_attr_index

    with pytest.raises(ValueError, match="unknown CelebA attribute"):
        _resolve_attr_index("NotAnAttribute")


# ---------------------------------------------------------------------------
# 3. Split ratios + 2-tuple batches
# ---------------------------------------------------------------------------

def test_split_ratios_match_config():
    """train/val/test_normal must respect train_ratio / val_ratio on the
    normal pool, and test_anomaly must contain every image with the target
    attribute set. Each loader must yield
    ``(images, anomaly_label, pretext_labels)`` 3-tuples matching the
    LRAD engine convention."""
    n_total = 200
    n_anomaly = 60
    n_normal = n_total - n_anomaly  # 140
    train_ratio = 0.8
    val_ratio = 0.1

    # Seed a few positives in the pretext attribute columns so the
    # multi-hot vectors have a meaningful distribution.
    # Eyeglasses=15, Male=20, Smiling=31, Young=39.
    mock_cls = _mock_celeba_class(
        n_total=n_total, attr_idx=15, n_anomaly=n_anomaly,
        pretext_idx_with_pos={20: 90, 39: 110, 31: 80},
    )
    cfg = _base_cfg(train_ratio=train_ratio, val_ratio=val_ratio)

    with patch("lrad.data.celeba.datasets.CelebA", mock_cls):
        from lrad.data.celeba import get_celeba_loaders
        loaders = get_celeba_loaders(cfg)

    expected_train = int(round(n_normal * train_ratio))
    expected_val = int(round(n_normal * val_ratio))
    expected_test_normal = n_normal - expected_train - expected_val

    assert len(loaders["train"].dataset) == expected_train
    assert len(loaders["val"].dataset) == expected_val
    assert len(loaders["test_normal"].dataset) == expected_test_normal
    assert len(loaders["test_anomaly"].dataset) == n_anomaly

    # Sanity: the four splits cover the full dataset exactly once.
    total = (len(loaders["train"].dataset)
             + len(loaders["val"].dataset)
             + len(loaders["test_normal"].dataset)
             + len(loaders["test_anomaly"].dataset))
    assert total == n_total

    # Batch contract: (images, anomaly_label, pretext_labels).
    images, labels, pretext = next(iter(loaders["train"]))
    assert images.dim() == 4 and images.size(1) == 3
    assert labels.dtype == torch.long
    assert torch.all(labels == 0)
    # Pretext labels: float multi-hot of shape (B, P=3) in {0, 1}.
    assert pretext.dim() == 2 and pretext.size(1) == 3
    assert pretext.dtype == torch.float32
    assert torch.all((pretext == 0) | (pretext == 1))

    images, labels, pretext = next(iter(loaders["test_anomaly"]))
    assert torch.all(labels == 1)
    assert pretext.size(1) == 3

    # Loader bundle exposes the resolved pretext attr names.
    assert loaders["pretext_attrs"] == ["Male", "Young", "Smiling"]


# ---------------------------------------------------------------------------
# 4. has_masks is always False
# ---------------------------------------------------------------------------

def test_has_masks_is_always_false():
    """CelebA ships no pixel masks — ``has_masks`` must be False so the
    runner falls back to image-level metrics only."""
    mock_cls = _mock_celeba_class(n_total=100, attr_idx=15, n_anomaly=30)
    cfg = _base_cfg()

    with patch("lrad.data.celeba.datasets.CelebA", mock_cls):
        from lrad.data.celeba import get_celeba_loaders
        loaders = get_celeba_loaders(cfg)

    assert loaders["has_masks"] is False
    assert loaders["attr_name"] == "Eyeglasses"
