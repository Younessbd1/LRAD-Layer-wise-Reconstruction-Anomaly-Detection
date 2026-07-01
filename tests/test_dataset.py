"""Unit tests for the OOD split logic.

These exercise the pure helpers behind ``get_celeba_loaders`` on a synthetic
attribute matrix, so they need neither the CelebA download nor torchvision's
dataset machinery. The point is to pin the "OOD = any configured accessory"
rule and the back-compat handling of a single attribute name.
"""

from __future__ import annotations

import pytest
import torch

from lrad.dataset import (
    OOD_ATTRS,
    _resolve,
    _resolve_ood_attrs,
    _split_in_ood,
)


def test_resolve_ood_attrs_accepts_none_str_and_list():
    assert _resolve_ood_attrs(None) == list(OOD_ATTRS)
    assert _resolve_ood_attrs("Eyeglasses") == ["Eyeglasses"]
    assert _resolve_ood_attrs(["Eyeglasses", "Wearing_Hat"]) == [
        "Eyeglasses", "Wearing_Hat",
    ]


def test_resolve_ood_attrs_rejects_empty_list():
    with pytest.raises(ValueError):
        _resolve_ood_attrs([])


def _attr_matrix(rows: int = 6) -> torch.Tensor:
    """A blank (rows, 40) CelebA-shaped attribute matrix, all attributes off."""
    return torch.zeros(rows, 40, dtype=torch.long)


def test_split_in_ood_is_a_union_over_attributes():
    """A face is OOD as soon as any one accessory is present."""
    g, h = _resolve("Eyeglasses"), _resolve("Wearing_Hat")
    attr = _attr_matrix(6)
    attr[1, g] = 1                       # glasses only
    attr[2, h] = 1                       # hat only
    attr[3, g] = 1
    attr[3, h] = 1                       # both
    # rows 0, 4, 5 stay clean

    in_rows, ood_rows = _split_in_ood(attr, [g, h])
    assert in_rows == [0, 4, 5]
    assert ood_rows == [1, 2, 3]


def test_split_in_ood_single_attribute_matches_legacy():
    """One attribute reproduces the old glasses-only split exactly."""
    g = _resolve("Eyeglasses")
    attr = _attr_matrix(4)
    attr[2, g] = 1
    in_rows, ood_rows = _split_in_ood(attr, [g])
    assert in_rows == [0, 1, 3]
    assert ood_rows == [2]
