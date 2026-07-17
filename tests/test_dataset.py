"""Unit tests for the OOD split logic.

These exercise the pure helpers behind ``get_celeba_loaders`` on a synthetic
attribute matrix, so they need neither the CelebA download nor torchvision's
dataset machinery. The point is to pin the "OOD = any configured accessory"
rule and the back-compat handling of a single attribute name.
"""

from __future__ import annotations

import pytest
import torch

from torch.utils.data import DataLoader, TensorDataset

from lrad.dataset import (
    OOD_ATTRS,
    _resolve,
    _resolve_ood_attrs,
    _split_in_ood,
    load_generalization_images,
    split_loader,
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


# ---------------------------------------------------------------------------
# load_generalization_images — arbitrary (non-CelebA) photos for
# scripts/run_generalization.py
# ---------------------------------------------------------------------------

def _write_photo(path, size: tuple[int, int], color) -> None:
    from PIL import Image

    Image.new("RGB", size, color=color).save(path)


def test_load_generalization_images_shape_and_range(tmp_path):
    # A landscape and a portrait photo — both must come out square.
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.png"
    _write_photo(p1, (120, 80), (255, 0, 0))
    _write_photo(p2, (60, 100), (0, 255, 0))

    out = load_generalization_images([p1, p2], image_size=32)
    assert out.shape == (2, 3, 32, 32)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_load_generalization_images_rejects_empty():
    with pytest.raises(ValueError):
        load_generalization_images([], image_size=32)


def test_load_generalization_images_handles_grayscale_and_rgba(tmp_path):
    from PIL import Image

    p_gray = tmp_path / "gray.png"
    p_rgba = tmp_path / "rgba.png"
    Image.new("L", (300, 900), 128).save(p_gray)     # grayscale, portrait
    Image.new("RGBA", (50, 50), (1, 2, 3, 255)).save(p_rgba)  # RGBA, tiny

    out = load_generalization_images([p_gray, p_rgba], image_size=64)
    assert out.shape == (2, 3, 64, 64)


def test_load_generalization_images_corrects_exif_orientation(tmp_path):
    """A phone photo stored sideways (EXIF orientation tag) must be rotated
    to its true orientation before the square crop, or the crop would cut
    along the wrong axis."""
    from PIL import Image

    img = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90 CW to display correctly"
    path = tmp_path / "sideways.jpg"
    img.save(path, format="JPEG", exif=exif)

    out = load_generalization_images([path], image_size=64)
    assert out.shape == (1, 3, 64, 64)


def _tensor_loader(n: int = 20, batch: int = 4) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.arange(n).float().view(-1, 1)),
        batch_size=batch, shuffle=False, num_workers=0,
    )


def test_split_loader_disjoint_and_exhaustive():
    a, b = split_loader(_tensor_loader(), 7, seed=0)
    va = torch.cat([x for (x,) in a]).ravel()
    vb = torch.cat([x for (x,) in b]).ravel()
    assert len(va) == 7 and len(vb) == 13
    merged = torch.cat([va, vb])
    assert set(merged.tolist()) == set(range(20))  # disjoint + exhaustive


def test_split_loader_deterministic_and_seed_sensitive():
    a1, _ = split_loader(_tensor_loader(), 7, seed=0)
    a2, _ = split_loader(_tensor_loader(), 7, seed=0)
    a3, _ = split_loader(_tensor_loader(), 7, seed=1)
    v1 = torch.cat([x for (x,) in a1])
    v2 = torch.cat([x for (x,) in a2])
    v3 = torch.cat([x for (x,) in a3])
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, v3)


def test_split_loader_preserves_batch_size_and_validates():
    a, b = split_loader(_tensor_loader(batch=4), 7, seed=0)
    assert a.batch_size == 4 and b.batch_size == 4
    with pytest.raises(ValueError):
        split_loader(_tensor_loader(), 0)
    with pytest.raises(ValueError):
        split_loader(_tensor_loader(), 20)
