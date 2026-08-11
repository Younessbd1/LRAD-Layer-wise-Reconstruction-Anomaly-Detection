"""Tests for the MVTec AD dataset, loaders, and the pretext-only trunk.

Everything runs against a synthetic MVTec tree built by the ``mvtec_root``
fixture — same directory layout, same file-naming conventions, tiny images
— so the suite needs no 5 GB download.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from lrad.cutpaste import cutpaste_batch
from lrad.decoder import build_decoders
from lrad.model import build_model
from lrad.mvtec import (
    MVTEC_CATEGORIES,
    MVTEC_TEXTURES,
    MVTecCategory,
    category_root,
    get_mvtec_loaders,
)
from lrad.train import train_decoders, train_model

RAW = 24          # on-disk image size
IMG = 16          # after resize
CROP = 12         # after centre crop


def _write(path, size=RAW, value=128, mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), value, dtype=np.uint8)
    if mode == "RGB":
        arr = np.stack([arr] * 3, axis=-1)
    Image.fromarray(arr, mode=mode).save(path)


@pytest.fixture
def mvtec_root(tmp_path):
    """A synthetic two-category MVTec tree under ``<tmp>/mvtec_anomaly_detection``."""
    root = tmp_path / "data"
    base = root / "mvtec_anomaly_detection"
    for cat in ("bottle", "carpet"):
        d = base / cat
        for i in range(6):
            _write(d / "train" / "good" / f"{i:03d}.png", value=100 + i)
        for i in range(3):
            _write(d / "test" / "good" / f"{i:03d}.png", value=110 + i)
        for defect in ("broken", "contamination"):
            for i in range(2):
                _write(d / "test" / defect / f"{i:03d}.png", value=200)
                # Masks are single-channel, 0/255, named "<stem>_mask.png".
                m = np.zeros((RAW, RAW), dtype=np.uint8)
                m[4:10, 4:10] = 255
                (d / "ground_truth" / defect).mkdir(parents=True, exist_ok=True)
                Image.fromarray(m, mode="L").save(
                    d / "ground_truth" / defect / f"{i:03d}_mask.png"
                )
    return root


def _cfg(root, category="bottle", **over):
    cfg = {
        "dataset": {
            "root": str(root), "category": category,
            "image_size": IMG, "crop_size": CROP,
            "batch_size": 4, "num_workers": 0, "val_ratio": 0.0,
            "pin_memory": False, "seed": 0,
        },
    }
    cfg["dataset"].update(over)
    return cfg


# ---------------------------------------------------------------------------
# layout / splits
# ---------------------------------------------------------------------------

def test_categories_and_textures_are_consistent():
    assert len(MVTEC_CATEGORIES) == 15
    assert len(set(MVTEC_CATEGORIES)) == 15
    assert MVTEC_TEXTURES <= set(MVTEC_CATEGORIES)
    assert len(MVTEC_TEXTURES) == 5


def test_category_root_accepts_both_layouts(mvtec_root):
    """<root>/mvtec_anomaly_detection/<cat> and <root>/<cat> both resolve."""
    assert category_root(mvtec_root, "bottle").name == "bottle"
    nested = mvtec_root / "mvtec_anomaly_detection"
    assert category_root(nested, "bottle").name == "bottle"
    with pytest.raises(FileNotFoundError, match="not found"):
        category_root(mvtec_root, "nope")


def test_split_sizes_and_ood_labels(mvtec_root):
    loaders = get_mvtec_loaders(_cfg(mvtec_root))
    assert len(loaders["train_ds"]) == 6      # train/good
    assert loaders["val"] is None             # val_ratio = 0
    assert len(loaders["test_in_ds"]) == 3    # test/good
    assert len(loaders["test_ood_ds"]) == 4   # 2 defects x 2
    assert loaders["defect_types"] == ["broken", "contamination"]
    assert loaders["category"] == "bottle"

    # Nominal splits are labelled 0, the anomalous split 1 — the label the
    # AUROC is computed against.
    for key in ("train", "test_in"):
        assert all(int(b[3].max()) == 0 for b in loaders[key])
    assert all(int(b[3].min()) == 1 for b in loaders["test_ood"])


def test_batch_contract_matches_celeba(mvtec_root):
    """4-tuple (image, cls, attrs, is_ood) so shared code needs no branch."""
    loaders = get_mvtec_loaders(_cfg(mvtec_root))
    img, cls, attrs, is_ood = next(iter(loaders["train"]))
    assert img.shape == (4, 3, CROP, CROP)
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert cls.shape == (4,)
    assert attrs.shape == (4, 0)     # no attribute labels exist in MVTec
    assert is_ood.shape == (4,)


def test_val_split_carves_off_training_images(mvtec_root):
    loaders = get_mvtec_loaders(_cfg(mvtec_root, val_ratio=0.5))
    assert len(loaders["train_ds"]) == 3
    assert loaders["val"] is not None
    assert len(loaders["val"].dataset) == 3


def test_missing_category_raises(mvtec_root):
    with pytest.raises(FileNotFoundError):
        get_mvtec_loaders(_cfg(mvtec_root, category="hazelnut"))


def test_category_is_required(mvtec_root):
    cfg = _cfg(mvtec_root)
    del cfg["dataset"]["category"]
    with pytest.raises(ValueError, match="category is required"):
        get_mvtec_loaders(cfg)


def test_defective_train_split_is_rejected(mvtec_root):
    """A mis-extracted archive must fail loudly, not train on defects."""
    bad = mvtec_root / "mvtec_anomaly_detection" / "bottle" / "train" / "broken"
    _write(bad / "000.png")
    with pytest.raises(RuntimeError, match="nominal only"):
        get_mvtec_loaders(_cfg(mvtec_root))


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def test_masks_align_with_images_and_stay_binary(mvtec_root):
    loaders = get_mvtec_loaders(_cfg(mvtec_root))
    ood = loaders["test_ood_ds"]
    m = ood.load_mask(0)
    assert m.shape == (CROP, CROP)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}, "mask must stay {0,1}"
    assert m.sum() > 0, "defective image must have a non-empty mask"
    # Same geometry as the image tensor it belongs to.
    assert ood[0][0].shape[-2:] == m.shape


def test_nominal_images_get_an_all_zero_mask(mvtec_root):
    """test/good has no mask on disk; a zero mask is the correct truth."""
    loaders = get_mvtec_loaders(_cfg(mvtec_root))
    m = loaders["test_in_ds"].load_mask(0)
    assert m.shape == (CROP, CROP)
    assert float(m.sum()) == 0.0


def test_mask_uses_nearest_not_bilinear(tmp_path):
    """A downscaled mask must not acquire fractional edge values."""
    p = tmp_path / "test" / "d" / "000.png"
    _write(p, size=64)
    mp = tmp_path / "gt" / "d" / "000_mask.png"
    mp.parent.mkdir(parents=True)
    arr = np.zeros((64, 64), np.uint8)
    arr[10:20, 10:20] = 255
    Image.fromarray(arr, mode="L").save(mp)

    ds = MVTecCategory([p], ["d"], is_ood=1, image_size=16, crop_size=16,
                       mask_root=tmp_path / "gt")
    m = ds.load_mask(0)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


def test_grayscale_source_images_become_three_channel(tmp_path):
    """grid/screw/zipper ship single-channel PNGs."""
    p = tmp_path / "000.png"
    _write(p, mode="L")
    ds = MVTecCategory([p], ["good"], is_ood=0, image_size=IMG, crop_size=CROP)
    assert ds[0][0].shape == (3, CROP, CROP)


def test_crop_larger_than_resize_is_rejected(tmp_path):
    p = tmp_path / "000.png"
    _write(p)
    with pytest.raises(ValueError, match="exceeds image_size"):
        MVTecCategory([p], ["good"], is_ood=0, image_size=16, crop_size=32)


# ---------------------------------------------------------------------------
# the pretext-only trunk trains on MVTec batches
# ---------------------------------------------------------------------------

def _mvtec_model_cfg(**over):
    model = {
        "channels": [8, 16, 32], "kernel_size": 3,
        "gender_head": False, "attrs_head": False,
        "cutpaste_head": True, "cutpaste_classes": 3,
    }
    model.update(over)
    return {"model": model, "dataset": {"image_size": IMG, "crop_size": CROP}}


def test_model_without_supervised_heads_omits_their_logits():
    m = build_model(_mvtec_model_cfg())
    assert m.head_gender is None and m.head_attrs is None
    assert m.n_attrs == 0 and m.n_gender == 0
    assert m.input_size == CROP     # the trunk sees the CROP, not image_size
    out = m(torch.rand(2, 3, CROP, CROP))
    assert set(out) == {"cutpaste_logits"}
    assert out["cutpaste_logits"].shape == (2, 3)


def test_model_needs_at_least_one_head():
    with pytest.raises(ValueError, match="at least one head"):
        build_model(_mvtec_model_cfg(cutpaste_head=False))


def test_celeba_model_defaults_are_unchanged():
    """Both supervised heads stay on by default — no CelebA config churn."""
    m = build_model({"model": {"channels": [8, 16]}, "dataset": {"image_size": IMG}})
    assert m.head_gender is not None and m.head_attrs is not None
    assert m.head_cutpaste is None
    assert set(m(torch.rand(2, 3, IMG, IMG))) == {"gender_logits", "attr_logits"}


def test_three_way_cutpaste_labels_are_in_range():
    g = torch.Generator().manual_seed(0)
    _, labels = cutpaste_batch(
        torch.rand(64, 3, IMG, IMG), prob=1.0, scar_prob=0.5,
        three_way=True, generator=g,
    )
    uniq = set(labels.tolist())
    assert uniq <= {0, 1, 2}
    assert {1, 2} <= uniq, "both box and scar classes must be populated"


def test_binary_cutpaste_is_unaffected_by_the_new_flag():
    g = torch.Generator().manual_seed(0)
    _, labels = cutpaste_batch(
        torch.rand(32, 3, IMG, IMG), prob=1.0, scar_prob=0.5, generator=g,
    )
    assert set(labels.tolist()) <= {0, 1}


def test_train_pretext_only_trunk_and_decoders(mvtec_root):
    """The full member recipe runs on MVTec batches with no labels."""
    loaders = get_mvtec_loaders(_cfg(mvtec_root))
    model = build_model(_mvtec_model_cfg())
    device = torch.device("cpu")

    hist = train_model(
        model, loaders["train"], None, epochs=2, lr=1e-3, device=device,
        log_every=1,
        cutpaste={"prob": 0.5, "scar_prob": 0.5, "loss_weight": 1.0},
    )
    assert len(hist["train_loss"]) == 2
    assert all(np.isfinite(v) for v in hist["train_loss"])
    assert hist["batch_cutpaste_acc"], "pretext head must be trained"
    # No supervised head ran, so its accuracy series stays at the 0 filler
    # rather than carrying a meaningless number.
    assert set(hist["batch_gender_acc"]) == {0.0}

    decoders = build_decoders(model, image_size=CROP)
    dh = train_decoders(model, decoders, loaders["train"], None,
                        epochs=1, lr=1e-3, device=device, log_every=1)
    assert len(dh["train_loss_per_block"][-1]) == len(model.channels)
    assert all(np.isfinite(v) for v in dh["train_loss_per_block"][-1])
