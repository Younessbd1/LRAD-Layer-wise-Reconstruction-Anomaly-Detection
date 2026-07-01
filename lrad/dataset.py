"""CelebA dataset for the gender + facial-attribute OOD task.

OOD is defined by a *set* of accessory attributes (``dataset.ood_attrs``,
default ``[Eyeglasses, Wearing_Hat]``). A face is out-of-distribution as
soon as it carries any one of them; the in-distribution pool is the clean
faces that carry none.

In-distribution protocol:
    train / val / test_in  : faces with none of the OOD attributes set
                             (no glasses, no sunglasses, no hat) — the
                             "clean" faces the classifier trains on
    test_ood               : faces with at least one OOD attribute set,
                             held out and only seen at evaluation time

Each in-distribution sample yields:

    image          : (3, H, W) float tensor in [0, 1]
    gender         : long scalar in {0, 1}     (1 == Male, 0 == Female)
    attrs          : (6,) float tensor of binary {0, 1} targets, in this order
                     [Young, Smiling, Mouth_Slightly_Open, High_Cheekbones,
                      Pointy_Nose, Oval_Face]
    is_ood         : long scalar (0 for in-dist, 1 for OOD).
                     Always 0 for the train/val/test_in loaders, 1 for
                     test_ood. Useful for the evaluator to assemble a
                     single (score, label) pair across the in/ood split.

The OOD attributes (and the other accessory attributes — Wearing_Earrings,
Heavy_Makeup) are deliberately *not* exposed as targets: we want the encoder
to learn identity-/expression-level facial features, not the accessories
themselves. Holding glasses and hats out entirely is what makes them a clean
OOD test — the model never gets to fit the occlusion they cause.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger("celeba_ood")


CELEBA_ATTRS: tuple[str, ...] = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
)

GENDER_ATTR: str = "Male"

# OOD is the union of these accessory attributes (any one present -> OOD).
# OOD_ATTR is kept as the historical single-attribute name; OOD_ATTRS is the
# default set the config falls back to when dataset.ood_attrs is unset.
OOD_ATTR: str = "Eyeglasses"
OOD_ATTRS: tuple[str, ...] = ("Eyeglasses", "Wearing_Hat")

ATTR_TARGETS: tuple[str, ...] = (
    "Young",
    "Smiling",
    "Mouth_Slightly_Open",
    "High_Cheekbones",
    "Pointy_Nose",
    "Oval_Face",
)


def _resolve(name: str) -> int:
    return CELEBA_ATTRS.index(name)


def _resolve_ood_attrs(value) -> list[str]:
    """Normalize the configured OOD attribute(s) to a list of names.

    Accepts a single name (the old ``dataset.ood_attr`` string), a list of
    names (``dataset.ood_attrs``), or ``None`` (fall back to ``OOD_ATTRS``).
    """
    if value is None:
        return list(OOD_ATTRS)
    if isinstance(value, str):
        return [value]
    names = list(value)
    if not names:
        raise ValueError("ood_attrs is empty — give at least one attribute")
    return names


def _split_in_ood(
    attr: torch.Tensor, ood_indices: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Partition row indices into (in-distribution, OOD).

    ``attr`` is the ``(N, 40)`` CelebA attribute matrix in {0, 1}. A row is
    OOD if *any* of the ``ood_indices`` columns is set, in-distribution if
    *all* of them are off.
    """
    cols = attr[:, list(ood_indices)]            # (N, k)
    ood_any = cols.to(torch.bool).any(dim=1)     # (N,)
    in_rows = (~ood_any).nonzero(as_tuple=True)[0].tolist()
    ood_rows = ood_any.nonzero(as_tuple=True)[0].tolist()
    return in_rows, ood_rows


_GENDER_IDX = _resolve(GENDER_ATTR)
_ATTR_IDX = [_resolve(a) for a in ATTR_TARGETS]


class CelebAFacialAttributes(Dataset):
    """Wraps a torchvision CelebA dataset, indexes a curated subset, and
    returns ``(image, gender, attrs, is_ood)`` 4-tuples."""

    def __init__(
        self,
        base: Dataset,
        indices: Sequence[int],
        is_ood: int,
    ):
        self.base = base
        self.indices = list(indices)
        self.is_ood = int(is_ood)

        attr = base.attr  # (N, 40) long tensor in {0, 1}
        self._gender = attr[:, _GENDER_IDX].to(torch.long)
        self._attrs = attr[:, _ATTR_IDX].to(torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gi = self.indices[i]
        img, _ = self.base[gi]
        return img, self._gender[gi], self._attrs[gi], self.is_ood


def get_celeba_loaders(cfg: dict) -> dict:
    """Build train / val / test_in / test_ood dataloaders.

    Reads ``cfg["dataset"]``:
        root          parent dir holding ``celeba/`` (default "./data")
        download      torchvision auto-download (default False)
        image_size    square resize (default 64)
        batch_size    default 64
        num_workers   default 4
        train_ratio   fraction of in-dist pool used for training (default 0.80)
        val_ratio     fraction used for validation (default 0.10)
        pin_memory    default True
        seed          split RNG seed (default 42)
        ood_attrs     attribute name or list defining OOD (default
                      ``[Eyeglasses, Wearing_Hat]``); ``ood_attr`` (singular)
                      is still accepted for back-compat

    Returns:
        dict with keys 'train', 'val', 'test_in', 'test_ood',
        'gender_attr', 'attr_targets', 'ood_attr' (display string) and
        'ood_attrs' (resolved list of names).
    """
    dcfg = cfg.get("dataset", {})
    root = dcfg.get("root", "./data")
    download = dcfg.get("download", False)
    image_size = dcfg.get("image_size", 64)
    batch_size = dcfg.get("batch_size", 64)
    num_workers = dcfg.get("num_workers", 4)
    train_ratio = dcfg.get("train_ratio", 0.80)
    val_ratio = dcfg.get("val_ratio", 0.10)
    pin_memory = dcfg.get("pin_memory", True)
    seed = dcfg.get("seed", 42)

    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    base = datasets.CelebA(
        root=root,
        split="all",
        target_type="attr",
        transform=transform,
        download=download,
    )

    ood_names = _resolve_ood_attrs(dcfg.get("ood_attrs", dcfg.get("ood_attr")))
    ood_indices = [_resolve(n) for n in ood_names]
    in_idx, ood_idx = _split_in_ood(base.attr, ood_indices)
    if not in_idx or not ood_idx:
        raise RuntimeError(
            "CelebA splits are empty — check the ood_attrs columns "
            f"{ood_names}"
        )
    logger.info(
        "OOD attributes: %s  →  in-dist=%d  ood=%d images",
        "+".join(ood_names), len(in_idx), len(ood_idx),
    )

    n = len(in_idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        in_idx, [n_train, n_val, n_test], generator=gen,
    )
    train_indices = [in_idx[i] for i in train_sub.indices]
    val_indices = [in_idx[i] for i in val_sub.indices]
    test_indices = [in_idx[i] for i in test_sub.indices]

    train_attrs = base.attr[train_indices][:, _ATTR_IDX].float()
    train_gender = base.attr[train_indices][:, _GENDER_IDX].float()
    logger.info(
        "Train pos rates: Male=%.1f%%  " % (train_gender.mean() * 100)
        + "  ".join(f"{n}={p*100:.1f}%" for n, p in
                    zip(ATTR_TARGETS, train_attrs.mean(dim=0).tolist()))
    )

    def _make(idx: list[int], is_ood: int, shuffle: bool) -> DataLoader:
        ds = CelebAFacialAttributes(base, idx, is_ood=is_ood)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    # ``val_ratio=0`` disables validation entirely (see the config note):
    # there is no split to wrap, so expose ``val=None`` rather than an empty
    # DataLoader. An empty loader would make every val metric collapse to 0
    # and silently trigger early stopping on a constant val_loss — the
    # training loops treat ``None`` as "no validation, run the full schedule".
    if val_indices:
        val_loader: DataLoader | None = _make(
            val_indices, is_ood=0, shuffle=False,
        )
    else:
        val_loader = None
        logger.info(
            "val_ratio=%.4g → no validation split: %d in-dist images train, "
            "%d held out for test_in (no early stopping, full schedule).",
            val_ratio, len(train_indices), len(test_indices),
        )

    return {
        "train": _make(train_indices, is_ood=0, shuffle=True),
        "val": val_loader,
        "test_in": _make(test_indices, is_ood=0, shuffle=False),
        "test_ood": _make(ood_idx, is_ood=1, shuffle=False),
        "gender_attr": GENDER_ATTR,
        "attr_targets": list(ATTR_TARGETS),
        # "ood_attr" stays a single display string for logs/summaries;
        # "ood_attrs" is the resolved list the split was actually built from.
        "ood_attr": "+".join(ood_names),
        "ood_attrs": ood_names,
    }
