"""MVTec AD dataset — per-category cold-start anomaly detection.

This is the MVTec counterpart of :mod:`lrad.dataset` (CelebA), built so the
rest of the pipeline (``lrad.train``, ``lrad.ensemble``, ``lrad.localized``,
``lrad.feature_error``, ``lrad.fusion``) runs unchanged on it.

**Why the protocol differs from CelebA.** MVTec AD [Bergmann et al., CVPR
2019] is a *cold-start* benchmark: the training set of each category holds
nominal images only, and carries **no labels of any kind** — no gender, no
attributes, nothing the CelebA heads could be trained against. The trunk is
therefore trained purely self-supervised, on the CutPaste pretext task
(``model.cutpaste_head: true``, ``model.gender_head/attrs_head: false``);
see :mod:`lrad.cutpaste`. Everything downstream — the frozen trunk, the
per-block decoders, the ensemble Bias/Variance decomposition — is identical
to the CelebA path.

Directory layout expected under ``dataset.root`` (the standard MVTec AD
release, unmodified)::

    <root>/mvtec_anomaly_detection/<category>/
        train/good/000.png ...              nominal, the only training data
        test/good/000.png ...               nominal test images
        test/<defect_type>/000.png ...      anomalous test images
        ground_truth/<defect_type>/000_mask.png

Split protocol, per category:

    train / val    : ``train/good`` (val carved off by ``val_ratio``)
    test_in        : ``test/good``
    test_ood       : every ``test/<defect>`` image, defect != "good"

Each sample yields the same 4-tuple the CelebA loaders emit, so no consumer
needs to know which dataset it is looking at::

    image      : (3, H, W) float tensor in [0, 1]
    cls_target : long scalar — always 0 (no class labels in MVTec; the
                 gender head is disabled, this slot is never read)
    attrs      : (0,) float tensor — empty (no attribute labels; the attr
                 head is disabled, this slot is never read)
    is_ood     : long scalar, 0 for nominal, 1 for defective

Preprocessing follows PatchCore for comparability: resize the shorter
representation to ``image_size`` then centre-crop to ``crop_size`` (256 →
224 in the paper). Ground-truth masks go through the *same* geometry with
nearest-neighbour interpolation so a mask pixel still lands on its image
pixel — see :func:`MVTecCategory.load_mask`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF

logger = logging.getLogger("celeba_ood")

# The 15 MVTec AD categories, in the order Table S1 of the PatchCore paper
# lists them (alphabetical), so our result tables line up with theirs
# column for column.
MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
)

# The five texture categories; the other ten are objects. Textures are
# translation-invariant, which matters for the localized scorer: its
# per-pixel reference statistics assume a spatially stable layout (true for
# aligned CelebA faces and for MVTec objects, only weakly true for textures).
MVTEC_TEXTURES: frozenset[str] = frozenset(
    {"carpet", "grid", "leather", "tile", "wood"}
)

DEFAULT_DIRNAME = "mvtec_anomaly_detection"
NOMINAL_DIR = "good"
_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def category_root(root: str | Path, category: str) -> Path:
    """Resolve ``<root>/<dirname>/<category>``, tolerating either layout.

    The MVTec tarball unpacks to a ``mvtec_anomaly_detection/`` directory,
    but people routinely point ``dataset.root`` straight at that directory.
    Accept both rather than making the config order-sensitive.
    """
    root = Path(root)
    for candidate in (root / DEFAULT_DIRNAME / category, root / category):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"MVTec category {category!r} not found under {root} — looked in "
        f"{root / DEFAULT_DIRNAME / category} and {root / category}. "
        f"Run scripts/download_mvtec.py to fetch the dataset."
    )


def _sorted_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in _IMG_SUFFIXES)


class MVTecCategory(Dataset):
    """One split of one MVTec category.

    Holds ``(image_path, defect_type)`` pairs and, for anomalous images,
    knows where the ground-truth mask lives. ``is_ood`` is fixed per split
    (0 for nominal splits, 1 for the anomalous one) exactly as in
    :class:`lrad.dataset.CelebAFacialAttributes`.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        defects: Sequence[str],
        *,
        is_ood: int,
        image_size: int = 256,
        crop_size: int | None = 224,
        mask_root: Path | None = None,
    ):
        if len(paths) != len(defects):
            raise ValueError("paths and defects must have the same length")
        self.paths = [Path(p) for p in paths]
        self.defects = list(defects)
        self.is_ood = int(is_ood)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size) if crop_size else int(image_size)
        self.mask_root = Path(mask_root) if mask_root is not None else None

        if self.crop_size > self.image_size:
            raise ValueError(
                f"crop_size {self.crop_size} exceeds image_size "
                f"{self.image_size}"
            )
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])
        # Empty attribute vector: MVTec has no attribute labels, and the
        # attr head is disabled. Shared across items (never mutated) so we
        # do not allocate one per __getitem__.
        self._empty_attrs = torch.zeros(0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_rgb(self, i: int):
        from PIL import Image
        # A few MVTec categories (grid, screw, zipper) ship single-channel
        # PNGs; convert unconditionally so every tensor is (3, H, W).
        return Image.open(self.paths[i]).convert("RGB")

    def __getitem__(self, i: int):
        img = self.transform(self._load_rgb(i))
        return img, 0, self._empty_attrs, self.is_ood

    def load_mask(self, i: int) -> torch.Tensor:
        """Ground-truth defect mask for item ``i`` as ``(H, W)`` in {0, 1}.

        Nominal images have no mask file on disk — MVTec ships masks only
        for defective images — so an all-zero mask is synthesized for them.
        That is the correct ground truth (no defective pixel) and is what
        keeps ``test/good`` images contributing true negatives to the pixel
        AUROC instead of being silently dropped.

        The mask goes through the same resize + centre-crop as the image
        but with NEAREST interpolation: bilinear would blend the {0, 1}
        boundary into fractional values and blur the defect edge, which
        biases both pixel AUROC and PRO.
        """
        from PIL import Image

        defect = self.defects[i]
        mask_path: Path | None = None
        if defect != NOMINAL_DIR and self.mask_root is not None:
            stem = self.paths[i].stem
            cand = self.mask_root / defect / f"{stem}_mask.png"
            if not cand.exists():  # a few releases drop the _mask suffix
                cand = self.mask_root / defect / f"{stem}.png"
            mask_path = cand if cand.exists() else None

        if mask_path is None:
            return torch.zeros(self.crop_size, self.crop_size,
                               dtype=torch.float32)

        m = Image.open(mask_path).convert("L")
        m = TF.resize(
            m, [self.image_size, self.image_size],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        m = TF.center_crop(m, [self.crop_size, self.crop_size])
        t = TF.to_tensor(m)[0]              # (H, W) in [0, 1]
        return (t > 0.5).to(torch.float32)  # binarize: masks are 0/255

    def defect_names(self) -> list[str]:
        return sorted(set(self.defects))


def _scan_split(
    cat_dir: Path, split: str,
) -> tuple[list[Path], list[str]]:
    """List ``(paths, defect_types)`` under ``<cat_dir>/<split>``."""
    split_dir = cat_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory {split_dir}")
    paths: list[Path] = []
    defects: list[str] = []
    for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        imgs = _sorted_images(sub)
        paths.extend(imgs)
        defects.extend([sub.name] * len(imgs))
    return paths, defects


def get_mvtec_loaders(cfg: dict) -> dict:
    """Build train / val / test_in / test_ood loaders for ONE category.

    Reads ``cfg["dataset"]``:
        root          parent dir holding ``mvtec_anomaly_detection/``
                      (default "./data")
        category      MVTec category name (required, e.g. "bottle")
        image_size    resize before cropping (default 256)
        crop_size     centre crop, the tensor the model sees (default 224);
                      set equal to image_size to disable cropping
        batch_size    default 32
        num_workers   default 4
        val_ratio     fraction of train/good held out for validation
                      (default 0.0 — the CelebA runs use no val split)
        pin_memory    default True
        seed          split RNG seed (default 42)

    Returns a dict with the same keys the CelebA loaders return, so callers
    stay dataset-agnostic, plus ``category``, ``defect_types`` and the
    ``test_in_ds`` / ``test_ood_ds`` datasets (needed by the pixel-level
    metrics, which read ground-truth masks the 4-tuple cannot carry).
    """
    dcfg = cfg.get("dataset", {})
    root = dcfg.get("root", "./data")
    category = dcfg.get("category")
    if not category:
        raise ValueError(
            "dataset.category is required for MVTec (one of "
            f"{', '.join(MVTEC_CATEGORIES)})"
        )
    if category not in MVTEC_CATEGORIES:
        logger.warning(
            "category %r is not one of the 15 standard MVTec categories",
            category,
        )

    image_size = dcfg.get("image_size", 256)
    crop_size = dcfg.get("crop_size", 224)
    batch_size = dcfg.get("batch_size", 32)
    num_workers = dcfg.get("num_workers", 4)
    val_ratio = dcfg.get("val_ratio", 0.0)
    pin_memory = dcfg.get("pin_memory", True)
    seed = dcfg.get("seed", 42)

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    cat_dir = category_root(root, category)
    mask_root = cat_dir / "ground_truth"

    train_paths, train_defects = _scan_split(cat_dir, "train")
    test_paths, test_defects = _scan_split(cat_dir, "test")
    if not train_paths:
        raise RuntimeError(f"no training images under {cat_dir / 'train'}")

    # MVTec's train split is nominal by construction; assert it rather than
    # trusting it, because a mis-extracted archive would silently train the
    # "cold-start" model on defects and quietly invalidate the whole run.
    stray = sorted(set(train_defects) - {NOMINAL_DIR})
    if stray:
        raise RuntimeError(
            f"{category}: train/ contains non-nominal subdirs {stray} — "
            "MVTec training data must be nominal only"
        )

    in_idx = [i for i, d in enumerate(test_defects) if d == NOMINAL_DIR]
    ood_idx = [i for i, d in enumerate(test_defects) if d != NOMINAL_DIR]
    if not in_idx or not ood_idx:
        raise RuntimeError(
            f"{category}: test split has {len(in_idx)} nominal and "
            f"{len(ood_idx)} anomalous images — expected both to be non-empty"
        )

    common = dict(image_size=image_size, crop_size=crop_size,
                  mask_root=mask_root)

    n = len(train_paths)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val
    if n_val > 0:
        gen = torch.Generator().manual_seed(seed)
        tr_sub, va_sub = random_split(range(n), [n_train, n_val],
                                      generator=gen)
        tr_rows, va_rows = list(tr_sub.indices), list(va_sub.indices)
    else:
        tr_rows, va_rows = list(range(n)), []

    def _nominal_ds(rows: list[int]) -> MVTecCategory:
        return MVTecCategory(
            [train_paths[i] for i in rows],
            [train_defects[i] for i in rows],
            is_ood=0, **common,
        )

    train_ds = _nominal_ds(tr_rows)
    val_ds = _nominal_ds(va_rows) if va_rows else None
    test_in_ds = MVTecCategory(
        [test_paths[i] for i in in_idx], [test_defects[i] for i in in_idx],
        is_ood=0, **common,
    )
    test_ood_ds = MVTecCategory(
        [test_paths[i] for i in ood_idx], [test_defects[i] for i in ood_idx],
        is_ood=1, **common,
    )

    defect_types = test_ood_ds.defect_names()
    logger.info(
        "MVTec %s: train=%d val=%d test_in=%d test_ood=%d  defects=[%s]",
        category, len(train_ds), len(val_ds) if val_ds else 0,
        len(test_in_ds), len(test_ood_ds), ", ".join(defect_types),
    )

    def _make(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    # ``val_ratio=0`` yields ``val=None``, the same contract the CelebA
    # loaders use: the training loops read None as "no validation, run the
    # full schedule" rather than early-stopping on an empty loader.
    return {
        "train": _make(train_ds, shuffle=True),
        "val": _make(val_ds, shuffle=False) if val_ds is not None else None,
        "test_in": _make(test_in_ds, shuffle=False),
        "test_ood": _make(test_ood_ds, shuffle=False),
        "train_ds": train_ds,
        "test_in_ds": test_in_ds,
        "test_ood_ds": test_ood_ds,
        "category": category,
        "defect_types": defect_types,
        # Mirrors the CelebA keys so shared code can log a task label
        # without branching on the dataset.
        "gender_attr": None,
        "attr_targets": [],
        "ood_attr": f"mvtec:{category}",
        "ood_attrs": defect_types,
    }
