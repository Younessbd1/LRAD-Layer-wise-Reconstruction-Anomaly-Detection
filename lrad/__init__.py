"""CelebA gender + attribute classifier with confidence-based OOD detection."""

__version__ = "0.3.0"

from .dataset import (
    ATTR_TARGETS,
    CELEBA_ATTRS,
    GENDER_ATTR,
    OOD_ATTR,
    CelebAFacialAttributes,
    get_celeba_loaders,
)
from .model import FacialCNN, build_model, count_parameters
from .train import evaluate_one_epoch, train_model
from .evaluate import (
    collect_predictions,
    evaluate,
    ood_auroc,
    per_attribute_accuracy,
)
from .utils import get_device, seed_everything, setup_logging

__all__ = [
    "ATTR_TARGETS",
    "CELEBA_ATTRS",
    "CelebAFacialAttributes",
    "FacialCNN",
    "GENDER_ATTR",
    "OOD_ATTR",
    "build_model",
    "collect_predictions",
    "count_parameters",
    "evaluate",
    "evaluate_one_epoch",
    "get_celeba_loaders",
    "get_device",
    "ood_auroc",
    "per_attribute_accuracy",
    "seed_everything",
    "setup_logging",
    "train_model",
]
