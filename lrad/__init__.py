"""CelebA gender + attribute classifier with confidence-based OOD detection."""

__version__ = "0.3.0"

from .anomaly_score import (
    aggregate_anomaly_score,
    cleaned_anomaly_maps,
    estimate_baseline_stats,
)
from .dataset import (
    ATTR_TARGETS,
    CELEBA_ATTRS,
    GENDER_ATTR,
    OOD_ATTR,
    CelebAFacialAttributes,
    get_celeba_loaders,
)
from .decoder import BlockDecoder, build_decoders
from .model import FacialCNN, build_model, count_parameters
from .plots import (
    plot_anomaly_cleaning_comparison,
    plot_bias_variance_maps,
    plot_per_block_auroc_bars,
    plot_per_block_breakdown,
    plot_recons_only,
    plot_score_distribution_comparison,
)
from .train import evaluate_one_epoch, train_decoders, train_model
from .evaluate import (
    collect_debiased_scores,
    collect_predictions,
    evaluate,
    evaluate_with_debiasing,
    ood_auroc,
    per_attribute_accuracy,
)
from .utils import get_device, seed_everything, setup_logging

__all__ = [
    "ATTR_TARGETS",
    "BlockDecoder",
    "CELEBA_ATTRS",
    "CelebAFacialAttributes",
    "FacialCNN",
    "GENDER_ATTR",
    "OOD_ATTR",
    "aggregate_anomaly_score",
    "build_decoders",
    "build_model",
    "cleaned_anomaly_maps",
    "collect_debiased_scores",
    "collect_predictions",
    "count_parameters",
    "estimate_baseline_stats",
    "evaluate",
    "evaluate_one_epoch",
    "evaluate_with_debiasing",
    "get_celeba_loaders",
    "get_device",
    "ood_auroc",
    "per_attribute_accuracy",
    "plot_anomaly_cleaning_comparison",
    "plot_bias_variance_maps",
    "plot_per_block_auroc_bars",
    "plot_per_block_breakdown",
    "plot_recons_only",
    "plot_score_distribution_comparison",
    "seed_everything",
    "setup_logging",
    "train_decoders",
    "train_model",
]
