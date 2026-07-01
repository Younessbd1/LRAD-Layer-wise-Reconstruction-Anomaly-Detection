"""CelebA gender + attribute classifier with confidence-based OOD detection."""

__version__ = "0.3.0"

from .anomaly_score import aggregate_anomaly_score
from .dataset import (
    ATTR_TARGETS,
    CELEBA_ATTRS,
    GENDER_ATTR,
    OOD_ATTR,
    OOD_ATTRS,
    CelebAFacialAttributes,
    get_celeba_loaders,
)
from .decoder import BlockDecoder, build_decoders
from .ensemble import (
    collect_decomposition_scores,
    decomposition_maps,
    evaluate_ensemble_decomposition,
    identity_residual,
    sample_block_recons,
    sample_decomposition,
)
from .model import FacialCNN, build_model, count_parameters
from .plots import (
    plot_bias_variance_vs_block,
    plot_bias_variance_vs_percentile,
    plot_decomposition_auroc_bars,
    plot_ensemble_decomposition,
    plot_ensemble_score_hists,
    plot_instance_decomposition,
    plot_mean_abs_bias,
    plot_per_block_breakdown,
    plot_recons_only,
    plot_variance_heatmaps,
    smooth_cam,
)
from .train import evaluate_one_epoch, train_decoders, train_model
from .evaluate import (
    collect_predictions,
    evaluate,
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
    "OOD_ATTRS",
    "aggregate_anomaly_score",
    "build_decoders",
    "build_model",
    "collect_decomposition_scores",
    "collect_predictions",
    "count_parameters",
    "decomposition_maps",
    "evaluate",
    "evaluate_ensemble_decomposition",
    "evaluate_one_epoch",
    "get_celeba_loaders",
    "get_device",
    "identity_residual",
    "ood_auroc",
    "per_attribute_accuracy",
    "plot_bias_variance_vs_block",
    "plot_bias_variance_vs_percentile",
    "plot_decomposition_auroc_bars",
    "plot_ensemble_decomposition",
    "plot_ensemble_score_hists",
    "plot_instance_decomposition",
    "plot_mean_abs_bias",
    "plot_per_block_breakdown",
    "plot_recons_only",
    "plot_variance_heatmaps",
    "sample_block_recons",
    "sample_decomposition",
    "seed_everything",
    "setup_logging",
    "smooth_cam",
    "train_decoders",
    "train_model",
]
