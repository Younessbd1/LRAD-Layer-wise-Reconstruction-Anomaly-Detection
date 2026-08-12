"""LRAD — layer-wise reconstruction anomaly detection.

Two tasks share this pipeline:

* **CelebA** (:mod:`lrad.dataset`) — supervised gender + attribute trunk,
  eyeglasses held out as OOD.
* **MVTec AD** (:mod:`lrad.mvtec`) — cold-start industrial anomaly
  detection, benchmarked against PatchCore. No labels exist, so the trunk
  trains on the CutPaste pretext task alone and localization is scored with
  pixel AUROC + PRO (:mod:`lrad.pixel_metrics`).
"""

__version__ = "0.3.0"

from .anomaly_score import aggregate_anomaly_score
from .arch_diagram import (
    classifier_n_params,
    decoder_n_params,
    render_classifier_svg,
    render_decoder_svg,
    render_ensemble_svg,
    resolve_member_configs,
)
from .config import apply_overrides, load_config, to_jsonable
from .dataset import (
    ATTR_TARGETS,
    CELEBA_ATTRS,
    GENDER_ATTR,
    OOD_ATTR,
    OOD_ATTRS,
    CelebAFacialAttributes,
    gather_samples,
    get_celeba_loaders,
)
from .decoder import BlockDecoder, build_decoders
from .ensemble import (
    collect_decomposition_scores,
    collect_eye_region_bias,
    decomposition_maps,
    evaluate_ensemble_decomposition,
    identity_residual,
    sample_block_recons,
    sample_decomposition,
)
from .evaluate import (
    collect_predictions,
    evaluate,
    ood_auroc,
    per_attribute_accuracy,
)
from .model import FacialCNN, build_model, count_parameters
from .mvtec import (
    MVTEC_CATEGORIES,
    MVTEC_TEXTURES,
    MVTecCategory,
    get_mvtec_loaders,
)
from .pixel_metrics import (
    collect_anomaly_maps,
    evaluate_pixel_metrics,
    pixel_auroc,
    pro_score,
)
from .plots import (
    plot_architecture_effect,
    plot_batch_loss,
    plot_bias_variance_vs_block,
    plot_bias_variance_vs_percentile,
    plot_decoder_history,
    plot_decomposition_auroc_bars,
    plot_ensemble_decomposition,
    plot_ensemble_score_hists,
    plot_instance_summary,
    plot_mean_abs_bias,
    plot_member_instance,
    plot_per_block_breakdown,
    plot_recons_only,
    plot_top_ood_glasses,
    plot_variance_heatmaps,
    smooth_cam,
)
from .train import evaluate_one_epoch, train_decoders, train_model
from .utils import get_device, seed_everything, setup_logging

__all__ = [
    "ATTR_TARGETS",
    "BlockDecoder",
    "CELEBA_ATTRS",
    "CelebAFacialAttributes",
    "FacialCNN",
    "GENDER_ATTR",
    "MVTEC_CATEGORIES",
    "MVTEC_TEXTURES",
    "MVTecCategory",
    "OOD_ATTR",
    "OOD_ATTRS",
    "aggregate_anomaly_score",
    "apply_overrides",
    "build_decoders",
    "build_model",
    "classifier_n_params",
    "collect_anomaly_maps",
    "collect_decomposition_scores",
    "collect_eye_region_bias",
    "collect_predictions",
    "count_parameters",
    "decomposition_maps",
    "evaluate",
    "evaluate_ensemble_decomposition",
    "evaluate_one_epoch",
    "evaluate_pixel_metrics",
    "gather_samples",
    "get_celeba_loaders",
    "get_device",
    "get_mvtec_loaders",
    "pixel_auroc",
    "pro_score",
    "identity_residual",
    "load_config",
    "ood_auroc",
    "per_attribute_accuracy",
    "plot_architecture_effect",
    "plot_batch_loss",
    "plot_bias_variance_vs_block",
    "plot_bias_variance_vs_percentile",
    "plot_decoder_history",
    "plot_decomposition_auroc_bars",
    "plot_ensemble_decomposition",
    "plot_ensemble_score_hists",
    "plot_instance_summary",
    "plot_mean_abs_bias",
    "plot_member_instance",
    "plot_per_block_breakdown",
    "plot_recons_only",
    "plot_top_ood_glasses",
    "plot_variance_heatmaps",
    "decoder_n_params",
    "render_classifier_svg",
    "render_decoder_svg",
    "render_ensemble_svg",
    "resolve_member_configs",
    "sample_block_recons",
    "sample_decomposition",
    "seed_everything",
    "setup_logging",
    "smooth_cam",
    "to_jsonable",
    "train_decoders",
    "train_model",
]
