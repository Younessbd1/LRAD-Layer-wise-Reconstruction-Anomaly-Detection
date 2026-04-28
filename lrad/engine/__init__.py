"""Training + evaluation engine for LRAD on Real-IAD."""

from .evaluator import (
    collect_anomaly_scores,
    evaluate_realiad,
    image_metrics,
    pixel_metrics,
)
from .realiad_trainer import (
    collect_features,
    train_classifier_pretext,
    train_decoders_with_val,
)

__all__ = [
    "collect_anomaly_scores",
    "collect_features",
    "evaluate_realiad",
    "image_metrics",
    "pixel_metrics",
    "train_classifier_pretext",
    "train_decoders_with_val",
]
