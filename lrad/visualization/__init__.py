"""Publication-quality plots for the LRAD pipeline."""

from ._style import apply_style, PALETTE, HEAT_CMAP, IMG_CMAP
from .heatmaps import (
    plot_heatmap_grid,
    plot_score_distributions,
    plot_roc_curves,
    plot_reconstruction_comparison,
)
from .activations import (
    plot_feature_maps,
    plot_activation_stats,
    plot_per_layer_reconstructions,
    plot_per_layer_errors,
    plot_activation_distributions,
    compute_activation_stats,
    format_stats_report,
)
from .realiad_plots import (
    plot_class_distribution,
    plot_feature_scatter,
    plot_pr_curves,
    plot_reconstruction_gallery,
    plot_training_curves,
)

apply_style()

__all__ = [
    "apply_style", "PALETTE", "HEAT_CMAP", "IMG_CMAP",
    # heatmaps
    "plot_heatmap_grid", "plot_score_distributions",
    "plot_roc_curves", "plot_reconstruction_comparison",
    # activations
    "plot_feature_maps", "plot_activation_stats",
    "plot_per_layer_reconstructions", "plot_per_layer_errors",
    "plot_activation_distributions",
    "compute_activation_stats", "format_stats_report",
    # realiad
    "plot_class_distribution", "plot_feature_scatter",
    "plot_pr_curves", "plot_reconstruction_gallery",
    "plot_training_curves",
]
