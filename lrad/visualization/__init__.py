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

apply_style()
