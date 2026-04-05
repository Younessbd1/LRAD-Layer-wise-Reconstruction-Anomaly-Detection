"""LRAD-UQ: Uncertainty-Quantified Anomaly Detection.

Extends the base LRAD pipeline with three UQ strategies:
  1. MC Dropout — stochastic decoders, T forward passes
  2. Deep Ensembles — M independently-trained decoders per layer
  3. Combined — reconstruction error + uncertainty for richer anomaly maps

Key outputs:
  - Reconstruction error map (same as base LRAD)
  - Epistemic uncertainty map (new: from MC Dropout or ensemble disagreement)
  - Mutual information map (new: captures "I don't know" regions)
  - Combined anomaly map (error × uncertainty weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .classifier import CNNClassifier, MLPClassifier
from .uq_decoders import (
    MCDropoutCNNDecoder,
    MCDropoutMLPDecoder,
    EnsembleDecoder,
)
from .decoder import CNNDecoder, MLPDecoder


class LRADModelUQ(nn.Module):
    """LRAD with Uncertainty Quantification.

    Args:
        classifier: Pre-trained and frozen classifier.
        uq_method: One of 'mc_dropout', 'ensemble', 'both'.
        mc_samples: Number of MC forward passes (for mc_dropout).
        n_ensemble: Number of ensemble members (for ensemble).
        dropout_rate: Dropout rate for MC Dropout decoders.
        decoder_layers: Which classifier layers to decode.
    """

    def __init__(
        self,
        classifier: CNNClassifier | MLPClassifier,
        uq_method: str = "mc_dropout",
        mc_samples: int = 20,
        n_ensemble: int = 5,
        dropout_rate: float = 0.2,
        decoder_layers: list[int] | None = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.classifier.freeze()

        self.uq_method = uq_method
        self.mc_samples = mc_samples
        self.n_ensemble = n_ensemble
        self.arch_type = "cnn" if isinstance(classifier, CNNClassifier) else "mlp"

        if self.arch_type == "cnn":
            n_layers = len(classifier.blocks)
        else:
            n_layers = len(classifier.layers)

        if decoder_layers is None:
            decoder_layers = list(range(n_layers))
        self.decoder_layers = decoder_layers

        # Build UQ decoders
        self.decoders = nn.ModuleList()
        for layer_idx in decoder_layers:
            self.decoders.append(self._build_uq_decoder(layer_idx))

    def _build_uq_decoder(self, layer_idx: int) -> nn.Module:
        """Build a UQ decoder for the given layer."""
        if self.arch_type == "cnn":
            channels = self.classifier.channels[:layer_idx + 2]
            spatial = self.classifier.spatial_sizes[:layer_idx + 2]

            if self.uq_method in ("mc_dropout", "both"):
                return MCDropoutCNNDecoder(channels, spatial, dropout_rate=0.2)
            elif self.uq_method == "ensemble":
                return EnsembleDecoder(
                    CNNDecoder, n_members=self.n_ensemble,
                    channels=channels, spatial_sizes=spatial,
                )
        else:
            act_dim = self.classifier.hidden_dims[layer_idx]
            out_dim = self.classifier.input_dim
            mid = (act_dim + out_dim) // 2
            hidden = [mid] if act_dim < out_dim // 2 else []
            shape = (1, self.classifier.input_size, self.classifier.input_size)

            if self.uq_method in ("mc_dropout", "both"):
                return MCDropoutMLPDecoder(
                    act_dim, out_dim, hidden_dims=hidden,
                    output_shape=shape, dropout_rate=0.3,
                )
            elif self.uq_method == "ensemble":
                return EnsembleDecoder(
                    MLPDecoder, n_members=self.n_ensemble,
                    activation_dim=act_dim, output_dim=out_dim,
                    hidden_dims=hidden, output_shape=shape,
                )

        raise ValueError(f"Unknown uq_method: {self.uq_method}")

    def forward(self, x: torch.Tensor) -> dict:
        """Standard forward pass (mean reconstruction)."""
        with torch.no_grad():
            logits, all_acts = self.classifier(x)

        activations = [all_acts[i] for i in self.decoder_layers]
        reconstructions = []
        for dec, act in zip(self.decoders, activations):
            recon = dec(act.detach())
            reconstructions.append(recon)

        return {"logits": logits, "reconstructions": reconstructions, "activations": activations}

    def compute_uq_anomaly_maps(
        self,
        x: torch.Tensor,
        fusion: str = "mean",
    ) -> dict:
        """Compute anomaly maps WITH uncertainty quantification.

        Returns:
            dict with:
                'error_maps': list of (B, 1, H, W) reconstruction error per layer
                'uncertainty_maps': list of (B, 1, H, W) epistemic uncertainty per layer
                'fused_error': (B, 1, H, W) fused reconstruction error
                'fused_uncertainty': (B, 1, H, W) fused uncertainty
                'combined': (B, 1, H, W) error × (1 + uncertainty) — boosted anomaly map
                'scores_error': (B,) image-level error scores
                'scores_uncertainty': (B,) image-level uncertainty scores
                'scores_combined': (B,) image-level combined scores
                'logits': (B, num_classes)
        """
        with torch.no_grad():
            logits, all_acts = self.classifier(x)

        H, W = x.shape[2], x.shape[3]
        error_maps = []
        uncertainty_maps = []

        for i, dec in enumerate(self.decoders):
            act = all_acts[self.decoder_layers[i]].detach()

            # Get reconstruction statistics
            if self.uq_method in ("mc_dropout", "both"):
                result = dec.mc_forward(act, n_samples=self.mc_samples)
            elif self.uq_method == "ensemble":
                result = dec.ensemble_forward(act)
            else:
                raise ValueError(f"Unknown uq_method: {self.uq_method}")

            mean_recon = result["mean"]
            variance = result["variance"]  # (B, 1, h, w)

            # Reconstruction error
            error = (mean_recon - x).pow(2).mean(dim=1, keepdim=True)  # (B, 1, h, w)

            # Upsample if needed
            if error.shape[2] != H or error.shape[3] != W:
                error = F.interpolate(error, (H, W), mode="bilinear", align_corners=False)
                variance = F.interpolate(variance, (H, W), mode="bilinear", align_corners=False)

            error_maps.append(error)
            uncertainty_maps.append(variance)

        # Fuse across layers
        stacked_err = torch.stack(error_maps, dim=0)
        stacked_unc = torch.stack(uncertainty_maps, dim=0)

        if fusion == "mean":
            fused_error = stacked_err.mean(dim=0)
            fused_unc = stacked_unc.mean(dim=0)
        elif fusion == "max":
            fused_error = stacked_err.max(dim=0).values
            fused_unc = stacked_unc.max(dim=0).values
        else:
            fused_error = stacked_err.mean(dim=0)
            fused_unc = stacked_unc.mean(dim=0)

        # Combined score: error boosted by uncertainty
        # High error + high uncertainty → very suspicious
        # High error + low uncertainty → confident anomaly
        # Low error + high uncertainty → model unsure, warrants attention
        norm_unc = fused_unc / (fused_unc.max() + 1e-8)
        combined = fused_error * (1.0 + norm_unc)

        return {
            "error_maps": error_maps,
            "uncertainty_maps": uncertainty_maps,
            "fused_error": fused_error,
            "fused_uncertainty": fused_unc,
            "combined": combined,
            "scores_error": fused_error.flatten(1).max(1).values,
            "scores_uncertainty": fused_unc.flatten(1).max(1).values,
            "scores_combined": combined.flatten(1).max(1).values,
            "logits": logits,
        }
