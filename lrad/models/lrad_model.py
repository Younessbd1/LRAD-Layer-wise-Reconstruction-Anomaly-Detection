"""LRAD Pipeline - wraps a frozen classifier with N trainable decoders.

This is the main model class. It handles:
  - Building decoders matched to each classifier layer
  - Computing multi-scale reconstruction errors
  - Fusing per-layer heatmaps into a single anomaly map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import CNNClassifier, MLPClassifier
from .decoder import CNNDecoder, MLPDecoder


class LRADModel(nn.Module):
    """Layer-wise Reconstruction Anomaly Detection model.

    Args:
        classifier: A pre-trained and frozen CNNClassifier or MLPClassifier.
        decoder_layers: Which classifier layers to attach decoders to.
                        Default None = all layers.
    """

    def __init__(
        self,
        classifier: CNNClassifier | MLPClassifier,
        decoder_layers: list[int] | None = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.classifier.freeze()

        self.arch_type = "cnn" if isinstance(classifier, CNNClassifier) else "mlp"

        # Determine which layers to decode
        if self.arch_type == "cnn":
            n_layers = len(classifier.blocks)
        else:
            n_layers = len(classifier.layers)

        if decoder_layers is None:
            decoder_layers = list(range(n_layers))
        self.decoder_layers = decoder_layers

        # Build decoders
        self.decoders = nn.ModuleList()
        for layer_idx in decoder_layers:
            self.decoders.append(self._build_decoder(layer_idx))

    def _build_decoder(self, layer_idx: int) -> nn.Module:
        """Construct the appropriate decoder for a given layer index."""
        if self.arch_type == "cnn":
            # channels[:layer_idx+2] gives [in_ch, ch1, ..., ch_{layer_idx+1}]
            channels = self.classifier.channels[:layer_idx + 2]
            spatial = self.classifier.spatial_sizes[:layer_idx + 2]
            return CNNDecoder(channels, spatial)
        else:
            act_dim = self.classifier.hidden_dims[layer_idx]
            out_dim = self.classifier.input_dim
            # Add a hidden layer for deeper decoders
            mid = (act_dim + out_dim) // 2
            hidden = [mid] if act_dim < out_dim // 2 else []
            return MLPDecoder(
                activation_dim=act_dim,
                output_dim=out_dim,
                hidden_dims=hidden,
                output_shape=(1, self.classifier.input_size, self.classifier.input_size),
            )

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass.

        Returns:
            dict with keys:
                'logits': classifier output (B, num_classes)
                'reconstructions': list of (B, C, H, W) reconstructions
                'activations': list of raw activations
        """
        with torch.no_grad():
            logits, all_activations = self.classifier(x)

        # Select activations for decoded layers
        activations = [all_activations[i] for i in self.decoder_layers]
        reconstructions = []

        for decoder, act in zip(self.decoders, activations):
            recon = decoder(act.detach())
            reconstructions.append(recon)

        return {
            "logits": logits,
            "reconstructions": reconstructions,
            "activations": activations,
        }

    def compute_anomaly_maps(
        self,
        x: torch.Tensor,
        fusion: str = "mean",
    ) -> dict:
        """Compute pixel-level anomaly heatmaps from reconstruction errors.

        Args:
            x: Input images (B, C, H, W).
            fusion: How to combine multi-scale maps. One of:
                    'mean' - average across decoder levels
                    'max'  - take pixel-wise maximum
                    'weighted' - weight by inverse reconstruction quality

        Returns:
            dict with keys:
                'per_layer': list of (B, 1, H, W) error maps per decoder
                'fused': (B, 1, H, W) combined anomaly map
                'scores': (B,) image-level anomaly scores
                'logits': (B, num_classes) classifier predictions
        """
        output = self.forward(x)
        recons = output["reconstructions"]

        H, W = x.shape[2], x.shape[3]
        per_layer_maps = []

        for recon in recons:
            # Pixel-wise squared error
            error = (recon - x).pow(2).mean(dim=1, keepdim=True)  # (B, 1, h, w)
            # Upsample to original resolution if needed
            if error.shape[2] != H or error.shape[3] != W:
                error = F.interpolate(error, size=(H, W), mode="bilinear", align_corners=False)
            per_layer_maps.append(error)

        # Fuse maps
        stacked = torch.stack(per_layer_maps, dim=0)  # (N_decoders, B, 1, H, W)

        if fusion == "mean":
            fused = stacked.mean(dim=0)
        elif fusion == "max":
            fused = stacked.max(dim=0).values
        elif fusion == "weighted":
            # Weight by global reconstruction quality (inverse MSE)
            weights = []
            for m in per_layer_maps:
                w = 1.0 / (m.mean() + 1e-8)
                weights.append(w)
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            fused = sum(w * m for w, m in zip(weights, per_layer_maps))
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

        # Image-level scores: max pixel anomaly per image
        scores = fused.flatten(1).max(dim=1).values  # (B,)

        return {
            "per_layer": per_layer_maps,
            "fused": fused,
            "scores": scores,
            "logits": output["logits"],
        }
