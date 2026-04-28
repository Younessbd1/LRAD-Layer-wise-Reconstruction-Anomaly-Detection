"""LRADModel — frozen DeepCNN classifier wrapped with one decoder per stage.

At inference time:
    1. Run the (frozen) classifier and capture per-stage activations.
    2. For each stage k, the matching decoder produces an image-space
       reconstruction \\hat{x}_k = D_k(a_k).
    3. Per-pixel squared error e_k = (\\hat{x}_k - x)^2 (averaged over
       channels) is the per-stage anomaly map.
    4. Maps are fused (mean / max / weighted) into a single anomaly heatmap;
       the image-level score is the maximum pixel of the fused map.

The classifier is *only* used as a feature extractor — its logits are never
read at inference. This is what lets us train it on any pretext task that
encourages useful features (rotation, category, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_cnn import DeepCNNClassifier
from .decoder import CNNDecoder, build_decoders


class LRADModel(nn.Module):
    """Layer-wise Reconstruction Anomaly Detection on a deep CNN backbone.

    Args:
        classifier: a (trained or untrained) DeepCNNClassifier; will be
                    frozen on construction.
        decoder_layers: indices of stages to attach decoders to.
                        Default: every stage.
        decoder_base_ch: width of the deepest decoder layer (see CNNDecoder).
    """

    def __init__(
        self,
        classifier: DeepCNNClassifier,
        decoder_layers: list[int] | None = None,
        decoder_base_ch: int = 128,
    ):
        super().__init__()
        self.classifier = classifier
        self.classifier.freeze()

        n_stages = len(classifier.stages)
        if decoder_layers is None:
            decoder_layers = list(range(n_stages))
        if not decoder_layers:
            raise ValueError("decoder_layers must be non-empty")
        if any(i < 0 or i >= n_stages for i in decoder_layers):
            raise ValueError(f"decoder_layers must be in [0, {n_stages})")
        self.decoder_layers = sorted(set(decoder_layers))

        # Build decoders only for the requested stages.
        all_decoders = build_decoders(
            stage_channels=classifier.stage_channels,
            spatial_sizes=classifier.spatial_sizes,
            in_channels=classifier.in_channels,
            base_ch=decoder_base_ch,
        )
        self.decoders = nn.ModuleList([all_decoders[i] for i in self.decoder_layers])

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """Run the classifier (no_grad) and every attached decoder."""
        with torch.no_grad():
            logits, all_activations = self.classifier(x)

        activations = [all_activations[i] for i in self.decoder_layers]
        reconstructions = [d(a.detach()) for d, a in zip(self.decoders, activations)]
        return {
            "logits": logits,
            "reconstructions": reconstructions,
            "activations": activations,
        }

    # ------------------------------------------------------------------
    def compute_anomaly_maps(
        self,
        x: torch.Tensor,
        fusion: str = "mean",
    ) -> dict:
        """Pixel-level anomaly heatmaps + image-level scores.

        Args:
            x: input batch (B, C, H, W) in [0, 1].
            fusion: 'mean' | 'max' | 'weighted'.

        Returns dict with keys:
            'per_layer'  : list of (B, 1, H, W) per-decoder error maps
            'fused'      : (B, 1, H, W) combined map
            'scores'     : (B,) image-level scores (max pixel of fused map)
            'logits'     : (B, num_classes) classifier output (for record only)
        """
        out = self.forward(x)
        recons = out["reconstructions"]

        H, W = x.shape[-2], x.shape[-1]
        per_layer: list[torch.Tensor] = []
        for r in recons:
            err = (r - x).pow(2).mean(dim=1, keepdim=True)  # (B, 1, h, w)
            if err.shape[-2:] != (H, W):
                err = F.interpolate(err, size=(H, W), mode="bilinear",
                                    align_corners=False)
            per_layer.append(err)

        stacked = torch.stack(per_layer, dim=0)  # (N, B, 1, H, W)

        if fusion == "mean":
            fused = stacked.mean(dim=0)
        elif fusion == "max":
            fused = stacked.amax(dim=0)
        elif fusion == "weighted":
            # Weight each decoder by inverse mean error (good decoders get
            # more weight). Weights are computed per-batch so that a single
            # rogue decoder can't dominate the fused map.
            weights = torch.stack([1.0 / (m.mean() + 1e-8) for m in per_layer])
            weights = weights / weights.sum()
            fused = sum(w * m for w, m in zip(weights, per_layer))
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        scores = fused.flatten(1).amax(dim=1)  # (B,)

        return {
            "per_layer": per_layer,
            "fused": fused,
            "scores": scores,
            "logits": out["logits"],
        }
