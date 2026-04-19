"""LRAD Nested Pipeline — wraps a frozen classifier with cascaded decoders.

Mirror of `LRADModel` but each decoder only inverts its adjacent classifier
block. Reconstructions from deep layers are produced by chaining decoders.

This is the alternative architecture proposed for comparison against the
parallel decoders in `lrad_model.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import CNNClassifier, MLPClassifier
from .nested_decoder import CNNNestedDecoder, MLPNestedDecoder


class LRADModelNested(nn.Module):
    """LRAD with nested (cascaded) decoders.

    Each decoder D_k inverts a single classifier block:
      D_0: a_0 → image
      D_k: a_k → a_{k-1}   for k >= 1

    At inference, reconstruction from level k is built by cascading:
      x̂_k = D_0(D_1(...D_k(a_k)))

    Args:
        classifier: A pre-trained CNNClassifier or MLPClassifier (will be frozen).
        decoder_layers: Which classifier layers to *report* reconstructions for.
                        Note: even if you only ask for layer k, decoders 0..k are
                        all built and trained, because the cascade requires them.
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

        if self.arch_type == "cnn":
            n_layers = len(classifier.blocks)
        else:
            n_layers = len(classifier.layers)

        if decoder_layers is None:
            decoder_layers = list(range(n_layers))

        # The cascade requires a contiguous chain from 0 up to max requested.
        max_layer = max(decoder_layers)
        self.decoder_layers = list(range(max_layer + 1))
        self.report_layers = sorted(set(decoder_layers))

        self.decoders = nn.ModuleList()
        for k in self.decoder_layers:
            self.decoders.append(self._build_nested_decoder(k))

    def _build_nested_decoder(self, k: int) -> nn.Module:
        if self.arch_type == "cnn":
            ch = self.classifier.channels
            sz = self.classifier.spatial_sizes
            return CNNNestedDecoder(
                in_channels=ch[k + 1], out_channels=ch[k],
                in_size=sz[k + 1], out_size=sz[k],
                is_image_target=(k == 0),
            )
        else:
            hd = self.classifier.hidden_dims
            in_dim = hd[k]
            if k == 0:
                shape = (1, self.classifier.input_size, self.classifier.input_size)
                return MLPNestedDecoder(in_dim, self.classifier.input_dim, output_shape=shape)
            return MLPNestedDecoder(in_dim, hd[k - 1], output_shape=None)

    def forward(self, x: torch.Tensor) -> dict:
        """Cascade forward.

        Returns:
            dict with:
              'logits': classifier output
              'reconstructions': list of x̂_k images for each report layer k
              'activations': list of raw a_k for each report layer
              'all_activations': full activation list (used by the trainer)
        """
        with torch.no_grad():
            logits, all_activations = self.classifier(x)

        reconstructions = []
        for k in self.report_layers:
            h = all_activations[k].detach()
            for j in range(k, -1, -1):
                h = self.decoders[j](h)
            reconstructions.append(h)

        return {
            "logits": logits,
            "reconstructions": reconstructions,
            "activations": [all_activations[k] for k in self.report_layers],
            "all_activations": all_activations,
        }

    def compute_anomaly_maps(self, x: torch.Tensor, fusion: str = "mean") -> dict:
        """Compute pixel-level anomaly heatmaps. Same fusion API as LRADModel."""
        output = self.forward(x)
        recons = output["reconstructions"]

        H, W = x.shape[2], x.shape[3]
        per_layer_maps = []
        for recon in recons:
            error = (recon - x).pow(2).mean(dim=1, keepdim=True)
            if error.shape[2] != H or error.shape[3] != W:
                error = F.interpolate(error, size=(H, W), mode="bilinear", align_corners=False)
            per_layer_maps.append(error)

        stacked = torch.stack(per_layer_maps, dim=0)
        if fusion == "mean":
            fused = stacked.mean(dim=0)
        elif fusion == "max":
            fused = stacked.max(dim=0).values
        elif fusion == "weighted":
            weights = torch.stack([1.0 / (m.mean() + 1e-8) for m in per_layer_maps])
            weights = weights / weights.sum()
            fused = sum(w * m for w, m in zip(weights, per_layer_maps))
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

        scores = fused.flatten(1).max(dim=1).values
        return {
            "per_layer": per_layer_maps,
            "fused": fused,
            "scores": scores,
            "logits": output["logits"],
        }
