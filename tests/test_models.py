"""Unit tests for LRAD models — verifies shapes, freezing, and reconstruction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.decoder import CNNDecoder, MLPDecoder
from lrad.models.lrad_model import LRADModel


# ──────────────────────────────────────────────
#  CNN Classifier
# ──────────────────────────────────────────────

class TestCNNClassifier:
    def setup_method(self):
        self.channels = [1, 8, 16, 32]
        self.model = CNNClassifier(self.channels, num_classes=4, input_size=28)

    def test_output_shapes(self):
        x = torch.randn(2, 1, 28, 28)
        logits, acts = self.model(x)
        assert logits.shape == (2, 4)
        assert len(acts) == len(self.channels) - 1

    def test_spatial_sizes_computed(self):
        assert len(self.model.spatial_sizes) == len(self.channels)
        assert self.model.spatial_sizes[0] == 28

    def test_freeze(self):
        self.model.freeze()
        for p in self.model.parameters():
            assert not p.requires_grad

    def test_activation_spatial_consistency(self):
        x = torch.randn(1, 1, 28, 28)
        _, acts = self.model(x)
        for i, act in enumerate(acts):
            assert act.shape[2] == self.model.spatial_sizes[i + 1]
            assert act.shape[1] == self.channels[i + 1]


# ──────────────────────────────────────────────
#  MLP Classifier
# ──────────────────────────────────────────────

class TestMLPClassifier:
    def setup_method(self):
        self.hidden = [512, 256, 128]
        self.model = MLPClassifier(784, self.hidden, num_classes=4, input_size=28)

    def test_output_shapes(self):
        x = torch.randn(2, 1, 28, 28)
        logits, acts = self.model(x)
        assert logits.shape == (2, 4)
        assert len(acts) == len(self.hidden)

    def test_activation_dims(self):
        x = torch.randn(3, 1, 28, 28)
        _, acts = self.model(x)
        for i, act in enumerate(acts):
            assert act.shape == (3, self.hidden[i])

    def test_freeze(self):
        self.model.freeze()
        for p in self.model.parameters():
            assert not p.requires_grad


# ──────────────────────────────────────────────
#  CNN Decoder
# ──────────────────────────────────────────────

class TestCNNDecoder:
    def test_reconstruction_shape(self):
        channels = [1, 8, 16]
        spatial = [28, 14, 7]
        decoder = CNNDecoder(channels, spatial)
        # Input: activation from layer 2 → (B, 16, 7, 7)
        x = torch.randn(2, 16, 7, 7)
        out = decoder(x)
        assert out.shape == (2, 1, 28, 28)

    def test_output_range(self):
        channels = [1, 8]
        spatial = [28, 14]
        decoder = CNNDecoder(channels, spatial)
        x = torch.randn(4, 8, 14, 14)
        out = decoder(x)
        # Sigmoid output → [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_layer_decoder(self):
        channels = [1, 16]
        spatial = [28, 14]
        decoder = CNNDecoder(channels, spatial)
        x = torch.randn(1, 16, 14, 14)
        out = decoder(x)
        assert out.shape == (1, 1, 28, 28)


# ──────────────────────────────────────────────
#  MLP Decoder
# ──────────────────────────────────────────────

class TestMLPDecoder:
    def test_reconstruction_shape(self):
        decoder = MLPDecoder(128, 784, output_shape=(1, 28, 28))
        x = torch.randn(2, 128)
        out = decoder(x)
        assert out.shape == (2, 1, 28, 28)

    def test_with_hidden_layers(self):
        decoder = MLPDecoder(64, 784, hidden_dims=[256], output_shape=(1, 28, 28))
        x = torch.randn(3, 64)
        out = decoder(x)
        assert out.shape == (3, 1, 28, 28)

    def test_output_range(self):
        decoder = MLPDecoder(128, 784, output_shape=(1, 28, 28))
        x = torch.randn(4, 128)
        out = decoder(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ──────────────────────────────────────────────
#  Full LRAD Model
# ──────────────────────────────────────────────

class TestLRADModelCNN:
    def setup_method(self):
        channels = [1, 8, 16, 32]
        self.classifier = CNNClassifier(channels, num_classes=4, input_size=28)
        self.classifier.freeze()
        self.model = LRADModel(self.classifier)

    def test_forward_keys(self):
        x = torch.randn(2, 1, 28, 28)
        out = self.model(x)
        assert "logits" in out
        assert "reconstructions" in out
        assert "activations" in out

    def test_reconstruction_shapes(self):
        x = torch.randn(2, 1, 28, 28)
        out = self.model(x)
        for recon in out["reconstructions"]:
            assert recon.shape == (2, 1, 28, 28)

    def test_anomaly_maps(self):
        x = torch.randn(4, 1, 28, 28).clamp(0, 1)
        result = self.model.compute_anomaly_maps(x, fusion="mean")
        assert result["fused"].shape == (4, 1, 28, 28)
        assert result["scores"].shape == (4,)
        assert len(result["per_layer"]) == len(self.model.decoders)

    def test_fusion_methods(self):
        x = torch.randn(2, 1, 28, 28).clamp(0, 1)
        for method in ["mean", "max", "weighted"]:
            result = self.model.compute_anomaly_maps(x, fusion=method)
            assert result["fused"].shape == (2, 1, 28, 28)

    def test_classifier_stays_frozen(self):
        x = torch.randn(2, 1, 28, 28)
        self.model(x)
        for p in self.model.classifier.parameters():
            assert not p.requires_grad


class TestLRADModelMLP:
    def setup_method(self):
        self.classifier = MLPClassifier(784, [256, 128], num_classes=4, input_size=28)
        self.classifier.freeze()
        self.model = LRADModel(self.classifier)

    def test_forward(self):
        x = torch.randn(2, 1, 28, 28)
        out = self.model(x)
        assert out["logits"].shape == (2, 4)
        assert len(out["reconstructions"]) == 2

    def test_anomaly_maps(self):
        x = torch.randn(3, 1, 28, 28).clamp(0, 1)
        result = self.model.compute_anomaly_maps(x)
        assert result["fused"].shape == (3, 1, 28, 28)
        assert result["scores"].shape == (3,)


# ──────────────────────────────────────────────
#  Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_single_decoder_layer(self):
        classifier = CNNClassifier([1, 16, 32], num_classes=2, input_size=28)
        classifier.freeze()
        model = LRADModel(classifier, decoder_layers=[0])
        assert len(model.decoders) == 1

        x = torch.randn(1, 1, 28, 28).clamp(0, 1)
        result = model.compute_anomaly_maps(x)
        assert result["fused"].shape == (1, 1, 28, 28)

    def test_rgb_input(self):
        classifier = CNNClassifier([3, 16, 32], num_classes=5, input_size=32)
        classifier.freeze()
        model = LRADModel(classifier)

        x = torch.randn(2, 3, 32, 32).clamp(0, 1)
        result = model.compute_anomaly_maps(x)
        assert result["fused"].shape == (2, 1, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
