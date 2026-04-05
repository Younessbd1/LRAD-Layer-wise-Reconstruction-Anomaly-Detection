"""Unit tests for LRAD-UQ models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from lrad.models.classifier import CNNClassifier, MLPClassifier
from lrad.models.uq_decoders import MCDropoutCNNDecoder, MCDropoutMLPDecoder, EnsembleDecoder
from lrad.models.decoder import CNNDecoder, MLPDecoder
from lrad.models.lrad_uq import LRADModelUQ


class TestMCDropoutCNNDecoder:
    def test_shape(self):
        dec = MCDropoutCNNDecoder([1, 8, 16], [28, 14, 7], dropout_rate=0.2)
        x = torch.randn(2, 16, 7, 7)
        out = dec(x)
        assert out.shape == (2, 1, 28, 28)

    def test_mc_forward(self):
        dec = MCDropoutCNNDecoder([1, 8, 16], [28, 14, 7])
        x = torch.randn(2, 16, 7, 7)
        result = dec.mc_forward(x, n_samples=5)
        assert result["mean"].shape == (2, 1, 28, 28)
        assert result["variance"].shape == (2, 1, 28, 28)
        assert result["samples"].shape == (5, 2, 1, 28, 28)

    def test_stochasticity(self):
        # Need >= 2 intermediate layers so dropout is actually inserted
        dec = MCDropoutCNNDecoder([1, 8, 16], [28, 14, 7], dropout_rate=0.5)
        x = torch.randn(1, 16, 7, 7)
        result = dec.mc_forward(x, n_samples=10)
        # Samples should differ (dropout is stochastic)
        assert result["variance"].max() > 0


class TestMCDropoutMLPDecoder:
    def test_shape(self):
        dec = MCDropoutMLPDecoder(128, 784, output_shape=(1, 28, 28))
        x = torch.randn(3, 128)
        out = dec(x)
        assert out.shape == (3, 1, 28, 28)

    def test_mc_forward(self):
        dec = MCDropoutMLPDecoder(64, 784, hidden_dims=[256], output_shape=(1, 28, 28))
        x = torch.randn(2, 64)
        result = dec.mc_forward(x, n_samples=8)
        assert result["mean"].shape == (2, 1, 28, 28)
        assert result["variance"].shape == (2, 1, 28, 28)


class TestEnsembleDecoder:
    def test_cnn_ensemble(self):
        ens = EnsembleDecoder(
            CNNDecoder, n_members=3,
            channels=[1, 8, 16], spatial_sizes=[28, 14, 7],
        )
        x = torch.randn(2, 16, 7, 7)
        out = ens(x)
        assert out.shape == (2, 1, 28, 28)

    def test_ensemble_forward(self):
        ens = EnsembleDecoder(
            CNNDecoder, n_members=3,
            channels=[1, 8], spatial_sizes=[28, 14],
        )
        x = torch.randn(2, 8, 14, 14)
        result = ens.ensemble_forward(x)
        assert result["mean"].shape == (2, 1, 28, 28)
        assert result["variance"].shape == (2, 1, 28, 28)
        assert len(result["member_outputs"]) == 3

    def test_mlp_ensemble(self):
        ens = EnsembleDecoder(
            MLPDecoder, n_members=4,
            activation_dim=128, output_dim=784, output_shape=(1, 28, 28),
        )
        x = torch.randn(2, 128)
        result = ens.ensemble_forward(x)
        assert result["mean"].shape == (2, 1, 28, 28)

    def test_get_optimizers(self):
        ens = EnsembleDecoder(
            CNNDecoder, n_members=3,
            channels=[1, 8], spatial_sizes=[28, 14],
        )
        opts = ens.get_optimizers(lr=0.01)
        assert len(opts) == 3


class TestLRADModelUQ:
    def test_mc_dropout_cnn(self):
        clf = CNNClassifier([1, 8, 16], num_classes=2, input_size=28)
        clf.freeze()
        model = LRADModelUQ(clf, uq_method="mc_dropout", mc_samples=5)
        x = torch.randn(2, 1, 28, 28).clamp(0, 1)
        result = model.compute_uq_anomaly_maps(x)
        assert result["fused_error"].shape == (2, 1, 28, 28)
        assert result["fused_uncertainty"].shape == (2, 1, 28, 28)
        assert result["combined"].shape == (2, 1, 28, 28)
        assert result["scores_error"].shape == (2,)
        assert result["scores_combined"].shape == (2,)

    def test_ensemble_cnn(self):
        clf = CNNClassifier([1, 8, 16], num_classes=2, input_size=28)
        clf.freeze()
        model = LRADModelUQ(clf, uq_method="ensemble", n_ensemble=3)
        x = torch.randn(2, 1, 28, 28).clamp(0, 1)
        result = model.compute_uq_anomaly_maps(x)
        assert result["fused_error"].shape == (2, 1, 28, 28)
        assert result["scores_combined"].shape == (2,)

    def test_mc_dropout_mlp(self):
        clf = MLPClassifier(784, [256, 128], num_classes=3, input_size=28)
        clf.freeze()
        model = LRADModelUQ(clf, uq_method="mc_dropout", mc_samples=5)
        x = torch.randn(2, 1, 28, 28).clamp(0, 1)
        result = model.compute_uq_anomaly_maps(x)
        assert result["fused_error"].shape == (2, 1, 28, 28)

    def test_classifier_stays_frozen(self):
        clf = CNNClassifier([1, 8, 16], num_classes=2, input_size=28)
        clf.freeze()
        model = LRADModelUQ(clf, uq_method="mc_dropout")
        x = torch.randn(1, 1, 28, 28).clamp(0, 1)
        model.compute_uq_anomaly_maps(x)
        for p in model.classifier.parameters():
            assert not p.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
