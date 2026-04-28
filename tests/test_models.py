"""Unit tests for the deep CNN LRAD pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from lrad.models import (
    DeepCNNClassifier,
    LRADModel,
    build_decoders,
    build_deep_cnn,
)
from lrad.models.decoder import CNNDecoder


# ---------------------------------------------------------------------------
# DeepCNNClassifier
# ---------------------------------------------------------------------------

class TestDeepCNNClassifier:
    def test_default_resnet18_shapes(self):
        m = build_deep_cnn(preset="resnet18", in_channels=3,
                           num_classes=4, input_size=224, stem="large")
        assert m.spatial_sizes == [224, 56, 28, 14, 7]
        x = torch.zeros(2, 3, 224, 224)
        logits, acts = m(x)
        assert logits.shape == (2, 4)
        assert [a.shape[1] for a in acts] == [64, 128, 256, 512]
        assert [a.shape[-1] for a in acts] == [56, 28, 14, 7]

    def test_freeze(self):
        m = build_deep_cnn(preset="resnet10", input_size=64)
        m.freeze()
        for p in m.parameters():
            assert not p.requires_grad

    def test_small_stem(self):
        m = DeepCNNClassifier(in_channels=1, num_classes=4, input_size=64,
                              stage_channels=(32, 64, 128, 256),
                              blocks_per_stage=(1, 1, 1, 1), stem="small")
        # /2 from stem, then /2 at each subsequent stage transition
        assert m.spatial_sizes == [64, 32, 16, 8, 4]

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            build_deep_cnn(preset="totally-fake")


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class TestCNNDecoder:
    def test_reconstruction_shape(self):
        d = CNNDecoder(in_ch=128, out_ch=3, in_size=14, out_size=224)
        x = torch.randn(2, 128, 14, 14)
        out = d(x)
        assert out.shape == (2, 3, 224, 224)

    def test_output_range(self):
        d = CNNDecoder(in_ch=64, out_ch=3, in_size=56, out_size=224)
        out = d(torch.randn(1, 64, 56, 56))
        assert 0.0 <= out.min().item() <= out.max().item() <= 1.0

    def test_build_decoders_aligns_with_classifier(self):
        m = build_deep_cnn(preset="resnet10", input_size=224)
        decs = build_decoders(m.stage_channels, m.spatial_sizes,
                              in_channels=m.in_channels)
        assert len(decs) == 4
        x = torch.zeros(1, 3, 224, 224)
        _, acts = m(x)
        for d, a in zip(decs, acts):
            assert d(a).shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# LRADModel
# ---------------------------------------------------------------------------

class TestLRADModel:
    def test_full_pipeline(self):
        m = build_deep_cnn(preset="resnet10", input_size=64)
        lrad = LRADModel(m)
        assert len(lrad.decoders) == 4
        x = torch.zeros(1, 3, 64, 64)
        out = lrad.compute_anomaly_maps(x, fusion="mean")
        assert out["fused"].shape == (1, 1, 64, 64)
        assert out["scores"].shape == (1,)
        assert len(out["per_layer"]) == 4

    def test_subset_of_decoders(self):
        m = build_deep_cnn(preset="resnet10", input_size=64)
        lrad = LRADModel(m, decoder_layers=[0, 3])
        assert len(lrad.decoders) == 2
        out = lrad.compute_anomaly_maps(torch.zeros(2, 3, 64, 64))
        assert len(out["per_layer"]) == 2

    def test_classifier_stays_frozen(self):
        m = build_deep_cnn(preset="resnet10", input_size=64)
        lrad = LRADModel(m)
        lrad(torch.zeros(1, 3, 64, 64))
        for p in lrad.classifier.parameters():
            assert not p.requires_grad

    def test_fusion_methods(self):
        m = build_deep_cnn(preset="resnet10", input_size=64)
        lrad = LRADModel(m)
        x = torch.zeros(1, 3, 64, 64)
        for method in ["mean", "max", "weighted"]:
            assert lrad.compute_anomaly_maps(x, fusion=method)["fused"].shape \
                   == (1, 1, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
