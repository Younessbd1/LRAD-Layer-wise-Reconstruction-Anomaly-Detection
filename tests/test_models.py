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


# ──────────────────────────────────────────────
#  Calibration (PaDiM-style baseline subtraction)
# ──────────────────────────────────────────────

class TestCalibration:
    def setup_method(self):
        torch.manual_seed(0)
        self.classifier = CNNClassifier([3, 16, 32], num_classes=4, input_size=32)
        self.classifier.freeze()
        self.model = LRADModel(self.classifier)

    def _normal_loader(self, n=8, batch_size=4):
        x = torch.randn(n, 3, 32, 32).clamp(0, 1)

        class _DS(torch.utils.data.Dataset):
            def __len__(self_): return n
            def __getitem__(self_, i):
                return x[i], torch.tensor(0), torch.zeros(4)

        return torch.utils.data.DataLoader(_DS(), batch_size=batch_size)

    def test_calibrate_sets_buffers(self):
        loader = self._normal_loader()
        assert self.model.calibrated.item() == 0.0
        self.model.calibrate(loader, device=torch.device("cpu"))
        assert self.model.calibrated.item() == 1.0
        for i in range(len(self.model.decoders)):
            buf = getattr(self.model, f"baseline_{i}")
            assert buf.dim() == 4 and buf.shape[0] == 1 and buf.shape[1] == 1
            assert (buf >= 0).all()

    def test_calibrate_reduces_baseline_error(self):
        """After calibration, the per-layer error of an in-distribution sample
        — once baseline-subtracted and clamped — should drop close to zero on
        average, since the baseline IS the average error on normals."""
        loader = self._normal_loader(n=16, batch_size=4)
        self.model.calibrate(loader, device=torch.device("cpu"))

        x = torch.randn(4, 3, 32, 32).clamp(0, 1)
        before = self.model.compute_anomaly_maps(x)
        # Toggle the buffer off to compare against the un-calibrated path.
        self.model.calibrated.fill_(0.0)
        after = self.model.compute_anomaly_maps(x)
        self.model.calibrated.fill_(1.0)
        # ``before`` ran with the baseline → mean fused intensity should be
        # strictly lower than the un-calibrated path.
        assert before["fused"].mean() < after["fused"].mean()

    def test_calibrate_buffer_reload(self):
        """Mirror the runner's save/load path: persist each baseline tensor,
        then re-register on a fresh model and verify behavior matches."""
        loader = self._normal_loader()
        self.model.calibrate(loader, device=torch.device("cpu"))

        saved = {f"baseline_{i}": getattr(self.model, f"baseline_{i}").clone()
                 for i in range(len(self.model.decoders))}
        saved["calibrated"] = self.model.calibrated.clone()

        fresh = LRADModel(CNNClassifier([3, 16, 32], num_classes=4, input_size=32))
        for name, tensor in saved.items():
            fresh.register_buffer(name, tensor)
        assert fresh.calibrated.item() == 1.0
        for i in range(len(fresh.decoders)):
            assert torch.allclose(getattr(fresh, f"baseline_{i}"),
                                  getattr(self.model, f"baseline_{i}"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
