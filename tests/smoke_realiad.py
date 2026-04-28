"""Fast smoke checks for the Real-IAD pipeline plumbing (no pytest needed).

Run from the repo root with::

    python tests/smoke_realiad.py

It exercises the interfaces without touching the Real-IAD zips, so it is
safe to invoke on any machine before submitting a SLURM job.
"""

from __future__ import annotations

import sys
import traceback

import torch


def _check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  [OK]  {name}")
        return True
    except Exception:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return False


def test_deep_cnn_shapes():
    from lrad.models import DeepCNNClassifier
    m = DeepCNNClassifier(in_channels=3, num_classes=4, input_size=224)
    assert m.spatial_sizes == [224, 56, 28, 14, 7], m.spatial_sizes
    x = torch.zeros(2, 3, 224, 224)
    logits, acts = m(x)
    assert logits.shape == (2, 4)
    assert [a.shape[-1] for a in acts] == [56, 28, 14, 7]


def test_lrad_with_deep_cnn():
    from lrad.models import DeepCNNClassifier, LRADModel
    m = DeepCNNClassifier(in_channels=3, num_classes=4, input_size=224)
    lrad = LRADModel(m)
    assert len(lrad.decoders) == 4
    x = torch.zeros(1, 3, 224, 224)
    out = lrad.compute_anomaly_maps(x)
    assert out["fused"].shape == (1, 1, 224, 224)
    assert out["scores"].shape == (1,)


def test_rotation_pretext():
    from lrad.engine.realiad_trainer import _rotate_batch
    x = torch.randn(3, 3, 32, 32)
    xs, y = _rotate_batch(x)
    assert xs.shape == (12, 3, 32, 32)
    assert torch.equal(y, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))


def test_image_metrics():
    import numpy as np
    from lrad.engine import image_metrics
    # Simulate a perfectly separable scoring.
    normal = np.random.default_rng(0).normal(0, 0.1, 200)
    anomaly = np.random.default_rng(1).normal(1, 0.1, 200)
    m = image_metrics(normal, anomaly)
    assert m["auroc"] > 0.99
    assert m["pr_auc"] > 0.99
    assert m["f1"] > 0.95


def test_override_parsing():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "run_realiad",
        Path(__file__).resolve().parent.parent / "scripts" / "run_realiad.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {"a": {"b": 1}, "c": 2.5}
    out = mod.apply_overrides(cfg, ["a.b=42", "c=3.14"])
    assert out["a"]["b"] == 42
    assert out["c"] == 3.14


def main() -> int:
    checks = [
        ("deep cnn shapes",   test_deep_cnn_shapes),
        ("lrad with deep cnn", test_lrad_with_deep_cnn),
        ("rotation pretext",  test_rotation_pretext),
        ("image metrics",     test_image_metrics),
        ("override parsing",  test_override_parsing),
    ]
    ok = 0
    for name, fn in checks:
        ok += _check(name, fn)
    print()
    print(f"{ok}/{len(checks)} smoke checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
