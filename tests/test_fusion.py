"""Tests for the locfre feature-error signal and the rank-fusion detector."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.decoder import build_decoders
from lrad.feature_error import (
    collect_locfre_scores,
    feature_error_maps,
    fit_feature_error_stats,
    locfre_scores_from_maps,
)
from lrad.fusion import (
    apply_signal_fusion,
    collect_fusion_signals,
    evaluate_fused_ood,
    evaluate_supervised_fusion,
    fit_signal_fusion,
    rank_fusion,
)
from lrad.model import FacialCNN

IMG = 16
N_MODELS = 3
BLOCKS = (0, 1)   # the tiny 2-block test model: 8x8 and 8x8/4x4 features
DEVICE = torch.device("cpu")


def _make_model(seed: int) -> tuple[FacialCNN, torch.nn.ModuleList]:
    torch.manual_seed(seed)
    model = FacialCNN(channels=(4, 8), input_size=IMG).eval()
    decoders = build_decoders(model, image_size=IMG).eval()
    return model, decoders


@pytest.fixture(scope="module")
def ensemble():
    pairs = [_make_model(s) for s in range(N_MODELS)]
    return [m for m, _ in pairs], [d for _, d in pairs]


def _loader(images: torch.Tensor, batch: int = 4):
    return [(images[i:i + batch],) for i in range(0, len(images), batch)]


# ---------------------------------------------------------------------------
# feature_error_maps / stats
# ---------------------------------------------------------------------------

def test_feature_error_maps_shapes_and_range(ensemble):
    models, decoders_list = ensemble
    images = torch.rand(5, 3, IMG, IMG)
    maps = feature_error_maps(models, decoders_list, images, DEVICE, BLOCKS)
    assert set(maps.keys()) == set(BLOCKS)
    for j, mp in maps.items():
        assert mp.shape[0] == 5
        assert mp.dim() == 3
        assert torch.isfinite(mp).all()
        # squared distance between unit vectors lives in [0, 4]
        assert (mp >= 0).all() and (mp <= 4.0 + 1e-5).all()


def test_feature_error_maps_validates(ensemble):
    models, decoders_list = ensemble
    images = torch.rand(2, 3, IMG, IMG)
    with pytest.raises(ValueError):
        feature_error_maps([], [], images, DEVICE, BLOCKS)
    with pytest.raises(ValueError):
        feature_error_maps(models, decoders_list, images, DEVICE, (99,))


def test_fit_feature_error_stats_contract(ensemble):
    models, decoders_list = ensemble
    ref = fit_feature_error_stats(
        models, decoders_list, _loader(torch.rand(8, 3, IMG, IMG)),
        DEVICE, BLOCKS,
    )
    assert ref["n_images"] == 8
    for j in BLOCKS:
        st = ref["stats"][j]
        assert st["mean"].shape == st["std"].shape
        assert (st["std"] >= ref["std_floor"]).all()
    with pytest.raises(ValueError):
        fit_feature_error_stats(
            models, decoders_list, [], DEVICE, BLOCKS,
        )


def test_locfre_scoring_reference_images_is_zero(ensemble):
    """Same image repeated as reference AND query: map == μ, z == 0."""
    models, decoders_list = ensemble
    one = torch.rand(1, 3, IMG, IMG).expand(6, -1, -1, -1).contiguous()
    ref = fit_feature_error_stats(
        models, decoders_list, _loader(one), DEVICE, BLOCKS,
    )
    scores = collect_locfre_scores(
        models, decoders_list, _loader(one[:2]), DEVICE, ref, BLOCKS,
    )
    for j in BLOCKS:
        assert scores[j].shape == (2,)
        assert np.allclose(scores[j], 0.0, atol=1e-4)


def test_locfre_scores_from_maps_shapes(ensemble):
    models, decoders_list = ensemble
    images = torch.rand(4, 3, IMG, IMG)
    ref = fit_feature_error_stats(
        models, decoders_list, _loader(torch.rand(8, 3, IMG, IMG)),
        DEVICE, BLOCKS,
    )
    maps = feature_error_maps(models, decoders_list, images, DEVICE, BLOCKS)
    out = locfre_scores_from_maps(maps, ref)
    for j in BLOCKS:
        assert out[j].shape == (4,)
        assert torch.isfinite(out[j]).all()


# ---------------------------------------------------------------------------
# rank_fusion
# ---------------------------------------------------------------------------

def test_rank_fusion_is_monotone_invariant():
    """Fusing one signal reproduces its ranking (AUROC-equivalent)."""
    s = np.array([0.3, 0.1, 0.9, 0.5])
    fused = rank_fusion([s])
    assert (np.argsort(fused) == np.argsort(s)).all()
    # a monotone transform of the signal changes nothing
    assert np.allclose(rank_fusion([np.exp(10 * s)]), fused)


def test_rank_fusion_separable_signals_stay_separable():
    """Two noisy-but-agreeing signals: fused keeps in < ood perfectly."""
    rng = np.random.RandomState(0)
    s1 = np.concatenate([rng.rand(50), rng.rand(50) + 2.0])
    s2 = np.concatenate([rng.rand(50), rng.rand(50) + 2.0])
    fused = rank_fusion([s1, s2])
    assert fused[:50].max() < fused[50:].min()


def test_rank_fusion_weights_and_validation():
    s1 = np.array([0.0, 1.0, 2.0])
    s2 = np.array([2.0, 1.0, 0.0])
    # weight fully on s2 → s2's ordering wins
    fused = rank_fusion([s1, s2], weights=[0.0, 1.0])
    assert (np.argsort(fused) == np.argsort(s2)).all()
    with pytest.raises(ValueError):
        rank_fusion([])
    with pytest.raises(ValueError):
        rank_fusion([s1, s2[:2]])
    with pytest.raises(ValueError):
        rank_fusion([s1], weights=[1.0, 2.0])


# ---------------------------------------------------------------------------
# collect_fusion_signals / evaluate_fused_ood
# ---------------------------------------------------------------------------

def test_collect_fusion_signals_keys_and_shapes(ensemble):
    models, decoders_list = ensemble
    ref = fit_feature_error_stats(
        models, decoders_list, _loader(torch.rand(8, 3, IMG, IMG)),
        DEVICE, BLOCKS,
    )
    n = 6
    sig = collect_fusion_signals(
        models, decoders_list, _loader(torch.rand(n, 3, IMG, IMG)),
        DEVICE, ref, BLOCKS,
    )
    expected = {f"locfre_b{j}" for j in BLOCKS} | {
        "unc_epistemic_combined", "ens_energy_gender",
    }
    assert set(sig.keys()) == expected
    for k, v in sig.items():
        assert v.shape == (n,), k
        assert np.isfinite(v).all(), k


def test_evaluate_fused_ood_end_to_end(ensemble):
    models, decoders_list = ensemble
    out = evaluate_fused_ood(
        models, decoders_list,
        _loader(torch.rand(8, 3, IMG, IMG)),
        _loader(torch.rand(6, 3, IMG, IMG)),
        _loader(torch.rand(6, 3, IMG, IMG)),
        DEVICE, blocks=BLOCKS,
    )
    assert "fused" in out["auroc"]
    # the fused recipe uses exactly locfre blocks + epistemic + energy,
    # not the extra head scores
    assert out["signals"] == [f"locfre_b{j}" for j in BLOCKS] + [
        "unc_epistemic_combined", "ens_energy_gender",
    ]
    for k in out["signals"]:
        assert k in out["auroc"]
        a = out["auroc"][k].get("auroc")
        assert 0.0 <= a <= 1.0
    assert out["fused_in"].shape == (6,)
    assert out["fused_ood"].shape == (6,)
    assert len(out["weights"]) == len(out["signals"])


# ---------------------------------------------------------------------------
# supervised fusion
# ---------------------------------------------------------------------------

def test_fit_apply_signal_fusion_separable():
    """LR fusion learns weights that separate held-out data perfectly when
    the calibration signals do."""
    rng = np.random.RandomState(0)
    mk = lambda lo: {"a": rng.rand(40) + lo, "b": rng.rand(40) - lo}
    calib = fit_signal_fusion(mk(0.0), mk(2.0))
    assert set(calib) == {"signals", "mu", "sd", "coef", "intercept"}
    import json
    json.dumps(calib)  # JSON-serializable contract
    te_in, te_ood = mk(0.0), mk(2.0)
    s_in = apply_signal_fusion(calib, te_in)
    s_ood = apply_signal_fusion(calib, te_ood)
    assert s_in.max() < s_ood.min()


def test_fit_signal_fusion_validates():
    with pytest.raises(ValueError):
        fit_signal_fusion({"a": np.zeros(3)}, {"b": np.ones(3)},
                          signals=["a", "b"])
    with pytest.raises(ValueError):
        fit_signal_fusion({"a": np.zeros(0)}, {"a": np.ones(3)})


def test_evaluate_supervised_fusion_end_to_end(ensemble):
    models, decoders_list = ensemble
    out = evaluate_supervised_fusion(
        models, decoders_list,
        _loader(torch.rand(8, 3, IMG, IMG)),   # ref stats
        _loader(torch.rand(6, 3, IMG, IMG)),   # cal in
        _loader(torch.rand(6, 3, IMG, IMG)),   # cal ood
        _loader(torch.rand(6, 3, IMG, IMG)),   # test in
        _loader(torch.rand(6, 3, IMG, IMG)),   # test ood
        DEVICE, blocks=BLOCKS,
    )
    assert "fused_rank" in out["auroc"]
    assert "fused_supervised" in out["auroc"]
    for k in ("fused_rank", "fused_supervised"):
        a = out["auroc"][k].get("auroc")
        assert 0.0 <= a <= 1.0
    assert out["calibration"]["signals"] == sorted(out["signals_in"])
    assert len(out["calibration"]["coef"]) == len(out["signals_in"])
