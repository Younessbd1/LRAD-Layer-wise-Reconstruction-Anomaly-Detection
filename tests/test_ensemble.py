"""Shape/contract tests for the ensemble bias/variance decomposition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lrad.decoder import build_decoders
from lrad.ensemble import (
    UNC_TERMS,
    block_reconstructions,
    collect_eye_region_bias,
    decomposition_maps,
    evaluate_ensemble_decomposition,
    evaluate_ensemble_uncertainty,
    identity_residual,
    mean_error_maps,
    min_error_maps,
    quantile_min_error_maps,
    sample_block_recons,
)
from lrad.evaluate import _bernoulli_entropy, _bernoulli_entropy_logits
from lrad.model import FacialCNN

IMG = 16
N_BLOCKS = 2
N_MODELS = 3
TERMS = ("risk", "bias", "variance")


def _make_model(seed: int) -> tuple[FacialCNN, torch.nn.ModuleList]:
    torch.manual_seed(seed)
    model = FacialCNN(channels=(4, 8), input_size=IMG).eval()
    decoders = build_decoders(model, image_size=IMG).eval()
    return model, decoders


@pytest.fixture(scope="module")
def setup():
    pairs = [_make_model(s) for s in range(N_MODELS)]
    models = [m for m, _ in pairs]
    decoders_list = [d for _, d in pairs]
    images = torch.rand(8, 3, IMG, IMG)
    recons = [
        block_reconstructions(models[i], decoders_list[i], images)
        for i in range(N_MODELS)
    ]
    maps = decomposition_maps(images, recons)
    return models, decoders_list, images, maps


def test_decomposition_map_shapes(setup):
    _, _, images, maps = setup
    assert set(maps.keys()) == set(range(N_BLOCKS))
    for k in range(N_BLOCKS):
        for t in TERMS:
            m = maps[k][t]
            assert m.shape == (images.shape[0], IMG, IMG)
            assert torch.isfinite(m).all()
        assert maps[k]["mean_recon"].shape == (images.shape[0], 3, IMG, IMG)


def test_risk_bias_variance_identity(setup):
    """Risk = Bias + Variance must hold pixelwise up to float32 noise."""
    _, _, _, maps = setup
    for k in range(N_BLOCKS):
        recombined = maps[k]["bias"] + maps[k]["variance"]
        assert torch.allclose(maps[k]["risk"], recombined, atol=1e-5)
    assert identity_residual(maps) < 1e-5


def test_terms_are_non_negative(setup):
    """Squared-error and spread terms are non-negative by construction."""
    _, _, _, maps = setup
    for k in range(N_BLOCKS):
        for t in TERMS:
            assert (maps[k][t] >= 0).all()


def test_single_model_has_zero_variance():
    """A 1-model 'ensemble' has no spread: Variance == 0 and Risk == Bias."""
    model, decoders = _make_model(0)
    images = torch.rand(5, 3, IMG, IMG)
    recons = [block_reconstructions(model, decoders, images)]
    maps = decomposition_maps(images, recons)
    for k in range(N_BLOCKS):
        assert torch.allclose(maps[k]["variance"],
                              torch.zeros_like(maps[k]["variance"]),
                              atol=1e-6)
        assert torch.allclose(maps[k]["risk"], maps[k]["bias"], atol=1e-6)


def test_identical_models_have_zero_variance():
    """If every member is the same model, the members never disagree."""
    model, decoders = _make_model(1)
    images = torch.rand(5, 3, IMG, IMG)
    recons = [
        block_reconstructions(model, decoders, images) for _ in range(4)
    ]
    maps = decomposition_maps(images, recons)
    for k in range(N_BLOCKS):
        assert maps[k]["variance"].max().item() < 1e-6


def test_decomposition_maps_rejects_empty():
    with pytest.raises(ValueError):
        decomposition_maps(torch.rand(2, 3, IMG, IMG), [])


def _recons(setup):
    models, decoders_list, images, _ = setup
    return images, [
        block_reconstructions(models[i], decoders_list[i], images)
        for i in range(N_MODELS)
    ]


def test_quantile_min_k1_equals_min(setup):
    """k=1 must reproduce min_error_maps exactly (bit-for-bit)."""
    images, recons = _recons(setup)
    qmin = quantile_min_error_maps(images, recons, k=1)
    mn = min_error_maps(images, recons)
    assert set(qmin.keys()) == set(mn.keys())
    for k in qmin:
        assert torch.equal(qmin[k], mn[k])


def test_quantile_min_is_monotone_in_k(setup):
    """The k-th smallest error is non-decreasing in k, and bounded by
    min (k=1) below and the per-pixel max (k=M) above."""
    images, recons = _recons(setup)
    prev = None
    for k in range(1, N_MODELS + 1):
        cur = quantile_min_error_maps(images, recons, k=k)
        if prev is not None:
            for b in cur:
                assert (cur[b] >= prev[b]).all()
        prev = cur


def test_quantile_min_shapes_and_finiteness(setup):
    images, recons = _recons(setup)
    qmin = quantile_min_error_maps(images, recons, k=2)
    mean = mean_error_maps(images, recons)
    assert set(qmin.keys()) == set(range(N_BLOCKS))
    for b in range(N_BLOCKS):
        assert qmin[b].shape == (images.shape[0], IMG, IMG)
        assert torch.isfinite(qmin[b]).all()
        assert (qmin[b] >= 0).all()
        # The k-th smallest of M values never exceeds M times their mean.
        assert (qmin[b] <= mean[b] * N_MODELS + 1e-6).all()


def test_quantile_min_rejects_bad_k(setup):
    images, recons = _recons(setup)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, recons, k=0)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, recons, k=N_MODELS + 1)
    with pytest.raises(ValueError):
        quantile_min_error_maps(images, [], k=1)


def test_sample_block_recons_keeps_members_separate(setup):
    """One reconstruction per member at a chosen block; bad blocks rejected."""
    models, decoders_list, images, _ = setup
    device = torch.device("cpu")
    imgs, recons = sample_block_recons(
        models, decoders_list, images, device, block=1,
    )
    assert imgs.shape == images.shape
    assert len(recons) == N_MODELS
    for r in recons:
        assert r.shape == (images.shape[0], 3, IMG, IMG)
        assert torch.isfinite(r).all()
    with pytest.raises(ValueError):
        sample_block_recons(models, decoders_list, images, device,
                            block=N_BLOCKS)


def test_evaluate_ensemble_decomposition(setup):
    """End-to-end: AUROC keys present and score arrays well-shaped."""
    models, decoders_list, images, _ = setup
    device = torch.device("cpu")
    in_imgs = torch.rand(10, 3, IMG, IMG)
    ood_imgs = torch.rand(7, 3, IMG, IMG)
    loaders = {
        "test_in": [(in_imgs[:6], None, None, None),
                    (in_imgs[6:], None, None, None)],
        "test_ood": [(ood_imgs, None, None, None)],
    }
    out = evaluate_ensemble_decomposition(
        models, decoders_list, loaders, device, agg="p95",
    )
    assert out["blocks"] == list(range(N_BLOCKS))
    for t in TERMS:
        assert len(out["per_block_auroc"][t]) == N_BLOCKS
        assert f"score_{t}_aggregated" in out["auroc"]
        for k in range(N_BLOCKS):
            assert f"score_{t}_per_block_{k}" in out["auroc"]
        assert out["scores_in"]["aggregated"][t].shape == (10,)
        assert out["scores_ood"]["aggregated"][t].shape == (7,)
        assert np.isfinite(out["scores_in"]["aggregated"][t]).all()
    # Per-member reconstruction-error scores + AUROC (architecture effect).
    assert len(out["member_auroc"]) == N_MODELS
    for m in range(N_MODELS):
        mau = out["member_auroc"][m]
        assert 0.0 <= mau["aggregated"] <= 1.0
        assert len(mau["per_block"]) == N_BLOCKS
        assert out["scores_in"]["per_member"]["aggregated"][m].shape == (10,)
        assert out["scores_ood"]["per_member"]["aggregated"][m].shape == (7,)
        for k in range(N_BLOCKS):
            assert (out["scores_in"]["per_member"]["per_block"][m][k].shape
                    == (10,))


def test_collect_eye_region_bias_shapes_and_score(setup):
    """Per-image eye-region bias stats: shapes, finiteness, score formula."""
    models, decoders_list, _, _ = setup
    device = torch.device("cpu")
    images = torch.rand(9, 3, IMG, IMG)
    loader = [(images[:5], None, None, None), (images[5:], None, None, None)]
    out = collect_eye_region_bias(
        models, decoders_list, loader, device, block=1,
    )
    for key in ("global_mean", "eye_mean", "score"):
        assert out[key].shape == (9,)
        assert np.isfinite(out[key]).all()
        assert (out[key] >= 0).all()
    # score = eye_mean * (eye_mean / global_mean)
    expected = out["eye_mean"] ** 2 / np.maximum(out["global_mean"], 1e-12)
    assert np.allclose(out["score"], expected, rtol=1e-5)


def test_collect_eye_region_bias_rejects_bad_region(setup):
    models, decoders_list, _, _ = setup
    device = torch.device("cpu")
    loader = [(torch.rand(2, 3, IMG, IMG), None, None, None)]
    with pytest.raises(ValueError):
        collect_eye_region_bias(
            models, decoders_list, loader, device, block=0,
            region=(0.6, 0.4, 0.0, 1.0),
        )


# ---------------------------------------------------------------------------
# Predictive-uncertainty (epistemic = variance) OOD score
# ---------------------------------------------------------------------------

def test_bernoulli_entropy_no_nan_on_saturated_logits():
    """Saturated attribute logits must not produce NaN entropy.

    Regression for the float32 bug: sigmoid(large logit) == 1.0, and the old
    clamp floor 1e-12 left it at 1.0, so (1-p).log() was -inf -> NaN. Both the
    logit-based path and the float32-safe probability path must stay finite.
    """
    logits = torch.tensor([[30.0, 40.0, -50.0, 0.0, 2.0]], dtype=torch.float32)
    h_logits = _bernoulli_entropy_logits(logits)
    assert torch.isfinite(h_logits).all()
    assert (h_logits >= 0).all()
    h_probs = _bernoulli_entropy(torch.sigmoid(logits))
    assert torch.isfinite(h_probs).all()


def _labelled_loader(images: torch.Tensor, n_attrs: int):
    """One-batch loader with the (img, gender, attrs, is_ood) tuple shape."""
    b = images.size(0)
    return [(
        images,
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, n_attrs),
        torch.zeros(b, dtype=torch.long),
    )]


def test_evaluate_ensemble_uncertainty(setup):
    """Total = Aleatoric + Epistemic identity, AUROC keys, finite scores."""
    models, _, _, _ = setup
    device = torch.device("cpu")
    n_attrs = models[0].n_attrs
    loaders = {
        "test_in": _labelled_loader(torch.rand(10, 3, IMG, IMG), n_attrs),
        "test_ood": _labelled_loader(torch.rand(7, 3, IMG, IMG), n_attrs),
    }
    out = evaluate_ensemble_uncertainty(models, loaders, device)
    for head in out["heads"]:
        for t in UNC_TERMS:
            assert t in out["auroc"][head]
        assert out["epistemic_auroc"][head] == out["epistemic_auroc"][head]


def test_epistemic_is_total_minus_aleatoric_and_nonneg():
    """Epistemic uncertainty (MI) is non-negative and obeys the identity."""
    from lrad.ensemble import _uncertainty_scores

    rng = np.random.default_rng(0)
    m, n, c, a = 5, 50, 2, 6
    g = rng.dirichlet([1.0, 1.0], size=(m, n))
    at = rng.uniform(0.0, 1.0, size=(m, n, a))
    s = _uncertainty_scores(g, at)
    for head in ("gender", "attrs", "combined"):
        total = s[head]["total"]
        recon = s[head]["aleatoric"] + s[head]["epistemic"]
        assert np.allclose(total, recon, atol=1e-10)
        assert (s[head]["epistemic"] >= -1e-9).all()
