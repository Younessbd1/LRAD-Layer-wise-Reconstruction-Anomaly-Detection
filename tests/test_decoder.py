"""Shape/architecture tests for the per-block decoders.

The decoders use a single upsampling architecture: every ×2 stage is a
learnable ConvTranspose2d(4×4, stride 2, padding 1) + BN + ReLU. Every
decoder of the full 5-block CelebA config must map its block activation to
a (3, 64, 64) reconstruction, whatever trunk variant (channel widths, conv
kernel size) the classifier uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lrad.decoder import BlockDecoder, build_decoders
from lrad.model import FacialCNN

# Full configs/celeba_ood.yaml trunk: 5 conv blocks on 64x64 input.
CHANNELS = (32, 64, 128, 256, 256)
IMG = 64


def _model(channels=CHANNELS, kernel_size: int = 3) -> FacialCNN:
    torch.manual_seed(0)
    return FacialCNN(channels=channels, input_size=IMG,
                     kernel_size=kernel_size).eval()


def test_all_five_decoders_output_image_shape():
    """Every BlockDecoder maps its activation back to (3, 64, 64)."""
    model = _model()
    decoders = build_decoders(model, image_size=IMG).eval()
    assert len(decoders) == len(CHANNELS)
    x = torch.rand(2, 3, IMG, IMG)
    with torch.no_grad():
        _, acts = model.forward_features(x)
        for k, dec in enumerate(decoders):
            out = dec(acts[k])
            assert out.shape == (2, 3, IMG, IMG), f"decoder {k}"
            assert torch.isfinite(out).all()
            # final Sigmoid keeps reconstructions in [0, 1]
            assert out.min() >= 0.0 and out.max() <= 1.0


def test_decoders_use_convtranspose_not_upsample():
    """The only upsampling stage is ConvTranspose2d — never nn.Upsample."""
    decoders = build_decoders(_model(), image_size=IMG)
    has_transpose = False
    for dec in decoders:
        for m in dec.modules():
            assert not isinstance(m, nn.Upsample)
            if isinstance(m, nn.ConvTranspose2d):
                has_transpose = True
                assert m.kernel_size == (4, 4)
                assert m.stride == (2, 2)
    assert has_transpose


def test_decoders_fit_every_member_variant():
    """Ensemble member variants (widths + kernel size) keep the decoder
    geometry valid: same block count, same spatial sizes, image-shaped
    reconstructions."""
    variants = [
        ((24, 48, 96, 192, 384), 3),
        ((16, 48, 96, 192, 320), 5),
        ((32, 48, 112, 224, 336), 5),
    ]
    x = torch.rand(2, 3, IMG, IMG)
    for channels, ks in variants:
        model = _model(channels=channels, kernel_size=ks)
        # kernel size only widens the receptive field; the spatial layout
        # (and hence the decoder stage count) is unchanged.
        assert model.block_spatial_sizes == [32, 16, 8, 4, 4]
        decoders = build_decoders(model, image_size=IMG).eval()
        with torch.no_grad():
            _, acts = model.forward_features(x)
            for k, dec in enumerate(decoders):
                out = dec(acts[k])
                assert out.shape == (2, 3, IMG, IMG), (channels, ks, k)
                assert torch.isfinite(out).all()


def test_channel_progression_preserved():
    """Each upsampling stage halves channels down to min_channels."""
    dec = BlockDecoder(in_channels=256, in_size=4, out_size=64,
                       min_channels=16)
    ups = [m for m in dec.net if isinstance(m, nn.ConvTranspose2d)]
    expected, ch = [], 256
    for _ in range(4):  # log2(64 / 4)
        new_ch = max(16, ch // 2)
        expected.append((ch, new_ch))
        ch = new_ch
    assert [(u.in_channels, u.out_channels) for u in ups] == expected
