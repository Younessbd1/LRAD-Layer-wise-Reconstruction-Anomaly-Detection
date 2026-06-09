"""Shape/architecture tests for the per-block decoders.

Pins the bilinear-Upsample + Conv replacement of the former
ConvTranspose2d stages: every decoder of the full 6-block CelebA config
must still map its block activation to a (3, 64, 64) reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lrad.decoder import BlockDecoder, build_decoders
from lrad.model import FacialCNN

# Full configs/celeba_ood.yaml trunk: 6 conv blocks on 64x64 input.
CHANNELS = (32, 64, 128, 256, 256, 256)
IMG = 64


def _model() -> FacialCNN:
    torch.manual_seed(0)
    return FacialCNN(channels=CHANNELS, input_size=IMG).eval()


def test_all_six_decoders_output_image_shape():
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


def test_decoders_use_bilinear_upsample_not_transposed_conv():
    """Upsampling is bilinear Upsample + Conv2d — no ConvTranspose2d left."""
    decoders = build_decoders(_model(), image_size=IMG)
    has_upsample = False
    for dec in decoders:
        for m in dec.modules():
            assert not isinstance(m, nn.ConvTranspose2d)
            if isinstance(m, nn.Upsample):
                assert m.mode == "bilinear"
                assert m.align_corners is False
                has_upsample = True
    assert has_upsample


def test_channel_progression_preserved():
    """Each upsampling stage still halves channels down to min_channels."""
    dec = BlockDecoder(in_channels=256, in_size=2, out_size=64,
                       min_channels=16)
    convs = [m for m in dec.net
             if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
    expected, ch = [], 256
    for _ in range(5):  # log2(64 / 2)
        new_ch = max(16, ch // 2)
        expected.append((ch, new_ch))
        ch = new_ch
    assert [(c.in_channels, c.out_channels) for c in convs] == expected
