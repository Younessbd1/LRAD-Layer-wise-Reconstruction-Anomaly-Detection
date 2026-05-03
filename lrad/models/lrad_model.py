"""LRADModel — frozen DeepCNN classifier wrapped with one decoder per stage.

At inference time:
    1. Run the (frozen) classifier and capture per-stage activations.
    2. For each stage k, the matching decoder produces an image-space
       reconstruction \\hat{x}_k = D_k(a_k).
    3. Per-pixel squared error e_k = (\\hat{x}_k - x)^2 (averaged over
       channels) is the per-stage anomaly map.
    4. Maps are fused (mean / max / weighted) into a single anomaly heatmap;
       the image-level score is the maximum pixel of the fused map.

The classifier is *only* used as a feature extractor — its logits are never
read at inference. This is what lets us train it on any pretext task that
encourages useful features (rotation, category, etc.).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_cnn import DeepCNNClassifier
from .decoder import CNNDecoder, build_decoders


def _gaussian_kernel2d(sigma: float, device, dtype) -> torch.Tensor:
    """Build a normalized 2D Gaussian kernel for separable smoothing."""
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma).pow(2))
    g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k.view(1, 1, *k.shape)


def _smooth(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth a (B, 1, H, W) map. ``sigma`` in pixels."""
    if sigma <= 0:
        return maps
    kernel = _gaussian_kernel2d(sigma, maps.device, maps.dtype)
    radius = kernel.shape[-1] // 2
    return F.conv2d(maps, kernel, padding=radius)


def _per_image_minmax(maps: torch.Tensor) -> torch.Tensor:
    """Min-max scale each (1, H, W) map to [0, 1]. Robust to constant maps.

    ``maps`` is (B, 1, H, W). Reduces over the spatial dims and reshapes the
    statistics to (B, 1, 1, 1) so the subtraction broadcasts back to the
    original 4D shape (a 3D ``(B, 1, 1)`` would right-align with the leading
    batch dim and produce a (B, B, H, W) outer-product shape).
    """
    B = maps.shape[0]
    flat = maps.reshape(B, -1)
    lo = flat.min(dim=1).values.view(B, 1, 1, 1)
    hi = flat.max(dim=1).values.view(B, 1, 1, 1)
    span = (hi - lo).clamp(min=1e-8)
    return (maps - lo) / span


class LRADModel(nn.Module):
    """Layer-wise Reconstruction Anomaly Detection on a deep CNN backbone.

    Args:
        classifier: a (trained or untrained) DeepCNNClassifier; will be
                    frozen on construction.
        decoder_layers: indices of stages to attach decoders to.
                        Default: every stage.
        decoder_base_ch: width of the deepest decoder layer (see CNNDecoder).
    """

    def __init__(
        self,
        classifier: DeepCNNClassifier,
        decoder_layers: list[int] | None = None,
        decoder_base_ch: int = 128,
    ):
        super().__init__()
        self.classifier = classifier
        self.classifier.freeze()

        n_stages = len(classifier.stages)
        if decoder_layers is None:
            decoder_layers = list(range(n_stages))
        if not decoder_layers:
            raise ValueError("decoder_layers must be non-empty")
        if any(i < 0 or i >= n_stages for i in decoder_layers):
            raise ValueError(f"decoder_layers must be in [0, {n_stages})")
        self.decoder_layers = sorted(set(decoder_layers))

        # Build decoders only for the requested stages.
        all_decoders = build_decoders(
            stage_channels=classifier.stage_channels,
            spatial_sizes=classifier.spatial_sizes,
            in_channels=classifier.in_channels,
            base_ch=decoder_base_ch,
        )
        self.decoders = nn.ModuleList([all_decoders[i] for i in self.decoder_layers])

        # Per-pixel "background" baseline for each decoder, populated by
        # ``calibrate``. Stored as buffers so they save with state_dict and
        # move with ``.to(device)``. Shape per buffer: (1, 1, h_layer, w_layer)
        # in the decoder's *native* output resolution. ``calibrated`` flips to
        # 1.0 when fitted; ``compute_anomaly_maps`` only subtracts when set.
        for i in range(len(self.decoders)):
            self.register_buffer(f"baseline_{i}", torch.zeros(0))
        self.register_buffer("calibrated", torch.zeros(1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, loader, device: torch.device | None = None) -> None:
        """Fit a per-pixel background error baseline on normal data.

        For each decoder ``i``, accumulate the squared reconstruction error
        averaged over channels at the decoder's *native* spatial resolution,
        then store the per-pixel mean across the loader as ``baseline_{i}``.

        At test time ``compute_anomaly_maps`` subtracts this baseline so that
        regions which are *consistently hard to reconstruct on normals*
        (e.g. face/hair edges, mouth, lighting hotspots on CelebA) stop
        dominating the heatmap, leaving only the genuinely anomalous regions
        bright. This is the same idea PaDiM uses to centre per-position
        feature distances; here we just centre per-pixel reconstruction
        error.

        ``loader`` should yield only normal samples — typically the same
        validation loader used for early stopping.
        """
        self.eval()
        if device is not None:
            self.to(device)
        else:
            device = next(self.parameters()).device

        sums: list[torch.Tensor] | None = None
        count = 0
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            output = self.forward(x)
            recons = output["reconstructions"]
            if sums is None:
                sums = [
                    torch.zeros(1, 1, r.shape[2], r.shape[3],
                                device=device, dtype=r.dtype)
                    for r in recons
                ]
            for i, recon in enumerate(recons):
                err = (recon - x).pow(2).mean(dim=1, keepdim=True)  # (B,1,h,w)
                sums[i] += err.sum(dim=0, keepdim=True)
            count += x.size(0)

        if sums is None or count == 0:
            raise ValueError("calibrate(): loader yielded no batches")

        for i, s in enumerate(sums):
            mean = (s / count).to(device)
            # Replace the (initially zero-sized) buffer with the fitted one.
            self.register_buffer(f"baseline_{i}", mean)
        self.calibrated.fill_(1.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """Run the classifier (no_grad) and every attached decoder."""
        with torch.no_grad():
            logits, all_activations = self.classifier(x)

        activations = [all_activations[i] for i in self.decoder_layers]
        reconstructions = [d(a.detach()) for d, a in zip(self.decoders, activations)]
        return {
            "logits": logits,
            "reconstructions": reconstructions,
            "activations": activations,
        }

    # ------------------------------------------------------------------
    def compute_anomaly_maps(
        self,
        x: torch.Tensor,
        fusion: str = "mean",
        smooth_sigma: float = 1.5,
        normalize_per_layer: bool = True,
        score_top_k_pct: float = 1.0,
    ) -> dict:
        """Pixel-level anomaly heatmaps + image-level scores.

        Args:
            x: Input images (B, C, H, W).
            fusion: How to combine multi-scale maps. One of:
                    'mean' — average across decoder levels
                    'max'  — take pixel-wise maximum
                    'weighted' — weight by inverse reconstruction quality
            smooth_sigma: σ (pixels) of Gaussian smoothing applied to each
                          per-layer map BEFORE fusion. Anomalies (sunglasses)
                          are spatially extended; smoothing suppresses
                          single-pixel reconstruction noise that otherwise
                          dominates ``max``-style scoring.
            normalize_per_layer: If True, per-image min-max each layer's
                                 map to [0, 1] before fusion so deep,
                                 upsampled blurry maps and shallow
                                 high-resolution maps contribute on equal
                                 footing.
            score_top_k_pct: Image-level score is the mean of the top
                             ``score_top_k_pct`` percent of fused pixels.
                             A spatially extended anomaly elevates many
                             pixels at once, so a top-k mean separates
                             classes far better than a single-pixel max.

        Returns:
            dict with keys:
                'per_layer': list of (B, 1, H, W) error maps per decoder
                             (smoothed; pre-normalization for visualization)
                'fused': (B, 1, H, W) combined anomaly map
                'scores': (B,) image-level anomaly scores
                'logits': (B, num_classes) classifier predictions
        """
        out = self.forward(x)
        recons = out["reconstructions"]

        H, W = x.shape[2], x.shape[3]
        raw_maps: list[torch.Tensor] = []

        use_baseline = bool(self.calibrated.item())
        for i, recon in enumerate(recons):
            error = (recon - x).pow(2).mean(dim=1, keepdim=True)  # (B, 1, h, w)
            # Per-pixel background subtraction at the decoder's native
            # resolution — see ``calibrate``. We clamp at 0 so negative
            # residuals (test image easier-than-average to reconstruct at
            # this pixel) don't pull down the score.
            if use_baseline:
                baseline = getattr(self, f"baseline_{i}")
                if baseline.numel() > 0:
                    error = (error - baseline).clamp(min=0)
            if error.shape[2] != H or error.shape[3] != W:
                error = F.interpolate(error, size=(H, W),
                                      mode="bilinear", align_corners=False)
            error = _smooth(error, sigma=smooth_sigma)
            raw_maps.append(error)

        # Per-layer min-max so all decoder scales contribute equally.
        if normalize_per_layer:
            fuse_inputs = [_per_image_minmax(m) for m in raw_maps]
        else:
            fuse_inputs = raw_maps

        stacked = torch.stack(fuse_inputs, dim=0)  # (N_decoders, B, 1, H, W)

        if fusion == "mean":
            fused = stacked.mean(dim=0)
        elif fusion == "max":
            fused = stacked.amax(dim=0)
        elif fusion == "weighted":
            weights = []
            for m in fuse_inputs:
                w = 1.0 / (m.mean() + 1e-8)
                weights.append(w)
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            fused = sum(w * m for w, m in zip(weights, fuse_inputs))
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        # Image-level score: mean of the top-k% of pixels. With H*W=4096 and
        # k=1.0%, that's ~41 pixels — covers the typical sunglasses region
        # without diluting the signal with the cheek/forehead background.
        flat = fused.flatten(1)
        n_pixels = flat.shape[1]
        k = max(1, int(math.ceil(n_pixels * score_top_k_pct / 100.0)))
        top_vals, _ = flat.topk(k, dim=1)
        scores = top_vals.mean(dim=1)  # (B,)

        return {
            "per_layer": raw_maps,        # un-normalized — visualization
            "fused": fused,               # post-fusion (for overlays)
            "scores": scores,
            "logits": out["logits"],
        }
