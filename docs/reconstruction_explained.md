# How LRAD Reconstruction Works

This document explains the pipeline stage-by-stage so you can interpret the
diagnostic plots produced by `scripts/inspect_model.py`.

## The core idea

Train a classifier on *only normal* data. At each hidden layer, attach a
**decoder** that tries to reconstruct the original image from that layer's
activations. When the decoders are later fed an **anomalous** image, they
reconstruct it poorly — because they've only ever seen normal activations.
The pixel-wise reconstruction error becomes the anomaly heatmap.

## Pipeline

```
                              ┌─────────────────────────────────┐
                              │      frozen classifier          │
  input x  ──▶  Block 0  ──▶  Block 1  ──▶  ...  ──▶  Block N  ──▶  logits
                  │              │                      │
                  │ a₀           │ a₁                   │ aₙ
                  ▼              ▼                      ▼
              Decoder 0     Decoder 1              Decoder N
                  │              │                      │
                  ▼              ▼                      ▼
                  x̂₀             x̂₁                    x̂ₙ
                  │              │                      │
           ┌──────┴──────────────┴──────────────────────┘
           │
           ▼
   per-pixel error:  eᵢ = (x̂ᵢ − x)²       (one heatmap per decoder)
           │
           ▼
   fusion (mean / max / weighted)  ──▶  final anomaly heatmap  ──▶  score
```

## Stage-by-stage

### 1. Classifier activations (aᵢ)

- The classifier is trained first, then **frozen**.
- At depth `i`, the activation `aᵢ` has shape `(B, Cᵢ, Hᵢ, Wᵢ)` for CNN,
  or `(B, Dᵢ)` for MLP.
- **Early layers** encode local textures / edges.
  **Deep layers** encode object-level, semantic features.
- *Plot:* `feature_maps_sample_*.png`, `activation_distributions.png`.

### 2. Decoders (x̂ᵢ = Decoderᵢ(aᵢ))

- Each decoder is an independent network, one per chosen depth.
- CNN decoder: mirror of the encoder using `ConvTranspose2d` — upsamples the
  activation map back to the original resolution `(H, W)` with `Sigmoid`
  at the end for pixel values in [0, 1].
- MLP decoder: Linear → ReLU → Linear → Sigmoid → reshape to `(C, H, W)`.
- **Trained only on normal data** to minimize MSE(x̂ᵢ, x).
- *Plot:* `per_layer_reconstructions.png`.

### 3. Per-layer error maps (eᵢ)

- For each decoder: `eᵢ = mean_channel((x̂ᵢ − x)²)` — a single-channel
  heatmap at image resolution.
- On normal data: small error (decoder was trained for this).
- On anomalous data: errors spike wherever the classifier's features
  can't be inverted back to the pixel.
- **Different layers catch different anomalies:** shallow-layer errors
  highlight texture/local defects; deep-layer errors highlight
  structural/semantic deviations.
- *Plot:* `per_layer_errors.png` — each tile's `s=` annotation is that
  layer's per-image score (the max pixel value in that error map).

### 4. Fusion

Multiple error maps are combined into one. Three options:

- **`mean`** — simple average. Robust baseline.
- **`max`** — pixel-wise max across layers. Catches any layer that fires.
  More sensitive, more false positives.
- **`weighted`** — inverse-quality weighting, favors decoders with lower
  average error (i.e. more "trusted" on normal data).

### 5. Image-level score

Final score = `max(fused_heatmap)`. A single pixel that's very wrong is
enough to flag the whole image as anomalous.

## Interpreting the inspection plots

| Plot | What to look for |
|---|---|
| `feature_maps_sample_*.png` | Channel activations per layer for one sample. Shallow rows should look like edges/textures; deep rows should look sparse and abstract. |
| `activation_stats.png` | Sparsity should increase with depth (deeper layers are more selective). Dead channels > 50% is a red flag. |
| `activation_distributions.png` | Histograms. With ReLU, expect a big spike at 0. Long positive tails are normal. If a deep layer has no spike, BatchNorm may be forcing non-zero means. |
| `per_layer_reconstructions.png` | Earlier decoders should reconstruct near-perfectly (more spatial info). Deeper decoders should look blurrier — they have to hallucinate detail. |
| `per_layer_errors.png` | On normal samples: uniformly low. On anomaly: one or more layers light up at the anomaly location. The `s=` scores tell you which layer contributes most. |

## When things go wrong

- **All layers perfectly reconstruct anomalies too** → decoders have
  over-generalized (maybe the normal set was too diverse, or decoders too
  high-capacity). Reduce decoder width, add more normal classes, or
  regularize.
- **All layers produce very high error on normal data** → decoders are
  under-trained. Increase decoder epochs.
- **Only one layer fires on anomalies** → fusion is probably drowning
  its signal. Try `fusion: weighted` or `fusion: max`.
- **Feature maps look identical across layers** → the classifier
  collapsed. Retrain with more epochs or lower LR.
