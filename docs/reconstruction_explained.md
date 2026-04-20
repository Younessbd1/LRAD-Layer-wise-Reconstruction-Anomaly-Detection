# How LRAD Reconstruction Works

Notes on the pipeline and how to read the plots from `scripts/inspect_model.py`.

## Idea

Train a classifier on only normal data. Attach one decoder per hidden layer that tries to reconstruct the input from that layer's activations. Feed an anomalous image: decoders reconstruct it badly because they never saw activations like that. Pixel-wise squared error = heatmap.

## Pipeline

```
  x -> Block0 -> Block1 -> ... -> BlockN -> logits  (frozen classifier)
         |         |                |
         a0        a1               aN
         v         v                v
       Dec0      Dec1             DecN
         |         |                |
         v         v                v
         x0        x1               xN   (reconstructions)

  error:  e_i = mean_channel((x_i - x)^2)     one map per decoder
  fuse:   mean / max / weighted  ->  final heatmap
  score:  max(final heatmap)
```

## Stages

### 1. Activations

Classifier is trained, then frozen. At depth `i`, `a_i` has shape `(B, C_i, H_i, W_i)` for CNN or `(B, D_i)` for MLP. Shallow layers = edges/textures. Deep layers = object-level stuff.

Plots: `feature_maps_sample_*.png`, `activation_distributions.png`.

### 2. Decoders

One independent network per depth.

- CNN decoder: mirror of the encoder with `ConvTranspose2d`, ends in `Sigmoid`, output `(H, W)` in [0, 1].
- MLP decoder: Linear -> ReLU -> Linear -> Sigmoid -> reshape `(C, H, W)`.

Trained on normal data only, MSE loss.

Plot: `per_layer_reconstructions.png`.

### 3. Per-layer error maps

`e_i = mean_channel((x_i - x)^2)`, single channel, image-resolution.

- Normal input: small error (decoder has seen this kind of activation).
- Anomaly: errors spike where features can't be inverted back to the pixel.

Shallow layers catch texture/local defects. Deep layers catch structural stuff. That's the whole reason for stacking them.

Plot: `per_layer_errors.png`. The `s=` on each tile is that layer's per-image score (max of its error map).

### 4. Fusion

Three options:

- `mean`: average. Default.
- `max`: pixel-wise max. More sensitive, more false positives.
- `weighted`: inverse-quality weighting, trusts decoders with lower avg normal error.

### 5. Image score

`max(fused_heatmap)`. One bad pixel is enough to flag the image.

## Reading the plots

| Plot | What to check |
|---|---|
| `feature_maps_sample_*.png` | Shallow rows: edges/textures. Deep rows: sparse, abstract. |
| `activation_stats.png` | Sparsity should go up with depth. Dead channels > 50% is bad. |
| `activation_distributions.png` | With ReLU expect a spike at 0 and a positive tail. No spike in deep layers usually means BatchNorm is pushing the mean off zero. |
| `per_layer_reconstructions.png` | Early decoders: near-perfect. Deep decoders: blurry (they have to guess detail). |
| `per_layer_errors.png` | Normal: flat. Anomaly: one or more layers light up at the defect. `s=` tells you which. |

## Things that go wrong

- Decoders reconstruct anomalies too well. Over-capacity or normal set too diverse. Shrink decoder width, narrow the normal class set, or regularize.
- High error even on normal data. Decoders under-trained. More epochs.
- Only one layer fires on anomalies. Fusion is drowning it. Try `weighted` or `max`.
- Feature maps look identical across layers. Classifier collapsed. Retrain, more epochs, lower LR.
