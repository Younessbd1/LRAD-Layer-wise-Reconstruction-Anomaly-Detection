# Ensemble mean error map — `mean_error_maps.png`

How `mean_error_maps.png` (produced by `scripts/run_ensemble.py`) is computed,
formula plus a worked example.

---

## 1. What we compute

For each conv block `k`, take **every model's** reconstruction-error map and
average it **pixel by pixel over the M models**. The error is the **squared**
error, not the absolute value.

```text
MeanErr_k(x)[i] = (1/M) Σₘ ( x[i] − f̂ᵐ_k(x)[i] )²
```

where:

- `x` = input image,
- `f̂ᵐ_k(x)` = model `m`'s reconstruction of the image at block `k`,
- `i` = a pixel; the squared error is **summed** over the 3 RGB channels
  (`Σ_c (x_c − f̂_c)²`, no mean, no square root), so a pixel lives in `[0, 3]`,
- `M` = number of models in the ensemble (here 10).

The result is one `(H, W)` map per block, drawn as usual:
`Original | Err L0 | Err L1 | … | Err Ln`, with a shared colour scale and each
tile's mean annotated under it.

![Per-block mean_m (x − f̂ᵐ)² for a few ID and OOD samples — the Risk
view.](figures/mean_error_maps.png)

---

## 2. This is the Risk term — and Risk = Bias + Variance

This is the key point. `MeanErr` (squared) **is exactly** the **Risk** term of
the bias/variance decomposition. Don't confuse it with the Bias:

| Quantity | Formula | What it is |
|---|---|---|
| **MeanErr / Risk** (this map) | `(1/M) Σₘ ( x − f̂ᵐ )²` | mean **of the errors** of each model |
| **Bias** (`mean_abs_bias.png`) | `( x − f̄ )²` with `f̄ = (1/M) Σₘ f̂ᵐ` | error of the **averaged** reconstruction |

The order of operations is reversed:

- **MeanErr**: square each model's error *first*, then average. → square, then mean.
- **Bias**: average the reconstructions *first*, then take the error.

With the **squared** error, the decomposition is exact, pixel by pixel:

```text
(1/M) Σₘ ( x − f̂ᵐ )²  =  ( x − f̄ )²  +  (1/M) Σₘ ( f̂ᵐ − f̄ )²
        Risk           =     Bias     +            Variance
```

This is exactly why we moved from `|·|` (L1) to `(·)²` (L2): with the absolute
value the identity is **false** (you only get the Jensen inequality `Bias ≤
MeanErr`), so the variance doesn't read off as a clean gap between the two maps.
In L2, the gap `MeanErr − Bias` **is** exactly the variance — the inter-model
disagreement we wanted to make visible.

---

## 3. Worked example

Take **1 pixel on a single channel**, **M = 3 models**, one block (a real image
would sum 3 such terms, one per RGB channel). True pixel value:

```text
x = 0.80
```

The 3 models' reconstructions for this pixel:

```text
f̂¹ = 0.50      f̂² = 0.90      f̂³ = 0.70
```

### MeanErr / Risk (this map)

Square each model's error, then average:

```text
( 0.80 − 0.50 )² = 0.09
( 0.80 − 0.90 )² = 0.01
( 0.80 − 0.70 )² = 0.01

MeanErr = (0.09 + 0.01 + 0.01) / 3 = 0.11 / 3 = 0.03667
```

### Bias

Average the reconstructions first:

```text
f̄ = (0.50 + 0.90 + 0.70) / 3 = 2.10 / 3 = 0.70

Bias = ( 0.80 − 0.70 )² = 0.01
```

### Variance

Spread of the models around `f̄`:

```text
( 0.50 − 0.70 )² = 0.04
( 0.90 − 0.70 )² = 0.04
( 0.70 − 0.70 )² = 0.00

Variance = (0.04 + 0.04 + 0.00) / 3 = 0.08 / 3 = 0.02667
```

### Takeaway

```text
Bias + Variance = 0.01 + 0.02667 = 0.03667 = MeanErr (Risk)   ✓
```

The identity lands **exactly**. The gap `MeanErr − Bias = 0.02667` isn't noise —
it's the variance, the fact that the 3 models disagree (`0.50`, `0.90`, `0.70`).
That gap was hidden as long as we used the absolute value.

---

## 4. The real computation (RGB + tensors)

Real images have 3 RGB channels and a batch. The code (`mean_error_maps` in
[`lrad/ensemble.py`](../lrad/ensemble.py)) does:

```python
recons = torch.stack(                       # (M, B, 3, H, W)
    [recons_per_model[m][k] for m in range(n_models)], dim=0,
)
se = ((images.unsqueeze(0) - recons) ** 2).sum(dim=2)    # (M, B, H, W)
out[k] = se.mean(dim=0)                                  # (B, H, W)
```

Step by step:

1. **`images.unsqueeze(0) - recons`** → signed error `x − f̂ᵐ`, shape `(M, B, 3, H, W)`.
2. **`** 2`** → square, per pixel and per channel. (Square **before** the channel
   sum, so no sign cancellation.)
3. **`.sum(dim=2)`** → **sum** over the **3 RGB channels** → `(M, B, H, W)`. (No
   mean, no square root: a pixel lives in `[0, 3]` — a single model's error map,
   same as the other plots.)
4. **`.mean(dim=0)`** → mean over the **M models** → `(B, H, W)`.

That's the section-1 formula applied to the whole batch at once.

The display is `plot_mean_error_maps` in [`lrad/plots.py`](../lrad/plots.py): one
row per image, one column per block, `viridis` colormap, **fixed scale `vmin=0,
vmax=3`** (3 channels, so the theoretical max is 3 — never a max from the
displayed subset, or two `.png` files wouldn't be comparable), and each tile's
mean written under it in white.

---

## 5. Reading the image

- **Rows**: the samples (ID on top, OOD below — see `row_labels`).
- **Columns**: `Original`, then one error map per block `L0…Ln`.
- **Colour**: brighter (yellow in viridis) = higher mean reconstruction error at
  that pixel.
- **Number under each tile**: mean error over the whole tile.

Expected: on **ID** images the maps stay dark everywhere; on **OOD** images they
light up where the anomaly sits (e.g. the glasses), because no ensemble member
reconstructs that region well.
