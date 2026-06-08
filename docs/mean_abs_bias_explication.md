# How the numbers under each tile of `mean_abs_bias.png` are computed

Each tile in `mean_abs_bias.png` has a small white number under it. Here is
exactly what that number is, with a full worked example.

> **In one sentence:** the number under a tile is the **spatial mean of the
> squared-error map** `(x − f̄_k)²` (**summed** over the RGB channels), where
> `f̄_k` is the *ensemble-mean* reconstruction at conv block `k`.

That's exactly the **L2 (squared) bias** term of the `Risk = Bias + Variance`
decomposition. (The filename `mean_abs_bias.png` is historical: since the switch
to the L2 norm the map is the **squared** error, not the absolute value.) Units:
pixels normalized to `[0, 1]`.

![Per-block (x − f̄)² for a few ID and OOD samples. ID rows stay dark; OOD rows
brighten on the glasses.](figures/mean_abs_bias.png)

---

## 1. What each tile shows

For each image (a row `ID i` or `OOD i`) we draw:

```text
Original | (x − f̄)² L0 | (x − f̄)² L1 | ... | (x − f̄)² Ln
```

- **Column 0**: the original image `x`.
- **Following columns**: a heatmap of the error `(x − f̄_k)²` for each conv block `k`.
- **The number under each map**: the mean of that error map over all pixels.

Every error tile shares the **fixed colour scale `vmin=0, vmax=3`** (3 RGB
channels summed, so the theoretical max is 3 — never a max recomputed on the
displayed subset, so any two `.png` files stay comparable) and a single colour
bar. In practice the **ID** rows stay dark everywhere and the **OOD** rows light
up over the glasses.

---

## 2. The compute chain (with code references)

### Step A — per-model reconstruction

Each of the `M = 10` models reconstructs the image at block `k`:

```text
f_{m,k}(x)        # shape (B, 3, H, W),  m = 0 .. M-1
```

Code: `block_reconstructions(...)`, called from `sample_decomposition`
in [`lrad/ensemble.py`](../lrad/ensemble.py).

### Step B — ensemble-mean reconstruction `f̄_k`

Average over the `M` models (`decomposition_maps` in
[`lrad/ensemble.py`](../lrad/ensemble.py)):

```python
recons      = torch.stack([recons_per_model[m][k] for m in range(M)], dim=0)  # (M, B, 3, H, W)
mean_recon  = recons.mean(dim=0)                                              # (B, 3, H, W)  == f̄_k
```

This `mean_recon` is what gets handed to the plot as `mean_recons` in
[`scripts/run_ensemble.py`](../scripts/run_ensemble.py).

### Step C — squared-error map (summed over RGB)

`plot_mean_abs_bias` in [`lrad/plots.py`](../lrad/plots.py) builds one
`(B, H, W)` error map per block:

```python
block_maps = [_sq_error(images_np, recons_np[j]) for j in range(n_blocks)]
```

with:

```python
def _sq_error(orig, recon):
    return ((orig - recon) ** 2).sum(axis=-1)    # (x - f̄)² summed over the 3 RGB channels
```

The square is taken **before** the channel sum, so no sign cancellation; no mean,
no square root, so a pixel lives in `[0, 3]`. Images are first clamped to `[0, 1]`
and reordered to `(H, W, 3)` by `_to_image_grid`.

### Step D — the displayed number = spatial mean

The shared grid helper `_block_heatmap_grid` (also in
[`lrad/plots.py`](../lrad/plots.py)) writes, under each tile, the mean of its
`(H, W)` map over **all** pixels:

```python
ax_e.text(0.5, 0.04, annot_fmt.format(float(e.mean())), ...)   # annot_fmt = "{:.3f}"
```

### Compact formula

For image `x`, block `k`, an ensemble of `M` models, an `H × W` image, 3 channels:

```text
f̄_k = (1/M) · Σ_{m=1..M} f_{m,k}(x)

number(k) = (1 / (H · W)) · Σ_{i,j} [ Σ_{c∈{R,G,B}} ( x_{i,j,c} − f̄_{k,i,j,c} )² ]
```

i.e. the **RGB-summed L2 error, averaged spatially** between the image and the
ensemble-mean reconstruction — exactly the bias term. (Sum over channels, no
`1/3`, no square root.)

---

## 3. Full worked example

To keep it readable: a tiny `H = W = 2` image (4 pixels), 3 RGB channels, and an
ensemble of `M = 3` models, for a single block `k`. (In the real plot: `H = W =
64`, `M = 10`.)

### Data: original image `x`

| pixel | R    | G    | B    |
|-------|------|------|------|
| (0,0) | 0.80 | 0.80 | 0.80 |
| (0,1) | 0.50 | 0.50 | 0.50 |
| (1,0) | 0.20 | 0.20 | 0.20 |
| (1,1) | 0.60 | 0.30 | 0.30 |

### The 3 models' reconstructions (block `k`)

| pixel | model 1 (R,G,B)      | model 2 (R,G,B)      | model 3 (R,G,B)      |
|-------|----------------------|----------------------|----------------------|
| (0,0) | 0.70, 0.70, 0.70     | 0.74, 0.74, 0.74     | 0.78, 0.78, 0.78     |
| (0,1) | 0.55, 0.55, 0.55     | 0.50, 0.50, 0.50     | 0.45, 0.45, 0.45     |
| (1,0) | 0.25, 0.25, 0.25     | 0.20, 0.20, 0.20     | 0.30, 0.30, 0.30     |
| (1,1) | 0.50, 0.30, 0.30     | 0.56, 0.36, 0.36     | 0.62, 0.30, 0.30     |

### Step B — ensemble mean `f̄_k`

| pixel | R    | G    | B    | detail                              |
|-------|------|------|------|-------------------------------------|
| (0,0) | 0.74 | 0.74 | 0.74 | (0.70+0.74+0.78)/3 = 0.74           |
| (0,1) | 0.50 | 0.50 | 0.50 | (0.55+0.50+0.45)/3 = 0.50           |
| (1,0) | 0.25 | 0.25 | 0.25 | (0.25+0.20+0.30)/3 = 0.25           |
| (1,1) | 0.56 | 0.32 | 0.32 | R: (0.50+0.56+0.62)/3=0.56 ; G,B: (0.30+0.36+0.30)/3=0.32 |

### Step C — square the error, then sum over RGB → `(H, W)` map

| pixel | (R)²                | (G)²  | (B)²  | RGB sum |
|-------|---------------------|-------|-------|-----------|
| (0,0) | (0.80−0.74)²=0.0036 | 0.0036| 0.0036| **0.0108**|
| (0,1) | (0.50−0.50)²=0.0000 | 0.0000| 0.0000| **0.0000**|
| (1,0) | (0.20−0.25)²=0.0025 | 0.0025| 0.0025| **0.0075**|
| (1,1) | (0.60−0.56)²=0.0016 | (0.30−0.32)²=0.0004 | 0.0004 | 0.0016+0.0004+0.0004 = **0.0024** |

### Step D — spatial mean = the number under the tile

```text
number(k) = (0.0108 + 0.0000 + 0.0075 + 0.0024) / 4
          = 0.0207 / 4
          = 0.005175
          ≈ 0.005     # shown as .3f under the tile
```

**Result: `0.005`** — that's the white number that would appear under this tile.

---

## 4. Reading it

- **Small number** (dark tile) → the mean reconstruction tracks the image well:
  low bias. That's the **ID** rows.
- **Large number** (bright tile) → strong reconstruction error = high bias, a sign
  of anomaly. That's the **OOD** rows, over the glasses region.
- Comparing columns (L0, L1, …) shows at **which conv block** the OOD bias becomes
  most discriminative.

> Link to the decomposition: this tile shows `(x − f̄_k)²`, which is **exactly**
> the *bias* term of `Risk = Bias + Variance` (see
> [`lrad/ensemble.py`](../lrad/ensemble.py), `decomposition_maps`, key `"bias"`).
> Same quantity as the "Bias" column of `ensemble_decomposition.png`.

---

## 5. Reproduce it (minimal snippet)

```python
import numpy as np

x = np.array([[[0.80,0.80,0.80],[0.50,0.50,0.50]],
              [[0.20,0.20,0.20],[0.60,0.30,0.30]]])          # (H, W, 3)

recons = np.array([
    [[[0.70]*3,[0.55]*3],[[0.25]*3,[0.50,0.30,0.30]]],       # model 1
    [[[0.74]*3,[0.50]*3],[[0.20]*3,[0.56,0.36,0.36]]],       # model 2
    [[[0.78]*3,[0.45]*3],[[0.30]*3,[0.62,0.30,0.30]]],       # model 3
])                                                            # (M, H, W, 3)

mean_recon = recons.mean(axis=0)                              # f̄_k  (step B)
err_map    = ((x - mean_recon) ** 2).sum(axis=-1)            # (x-f̄)² summed over RGB (step C)
number     = float(err_map.mean())                           # spatial mean (step D)

print(round(number, 3))   # -> 0.005
```
