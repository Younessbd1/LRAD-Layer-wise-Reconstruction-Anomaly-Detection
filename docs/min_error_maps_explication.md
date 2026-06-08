# Ensemble *minimum* error map — `min_error_maps.png`

How `min_error_maps.png` (produced by `scripts/run_ensemble.py`) is computed,
formula plus a worked example. It's the **minimum** variant of
[`mean_error_maps.png`](mean_error_maps_explication.md): same visualization, but
the mean over models is replaced by the **per-pixel minimum**.

---

## 1. What we compute

For each conv block `k`, take **every model's** reconstruction-error map and, at
each pixel, keep the **smallest** error among the `M` models. The error is the
**squared** error, as everywhere else since the move to L2.

```text
MinErr_k(x)[i] = min_m ( x[i] − f̂ᵐ_k(x)[i] )²
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

![Per-block min_m (x − f̂ᵐ)² — only regions no member can reconstruct stay
bright.](figures/min_error_maps.png)

> **Interpretation.** Where `MeanErr` answers "on average, how badly does the
> ensemble miss this pixel", `MinErr` answers "how well does the **best** member
> do on this pixel". A region stays bright (high error) only if **no** model can
> reconstruct it — a stricter OOD signal that a single bad member can't inflate.

---

## 2. Min vs Mean: more than just the operator

The only difference from `MeanErr` is the aggregation over models: `min` instead
of `mean`. We **always** have:

```text
min_m ( x − f̂ᵐ )²  ≤  (1/M) Σₘ ( x − f̂ᵐ )²
   MinErr         ≤        MeanErr
```

(the minimum of a set is always ≤ its mean). The two maps are equal only when
every model has the same error at that pixel. So the gap between them comes,
again, from **inter-model disagreement**: the more the models diverge, the darker
`MinErr` is relative to `MeanErr`, since a single member landing it is enough.

| Quantity | Formula | What it is |
|---|---|---|
| **MeanErr / Risk** | `(1/M) Σₘ ( x − f̂ᵐ )²` | **mean** error of the ensemble |
| **MinErr** (this map) | `min_m ( x − f̂ᵐ )²` | error of the **best** model |
| **Bias** | `( x − f̄ )²` | error of the **averaged** reconstruction |

Careful: `MinErr` is **not** the bias and has **no** fixed inequality with it (the
min can sit above or below the bias depending on the pixel).

---

## 3. Worked example

Reuse the exact `mean_error_maps` example: **1 pixel on a single channel**, **M =
3 models**, one block (a real image would sum 3 terms, one per RGB channel). True
pixel value:

```text
x = 0.80
```

The 3 models' reconstructions for this pixel:

```text
f̂¹ = 0.50      f̂² = 0.90      f̂³ = 0.70
```

### Step 1 — each model's squared error

```text
( 0.80 − 0.50 )² = 0.09      (model 1)
( 0.80 − 0.90 )² = 0.01      (model 2)
( 0.80 − 0.70 )² = 0.01      (model 3)
```

### Step 2 — keep the minimum (instead of the mean)

```text
MinErr  = min(0.09, 0.01, 0.01) = 0.01
MeanErr = (0.09 + 0.01 + 0.01)/3 = 0.03667   (for comparison)
```

### Takeaway

```text
MinErr = 0.01   ≤   MeanErr = 0.03667
```

Model 1 misses (`0.09`), but models 2 and 3 nail it (`0.01`). `MeanErr` keeps a
blend of all three and stays at `0.03667`; `MinErr` keeps only the best (`0.01`)
and so "forgives" the bad model. If instead **all three** had missed (say squared
errors `0.09, 0.078, 0.096`), `MinErr` would have stayed high (`0.078`) — exactly
the "nobody can reconstruct this" case typical of an OOD anomaly.

---

## 4. The real computation (RGB + tensors)

Real images have 3 RGB channels and a batch. The code (`min_error_maps` in
[`lrad/ensemble.py`](../lrad/ensemble.py)) does:

```python
recons = torch.stack(                       # (M, B, 3, H, W)
    [recons_per_model[m][k] for m in range(n_models)], dim=0,
)
se = ((images.unsqueeze(0) - recons) ** 2).sum(dim=2)    # (M, B, H, W)
out[k] = se.min(dim=0).values                            # (B, H, W)
```

Step by step:

1. **`images.unsqueeze(0) - recons`** → signed error `x − f̂ᵐ`, shape `(M, B, 3, H, W)`.
2. **`** 2`** → square, per pixel and per channel (square **before** the channel
   sum, so no sign cancellation).
3. **`.sum(dim=2)`** → **sum** over the **3 RGB channels** → `(M, B, H, W)`. (No
   mean, no square root: a pixel lives in `[0, 3]` — a single model's error map,
   same as the other plots.)
4. **`.min(dim=0).values`** → minimum over the **M models** → `(B, H, W)`.

The **only** line that differs from `mean_error_maps` is the last:
`se.mean(dim=0)` → `se.min(dim=0).values`. `torch.min(dim=...)` returns a
`(values, indices)` pair; we keep only `.values` (the best model's error), not
`.indices` (which model was best).

The display is `plot_min_error_maps` in [`lrad/plots.py`](../lrad/plots.py): one
row per image, one column per block, `viridis` colormap, **fixed scale `vmin=0,
vmax=3`** (3 channels, never a max from the displayed subset), and each tile's
mean written under it in white. The colour-bar label reads `min_m (x − f̂^m)²`.

---

## 5. Reading the image

- **Rows**: the samples (ID on top, OOD below — see `row_labels`).
- **Columns**: `Original`, then one error map per block `L0…Ln`.
- **Colour**: brighter (yellow in viridis) = higher error from the **best** model,
  i.e. a region harder for the **whole** ensemble to reconstruct.
- **Number under each tile**: mean of the minimum-error map over the whole tile.

Expected: on **ID** images the maps are even darker than for `MeanErr` (at least
one model reconstructs well everywhere). On **OOD** images only the regions where
**no** model copes stay lit (e.g. the glasses) — the noise from a single weak
member drops out, which sharpens the anomaly.

---

## 6. Regenerating it

`min_error_maps.png` (like `mean_error_maps.png`) is written on every run of the
pipeline, in the **same run** as training — there's no separate evaluation job
anymore. On Grid'5000:

```bash
./scripts/oar_run_ensemble.sh                       # immediate submission
./scripts/oar_run_ensemble.sh '2026-06-10 20:00:00' # advance reservation
```

Locally (no OAR):

```bash
python scripts/run_ensemble.py --config configs/celeba_ood.yaml
```

> `run_ensemble.py` still has an `--eval-only` flag (reloads already-trained
> weights to redo only the decomposition + plots), but since we moved to a very
> short schedule (2 epochs, to keep the ensemble diverse), the simplest path is
> just to rerun a full job.
