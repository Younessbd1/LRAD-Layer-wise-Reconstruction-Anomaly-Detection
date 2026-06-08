# Clarifications — Bias / Variance decomposition

Loose ends and things that confused me while building the decomposition,
written down so they stay answered.

---

## 1. Reconstruction error vs Risk

The **reconstruction error** of a single model `m` at pixel `i` is:

```text
err_m(x)[i] = ( x[i] − f̂ᵐ(x)[i] )²
```

That's the number you get from running one model on one image. Each model gives a
different value, so this error is random in *which* of the M models you happened
to pick.

The **Risk** is the mean of those errors over all M models:

```text
Risk(x)[i] = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
```

So Risk is the average of the individual errors — what you'd expect from
deploying a single model drawn at random from the ensemble.

A concrete analogy: if 3 archers each take one shot at a target, the
reconstruction error is one archer's distance; the Risk is the average distance
of the 3 shots.

---

## 2. Why Risk is the "expected cost of a random model"

Drawing a model at random from M models means drawing `m` uniformly from
`{1, 2, …, M}`, each with probability `1/M`. The expected cost is:

```text
E_m[ (x[i] − f̂ᵐ(x)[i])² ]
  = Σₘ P(m) · (x[i] − f̂ᵐ(x)[i])²
  = Σₘ (1/M) · (x[i] − f̂ᵐ(x)[i])²
  = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
  = Risk(x)[i]
```

That's exactly the Risk formula — no approximation: Risk is the empirical mean of
the squared error over the ensemble.

### Worked example

Take M = 3 models on a pixel whose true value is `x[i] = 0.8`:

| Model | Prediction | Squared error |
|-------|-----------|-----------------|
| m = 1 | 0.60      | (0.8 − 0.60)² = 0.0400 |
| m = 2 | 0.80      | (0.8 − 0.80)² = 0.0000 |
| m = 3 | 1.00      | (0.8 − 1.00)² = 0.0400 |

```text
Risk = (0.0400 + 0.0000 + 0.0400) / 3 = 0.0267
```

Deploy a single random model and its expected error is `0.0267`. Model 2 would
score `0.0000`, models 1 and 3 would score `0.0400` — Risk folds all of that into
one number.

---

## 3. The consensus model (the "fictional" model)

### Consensus model = fictional model = f̄

```text
f̄(x)[i] = (1/M) Σₘ f̂ᵐ(x)[i]
```

This "model" doesn't exist in memory as a trained network. It's an
**abstraction**: if you could build one network whose predictions were exactly
the mean of the M networks, that would be the consensus model. I call it
"fictional" because it's never trained — it falls out mechanically as the average
of the outputs.

### Example

Same pixel `x[i] = 0.8`, M = 3 models, predictions 0.60, 0.80, 1.00.

```text
f̄(x)[i] = (0.60 + 0.80 + 1.00) / 3 = 0.800
```

The fictional model predicts exactly `0.800`. The Bias at this pixel is:

```text
Bias = (0.8 − 0.800)² = 0.0000
```

The individual models were off (errors of ±0.2), but their mean is spot on.
→ zero Bias: model diversity cancelled the errors.

### Counter-example: OOD (glasses)

On a pixel that sits on an eyeglass frame, all 3 models reconstruct a patch of
skin — none of them knows how to draw the glasses. Predictions: 0.75, 0.78, 0.72;
true value `x[i] = 0.20` (dark frame).

```text
f̄(x)[i] = (0.75 + 0.78 + 0.72) / 3 = 0.750
Bias    = (0.20 − 0.750)² = 0.3025
```

The consensus model is badly wrong. This isn't missing diversity — the whole
ensemble simply has no concept of glasses.

---

## 4. Bias = irreducible error: but is it really zero on ID?

### What the formula says

```text
Bias(x)[i] = ( x[i] − f̄(x)[i] )²
```

This error is "irreducible" in a specific sense: however many models you add, the
mean `f̄` converges to `E[f̂(x)[i]]` — the expectation over random initializations.
If that expectation isn't `x[i]`, the Bias stays positive even as M → ∞.

### On ID, the Bias is not exactly zero

"Bias ≈ 0 on ID" is a teaching simplification. In practice, even on normal images
the Bias is small but nonzero, for a few reasons:

**a) Edges and fine contours**

A ConvTranspose2D can't perfectly reconstruct sharp transitions between colour
regions. The spatial resolution lost in encoding (max pooling) isn't fully
recovered. These edge errors are systematic — they persist no matter how many
models you average.

**b) ConvTranspose2D checkerboard artifacts**

ConvTranspose2D produces checkerboard artifacts on some pixels from the uneven
overlap of its kernels. These are baked into the architecture — they depend on
stride and kernel size, not on initialization — so they feed the Bias, not the
Variance.

```text
x[i]  = 0.52  (edge pixel, normal ID image)
f̄[i]  = 0.58  (ConvTranspose2D overlap artifact)
Bias  = (0.52 − 0.58)² = 0.0036   ← small but nonzero
```

**c) Limited architecture capacity**

The encoder compresses the image through conv layers; fine detail (hair texture,
skin pores) can't fit in the latent space, and decompression can't invent it
back. That loss is fundamental — even the consensus `f̄` can't reconstruct what
the representation no longer holds.

### Where does this sit in the formula?

There's no separate term — it's all inside `( x[i] − f̄(x)[i] )²`. That's why we
call it "irreducible": it isn't random model-to-model error (which averages away),
it's a systematic limit of what the ensemble can represent.

On ID: these irreducible errors are small → small Bias.
On OOD: an extra systematic error stacks on top (the unknown region) → large Bias.

---

## 5. High Risk on ID when models are individually bad but cancel out

This is the case where the models have **opposing** errors that average to zero —
and it threw me at first.

### Numerical example

Pixel `x[i] = 0.50`, 4 models:

| Model | Prediction | Error (x − f̂ᵐ) |
|-------|-----------|-------------------|
| m = 1 | 0.20      | +0.30 |
| m = 2 | 0.80      | −0.30 |
| m = 3 | 0.15      | +0.35 |
| m = 4 | 0.85      | −0.35 |

Every model is way off. And yet:

```text
f̄ = (0.20 + 0.80 + 0.15 + 0.85) / 4 = 2.00 / 4 = 0.50

Bias     = (0.50 − 0.50)² = 0.0000   ← perfect!
Variance = (1/4) × [(0.20−0.50)² + (0.80−0.50)² + (0.15−0.50)² + (0.85−0.50)²]
         = (1/4) × [0.0900 + 0.0900 + 0.1225 + 0.1225]
         = (1/4) × 0.4250
         = 0.1063

Risk = Bias + Variance = 0.0000 + 0.1063 = 0.1063
```

The Risk is high (0.1063) on a perfectly normal ID image — because each model is
far off, but their errors are exactly opposed.

A single random model would average an error of `0.1063`. The ensemble mean has
error 0.

→ This is why Risk is a poor OOD signal: it can be high even on normal images,
driven by model scatter rather than a real anomaly.

---

## 6. The ensemble-collapse problem

### Why it happens even with different seeds

Changing the seed changes:
- the batch order during training,
- the weight initialization.

But it does **not** change:
- the architecture (same layers, same structure),
- the training data (same distribution, same learned patterns),
- the loss (same objective),
- the inductive biases (a ConvNet favours local textures, translation
  invariance, and so on).

So the M models get different weights but converge to **similar solutions in
function space**. On ID inputs they answer slightly differently (hence Variance >
0 but small). On OOD inputs that leave their shared distribution, they all
extrapolate to the same wrong answer — the one their common inductive bias
dictates.

### Concrete example

The ensemble is trained only on glasses-free faces. Every model learns that the
region around the eyes = skin + eyebrows, encoded in the filters of the first few
layers.

When a glasses image arrives:

```text
Model 1 (seed=42) :  "something odd around the eyes → reconstruct skin"
Model 2 (seed=7)  :  "something odd around the eyes → reconstruct skin"
Model 3 (seed=123):  "something odd around the eyes → reconstruct skin"

f̄ ≈ skin   (strong consensus)
Variance ≈ 0 (they agree)
Bias >> 0  (the real image shows glasses, not skin)
```

It's not a seed issue — all 3 models learned the same "eye region = skin"
concept. Different seeds just gave them slightly different weights to reach the
same functional result.

### What would actually add diversity

- Heterogeneous architectures (ConvNet + ViT + ResNet)
- Different training data (disjoint subsets)
- Different losses (MSE + perceptual + adversarial)

With M models that share architecture, data, and only differ by seed, the
Variance stays a weak signal for **systematic** OODs like glasses. That's exactly
why Bias is the better signal.

---

## 7. Why we estimate with the mean everywhere

### What does the mean summarize?

The mean of `{v₁, v₂, …, vₙ}` is:

```text
v̄ = (1/n) Σᵢ vᵢ
```

It summarizes the **central tendency** of the distribution. It's the value that
minimizes the total squared error:

```text
v̄ = argmin_c Σᵢ (vᵢ − c)²
```

In other words: if you must pick one number to stand in for the whole
distribution, the mean is optimal in the least-squares sense.

### Why use it everywhere

**Case 1 — mean over the M models to get f̄**

We have M predictions `{f̂¹(x)[i], …, f̂ᴹ(x)[i]}` and want the best single estimate
from the ensemble. The mean is MSE-optimal: if the model errors are independent
and zero-mean, averaging cuts the noise by a factor of M.

```text
Variance of f̄ = Variance of f̂ᵐ / M   (if the errors are independent)
```

The larger M, the closer f̄ gets to the true value.

**Case 2 — spatial mean over pixels to get a scalar**

We have an `(H, W)` error map and want one number per image to compare images.
The spatial mean summarizes the overall anomaly intensity across the whole image.

```text
2×2 image, error map:
  [ 0.0  0.0 ]   ← normal region
  [ 0.0  0.8 ]   ← anomalous pixel (glasses)

mean = (0.0 + 0.0 + 0.0 + 0.8) / 4 = 0.200
```

The value `0.200` captures that something's wrong somewhere, even though three of
four pixels are fine.

**Case 3 — the number under each tile in `mean_abs_bias.png`**

In `mean_abs_bias.png`, the number under a tile is `(1/HW) Σᵢⱼ (x − f̄)²` with the
squared error summed over the 3 RGB channels. (The filename is historical — since
the switch to L2 the map is the **squared** error, not the absolute value.) It
boils the reconstruction-error intensity for that image at that block down to one
number.

### The mean vs other summaries

| Summary | What it captures | Used in this project for |
|--------|-----------------|----------------------|
| Mean | Global error, sensitive to the whole image | `mean_abs_bias`, Risk, Bias, Variance maps |
| Maximum | The single most anomalous pixel | `agg="max"` in `_reduce_over_pixels` |
| 95th percentile | A compromise — ignores lone hot pixels, fires on anomalous regions | `agg="p95"` (default in `aggregate_anomaly_score`) |

We use p95 as the default precisely because the mean is too diluted over 64×64
pixels: a small pair of glasses only touches a fraction of them, and the mean
washes the signal out. The p95 guarantees that if 5% of pixels are strongly
anomalous, the global score reflects it.

### The mean is not the distribution

The mean is a **summary**, not the full distribution. Two images can share the
same mean error with very different distributions:
- Image A: slightly blurry everywhere → 64×64 pixels at error 0.010
- Image B: a sharp pair of glasses → a few pixels at error 0.640, the rest at 0.000

Same mean = 0.010, but image B is OOD. That's why we also look at the max and the
p95 — and why the heatmaps stay complementary to the scalars.
