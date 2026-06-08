# Bias / Variance decomposition of a deep ensemble

## Contents

1. [Context and motivation](#1-context-and-motivation)
2. [The three terms: definitions and formulas](#2-the-three-terms-definitions-and-formulas)
3. [Proof: Risk = Bias + Variance](#3-proof-risk--bias--variance)
4. [Interpretation](#4-interpretation)
5. [Full worked example on an RGB pixel](#5-full-worked-example-on-an-rgb-pixel)
6. [Why the Bias is the anomaly score](#6-why-the-bias-is-the-anomaly-score)
7. [Why the Variance is a weak OOD signal](#7-why-the-variance-is-a-weak-ood-signal)

---

## 1. Context and motivation

We train **M independent models** (a deep ensemble) — same architecture, same
data, different random initializations. For each convolutional block `k`, every
model `m` produces a reconstruction `f̂ᵐₖ(x)` of the input image `x`.

The goal is to flag **out-of-distribution** (OOD) images — faces with
eyeglasses, when training only ever saw faces without them.

The question from the start was: **what signal says an image is OOD?**

- A single model's error is too noisy to trust.
- The mean over M models is steadier, but how do we read it?
- How do we split that error so the part that comes from a genuine anomaly is
  separated from the part that's just models disagreeing with each other?

That split is exactly what the **bias/variance decomposition** gives us.

![Per-block Risk / Bias / Variance maps. ID rows stay dark; the OOD rows light
up around the eyeglasses.](figures/ensemble_decomposition.png)

---

## 2. The three terms: definitions and formulas

Fix a block `k`, an image `x`, and a pixel `i`, with the M reconstructions `f̂ᵐ`
and their ensemble mean. Throughout, the per-pixel squared error `( · )²` is the
**sum over the 3 RGB channels** `Σ_c ( x_c − f̂_c )²` (no mean, no square root),
so a single pixel lives in `[0, 3]`:

```
f̄(x)[i] = (1/M) Σₘ f̂ᵐ(x)[i]
```

### Risk

```
Risk_k(x)[i] = (1/M) Σₘ ( x[i] − f̂ᵐ(x)[i] )²
```

The **mean squared error over the M models** at this pixel — the expected cost
of drawing one model at random from the ensemble.

### Bias

```
Bias_k(x)[i] = ( x[i] − f̄(x)[i] )²
```

The squared error of the **consensus model** — the hypothetical model whose
prediction is exactly the ensemble mean. It measures the **irreducible** error:
what survives even after averaging every model together.

### Variance

```
Variance_k(x)[i] = (1/M) Σₘ ( f̂ᵐ(x)[i] − f̄(x)[i] )²
```

The mean spread of the models around their own average — the **epistemic
uncertainty**, i.e. what the models can't agree on.

### Exact identity

The three terms satisfy the following algebraic identity, **pixel by pixel**:

```
Risk = Bias + Variance
```

---

## 3. Proof: Risk = Bias + Variance

### Notation

For a fixed pixel `i`, write:

```
a    = x[i]               (ground-truth value)
bₘ   = f̂ᵐ(x)[i]          (model m's reconstruction)
b̄    = (1/M) Σₘ bₘ        (ensemble mean = f̄(x)[i])
```

### Expansion

Start from the Risk and split each error into two pieces:

```
a − bₘ = (a − b̄) + (b̄ − bₘ)
          ───────   ─────────
          consensus  model m's
          error      deviation
                     from the mean
```

Expand the square:

```
Risk = (1/M) Σₘ (a − bₘ)²
     = (1/M) Σₘ [ (a − b̄) + (b̄ − bₘ) ]²
     = (1/M) Σₘ [ (a − b̄)²  +  2(a − b̄)(b̄ − bₘ)  +  (b̄ − bₘ)² ]
```

Distribute the sum across the three terms:

```
     = (a − b̄)²
     + 2(a − b̄) · (1/M) Σₘ (b̄ − bₘ)
     + (1/M) Σₘ (b̄ − bₘ)²
```

### The cross term vanishes

The middle term contains:

```
(1/M) Σₘ (b̄ − bₘ) = b̄ − (1/M) Σₘ bₘ = b̄ − b̄ = 0
```

The mean of the deviations from the mean is always zero, so the cross term drops.

### Result

```
Risk = (a − b̄)²  +  (1/M) Σₘ (b̄ − bₘ)²
       ─────────      ──────────────────────
         Bias                 Variance

⟹  Risk = Bias + Variance     ∎
```

This is an **exact** identity, not an approximation. It holds pixel by pixel, up
to floating-point error (the code checks the max residual at run time — it lands
around `1e-7`).

---

## 4. Interpretation

### Risk — "what if I draw a model at random?"

Risk is the expected error of a **randomly picked** ensemble member — what we'd
pay by deploying a single model chosen by lottery.

- **High on ID** when the members are individually poor but cancel out on average.
- **High on OOD** in all cases.
- **Limitation**: it mixes two signals (systematic error + disagreement), which
  makes it less diagnostic.

### Bias — "even in agreement, does the ensemble get it wrong?"

Bias is the error of the **consensus model** `f̄`. The underlying question: *even
with the best available estimator (the mean), is the reconstruction right?*

A caveat on the name: "bias" here is not a constant architectural offset. It's a
**per-image** quantity:

- On a normal face (ID): `f̄` reconstructs well → Bias ≈ 0.
- On a face with glasses (OOD): `f̄` can't reconstruct the glasses → Bias >> 0.

This is the **irreducible** error — what no amount of model diversity can fix,
because the failure is systematic for that input.

| Lens | Reading |
|---|---|
| Statistical | Error of the optimal estimator (the mean) |
| Reconstruction | What the ensemble can't rebuild, even in consensus |
| OOD | Signature of an input outside the training distribution |
| Decision | High Bias → the image is probably OOD |

### Variance — "do the models agree with each other?"

Variance measures **inter-model disagreement** — their spread around their own
mean. It's the **epistemic uncertainty** signal: models extrapolate differently
on unfamiliar inputs.

- On ID: the models learned the same patterns → little disagreement → Variance ≈ 0.
- On OOD: the models extrapolate differently → disagreement → higher Variance.

| Lens | Reading |
|---|---|
| Statistical | Spread of the individual predictions around the mean |
| Uncertainty | Epistemic-uncertainty signal |
| Localization | The variance map shows *where* the models hesitate |
| Diagnostic | Useful for understanding uncertain regions, not for the final call |

---

## 5. Full worked example on an RGB pixel

### Setup

- **M = 3 models**, **1 pixel** at position `(h, w)` of some block `k`.
- The image is RGB, so the per-pixel squared error is the **sum over the 3
  channels** `Σ_c (x_c − f̂_c)²` (no mean, no square root) — a pixel lives in `[0, 3]`.

### Ground-truth pixel

```
x[i] = (R = 0.9,  G = 0.6,  B = 0.3)
```

### The 3 models' reconstructions

| Model | R    | G    | B    |
|-------|------|------|------|
| f̂¹    | 0.70 | 0.50 | 0.20 |
| f̂²    | 0.80 | 0.60 | 0.30 |
| f̂³    | 0.90 | 0.70 | 0.40 |

### Step 1 — ensemble mean f̄ (per channel)

```
f̄_R = (0.70 + 0.80 + 0.90) / 3 = 2.40 / 3 = 0.800
f̄_G = (0.50 + 0.60 + 0.70) / 3 = 1.80 / 3 = 0.600
f̄_B = (0.20 + 0.30 + 0.40) / 3 = 0.90 / 3 = 0.300
```

### Step 2 — Risk

Each model's squared error, **summed** over the 3 RGB channels:

```
se₁ = (0.9 − 0.70)² + (0.6 − 0.50)² + (0.3 − 0.20)²
    =   0.0400      +    0.0100      +    0.0100
    = 0.0600

se₂ = (0.9 − 0.80)² + (0.6 − 0.60)² + (0.3 − 0.30)²
    =   0.0100      +    0.0000      +    0.0000
    = 0.0100

se₃ = (0.9 − 0.90)² + (0.6 − 0.70)² + (0.3 − 0.40)²
    =   0.0000      +    0.0100      +    0.0100
    = 0.0200
```

```
Risk = (1/M) × [se₁ + se₂ + se₃]    (mean over the M = 3 models)
     = (1/3) × [0.0600 + 0.0100 + 0.0200]
     = (1/3) × 0.0900
     = 0.0300
```

### Step 3 — Bias

Squared error of the consensus model, **summed** over the 3 channels:

```
Bias = (0.9 − 0.800)² + (0.6 − 0.600)² + (0.3 − 0.300)²
     =   (0.100)²     +      0          +       0
     = 0.0100
```

Only the **red** channel contributes — green and blue are reconstructed
perfectly on average.

### Step 4 — Variance

Each model's deviation from f̄, **summed** over the 3 channels:

```
var₁ = (0.70 − 0.800)² + (0.50 − 0.600)² + (0.20 − 0.300)²
     =     0.0100      +     0.0100      +     0.0100
     = 0.0300

var₂ = (0.80 − 0.800)² + (0.60 − 0.600)² + (0.30 − 0.300)²
     =     0.0000      +      0.0000     +      0.0000
     = 0.0000

var₃ = (0.90 − 0.800)² + (0.70 − 0.600)² + (0.40 − 0.300)²
     =     0.0100      +     0.0100      +     0.0100
     = 0.0300
```

```
Variance = (1/M) × [var₁ + var₂ + var₃]    (mean over the M = 3 models)
         = (1/3) × [0.0300 + 0.0000 + 0.0300]
         = (1/3) × 0.0600
         = 0.0200
```

### Step 5 — check the identity

```
Bias + Variance = 0.0100 + 0.0200 = 0.0300 = Risk  ✓
```

### Summary

| Term     | Value    | What it means for this pixel                          |
|----------|----------|-------------------------------------------------------|
| Risk     | 0.0300   | Mean error of a randomly drawn model                  |
| Bias     | 0.0100   | Consensus error — R channel slightly underestimated   |
| Variance | 0.0200   | Disagreement — models 1 and 3 pull apart              |

Here the **variance dominates** (the models scatter between 0.70 and 0.90 on R)
and the **bias is small** (the mean 0.80 is only 0.10 off 0.90). This pixel reads
as barely anomalous.

---

## 6. Why the Bias is the anomaly score

### What each term captures on ID vs OOD

| Situation | Bias | Variance | Diagnosis |
|---|---|---|---|
| ID, normal | ≈ 0 | ≈ 0 | Normal image |
| OOD, models diverge | **High** | High | OOD caught by both |
| OOD, models fail the same way | **High** | ≈ 0 | Only Bias catches it |

### Why not Risk?

```
Risk = Bias + Variance
```

Risk blends both signals. On ID data, members that are individually imperfect but
average out well give a high Risk and a low Bias — so Risk is the noisier signal.

### Why not Variance?

Variance is zero when every model fails the same way. If all M models learned the
same wrong representation of an OOD pattern, they produce the same bad
reconstruction — the variance stays low and the anomaly slips through.

### Why Bias?

Bias measures whether the consensus model `f̄` fails to reconstruct the input.
Even if every model is wrong in the same way, `f̄` is still wrong, so `(x − f̄)²`
is large.

```
Extreme case: all f̂ᵐ identical
→ f̄ = f̂ᵐ for every m
→ Variance = 0
→ Risk = Bias
→ Bias still captures the anomaly
```

So Bias is the most **robust and direct** signal — it simply says *"the ensemble
can't reconstruct this image."*

---

## 7. Why the Variance is a weak OOD signal

### The ensemble-collapse problem

When the M models share architecture, data, and a similar loss, they tend to
learn the same inductive biases. Faced with an OOD input they all extrapolate the
same way, so the disagreement stays small:

```
f̂¹(glasses) ≈ f̂²(glasses) ≈ f̂³(glasses)   (all reconstruct a glasses-free face)
→ Variance ≈ 0
→ but (x − f̄)² >> 0   (the glasses still aren't reconstructed)
→ Bias catches it, Variance misses it
```

That's exactly what we see empirically — the AUROCs confirm Bias > Risk > Variance.

### Where Variance still earns its keep

- **Visualization**: the variance map localizes *where* the models hesitate, even
  if it doesn't drive the final decision.
- **Diagnostics**: high variance on an ID region can flag a structurally hard area
  (occlusion, blur).
- **Validation**: comparing the three terms' AUROCs is the empirical proof that
  Bias is the right pick — without it, choosing Bias would just be a hunch.

### Summary

```
Variance → "the models are uncertain here"   (uncertainty signal)
Bias     → "even the consensus is wrong here" (error signal)

For OOD detection: Bias >> Variance in robustness
For interpretability: the two are complementary
```
