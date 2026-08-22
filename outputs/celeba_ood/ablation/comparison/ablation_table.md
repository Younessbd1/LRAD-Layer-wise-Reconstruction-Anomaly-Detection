# CelebA OOD ablation — results

Arms (10-member ensembles, 128 px, seeds 42..51):

- **Baseline (same arch)** — `/home/ybahaddo/lrad/outputs/celeba_ood/ablation/baseline_20260819_133802_6866854`
- **Arch diversity** — `/home/ybahaddo/lrad/outputs/celeba_ood/ablation/arch_20260819_142151_6866855`
- **CutPaste** — `/home/ybahaddo/lrad/outputs/celeba_ood/ablation/cutpaste_20260819_192140_6866856`
- **Arch + CutPaste** — `/home/ybahaddo/lrad/outputs/celeba_ood/LASTOF_RESULTS`

| Metric (OOD AUROC) | Baseline (same arch) | Arch diversity | CutPaste | Arch + CutPaste |
|---|---|---|---|---|
| fused (supervised) | 0.8730 | 0.8563 | 0.8710 | 0.8638 |
| fused (rank) | 0.8420 | 0.8415 | 0.8247 | 0.8040 |
| locfre b1 | 0.7833 | 0.7849 | 0.7928 | 0.7898 |
| locfre b2 | 0.7406 | 0.7677 | 0.7600 | 0.7773 |
| locfre b3 | 0.7710 | 0.7916 | 0.7825 | 0.7952 |
| cutpaste P(altered) | — | — | 0.7500 | 0.6738 |
| gender energy | 0.8112 | 0.7787 | 0.7435 | 0.7127 |
| epistemic MI | 0.7336 | 0.7393 | 0.5717 | 0.5387 |
| decomp bias (p95) | 0.6246 | 0.6274 | 0.6214 | 0.6228 |
| decomp risk (p95) | 0.6339 | 0.6350 | 0.6294 | 0.6304 |
| decomp variance (p95) | 0.6510 | 0.6457 | 0.6452 | 0.6425 |
| localized z-score (risk) | 0.6988 | 0.7012 | 0.6996 | 0.7029 |
| member recon AUROC (mean ± sd) | 0.6348 ± 0.0031 | 0.6375 ± 0.0044 | 0.6301 ± 0.0024 | 0.6319 ± 0.0039 |
| member gender acc (mean ± sd) | 0.9717 ± 0.0150 | 0.9557 ± 0.0305 | 0.9374 ± 0.1033 | 0.9401 ± 0.1041 |

## Case studies — Δ AUROC vs baseline

### 1. Arch diversity vs baseline

| Metric | Δ vs baseline |
|---|---|
| fused (supervised) | -0.0167 |
| fused (rank) | -0.0005 |
| locfre b3 | +0.0206 |
| gender energy | -0.0325 |
| epistemic MI | +0.0056 |
| decomp bias (p95) | +0.0028 |
| decomp variance (p95) | -0.0053 |
| localized z-score (risk) | +0.0024 |

### 2. CutPaste vs baseline

| Metric | Δ vs baseline |
|---|---|
| fused (supervised) | -0.0020 |
| fused (rank) | -0.0173 |
| locfre b3 | +0.0115 |
| gender energy | -0.0677 |
| epistemic MI | -0.1619 |
| decomp bias (p95) | -0.0033 |
| decomp variance (p95) | -0.0058 |
| localized z-score (risk) | +0.0008 |

### 3. Arch + CutPaste vs baseline

| Metric | Δ vs baseline |
|---|---|
| fused (supervised) | -0.0092 |
| fused (rank) | -0.0380 |
| locfre b3 | +0.0241 |
| gender energy | -0.0986 |
| epistemic MI | -0.1949 |
| decomp bias (p95) | -0.0018 |
| decomp variance (p95) | -0.0085 |
| localized z-score (risk) | +0.0040 |

