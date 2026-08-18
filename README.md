# LRAD — Layer-wise Reconstruction Anomaly Detection

OOD detection on CelebA faces via deep-ensemble reconstruction error decomposed into bias (anomaly) and variance (epistemic uncertainty).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

## What it does

A from-scratch convolutional classifier is trained on normal (glasses-free) CelebA faces; lightweight decoders attached to each conv block learn to invert that block's frozen activations back to a full-resolution `(3, H, W)` reconstruction (64 px in `celeba_ood.yaml`, 128 px in `celeba_ood_128.yaml`). Training `M` such models independently forms an **architecturally diverse deep ensemble** — each member has its own channel widths and conv kernel size (`ensemble.member_variants`), on top of its own weight init and SGD shuffle order — whose per-pixel reconstruction risk decomposes exactly as `Risk = Bias + Variance`. The **bias** term — the irreducible error of the consensus reconstruction — is the anomaly score; faces wearing **eyeglasses** are held out as OOD (the attribute set is configurable) and score higher.

## What was done in this run

- **Single decoder architecture — ConvTranspose2d only**: every ×2 upsampling stage is a learnable `ConvTranspose2d(4×4, stride 2)` + BN + ReLU; the bilinear resize-then-conv variant (and the `upsample` config knob) was removed from the whole pipeline.
- **Per-member architectures**: `ensemble.member_variants` gives each of the 10 members its own trunk (channel widths + 3×3 or 5×5 convs, ~0.9M–2.7M parameters) while keeping the exact same 5-block / pool-after-first-4 spatial layout, so the per-block decompositions stay aligned across members.
- **Glasses-only OOD**: `dataset.ood_attrs: [Eyeglasses]` — training/val/test_in contain only faces **without** eyeglasses (normal images); every face wearing glasses (sunglasses included) is held out as OOD. Wearing_Hat was dropped from the OOD set.
- **Eye-region attribute targets**: 4 of the 6 attribute heads are eye-region attributes (`Arched_Eyebrows, Bushy_Eyebrows, Narrow_Eyes, Bags_Under_Eyes`; `Smiling, Young` stay as global targets). Training only sees glasses-free faces, so the classifier must attend to the eye area — eyeglasses then occlude exactly that evidence, maximizing the ID/OOD activation gap the bias picks up.
- **Schedules**: 20 classifier epochs + 25 decoder epochs per member (`val_ratio=0`, no early stopping).
- **Per-instance figures restructured**: each test face gets its own folder with one figure **per ensemble member** (`model_01.png` … `model_10.png`: Original | Recon | Error) plus a `summary.png` (Original | Bias | Mean error | Min error on its raw scale | smoothed-bias overlay). The per-tile numeric score annotations were removed everywhere — the maps read through the shared colour bars. `overlay_sigma` (now 3) and `overlay_power` affect **display only**, never the score.
- **Top-10 OOD eyeglasses ranking**: the full OOD test set (restricted to `Eyeglasses` faces) is ranked by eye-region bias strength × concentration and the winners are written to `plots/top_ood_glasses.png` + `top_ood_glasses.json`.
- **Conference-paper figure styling**: serif (Times-like) typeface with STIX math, the Okabe–Ito colorblind-safe palette with fixed role assignments (ID = blue, OOD = vermillion, variance = orange), recessive axes (no top/right spines, light grid), 300 dpi export, fixed colour scales so figures stay comparable across runs.

## Architecture

**Classifier** — `FacialCNN`, 5-block conv trunk with gender and attribute heads; each block is `Conv(k×k) + BN + ReLU (+ MaxPool2×2, except the last)` and exposes its post-activation tensor for the downstream decoders. The single-model default is `channels = [32, 64, 128, 256, 256]`, `kernel_size = 3`; ensemble members override both via `ensemble.member_variants`.

**Decoders** — one `BlockDecoder` per conv block, upsampling frozen activations back to `(3, image_size, image_size)` through ×2 stages of `ConvTranspose2d(4×4, stride 2) + BN + ReLU`, halving channels per stage down to 16, ending in a 1×1 conv + Sigmoid.

### Diagrams

Three figures in [docs/diagrams/](docs/diagrams/) document the pipeline, each as a `.pdf` (vector source) and a `.png` (what renders below). All three describe the **128 px configuration** ([`configs/celeba_ood_128.yaml`](configs/celeba_ood_128.yaml)) — the run behind `outputs/celeba_ood/LASTOF_RESULTS`, which still carried the CutPaste pretext head. That head has since been removed from the CelebA path (see below), so the `head_cutpaste` box in the figures no longer corresponds to a CelebA member.

| Figure | Question it answers | Source of truth |
| --- | --- | --- |
| [classifier_pipeline.png](docs/diagrams/classifier_pipeline.png) | What does **one member** compute, from input tensor to three heads? | [`lrad/model.py`](lrad/model.py), [`lrad/train.py`](lrad/train.py) |
| [encoder_decoder.png](docs/diagrams/encoder_decoder.png) | Where do the **decoders tap** the frozen trunk, and what do they undo? | [`lrad/decoder.py`](lrad/decoder.py) |
| [ensemble_diversity_cubes.png](docs/diagrams/ensemble_diversity_cubes.png) | How do the **10 members** differ, and what stays invariant? | `ensemble.member_variants` |

Every number printed in them (channel widths, spatial sizes, loss weights, parameter counts, seeds) is the one the config resolves to. The same quantities are computed analytically in [`lrad/arch_diagram.py`](lrad/arch_diagram.py) and checked against `sum(p.numel() …)` on the real torch modules in `tests/test_arch_diagram.py`, which is what makes them auditable. A plainer SVG rendering of the same three views is generated straight from the config — use it when the config has moved and these figures have not been redrawn yet:

```bash
python scripts/generate_arch_svg.py --config configs/celeba_ood_128.yaml \
    --out-dir docs/diagrams
```

`run_ensemble.py` also drops a copy of the ensemble view into `<run>/ensemble/architectures.svg`, annotated with the parameter counts of the members that were actually instantiated.

#### 1. Classifier pipeline — one member, end to end

![Classifier pipeline](docs/diagrams/classifier_pipeline.png)

**How to read it.** One row per stage, top to bottom, in execution order. Each row shows the tensor *leaving* that stage as a cube whose **face is the spatial extent** (`H×W`) and whose **depth is the channel count** — so the cubes flatten into slabs as the trunk trades resolution for width. The label on each arrow is the operation that produced the next row. Colour is only a category: blue = tensor, teal = conv block, purple = task head, orange = self-supervised head. The three boxes at the bottom read the *same* pooled vector in parallel, which is why they hang off one bus rather than off each other.

The trunk is deliberately plain — five `Conv(k×k, pad k/2, bias=False) + BatchNorm2d + ReLU` blocks, `MaxPool2×2` after every block **except the last**, then `AdaptiveAvgPool2d(1×1)`. No dropout, no pretrained weights, no skip connections. At 128 px the spatial trace is:

```text
3×128×128 → 32×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 256×8×8 → GAP → h ∈ R^256
```

Pooling stops one block early on purpose: the last block keeps an 8×8 map so GAP averages over a real spatial extent rather than a single cell, and — more importantly — block 4 and block 5 share a resolution, which lets the deepest two decoders be architecturally identical and their reconstructions directly comparable.

Three linear heads read the 256-d pooled vector:

- **`head_gender` (256→2)** — softmax + CE on the `Male` attribute. Its job in this project is *not* accuracy (it sits near 0.63); it is to produce a logit vector whose **energy** and **entropy** move when the classifier hesitates. `ens_energy_gender` is a fusion input worth 0.713 AUROC on its own.
- **`head_attrs` (256→6)** — sigmoid + BCE, weighted `attr_loss_weight: 2.0`. Four of the six targets (`Arched_Eyebrows, Bushy_Eyebrows, Narrow_Eyes, Bags_Under_Eyes`) are **periocular**. This is the central design choice of the whole task: training only ever sees glasses-free faces, so the trunk must learn to read the eye region to satisfy these heads — and eyeglasses at test time occlude exactly that evidence. The resulting ID/OOD activation gap is what everything downstream measures.
- **`head_cutpaste` (256→2)** — the self-supervised pretext head, active only when `model.cutpaste_head: true`. **No CelebA config sets that flag any more**: the head is retained solely for MVTec AD ([`configs/mvtec.yaml`](configs/mvtec.yaml)), a cold-start benchmark with no labels at all, whose trunk trains on this pretext alone. On the historical `LASTOF_RESULTS` run it was on, and `P(altered | x)` scored 0.674 AUROC standalone; the numbers in the table below are from that run.

On CelebA the total loss is `CE(gender) + 2.0 · BCE(attrs)`, one `.backward()` per step.

The baseline member (`channels [32, 64, 128, 256, 256]`, `k=3`) is **981,288 parameters** — 979,232 in the trunk, 2,056 across the two heads (it was 981,802 with the CutPaste head on, as in `LASTOF_RESULTS`). The trunk is ~99.8 % of the model; the heads are almost free.

> The figure's gender-head caption still reads *"→ MSP, entropy, energy"*. **MSP was since removed** from [`lrad/evaluate.py`](lrad/evaluate.py): on two classes `1 − maxₑ pₑ` is a monotone function of `H(p)`, so it is rank-equivalent to the gender entropy and contributes no information the AUROC can see. Only `entropy` and `energy` are computed today.

#### 2. Encoder–decoder architecture — where the decoders tap

![Encoder–decoder architecture](docs/diagrams/encoder_decoder.png)

**How to read it.** The top row is the same forward pass as figure 1, but unrolled left to right with every sub-operation broken out (`Conv → BN+ReLU → Pool` instead of one fused "block"), so the shape at each intermediate point is explicit. Box **depth (Z) is the channel count**, box **face (X/Y) is the spatial extent**, and kernel/stride sit above each box — which is why the boxes start as thin wide sheets (`128×128×3`) and end as narrow deep bars (`8×8×256`). The dashed drops labelled `Dec L0 … Dec L3` are the **decoder taps**: the points where a block's post-pool activation is branched off. The bottom-right chain expands one of them — the `L4` bottleneck tap — into the actual `BlockDecoder` layers, because all five decoders share that structure and only differ in how many stages they need.

After classifier training the trunk is **frozen** (`eval()`, `requires_grad = False`) and one `BlockDecoder` is trained per conv block to reconstruct the original image from that block's activations alone. Each decoder is a stack of `n_up = log₂(image_size / block_size)` learnable ×2 stages — `ConvTranspose2d(4×4, stride 2, pad 1) + BN + ReLU` — halving channels at every stage down to a floor of 16, closed by `Conv1×1 → 3` + `Sigmoid` so outputs land in `[0,1]` like the inputs. The five decoders are optimised jointly by one Adam on `Σₖ MSE(dₖ(aₖ(x)), x)`.

The 4×4/stride-2/pad-1 kernel is chosen because it doubles H and W *exactly*, with no output-size ambiguity — which is what keeps block *k* meaning the same spatial scale across every ensemble member.

Two simplifications in the drawing, so the figure and the code can be read side by side: the expanded chain shows the four ×2 stages of `dec L4` (`8→16→32→64→128`) but folds the last stage's 16-channel output together with the closing `Conv1×1 → 3` into the single `Recon` box; and the taps are drawn on the pooled output of each block, which is what `FacialCNN` actually exposes.

| Decoder | Input activation | `n_up` | Channel path | Params |
| --- | --- | --- | --- | --- |
| `dec L0` | 32×64×64 | 1 | 32→16 | 8,275 |
| `dec L1` | 64×32×32 | 2 | 64→32→16 | 41,107 |
| `dec L2` | 128×16×16 | 3 | 128→64→32→16 | 172,307 |
| `dec L3` | 256×8×8 | 4 | 256→128→64→32→16 | 696,851 |
| `dec L4` | 256×8×8 | 4 | 256→128→64→32→16 | 696,851 |
| | | | **total** | **1,615,391** |

Note the shape of that cost: the two deepest decoders carry 86 % of the parameters, because they start from the widest activation *and* need the most upsampling stages. `dec L3` and `dec L4` are identical in structure — that is the block-4/block-5 shared resolution showing up again.

Reconstruction quality degrades monotonically with tap depth, and by a lot — final-epoch MSE for the baseline member runs `0.00043 → 0.00150 → 0.00327 → 0.00532 → 0.01124` from `L0` to `L4`, a factor of ~26. That is mechanical: successive max-pools destroy spatial detail, so a decoder starting from an 8×8 map can only ever paint a blurred face. **This error floor is why every pixel-space score plateaus around 0.62–0.70** — on a clean face, hair and background already contribute thousands of high-error pixels, while a pair of glasses touches a few hundred.

The reconstructions `f̂_k` are the substrate for every reconstruction-based score: the exact per-pixel `Risk = Bias + Variance` decomposition across the ensemble, the localized z-scored patch-max signals, and the per-block `recon_Lk` / `error_Lk` figures.

#### 3. Ensemble diversity — the ten members side by side

![DeepEnsemble member architectures](docs/diagrams/ensemble_diversity_cubes.png)

**How to read it.** One row per member, one column per conv block (`L0 … L4`, with the column header giving the spatial size that *every* member shares at that depth). **Cube height is the channel width** on a common scale (16 → 384) and **colour is the kernel size** (blue = 3×3, green = 5×5). So each row is a staircase, and the interesting content is the *shape* of the staircase: model 2 stays slim then flares at `L4` (24-48-96-192-**384**), model 7 is nearly flat (64-96-160-224-288), model 4 starts at just 16 channels but compensates with a 5×5 receptive field. Seed, kernel and parameter count are printed next to each member's name. Read down a column and every cube sits at the same spatial resolution — that alignment is the whole point of the figure.

The ensemble is *architecturally* diverse, not just seed-diverse. Ten members, seeded `base_seed + i` = 42…51, each with its own channel widths and conv kernel size from `ensemble.member_variants`:

| # | Seed | Channels | Kernel | Params | Intent |
| --- | --- | --- | --- | --- | --- |
| 1 | 42 | 32-64-128-256-256 | 3×3 | 981,802 | baseline |
| 2 | 43 | 24-48-96-192-384 | 3×3 | 887,266 | slim early, wide late |
| 3 | 44 | 48-96-192-256-256 | 3×3 | 1,245,114 | wide early |
| 4 | 45 | 16-48-96-192-320 | 5×5 | 2,136,954 | slim + large receptive field |
| 5 | 46 | 40-80-160-320-320 | 3×3 | 1,532,530 | 1.25× width |
| 6 | 47 | 32-64-128-256-256 | 5×5 | 2,720,042 | baseline, large RF |
| 7 | 48 | 64-96-160-224-288 | 3×3 | 1,102,986 | flat progression |
| 8 | 49 | 24-64-128-192-256 | 5×5 | 2,092,098 | narrow late, large RF |
| 9 | 50 | 48-64-96-192-384 | 3×3 | 919,098 | heavy head + tail |
| 10 | 51 | 32-48-112-224-336 | 5×5 | 2,688,874 | steep growth, large RF |

The **invariant** that makes this work: every variant keeps the same 5-block / pool-after-first-4 layout. `kernel_size` changes only the receptive field (padding preserves H and W), so all ten members share the spatial trace `128 → 64 → 32 → 16 → 8 → 8`. Block *k* therefore means the same scale in every member, the per-block decompositions stay aligned, and averaging reconstructions across members is well-defined. `resolve_member_configs` enforces the matching block count and raises if a variant disagrees.

Members span 0.89 M to 2.72 M parameters — a 3× spread. That heterogeneity is the point: independent inits alone produce members that fail on the *same* inputs, which collapses the variance term. Different widths and receptive fields make members extrapolate differently off-distribution, which is precisely what the epistemic/variance signal needs.

#### What the diagrams do not cover

The three figures document the *models*. They do not document the **scoring stack**, which is where the headline number comes from — [`lrad/fusion.py`](lrad/fusion.py) calls rank fusion "the headline detector". On the `LASTOF_RESULTS` run:

| Signal | AUROC |
| --- | --- |
| `p95` bias (the original reconstruction score) | 0.623 |
| `zscore_risk_aggregated` (localized, [`lrad/localized.py`](lrad/localized.py)) | 0.703 |
| `ens_energy_gender` | 0.713 |
| `cutpaste_prob` | 0.674 |
| `locfre_b1` / `locfre_b2` / `locfre_b3` ([`lrad/feature_error.py`](lrad/feature_error.py)) | 0.790 / 0.777 / 0.795 |
| **`fused_rank`** (label-free rank fusion of all six) | **0.804** |
| **`fused_supervised`** (logistic fit on a 50 % OOD calibration slice) | **0.864** |

The gap between the raw bias term (0.623) and the fused detector (0.804 label-free) is the substance of the project, and no current diagram shows it. [`docs/Documentation.md`](docs/Documentation.md) is the full report on that run: every formula, every figure it produces, worked numeric examples, and a critical reading of the anomalies (French).

## MVTec AD — the PatchCore comparison

The same decomposition runs as a **cold-start industrial anomaly detector** on
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), to be
benchmarked head-to-head against
[PatchCore](https://arxiv.org/abs/2106.08265) (Roth et al., CVPR 2022).

**What changes, and why.** MVTec's training split holds nominal images only and
carries **no labels of any kind** — the gender and attribute heads have nothing
to learn from. So the trunk is trained **purely self-supervised**, on the 3-way
CutPaste pretext task (intact / box patch / scar; Li et al., CVPR 2021), and the
supervised heads are switched off (`model.gender_head: false`,
`attrs_head: false`). Everything downstream is unchanged: the trunk is frozen,
one `BlockDecoder` is fitted per conv block, and the anomaly score is the same
ensemble Bias term `(x − f̄_k)²`. Members still differ by seed and architecture.

The protocol mirrors PatchCore's: one independent ensemble **per category**,
trained on that category's nominal images only, 256 → 224 centre-crop
preprocessing, and the 15-category mean reported at the end.

Because MVTec ships ground-truth masks, all three of PatchCore's metrics are
computed — `lrad/pixel_metrics.py`:

| Metric | What it measures | PatchCore−10% |
| --- | --- | --- |
| image AUROC | detection: is this image defective? | 99.0 |
| pixel AUROC | localization, pooled over all pixels | 98.1 |
| PRO | localization, **equal weight per defect region** | 93.5 |

PRO is the honest one. Pixel AUROC is dominated by large defects — a model that
finds every big blob and misses every hairline scratch still scores well, because
the big blob supplies thousands of positive pixels and the scratch a few dozen.
PRO averages the per-region recovery with weight 1 per connected component
whatever its area, then integrates against FPR up to 0.30 and normalizes. Note
its random baseline is **0.15**, not 0.5: a chance localizer traces the diagonal,
and `∫₀^0.3 x dx / 0.3 = 0.15`.

```bash
# 1. fetch the dataset (~4.9 GB; CC BY-NC-SA 4.0, non-commercial research only)
python scripts/download_mvtec.py --root ./data
python scripts/download_mvtec.py --root ./data --verify-only   # check a copy

# 2. run the pilot categories from the config (bottle, carpet, screw, transistor)
python scripts/run_mvtec.py --config configs/mvtec.yaml

# 3. all 15 categories
python scripts/run_mvtec.py --config configs/mvtec.yaml --categories all

# 4. re-score already-trained members
python scripts/run_mvtec.py --config configs/mvtec.yaml \
    --output-dir outputs/mvtec/my_run --eval-only
```

### Outputs

```text
<run>/
├── results.md                     # per-category table vs PatchCore
├── summary.json                   # every metric, machine-readable
├── vs_patchcore.png               # grouped bars, all 3 metrics per category
└── <category>/
    ├── summary.json
    ├── model_<i>/
    │   ├── weights/{model,decoders}.pt
    │   ├── history.json
    │   └── plots/{batch_loss,decoder_loss}.png
    └── plots/
        ├── curves.png             # image ROC + PRO curve (shaded integral)
        ├── auroc_per_block.png    # detection AUROC per block, R/B/V
        ├── score_hists.png        # nominal vs defective score distributions
        ├── architecture_effect.png    # member architecture vs its AUROC
        ├── instances_best.png     # Input | Anomaly map | GT | Overlay
        └── instances_worst.png    # the lowest-scoring defects — the failures
```

`instances_worst.png` is deliberately written next to `instances_best.png`:
showing only the top detections makes any category look solved, when the
low-scoring tail is precisely what the PRO number is reacting to. The overlay
column carries the ground-truth contour on top of the heatmap, so "did the hot
region land inside the real defect" is answerable at a glance.

`results.md` and `summary.json` are rewritten after **every** category, so a job
killed at the walltime still leaves a complete table for whatever finished. A
category that fails (missing download, OOM) is logged and skipped rather than
taking the whole run down, and a plotting failure never costs a category its
numbers.

### Schedule: steps, not epochs

MVTec categories hold 60 (toothbrush) to 391 (hazelnut) training images. At batch
32 a fixed epoch count therefore buys 2–13 optimizer steps per epoch — a **6.5×
spread** in how much training each category receives, which alone would make the
per-category table incomparable. `configs/mvtec.yaml` sets `training.steps` and
`training.decoders.steps` instead, and the runner converts each into a per-category
epoch count.

The absolute budget matters as much as the fairness. At a few hundred decoder
steps the decoders never individuate: they converge to the dataset mean, and the
bias map degenerates into `|x − mean image|`, which correlates ~0.85 with plain
image brightness and scores **below 0.5** pixel AUROC — MVTec defects are often
darker, and therefore *easier* to reconstruct, than nominal texture.
`lrad/pixel_metrics.py` warns when it detects that collapse (maps varying far
less between images than within one) instead of letting it pass as a result.

Budget ~30–45 min per member at 224 px on an A40 with the default 4,000 + 8,000
steps: the 4-category pilot at 4 members is ~8–12 h, all 15 categories ~30–45 h.
Run the pilot first and read the real per-member time out of the log before
committing to the full sweep.

## Features

- Exact pixelwise `Risk = Bias + Variance` decomposition, verified at runtime
- Architecturally diverse deep ensemble: per-member channel widths + kernel size, independent weight init + SGD shuffle order (no regularizer)
- Predictive-uncertainty decomposition on classifier heads (total = aleatoric + epistemic / MI)
- Per-instance figures: one figure per member (recon + error) and a bias / mean / min + overlay summary, per test face
- Top-N OOD eyeglasses faces ranked by eye-region bias (strength × concentration)
- Configurable OOD attribute set (default: eyeglasses only)
- Second benchmark on MVTec AD: per-category cold-start runs, pretext-only trunk, image AUROC + pixel AUROC + PRO against PatchCore
- Epoch-variability study: traces inter-model variability σ(e) from per-epoch decoder checkpoints
- Single YAML config shared by both single-model and ensemble runners; dotted `key=value` CLI overrides (lists supported, e.g. `model.channels=[...]`)

## Tech stack

| Component | Libraries |
| --- | --- |
| Runtime | Python 3.10+, PyTorch 2.4, torchvision |
| Numerics | NumPy < 2.0, SciPy (overlay smoothing), scikit-learn |
| Visualisation | matplotlib (300 dpi export) |
| Config | PyYAML |

## Prerequisites

- Python ≥ 3.10
- CelebA dataset placed at `data/celeba/` (official source, or set `dataset.download: true` in the config to let torchvision fetch it)
- For the MVTec task: MVTec AD at `data/mvtec_anomaly_detection/` via `python scripts/download_mvtec.py`
- CUDA optional but recommended for ensemble runs

## Installation

```bash
# editable install, loose deps (CPU or pre-installed torch)
pip install -e .

# pinned CUDA 11.8 stack (Grid'5000 / Ampere GPUs)
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

## Quick start

```bash
# 1. ensemble + bias/variance decomposition (primary pipeline)
python scripts/run_ensemble.py --config configs/celeba_ood.yaml

# 2. single model (classifier + decoders + confidence/fusion scores)
python scripts/run_celeba.py --config configs/celeba_ood.yaml

# 3. re-run decomposition on already-trained models
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --output-dir outputs/celeba_ood/my_run --eval-only

# 4. run tests
pytest
```

Both runners accept `--override key=value …` for dotted config overrides and `--no-plots`.

## Grid'5000 batch runs (Nancy)

From the Nancy frontend, in `~/lrad` (OAR directives — cluster, GPU, walltime — are embedded in the scripts):

```bash
# main run: 10-model ensemble, one architecture per member
./scripts/oar_run_ensemble.sh

# track / cancel
oarstat -u $USER
tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
oardel <jobid>
```

The job requests **1 GPU on `graffiti` (RTX 2080 Ti, 11 GiB) in the production/Abaca queue, walltime 48 h**. graffiti is the largest GPU pool at Nancy (12 nodes × 4 GPUs), so allocation is usually fast; the run needs < 10 GiB of VRAM, and ~3 h/model × 10 models fits comfortably in 48 h (a 2080 Ti is ~2–3× slower than the A100 this used to run on). The walltime is sized generously on purpose: an OAR job is cut at its walltime even mid-epoch and the ensemble run does not checkpoint. Note the production queue does **not** allow advance reservations (`oarsub -r`) — submission only. Fallback clusters (edit the `#OAR -p` line): `gruss` (2× A40 45 GiB) or `grue` (4× T4 15 GiB).

## Usage examples

**Override ensemble size and learning rate:**

```bash
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --override ensemble.size=3 training.lr=5e-4
```

**Trace inter-model variability vs training epochs** (requires `training.save_every_epoch: true`):

```bash
python scripts/epoch_variability_study.py \
    --config configs/celeba_ood.yaml \
    --run-dir outputs/celeba_ood/my_run \
    --all-blocks --include-ood
```

## Configuration

`configs/celeba_ood.yaml` is the single source of truth for both runners.

| Section | Key parameters |
| --- | --- |
| `dataset` | `root`, `image_size`, `batch_size`, `train_ratio`, `val_ratio`, `ood_attrs` (attribute name or list defining OOD; default `[Eyeglasses]`) |
| `model` | `channels` (one int per conv block), `kernel_size` (3/5), `n_attrs`, `n_gender` — the single-model architecture; ensemble members use `ensemble.member_variants` |
| `training` | `epochs` (20), `lr`, `attr_loss_weight`, `save_every_epoch`; nested `decoders: {epochs (25), lr}` |
| `evaluation` | `n_viz_in_samples`, `n_viz_ood_samples`, `viz_seed`, `score_comparison_block`, `score_comparison_k`, `n_instances_in`, `n_instances_ood`, `instance_block`, `overlay_sigma`, `overlay_power` (display-only knobs of the bias overlay), `n_top_ood` (size of the top OOD eyeglasses ranking) |
| `ensemble` | `size`, `base_seed` (model `i` uses `base_seed + i`), `agg` (`mean` / `max` / `p95`), `member_variants` (per-member `{channels, kernel_size}`, cycled if `size` exceeds the list) |

## Project structure

```text
lrad/
├── lrad/                          # library (flat layout)
│   ├── dataset.py                 # CelebA loaders; configurable accessory OOD split
│   ├── mvtec.py                   # MVTec AD: per-category loaders + ground-truth masks
│   ├── config.py                  # YAML load + dotted `key=value` CLI overrides
│   ├── model.py                   # FacialCNN: conv trunk + optional gender/attrs/cutpaste heads
│   ├── decoder.py                 # per-block ConvTranspose2d decoders
│   ├── cutpaste.py                # CutPaste pretext augmentation (patch/scar, 2- or 3-way)
│   ├── pixel_metrics.py           # pixel AUROC + PRO for anomaly localization
│   ├── train.py                   # classifier + decoder training loops
│   ├── evaluate.py                # accuracy + classifier-confidence OOD AUROC
│   ├── anomaly_score.py           # per-pixel error + pixel→scalar reductions
│   ├── ensemble.py                # bias/variance decomposition + eye-region bias
│   ├── localized.py               # per-pixel z-score + multi-scale patch-max
│   ├── feature_error.py           # localized feature-reconstruction error (locfre)
│   ├── fusion.py                  # rank / supervised fusion — headline detector
│   ├── arch_diagram.py            # member-config resolution, param counts, SVG renderers
│   ├── plots.py                   # all figures (300 dpi, paper styling)
│   └── utils.py                   # device, seeding, logging
├── configs/
│   ├── celeba_ood.yaml            # base config (64 px)
│   ├── celeba_ood_128.yaml        # 128 px, supervised heads only
│   └── mvtec.yaml                 # MVTec AD, per-category, pretext-only trunk
├── docs/
│   ├── Documentation.md           # full report on the LASTOF_RESULTS run (FR)
│   └── diagrams/                  # the three figures above, .pdf source + .png
├── scripts/
│   ├── run_celeba.py              # single-model pipeline
│   ├── run_ensemble.py            # ensemble + decomposition + instance figures
│   ├── run_mvtec.py               # MVTec per-category ensembles + PatchCore table
│   ├── download_mvtec.py          # fetch/extract/verify MVTec AD
│   ├── run_localized.py           # localized z-score / patch-max scoring
│   ├── run_fused.py               # fused (locfre + epistemic + energy) AUROC
│   ├── run_gridsearch.py          # CutPaste hyperparameter search
│   ├── run_generalization.py      # bias overlay on non-CelebA photos (sanity probe)
│   ├── generate_arch_svg.py       # regenerate the three diagrams as SVG
│   ├── epoch_variability_study.py # σ(e) variability vs decoder epochs
│   └── oar_run_*.sh               # Grid'5000 OAR jobs (ensemble, fused, gridsearch, …)
└── tests/                         # pytest: anomaly score, decomposition,
                                   #         decoders, training, plots, checkpointing
```

Outputs are gitignored. An ensemble run writes per-model results under `model_<i>/` (weights, history, plots) and the decomposition under `ensemble/` (AUROC table, identity residual, all heatmap figures, `top_ood_glasses.png` + `.json`), with the per-instance figures under `ensemble/plots/instances_{in,ood}/<ID|OOD>_XX/` (one folder per face: `model_01.png` … `model_10.png` + `summary.png`).

## Contributing

Open an issue or pull request. Run `pytest` and confirm no regressions before submitting.

## License

MIT — see [LICENSE](LICENSE).
