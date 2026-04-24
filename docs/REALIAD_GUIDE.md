# LRAD on Real-IAD — End-to-End Guide

Train, evaluate, and visualise the LRAD (Layer-wise Reconstruction Anomaly
Detection) model on the Real-IAD industrial anomaly dataset, targeted at
running on the **Grid'5000 cluster at LORIA (Nancy)** via SLURM.

This guide covers:

1. [Setup](#1-setup)
2. [Dataset preparation](#2-dataset-preparation)
3. [Model & method](#3-model--method)
4. [Training & evaluation](#4-training--evaluation)
5. [Outputs & interpretation](#5-outputs--interpretation)
6. [Grid'5000 / SLURM workflow](#6-grid5000--slurm-workflow)
7. [Ablations & extensions](#7-ablations--extensions)

---

## 1. Setup

### 1.1 Clone and install

```bash
git clone <repo-url> lrad
cd lrad

# Local workstation (Python 3.10+):
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
```

On Grid'5000 Nancy the venv is created (and re-used) automatically by
`scripts/slurm/env_setup.sh`; you do **not** need to run `pip install`
manually on a compute node.

### 1.2 Dependencies

- Python ≥ 3.10, PyTorch ≥ 2.0, torchvision ≥ 0.15
- numpy, matplotlib, scikit-learn, pyyaml, Pillow

### 1.3 Repository layout

```
lrad/
├── lrad/                           # Library
│   ├── models/
│   │   ├── classifier.py           # shallow CNN / MLP
│   │   ├── decoder.py              # shallow decoders
│   │   ├── deep_cnn.py             # DeepCNNClassifier (ResNet-style)
│   │   ├── deep_decoder.py         # DeepCNNDecoder (multi-step upsample)
│   │   └── lrad_model.py           # LRADModel glue
│   ├── data/
│   │   ├── datasets.py             # MNIST / CIFAR loaders
│   │   └── realiad.py              # Real-IAD manifest-driven loader
│   ├── engine/
│   │   ├── trainer.py              # baseline train loops
│   │   ├── realiad_trainer.py      # pretext-aware loops with validation
│   │   ├── evaluator.py            # baseline AUROC
│   │   └── realiad_eval.py         # AUROC, PR-AUC, F1, pixel-AUROC
│   └── visualization/
│       ├── heatmaps.py             # heatmap grids, score plots, ROC
│       └── realiad_plots.py        # training curves, PR, scatter, etc.
├── configs/
│   └── realiad.yaml                # the one config you edit
├── scripts/
│   ├── prepare_realiad.py          # zips -> manifest.csv
│   ├── analyze_realiad.py          # paper-style dataset figures
│   ├── verify_realiad.py           # integrity scan
│   ├── run_realiad.py              # train + eval + plots
│   ├── ablation_depth.py           # resnet10/18/34 sweep
│   └── slurm/                      # SLURM batch scripts
└── docs/
    └── REALIAD_GUIDE.md            # this file
```

---

## 2. Dataset preparation

Real-IAD ships as 30 category zip archives (one per product).
You do **not** need to unpack them — the loader reads straight from zip.

### 2.1 Expected layout

```
$HOME/Real-IAD/realiad_1024/
├── audiojack.zip
├── bottle_cap.zip
├── ...
└── zipper.zip
```

Inside each zip, files follow the canonical Real-IAD convention:

```
audiojack/OK/S0001/audiojack_0001_OK_C1_<ts>.jpg
audiojack/NG/BX/S0001/audiojack__0001_NG_BX_C1_<ts>.jpg
audiojack/NG/BX/S0001/audiojack__0001_NG_BX_C1_<ts>.png    (mask)
```

### 2.2 Build the manifest

The single source of truth for the loader is `manifest.csv`, generated
once from the zips:

```bash
python scripts/prepare_realiad.py \
    --data-root $HOME/Real-IAD/realiad_1024 \
    --out-dir   outputs/realiad/manifest
```

This writes:

| file | purpose |
|------|---------|
| `manifest.csv`   | one row per (category, sample, view) with its mask |
| `metadata.json`  | per-category OK/NG counts + defect vocabulary |
| `warnings.txt`   | missing views or orphan masks |

### 2.3 Sanity-check and explore

```bash
# Integrity scan (reads 10% of JPEGs/PNGs, flags corruption)
python scripts/verify_realiad.py \
    --manifest outputs/realiad/manifest/manifest.csv \
    --out-dir  outputs/realiad/verify --sample-frac 0.1

# Paper-style dataset figures (class distribution, sample grid,
# defect gallery, per-category mosaics, etc.)
python scripts/analyze_realiad.py \
    --manifest outputs/realiad/manifest/manifest.csv \
    --out-dir  outputs/realiad/analysis
```

`outputs/realiad/analysis/` ends up with:

- `counts_ok_ng.png`, `counts_grouped.png`, `dataset_card.md`
  → class distribution tables and paper-style Fig 2(a)/(d) plots
- `multiview_showcase.png`, `defect_gallery.png`, `samples/<cat>.png`
  → qualitative sample grids (paper Fig 3 / Fig 4 / per-category mosaics)
- `defect_area_hist.png`, `defect_aspect_ratio_hist.png`
  → statistical views of defect size/shape
- `views_coverage.png`, `pixel_stats.png`, `stats.json`

### 2.4 Preprocessing performed at load-time

`lrad/data/realiad.py`:

- **Resize** to `image_size` (default 224, square) — bilinear for images,
  nearest for masks so binary labels are preserved.
- **ToTensor** → scales pixels to `[0, 1]`. No channel-wise
  normalisation is applied; the decoders end with Sigmoid so they must
  reconstruct a `[0, 1]` signal.
- **Splits** are deterministic by `hash(category, sample_id)` with the
  one-class constraint: every view of a defective part goes to
  `test_anomaly`; OK parts are partitioned into `train` / `val` /
  `test_normal` with ratios `0.80 / 0.10 / 0.10` (configurable).

---

## 3. Model & method

### 3.1 LRAD in one paragraph

LRAD first trains a classifier on **normal** data (or a self-supervised
pretext over normal images), then **freezes** it and trains a bank of
decoders — one per classifier stage — to reconstruct the input image
from that stage's activations. At test time, a region of an image that
the decoders cannot reconstruct from their normality-biased features
exhibits a large pixel-wise reconstruction error. The per-stage error
maps are fused (mean, max, or a reconstruction-weighted average) into a
single anomaly map; its maximum is the image-level anomaly score.

### 3.2 Mathematical formulation

Let `f_θ` be the frozen classifier with stage outputs
`a_k = f_θ^(k)(x) ∈ ℝ^{C_k × H_k × W_k}` for stages `k = 1..K`, and let
`g_{φ_k}` be a decoder mapping `a_k` back to image space.

**Training (phase 2)** minimises per-stage mean squared error on the
normal-only training set `𝒟_train`:

```
min_{φ_k}   E_{x ~ 𝒟_train}  ‖ g_{φ_k}(f_θ^(k)(x)) − x ‖²₂
```

**Inference** produces per-stage error maps

```
E_k(x, u, v) = 1/C · Σ_c ( g_{φ_k}(a_k)[c, u, v] − x[c, u, v] )²
```

and a fused heatmap

```
H(x) = Fuse_k  U(E_k)         (U = bilinear upsample to H×W)
```

where `Fuse` is mean / max / inverse-MSE-weighted mean.  The
image-level score is `s(x) = max_{u,v} H(x)[u, v]`.

### 3.3 Why it detects anomalies

- The classifier has never been exposed to NG patterns, so its filters
  at every depth are tuned to the statistics of the normal class.
- Decoders learn the inverse map **only on the manifold of normal
  inputs**. A defect that sits off that manifold produces unusual
  activations, and the decoder — never trained on such states —
  reconstructs a "normalised" version of the region, not the defect.
- The residual between input and reconstruction is therefore large
  exactly where the defect lies.

### 3.4 Why a deeper CNN

The shallow `CNNClassifier` halves resolution on every block (strided
conv), so a 4-block stack on 224×224 images collapses the spatial
dimension to 14×14 by stage 4 without ever learning local texture at the
original resolution. For Real-IAD defects — pit/scratch/abrasion sizes
range from ~0.1% to 2% of the image — this is harmful.

`DeepCNNClassifier` instead uses a ResNet-18 layout:

| site   | channels | spatial (224 input) | decoder receives |
|--------|----------|---------------------|-------------------|
| stage1 | 64  | 56×56 | local texture, fine defects |
| stage2 | 128 | 28×28 | mid-scale structure |
| stage3 | 256 | 14×14 | part-level layout |
| stage4 | 512 | 7×7   | global object appearance |

**Architectural choices and the inductive biases they add**

- **Residual blocks (skip connections)** — keep gradients healthy during
  the weakly-supervised rotation pretext task (phase 1).
- **7×7 + MaxPool stem** — large receptive field at the very first
  activation site matches the visual scale of medium-size defects.
- **Four stages, doubling width** — lets each decoder specialise: stage 1
  picks up sub-pixel texture anomalies, stage 4 catches mis-shapen
  parts.
- **BatchNorm + ReLU + Kaiming init** — standard, makes training at
  224×224 stable on a single GPU with batch 32.

Choosing among `resnet10 / resnet18 / resnet34` trades off compute for
depth; `scripts/ablation_depth.py` sweeps all three and collates metrics.

### 3.5 Phase 1: handling the one-class problem

A standard supervised classifier is degenerate when the only label
available is "normal". `train_classifier_pretext` offers two paths:

- **Rotation pretext** (default, single-category runs). Each training
  batch is augmented with its 4 canonical rotations and the classifier
  predicts the rotation index. This supplies meaningful gradients
  without any anomaly labels, and the resulting features still
  specialise to normal-image statistics.
- **Category classification** (multi-category runs). If `dataset.
  categories` lists several products, the pretext becomes
  `predict-the-product`; the label comes from the manifest directly.

---

## 4. Training & evaluation

### 4.1 One command

```bash
python scripts/run_realiad.py --config configs/realiad.yaml
```

which performs, in order:

1. seed every RNG (Python, NumPy, torch, cuDNN deterministic)
2. build the four data loaders
3. build `DeepCNNClassifier` and `LRADModel` (4 decoders)
4. **Phase 1** — classifier pretext training (cosine LR schedule, AdamW)
5. **Phase 2** — decoder training (per-decoder AdamW + cosine schedule,
   per-epoch train **and** val MSE, best-epoch checkpoint)
6. **Evaluation** on `test_normal` + `test_anomaly`
7. All visualisations
8. `summary.json` with the full metric dump + resolved config

### 4.2 Config cheat sheet (`configs/realiad.yaml`)

| key | default | meaning |
|-----|---------|---------|
| `dataset.categories` | `[audiojack]` | null = all 30 categories |
| `dataset.image_size` | `224` | square resize target |
| `dataset.batch_size` | `32` | 32 fits 16 GB GPUs at 224×224 |
| `dataset.train_ratio` | `0.80` | OK share going to train |
| `dataset.val_ratio` | `0.10` | OK share going to validation |
| `dataset.load_masks` | `true` | yields masks on test_anomaly |
| `model.preset` | `resnet18` | resnet10 / resnet18 / resnet34 / resnet18_slim |
| `training.classifier.pretext` | `rotation` | rotation / category |
| `training.classifier.epochs` | `30` | |
| `training.classifier.lr` | `1e-3` | AdamW, cosine to 0 |
| `training.classifier.weight_decay` | `1e-4` | |
| `training.decoders.epochs` | `60` | |
| `training.decoders.lr` | `1e-3` | |
| `training.decoders.weight_decay` | `1e-5` | |
| `evaluation.fusion` | `weighted` | mean / max / weighted |
| `evaluation.n_display` | `10` | qualitative grid width |

**Ad-hoc overrides** without editing the YAML:

```bash
python scripts/run_realiad.py --config configs/realiad.yaml \
    --override model.preset=resnet34 \
               training.decoders.epochs=80 \
               dataset.batch_size=16
```

### 4.3 Evaluation metrics

Image-level (normal vs. anomaly score distributions):

- **AUROC** — ranking quality, threshold-free.
- **PR-AUC** (average precision) — more informative than AUROC when
  positives are rare (common in anomaly detection).
- **Best F1** — precision-recall trade-off at the operating point
  `argmax_τ F1(τ)`. Precision, recall, and τ are reported alongside.

Pixel-level (when masks are provided):

- **Pixel-AUROC** and **Pixel-PR-AUC** on flattened (heatmap, mask)
  pairs. For very large test sets the pixels are subsampled with
  stratification to stay within `max_pixels=2_000_000`.

---

## 5. Outputs & interpretation

Every run produces the following tree under `experiment.output_dir`:

```
outputs/realiad/run/
├── config.resolved.yaml        # exact config used (incl. CLI overrides)
├── summary.json                # every metric + param count
├── cls_history.json            # per-epoch classifier train/val loss+acc
├── dec_history.json            # per-decoder per-epoch train/val MSE
├── weights/
│   ├── classifier.pt
│   ├── decoder_0.pt … decoder_3.pt
│   └── decoders_best.pt        # best val-sum snapshot
├── logs/
│   └── lrad.log                # rotating console log (file copy)
└── plots/
    ├── class_distribution.png    # OK/NG stacked bar chart
    ├── training_curves.png       # classifier + per-decoder loss/acc
    ├── score_distribution.png    # histogram+KDE, threshold line, stats
    ├── roc_curves.png            # ROC with Youden-J operating point
    ├── pr_curves.png             # PR curve with AP + best F1
    ├── heatmaps_normal.png       # input / recon / error / fused / overlay
    ├── heatmaps_anomaly.png      # idem for bracketed anomaly samples
    ├── reconstruction_gallery.png# compact per-sample input/recon/heatmap
    └── feature_scatter.png       # 2-D PCA of last-stage activations
```

### 5.1 How to read the anomaly score

- The **image-level score** is the **max pixel value of the fused
  heatmap**. Images with score above `image.threshold` from
  `summary.json` are flagged anomalous (best-F1 operating point).
- **Per-layer layouts** in `heatmaps_*.png` let you see *which scale*
  fires: a defect lit up only in `Error L0` is a texture-scale defect;
  lit up in `Error L3` is a structural defect.
- **`score_distribution.png`** overlays normal and anomaly score
  histograms; large overlap means the model struggles, clean separation
  means the task is solved.
- **`feature_scatter.png`** projects the deepest-stage features to 2-D;
  a visibly separate NG cluster suggests the backbone is already doing
  a lot of the work (sometimes pooled features alone suffice).

### 5.2 Interpreting the training curves

- Classifier plot — the pretext task should converge to >90% accuracy
  on rotation; lower values suggest the backbone isn't learning.
- Decoder plot — **log-scale MSE** per stage. Stage 0 should drop the
  fastest (most spatial information), stage 3 the slowest (bottleneck).
  If val MSE diverges from train MSE, lower decoder lr or add more
  `weight_decay`.

### 5.3 summary.json shape

```json
{
  "experiment": "...",
  "seed": 42,
  "categories": ["audiojack"],
  "n_train": 2000, "n_val": 250, "n_test_normal": 250, "n_test_anomaly": 680,
  "classifier": {
    "architecture": "DeepCNNClassifier",
    "pretext": "rotation",
    "stages": [64, 128, 256, 512],
    "blocks_per_stage": [2, 2, 2, 2],
    "parameters": 11200000
  },
  "decoders": {"count": 4, "parameters_total": 8_000_000},
  "metrics": {
    "image": {"auroc": 0.92, "pr_auc": 0.87, "f1": 0.82,
              "precision": 0.79, "recall": 0.85, "threshold": 0.041},
    "pixel": {"auroc": 0.94, "pr_auc": 0.48, "n_pixels": 1999500}
  }
}
```

---

## 6. Grid'5000 / SLURM workflow

Grid'5000's Nancy site at LORIA exposes SLURM with `--gres=gpu:1` on
the GPU clusters (e.g. Graffiti, Grele, Grue). All batch scripts live
under `scripts/slurm/`.

> **Note.** Grid'5000 historically uses OAR as its default scheduler. If
> your reserved cluster exposes only OAR, run `oarsub -l ... -S script.sh`;
> the SLURM scripts below document resource requirements via `#SBATCH`
> directives that map cleanly to OAR constraints.

### 6.1 Environment bootstrap (`scripts/slurm/env_setup.sh`)

Sourced by every batch script. It:

1. clears the module environment and tries to `module load cuda`
   (12.1 → 11.8 → 11.7 fallback chain),
2. creates `$HOME/lrad-venv` on first call (`python -m venv` +
   `pip install -e .`), activates it on subsequent calls,
3. sets `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for
   deterministic cuBLAS,
4. points `TORCH_HOME` at `/tmp/$USER/torch_cache` to keep the 10 GB
   `/home` quota safe.

Override defaults via environment variables:

```bash
export LRAD_ROOT=$HOME/lrad
export LRAD_VENV=$HOME/envs/lrad
export LRAD_DATA_ROOT=$HOME/Real-IAD/realiad_1024
```

### 6.2 Typical submission sequence

```bash
# 1) CPU job: build manifest + analysis
sbatch scripts/slurm/prepare_realiad.sbatch

# 2) GPU job: train + eval
sbatch scripts/slurm/train_realiad.sbatch                         # default config
sbatch --export=ALL,LRAD_CONFIG=configs/realiad.yaml \
       scripts/slurm/train_realiad.sbatch \
       --override model.preset=resnet34

# 3) GPU job: re-evaluate without retraining
sbatch --export=ALL,LRAD_CONFIG=configs/realiad.yaml,LRAD_OUTPUT_DIR=outputs/realiad/run \
       scripts/slurm/eval_realiad.sbatch

# 4) GPU job: depth ablation (runs three presets sequentially)
sbatch scripts/slurm/ablation_depth.sbatch
```

### 6.3 Resource sizing

| job | time | CPUs | RAM | GPUs | notes |
|-----|------|------|-----|------|-------|
| `prepare_realiad` | 30 min | 4 | 8 GB | 0 | manifest + analysis figures |
| `train_realiad` (1 category, resnet18) | ~2h on one V100/A100 | 8 | 32 GB | 1 | 30 cls epochs + 60 dec epochs |
| `train_realiad` (30 cats, resnet18) | 8–12h | 8 | 32 GB | 1 | shrink `batch_size` if OOM |
| `ablation_depth` | ~6h per preset | 8 | 32 GB | 1 | use 3 separate jobs to parallelise |

### 6.4 Reproducibility knobs

- `experiment.seed` seeds Python / NumPy / torch / cuDNN.
- `torch.backends.cudnn.deterministic = True` (in `seed_everything`).
- `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the env
  bootstrap.
- The resolved config is snapshotted to `config.resolved.yaml` alongside
  the weights so a rerun with the same `--output-dir` reproduces bit-
  identical splits (the manifest-hash split is deterministic too).

---

## 7. Ablations & extensions

### 7.1 Depth ablation

`scripts/ablation_depth.py` fires `run_realiad.py` three times with
different `model.preset` values and writes:

- `outputs/realiad/ablation_depth/<preset>/summary.json`
- `outputs/realiad/ablation_depth/depth_comparison.csv`
- `outputs/realiad/ablation_depth/depth_comparison.png` (bar chart of
  image AUROC / PR-AUC / F1 per preset)

### 7.2 Fusion ablation

Toggle `evaluation.fusion` between `mean`, `max`, `weighted` and compare
the resulting `summary.json` files; the `--eval-only` mode makes this
instant once weights exist.

### 7.3 Suggested further enhancements

1. **Feature-bank baselines** — contrast LRAD against PaDiM /
   PatchCore by adding a `scripts/run_patchcore.py`; Real-IAD's
   multi-view structure makes the bank richer.
2. **Pretraining the classifier with DINO v2 / ImageNet** — swap the
   rotation pretext for a frozen foundation backbone and train only the
   decoders (turn `training.classifier.epochs = 0`).
3. **Per-category thresholds** — currently the operating threshold is
   global; groupby category and fit per-category thresholds for higher
   F1 when categories have very different background contrasts.
4. **Multi-view fusion** — aggregate fused heatmaps across the 5 Real-IAD
   camera views (C1–C5) of the same part to boost recall for defects
   visible on only some angles.
5. **Uncertainty-aware decoders** — `lrad/models/lrad_uq.py` and
   `lrad/engine/uq_engine.py` provide an MC-dropout decoder you can
   plug in; the anomaly score becomes residual ± predictive σ.

### 7.4 Troubleshooting

| symptom | likely cause | fix |
|---------|--------------|-----|
| GPU OOM at 224² | batch too large | `--override dataset.batch_size=16` or use `model.preset=resnet18_slim` |
| val MSE diverges | decoder overfits | raise `training.decoders.weight_decay` to `1e-4` |
| AUROC near 0.5 | rotation pretext failed | check `training_curves.png` — accuracy < 40% means labels got shuffled or batch is too small |
| pixel-AUROC is nan | masks all zero in batch | ensure `dataset.load_masks: true` and that the chosen category actually ships masks |
| SLURM job can't find CUDA | module load failed silently | run `module spider cuda` on the cluster to see exact names; set `LRAD_CUDA_MODULE=cuda/X.Y` and adapt `env_setup.sh` |
