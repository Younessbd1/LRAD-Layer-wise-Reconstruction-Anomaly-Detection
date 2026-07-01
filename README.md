# LRAD — Layer-wise Reconstruction Anomaly Detection

OOD detection on CelebA faces via deep-ensemble reconstruction error decomposed into bias (anomaly) and variance (epistemic uncertainty).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

## What it does

A from-scratch convolutional classifier is trained on accessory-free CelebA faces; lightweight decoders attached to each conv block learn to invert that block's frozen activations back to a `(3, 64, 64)` reconstruction. Training `M` such models independently forms a **plain deep ensemble** — diversity comes only from the per-seed weight init and the SGD shuffle order — whose per-pixel reconstruction risk decomposes exactly as `Risk = Bias + Variance`. The **bias** term — the irreducible error of the consensus reconstruction — is the anomaly score; faces wearing **eyeglasses or a hat** are held out as OOD (the attribute set is configurable) and score higher.

## What was done in this run

- **Removed Randomised MAP Sampling (anchored ensembling) and weight decay** from the whole pipeline: no `anchor_lambda`, no L2-to-anchor penalty in `train_decoders`, no `weight_decay` on either optimizer. The ensemble is now a clean deep ensemble, exactly as the LRAD note calls for.
- **Per-instance test figures**: for each test face, one standalone figure with the 10 members' reconstructions, the 10 per-model error maps, the Bias map, the Mean-error map, the **Min-error map on its raw scale (no vmin/vmax)**, and the Gaussian-smoothed bias painted over the face (`smooth_cam`, `inferno`, `alpha = cam ** overlay_power`). `overlay_sigma` and `overlay_power` are exposed as config knobs and affect **display only**, never the score. ~20 IND + ~20 OOD instances are saved per run under `ensemble/plots/instances_{in,ood}/`.
- **OOD set extended with hats**: `dataset.ood_attrs: [Eyeglasses, Wearing_Hat]` — in-distribution faces carry none of the OOD accessories.
- **ConvTranspose2d decoder variant** run in parallel: exactly the same 5-block trunk `channels = (32, 64, 128, 256, 256)`, with every decoder upsampling stage replaced by `ConvTranspose2d(4×4, stride 2)` (`scripts/oar_run_ensemble_transpose.sh`).
- **High-resolution rendering** for every figure: 300 dpi export, modern sans-serif typeface (Inter/Helvetica fallback chain), `constrained_layout`, hidden spines/ticks on image axes, colorbars where relevant, fixed colour scales so figures stay comparable across runs.
- **Grid'5000 (Nancy) batch scripts**: both variants submit via OAR to the `gratouille` cluster (1× A100 40 GB — the job needs < 10 GB of VRAM; the A100 is there to fit 10 models × (25 + 25) epochs in the window), walltime 24 h, advance-reservation supported.

## Architecture

**Classifier** — `FacialCNN`, shared 5-block conv trunk (L0…L4, `channels = [32, 64, 128, 256, 256]`) with gender and attribute heads. Each block exposes its post-activation tensor for the downstream decoders:

![Classifier diagram](docs/diagrams/Classifier.svg)

**Decoders** — one `BlockDecoder` per conv block, upsampling frozen activations back to `(3, 64, 64)` through ×2 stages (`bilinear` resize-then-conv by default, `ConvTranspose2d` as the variant):

![Decoder diagram](docs/diagrams/Decoder.svg)

## Features

- Exact pixelwise `Risk = Bias + Variance` decomposition, verified at runtime
- Plain deep ensemble: diversity comes only from independent weight init + SGD shuffle order (no regularizer)
- Predictive-uncertainty decomposition on classifier heads (total = aleatoric + epistemic / MI)
- Per-instance figures: each face's reconstruction + error across every member, the bias / mean / min summary, and the smoothed bias overlaid on the image
- Configurable OOD attribute set (eyeglasses, hats, …) and decoder upsampling (bilinear resize-then-conv or transposed conv)
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
# main run: 10-model ensemble, bilinear decoders (submit immediately)
./scripts/oar_run_ensemble.sh

# ConvTranspose2d variant on the 5-block trunk (can run in parallel)
./scripts/oar_run_ensemble_transpose.sh

# advance reservation (guaranteed window, never preempted)
./scripts/oar_run_ensemble.sh '2026-07-03 20:00:00'

# track / cancel
oarstat -u $USER
tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
oardel <jobid>
```

Both jobs reserve **1 GPU on `gratouille` (A100 40 GB) for 24 h**. The reservation is sized generously on purpose: an OAR job is cut at its walltime even mid-epoch and the ensemble run does not checkpoint.

## Usage examples

**Override ensemble size and learning rate:**

```bash
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --override ensemble.size=3 training.lr=5e-4
```

**Run the ConvTranspose2d decoder variant locally:**

```bash
python scripts/run_ensemble.py --config configs/celeba_ood.yaml \
    --override training.decoders.upsample=transpose
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
| `dataset` | `root`, `image_size`, `batch_size`, `train_ratio`, `val_ratio`, `ood_attrs` (attribute name or list defining OOD; default `[Eyeglasses, Wearing_Hat]`) |
| `model` | `channels` (one int per conv block), `n_attrs`, `n_gender` |
| `training` | `epochs`, `lr`, `attr_loss_weight`, `save_every_epoch`; nested `decoders: {epochs, lr, upsample}` (`upsample`: `bilinear` / `transpose`) |
| `evaluation` | `n_viz_in_samples`, `n_viz_ood_samples`, `viz_seed`, `score_comparison_block`, `score_comparison_k`, `n_instances_in`, `n_instances_ood`, `instance_block`, `overlay_sigma`, `overlay_power` (the last two are display-only knobs of the bias overlay) |
| `ensemble` | `size`, `base_seed` (model `i` uses `base_seed + i`), `agg` (`mean` / `max` / `p95`) |

## Project structure

```text
lrad/
├── lrad/                          # library (flat layout)
│   ├── dataset.py                 # CelebA loaders; configurable accessory OOD split
│   ├── model.py                   # FacialCNN: conv trunk + gender/attrs heads
│   ├── decoder.py                 # per-block decoders (bilinear / transpose stages)
│   ├── train.py                   # classifier + decoder training loops
│   ├── evaluate.py                # accuracy + classifier-confidence OOD AUROC
│   ├── anomaly_score.py           # per-pixel error + pixel→scalar reductions
│   ├── ensemble.py                # bias/variance decomposition
│   ├── plots.py                   # all figures (300 dpi, incl. per-instance view)
│   └── utils.py                   # device, seeding, logging
├── configs/celeba_ood.yaml        # single config file
├── docs/diagrams/                 # architecture diagrams (SVG, embedded above)
├── scripts/
│   ├── run_celeba.py              # single-model pipeline
│   ├── run_ensemble.py            # ensemble + decomposition + instance figures
│   ├── epoch_variability_study.py # σ(e) variability vs decoder epochs
│   ├── oar_run_ensemble.sh        # Grid'5000 OAR job — bilinear (default) run
│   └── oar_run_ensemble_transpose.sh  # Grid'5000 OAR job — transpose 5-block variant
└── tests/                         # pytest: anomaly score, decomposition,
                                   #         decoders, training, plots, checkpointing
```

Outputs are gitignored. An ensemble run writes per-model results under `model_<i>/` (weights, history, plots) and the decomposition under `ensemble/` (AUROC table, identity residual, all heatmap figures), with the per-instance figures under `ensemble/plots/instances_in/` and `ensemble/plots/instances_ood/` (one PNG per face, `ID_XX.png` / `OOD_XX.png`).

## Contributing

Open an issue or pull request. Run `pytest` and confirm no regressions before submitting.

## License

MIT — see [LICENSE](LICENSE).
