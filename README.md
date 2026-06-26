# LRAD — Layer-wise Reconstruction Anomaly Detection

OOD detection on CelebA faces via deep-ensemble reconstruction error decomposed into bias (anomaly) and variance (epistemic uncertainty).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

## What it does

A from-scratch convolutional classifier is trained on accessory-free CelebA faces; lightweight decoders attached to each conv block learn to invert that block's frozen activations back to a `(3, 64, 64)` reconstruction. Training `M` such models independently forms a deep ensemble whose per-pixel reconstruction risk decomposes exactly as `Risk = Bias + Variance`. The **bias** term — the irreducible error of the consensus reconstruction — is the anomaly score; faces wearing eyeglasses or a hat are held out as OOD (the attribute set is configurable) and score higher.

## Architecture

**Classifier** — convolutional trunk with gender and attribute heads:

![Classifier diagram](docs/diagrams/Classifier.svg)

**Decoder** — per-block reconstruction head, upsampling frozen activations back to `(3, 64, 64)`:

![Decoder diagram](docs/diagrams/Decoder.svg)

## Features

- Exact pixelwise `Risk = Bias + Variance` decomposition, verified at runtime
- Anchored ensembling: L2-toward-random-init penalty calibrates the epistemic (variance) term across members
- Predictive-uncertainty decomposition on classifier heads (total = aleatoric + epistemic / MI)
- Per-instance figures: each face's reconstruction + error across every member, the bias / mean / min summary, and the smoothed bias overlaid on the image
- Configurable OOD attribute set (eyeglasses, hats, …) and decoder upsampling (bilinear resize-then-conv or transposed conv)
- Epoch-variability study: traces inter-model variability σ(e) from per-epoch decoder checkpoints
- Single YAML config shared by both single-model and ensemble runners; dotted `key=value` CLI overrides

## Tech stack

| Component | Libraries |
| --- | --- |
| Runtime | Python 3.10+, PyTorch 2.4, torchvision |
| Numerics | NumPy < 2.0, scikit-learn |
| Visualisation | matplotlib |
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
| `dataset` | `root`, `image_size`, `batch_size`, `train_ratio`, `val_ratio`, `ood_attrs` (attribute name or list defining OOD) |
| `model` | `channels` (one int per conv block), `n_attrs`, `n_gender` |
| `training` | `epochs`, `lr`, `weight_decay`, `attr_loss_weight`, `save_every_epoch`; nested `decoders: {epochs, lr, anchor_lambda, upsample}` (`upsample`: `bilinear` / `transpose`) |
| `evaluation` | `n_viz_in_samples`, `n_viz_ood_samples`, `viz_seed`, `score_comparison_block`, `score_comparison_k`, `n_instances_in`, `n_instances_ood`, `overlay_sigma`, `overlay_power` |
| `ensemble` | `size`, `base_seed` (model `i` uses `base_seed + i`), `agg` (`mean` / `max` / `p95`) |

## Project structure

```text
lrad/
├── lrad/                          # library (flat layout)
│   ├── dataset.py                 # CelebA loaders; configurable accessory OOD split
│   ├── model.py                   # FacialCNN: conv trunk + gender/attrs heads
│   ├── decoder.py                 # per-block reconstruction decoders
│   ├── train.py                   # classifier + decoder training loops
│   ├── evaluate.py                # accuracy + classifier-confidence OOD AUROC
│   ├── anomaly_score.py           # per-pixel error + pixel→scalar reductions
│   ├── ensemble.py                # bias/variance decomposition
│   ├── plots.py                   # all figures
│   └── utils.py                   # device, seeding, logging
├── configs/celeba_ood.yaml        # single config file
├── scripts/
│   ├── run_celeba.py              # single-model pipeline
│   ├── run_ensemble.py            # ensemble + decomposition
│   └── epoch_variability_study.py # σ(e) variability vs decoder epochs
└── tests/                         # pytest: anomaly score, decomposition,
                                   #         decoders, training, checkpointing
```

Outputs are gitignored. An ensemble run writes per-model results under `model_<i>/` (weights, history, plots) and the decomposition under `ensemble/` (AUROC table, identity residual, all heatmap figures), with the per-instance figures under `ensemble/plots/instances_in/` and `ensemble/plots/instances_ood/` (one PNG per face).

## Contributing

Open an issue or pull request. Run `pytest` and confirm no regressions before submitting.

## License

MIT — see [LICENSE](LICENSE).
