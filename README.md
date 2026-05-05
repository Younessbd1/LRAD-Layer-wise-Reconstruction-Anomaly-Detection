# CelebA Multi-Task CNN — Confidence-Based OOD Detection

> Train a multi-head CNN from scratch on CelebA faces *without sunglasses*.
> Use the model's predictive confidence to flag sunglasses as OOD.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Concept

A classifier trained only on glasses-free faces never learns features that
explain occluded eye regions. When a sunglasses face arrives at test time,
the classifier becomes uncertain on every head → **high predictive
entropy** = OOD.

```
                  TRAINING (no glasses)                          TESTING
                  ──────────────────────                         ───────
   Image (Eyeglasses=0)                                  Image (Eyeglasses=1)
        │                                                      │
        ▼                                                      ▼
   Conv backbone ─┬─► gender head  (softmax, CE)        same model
                  │                                            │
                  └─► attrs  head  (sigmoid, BCE × 6)          ▼
                                                       confused predictions
                                                       → high entropy = OOD
```

In-distribution targets:

| Head    | Attribute(s)                                                                 | Loss                |
|---------|------------------------------------------------------------------------------|---------------------|
| gender  | Male / Female                                                                | CrossEntropy        |
| attrs   | Young, Smiling, Mouth_Slightly_Open, High_Cheekbones, Pointy_Nose, Oval_Face | BCEWithLogitsLoss   |

Combined loss:

```
loss = CE(gender_logits, gender_target)
     + BCE(attr_logits,  attr_target)
loss.backward()
```

OOD score variants exposed at evaluation:
* `score_msp`              = 1 − max softmax probability of the gender head
* `score_entropy_gender`   = entropy of gender softmax
* `score_entropy_attrs`    = mean Bernoulli entropy across the 6 attribute heads
* `score_entropy_combined` = `score_entropy_gender + score_entropy_attrs`
  (default — uses signal from every head)

AUROC is computed by labelling glasses-free test images as 0 and Eyeglasses=1
images as 1 on each score above.

---

## Project structure

```
lrad/
├── lrad/                       # core library (flat layout)
│   ├── dataset.py              # CelebA loaders: filters Eyeglasses=1 from train
│   ├── model.py                # FacialCNN: 5-block conv trunk + 2 heads
│   ├── train.py                # combined CE + BCE training loop
│   ├── evaluate.py             # accuracy + OOD AUROC (entropy / MSP)
│   ├── utils.py                # device, seeding, logging
│   └── __init__.py
├── configs/
│   └── celeba_ood.yaml         # single config — used by run_celeba.py
├── scripts/
│   ├── run_celeba.py           # CLI orchestrator
│   └── oar_run_celeba.sh       # Grid'5000 / OAR submission wrapper
├── tests/__init__.py
├── data/celeba/...             # standard torchvision CelebA layout
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick start

```bash
pip install -e .

# Local run
python scripts/run_celeba.py --config configs/celeba_ood.yaml

# Eval only (load weights from output_dir)
python scripts/run_celeba.py --config configs/celeba_ood.yaml --eval-only

# Override config keys ad-hoc
python scripts/run_celeba.py --config configs/celeba_ood.yaml \
    --override training.epochs=50 dataset.batch_size=128
```

Outputs land in `outputs/celeba_ood/run/`:
* `weights/model.pt`           — best-val checkpoint
* `history.json`               — per-epoch training metrics
* `summary.json`                — final accuracies + OOD AUROC numbers
* `plots/training_history.png`  — loss + accuracy curves
* `plots/score_dist_*.png`      — in-dist vs OOD score histograms
* `plots/roc_ood.png`           — ROC curves for every OOD-score variant

### Grid'5000 / OAR

```bash
./scripts/oar_run_celeba.sh         # creates log dirs and submits the job
oarstat -u $USER
tail -f outputs/celeba_ood/run/logs/oar.<jobid>.stdout
```

### CelebA download

If the standard torchvision Google-Drive download fails on a compute node,
fetch the archive once on the frontend and unpack it under
`data/celeba/` with the layout:

```
data/celeba/img_align_celeba/        (202,599 .jpg files)
data/celeba/list_attr_celeba.txt
data/celeba/list_eval_partition.txt
data/celeba/identity_CelebA.txt
data/celeba/list_bbox_celeba.txt
data/celeba/list_landmarks_align_celeba.txt
```

---

## License

MIT — see [LICENSE](LICENSE).
