# LRAD - Layer-wise Reconstruction Anomaly Detection

> Train a classifier on normal data. Freeze it. Train decoders on its hidden activations.  
> At test time, reconstruction error = anomaly heatmap.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Concept

LRAD exploits the fact that a classifier's intermediate representations are **tuned to its training distribution**. When an out-of-distribution (OOD) input passes through the frozen classifier, its activations in anomalous spatial regions become unusual - decoders trained to reconstruct normal images from those activations **fail locally**, producing high reconstruction error exactly where the anomaly is.

```
                    TRAINING                              TESTING
                    ────────                              ───────
  Normal Image ──► Classifier (freeze) ──► Activations    Test Image ──► Same Classifier ──► Activations
                                               │                                                  │
                                               ▼                                                  ▼
                                     Decoder_k(act_k) ──► Recon      Decoder_k(act_k) ──► Recon
                                           │                               │
                                     Loss(Recon, Image) = 0          |Recon - Image|² = HEATMAP
```

### Key innovations over vanilla autoencoders

1. **Multi-scale heatmaps** - Each decoder level captures anomalies at a different granularity (fine texture vs. structural), then they're fused into one map.
2. **Classifier bias** - The frozen classifier acts as a learned feature extractor biased toward normal-class semantics, making OOD reconstruction harder than a generic autoencoder would.
3. **No anomaly labels needed** - Pure one-class learning. Only normal samples required for training.

---

## Project structure

```
lrad/
├── lrad/                        # Core library
│   ├── models/
│   │   ├── classifier.py        # CNN and MLP classifiers
│   │   ├── decoder.py           # Spatial (CNN) and reshape (MLP) decoders
│   │   └── lrad_model.py        # Full LRAD pipeline (classifier + decoders)
│   ├── data/
│   │   └── datasets.py          # Dataset loading, filtering, OOD splits
│   ├── engine/
│   │   ├── trainer.py           # Training loops for classifier & decoders
│   │   └── evaluator.py         # Anomaly scoring, metrics (AUROC, pixel-AUROC)
│   ├── visualization/
│   │   └── heatmaps.py          # Heatmap rendering, multi-scale fusion
│   └── utils/
│       └── helpers.py           # Device, seeding, logging utilities
├── configs/
│   ├── mnist_mlp.yaml           # MLP config for MNIST protocol
│   ├── mnist_cnn.yaml           # CNN config for MNIST protocol
│   └── cifar10_cnn.yaml         # CNN config for CIFAR-10 protocol
├── scripts/
│   └── run_experiment.py        # Main entry point
├── tests/
│   └── test_models.py           # Unit tests
├── docs/
│   └── METHODOLOGY.md           # Detailed methodology writeup
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick start

```bash
# Install
pip install -e .

# Run MNIST MLP experiment (Protocol 1)
python scripts/run_experiment.py --config configs/mnist_mlp.yaml

# Run MNIST CNN experiment
python scripts/run_experiment.py --config configs/mnist_cnn.yaml

# Run CIFAR-10 experiment
python scripts/run_experiment.py --config configs/cifar10_cnn.yaml
```

---

## Protocols

## Protocol 1 - MNIST + MLP
Train an MLP classifier on digits `[0,1,2,3]`. Train N decoders (one per hidden layer) to reconstruct 28×28 images from each layer's activations. Test on held-out `[0,1,2,3]` (should reconstruct well) and on `[4,...,9]` + Fashion-MNIST (should reconstruct poorly -> anomaly heatmap).

## Protocol 2 - MNIST + CNN
Same as Protocol 1, but with spatial convolutions. Decoders use transposed convolutions, preserving spatial structure. Heatmaps are naturally pixel-aligned.

## Protocol 3 - CIFAR-10 + CNN
Train on one CIFAR-10 class (e.g., "airplane"). Test against other classes. Demonstrates scaling to RGB, 32×32, more complex textures.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{lrad2025,
  title={LRAD: Layer-wise Reconstruction Anomaly Detection},
  author={Youness},
  year={2025},
  note={Research internship project - Uncertainty Quantification for Anomaly Detection}
}
```

---

## License

MIT - see [LICENSE](LICENSE).
