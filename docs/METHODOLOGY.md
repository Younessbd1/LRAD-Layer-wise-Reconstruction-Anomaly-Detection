# LRAD — Methodology

## 1. Problem formulation

We address **cold-start visual anomaly detection**: given only normal (defect-free) images at training time, detect and localise anomalous regions in unseen test images. No anomaly labels are available during training.

### Formal setup

- **Training set** X_N = {x₁, ..., x_n} where all xᵢ are normal.
- **Test set** X_T = {x₁, ..., x_m} containing both normal and anomalous samples.
- **Goal**: For each test image xₜ, produce:
  - An image-level anomaly score s(xₜ) ∈ ℝ
  - A pixel-level anomaly heatmap H(xₜ) ∈ ℝ^(H×W)

## 2. Architecture

### 2.1 Classifier as feature extractor

We train a standard classifier (CNN or MLP) on the normal data. The key insight: a classifier trained exclusively on normal-class data develops intermediate representations **biased toward the normal distribution**. These representations encode the expected structure, texture, and semantics of normal images.

#### CNN variant
```
x → [Conv-BN-ReLU]₁ → [Conv-BN-ReLU]₂ → ... → [Conv-BN-ReLU]_K → GAP → FC → logits
         ↓ a₁              ↓ a₂                      ↓ a_K
```

Each activation aₖ ∈ ℝ^(Cₖ × Hₖ × Wₖ) retains spatial structure, allowing direct spatial heatmap computation.

#### MLP variant
```
x.flatten() → [Linear-BN-ReLU]₁ → ... → [Linear-BN-ReLU]_K → FC → logits
                   ↓ a₁                      ↓ a_K
```

Activations aₖ ∈ ℝ^(dₖ) are 1D vectors. Heatmaps are produced by the decoder's reshape operation.

### 2.2 Layer-wise decoders

After training, the classifier is **frozen** (all gradients disabled). We then train K independent decoders {D₁, ..., D_K}, one per hidden layer, to reconstruct the original input image from each layer's activation:

```
D_k(a_k) ≈ x    for all x ∈ X_N
```

Each decoder is trained with MSE loss:

```
L_k = (1/N) Σᵢ ||D_k(a_k(xᵢ)) - xᵢ||²
```

#### CNN decoder (ConvTranspose2d)
Mirrors the encoder path with transposed convolutions. Output padding is computed automatically to match target spatial sizes. Final layer uses Sigmoid to produce pixel values in [0, 1].

#### MLP decoder (Linear)
Symmetric linear layers mapping the hidden dimension back to the flattened image dimension, then reshapes to (C, H, W).

### 2.3 Why freeze the classifier?

Freezing is critical. If the classifier were trainable during decoder training, the system would degenerate into a standard autoencoder — the classifier would adapt its representations to make reconstruction easy for ANY input, including anomalies. By freezing, we ensure the representations remain biased toward normal data, making OOD reconstruction inherently difficult.

## 3. Anomaly detection at test time

### 3.1 Per-layer error maps

For a test image xₜ, we compute the reconstruction from each decoder and the pixel-wise squared error:

```
E_k(xₜ) = (D_k(a_k(xₜ)) - xₜ)²    ∈ ℝ^(C × H × W)
```

Averaging over channels gives a spatial heatmap per layer:

```
H_k(xₜ) = mean_c(E_k(xₜ))    ∈ ℝ^(H × W)
```

### 3.2 Multi-scale fusion

Different decoder levels capture anomalies at different granularities:
- **Shallow decoders** (from early layers): Capture fine-grained texture anomalies but may be noisy.
- **Deep decoders** (from late layers): Capture structural/semantic anomalies but have lower spatial resolution.

We fuse per-layer heatmaps into a single anomaly map. Three strategies:

**Mean fusion**: H(xₜ) = (1/K) Σₖ H_k(xₜ)

**Max fusion**: H(xₜ) = max_k H_k(xₜ)    (pixel-wise)

**Weighted fusion**: H(xₜ) = Σₖ wₖ · H_k(xₜ)  where wₖ ∝ 1/MSE_k (better-reconstructing decoders get more weight)

### 3.3 Image-level score

The image-level anomaly score is the maximum pixel value in the fused heatmap:

```
s(xₜ) = max_{h,w} H(xₜ)[h, w]
```

This follows the PatchCore principle: an image is anomalous as soon as a single spatial region is anomalous.

## 4. Evaluation protocol

### 4.1 MNIST protocol

| Split | Content | Expected behaviour |
|-------|---------|-------------------|
| Train | Digits [0,1,2,3] | Classifier learns, decoders learn to reconstruct |
| Test normal | Digits [0,1,2,3] (test set) | Low reconstruction error, low heatmap intensity |
| Test anomaly (near-OOD) | Digits [4,5,6,7,8,9] | Higher error — digit structure differs |
| Test anomaly (far-OOD) | Fashion-MNIST | Much higher error — completely different distribution |

### 4.2 Metrics

- **Image-level AUROC**: Area under ROC curve using s(xₜ) to separate normal vs anomaly.
- **Score distribution separation**: Visual histogram of normal vs anomaly scores.

## 5. Connection to PatchCore and UQ methods

### Similarities with PatchCore
- Both use **frozen pretrained features** as the representation backbone.
- Both produce **spatial anomaly maps** for localization.
- Both follow the principle that **a single anomalous patch is sufficient** for image-level detection.

### Key differences
| Aspect | PatchCore | LRAD |
|--------|-----------|------|
| Feature source | ImageNet-pretrained (external) | Task-trained on normal data (internal) |
| Scoring method | 1-NN distance to memory bank | Reconstruction error from decoders |
| Memory requirement | Stores coreset of patch features | Stores decoder weights |
| Adaptation | No adaptation to target domain | Classifier IS adapted |

### Path to uncertainty quantification

LRAD provides a natural bridge to UQ methods:

1. **MC Dropout in decoders**: Add dropout layers to decoders, perform T forward passes at test time. The variance of reconstructions across passes captures epistemic uncertainty about the reconstruction — high variance = the decoder is unsure = potential anomaly.

2. **Decoder ensembles**: Train M decoders with different initialisations for each layer. Disagreement between ensemble members quantifies model uncertainty.

3. **Evidential reconstruction**: Replace MSE loss with a Normal-Inverse-Gamma (NIG) prior, predicting both the reconstruction AND its uncertainty in a single forward pass.

## 6. Scaling to complex scenes

For multi-object images with localised anomalies:

1. **Higher-resolution backbones**: Use deeper CNNs (ResNet-18) as the classifier to capture richer spatial features.
2. **Skip connections in decoders**: U-Net style connections from encoder to decoder preserve fine spatial detail.
3. **Patch-level evaluation**: Divide the image into overlapping patches, score each independently, stitch into a full heatmap.
4. **Attention-guided fusion**: Learn attention weights over decoder levels instead of fixed fusion.
