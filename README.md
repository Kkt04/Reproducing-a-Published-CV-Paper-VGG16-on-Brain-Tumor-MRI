# Flatten or Pool? Revisiting VGG16 Transfer Learning for Brain Tumor MRI Classification

> An independent reproduction of a VGG16 brain tumor classifier, a duplicate-image audit of its test set, and a pre-registered single-variable experiment: **Flatten vs Global Average Pooling (GAP)**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Task](https://img.shields.io/badge/Task-4--class%20MRI%20classification-teal)
![Status](https://img.shields.io/badge/Status-Research%20reproduction-lightgrey)

**Authors:** Dally R, Kalash Kumari Thakur
**Institution:** Rishihood University, India
**Course context:** Computer Vision research-reproduction project (M1-M5)

> **Disclaimer:** This is an academic study. **No clinical claim is made.** The models classify single 2D slices from a public dataset and must not be used for diagnosis.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Findings](#key-findings)
3. [Dataset](#dataset)
4. [Method](#method)
5. [Results](#results)
6. [Discussion](#discussion)
7. [Limitations](#limitations)
8. [Future Work](#future-work)
9. [Repository Structure](#repository-structure)
10. [Getting Started](#getting-started)
11. [Reproducibility](#reproducibility)
12. [References](#references)
13. [Authors](#authors)

---

## Overview

Deep learning models for brain tumor MRI classification routinely report **96-99.6%** accuracy on public benchmarks. When accuracy is near a ceiling, the more useful questions are: *can the numbers be reproduced, how were they measured, and which errors remain?*

We study the model of **Wong et al. (PLoS ONE, 2025)**, an ImageNet-pretrained VGG16 fine-tuned on MRI images, which reports **99.24%** accuracy on four classes:

`glioma` · `meningioma` · `pituitary tumor` · `no tumor`

No code was released by the original authors, so we rebuilt the method from its description and documented every unspecified setting.

### Contributions

1. **Independent reproduction** of the base model, with all unspecified settings documented.
2. **Duplicate audit** showing that 25% of test images match a training image, and measuring how much this inflates accuracy.
3. **Single-variable experiment** (Flatten vs GAP) with predictions written down *before* running it, plus a second independent baseline run.
4. **Failure analysis** using confusion matrices, misclassified examples and Grad-CAM.

---

## Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | The reported accuracy was only partly reproduced | 95.25% vs 99.24% (95.38% in a second run) |
| 2 | The gap is concentrated in **glioma** | Glioma recall 84.5% vs 100% in the original |
| 3 | **25% of test images are duplicates of training images** | 202 / 800 share an image hash with a training image |
| 4 | Leakage inflates accuracy | 99.50% on duplicates vs 93.81% on unseen images |
| 5 | **Hypothesis refuted**: GAP did *not* help | Accuracy 95.38% to 92.25%, glioma recall 85.0% to 72.5% |
| 6 | Overfitting comes mainly from **fine-tuning**, not the head | Both models reach similar validation loss (~1.0-1.1) in stage 2 |

---

## Dataset

The base paper combines three public sources (Figshare, SARTAJ, Br35H), also distributed on Kaggle:

- **Brain Tumor MRI Dataset** by M. Nickparvar, Kaggle: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

The Kaggle copy we used is a **different, balanced version** compared with the base paper. To match the base paper's validation/test sizes, the original test folder is split equally into validation and test sets per class.

| | Wong et al. | This study |
|---|---|---|
| Total images | 7,023 | 7,200 |
| Training (real / after augmentation) | 5,712 / 17,136 | 5,600 / 16,800 |
| Validation / Test | 656 / 655 | 800 / 800 |
| Class balance | Unbalanced | 25% each |

> The dataset is **not** included in this repository. Download it from Kaggle (see [Getting Started](#getting-started)).

---

## Method

### Pipeline

```
MRI slice -> grayscale -> resize 224x224 -> replicate to 3 channels -> rescale [0,1]
          -> augment (x3, training only)
          -> VGG16 (ImageNet, blocks 1-3 frozen, blocks 4-5 fine-tuned in stage 2)
          -> Flatten  OR  Global Average Pooling   <-- the only change
          -> Dense 256 (ReLU)
          -> Dense 4 (softmax)
```

- **Augmentation (offline):** shear and zoom range 0.2, horizontal flip. Two augmented copies per training image, tripling the training set (5,600 to 16,800). Validation and test images are not augmented.

### The single variable: Flatten vs GAP

VGG16 outputs a `7 x 7 x 512` block: 512 feature maps, each recording where one visual pattern occurs.

| Head | Values into Dense layer | Weights in Dense 256 |
|------|-------------------------|----------------------|
| **Flatten** (baseline) | 25,088 | 25,088 x 256 + 256 = **6,422,784** (~6.4 M) |
| **GAP** (tested) | 512 | 512 x 256 + 256 = **131,328** (~0.13 M) |

GAP has about **49x fewer weights**. Flatten keeps *where* each pattern occurs; GAP keeps only *how strongly* it occurs.

### Hypothesis (stated before running)

> Replacing Flatten with GAP will reduce overfitting, because a classifier with 49x fewer weights cannot easily memorize ~5,600 real training images and must rely on general tumor features instead.

### Pre-registered predictions

| Measure | Prediction | Reason |
|---------|------------|--------|
| Test accuracy | +1 to +3 pts | less memorization |
| Validation loss | 30-60% lower | fewer confident errors |
| Train-val loss gap | smaller | direct overfitting sign |
| Glioma recall | higher | weakest class |
| Pituitary recall | may decrease (flagged risk) | position cue lost |
| Macro AUC | slightly higher | better ranking |

### Two-stage training (identical for both models)

| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Trainable | Head only (VGG16 frozen) | Head + VGG16 blocks 4-5 (~13 M backbone weights) |
| Optimizer / LR | Adam, 1e-3 | Adam, 1e-4 |
| Epochs | 50 | 50 |
| Batch size / steps per epoch | 32 / 179 | 32 / 179 |
| Loss | Categorical cross-entropy | Categorical cross-entropy |
| Seed / hardware | 42 / NVIDIA T4 (Kaggle), TensorFlow 2.20 | same |

### Evaluation

- Accuracy, per-class recall (sensitivity) and specificity
- Macro one-vs-rest AUC
- Log loss on full validation and test sets
- Mean train-validation loss gap over the last 5 epochs
- Expected Calibration Error (ECE, 15 bins)
- **Leakage audit:** 64-bit difference hash (dHash) for every image; a test image is flagged if its hash equals that of any training image. This is a **lower bound** (different slices of the same patient are not detected).
- Qualitative analysis with confusion matrices, most-confident errors and Grad-CAM

---

## Results

### 1. Reproduction of the base paper

| Metric (%) | Wong et al. | Ours |
|------------|-------------|------|
| Accuracy | 99.24 | 95.25 |
| Macro sensitivity | 99.18 | 95.25 |
| Macro specificity | 99.75 | 98.42 |

**Per-class recall (%)**

| Class | Wong et al. | Ours |
|-------|-------------|------|
| Glioma | 100 | **84.5** |
| Meningioma | 97.39 | 97 |
| No tumor | 100 | 100 |
| Pituitary | 99.33 | 99.5 |

Our reproduction gets 762 of 800 test images correct. Glioma accounts for **31 of 38 errors** (17 predicted as meningioma, 12 as no tumor, 2 as pituitary).

### 2. Test-set leakage audit

| Subset | Images | Accuracy |
|--------|--------|----------|
| Test images matching a training image | 202 (25.3%) | 99.50% (201/202) |
| Truly unseen test images | 598 | 93.81% (561/598) |
| All test images | 800 | 95.25% |

Repeated images raise measured accuracy by **1.44 points** in our setting. The original split may be affected similarly but was not audited.

### 3. Baseline stability (two independent Flatten runs)

| Run | Accuracy (%) | Glioma recall (%) | Unseen acc. (%) |
|-----|--------------|-------------------|-----------------|
| Reproduction run | 95.25 | 84.5 | 93.81 |
| Ablation run | 95.38 | 85.0 | 93.98 |
| Difference | 0.13 | 0.5 | 0.17 |

Run-to-run variation is small, so the glioma weakness is **systematic**, not chance.

### 4. Flatten vs GAP (same run, same data, same seed)

| Metric | Flatten | GAP |
|--------|---------|-----|
| Test accuracy (%) | **95.38** | 92.25 |
| Macro AUC | **0.989** | 0.986 |
| Test loss | **0.450** | 0.755 |
| Validation loss (full set) | **0.616** | 0.899 |
| Train-val loss gap (last 5 epochs) | **1.061** | 1.090 |
| Recall: glioma (%) | **85.0** | 72.5 |
| Recall: meningioma (%) | **97.5** | 96.5 |
| Recall: no tumor (%) | 100.0 | 100.0 |
| Recall: pituitary (%) | 99.0 | **100.0** |
| Macro specificity (%) | **98.46** | 97.42 |
| ECE (lower is better) | **0.037** | 0.056 |
| Accuracy, unseen images (%) | **93.98** | 89.80 |
| Accuracy, duplicate images (%) | 99.50 | 99.50 |

### 5. Predictions vs observed outcomes

| Measure | Predicted | Observed | Held? |
|---------|-----------|----------|-------|
| Test accuracy | +1 to +3 pts | -3.13 pts | No |
| Validation loss | 30-60% lower | 46% higher | No |
| Train-val gap | smaller | 1.06 to 1.09 | No |
| Glioma recall | higher | 85.0 to 72.5% | No |
| Macro AUC | slightly higher | 0.989 to 0.986 | No |
| Pituitary recall (risk) | may decrease | 99 to 100% | Risk avoided |

**Result: H1 refuted.** All five directional predictions failed.

### 6. Overfitting by training stage

Mean of the last 10 epochs of each stage (training-time validation subset):

| | Stage 1 (frozen) Flatten | Stage 1 GAP | Stage 2 (fine-tuned) Flatten | Stage 2 GAP |
|---|---|---|---|---|
| Training loss | 0.022 | 0.150 | 0.011 | 0.017 |
| Validation loss | 1.284 | **0.679** | 1.029 | 1.100 |
| Validation accuracy (%) | **90.1** | 85.9 | 93.3 | 92.6 |

- **Stage 1 (frozen backbone):** GAP's validation loss is ~47% lower than Flatten's, so the idea *worked* here.
- **Stage 2 (fine-tuned):** both models drive training loss toward zero and reach similar validation loss (~1.0-1.1).

> Note: training-time validation uses only the first 672 of 800 validation images (to follow the base paper's 21 validation steps). Because images are ordered by class, 128 pituitary images are omitted, so these values differ from the full-set numbers in the table above. They are comparable between models, not across tables.

### 7. Glioma failure analysis

| Model | Glioma errors | to Meningioma | to No tumor | to Pituitary |
|-------|---------------|---------------|-------------|--------------|
| Flatten (ablation run) | 30 | 19 | 10 | 1 |
| GAP | 55 | 26 | 19 | 10 |

Observed traits of the baseline's most confident mistakes: large, sharply bounded masses resembling meningiomas, mixed MRI sequences and orientations, at least one CT-like image, and repeated glioma images in the test set. Grad-CAM maps are often nearly empty for the Flatten model (consistent with saturated, over-confident softmax outputs). This assessment is qualitative.

---

## Discussion

**Why the hypothesis failed.** It rested on two assumptions: (1) the classifier head was the main source of overfitting, and (2) spatial position information was dispensable. Both proved wrong.

- The head *did* drive overfitting while the backbone was frozen, and GAP removed much of it.
- But fine-tuning ~13 M backbone weights on a training set built from fixed augmented copies reintroduced overfitting that the head cannot control.
- Discarding spatial layout cost discriminative information: GAP's glioma errors rose from 30 to 55.
- The anticipated pituitary risk did not occur (recall 100%), suggesting pituitary tumors are recognized from appearance rather than position.

**Why the reproduction gap (95.25% vs 99.24%)?** Likely contributors: a different dataset version, unreported augmentation strengths, random initialization and GPU non-determinism, and leakage of unknown size in the original split. Without the original code these factors cannot be separated.

---

## Limitations

- Single public dataset, **no external validation**.
- GAP was trained **only once**; two baseline runs give only a rough estimate of variation (though the 3.13-point drop is far larger than the 0.13-point baseline difference).
- Duplicate audit misses different slices of the same patient; **no patient-level split** was possible (no patient identifiers).
- Leaked images were measured but **not removed**.
- Models classify single 2D slices; the dataset mixes MRI sequences and contains at least one image that appears to be CT.
- Grad-CAM analysis is qualitative only.
- **No clinical claim is made.**

---

## Future Work

1. **Online augmentation:** generate fresh random augmentations every epoch instead of two fixed offline copies, to reduce stage-2 overfitting and improve glioma recall.
2. **Best-checkpoint selection:** keep the checkpoint with the lowest validation loss to avoid the late-training rise in validation loss.
3. Patient-level splits, multiple seeds, external test data.
4. Quantitative heatmap evaluation (e.g., overlap of Grad-CAM with tumor regions).

---

## Repository Structure

> TODO: update this tree to match your actual repository.

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_reproduction.ipynb        # Flatten baseline reproduction
│   ├── 02_duplicate_audit.ipynb     # dHash leakage audit
│   └── 03_flatten_vs_gap.ipynb      # single-variable ablation
├── figures/                          # confusion matrices, Grad-CAM, curves
├── paper/
│   └── Final_Paper.pdf
└── presentation/
    └── final_presentation.pdf
```

---

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.20
- A GPU is recommended (we used an NVIDIA T4 on Kaggle)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Download the dataset

```bash
# Requires a configured Kaggle API token
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```

### Run

> TODO: replace with your actual commands or notebook order.

```bash
# 1. Reproduce the Flatten baseline
jupyter notebook notebooks/01_reproduction.ipynb

# 2. Run the duplicate audit
jupyter notebook notebooks/02_duplicate_audit.ipynb

# 3. Run the Flatten vs GAP ablation
jupyter notebook notebooks/03_flatten_vs_gap.ipynb
```

---

## Reproducibility

| Item | Value |
|------|-------|
| Random seed | 42 |
| Hardware | NVIDIA T4 (Kaggle) |
| Framework | TensorFlow 2.20 |
| Baseline runs | 2 independent runs |
| GAP runs | 1 |
| Image preprocessing | grayscale, 224x224, 3-channel replicate, rescale to [0,1] |

GPU non-determinism means exact numbers may vary slightly between runs (our two baseline runs differed by 0.13 accuracy points).

---

## References

1. Y. Wong, E. L. M. Su, C. F. Yeong, W. Holderbaum, and C. Yang, "Brain tumor classification using MRI images and deep learning techniques," *PLoS ONE*, vol. 20, no. 5, e0322624, 2025. *(base paper)*
2. R. Disci, F. Gurcan, and A. Soylu, "Advanced brain tumor classification in MR images using transfer learning and pre-trained deep CNN models," *Cancers*, vol. 17, no. 1, p. 121, 2025.
3. M. M. Zahoor et al., "Brain tumor MRI classification using a novel deep residual and regional CNN," *Biomedicines*, 2024.
4. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015.
5. J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. CVPR*, 2009.
6. M. Lin, Q. Chen, and S. Yan, "Network in network," in *Proc. ICLR*, 2014. *(global average pooling)*
7. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. ICCV*, 2017.
8. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017.
9. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.
10. E. Yagis et al., "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Sci. Rep.*, vol. 11, 22544, 2021.
11. M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle. <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

See `paper/Final_Paper.pdf` for the full reference list (20 references).

---

## Authors

- **Dally R**, Rishihood University
- **Kalash Kumari Thakur**, Rishihood University

If you use or build on this work, please cite the paper or link to this repository.
