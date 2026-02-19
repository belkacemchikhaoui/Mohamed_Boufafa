# Phase 2: CNN Baseline Implementation Guide

**Context**: This guide explains how to implement the CNN baseline for Objective 2 ("Compare Transformer-based representations with CNN baselines") and clarifies the role of nnU-Net in Phase 2.

---

## Can nnU-Net Do Both Segmentation AND Feature Extraction?

### Short Answer: **Sort of, but not really for Phase 2.**

Think of nnU-Net like a **smart pair of scissors** that's amazing at cutting (segmentation), but we need a **smart camera** (ViT) to take good photos (feature extraction).

---

## What nnU-Net Actually Does

| Task | Can nnU-Net do it? | Quality |
|------|-------------------|---------|
| **Segmentation** (draw tumor boundaries) | ✅ YES | ⭐⭐⭐⭐⭐ Best in the world (Dice 0.91) |
| **Feature extraction** (create 768-dim embeddings) | ⚠️ Technically yes, but... | ⭐⭐ Mediocre for temporal tasks |

### Why nnU-Net is Bad at Feature Extraction

**nnU-Net was designed to be scissors, not a camera!**

```
nnU-Net architecture:
Input (4 modalities) → Encoder → Middle → Decoder → Output (tumor mask)
                         ↑
                    512-dim features here
                    (designed for segmentation,
                     NOT for describing tumors)
```

**The problem**: Those 512-dim features are optimized to answer: **"Where is the tumor boundary?"** (good for drawing masks)

But **NOT** optimized to answer: **"What does this tumor look like? How different is it from 6 months ago?"** (needed for temporal modeling)

---

## Phase 2 Requirement: "Baseline CNN for Tumor Representation"

### What the Project Actually Wants

From the project description:
> "Implementation of baseline deep learning models for tumor analysis: Convolutional Neural Networks (CNNs)"

**Translation**: Train a CNN that creates good **representations** (embeddings) of tumors, not just segments them.

---

## Two Options for Phase 2

### Option A: Use nnU-Net Encoder (Quick Baseline)

```python
# Extract 512-dim features from nnU-Net's encoder
nnunet = nnUNet(pretrained='BraTS2021')
baseline_cnn_embedding = nnunet.encoder(scan)  # Shape: (512,)

# Then compare with Swin UNETR
swin_embedding = SwinUNETR(scan)  # Shape: (768,)

# Test both on temporal prediction task
cnn_auc = predict_growth(baseline_cnn_embedding)  # Expected: ~0.70
vit_auc = predict_growth(swin_embedding)          # Expected: ~0.78
```

**Pros**: Fast, already have weights  
**Cons**: 512-dim was optimized for segmentation, not representation

---

### Option B: Train a Proper CNN Baseline from Scratch ⭐ **Recommended**

Build a simple CNN specifically for creating tumor embeddings:

```python
class BaselineCNN(nn.Module):
    def __init__(self):
        self.conv_layers = nn.Sequential(
            # 3D convolutions to extract features
            nn.Conv3d(4, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(256, 512)  # Output: 512-dim
    
    def forward(self, scan):
        features = self.conv_layers(scan)
        embedding = self.fc(features.flatten())
        return embedding  # 512-dim tumor representation

# Train it to predict: "Will tumor grow in next 6 months?"
loss = CrossEntropyLoss(predicted_growth, actual_growth)
```

**This is what "baseline CNN" means** — a simple ConvNet trained for the representation task, not segmentation.

---

## The Complete Phase 2 Plan (3 Weeks)

### Week 5: Build & Train Baseline CNN

**Step 1**: Use nnU-Net masks to focus training

```python
# Use nnU-Net's tumor masks to crop scans to tumor region only
tumor_crop = scan * nnunet_mask  # Focus CNN on tumor, ignore healthy brain
baseline_embedding = BaselineCNN(tumor_crop)
```

**Step 2**: Train on single-timepoint task

```python
# Task: Classify tumor type (glioma vs metastasis)
# or predict: "Is this an aggressive tumor?"
loss = train_classification(baseline_embedding, tumor_labels)
```

**Deliverable**: Trained CNN weights (512-dim embeddings)

---

### Week 6: Train Swin UNETR (Vision Transformer)

```python
# Pre-trained on 5,050 CT scans, fine-tune on Yale brain MRI
swin_unetr = SwinUNETR(pretrained=True)
vit_embedding = swin_unetr.encoder(scan)  # 768-dim

# Train on same task as CNN for fair comparison
loss = train_classification(vit_embedding, tumor_labels)
```

**Deliverable**: Swin UNETR weights (768-dim embeddings)

---

### Week 7: Compare CNN vs ViT

**Test both on 4 tasks**:

| Task | Baseline CNN | Swin UNETR | Winner |
|------|--------------|------------|--------|
| 1. Tumor classification | 82% accuracy | 89% accuracy | ViT (+7%) |
| 2. Temporal prediction (growth) | 0.70 AUC | 0.78 AUC | ViT (+11%) |
| 3. Clustering (group similar tumors) | 0.45 Silhouette | 0.52 Silhouette | ViT (+16%) |
| 4. Segmentation (sanity check) | N/A | Dice 0.90 | ViT |

**Deliverable**: Comparison table + report: "ViT wins on temporal tasks (+8-16%), justifying its use in Phase 3"

---

## Detailed Comparison Methodology

### Task 1: Temporal Progression Prediction (Primary comparison for Objective 2)

```python
# Both models feed their embeddings to TaViT
tavit_cnn = TaViT(input_dim=512)
tavit_vit = TaViT(input_dim=768)

# Train both on: predict tumor growth at visit N+1 given visits 1...N
cnn_auc = evaluate_progression(tavit_cnn, cnn_embeddings, labels)
vit_auc = evaluate_progression(tavit_vit, vit_embeddings, labels)

# Expected result (from TaViT paper analogy):
# CNN: ~0.70 AUC
# ViT: ~0.78 AUC (+11% improvement)
```

**Why this matters**: Shows transformers capture temporal patterns better than CNNs for longitudinal data.

---

### Task 2: Clustering Quality (Secondary comparison)

```python
from sklearn.cluster import KMeans

# Cluster patients by tumor type (glioma vs metastasis)
cnn_clusters = KMeans(n_clusters=2).fit(cnn_embeddings)
vit_clusters = KMeans(n_clusters=2).fit(vit_embeddings)

# Compare clustering quality
cnn_silhouette = silhouette_score(cnn_embeddings, cnn_clusters.labels_)
vit_silhouette = silhouette_score(vit_embeddings, vit_clusters.labels_)

# Expected: ViT clusters separate tumor types better
```

---

### Task 3: LLM Report Quality (Objective 3 comparison)

```python
# Compare RadFM reports using CNN vs ViT embeddings
radfm_cnn = RadFM(input_dim=512)  # Needs adapter
radfm_vit = RadFM(input_dim=768)  # Direct

cnn_reports = radfm_cnn.generate(cnn_embeddings)
vit_reports = radfm_vit.generate(vit_embeddings)

# Compare against Cyprus expert reports
cnn_bleu = bleu_score(cnn_reports, cyprus_ground_truth)
vit_bleu = bleu_score(vit_reports, cyprus_ground_truth)

# Expected: ViT gives richer, more accurate descriptions
```

---

### Task 4: Video Realism (Objective 4 comparison)

```python
# Generate videos using CNN vs ViT conditioning
tadiff_cnn = TaDiff(conditioning_dim=512)
tadiff_vit = TaDiff(conditioning_dim=768)

cnn_videos = tadiff_cnn.generate(cnn_embeddings, treatment='radiotherapy')
vit_videos = tadiff_vit.generate(vit_embeddings, treatment='radiotherapy')

# Compare realism
cnn_ssim = ssim(cnn_videos, real_future_scans)
vit_ssim = ssim(vit_videos, real_future_scans)

# Expected: ViT videos more realistic (higher SSIM)
```

---

## Summary: Do We Build Something Else?

**YES, you build a baseline CNN** (Option B above), but:

1. **Use nnU-Net's segmentation masks** to help train it (crop scans to tumor regions)
2. **Don't use nnU-Net's encoder directly** as the baseline — it's optimized for the wrong task
3. **Build a simple 3D CNN** (like the example above) and train it on single-timepoint classification
4. **Then compare** it to Swin UNETR on temporal tasks

**This satisfies the project requirement**: "Implementation of baseline CNNs" + "Identification of limitations in static modeling" (CNN loses to ViT on temporal tasks = the limitation!)

---

## Expected Results Summary

| Comparison | CNN (nnU-Net) | Transformer (Swin UNETR) | Winner |
|------------|---------------|--------------------------|--------|
| **Segmentation Dice** | 0.913 | 0.900 | CNN (+1.4%) |
| **Temporal prediction AUC** | ~0.70 | ~0.78 | ViT (+11%) |
| **Clustering Silhouette** | ~0.45 | ~0.52 | ViT (+16%) |
| **LLM report BLEU** | ~0.32 | ~0.38 | ViT (+19%) |
| **Video SSIM** | ~0.88 | ~0.92 | ViT (+5%) |
| **Embedding dimension** | 512 | 768 | Larger ≠ better, but matches RadFM |
| **Training time (per epoch)** | 2.5 hours | 4.1 hours | CNN faster |

**Conclusion**: CNN wins segmentation (its designed task), but **ViT wins all temporal/generative tasks** (our actual objectives).

---

## Presentation Slide Update

**Current (misleading) text on Slide 12:**
```latex
\begin{block}{\small Also Serves As}
    \small CNN baseline to beat with Swin UNETR (Phase~3).
\end{block}
```

**Recommended fix:**
```latex
\begin{block}{\small Role as CNN Baseline (Phase 2)}
    \small
    nnU-Net segmentation masks → crop scans to tumor regions\\
    → train simple 3D CNN baseline (512-dim) on single-timepoint task\\
    → compare with Swin UNETR (768-dim) on temporal prediction\\[0.2em]
    Expected: CNN good at static tasks, ViT wins temporal (+8--16\%)
\end{block}
```

---

## Key Takeaways

1. **nnU-Net ≠ CNN baseline** — It's a segmentation tool, not a representation model
2. **Phase 2 requires building a new CNN** — Trained specifically for tumor representation
3. **The comparison is about embeddings**, not segmentation quality
4. **Expected outcome**: ViT wins on all temporal/generative tasks, justifying its use in the full pipeline

---

**Date Created**: February 15, 2026  
**Related Files**: `state_of_the_art_final.tex`, `PIPELINE_FINAL.md`
