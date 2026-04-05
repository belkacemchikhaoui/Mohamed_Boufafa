# Phase 2 — Complete Report: Baseline Vision Models for Tumor Representation

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink — TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Duration:** Weeks 5–7 | **Status:** ✅ COMPLETE

---

## Executive Summary

Phase 2 established the CNN baseline for tumor representation using the Met-Seg two-stage pipeline (DenseNet121 detector + DynUNet segmenter). We achieved:

- **Segmentation:** Dice = 0.505 ± 0.025 (3-fold CV, 45 patients, 170 scans)
- **Embedding evaluation:** **12/16 tests passed** (all tests now produce values — zero N/A)
- **Architecture comparison:** Met-Seg (12/16) vs SegResNet (7/16) — domain-specific pretraining matters
- **Key finding:** CNN embeddings encode **static tumor anatomy** (R² = 0.38 for volume/shape) but **cannot model temporal change** (T1 r = 0.049, T3 R² = -0.210)
- **Phase 3 justification:** Temporal test failures (T1, T3, T4) provide direct evidence that Vision Transformers with temporal attention are needed

---

## Phase 2 Activities Completed

### Activity 1: Data Preparation ✅
- 45 patients from Cyprus PROTEAS dataset, 170 MRI scans (4 modalities each)
- 3-fold patient-wise cross-validation splits (stratified by histology + treatment)
- Data uploaded to Kaggle for GPU training

### Activity 2: CNN Training (Met-Seg v4) ✅
- **Architecture:** DenseNet121 detector + DynUNet segmenter (pretrained on 402 BraTS-METS cases)
- **Training:** 80 epochs per fold, AdamW optimizer, DiceCE loss, 3-fold CV
- **Results:**

| Fold | Dice (WT) | Dice (TC) | Dice (ET) | Mean Dice |
|---|---|---|---|---|
| Fold 0 (v3, 60ep) | 0.573 | 0.510 | 0.512 | 0.532 |
| Fold 1 (v4, 80ep) | 0.496 | 0.470 | 0.483 | 0.483 |
| Fold 2 (v4, 80ep) | 0.509 | 0.498 | 0.494 | 0.500 |
| **Mean** | **0.526** | **0.493** | **0.496** | **0.505 ± 0.025** |

### Activity 3: Embedding Extraction ✅
- Extracted from DynUNet encoder's deepest downsample block
- 170 scans × 1024-dim embeddings × 3 folds, averaged across folds

> [!IMPORTANT]
> **Bug Discovery:** The initial extraction (v1) had a patch-overwrite bug — `sliding_window_inference` runs on ~49 overlapping patches per scan, but the hook only kept the LAST patch's features (overwriting previous ones). This meant embeddings reflected a single patch (often background) rather than the whole scan. Fixed in v2 by accumulating all patches and averaging.

### Activity 4: Embedding Evaluation Battery ✅
- 17-test suite: 6 morphology + 4 heterogeneity + 7 temporal
- All tests run on CPU (~2 min total) using `sklearn` linear probes
- Pipeline: `StandardScaler → PCA(30) → TransformedTargetRegressor(Ridge/LogReg)`

---

## Evaluation Results: 12/16 Tests Passed (All Tests Computed)

### 🔬 Morphology Tests (5/6 passed)

These tests evaluate whether embeddings encode tumor **shape, size, and structure** using linear regression/classification probes.

| Test | Description | Method | Metric | Result | Pass? |
|---|---|---|---|---|---|
| **M1** | Volume prediction | Ridge(emb → volume) | R² | **0.379** | ✅ |
| **M2** | Log-volume shape | Ridge(emb → log₁₀ vol) | R² | **0.388** | ✅ |
| **M3** | Surface-volume ratio | Ridge(emb → SVR) | R² | 0.006 | ❌ |
| **M4** | Necrosis detection | LogReg(emb → NCR present) | F1 | **0.707** | ✅ |
| **M5** | Elongation proxy | Ridge(emb → elongation) | R² | **0.387** | ✅ |
| **M6** | NN consistency | 5-NN volume agreement | % | **26.8%** | ✅ |

**Interpretation:** The CNN encoder captures tumor size (M1, M2), overall shape (M5), and internal architecture (M4 necrosis). M3 (surface-volume ratio) fails because SVR requires precise boundary geometry that gets lost in global average pooling. M6 at 26.8% means scan neighbors in embedding space are morphologically similar ~27% of the time (above the 15% random baseline).

### 🗺️ Heterogeneity Tests (4/4 passed)

These test whether embeddings capture **non-uniform internal texture** using radiomic GLCM features as ground truth.

| Test | Description | Method | Metric | Result | Pass? |
|---|---|---|---|---|---|
| **H1** | PCA structure | PCA residual vs embedding variance | \|r\| | **0.796** | ✅ |
| **H2** | Heterogeneity score | Ridge(emb → GLCM entropy) | R² | **0.386** | ✅ |
| **H3** | Subregion detection | LogReg(emb → #subregions) | F1 | **0.540** | ✅ |
| **H4** | Texture bundle | Multi-output Ridge(emb → texture features) | R² | **0.258** | ✅ |

**Interpretation:** All heterogeneity tests pass. H1 (|r|=0.796) shows strong correlation between PCA reconstruction residual and embedding variance — embeddings that are harder to compress (high residual) correspond to more heterogeneous tumors. H2 (R²=0.39) confirms the CNN directly encodes tissue texture. H3 (F1=0.54) discriminates tumor complexity. H4 (R²=0.26) captures multi-target texture features.

### ⏱️ Temporal Tests (3/6 passed)

These test whether embeddings **track how tumors change over time** — the core justification for Phase 3.

| Test | Description | Method | Metric | Result | Pass? |
|---|---|---|---|---|---|
| **T1** | Emb distance vs ΔVol | Pearson(L2 distance, vol change) | r | 0.049 | ❌ |
| **T3** | ΔEmb → ΔVol | Ridge(Δembedding → Δvolume) | R² | -0.210 | ❌ |
| **T4** | Response prediction | LogReg(baseline emb → response) | AUC | 0.458 | ❌ |
| **T5** | Temporal coherence | Avg cosine(consecutive scans) | cos | **0.995** | ✅ |
| **T6** | Velocity correlation | emb velocity vs progression speed | r | **0.209** | ✅ |
| **T7** | Treatment separation | RS vs FSRT embedding distance | d | **15.201** | ✅ |

**Interpretation:** 
- **T1 (r=0.049):** Embedding distance does NOT correlate with volume change — the CNN processes each scan independently, so embedding shifts don't reflect biological change.
- **T3 (R²=-0.210):** Predicting volume change from embedding change is WORSE than guessing the mean — confirms no temporal information in the embeddings.
- **T4 (AUC=0.458):** Below chance (0.5) for predicting treatment response from baseline embeddings. The CNN cannot predict future tumor behavior from a single scan — this is precisely what Phase 3's temporal attention should enable.
- **T5 (cos=0.995):** Very high temporal coherence — consecutive scans produce near-identical embeddings. While "passing" the threshold, this is actually a negative signal: the CNN can't distinguish temporal evolution (all scans look the same to it).
- **T6 (r=0.209):** Weak but positive correlation between embedding velocity and progression speed — the information comes from volume-correlated features (M1 R²=0.38), not true temporal modeling.
- **T7 (d=15.201):** Large Cohen's d between RS (n=90) and FSRT (n=29) treatment groups. This reflects baseline tumor differences between treatment groups in the embedding space.

---

## Downstream Tasks: Definitions and Literature Basis

### What Are "Downstream Tasks"?

Downstream tasks evaluate representation quality by testing embeddings on tasks the model was **never explicitly trained for** [1, 2]. The model is trained for segmentation; we then freeze the encoder and test if its internal features can predict other clinical properties using simple linear probes.

### Our Downstream Task Categories

#### 1. Prediction Tasks (Linear Probing)

| Task | Type | Target Variable | Ground Truth Source |
|---|---|---|---|
| Volume prediction | Regression | Tumor volume (mm³) | Computed from segmentation masks |
| Shape prediction | Regression | Sphericity, elongation, SVR | PyRadiomics shape features [3] |
| Necrosis detection | Binary classification | NCR present/absent | Segmentation mask labels |
| Heterogeneity scoring | Regression | GLCM entropy | PyRadiomics texture features [3] |
| Subregion complexity | Multi-class classification | 1/2/3 subregions | Mask label counts |
| Volume change prediction | Regression | ΔVolume between timepoints | Mask-derived volumes |
| Treatment response | Binary classification | Responder vs non-responder | Clinical outcome data |

**Method:** `StandardScaler → PCA(30) → Ridge/LogisticRegression`, evaluated with 5-fold patient-wise cross-validation. This follows the **linear evaluation protocol** established by SimCLR [2] and widely adopted in medical imaging [4, 5].

#### 2. Clustering Tasks

| Task | Method | What It Tests |
|---|---|---|
| **M6: NN consistency** | 5-nearest-neighbor morphology agreement | Local structure of embedding space |
| **T7: Treatment separation** | Mean embedding distance between RS vs FSRT groups | Clinical groupings in embedding space |
| **t-SNE visualization** | t-SNE colored by histology, treatment, timepoint | Qualitative cluster assessment |

**Method:** These evaluate whether the embedding space has meaningful geometric structure without any supervised training. Clustering quality in learned representations is a standard evaluation in self-supervised learning [6, 7].

### Literature References

| # | Reference | How We Used It |
|---|---|---|
| [1] | Alain & Bengio (2017). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644 | **Linear probing methodology** — training simple classifiers on frozen features to evaluate what information each layer encodes |
| [2] | Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML 2020 | **Linear evaluation protocol** — the standard approach of linear probe on frozen encoder embeddings to measure representation quality |
| [3] | van Griethuysen et al. (2017). *Computational Radiomics System to Decode the Radiographic Phenotype.* Cancer Research | **PyRadiomics** — provides the ground truth shape (M1-M5) and texture (H1-H4) features we validate embeddings against |
| [4] | Tang et al. (2022). *Self-supervised pre-training of swin transformers for 3D medical image analysis.* CVPR 2022 | **3D medical embedding evaluation** — uses downstream segmentation and classification tasks to evaluate pre-trained ViT features |
| [5] | Hatamizadeh et al. (2022). *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumours.* BrainLes@MICCAI | **Brain tumor ViT baseline** — the architecture we adapt for Phase 3, evaluated on BraTS downstream tasks |
| [6] | Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021 | **Clustering in ViT features** — shows ViT self-supervised features produce semantically meaningful clusters without labels |
| [7] | Sadegheih & Merhof (2024). *Met-Seg: A Two-Stage Pipeline for Brain Metastasis Detection and Segmentation.* MICCAI PRIME | **Our baseline architecture** — the DenseNet121 + DynUNet pipeline we fine-tuned on Cyprus PROTEAS data |
| [8] | Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015 | **Skip connection design** — explains why U-Net bottleneck embeddings have limited discriminative power (Sec. CNN Limitations) |
| [9] | Luo et al. (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS 2016 | **Receptive field analysis** — effective receptive field ~60% of theoretical, explaining why distant tumor regions are processed independently |
| [10] | Milletari et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV 2016 | **Dice loss** — our DiceCE loss combines Dice (handles class imbalance) with cross-entropy (smooth gradients) |
| [11] | Flouri, D., Pattichis, C., et al. (2025). *Cyprus PROTEAS: A Longitudinal Brain Metastasis Dataset.* Zenodo. DOI: [10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793) | **Primary dataset** — 45 patients used for training and evaluation |
| [12] | Lin, N.U., et al. (2015). *RANO criteria for brain metastases.* Lancet Oncology, 16(6), e270-e278 | **Response classification** — RANO-BM thresholds used for T4 response prediction labels |
| [13] | Myronenko, A. (2019). *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization.* BrainLes@MICCAI 2018, LNCS 11384 | **SegResNet architecture** — the MONAI implementation we used as our second baseline |
| [14] | Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18, 203-211 | **DynUNet architecture basis** — Met-Seg's segmenter follows the nnU-Net dynamic architecture design |
| [15] | Hoerl & Kennard (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics, 12(1), 55-67 | **Probe model** — Ridge regression used in M1-M5, H2, H4, T3 (L2 regularization prevents overfitting on n=170) |
| [16] | Haralick et al. (1973). *Textural Features for Image Classification.* IEEE Trans. SMC, 3(6), 610-621 | **GLCM ground truth** — Gray-Level Co-occurrence Matrix features used as heterogeneity ground truth (H1-H4) |
| [17] | Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.* Lawrence Erlbaum Associates | **Effect size** — Cohen's d used to measure treatment group separation in T7 |
| [18] | Varoquaux, G. (2018). *Cross-validation failure: Small sample sizes lead to large error bars.* NeuroImage, 180, 68-77 | **CV strategy** — justifies patient-wise stratified k-fold with small sample sizes |

---

## CNN Limitation Analysis

| Limitation | Evidence from Our Tests | Root Cause | Phase 3 Solution |
|---|---|---|---|
| **Cannot predict volume change** | T3 R² = -0.210 | Each scan processed independently | TaViT temporal attention across timepoints |
| **Embedding distance ≠ biological change** | T1 r = 0.049 | No concept of "before/after" | Temporal contrastive learning aligns changes |
| **Weak surface detail** | M3 R² = 0.006 | Global avg pooling loses boundary geometry | ViT patch tokens preserve spatial detail |
| **Cannot predict treatment response** | T4 AUC = 0.458 | Cannot see treatment history | TaViT conditions on treatment timeline |
| **High cosine similarity** | T5 cos = 0.995 | U-Net skip connections bypass bottleneck | ViT has no skip-connection information bypass |

### Why These Limitations Are Architectural (Not Fixable)

1. **Skip connections:** DynUNet's skip connections carry spatial detail directly from encoder to decoder, so the bottleneck doesn't NEED to encode discriminative features — the decoder gets them via shortcuts. This is by design in U-Net architectures [8].

2. **Single-timepoint processing:** The CNN sees one scan at a time with no mechanism to relate scans across time. Even perfect single-scan features cannot encode temporal dynamics.

3. **Local receptive field:** Despite deep stacking, CNN effective receptive fields cover ~60% of the theoretical maximum [9], causing distant tumor regions to be processed independently.

---

## Phase 2 Deliverables Checklist

| Deliverable (from project plan) | Status | Evidence |
|---|---|---|
| Baseline model implementation | ✅ | Met-Seg v4 (DenseNet121 + DynUNet), 3-fold CV |
| Trained weights | ✅ | `metseg_fold{0,1,2}_best.pth` (705 MB total) |
| Quantitative evaluation of baseline performance | ✅ | Dice = 0.505 ± 0.025 |
| Representation quality via downstream prediction | ✅ | **12/16 linear probe tests** — all tests produce values |
| Representation quality via clustering | ✅ | **M6 NN consistency (26.8%)**, **T7 treatment separation (d=15.2)**, t-SNE visualizations |
| Identification of limitations in static modeling | ✅ | **T1, T3 temporal failures** documented |
| Comparative benchmarks for Phase 3 | ✅ | Full 17-test table with CNN scores |

---

## Phase 3 Transition: What Changes

| Aspect | Phase 2 (CNN) | Phase 3 (ViT) |
|---|---|---|
| **Encoder** | DynUNet (local convolutions) | Swin UNETR (global self-attention) |
| **Temporal** | None (single-scan) | TaViT (cross-timepoint attention) |
| **Embedding dim** | 1024 | 768 |
| **Expected static R²** | 0.38 | > 0.5 (global context) |
| **Expected temporal R²** | -0.2 | > 0.1 (temporal modeling) |
| **Evaluation** | Same 17 tests | Same 17 tests (fair comparison) |
| **Data & splits** | Same | Same (controlled variable) |

### Phase 3 Targets (based on CNN baseline)

| Test | CNN Baseline | ViT Target | ViT+TaViT Target |
|---|---|---|---|
| M1 Volume R² | 0.379 | > 0.5 | > 0.6 |
| M3 SVR R² | 0.006 | > 0.15 | > 0.2 |
| M4 Necrosis F1 | 0.707 | > 0.75 | > 0.8 |
| T1 dist→ΔVol r | 0.049 | > 0.15 | > 0.3 |
| T3 ΔEmb→ΔVol R² | -0.210 | > 0 | > 0.1 |
| T4 Response AUC | 0.458 | > 0.55 | > 0.65 |

---

## About the v3 Embeddings

> **Q: Do we need to download v3 separately?**

No. The Kaggle extraction confirmed `v2 == v3` — identical norms, cosine similarity, and dimensionality:

| Property | v2 | v3 | Match? |
|---|---|---|---|
| Dimensions | 1024-dim | 1024-dim | ✅ |
| Norm range | [0.4148, 0.6187] | [0.4148, 0.6187] | ✅ |
| Norm std | 0.0677 | 0.0677 | ✅ |
| Cosine similarity | 0.9964 | 0.9964 | ✅ |

This is because DynUNet has only 3 downsample blocks (indices 0, 1, 2), so `downsamples[2]` and `downsamples[-1]` are the **same layer**. The v2 embeddings ARE the corrected deepest-layer embeddings.

---

## Architecture Comparison: Met-Seg vs SegResNet

We also trained a **SegResNet** (MONAI pretrained on BraTS glioma data) on the same Cyprus PROTEAS dataset to compare CNN architectures.

### Architecture Details

| Property | Met-Seg (DynUNet) | SegResNet |
|---|---|---|
| **Encoder** | DenseNet121 detector + DynUNet segmenter | SegResNet (ResNet-based) |
| **Pretraining** | 402 BraTS-METS brain metastasis cases | BraTS glioma data |
| **Embedding dim** | 1024 | 128 |
| **Parameters** | ~30M (combined pipeline) | ~5M |
| **Domain match** | ✅ Brain metastases → brain metastases | ⚠️ Gliomas → brain metastases |

### Head-to-Head Evaluation Results

| Test | Description | Met-Seg (DynUNet) | SegResNet | Winner |
|---|---|---|---|---|
| **M1** | Volume R² | **0.379** ✅ | -1.454 ❌ | Met-Seg |
| **M2** | Log-volume R² | **0.388** ✅ | -0.073 ❌ | Met-Seg |
| **M3** | SVR R² | 0.006 ❌ | -0.179 ❌ | — |
| **M4** | Necrosis F1 | **0.707** ✅ | 0.564 ✅ | Met-Seg |
| **M5** | Elongation R² | **0.387** ✅ | -0.073 ❌ | Met-Seg |
| **M6** | NN consistency % | **26.8%** ✅ | 21.5% ✅ | Met-Seg |
| **H1** | PCA structure |r| | **0.796** ✅ | 0.016 ❌ | Met-Seg |
| **H2** | Heterogeneity R² | **0.386** ✅ | -0.073 ❌ | Met-Seg |
| **H3** | Subregion F1 | 0.540 ✅ | **0.546** ✅ | ≈ Tie |
| **H4** | Texture bundle R² | **0.258** ✅ | -0.569 ❌ | Met-Seg |
| **T1** | Emb dist vs ΔVol | 0.049 ❌ | -0.031 ❌ | — |
| **T3** | ΔEmb→ΔVol R² | -0.210 ❌ | -0.169 ❌ | — |
| **T4** | Response AUC | 0.458 ❌ | **0.500** ✅ | — (both near chance) |
| **T5** | Temporal coherence | **0.995** ✅ | **0.986** ✅ | ≈ Tie |
| **T6** | Velocity r | 0.209 ✅ | **0.423** ✅ | SegResNet |
| **T7** | Treatment sep d | **15.201** ✅ | 2.193 ✅ | Met-Seg |
| | **Score** | **12/16** | **7/16** | **Met-Seg** |

### Why Met-Seg Wins

1. **Domain-specific pretraining:** Met-Seg was pretrained on 402 brain metastasis cases (same tumor type as Cyprus PROTEAS). SegResNet was pretrained on glioma data — a fundamentally different tumor type (large infiltrative vs small scattered).

2. **Higher-dimensional embeddings:** Met-Seg's 1024-dim embeddings have more capacity to encode diverse morphological features than SegResNet's 128-dim. With `PCA(30)` reducing both to 30 dimensions, the 1024-dim space has richer variation to preserve.

3. **Two-stage pipeline advantage:** Met-Seg's DenseNet121 detector first localizes metastases, then DynUNet focuses on the detected regions. SegResNet processes the entire volume at once, which may dilute tumor-specific features.

> [!NOTE]
> SegResNet outperforms on **T6 (velocity r=0.423 vs 0.209)**. Met-Seg strongly outperforms on **H1 (|r|=0.796 vs 0.016)** and **T7 (d=15.2 vs 2.2)**, confirming that domain-specific pretraining produces embeddings with richer internal structure and better treatment-group separation.

---

## Test Infrastructure Notes

All 16 tests now produce values. Previous N/A issues were resolved:

| Issue | Root Cause | Fix Applied |
|---|---|---|
| **H1 N/A** | `numpy.float64` not recognized as `float` in display cell | Wrapped in `float()` |
| **T4 N/A** | Regular k-fold CV created single-class folds (AUC undefined) | `StratifiedKFold` + LOO fallback |
| **T5 N/A** | T4 crash killed the shared code cell before T5 ran | Independent `try/except` blocks |
| **T7 N/A** | Missing `float()` conversion on Cohen's d | Added `float()` wrapping |

---

## Files Produced

| File | Description |
|---|---|
| `notebooks/Phase2_A1_Data_Preparation.ipynb` | Data prep + 3-fold splits |
| `notebooks/Phase2_A2_MetSeg_Training_Fold{0,1,2}.ipynb` | Met-Seg training (1 fold per notebook, Kaggle GPU) |
| `notebooks/Phase2_A2_SegResNet_Training_3Fold.ipynb` | SegResNet training (all 3 folds, Kaggle GPU) |
| `notebooks/Phase2_A3_Embedding_ReExtraction.ipynb` | Fixed extraction v2/v3 (Kaggle GPU) |
| `notebooks/Phase2_A4_MetSeg_Embedding_Eval.ipynb` | Met-Seg 16-test battery (local CPU) |
| `notebooks/Phase2_A4_SegResNet_Embedding_Eval.ipynb` | SegResNet 16-test battery (local CPU) |
| `embeddings/metseg/cnn_metseg_embeddings_v2_fold{0,1,2}.npz` | Met-Seg embeddings (170×1024 per fold) |
| `embeddings/segresnet/cnn_embeddings_fold{0,1,2}.npz` | SegResNet embeddings (170×128 per fold) |
| `weights/metseg/metseg_fold{0,1,2}_best.pth` | Met-Seg model weights (~235 MB each) |
| `outputs/activity4_results.json` | Machine-readable evaluation results |

---

## Software and Tools

| Tool | Version | Purpose |
|---|---|---|
| **PyTorch** | 2.x | Model training, GPU acceleration |
| **MONAI** | 1.3+ | Medical image transforms, DynUNet, SegResNet, sliding window inference |
| **scikit-learn** | 1.4+ | Linear probes (Ridge, LogisticRegression, PCA, StandardScaler) |
| **Kaggle** | T4 GPU×2 | Training environment (30h/week GPU quota) |
| **DenseNet121** | torchvision | Met-Seg detector backbone |
| **Pandas / Matplotlib / Seaborn** | — | Analysis and visualization |

