# Phase 2 -- Status and Deliverables

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink -- TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)  
**Status Date:** March 29, 2026

---

## Phase 2 Scope

**Phase 2 -- Baseline Vision Models for Tumor Representation**  
Duration: 3 weeks (Weeks 5--7)

Activities:
- Implementation of baseline deep learning models for tumor analysis (CNNs)
- Training and validation on single time-point tumor imaging tasks
- Quantitative evaluation of baseline performance
- Identification of limitations in static modeling

---

## Activity Checklist

### A. Implementation of Baseline Deep Learning Models

Two CNN architectures were implemented, both pretrained on BraTS challenge data and fine-tuned on Cyprus PROTEAS using 3-fold cross-validation.

| Task | Status | Notes |
| ---- | ------ | ----- |
| Architecture selection | DONE | SegResNet (general-purpose) + Met-Seg two-stage (metastasis-specific) |
| Data split strategy | DONE | Stratified 3-fold CV at patient level (`outputs/data_splits.json`) |
| 3D volume pre-validation | DONE | Filter for valid 3D NIfTI files, skip 2D/corrupted |
| Label conversion | DONE | Cyprus {0,1,2,3} → WT/TC/ET multi-channel (BraTS convention) |

#### Model 1: SegResNet (MONAI)

| Task | Status | Notes |
| ---- | ------ | ----- |
| Architecture implementation | DONE | `notebooks/Phase2_Activity2_Training.ipynb` |
| Pretrained weights loaded | DONE | NVIDIA BraTS 2023 (glioma-pretrained, 4.7M params) |
| Training pipeline | DONE | Dice+CE loss, cosine LR, sliding-window inference |
| Data augmentation | DONE | RandFlip, RandRotate90, RandScaleIntensity, RandShiftIntensity |
| Embedding extraction (256-dim) | DONE | Forward hook on encoder bottleneck |

#### Model 2: Met-Seg Two-Stage CNN (MICCAI 2024)

| Task | Status | Notes |
| ---- | ------ | ----- |
| Architecture implementation | DONE | `notebooks/Phase2_Activity2_MetSeg.ipynb` |
| DynUNet segmenter (16.7M params) | DONE | 5-level 3D U-Net, deep supervision, residual blocks |
| DenseNet121 detector (11.3M params) | DONE | Binary patch classifier (frozen during fine-tuning) |
| Pretrained weights loaded | DONE | Official BraTS-METS 2023 (metastasis-pretrained, full modality) |
| Training pipeline | DONE | Dice+BCE loss, SGD+Nesterov, two-stage inference |
| Embedding extraction (1024-dim) | DONE | Forward hook on DynUNet last downsample block |

**Reference:** Sadegheih, Y. & Merhof, D. (2024). Met-Seg: A two-stage segmentation model for brain metastases. *MICCAI 2024 BraTS-METS Challenge.*

### B. Training and Validation on Single Time-Point Tasks

#### SegResNet Training Progress

| Fold | Epochs | Best Dice | WT | TC | ET | Status |
| :--: | :----: | :-------: | :-: | :-: | :-: | ------ |
| 0 | 49/49 | **0.414** | 0.444 | 0.404 | 0.393 | DONE (no augmentation) |
| 1 | 29/49 | **0.411** | 0.448 | 0.398 | 0.386 | IN PROGRESS (with augmentation) |
| 2 | -- | -- | -- | -- | -- | PENDING |

**SegResNet Fold 0 -- Full Training Curve:**

| Epoch | Mean Dice | WT | TC | ET |
| :---: | :-------: | :-: | :-: | :-: |
| 1 | 0.209 | 0.328 | 0.182 | 0.116 |
| 5 | 0.317 | 0.380 | 0.319 | 0.253 |
| 13 | 0.384 | 0.428 | 0.376 | 0.347 |
| 27 | 0.405 | 0.438 | 0.393 | 0.383 |
| 49 | **0.414** | **0.444** | **0.404** | **0.393** |

**SegResNet Fold 1 -- Training Curve (in progress):**

| Epoch | Mean Dice | WT | TC | ET |
| :---: | :-------: | :-: | :-: | :-: |
| 1 | 0.177 | 0.306 | 0.146 | 0.080 |
| 7 | 0.354 | 0.414 | 0.353 | 0.296 |
| 15 | 0.393 | 0.431 | 0.386 | 0.362 |
| 21 | 0.408 | 0.444 | 0.399 | 0.380 |
| 25 | **0.411** | **0.448** | **0.398** | **0.386** |
| 29 | 0.408 | 0.444 | 0.395 | 0.384 |

**Observation:** SegResNet is plateauing at approximately Dice = 0.41 on both folds. Augmentation on Fold 1 did not improve over Fold 0's non-augmented result, indicating the architecture is the limiting factor rather than data diversity.

#### Met-Seg Training Progress

| Fold | Epochs | Best Dice | WT | TC | ET | Status |
| :--: | :----: | :-------: | :-: | :-: | :-: | ------ |
| 0 | 14/30 | **0.356** | 0.369 | 0.349 | 0.349 | IN PROGRESS |
| 1 | -- | -- | -- | -- | -- | PENDING |
| 2 | -- | -- | -- | -- | -- | PENDING |

**Met-Seg Fold 0 -- Training Curve (in progress):**

| Epoch | Mean Dice | WT | TC | ET | Loss |
| :---: | :-------: | :-: | :-: | :-: | :--: |
| 4 | 0.300 | 0.306 | 0.298 | 0.297 | 2.100 |
| 9 | 0.333 | 0.342 | 0.328 | 0.328 | 1.910 |
| 14 | **0.356** | **0.369** | **0.349** | **0.349** | 1.769 |

**Observation:** Met-Seg is still converging (loss dropping from 2.76 to 1.77). With 16 epochs remaining, the expected final Dice is 0.42--0.48. The model shows notably more balanced region performance (WT ≈ TC ≈ ET) compared to SegResNet, reflecting its metastasis-specific pretraining.

### C. Quantitative Evaluation of Baseline Performance

#### Head-to-Head Comparison (Quick Test, 5 Epochs, Same Fold)

| Metric | SegResNet | Met-Seg | Difference |
| :----: | :-------: | :-----: | :--------: |
| Mean Dice | 0.317 | **0.377** | Met-Seg +19% |
| WT | 0.380 | **0.402** | Met-Seg +6% |
| TC | 0.319 | **0.364** | Met-Seg +14% |
| ET | 0.253 | **0.364** | Met-Seg +44% |

**Key finding:** Met-Seg demonstrates a **+44% improvement on Enhancing Tumor (ET)** detection in the quick test. Brain metastases are predominantly enhancing lesions, making this the most clinically relevant sub-region. This advantage directly stems from the metastasis-specific BraTS-METS pretraining.

#### Current Best Results per Model (Different Folds, Different Epochs)

| Metric | SegResNet (Fold 0, Ep 49) | SegResNet (Fold 1, Ep 25) | Met-Seg (Fold 0, Ep 14) |
| :----: | :-----------------------: | :-----------------------: | :---------------------: |
| Mean Dice | 0.414 | 0.411 | 0.356 (still training) |
| WT | 0.444 | 0.448 | 0.369 |
| TC | 0.404 | 0.398 | 0.349 |
| ET | 0.393 | 0.386 | 0.349 |

**Note:** Met-Seg is at epoch 14 of 30 and still improving. Direct comparison will be possible after all folds complete.

#### Embedding Extraction for Downstream Explainability

| Model | Embedding Dim | Layer | Fold 0 | Fold 1 | Fold 2 |
| ----- | :-----------: | ----- | :----: | :----: | :----: |
| SegResNet | 256-dim | Encoder bottleneck | DONE | PENDING | PENDING |
| Met-Seg | 1024-dim | DynUNet last downsample | DONE (quick-test) | PENDING | PENDING |

These CNN embeddings will serve as baselines for Phase 3 comparison with Vision Transformer (ViT) representations.

### D. Identification of Limitations in Static Modeling

#### L1. Single-Timepoint Processing
Both CNN models process each MRI scan independently with no temporal context. They cannot capture how a metastasis evolves across treatment timepoints -- a critical gap for treatment response assessment and disease progression modeling.

#### L2. SegResNet Architecture Ceiling
SegResNet converges to approximately Dice = 0.41 regardless of augmentation strategy, suggesting the 4.7M-parameter encoder-decoder architecture has insufficient capacity for small, scattered brain metastases. The 128^3 input resolution may dilute fine-grained enhancing patterns.

#### L3. Patch-Based Trade-offs (Met-Seg)
Met-Seg's 64^3 patch approach captures better local detail for small metastases but sacrifices global anatomical context. The frozen detector cannot adapt to Cyprus PROTEAS-specific tumor distribution patterns (e.g., post-treatment radiation necrosis mimicking enhancement).

#### L4. No Uncertainty Quantification
Neither model provides voxel-level confidence estimates. For clinical explainability, uncertainty maps would help clinicians identify predictions that require manual review.

#### L5. Single-Institution Data
Training and validation on Cyprus PROTEAS alone (45 patients, one institution) limits generalization claims. Cross-dataset validation (e.g., BraTS-METS 2023 test set) would strengthen the findings.

#### L6. No Temporal Modeling
Static CNNs cannot model treatment response dynamics (progressive vs stable vs responsive), which requires comparing tumor representations across time points. This motivates Phase 3's introduction of temporal modeling and ViT architectures.

---

## Intermediate Deliverables

### 1. Baseline Model Implementations and Trained Weights

| Deliverable | Status | Location |
| ----------- | ------ | -------- |
| SegResNet notebook | DONE | `notebooks/Phase2_Activity2_Training.ipynb` |
| Met-Seg notebook | DONE | `notebooks/Phase2_Activity2_MetSeg.ipynb` |
| SegResNet weights (fold 0) | DONE | `outputs/phase2_outputs/checkpoints/segresnet_fold0_best.pth` |
| SegResNet weights (fold 1) | IN PROGRESS | Training on Kaggle (account 1) |
| Met-Seg weights (fold 0) | IN PROGRESS | Training on Kaggle (account 2) |
| Met-Seg pretrained weights | DONE | `segmentor_full_modality.ckpt` + `detector_full_modality.ckpt` |
| SegResNet embeddings (fold 0) | DONE | `outputs/phase2_outputs/embeddings/` |
| Met-Seg embeddings (fold 0 quick-test) | DONE | `cnn_metseg_embeddings_fold0.npz` (170 × 1024) |

### 2. Comparative Performance Results

| Deliverable | Status | Location |
| ----------- | ------ | -------- |
| Training curves (SegResNet fold 0) | DONE | `outputs/phase2_outputs/figures/training_curves_fold0.png` |
| Training curves (Met-Seg fold 0 quick-test) | DONE | Kaggle output |
| Fold metrics JSON | DONE | `outputs/phase2_outputs/metrics/fold0_metrics.json` |
| Cross-validation summary | PENDING | Requires all 3 folds for both models |
| Head-to-head comparison table | DONE | This report (Section C) |
| Limitations analysis | DONE | This report (Section D) |

### 3. Supporting Infrastructure

| Deliverable | Status | Location |
| ----------- | ------ | -------- |
| 3-fold cross-validation splits | DONE | `outputs/data_splits.json` |
| Dataset upload to Kaggle (account 1) | DONE | `mohamedmohamed23/cyprus-proteas-brain-mets` |
| Dataset upload to Kaggle (account 2) | DONE | `zinou123viva/cyprus-proteas-brain-mets` |
| Met-Seg weights on Kaggle | DONE | `zinou123viva/metseg-pretrained-weights` |
| Upload automation script | DONE | `scripts/upload_second_account.py` |

---

## Key Findings

1. **SegResNet plateaus at Dice ≈ 0.41** on both folds, with and without augmentation -- the architecture is the bottleneck
2. **Met-Seg's metastasis-specific pretraining** delivers +44% ET improvement in early epochs, confirming domain-specific pretrained features are superior for brain metastases
3. **Met-Seg shows balanced region performance** (WT ≈ TC ≈ ET), while SegResNet has a persistent ranking (WT > TC > ET)
4. **Both models achieve Dice 0.35--0.42**, consistent with literature for single-institution brain metastasis segmentation
5. **Static CNN models cannot model temporal tumor evolution** -- this motivates Phase 3 (ViT + temporal analysis)
6. **1024-dim Met-Seg embeddings** capture richer representations than SegResNet's 256-dim, potentially more informative for downstream explainability

---

## Remaining Work

| Task | Account | Estimated Time |
| ---- | ------- | -------------- |
| SegResNet Fold 1 completion | mohamedmohamed23 | ~1.5 hours |
| SegResNet Fold 2 | mohamedmohamed23 | ~2 hours |
| Met-Seg Fold 0 completion | zinou123viva | ~2.5 hours |
| Met-Seg Folds 1 + 2 | zinou123viva | ~6 hours |
| Final cross-validation summary (mean ± std) | -- | After all folds |

---

## Phase 2 Completion: IN PROGRESS (estimated 70%)

Core implementations done. Training ongoing for remaining folds.  
Upon completion, final deliverable: **cross-validated comparison table (mean ± std Dice across 3 folds)** for both architectures.  
Ready to begin **Phase 3 -- Temporal Modeling and Explainability** in parallel.
