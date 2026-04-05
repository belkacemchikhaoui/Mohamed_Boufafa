# Progress Report — Phases 1 & 2 Complete

**Student:** Mohamed Boufafa  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Program:** Mitacs Globalink — TÉLUQ University  
**Project:** Explainable Disease Progression & Counterfactual Video Generation  
**Date:** April 5, 2026

---

## Summary

Phases 1 and 2 are complete. The Cyprus PROTEAS brain metastases dataset (45 patients, 171 annotated MRI scans across 187 timepoints) has been prepared, validated, and used to train and evaluate two CNN architectures as baselines for tumor representation. The key outcome is a quantitative demonstration that CNN embeddings encode static tumor anatomy but cannot model temporal change — providing direct justification for the Phase 3 transition to Vision Transformers.

---

## Phase 1 — Oncology Medical Imaging Preparation and Exploration (Weeks 1–4) ✅

### Activities Completed

| Activity | Status | Key Result |
|---|---|---|
| Dataset selection and acquisition | ✅ | Cyprus PROTEAS — 45 patients, 187 MRI timepoints, 4 modalities |
| Intensity normalization | ✅ | Dataset already Z-score normalized (validated: error = 0.0) |
| Spatial resampling | ✅ | Already 1×1×1 mm isotropic (validated: 45/45) |
| Longitudinal alignment | ✅ | 744/744 affine checks passed (100% co-registered) |
| Tumor annotation verification | ✅ | 171/187 masks valid; P28-P39 skull-stripping corrected |
| Tumor size & location analysis | ✅ | Volumes computed: mean WT = 22.8 cm³ (CV=1.61) |
| Temporal variability analysis | ✅ | Response trajectories classified; 100%→23% follow-up retention |
| Longitudinal organization | ✅ | Timeline CSV with day gaps; 187-row inventory |

### Deliverables

| Deliverable | Location |
|---|---|
| EDA notebook (40 cells, 9 sections) | `Phase1/notebooks/Phase1_Complete_EDA.ipynb` |
| Skull-stripping correction notebook | `Phase1/notebooks/Skull_Stripping_P28_P39.ipynb` |
| Pipeline & deliverables report | `Phase1/reports/Phase1_Pipeline_and_Deliverables.md` |
| Clinical data analysis report | `Phase1/reports/Phase1_Clinical_Data_Analysis.md` |
| Cleaned clinical data (28 cols) | `Phase1/outputs/PROTEAS_Clinical_Cleaned.xlsx` |
| Timeline CSV | `Phase1/outputs/cyprus_patient_timelines.csv` |
| Radiomic features (7,980) | `Phase1/outputs/radiomic_features.csv` |
| Tumor volumes | `Phase1/outputs/tumor_volumes.csv` |

---

## Phase 2 — Baseline Vision Models for Tumor Representation (Weeks 5–7) ✅

### Activities Completed

| Activity | Status | Key Result |
|---|---|---|
| Data preparation (A1) | ✅ | Stratified 3-fold + 5-fold CV splits at patient level |
| Model training (A2) | ✅ | Met-Seg (DynUNet) + SegResNet, 3-fold CV each |
| Embedding extraction (A3) | ✅ | 1024-dim (Met-Seg) + 128-dim (SegResNet), 3 folds |
| Evaluation battery (A4) | ✅ | 16-test battery: Met-Seg 12/16, SegResNet 7/16 |

### Segmentation Results (3-Fold Cross-Validation)

| Model | Mean Dice | WT | TC | ET |
|---|---|---|---|---|
| **Met-Seg v3/v4** | **0.505 ± 0.024** | 0.519 | 0.495 | 0.497 |
| Met-Seg v1 | 0.413 ± 0.016 | 0.426 | 0.408 | 0.406 |
| SegResNet | 0.368 ± 0.044 | 0.399 | 0.361 | 0.344 |

Met-Seg outperforms SegResNet by +37.2%, confirming that domain-specific pretraining (brain metastases vs glioma) is critical.

### Embedding Evaluation (16-Test Battery)

| Category | Met-Seg | SegResNet |
|---|---|---|
| Morphology (M1-M6) | **5/6** | 2/6 |
| Heterogeneity (H1-H4) | **4/4** | 2/4 |
| Temporal (T1-T7) | **3/6** | 3/6 |
| **Total** | **12/16** | **7/16** |

### Key Findings

1. **CNN embeddings encode tumor anatomy** — Volume prediction R²=0.38, necrosis detection F1=0.71
2. **CNN embeddings capture heterogeneity** — PCA structure |r|=0.80, GLCM entropy R²=0.39
3. **CNN embeddings CANNOT model temporal change** — T1 (r=0.05) and T3 (R²=-0.21) fail
4. **Response prediction below chance** — T4 AUC=0.46, confirming CNN cannot predict treatment outcomes
5. **Architecture matters** — Met-Seg (12/16) vs SegResNet (7/16), domain-specific pretraining critical

### Deliverables

| Deliverable | Location |
|---|---|
| Data preparation notebook | `Phase2/notebooks/Phase2_A1_Data_Preparation.ipynb` |
| Met-Seg training (3 folds) | `Phase2/notebooks/Phase2_A2_MetSeg_Training_Fold{0,1,2}.ipynb` |
| SegResNet training (3-fold) | `Phase2/notebooks/Phase2_A2_SegResNet_Training_3Fold.ipynb` |
| Embedding re-extraction | `Phase2/notebooks/Phase2_A3_Embedding_ReExtraction.ipynb` |
| Met-Seg evaluation (12/16) | `Phase2/notebooks/Phase2_A4_MetSeg_Embedding_Eval.ipynb` |
| SegResNet evaluation (7/16) | `Phase2/notebooks/Phase2_A4_SegResNet_Embedding_Eval.ipynb` |
| Complete report (18 refs) | `Phase2/reports/Phase2_Complete_Report.md` |
| Evaluation report (14 refs) | `Phase2/reports/Phase2_Activity4_Report.md` |
| Met-Seg embeddings (3 folds) | `Phase2/embeddings/metseg/` |
| SegResNet embeddings (3 folds) | `Phase2/embeddings/segresnet/` |
| Met-Seg weights (3 folds) | `Phase2/weights/metseg/` |
| Data splits (3-fold + 5-fold) | `Phase2/outputs/data_splits.json` |

---

## Phase 3 Justification

The CNN temporal test failures provide direct evidence for the Phase 3 transition:

| Limitation | Evidence | Phase 3 Solution |
|---|---|---|
| No temporal awareness | T1 r=0.049 (embedding distance ≠ volume change) | TaViT cross-attention across timepoints |
| Cannot predict change | T3 R²=-0.210 (worse than mean) | ViT temporal embeddings |
| Cannot predict response | T4 AUC=0.458 (below chance) | Longitudinal trajectory modeling |
| High coherence = static | T5 cos=0.995 (all scans look same) | Temporal differentiation via attention |

---

## Next Steps

1. **Phase 3 — Temporal Modeling:** Implement Swin UNETR + TaViT for temporal-aware embeddings
2. **Re-evaluate with same 16-test battery** — direct comparison with CNN baseline
3. **Target:** Improve temporal tests (T1, T3, T4) while maintaining morphology performance
