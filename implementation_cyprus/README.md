# Explainable Disease Progression & Counterfactual Video Generation

**Student:** Mohamed Boufafa  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Program:** Mitacs Globalink — TÉLUQ University  
**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)

---

## Project Overview

This project develops temporal-aware deep learning representations for brain metastasis MRI analysis, with the ultimate goal of explainable disease progression modeling and counterfactual video generation.

### Phase Status

| Phase | Title | Duration | Status | Score |
|---|---|---|---|---|
| **Phase 1** | Oncology Medical Imaging Preparation and Exploration | Weeks 1–4 | ✅ Complete | All 8 activities done |
| **Phase 2** | Baseline Vision Models for Tumor Representation | Weeks 5–7 | ✅ Complete | Met-Seg 12/16, SegResNet 7/16 |
| **Phase 3** | Temporal Modeling and Explainability | Weeks 8–10 | 🔜 Next | — |
| **Phase 4** | LLM-based Narrative Generation | Weeks 11–13 | — | — |
| **Phase 5** | Counterfactual Video Generation | Weeks 14–16 | — | — |

---

## Directory Structure

| Folder | Contents |
|---|---|
| `Phase1/notebooks/` | EDA notebook (40 cells), skull-stripping fix |
| `Phase1/reports/` | 5 deliverable reports (pipeline, clinical, EDA, plan, status) |
| `Phase1/outputs/` | CSVs, Excel, tumor volumes, radiomic features |
| `Phase2/notebooks/` | 8 notebooks (A1 data prep → A4 evaluation) |
| `Phase2/reports/` | 4 deliverable reports (complete, activity4, data prep, status) |
| `Phase2/embeddings/` | Met-Seg (v1+v2 × 3 folds) + SegResNet (3 folds) |
| `Phase2/weights/` | Met-Seg trained weights (3 folds) + pretrained checkpoints |
| `Phase2/outputs/` | Evaluation figures, results JSON, data splits |
| `Phase2/scripts/` | 3 helper scripts (notebook generation, evaluation) |
| `Phase2/training_logs/` | Training logs and metrics JSON |
| `meeting_reports/` | March 11 meeting report + supervisor progress report |
| `study_guides/` | 9 personal reference docs (methodology, explanations) |
| `Data/` | Raw dataset (45 patients, 187 timepoints, 4 modalities) |

---

## How to Review — Step by Step

### 📖 Phase 1: Data Preparation (start here)

| Step | Action | File to Open |
|---|---|---|
| **1.1** | Read the complete summary | `Phase1/reports/Phase1_Pipeline_and_Deliverables.md` |
| **1.2** | Check all activities are done | `Phase1/reports/PHASE1_STATUS.md` |
| **1.3** | Walk through the EDA notebook | `Phase1/notebooks/Phase1_Complete_EDA.ipynb` |
| **1.4** | Understand the methodology | `Phase1/reports/Phase1_EDA_Explanation.md` |
| **1.5** | Review clinical findings | `Phase1/reports/Phase1_Clinical_Data_Analysis.md` |
| **1.6** | Check skull-stripping fix | `Phase1/notebooks/Skull_Stripping_P28_P39.ipynb` |

**Key things to verify in Phase 1:**
- [ ] Dataset has 45 patients, 187 timepoints, 171 masks
- [ ] Z-score normalization validated (error = 0.0)
- [ ] 744 alignment checks passed (100%)
- [ ] P28-P39 skull-stripping corrected
- [ ] 5 clinical data quality fixes applied
- [ ] Cleaned Excel has 28 columns

### 🧠 Phase 2: CNN Baseline (after Phase 1)

| Step | Action | File to Open |
|---|---|---|
| **2.1** | Read the complete summary | `Phase2/reports/Phase2_Complete_Report.md` |
| **2.2** | Check all training results | `Phase2/reports/PHASE2_STATUS.md` |
| **2.3** | Review data preparation | `Phase2/notebooks/Phase2_A1_Data_Preparation.ipynb` |
| **2.4** | Review Met-Seg fold 0 training | `Phase2/notebooks/Phase2_A2_MetSeg_Training_Fold0.ipynb` |
| **2.5** | Review SegResNet training | `Phase2/notebooks/Phase2_A2_SegResNet_Training_3Fold.ipynb` |
| **2.6** | Review embedding re-extraction | `Phase2/notebooks/Phase2_A3_Embedding_ReExtraction.ipynb` |
| **2.7** | Review Met-Seg evaluation (12/16) | `Phase2/notebooks/Phase2_A4_MetSeg_Embedding_Eval.ipynb` |
| **2.8** | Review SegResNet evaluation (7/16) | `Phase2/notebooks/Phase2_A4_SegResNet_Embedding_Eval.ipynb` |
| **2.9** | Read evaluation analysis | `Phase2/reports/Phase2_Activity4_Report.md` |
| **2.10** | Read data preparation details | `Phase2/reports/Phase2_Data_Preparation_Report.md` |

**Key things to verify in Phase 2:**
- [ ] Met-Seg Dice = 0.505 ± 0.024 (3-fold CV)
- [ ] SegResNet Dice = 0.368 ± 0.044 (3-fold CV)
- [ ] Met-Seg embeddings pass 12/16 tests
- [ ] SegResNet embeddings pass 7/16 tests
- [ ] Temporal tests T1, T3, T4 FAIL (justifies Phase 3)
- [ ] Morphology tests M1, M2, M4, M5 PASS (CNN encodes anatomy)
- [ ] All heterogeneity tests H1-H4 PASS

### 📄 Supervisor Reports

| Step | Action | File to Open |
|---|---|---|
| **3.1** | Read combined progress report | `meeting_reports/SUPERVISOR_PROGRESS_REPORT.md` |
| **3.2** | Check March 11 meeting notes | `meeting_reports/meeting_report_march11.pdf` |
