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

```
implementation_cyprus/
│
├── README.md                    ← You are here
│
├── Phase1/                      ← ONCOLOGY DATA PREPARATION
│   ├── notebooks/
│   │   ├── Phase1_Complete_EDA.ipynb         Main EDA (40 cells, 9 sections)
│   │   └── Skull_Stripping_P28_P39.ipynb     Data correction
│   ├── reports/
│   │   ├── Phase1_Complete_Report.md         📋 START HERE — full summary
│   │   ├── PHASE1_PLAN.md                    Activity plan
│   │   ├── PHASE1_STATUS.md                  Completion checklist
│   │   ├── Phase1_EDA_Report.md              Clinical findings
│   │   └── Phase1_EDA_Explanation.md         Methodology
│   └── outputs/                              → links to outputs/
│
├── Phase2/                      ← CNN BASELINE MODELS
│   ├── notebooks/
│   │   ├── Phase2_Activity1_Data_Preparation.ipynb   Data splits
│   │   ├── Phase2_Activity2_MetSeg.ipynb             Met-Seg training
│   │   ├── Phase2_Activity2_Training.ipynb           SegResNet training
│   │   ├── Phase2_EmbeddingFix_ReExtract.ipynb       Embedding extraction
│   │   ├── Phase2_Activity4_Evaluation.ipynb         Met-Seg eval (12/16)
│   │   └── Phase2_Activity4_SegResNet.ipynb          SegResNet eval (7/16)
│   ├── reports/
│   │   ├── Phase2_Complete_Report.md         📋 START HERE — full summary
│   │   ├── PHASE2_PLAN.md                    Architecture & strategy
│   │   ├── PHASE2_STATUS.md                  Training curves & status
│   │   ├── Phase2_Activity4_Report.md        Evaluation analysis
│   │   ├── Phase2_Activity1_DataPrep_Report.md  Data preparation
│   │   ├── Phase2_Activity2_QuickTest_Guide.md  Training guide
│   │   └── emb_extr_V1_vs_V2_Comparison.md     Embedding versions
│   ├── outputs/                              → links to outputs/
│   └── scripts/
│       ├── make_embedding_fix_nb.py          Generates extraction notebook
│       ├── make_activity4_nb.py              Generates evaluation notebook
│       └── activity4_evaluation.py           Evaluation functions
│
├── reports/                     ← SUPERVISOR & CROSS-PHASE REPORTS
│   ├── SUPERVISOR_PROGRESS_REPORT.md         Combined Phase 1+2 report
│   ├── progress_report_april2.tex            LaTeX version
│   ├── progress_report_april2.pdf            PDF version (2 pages)
│   ├── meeting_report_march11.tex            March meeting report
│   ├── meeting_report_march11.pdf            March meeting PDF
│   └── NOTEBOOKS_GUIDE.md                    Full project file guide
│
├── Data/                        ← RAW DATASET (45 patients)
├── outputs/                     ← ALL GENERATED DATA
├── notebooks/                   ← ORIGINAL NOTEBOOK LOCATION (kept for compatibility)
└── scripts/                     ← UTILITY SCRIPTS (Kaggle upload, etc.)
```

---

## How to Review — Step by Step

### 📖 Phase 1: Data Preparation (start here)

| Step | Action | File to Open |
|---|---|---|
| **1.1** | Read the complete summary | `Phase1/reports/Phase1_Complete_Report.md` |
| **1.2** | Check all activities are done | `Phase1/reports/PHASE1_STATUS.md` |
| **1.3** | Walk through the EDA notebook | `Phase1/notebooks/Phase1_Complete_EDA.ipynb` |
| **1.4** | Understand the methodology | `Phase1/reports/Phase1_EDA_Explanation.md` |
| **1.5** | Review clinical findings | `Phase1/reports/Phase1_EDA_Report.md` |
| **1.6** | Check skull-stripping fix | `Phase1/notebooks/Skull_Stripping_P28_P39.ipynb` |

**Key things to verify in Phase 1:**
- [ ] Dataset has 45 patients, 186 timepoints, 171 masks
- [ ] Z-score normalization validated (error = 0.0)
- [ ] 728 alignment checks passed (100%)
- [ ] P28-P39 skull-stripping corrected
- [ ] 5 clinical data quality fixes applied
- [ ] Cleaned Excel has 28 columns

### 🧠 Phase 2: CNN Baseline (after Phase 1)

| Step | Action | File to Open |
|---|---|---|
| **2.1** | Read the complete summary | `Phase2/reports/Phase2_Complete_Report.md` |
| **2.2** | Understand the architecture choices | `Phase2/reports/PHASE2_PLAN.md` |
| **2.3** | Check all training results | `Phase2/reports/PHASE2_STATUS.md` |
| **2.4** | Review data preparation | `Phase2/notebooks/Phase2_Activity1_Data_Preparation.ipynb` |
| **2.5** | Review Met-Seg training | `Phase2/notebooks/Phase2_Activity2_MetSeg.ipynb` |
| **2.6** | Review SegResNet training | `Phase2/notebooks/Phase2_Activity2_Training.ipynb` |
| **2.7** | **Run** Met-Seg evaluation | `Phase2/notebooks/Phase2_Activity4_Evaluation.ipynb` |
| **2.8** | **Run** SegResNet evaluation | `Phase2/notebooks/Phase2_Activity4_SegResNet.ipynb` |
| **2.9** | Read evaluation analysis | `Phase2/reports/Phase2_Activity4_Report.md` |

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
| **3.1** | Read combined progress report | `reports/SUPERVISOR_PROGRESS_REPORT.md` |
| **3.2** | Review PDF for supervisor | `reports/progress_report_april2.pdf` |
| **3.3** | Check March meeting notes | `reports/meeting_report_march11.pdf` |
