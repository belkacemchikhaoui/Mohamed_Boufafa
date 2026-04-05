# Cyprus PROTEAS — Project File Guide

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink — TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Last Updated:** April 2, 2026

---

## Project Structure

```
implementation_cyprus/
├── PHASE1_PLAN.md                       Phase 1 plan (activities, deliverables, findings)
├── PHASE1_STATUS.md                     Phase 1 checklist (all DONE)
├── PHASE2_PLAN.md                       Phase 2 plan (architectures, Kaggle GPU strategy)
├── PHASE2_STATUS.md                     Phase 2 checklist (all DONE)
├── SUPERVISOR_PROGRESS_REPORT.md        Combined progress report for supervisor
├── NOTEBOOKS_GUIDE.md                   This file
│
├── notebooks/
│   │── Phase 1 ─────────────────────────────────────────────────
│   ├── Phase1_Complete_EDA.ipynb              Main EDA notebook (40 cells, 9 sections)
│   ├── Skull_Stripping_P28_P39.ipynb          Skull-stripping fix for P28-P39
│   ├── Phase1_EDA_Explanation.md              Section-by-section methodology
│   ├── Phase1_EDA_Report.md                   Results + clinical interpretation
│   │
│   │── Phase 2 ─────────────────────────────────────────────────
│   ├── Phase2_Activity1_Data_Preparation.ipynb   Data splits + MONAI pipeline setup
│   ├── Phase2_Activity1_DataPrep_Report.md       Data preparation report
│   ├── Phase2_Activity2_MetSeg.ipynb             Met-Seg CNN training (Kaggle GPU)
│   ├── Phase2_Activity2_Training.ipynb           SegResNet CNN training (Kaggle GPU)
│   ├── Phase2_Activity2_QuickTest_Guide.md       Quick test guide for training
│   ├── Phase2_EmbeddingFix_ReExtract.ipynb       Fixed embedding extraction (Kaggle)
│   ├── Phase2_Activity4_Evaluation.ipynb         Met-Seg embedding evaluation (16 tests)
│   ├── Phase2_Activity4_SegResNet.ipynb          SegResNet embedding evaluation (16 tests)
│   ├── Phase2_Activity4_Report.md                Evaluation battery report
│   └── emb_extr_V1_vs_V2_Comparison.md           Embedding extraction version comparison
│
├── outputs/
│   │── Clinical & Metadata ──────────────────────────────────────
│   ├── PROTEAS_Clinical_Cleaned.xlsx             Cleaned clinical data (47 rows, 28 cols)
│   ├── cyprus_patient_timelines.csv               Longitudinal timeline with day gaps
│   ├── cyprus_inventory.csv                       Per-timepoint inventory (186 rows)
│   ├── data_splits.json                           3-fold CV splits (patient-level)
│   ├── clinical_data.json                         JSON version of clinical metadata
│   │
│   │── Volumes & Features ───────────────────────────────────────
│   ├── tumor_volumes.csv                          Tumor volumes per scan (171 rows)
│   ├── radiomic_features.csv                      7,980 radiomic features
│   ├── activity4_results.json                     Evaluation battery results (JSON)
│   │
│   │── Visualizations ───────────────────────────────────────────
│   ├── eda/                                       Phase 1 EDA figures
│   ├── 3d_overlays/                               T1c + mask 3D overlay NIfTI files
│   ├── activity4_figures/                         Met-Seg evaluation plots (t-SNE, etc.)
│   ├── activity4_figures_segresnet/               SegResNet evaluation plots
│   ├── skull_stripping_verification.png           Before/after skull-strip comparison
│   │
│   │── Reports ──────────────────────────────────────────────────
│   ├── Phase2_Complete_Report.md                  Final Phase 2 report (12/16, SegResNet)
│   ├── CYPRUS_DATASET_DEEP_ANALYSIS.md            Dataset analysis document
│   └── CYPRUS_DATASET_STRUCTURE_AND_ALIGNMENT.md  Data structure documentation
│
├── scripts/
│   ├── make_embedding_fix_nb.py                   Generates embedding extraction notebook
│   ├── make_activity4_nb.py                       Generates Activity 4 evaluation notebook
│   ├── upload_checkpoints_2nd_account.py          Kaggle checkpoint upload
│   └── upload_to_kaggle.py                        Dataset upload to Kaggle
│
├── Data/Cyprus-PROTEAS-zips/                      Raw dataset (45 patient directories)
├── detector_weight (full modality).ckpt           Met-Seg DenseNet121 detector (137 MB)
├── segmentor_weight (full modality).ckpt          Met-Seg DynUNet segmenter (179 MB)
└── meeting_report_march11.tex + .pdf              LaTeX meeting report
```

### External Data (outside implementation_cyprus/)

```
folder{0,1,2}/
├── cnn_metseg_embeddings_v2_fold{0,1,2}.npz      Met-Seg embeddings (170 × 1024-dim)
└── metseg_fold{0,1,2}_best.pth                    Met-Seg trained weights

cnn_emb_SegResNet/
└── cnn_embeddings_fold{0,1,2}.npz                 SegResNet embeddings (170 × 128-dim)
```

---

## Notebooks Summary

### Phase 1 — Oncology Medical Imaging Preparation and Exploration

| Notebook | Cells | Purpose | Runs On |
|---|---|---|---|
| `Phase1_Complete_EDA.ipynb` | 40 | Complete EDA: data loading, normalization validation, alignment checks, tumor analysis, radiomics, clinical cross-tabs | Local CPU |
| `Skull_Stripping_P28_P39.ipynb` | 5 steps | Fix skull-stripping inconsistency for patients P28-P39 | Local CPU |

### Phase 2 — Baseline Vision Models for Tumor Representation

| Notebook | Purpose | Runs On |
|---|---|---|
| `Phase2_Activity1_Data_Preparation.ipynb` | 3-fold CV splits, data validation, MONAI dataset setup | Local CPU |
| `Phase2_Activity2_MetSeg.ipynb` | Met-Seg (DynUNet) training — 3 folds, 60-80 epochs | Kaggle GPU |
| `Phase2_Activity2_Training.ipynb` | SegResNet training — 3 folds, 50 epochs | Kaggle GPU |
| `Phase2_EmbeddingFix_ReExtract.ipynb` | Fixed embedding extraction with patch accumulation | Kaggle GPU |
| `Phase2_Activity4_Evaluation.ipynb` | Met-Seg 16-test evaluation battery — **12/16 passed** | Local CPU |
| `Phase2_Activity4_SegResNet.ipynb` | SegResNet 16-test evaluation battery — **7/16 passed** | Local CPU |

---

## Reports Summary

| File | Type | Content |
|---|---|---|
| `PHASE1_PLAN.md` | Plan | Activities A-H, deliverables, key findings |
| `PHASE1_STATUS.md` | Checklist | All 8 activities marked DONE |
| `Phase1_EDA_Explanation.md` | Methodology | Section-by-section how each analysis works |
| `Phase1_EDA_Report.md` | Results | Clinical interpretation of all EDA findings |
| `PHASE2_PLAN.md` | Plan | Architecture details, training strategy, evaluation battery |
| `PHASE2_STATUS.md` | Checklist | All activities marked DONE, training curves |
| `Phase2_Complete_Report.md` | Report | Final results: 12/16 Met-Seg, 7/16 SegResNet, literature refs |
| `SUPERVISOR_PROGRESS_REPORT.md` | Progress | Combined Phase 1+2 report for Dr. Chikhaoui |

---

## Data Sizes

| Component | Size | Phase |
|---|---|---|
| BraTS MRI (4 mod × 186 tp) | 1.67 GB | 1-5 |
| Tumor masks (171 tp) | 3.7 MB | 1-3 |
| Brain masks (45) | 11 MB | 1 |
| CT scans (45) | 388 MB | 4-5 |
| RTP dose plans (45) | 573 MB | 4-5 |
| Clinical + radiomics (Excel) | 6.3 MB | 1-5 |
| Met-Seg weights (3 folds) | ~705 MB | 2 |
| Met-Seg embeddings (3 folds) | ~2.2 MB | 2 |
| SegResNet embeddings (3 folds) | ~260 KB | 2 |
| **Total project** | **~3.4 GB** | |
