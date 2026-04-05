# Phase 2 — Activity 1: Data Preparation Report

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink — TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Date:** March 25, 2026  
**Status:** ✅ COMPLETE  
**Notebook:** `Phase2_A1_Data_Preparation.ipynb`

---

## 1. Objective

Activity 1 builds the data infrastructure required for CNN training in Phase 2. This involves scanning all patient directories, building a complete scan inventory, identifying valid training samples, and creating reproducible cross-validation splits that will be reused in Phase 3 for fair comparison.

---

## 2. Dataset Overview

### 2.1 Scan Inventory

| Metric | Count |
|---|---|
| Total patient directories | 45 |
| Unique patient groups (a/b merged) | 40 |
| Total MRI timepoints | 187 |
| Timepoints with expert masks | **171** |
| Timepoints without masks | 16 |
| Modalities per scan | 4 (T1, T1c, T2, FLAIR) |

### 2.2 Patients with a/b Splits

Five patients have two separate lesion directories representing distinct treated lesions in the same brain. These are always kept in the **same fold** during cross-validation to prevent data leakage [1]:

| Patient Group | Directories | Total Scans |
|---|---|---|
| P04 | P04a, P04b | 10 |
| P07 | P07a, P07b | 6 |
| P17 | P17a, P17b | 11 |
| P20 | P20a, P20b | 8 |
| P23 | P23a, P23b | 9 |

### 2.3 Missing Masks (16 Timepoints)

These follow-up scans have BraTS MRI data but **no expert tumor annotations**. Likely reasons: tumor fully resolved post-treatment, or expert did not annotate late follow-ups.

| Patient | Missing Visits | Count |
|---|---|---|
| P09 | fu2 | 1 |
| P15 | fu1, fu4 | 2 |
| P16 | fu1, fu2, fu3 | 3 |
| P19 | fu4, fu5 | 2 |
| P20b | fu1, fu2 | 2 |
| P29 | fu5 | 1 |
| P32 | fu2, fu3, fu4, fu5 | 4 |
| P38 | fu3 | 1 |

**Impact on training:** These 16 scans are excluded from segmentation training but are included during embedding extraction (forward pass without mask comparison).

### 2.4 Data Path Normalization

Two naming inconsistencies were handled automatically:
- **P01, P30:** Use `"tumor segmentation"` (with space) instead of `"tumor_segmentation"` (underscore)
- **P31:** BraTS directories named `Fu1`, `Fu2`, `Fu3` (capital F) but mask files use lowercase `fu1`

---

## 3. Stratified Cross-Validation Splits

### 3.1 Stratification Design

Splits are created at the **patient group level** (not scan level) to prevent data leakage [1, 2]. Each patient group is assigned a stratification label combining treatment type and tumor histology:

| Stratification Label | Patient Groups | % |
|---|---|---|
| RS_NSCLC | 25 | 62.5% |
| FSRT_Breast | 8 | 20.0% |
| RS_Breast | 6 | 15.0% |
| RS_Other (SCLC) | 1 | 2.5% |

The single SCLC patient is merged with RS_NSCLC for fold assignment (insufficient samples for its own stratum).

### 3.2 Three-Fold Split (Primary)

| Fold | Train Groups | Train Scans | Val Groups | Val Scans |
|---|---|---|---|---|
| 0 | 26 | 114 | 14 | 56 |
| 1 | 27 | 119 | 13 | 51 |
| 2 | 27 | 107 | 13 | 63 |
| **Total** | | | **40** | **171** |

### 3.3 Five-Fold Split (For Final Publication)

| Fold | Train Groups | Train Scans | Val Groups | Val Scans |
|---|---|---|---|---|
| 0 | 32 | 143 | 8 | 27 |
| 1 | 32 | 132 | 8 | 38 |
| 2 | 32 | 138 | 8 | 32 |
| 3 | 32 | 142 | 8 | 28 |
| 4 | 32 | 125 | 8 | 45 |
| **Total** | | | **40** | **170** |

### 3.4 Integrity Checks (All Passed ✅)

- ✅ No train/test patient overlap in any fold
- ✅ a/b patient groups always assigned to the same fold
- ✅ Every patient group appears in validation exactly once
- ✅ All 171 masked scans accounted for
- ✅ All file paths relative to DATA_ROOT (portable across local and Kaggle environments)

---

## 4. Output Files

| File | Location | Description |
|---|---|---|
| `data_splits.json` | `Phase2/outputs/` | Train/val splits for 3-fold and 5-fold CV. Shared with Phase 3 for fair comparison. |
| `data_inventory.csv` | `outputs/` | Complete 187-entry scan inventory with paths and mask status. |
| `phase2_data_summary.json` | `Phase2/outputs/` | Summary statistics (scan counts, group counts). |

---

## 5. Design Decisions

| Decision | Rationale |
|---|---|
| **Patient-level splits** (not scan-level) | Prevents data leakage — same brain in train and test would inflate scores [1] |
| **Stratified by treatment × histology** | Ensures each fold has proportional mix of RS/FSRT and tumor types [2] |
| **a/b patients grouped** | Same patient's two lesions are correlated — splitting would leak information |
| **3-fold primary, 5-fold later** | 3-fold is time-efficient for initial experiments; 5-fold for publication-quality CI [3] |
| **JSON format** | Portable across local machine and Kaggle; shared with Phase 3 |
| **Relative paths** | Paths are relative to DATA_ROOT, enabling same splits on any machine |

---

## References

| # | Reference | How We Used It |
|---|---|---|
| [1] | Chollet, F. (2018). *Deep Learning with Python.* Manning Publications. Ch. 4.2 | Patient-level split rationale — avoiding data leakage in medical imaging |
| [2] | Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825-2830 | `StratifiedKFold` implementation for balanced fold assignment |
| [3] | Varoquaux, G. (2018). *Cross-validation failure: Small sample sizes lead to large error bars.* NeuroImage, 180, 68-77 | Justification for starting with 3-fold (sufficient for paired comparison) before 5-fold |
| [4] | Flouri, D., Pattichis, C., et al. (2025). *Cyprus PROTEAS: A Longitudinal Brain Metastasis Dataset.* Zenodo. DOI: 10.5281/zenodo.17253793 | Primary dataset — 45 patients, 187 timepoints |
