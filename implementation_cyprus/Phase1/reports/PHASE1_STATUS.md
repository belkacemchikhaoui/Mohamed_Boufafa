# Phase 1 -- Status and Deliverables

**Project:** Explainable Disease Progression and Counterfactual Video Generation
**Program:** Mitacs Globalink -- TELUQ University
**Supervisor:** Dr. Belkacem Chikhaoui
**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)
**Status Date:** March 24, 2026

---

## Phase 1 Scope

**Phase 1 -- Oncology Medical Imaging Preparation and Exploration**
Duration: 4 weeks (Weeks 1-4)

---

## Activity Checklist

### A. Selection and acquisition of public cancer imaging datasets

| Task                       | Status | Notes                                                                                  |
| -------------------------- | ------ | -------------------------------------------------------------------------------------- |
| Dataset selection          | DONE   | Cyprus PROTEAS chosen (brain metastases, longitudinal, public)                         |
| Dataset acquisition        | DONE   | 45 patient directories downloaded from Zenodo                                          |
| Clinical metadata loaded   | DONE   | 47 rows, 21 columns + 7 derived columns                                                |
| Clinical data cleaned      | DONE   | `outputs/PROTEAS_Clinical_Cleaned.xlsx` (28 columns)                                 |
| Data quality fixes applied | DONE   | 5 fixes: dose space, chemo case, treatment hierarchy, location normalization, KPS flag |

### B. Preprocessing -- Intensity normalization

| Task                       | Status    | Notes                                                 |
| -------------------------- | --------- | ----------------------------------------------------- |
| Z-score validation         | DONE      | Mean error = 0.0, Std error = 0.0 across all patients |
| Dataset already normalized | CONFIRMED | No additional normalization needed                    |

### C. Preprocessing -- Spatial resampling

| Task                 | Status    | Notes                                |
| -------------------- | --------- | ------------------------------------ |
| Resolution check     | DONE      | 45/45 patients at 1x1x1 mm isotropic |
| Grid dimensions      | DONE      | All 240x240x155 voxels               |
| Orientation          | DONE      | All LPS                              |
| No resampling needed | CONFIRMED | Dataset already standardized         |

### D. Longitudinal image alignment across time points

| Task                   | Status    | Notes                                                     |
| ---------------------- | --------- | --------------------------------------------------------- |
| Intra-visit alignment  | DONE      | 180/180 (100%) modality pairs match                       |
| Inter-visit alignment  | DONE      | 564/564 (100%) cross-visit pairs match                    |
| CoM shift analysis     | DONE      | Mean ~2 mm, max ~6 mm (within tolerance)                  |
| Total checks           | DONE      | 744 affine checks across all patients, modalities, visits |
| No registration needed | CONFIRMED | Dataset already co-registered                             |

### E. Verification and processing of tumor annotations

| Task                       | Status | Notes                                                |
| -------------------------- | ------ | ---------------------------------------------------- |
| Label validation           | DONE   | 171/171 masks valid (subset of {0,1,2,3})            |
| Empty mask check           | DONE   | 0 empty masks                                        |
| Label distribution         | DONE   | {0,1,2,3}=84, {0,2,3}=72, {0,2}=14, {0,3}=1          |
| Missing masks documented   | DONE   | 16/187 visits without masks (likely resolved tumors) |
| 3D overlay generation      | DONE   | NIfTI overlays in `outputs/3d_overlays/`           |
| Skull-stripping correction | DONE   | P28-P39 fixed, P31 float precision resolved          |
| Brain tissue labels        | DONE   | Labels 10/30/40/50 characterized                     |

### F. Exploratory data analysis -- Tumor sizes and locations

| Task                            | Status | Notes                                           |
| ------------------------------- | ------ | ----------------------------------------------- |
| Tumor volumes (NCR, ET, ED, WT) | DONE   | 171 timepoints analyzed                         |
| Spatial distributions           | DONE   | Centroid scatter, lobe distributions            |
| Lesion location standardization | DONE   | 37 labels -> 7 lobes + side + multi-lesion flag |
| Cross-tabulations               | DONE   | 6 clinically relevant cross-tabs (visual)       |

### G. Exploratory data analysis -- Inter-patient and temporal variability

| Task                        | Status | Notes                                               |
| --------------------------- | ------ | --------------------------------------------------- |
| Temporal trajectories       | DONE   | Volume tracking over all follow-ups                 |
| Response classification     | DONE   | Progressive/Stable/Responsive (RANO-like)           |
| Radiomic features           | DONE   | 7,980 features analyzed (7 categories, PCA)         |
| CT/RTP analysis             | DONE   | NaN-safe, cross-referenced with clinical metadata   |
| Dose validation             | DONE   | Prescribed vs hotspot ratios confirmed (1.12-1.30x) |
| Performance status analysis | DONE   | WHO PS + KPS with clinical interpretation           |
| Follow-up retention         | DONE   | 100% -> 23% dropout curve documented                |

### H. Organization of longitudinal patient imaging sequences

| Task                        | Status | Notes                                             |
| --------------------------- | ------ | ------------------------------------------------- |
| Timeline extraction         | DONE   | `outputs/cyprus_patient_timelines.csv`          |
| Day gaps computed           | DONE   | Days between visits and from baseline             |
| Split patient investigation | DONE   | 5 patients with a/b = distinct spatial metastases |
| Patient inventory           | DONE   | `outputs/cyprus_inventory.csv` (186 rows)       |

---

## Intermediate Deliverables

### 1. Reproducible imaging preprocessing pipeline

| Deliverable              | Status | Location                                    |
| ------------------------ | ------ | ------------------------------------------- |
| Skull-stripping notebook | DONE   | `notebooks/Skull_Stripping_P28_P39.ipynb` |
| Backups of original data | DONE   | `outputs/pre_skullstrip_backup/`          |

**Note:** The dataset was already preprocessed (normalized, resampled, aligned). The pipeline focused on identifying and correcting the skull-stripping inconsistency (P28-P39) and validating that no additional preprocessing was needed.

### 2. Cleaned, standardized, and temporally organized datasets

| Deliverable               | Status | Location                                            |
| ------------------------- | ------ | --------------------------------------------------- |
| Cleaned clinical metadata | DONE   | `outputs/PROTEAS_Clinical_Cleaned.xlsx` (28 cols) |
| Patient timelines         | DONE   | `outputs/cyprus_patient_timelines.csv`            |
| Patient inventory         | DONE   | `outputs/cyprus_inventory.csv`                    |
| 3D tumor overlays         | DONE   | `outputs/3d_overlays/*.nii.gz`                    |
| Skull-stripped MRIs       | DONE   | All 45 patients now consistent                      |

### 3. Exploratory data analysis report

| Deliverable                   | Status | Location                                           |
| ----------------------------- | ------ | -------------------------------------------------- |
| Complete EDA notebook         | DONE   | `notebooks/Phase1_Complete_EDA.ipynb` (40 cells) |
| EDA methodology report        | DONE   | `notebooks/Phase1_EDA_Explanation.md`            |
| EDA results + clinical report | DONE   | `notebooks/Phase1_EDA_Report.md`                 |
| Supervisor progress report    | DONE   | `SUPERVISOR_PROGRESS_REPORT.md`                  |
| Meeting report (LaTeX)        | DONE   | `meeting_report_march11.tex` + `.pdf`          |

---

## Key Findings

1. **Dataset is high quality**: Pre-normalized, co-registered, isotropic -- minimal preprocessing needed
2. **Skull-stripping inconsistency**: P28-P39 not stripped (now fixed)
3. **Clinical data quality issues**: 5 fixes applied (dose space, chemo case, treatment labels, locations, KPS)
4. **Treatment**: RS (76.6%) vs FSRT (23.4%) -- distinct strategies, must not merge
5. **KPS eligibility**: 2 patients (P33a/b) below SRS threshold (KPS=50)
6. **Lesion locations**: Cerebellum most common (consistent with lung/breast primary)
7. **Follow-up dropout**: 100% at baseline -> 23% at 12 months
8. **Radiomics**: 7,980 features across 4 masks x 4 modalities x 5 timepoints x 7 categories

---

## Phase 1 Completion: COMPLETE

All 8 activities (A-H) and all 3 intermediate deliverables are done.
Ready to proceed to **Phase 2 -- Baseline Vision Models for Tumor Representation**.
