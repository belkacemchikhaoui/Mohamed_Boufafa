# Phase 1 -- Oncology Medical Imaging Preparation and Exploration

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink -- TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)  
**Source:** [Zenodo DOI: 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793)  
**Duration:** 4 weeks (Weeks 1-4)

---

## Dataset Overview

| Aspect | Value |
| ------ | ----- |
| Patients | 40 unique (45 directories, 5 with a/b spatial splits) |
| Total MRI timepoints | 187 (BraTS/ folders, 4 modalities each: T1, T1c, T2, FLAIR) |
| Expert segmentation masks | 171 (labels: 0=BG, 1=NCR, 2=ET, 3=ED) |
| CT scans | 45 |
| RT dose plans (RTP) | 45 |
| Radiomic features | 7,980 pre-extracted (7 categories x 4 masks x 4 modalities x 5 timepoints) |
| Clinical metadata | 47 rows, 21 columns (demographics, treatment, performance status) |
| Resolution | 1x1x1 mm isotropic, 240x240x155 voxels |
| Size on disk | ~15.1 GB (zipped) |

### Per-Patient Structure

| Path | Contents |
|---|---|
| `P??/BraTS/baseline/` | t1.nii.gz, t1c.nii.gz, t2.nii.gz, fla.nii.gz |
| `P??/BraTS/fu1/ ... fu5/` | Follow-ups (~6 weeks to ~12 months post-treatment) |
| `P??/P??_brain_mask.nii.gz` | Healthy tissue labels (10=ventricles, 30=WM, 40=GM, 50=CSF) |
| `P??/P??_CT.nii.gz` | CT scan (Hounsfield Units, for RT planning) |
| `P??/P??_RTP.nii.gz` | 3D radiation dose map (Grays) |
| `P??/tumor_segmentation/P??_tumor_mask_*.nii.gz` | Segmentation (1=NCR, 2=ET, 3=ED) |

### Key Medical Context

| Term | Full Name | Meaning |
| ---- | --------- | ------- |
| NSCLC | Non-Small Cell Lung Cancer | Primary cancer in ~60% of cohort |
| SCLC | Small Cell Lung Cancer | Aggressive, ~10% of cohort |
| RS/SRS | (Stereotactic) Radiosurgery | Single-fraction, high dose (~18-24 Gy), 36 cases |
| FSRT | Fractionated Stereotactic RT | Multi-fraction (3x ~12 Gy), 11 cases |
| KPS | Karnofsky Performance Status | 0-100 scale (>70 = SRS eligible) |
| RTP | Radiotherapy Treatment Plan | 3D dose distribution, hotspot = 120-130% of marginal dose |

---

## Activities and Notebooks

### Activity A: Dataset Selection, Acquisition, Clinical Integration

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 1 + Section 7)

| Task | Status | Result |
| ---- | ------ | ------ |
| Dataset selection | DONE | Cyprus PROTEAS (longitudinal, expert labels, RTP, radiomics) |
| Timeline extraction | DONE | `outputs/cyprus_patient_timelines.csv` |
| Split patient investigation | DONE | 5 patients with a/b = distinct spatial metastases |
| Cross-modality visualization | DONE | 4-patient, 6-modality side-by-side views |
| Clinical data loading + cleaning | DONE | 5 quality fixes, `outputs/PROTEAS_Clinical_Cleaned.xlsx` (28 cols) |

**Clinical data quality fixes applied:**
1. Dose trailing space (`'20 Gy '` -> `'20 Gy'`)
2. Prior Chemo case normalization (`'yes'` -> `'Yes'`)
3. Treatment hierarchy: 7 annotations -> RS/FSRT groups + re-irradiation flag
4. Lesion locations: 37 raw labels -> 7 lobes + side + multi-lesion flag
5. KPS eligibility flag: P33a/b with KPS=50 (below SRS threshold)

### Activity B: Intensity Normalization

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 2.1)

| Task | Status | Result |
| ---- | ------ | ------ |
| Z-score validation | DONE | Mean error = 0.0, Std error = 0.0 |
| Diagnostic plots | DONE | Per-patient scatter, raw stats, dynamic range |
| Conclusion | -- | Dataset already normalized, no additional processing needed |

### Activity C: Spatial Resampling

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 2.2)

| Task | Status | Result |
| ---- | ------ | ------ |
| Resolution check | DONE | 45/45 at 1x1x1 mm isotropic |
| Grid dimensions | DONE | All 240x240x155 |
| Conclusion | -- | No resampling needed |

### Activity D: Longitudinal Alignment

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 3)

| Task | Status | Result |
| ---- | ------ | ------ |
| Intra-visit alignment | DONE | 180/180 (100%) |
| Inter-visit alignment | DONE | 564/564 (100%) |
| CoM shift | DONE | Mean ~2 mm, max ~6 mm |
| Total checks | DONE | 744 affine checks |
| Conclusion | -- | Already co-registered, no registration needed |

### Activity E: Tumor Annotation Verification

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 4) + `Skull_Stripping_P28_P39.ipynb`

| Task | Status | Result |
| ---- | ------ | ------ |
| Label validation | DONE | 171/171 valid, 0 empty |
| Missing masks | DONE | 16/187 visits (likely resolved tumors) |
| 3D overlays | DONE | `outputs/3d_overlays/*.nii.gz` |
| Skull-stripping check | DONE | P28-P39 not stripped |
| Skull-stripping fix | DONE | Brain masks applied, P31 float32 fix |
| Brain tissue analysis | DONE | Ventricles, WM, GM, CSF characterized |

### Activities F, G: Exploratory Data Analysis

**Notebook:** `Phase1_Complete_EDA.ipynb` (Sections 5-8)

| Task | Status | Result |
| ---- | ------ | ------ |
| Tumor volumes (NCR, ET, ED, WT) | DONE | 171 timepoints, high CV |
| Spatial distributions | DONE | Centroid scatter, standardized lobes |
| Temporal trajectories | DONE | RANO-like response classification |
| Radiomic features | DONE | 7,980 features, PCA, correlation analysis |
| CT analysis | DONE | HU distributions, all 45 patients |
| RTP analysis | DONE | NaN-filtered, cross-referenced with Rx dose |
| Clinical cross-tabulations | DONE | 6 visual panels |

### Activity H: Longitudinal Organization

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 1 + 9)

| Task | Status | Result |
| ---- | ------ | ------ |
| Timeline CSV | DONE | Day gaps, visit ordering |
| Patient inventory | DONE | 186-row inventory CSV |
| Follow-up retention curve | DONE | 100% -> 23% at 12 months |

---

## Deliverables

### 1. Reproducible imaging preprocessing pipeline

| File | Description |
| ---- | ----------- |
| `notebooks/Skull_Stripping_P28_P39.ipynb` | Skull-stripping correction (12 patients, 5 steps) |
| `outputs/pre_skullstrip_backup/` | Original MRI backups |

Note: The dataset was already preprocessed. The pipeline focused on identifying and correcting the P28-P39 skull-stripping inconsistency.

### 2. Cleaned, standardized, temporally organized datasets

| File | Description |
| ---- | ----------- |
| `outputs/PROTEAS_Clinical_Cleaned.xlsx` | 47 rows, 28 columns (7 new derived) |
| `outputs/cyprus_patient_timelines.csv` | Longitudinal timeline with day gaps |
| `outputs/cyprus_inventory.csv` | Per-timepoint inventory (186 rows) |
| `outputs/3d_overlays/*.nii.gz` | T1c + colored mask 3D overlays |
| `outputs/skull_stripping_verification.png` | Before/after comparison |

### 3. Exploratory data analysis report

| File | Description |
| ---- | ----------- |
| `notebooks/Phase1_Complete_EDA.ipynb` | Complete EDA notebook (40 cells) |
| `notebooks/Phase1_EDA_Explanation.md` | Section-by-section methodology |
| `notebooks/Phase1_EDA_Report.md` | Results + clinical interpretation (328 lines) |
| `SUPERVISOR_PROGRESS_REPORT.md` | Progress report for supervisor |
| `PHASE1_STATUS.md` | Activity checklist with completion status |
| `meeting_report_march11.tex` + `.pdf` | LaTeX meeting report |

---

## Key Findings

1. Dataset is high quality: pre-normalized, co-registered, isotropic
2. Skull-stripping inconsistency (P28-P39) identified and corrected
3. Clinical data: 5 quality fixes, RS (76.6%) vs FSRT (23.4%) treatment groups
4. 2 patients (P33a/b) below SRS eligibility threshold (KPS=50)
5. Cerebellum is the most common lesion location (consistent with lung/breast primary)
6. Follow-up dropout: 100% -> 23% at 12 months
7. RTP hotspot = 120-130% of prescribed marginal dose (confirmed)
8. 7,980 radiomic features available across all combinations

---

## Status: COMPLETE

All 8 activities and all 3 deliverables are done. Ready for Phase 2.
