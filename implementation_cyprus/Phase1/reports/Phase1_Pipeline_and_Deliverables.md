# Phase 1 — Complete Report: Oncology Medical Imaging Preparation and Exploration

**Project:** Explainable Disease Progression and Counterfactual Video Generation
**Program:** Mitacs Globalink — TELUQ University
**Supervisor:** Dr. Belkacem Chikhaoui
**Duration:** Weeks 1–4 | **Status:** ✅ COMPLETE

---

## Executive Summary

Phase 1 prepared the Cyprus PROTEAS brain metastases dataset for deep learning analysis. We performed comprehensive data validation, preprocessing correction, and exploratory analysis across 45 patients with 187 longitudinal MRI visits. Key outcomes:

- **Dataset validation:** 100% preprocessed — already normalized, resampled, and co-registered
- **Data quality fix:** Skull-stripping inconsistency identified and corrected for patients P28-P39
- **Clinical data:** 5 quality fixes applied, 7 derived features created (28 total columns)
- **EDA:** Complete analysis of tumor volumes, spatial distributions, temporal trajectories, radiomics, CT, and radiation therapy planning data
- **Key finding:** The dataset is uniquely suited for temporal modeling — longitudinal follow-ups with multi-class tumor annotations, a combination not available in other public brain metastasis datasets

---

## Dataset: Cyprus PROTEAS Brain Metastases

**Source:** Flouri et al. (2025), Zenodo DOI: 10.5281/zenodo.17253793

| Property                     | Value                                                            |
| ---------------------------- | ---------------------------------------------------------------- |
| **Patients**           | 40 unique (45 directories — 5 patients with a/b spatial splits) |
| **Visit dates**        | 236 (from DICOM data)                                            |
| **MRI visits**         | 187 (BraTS/ folders with 4 modalities each: T1, T1c, T2, FLAIR) |
| **Segmentation masks** | 171 (labels: 0=BG, 1=NCR, 2=ET, 3=ED)                            |
| **CT scans**           | 45                                                               |
| **RT dose plans**      | 45                                                               |
| **Radiomic features**  | 7,980 pre-extracted                                              |
| **Clinical metadata**  | 47 rows, 21 columns                                              |
| **Resolution**         | 1×1×1 mm isotropic, 240×240×155 voxels                       |
| **Size**               | ~15.1 GB (including DICOM)                                       |

### Per-Patient Data Structure

```
P??/
├── BraTS/
│   ├── baseline/          T1, T1c, T2, FLAIR
│   ├── fu1/               Follow-up 1 (~6 weeks post-treatment)
│   ├── fu2/ ... fu5/      Follow-ups 2-5 (~3, 6, 9, 12 months)
├── P??_brain_mask.nii.gz  Healthy tissue labels (10=ventricles, 30=WM, 40=GM, 50=CSF)
├── P??_CT.nii.gz          Planning CT (Hounsfield Units)
├── P??_RTP.nii.gz         3D radiation dose map (Gy)
└── tumor_segmentation/
    ├── P??_tumor_mask_baseline.nii.gz   Multi-class (1=NCR, 2=ET, 3=ED)
    └── P??_tumor_mask_fu?.nii.gz
```

### Why This Dataset Was Selected

| Criterion                            | Cyprus PROTEAS                   | UCSF-BMSR                  | Stanford BrainMetShare                     |
| ------------------------------------ | -------------------------------- | -------------------------- | ------------------------------------------ |
| **Longitudinal follow-up**     | ✅ 3-7 visits/patient            | ❌ Single timepoint        | ❌ Single timepoint                        |
| **Multi-class segmentation**   | ✅ NCR, ET, ED                   | ❌ Binary only             | ❌ Binary only                             |
| **Modalities**                 | T1, T1c, T2, FLAIR (4)           | T1-pre, T1-post, FLAIR (3) | T1-pre, T1-post-SE, T1-post-GRE, FLAIR (4) |
| **Treatment response data**    | ✅ RS vs FSRT, clinical outcomes | ❌ No treatment data       | ❌ No treatment data                       |
| **Radiation therapy planning** | ✅ 3D dose maps                  | ❌ Not available           | ❌ Not available                           |

Cyprus PROTEAS is the **only publicly available brain metastasis dataset** with all five features, making it uniquely suited for our project's temporal modeling and explainability goals.

---

## Activities Completed

### Activity A: Dataset Selection and Acquisition ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Sections 1, 7)

| Task                        | Result                                                                           |
| --------------------------- | -------------------------------------------------------------------------------- |
| Dataset selection           | Cyprus PROTEAS chosen (longitudinal, expert labels, RTP, radiomics)              |
| Timeline extraction         | `outputs/cyprus_patient_timelines.csv` — day gaps between all visits          |
| Split patient investigation | 5 patients (P04, P07, P17, P20, P23) with a/b = distinct spatial metastases      |
| Clinical data loading       | 47 rows, 21 raw columns from Excel                                               |
| Clinical data cleaning      | 5 quality fixes applied →`outputs/PROTEAS_Clinical_Cleaned.xlsx` (28 columns) |

**Clinical data quality fixes:**

1. **Dose trailing space:** `'20 Gy '` → `'20 Gy'`
2. **Prior Chemo case:** `'yes'` → `'Yes'` (standardized)
3. **Treatment hierarchy:** 7 raw treatment annotations → RS/FSRT binary groups + re-irradiation flag
4. **Lesion locations:** 37 raw labels → 7 standardized lobes + laterality + multi-lesion flag
5. **KPS eligibility:** P33a/b flagged — KPS=50 is below SRS threshold (typically ≥70)

### Activity B: Intensity Normalization ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 2.1)

| Test                 | Result                                                                          |
| -------------------- | ------------------------------------------------------------------------------- |
| Z-score validation   | Mean error = 0.0, Std error = 0.0 across all patients                           |
| Dynamic range check  | Consistent across modalities                                                    |
| **Conclusion** | **Dataset already Z-score normalized — no additional processing needed** |

### Activity C: Spatial Resampling ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 2.2)

| Test                 | Result                                                 |
| -------------------- | ------------------------------------------------------ |
| Resolution check     | 45/45 patients at 1×1×1 mm isotropic                 |
| Grid dimensions      | All 240×240×155 voxels                               |
| Orientation          | All LPS (Left-Posterior-Superior)                      |
| **Conclusion** | **Already standardized — no resampling needed** |

### Activity D: Longitudinal Image Alignment ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 3)

| Test                          | Checked               | Result                                                    |
| ----------------------------- | --------------------- | --------------------------------------------------------- |
| Intra-visit alignment         | 180 modality pairs    | 100% match                                                |
| Inter-visit alignment         | 564 cross-visit pairs | 100% match                                                |
| Center-of-mass shift          | All patient pairs     | Mean ~2 mm, max ~6 mm                                     |
| **Total affine checks** | **744**         | **All passed**                                      |
| **Conclusion**          |                       | **Already co-registered — no registration needed** |

### Activity E: Tumor Annotation Verification ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Section 4) + `Skull_Stripping_P28_P39.ipynb`

| Task                  | Result                                                                 |
| --------------------- | ---------------------------------------------------------------------- |
| Label validation      | 171/171 masks valid (subset of {0, 1, 2, 3})                           |
| Empty mask check      | 0 empty masks found                                                    |
| Label distribution    | {0,1,2,3}=84, {0,2,3}=72, {0,2}=14, {0,3}=1                            |
| Missing masks         | 16/187 visits without masks (likely resolved tumors after treatment)   |
| 3D overlay generation | T1c + colored tumor mask overlays in `outputs/3d_overlays/`          |
| Skull-stripping check | **P28-P39 not skull-stripped** (inconsistency discovered)        |
| Skull-stripping fix   | Brain masks applied to all 12 patients, P31 float32 precision resolved |
| Brain tissue analysis | Ventricles (10), WM (30), GM (40), CSF (50) characterized              |

### Activities F & G: Exploratory Data Analysis ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Sections 5-8)

#### Tumor Size and Location Analysis

| Finding              | Detail                                                                               |
| -------------------- | ------------------------------------------------------------------------------------ |
| Volume range         | 18 mm³ to 205,349 mm³ (high variability, CV > 2.0)                                 |
| Dominant subregion   | Edema (ED) typically largest, followed by Enhancing Tumor (ET)                       |
| Necrosis prevalence  | 49% of scans show necrosis (NCR label present)                                       |
| Most common location | Cerebellum (consistent with lung/breast primary cancer metastasis patterns)          |
| Spatial distribution | Supratentorial and infratentorial scatter — validates whole-brain analysis approach |

#### Temporal Trajectory Analysis

| Finding             | Detail                                                                           |
| ------------------- | -------------------------------------------------------------------------------- |
| Follow-up retention | 100% at baseline → 23% at 12 months                                             |
| Response patterns   | Progressive, stable, and responsive trajectories identified (RANO-like criteria) |
| Treatment groups    | RS (76.6%, n=36) vs FSRT (23.4%, n=11) — distinct strategies                    |
| KPS distribution    | Median 80%, 2 patients (P33a/b) below SRS threshold at KPS=50                    |

#### Demographics

| Finding             | Detail                                                                           |
| ------------------- | -------------------------------------------------------------------------------- |
| Age                 | Mean 60.0, median 61, range 39-82                                                |
| Gender              | 53% Female (25), 47% Male (22)                                                   |
| Primary cancer      | NSCLC 62% (29), Breast 36% (17), SCLC 2% (1)                                    |

#### Radiomic Feature Analysis

| Property           | Value                                                                               |
| ------------------ | ----------------------------------------------------------------------------------- |
| Total features     | 7,980 (7 categories × 4 masks × 4 modalities × 5 timepoints)                     |
| PCA analysis       | First 10 components capture ~70% variance                                           |
| Feature categories | First-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM, shape                                 |
| Cross-correlations | High inter-category correlation confirms redundancy — supports PCA-based reduction |

#### CT and Radiation Therapy Planning Analysis

| Finding             | Detail                                                                 |
| ------------------- | ---------------------------------------------------------------------- |
| CT Hounsfield units | Standard brain range (20-40 HU for GM/WM)                              |
| RTP dose validation | Hotspot = 120-130% of prescribed marginal dose (confirmed)             |
| Dose range          | 18-24 Gy (RS) and 3×12 Gy (FSRT) — consistent with clinical metadata |
| NaN handling        | RTP files contain background NaN — filtered safely                    |

### Activity H: Longitudinal Organization ✅

**Notebook:** `Phase1_Complete_EDA.ipynb` (Sections 1, 9)

| Deliverable               | Detail                                                               |
| ------------------------- | -------------------------------------------------------------------- |
| Timeline CSV              | `outputs/cyprus_patient_timelines.csv` — day gaps, visit ordering |
| Patient inventory         | `outputs/cyprus_inventory.csv` — 186-row per-timepoint inventory  |
| Follow-up retention curve | 100% → 98% → 87% → 62% → 38% → 23% (baseline through 12 months) |

---

## Deliverables

### 1. Reproducible Imaging Preprocessing Pipeline ✅

| File                                         | Description                                                       |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `notebooks/Skull_Stripping_P28_P39.ipynb`  | 5-step skull-stripping correction for 12 patients                 |
| `outputs/pre_skullstrip_backup/`           | Original MRI backups before correction (local machine not pushed) |
| `outputs/skull_stripping_verification.png` | Before/after visual comparison                                    |

> [!NOTE]
> The dataset was already preprocessed (normalized, resampled, aligned). The pipeline focused on identifying and correcting the P28-P39 skull-stripping inconsistency and validating that no additional preprocessing was needed.

### 2. Cleaned, Standardized, and Temporally Organized Datasets ✅

| File                                      | Description                              |
| ----------------------------------------- | ---------------------------------------- |
| `outputs/PROTEAS_Clinical_Cleaned.xlsx` | 47 rows, 28 columns (7 derived features) |
| `outputs/cyprus_patient_timelines.csv`  | Longitudinal timeline with day gaps      |
| `outputs/cyprus_inventory.csv`          | 186-row per-timepoint inventory          |
| `outputs/tumor_volumes.csv`             | Tumor volumes per scan (NCR, ED, ET, WT) |
| `outputs/radiomic_features.csv`         | 7,980 radiomic features                  |
| `outputs/3d_overlays/*.nii.gz`          | T1c + colored tumor mask 3D overlays     |

### 3. Exploratory Data Analysis Report ✅

| File                                    | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `notebooks/Phase1_Complete_EDA.ipynb` | Complete EDA notebook (40 cells, 9 sections)  |
| `notebooks/Phase1_EDA_Explanation.md` | Section-by-section methodology (180 lines)    |
| `Phase1_Clinical_Data_Analysis.md`    | Results + clinical interpretation (331 lines) |
| `outputs/eda/figures/`                | All EDA visualizations                        |

---

## Key Findings

1. **High-quality dataset:** Pre-normalized (Z-score), co-registered, 1 mm isotropic — minimal preprocessing needed
2. **Skull-stripping inconsistency:** P28-P39 not skull-stripped — discovered and corrected
3. **Clinical data:** 5 quality fixes applied; RS (76.6%) vs FSRT (23.4%) treatment groups identified
4. **KPS eligibility concern:** 2 patients (P33a/b) have KPS=50, below typical SRS threshold of 70
5. **Lesion locations:** Cerebellum most common — consistent with lung/breast primary metastasis patterns
6. **Temporal dropout:** 100% at baseline → 23% at 12 months (typical for brain met populations)
7. **RTP validation:** Hotspot doses = 120-130% of marginal dose (confirmed clinically appropriate)
8. **Radiomic redundancy:** 7,980 features highly correlated — PCA-based reduction justified
9. **Missing masks:** 16/187 visits lack tumor annotations — likely treatment-resolved tumors (positive clinical outcome)

---

## Files Produced

| File                                        | Description                  |
| ------------------------------------------- | ---------------------------- |
| `notebooks/Phase1_Complete_EDA.ipynb`     | Main EDA notebook            |
| `notebooks/Skull_Stripping_P28_P39.ipynb` | Skull-stripping fix          |
| `notebooks/Phase1_EDA_Explanation.md`     | Methodology guide            |
| `notebooks/Phase1_EDA_Report.md`          | Clinical results report      |
| `outputs/PROTEAS_Clinical_Cleaned.xlsx`   | Cleaned clinical data        |
| `outputs/cyprus_patient_timelines.csv`    | Longitudinal timeline        |
| `outputs/cyprus_inventory.csv`            | Per-timepoint inventory      |
| `outputs/tumor_volumes.csv`               | Tumor volumes                |
| `outputs/radiomic_features.csv`           | Radiomic features            |
| `outputs/3d_overlays/`                    | 3D NIfTI overlays            |
| `outputs/eda/figures/`                    | EDA figures                  |
| `PHASE1_PLAN.md`                          | Phase 1 activity plan        |
| `PHASE1_STATUS.md`                        | Phase 1 completion checklist |

---

## References

### Dataset

| # | Reference | How We Used It |
|---|---|---|
| [D1] | Flouri, D., Pattichis, C., et al. (2025). *Cyprus PROTEAS: A Longitudinal Brain Metastasis Dataset with Multi-class Segmentation, CT, and Radiation Therapy Planning Data.* Zenodo. DOI: [10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793) | **Primary dataset** — 45 patients, 187 MRI visits, 171 expert segmentations, CT, RTP, radiomics |
| [D2] | Moawad, A.W., et al. (2023). *Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population.* arXiv:2305.19369 | **BraTS segmentation protocol** — our Cyprus PROTEAS labels follow the BraTS convention (NCR=1, ET=2, ED=3) |
| [D3] | Grovik, E., et al. (2020). *Deep learning enables automatic detection and segmentation of brain metastases on multi-sequence MRI.* Journal of Magnetic Resonance Imaging, 51(1), 175-182 | **Stanford BrainMetShare** — alternative dataset evaluated but rejected (single timepoint, binary masks) |
| [D4] | Rudie, J.D., et al. (2024). *The University of California San Francisco Brain Metastases Stereotactic Radiosurgery (UCSF-BMSR) MRI Dataset.* Radiology: Artificial Intelligence, 6(1) | **UCSF-BMSR** — alternative dataset evaluated but rejected (single timepoint, binary masks, no treatment data) |

### Preprocessing Methodology

| # | Reference | How We Used It |
|---|---|---|
| [P1] | Bakas, S., et al. (2017). *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117 | **BraTS preprocessing pipeline** — Z-score normalization on brain-only voxels, 1mm isotropic resampling, co-registration protocol |
| [P2] | Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18, 203-211 | **Preprocessing best practices** — intensity normalization, resampling strategies for 3D medical images |
| [P3] | Kamnitsas, K., et al. (2017). *Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation.* Medical Image Analysis, 36, 61-78 | **Skull-stripping methodology** — brain mask application for removing extracranial tissue |

### Radiomic Features

| # | Reference | How We Used It |
|---|---|---|
| [R1] | van Griethuysen, J.J.M., et al. (2017). *Computational Radiomics System to Decode the Radiographic Phenotype.* Cancer Research, 77(21), e104-e107 | **PyRadiomics** — defines the 7 radiomic feature categories (shape, first-order, GLCM, GLDM, GLRLM, GLSZM, NGTDM) used in the dataset's 7,980 pre-extracted features |
| [R2] | Lambin, P., et al. (2017). *Radiomics: the bridge between medical imaging and personalized medicine.* Nature Reviews Clinical Oncology, 14(12), 749-762 | **Radiomics framework** — theoretical basis for using quantitative imaging features as biomarkers |
| [R3] | Aerts, H.J.W.L., et al. (2014). *Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach.* Nature Communications, 5, 4006 | **Radiomics validation** — demonstrated that radiomic features have prognostic value in cancer, motivating our Phase 1 feature analysis |

### Clinical Standards

| # | Reference | How We Used It |
|---|---|---|
| [C1] | Lin, N.U., et al. (2015). *Response Assessment in Neuro-Oncology (RANO) criteria for brain metastases.* The Lancet Oncology, 16(6), e270-e278 | **RANO-BM criteria** — our response classification (Progressive >+25%, Stable, Responsive >-25%) follows simplified RANO-BM volume-based thresholds |
| [C2] | Vogelbaum, M.A., et al. (2022). *Treatment for Brain Metastases: ASCO-SNO-ASTRO Guideline.* Journal of Clinical Oncology, 40(5), 492-516 | **SRS eligibility** — KPS ≥70 threshold used to flag P33a/b; RS vs FSRT treatment selection criteria |
| [C3] | Yamamoto, M., et al. (2014). *Stereotactic radiosurgery for patients with multiple brain metastases (JLGK0901): a multi-institutional prospective observational study.* The Lancet Oncology, 15(4), 387-395 | **SRS dosimetry** — our dose cross-reference (marginal vs hotspot, 120-130% ratio) follows standard SRS practice |

### Software and Tools

| Tool | Version | Purpose |
|---|---|---|
| **NiBabel** | 5.x | NIfTI file I/O, affine matrix extraction, orientation checking |
| **NumPy** | 1.26+ | Volume computation (voxel counting), Z-score validation, array operations |
| **Pandas** | 2.x | Clinical data cleaning, timeline construction, CSV/Excel I/O |
| **Matplotlib** / **Seaborn** | 3.8+ / 0.13+ | All EDA visualizations (violin plots, scatter, pie charts, heatmaps) |
| **scikit-learn** | 1.4+ | PCA for radiomic feature reduction |
| **openpyxl** | 3.x | Excel file reading/writing for clinical metadata |
| **ITK-SNAP** / **3D Slicer** | — | 3D visualization of NIfTI overlays (external tools) |
