# Phase 1 EDA -- Results and Clinical Interpretation

**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)
**Date:** March 2026

---

## 1. Dataset Overview

| Metric                    | Value                                                     |
| ------------------------- | --------------------------------------------------------- |
| Patient directories       | 45 (40 unique + 5 a/b spatial splits)                     |
| Total MRI timepoints      | 186                                                       |
| Expert segmentation masks | 171 (15 visits without masks)                             |
| CT scans                  | 45/45 (100%)                                              |
| RT dose plans             | 45/45 (100%)                                              |
| Resolution                | 1x1x1 mm isotropic, 240x240x155                           |
| Alignment                 | 728/728 affine checks passed                              |
| Skull-stripping           | P01-P27, P40 stripped; P28-P39 corrected with brain masks |

---

## 2. Patient Demographics

| Variable                 | Value                                   |
| ------------------------ | --------------------------------------- |
| **Age**            | Mean 62.4 +/- 10.5 years (range: 38-82) |
| **Gender**         | ~60% Male, ~40% Female                  |
| **Race**           | 100% White/Caucasian (Cyprus cohort)    |
| **Primary cancer** | NSCLC 60%, Breast 30%, SCLC 10%         |

### Tumour Histology

- **NSCLC** (Non-Small Cell Lung Cancer): Most common lung cancer (~85% of all lung cancers). The dominant primary in this cohort, consistent with NSCLC being the leading cause of brain metastases worldwide.
- **SCLC** (Small Cell Lung Cancer): Aggressive, high rate of brain metastasis. Only ~10% of this cohort.
- **Breast**: Second most common source of brain mets. ~30% of cohort.

---

## 3. Treatment Distribution

### Two Treatment Groups

| Group                             | N  | %     | Description                            |
| --------------------------------- | -- | ----- | -------------------------------------- |
| **RS (Radiosurgery)**       | 36 | 76.6% | Single-fraction, high dose (~18-24 Gy) |
| **FSRT (Fractionated SRT)** | 11 | 23.4% | Multiple fractions (typically 3x8 Gy)  |

RS and FSRT are **distinct treatment strategies** -- they should NOT be merged unless studying stereotactic techniques as a whole.

### RS Subgroup Detail (36 cases)

| Annotation                 | N  | Meaning                                                      |
| -------------------------- | -- | ------------------------------------------------------------ |
| RS                         | 33 | Standard single-fraction radiosurgery                        |
| RS (DP after previous WBR) | 1  | RS for distant progression after prior Whole Brain Radiation |
| RS after previous PCI      | 1  | RS after Prophylactic Cranial Irradiation                    |
| RS (previous WBR)          | 1  | RS in patient with WBR history                               |

The 3 rare variants are RS in **re-irradiation** scenarios -- same treatment modality but clinically different context (prior radiation affects tissue tolerance).

### FSRT Subgroup Detail (11 cases)

| Annotation                       | N | Meaning                                      |
| -------------------------------- | - | -------------------------------------------- |
| FSRT stereotactic boost after OP | 9 | Post-surgical fractionated boost             |
| FSRT (DP after Cyberknife)       | 1 | FSRT after prior Cyberknife SRS failure      |
| FSRT due to size                 | 1 | Lesion too large for safe single-fraction RS |

### Re-irradiation Cases

13 patients (28%) had prior radiation treatment (WBR, PCI, or Cyberknife). These cases should be flagged as a covariate -- prior radiation significantly affects tissue tolerance and outcomes.

---

## 4. Dose Distribution

| Dose  | N  | Treatment                        | Context                                        |
| ----- | -- | -------------------------------- | ---------------------------------------------- |
| 20 Gy | 33 | RS (1 fraction)                  | Standard SRS dose                              |
| 35 Gy | 11 | FSRT (3 fractions ~11.7 Gy each) | Total fractionated dose                        |
| 18 Gy | 3  | RS (1 fraction)                  | Lower dose (smaller lesions or re-irradiation) |

**Important:** The Excel column "Rx dose at tumor margins" reports the **marginal** dose (at the tumor boundary). The RTP NIfTI files show that the actual maximum dose at the tumor **center** (hotspot) is ~25 Gy for a 20 Gy marginal prescription. This is standard in SRS -- the hotspot is typically 120-130% of the marginal dose.

---

## 5. Performance Status

### WHO Performance Status (43 valid, 4 missing)

| Score | Meaning                             | N  | %     |
| ----- | ----------------------------------- | -- | ----- |
| 0     | Fully active, no restrictions       | 1  | 2.3%  |
| 1     | Light activity only, ambulatory     | 27 | 62.8% |
| 2     | Self-care only, unable to work      | 13 | 30.2% |
| 3     | Limited self-care, mostly bedridden | 2  | 4.7%  |

Mean WHO PS = 1.4 -- cohort sits between "restricted but ambulatory" and "self-care only."

**Clinical concern:** ~35% are WHO 2-3, which is a functionally compromised population. WHO 3 patients are borderline for aggressive treatment.

### Karnofsky Performance Status (43 valid, 4 missing)

| KPS | Meaning                          | N  | %     |
| --- | -------------------------------- | -- | ----- |
| 50  | Requires considerable assistance | 2  | 4.7%  |
| 70  | Cares for self, cannot work      | 8  | 18.6% |
| 80  | Normal activity with effort      | 20 | 46.5% |
| 90  | Minor symptoms, near normal      | 12 | 27.9% |
| 100 | Fully normal                     | 1  | 2.3%  |

Mean KPS = 80% -- generally acceptable for SRS.

### Critical Finding: KPS Eligibility

KPS >= 70 is the standard SRS eligibility threshold. **2 patients (P33a, P33b) have KPS = 50**, below threshold. These patients likely received treatment under exceptional/compassionate circumstances or with palliative intent.

### WHO vs KPS Correlation

The expected mapping (WHO 0 = KPS 90-100, WHO 1 = KPS 70-80, WHO 2 = KPS 50-60) shows slight discordance: KPS mean (80) is higher than WHO 1.4 would predict. This is common due to inter-rater variability.

**Recommendations:**

- Flag KPS=50 cases as a subgroup (may skew outcomes)
- Handle the 4 missing values carefully (missing PS data is often non-random)
- Use KPS as primary PS variable (finer granularity, more common in SRS literature)

---

## 6. Lesion Locations

### Data Quality Issue

The raw data has **37 unique location labels** due to inconsistent annotation:

- "cerebellum", "Rt Cerebellum", "Lt cerebellum" = same region, 3 labels
- "Lt parietal", "Lt Parietal" = same, different capitalization
- Multi-lesion entries: "Lt parietal, and 2 lesions in cerebellum"

### Standardized Distribution (after cleaning)

| Lobe                 | N  | Clinical Note                                    |
| -------------------- | -- | ------------------------------------------------ |
| **Cerebellum** | 12 | Most common brain met site (lung/breast primary) |
| **Parietal**   | 8  | Common supratentorial location                   |
| **Frontal**    | 6  | Common supratentorial location                   |
| **Temporal**   | 3  | --                                               |
| **Occipital**  | 2  | --                                               |
| **Thalamus**   | 1  | Deep location, technically challenging for SRS   |
| **Mixed**      | 15 | Multiple lobes involved                          |

### Laterality

| Side                | N  |
| ------------------- | -- |
| Left                | 16 |
| Right               | 14 |
| Bilateral           | 9  |
| Midline/Unspecified | 8  |

Multi-lesion patients (3 cases) had oligometastatic disease treated in the same session -- important for dosimetric complexity and outcomes analysis.

---

## 7. Prior Chemotherapy

After case normalization: **Yes: 45 (95.7%), No: 2 (4.3%)**

Nearly all patients received systemic chemotherapy before radiation. This is expected -- brain metastases typically appear late in cancer progression, after systemic treatment has been tried.

---

## 8. Extracranial Disease

| Status                       | Meaning                                |
| ---------------------------- | -------------------------------------- |
| Stable extracranial          | Cancer outside brain is not growing    |
| Progressive extracranial     | Cancer outside brain is growing        |
| NED (No Evidence of Disease) | No detectable cancer outside the brain |

---

## 9. Follow-up Retention

| Timepoint | Available | %    |
| --------- | --------- | ---- |
| Baseline  | 47/47     | 100% |
| 6 weeks   | 46/47     | 98%  |
| 3 months  | 41/47     | 87%  |
| 6 months  | 29/47     | 62%  |
| 9 months  | 18/47     | 38%  |
| 12 months | 11/47     | 23%  |

Significant dropout at 6+ months is clinically expected -- patients may be deceased, transferred to other care, or lost to follow-up. This affects longitudinal analysis power.

---

## 10. Tumor Characteristics

### Segmentation Label Distribution

| Labels    | N  | Meaning                                     |
| --------- | -- | ------------------------------------------- |
| {0,1,2,3} | 84 | All subregions present (NCR + ET + ED)      |
| {0,2,3}   | 72 | ET + ED, no necrotic core                   |
| {0,2}     | 14 | ET only (no edema, no necrosis)             |
| {0,3}     | 1  | ED only (edema without enhancing component) |

15 visits have no mask at all -- likely complete tumor response after treatment.

### Volume Statistics

Whole tumor volumes show right-skewed distribution (most tumors small, few very large). Coefficient of Variation > 1 indicates high inter-patient heterogeneity, which is clinically expected in brain metastases.

### Response Classification

Using RANO-like criteria (>25% change from baseline):

- **Progressive**: Volume increase >25%
- **Stable**: Volume change within 25%
- **Responsive**: Volume decrease >25%

---

## 11. CT and Radiotherapy Plans

### CT Files (`P??_CT.nii.gz`)
Computed Tomography scans in Hounsfield Units (HU): air=-1000, water=0, bone=+1000. Used for electron density mapping in radiation dose calculation.

### RTP Files (`P??_RTP.nii.gz`) -- How to Interpret

Each voxel = absorbed radiation dose in Grays (Gy). When opened in a viewer:

| What You See             | What It Means                                     |
| ------------------------ | ------------------------------------------------- |
| Bright center            | Tumor location -- highest dose (hotspot)          |
| Skull-shaped halo        | Entrance dose -- beams passing through cranium    |
| Streak patterns          | Individual beam paths (SRS uses 5-10+ beams)      |
| Dark regions             | Spared brain tissue (minimal radiation)           |
| NaN voxels               | Outside dose calculation grid (filtered out)      |

The skull shape appears because multiple radiation beams enter from different angles through the cranium, all converging on the tumor.

### Prescribed vs Actual Dose (Cross-Reference)

| Prescribed (Excel) | Actual Hotspot (RTP) | Ratio | Treatment |
| ------------------- | -------------------- | ----- | --------- |
| 18 Gy               | 22.7 Gy              | 1.26x | RS        |
| 20 Gy               | 26.0 Gy              | 1.30x | RS        |
| 35 Gy               | 39.1 Gy              | 1.12x | FSRT      |

The hotspot is always 112-130% of the marginal dose -- this is standard and expected. FSRT has a lower ratio because fractionation produces more uniform dose distribution.

### Clinical Significance

The RTP + MRI overlay is central to this project:
1. **Radiation necrosis** occurs in the high-dose zone (>12 Gy single fraction)
2. **Tumor recurrence** outside the high-dose zone = treatment failure
3. Distinguishing necrosis from recurrence on follow-up MRI is the core diagnostic challenge

---

## 11b. Radiomic Feature Analysis

### Data Structure
7,980 features extracted across 4 dimensions:
- **4 masks**: all (whole tumor), necrosis, oedema, tumor (enhancing)
- **4 modalities**: FLAIR, T1, T1c, T2
- **5 timepoints**: Baseline + 4 follow-ups
- **7 categories**: shape, firstorder, GLCM, GLDM, GLRLM, GLSZM, NGTDM

### Feature Categories

| Category | N features | What It Captures |
| -------- | ---------- | ---------------- |
| Shape | 14 | 3D tumor geometry (volume, sphericity, elongation) |
| First-order | 16 | Intensity statistics (mean, entropy, skewness) |
| GLCM | 24 | Spatial texture from neighboring voxel pairs |
| GLDM | 14 | Regional homogeneity (connected component sizes) |
| GLRLM | 16 | Directional texture (run lengths of same intensity) |
| GLSZM | 16 | 3D texture zones (connected regions of same intensity) |
| NGTDM | 5 | Visual texture perception (coarseness, complexity) |

### Clinical Relevance
Radiomic features can distinguish radiation necrosis from recurrence:
- Recurrent tumors: high entropy, low uniformity (heterogeneous)
- Radiation necrosis: low entropy, high uniformity (more uniform)
- Delta-radiomics (temporal feature changes) predict treatment response

---

## 12. Data Quality Issues Resolved

| Issue                            | Fix                                        | Impact                 |
| -------------------------------- | ------------------------------------------ | ---------------------- |
| Dose `'20 Gy '` trailing space | `.str.strip()`                           | 1 case reclassified    |
| Prior Chemo `'yes'`/`'Yes'`  | `.str.capitalize()`                      | 19 cases unified       |
| 7 treatment annotations          | Hierarchical RS/FSRT + re-irradiation flag | Clean for analysis     |
| 37 location labels               | 7 lobes + side + multi-lesion flag         | Analyzable categories  |
| KPS=50 below threshold           | Flagged with `KPS_Below_Threshold`       | 2 cases identified     |
| P28-P39 not skull-stripped       | Brain mask applied, backups saved          | 12 patients fixed      |
| P31 float precision              | Explicit zeroing + float32                 | 1 case fixed           |
| RTP NaN values                   | `np.isfinite()` filtering                | All 45 RTP files valid |

---

## 13. Cleaned Data Outputs

| File                                         | Description                         |
| -------------------------------------------- | ----------------------------------- |
| `outputs/PROTEAS_Clinical_Cleaned.xlsx`    | 47 rows, 28 columns (7 new derived) |
| `outputs/cyprus_patient_timelines.csv`     | Longitudinal timeline               |
| `outputs/3d_overlays/*.nii.gz`             | 3D tumor overlay files              |
| `outputs/skull_stripping_verification.png` | Before/after comparison             |
| `outputs/eda/figures/*.png`                | All EDA visualizations              |

### New Columns in Cleaned Excel

| Column                  | Values                                                              |
| ----------------------- | ------------------------------------------------------------------- |
| `Treatment_Group`     | RS or FSRT                                                          |
| `Re_Irradiation`      | True/False                                                          |
| `Lesion_Lobe`         | Frontal, Parietal, Temporal, Occipital, Cerebellum, Thalamus, Mixed |
| `Lesion_Side`         | Left, Right, Bilateral, Midline                                     |
| `Multi_Lesion`        | True/False                                                          |
| `KPS_Below_Threshold` | True/False (< 70)                                                   |
| `Dose_Gy`             | 18, 20, or 35                                                       |
