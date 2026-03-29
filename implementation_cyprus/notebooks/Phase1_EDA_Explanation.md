# Phase 1 EDA -- Methodology and Explanation

**Notebook:** `Phase1_Complete_EDA.ipynb`  
**Dataset:** Cyprus PROTEAS Brain Metastases (Flouri et al., 2025)  
**Patients:** 40 unique (45 directories) | **Timepoints:** 186 | **Modalities:** T1, T2, T1c, FLAIR + CT + RTP

---

## Medical Background

### Imaging Modalities

| Modality | Full Name | What It Shows |
|----------|-----------|---------------|
| **T1** | T1-weighted MRI | Structural anatomy; grey matter appears darker than white matter |
| **T1c** | T1 + Gadolinium contrast | Enhancing tumor lights up (contrast leaks through broken blood-brain barrier) |
| **T2** | T2-weighted MRI | Fluid and edema appear bright; good for detecting swelling |
| **FLAIR** | Fluid-Attenuated Inversion Recovery | Like T2 but CSF is dark; better for edema near ventricles |
| **CT** | Computed Tomography | X-ray based; shows skull geometry in Hounsfield Units (HU) for radiation planning |
| **RTP** | Radiotherapy Treatment Plan (RT Dose) | 3D volume where each voxel = radiation dose in Grays (Gy); multiple beams converge on tumor |

### Tumor Segmentation Labels (from paper)

| Label | Structure | Color |
|-------|-----------|-------|
| 0 | Background | Transparent |
| 1 | Necrotic Core (NCR) -- dead tissue at tumor center | Red |
| 2 | Enhancing Tumor (ET) -- actively growing region | Blue |
| 3 | Peritumoral Edema (ED) -- swelling around tumor | Green |

Note: This differs from BraTS standard (which uses label 4 for ET).

### Brain Tissue Labels (brain mask files only)

| Label | Structure | Clinical Utility |
|-------|-----------|------------------|
| 10 | Ventricles | Volume changes indicate mass effect (tumor pushing brain tissue) or hydrocephalus |
| 30 | White Matter (WM) | Tumor proximity to WM tracts affects neurological function and surgical risk |
| 40 | Grey Matter (GM) | Tumor-to-GM ratio contextualizes burden; GM atrophy may indicate radiation damage |
| 50 | CSF | Volume changes indicate edema progression or blocked CSF flow |

### Treatment Types

| Abbreviation | Full Name | Description |
|-------------|-----------|-------------|
| **RS / SRS** | (Stereotactic) Radiosurgery | Single-fraction, high dose (~18-24 Gy). Multiple beams converge on tumor. |
| **FSRT** | Fractionated Stereotactic RT | Multiple fractions (typically 3x8 Gy). Used for larger tumors or near sensitive structures. |
| **WBR** | Whole Brain Radiation | Entire brain irradiated; used as salvage or prophylaxis. |
| **PCI** | Prophylactic Cranial Irradiation | Preventive whole-brain radiation (common in SCLC). |

RS and FSRT are distinct treatment strategies and should NOT be merged in analysis.

---

## Section-by-Section Methodology

### Section 1: Dataset Acquisition and Timeline Extraction (Activities A, H)

- Scans `Data/Cyprus-PROTEAS-zips/` to find all 45 patient directories
- Extracts visit dates from DICOM folder names (e.g., `study_2020-03-16`)
- Computes day gaps between consecutive visits and from baseline
- Outputs: `cyprus_patient_timelines.csv`
- Investigates split patients (P04, P07, P17, P20, P23): per paper, all represent spatially distinct brain metastases requiring separate radiotherapy rounds
- Cross-modality visualization: 4 random patients, all 6 data types side-by-side

### Section 2: Preprocessing Validation (Activities B, C)

**2.1 Intensity Normalization:** Z-score normalization `z = (x - mean) / std` on brain-only voxels. Validates mean=0, std=1 across all patients. Three diagnostic plots.

**2.2 Spatial Resampling:** Verifies 1x1x1 mm isotropic, 240x240x155 grid, LPS orientation. Result: 45/45 pass.

**2.3 Skull-Stripping Consistency:** Measures non-zero voxel percentage per patient. P01-P27 and P40 are properly skull-stripped (~14-19%), P28-P39 retain extracranial tissue (~44-85%). Brain masks available to fix. Separate notebook: `Skull_Stripping_P28_P39.ipynb`.

### Section 3: Longitudinal Alignment (Activity D)

Compares NIfTI affine matrices and brain Center-of-Mass shifts:
- Intra-visit (modalities within same session): 176/176 match
- Inter-visit (follow-ups vs baseline): 552/552 match
- CoM shift: ~2 mm average (within clinical tolerance)
- Total: 728 checks. No additional registration needed.

### Section 4: Tumor Annotation Verification (Activity E)

**Label validation:** 171/171 masks valid, 0 empty. Distribution: {0,1,2,3}=84, {0,2,3}=72, {0,2}=14, {0,3}=1.

**Missing masks:** 15/186 visits (8 patients) lack masks. Likely: complete tumor response, lesion too small, or scan not annotated. All visits shown in visualizations (with/without mask).

**3D Overlays:** Generated `{patient}_t1c_with_mask_baseline.nii.gz` files (T1c + colored mask at 50% opacity) for 3D viewing in ITK-SNAP/Slicer.

**Brain tissue analysis:** Volume distributions of ventricles, WM, GM, CSF from brain mask files.

### Section 5: Tumor Characteristics EDA (Activities F, G)

**5.1 Volumes:** NCR, ET, ED, WT computed from all 171 masks (1 voxel = 1 mm^3 at isotropic resolution). High inter-patient variability (CV > 1).

**5.2 Composition and Spatial:** Subregion pie chart, violin plots, centroid scatter.

**5.3 Temporal Trajectories:** Volume tracking over time. RANO-like response classification (Progressive >+25%, Stable, Responsive >-25%).

### Section 6: Radiomic Feature Analysis (Activities F, G)

`PROTEAS-MRI_radiomics_data.xlsx`: 7,980 rows of pre-extracted radiomic features. Each feature name encodes:

`{mask}__{modality}__{timepoint}__original_{category}_{feature}`

- **4 masks**: mask_all (whole tumor), mask_necrosis (NCR), mask_oedema (edema), mask_tumor (enhancing)
- **4 modalities**: fla (FLAIR), t1, t1c, t2
- **5 timepoints**: Baseline, follow_up_1..4
- **7 categories** (105 unique features per combination):

| Category | N | What It Captures |
|----------|---|-----------------|
| Shape (14) | Geometry | Volume, sphericity, elongation, surface area, max diameter |
| First-order (16) | Intensity stats | Mean, entropy, skewness, kurtosis, energy, uniformity |
| GLCM (24) | Pair texture | Co-occurrence of neighboring voxel intensities (contrast, correlation, joint entropy) |
| GLDM (14) | Regional uniformity | Connected regions of same gray level (dependence emphasis) |
| GLRLM (16) | Directional texture | Runs of same intensity in one direction (run length emphasis) |
| GLSZM (16) | 3D zone texture | Connected 3D zones of same gray level (size zone emphasis) |
| NGTDM (5) | Perceived texture | Coarseness, contrast, busyness, complexity, strength |

Analysis: (1) structural breakdown, (2) per-category distribution histograms, (3) correlation heatmap, (4) PCA with variance explained + scatter by mask type.

### Section 7: Clinical Metadata -- Deep Analysis and Cleaning (Activity A)

Loads raw Excel (47 rows, 21 columns). Applies 5 data quality fixes:
1. Dose trailing space (`'20 Gy '` -> `'20 Gy'`)
2. Prior Chemo case normalization (`'yes'` -> `'Yes'`)
3. Treatment hierarchy: 7 annotations -> RS (36) + FSRT (11) + re-irradiation flag
4. Lesion locations: 37 raw labels -> 7 standardized lobes + side + multi-lesion flag
5. KPS eligibility flag (2 patients with KPS=50 below SRS threshold)

Saves cleaned version: `outputs/PROTEAS_Clinical_Cleaned.xlsx` (28 columns).

Produces per-column visualizations and 6 clinically relevant cross-tabulations.

### Section 8: CT and RTP Analysis

**CT (`P??_CT.nii.gz`):** Computed Tomography in Hounsfield Units. Used for electron density in dose calculation. Distributions analyzed for all 45 patients.

**RTP (`P??_RTP.nii.gz`):** 3D radiation dose maps. Each voxel = dose in Gy.

How to interpret RTP visually:
- **Bright center** = tumor hotspot (maximum dose)
- **Skull-shaped halo** = entrance dose from beams passing through cranium
- **Streak patterns** = individual beam paths (SRS uses 5-10+ beams from different angles)
- **NaN voxels** = outside the dose calculation grid

Dose cross-reference with clinical metadata:
- 18 Gy marginal -> 22.7 Gy hotspot (ratio 1.26x)
- 20 Gy marginal -> 26.0 Gy hotspot (ratio 1.30x)
- 35 Gy marginal (FSRT) -> 39.1 Gy hotspot (ratio 1.12x, lower because fractionation is more uniform)

### Section 9: Pipeline Summary

Summary table with all key metrics, Phase 1 activity checklist, skull-stripping status, and next steps.

---

## Key Technical Decisions

1. **Z-score normalization** -- standard in BraTS pipelines
2. **No additional registration** -- dataset already co-registered (728/728 checks)
3. **Label convention from paper** -- {0,1,2,3} where 1=NCR, 2=ET, 3=ED
4. **NaN-safe RTP analysis** -- `np.isfinite()` filtering
5. **Clinical data cleaning** -- hierarchical treatment, standardized locations, KPS flags
6. **Skull-stripping correction** -- explicit zeroing + float32 (handles P31 edge case)

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/PROTEAS_Clinical_Cleaned.xlsx` | Cleaned clinical data (28 columns) |
| `outputs/cyprus_patient_timelines.csv` | Longitudinal timeline with day gaps |
| `outputs/3d_overlays/*.nii.gz` | T1c + mask 3D overlay files |
| `outputs/eda/figures/*.png` | All EDA visualizations |
| `outputs/skull_stripping_verification.png` | Before/after skull-strip comparison |
| `outputs/pre_skullstrip_backup/` | Original MRI backups for P28-P39 |
