# 📁 Phase 1 Outputs Guide
## Yale Brain Metastases Longitudinal Dataset

**Generated**: March 10, 2026  
**Notebook**: `implementation/notebooks/01_data_audit_inventory.ipynb`  
**Status**: ✅ All 20 code cells executed clean — 0 errors

---

## 📊 Output Files Summary

| File | Rows / Entries | Size (approx) | Purpose |
|------|---------------|---------------|---------|
| `data_inventory.csv` | 11,877 rows × 35 cols | ~5 MB | Master ledger of every visit |
| `acquisition_parameters.csv` | 33,804 rows × 11 cols | ~4 MB | Per-file MRI sequence parameters |
| `metadata.json` | 1,430 patients | ~15 MB | Clinical + scanner + per-visit acquisition params |
| `patient_timelines.json` | 1,430 patients | ~5 MB | Sorted visit sequences with temporal deltas |
| `splits.json` | 1,430 patients | ~50 KB | Locked train/val/test split (seed=42) |
| `AUDIT_REPORT.md` | — | ~5 KB | Human-readable audit summary |
| `eda/figures/01_full_eda_dashboard.png` | — | ~500 KB | 6-panel EDA dashboard |
| `eda/figures/02_top20_patients.png` | — | ~200 KB | Top 20 patients by visit count |
| `eda/figures/03_modality_heatmap.png` | — | ~200 KB | Modality availability heatmap |

---

## 1. `data_inventory.csv` — Master Visit Ledger

**Every row = one patient visit.** This is the primary input for all downstream DataLoaders.

### Schema (35 columns)

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Patient identifier, e.g. `YG_01M98EKKAR50` |
| `visit_date` | str | ISO date of the visit, e.g. `2016-11-13` |
| `visit_dir` | str | Absolute path to the visit folder |
| `visit_idx` | int | 0-indexed visit number within this patient (0 = baseline) |
| `days_since_baseline` | int | Days from patient's first visit to this visit |
| `interval_days` | float | Days between this visit and the previous one (NaN for visit_idx=0) |
| `same_day_flag` | bool | True if interval < 2 days (possible duplicate scan) |
| `has_PRE` | bool | T1 pre-contrast scan present |
| `has_POST` | bool | T1 post-contrast (T1c) scan present |
| `has_T2` | bool | T2 scan present |
| `has_FLAIR` | bool | FLAIR scan present |
| `n_modalities` | int | Number of modalities present (0–4) |
| `training_usability` | str | See usability labels below |
| `all_headers_ok` | bool | All NIfTI headers loaded without errors |
| `n_nii_files` | int | Number of `.nii.gz` files found in this visit folder |
| `path_PRE` | str | Absolute path to PRE NIfTI (NaN if absent) |
| `path_POST` | str | Absolute path to POST NIfTI (NaN if absent) |
| `path_T2` | str | Absolute path to T2 NIfTI (NaN if absent) |
| `path_FLAIR` | str | Absolute path to FLAIR NIfTI (NaN if absent) |
| `echo_time_ms_flair` | float | Echo time (TE) in ms for FLAIR scan |
| `echo_time_ms_post` | float | Echo time (TE) in ms for POST scan |
| `echo_time_ms_pre` | float | Echo time (TE) in ms for PRE scan |
| `echo_time_ms_t2` | float | Echo time (TE) in ms for T2 scan |
| `inversion_time_ms_flair` | float | Inversion time (TI) in ms for FLAIR |
| `inversion_time_ms_post` | float | Inversion time (TI) in ms for POST |
| `inversion_time_ms_pre` | float | Inversion time (TI) in ms for PRE |
| `inversion_time_ms_t2` | float | Inversion time (TI) in ms for T2 |
| `repetition_time_ms_flair` | float | Repetition time (TR) in ms for FLAIR |
| `repetition_time_ms_post` | float | Repetition time (TR) in ms for POST |
| `repetition_time_ms_pre` | float | Repetition time (TR) in ms for PRE |
| `repetition_time_ms_t2` | float | Repetition time (TR) in ms for T2 |
| `slice_thickness_mm_flair` | float | Slice thickness in mm for FLAIR |
| `slice_thickness_mm_post` | float | Slice thickness in mm for POST |
| `slice_thickness_mm_pre` | float | Slice thickness in mm for PRE |
| `slice_thickness_mm_t2` | float | Slice thickness in mm for T2 |

### `training_usability` Labels

| Label | Count | % | Meaning | Use in Training |
|-------|-------|---|---------|-----------------|
| `FULL` | 4,364 | 36.7% | All 4 modalities present (PRE, POST, T2, FLAIR) | Full supervised training |
| `PARTIAL_GOOD` | 2,278 | 19.2% | POST + FLAIR present (± PRE, T2) | Supervised with modality dropout |
| `POST_ONLY` | 2,352 | 19.8% | POST present, FLAIR absent | Self-supervised pretraining only |
| `UNUSABLE` | 2,883 | 24.3% | No POST (T1c) at all | Temporal context / excluded from training |
| `CORRUPT` | 0 | 0.0% | NIfTI header load error | Excluded entirely |

> **Supervised training pool**: `FULL` + `PARTIAL_GOOD` = **6,642 visits (55.9%)**  
> **Pretraining pool**: above + `POST_ONLY` = **8,994 visits (75.7%)**

### Downstream Usage
- **Phase 2 CNN DataLoader**: reads `path_POST` for all rows where `split='train'` and `training_usability != 'UNUSABLE'`
- **Phase 3 TaViT**: reads `path_POST`, `path_FLAIR` for `FULL` + `PARTIAL_GOOD` rows, grouped by `patient_id`, sorted by `days_since_baseline`
- **ComBat harmonization**: uses `slice_thickness_mm_*`, `repetition_time_ms_*` columns to detect scanner-driven batch effects

---

## 2. `acquisition_parameters.csv` — Per-File MRI Sequence Parameters

**Every row = one NIfTI file.** Long-format (4 rows per visit maximum).

### Schema (11 columns)

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Patient identifier |
| `visit_date` | str | ISO date |
| `modality` | str | `PRE`, `POST`, `T2`, or `FLAIR` |
| `file_name` | str | NIfTI filename, e.g. `YG_01M98EKKAR50_2016-11-13_10-16-23_PRE.nii.gz` |
| `sequence_class` | str | High-level sequence type (e.g. `PRE`, `POST`) |
| `sequence_tags` | str | Scanner sequence tag string (e.g. `ax_t1_se_pre`) |
| `slice_thickness_mm` | float | Slice thickness in mm |
| `spacing_between_slices_mm` | float | Spacing between slices in mm |
| `repetition_time_ms` | float | TR in milliseconds |
| `echo_time_ms` | float | TE in milliseconds |
| `inversion_time_ms` | float | TI in milliseconds (NaN for sequences without inversion) |

### Stats
- 33,804 rows (one per NIfTI file)
- 100% match rate to `data_inventory.csv` file paths
- Unique sequence tags: varies by modality — see Cell 22 in notebook for per-modality breakdown

### Downstream Usage
- **ComBat / harmonization (Phase 3)**: TR, TE, TI, slice thickness are key covariates for scanner harmonization
- **Data quality filtering**: flag outlier voxel spacings or unrealistic parameter values before training

---

## 3. `metadata.json` — Clinical + Scanner + Acquisition Metadata

**Key = `patient_id`. One entry per patient containing clinical info, scanner info, and per-visit acquisition parameters.**

### Structure

```json
{
  "YG_01M98EKKAR50": {
    "age_at_first_imaging": 71,
    "sex": "Female",
    "scanner_vendor": "SIEMENS",
    "scanner_site": "Yale",
    "visits": [
      {
        "visit_date": "2016-11-13",
        "study_datetime": "2016-11-13_10-16-23",
        "vendor": "SIEMENS",
        "model": "Verio",
        "field_strength_tesla": 3.0,
        "2d_3d_acquisition": "2D",
        "scanner_site": "Yale",
        "pre_included": 1.0,
        "post_included": 1.0,
        "t2_included": 1.0,
        "flair_included": 1.0,
        "acquisition_params": {
          "FLAIR": {
            "sequence_tags": "ax_flair_blade",
            "slice_thickness_mm": 5.0,
            "spacing_between_slices_mm": 5.0,
            "repetition_time_ms": 9000.0,
            "echo_time_ms": 94.0,
            "inversion_time_ms": 2500.0
          },
          "POST": { ... },
          "PRE": { ... },
          "T2": { ... }
        }
      }
    ]
  }
}
```

### Coverage
- **1,430 patients** with clinical data (100% coverage)
- **11,883 visits** enriched with per-visit `acquisition_params`
- Fields available: `age_at_first_imaging`, `sex`, `scanner_vendor`, `scanner_model`, `field_strength_tesla`, `2d_3d_acquisition`, `scanner_site`

### Downstream Usage
- **Phase 4 LLM narrative**: `age`, `sex` → clinical context for LLM report generation
- **Phase 3 ComBat harmonization**: `field_strength_tesla`, `vendor`, `model` → batch effect covariates
- **Stratified analysis**: split by scanner vendor/field strength to analyze domain shift

---

## 4. `patient_timelines.json` — Longitudinal Visit Sequences

**Key = `patient_id`. Sorted chronologically.**

### Structure

```json
{
  "YG_01M98EKKAR50": {
    "n_visits": 3,
    "first_visit": "2016-11-13",
    "last_visit": "2016-12-20",
    "followup_days": 37,
    "visits": [
      {
        "visit_idx": 0,
        "visit_date": "2016-11-13",
        "days_since_baseline": 0,
        "interval_days": null,
        "usability": "FULL",
        "modalities": ["PRE", "POST", "T2", "FLAIR"]
      },
      {
        "visit_idx": 1,
        "visit_date": "2016-11-27",
        "days_since_baseline": 14,
        "interval_days": 14,
        "usability": "POST_ONLY",
        "modalities": ["POST"]
      }
    ]
  }
}
```

### Dataset-Level Stats
| Metric | Value |
|--------|-------|
| Patients | 1,430 |
| Median visits per patient | 6 |
| Max visits (YG_C9D2TEGNY08A) | 66 visits over 11 years |
| Median follow-up | 288 days (~9.6 months) |
| Max follow-up | 5,061 days (13.9 years) |
| Patients with only 1 visit | 131 |
| Patients with ≥ 3 visits | 1,126 |

### Downstream Usage
- **Phase 3 TaViT time encoding**: `days_since_baseline` → positional encoding for transformer attention
- **Phase 3 TaViT input construction**: build patient sequence `[visit_0, visit_1, ..., visit_N]` in order
- **Longitudinal eligibility filtering**: exclude `n_visits == 1` patients from longitudinal training

---

## 5. `splits.json` — Locked Train/Val/Test Split

> ⚠️ **DO NOT REGENERATE.** This file is locked with `seed=42`. All phases read from this file.

### Structure

```json
{
  "train": ["YG_...", "YG_...", ...],   // 1,144 patients
  "val":   ["YG_...", "YG_...", ...],   // 143 patients
  "test":  ["YG_...", "YG_...", ...]    // 143 patients
}
```

### Split Strategy
- **Patient-level split**: no patient's visits appear in more than one set (no leakage)
- **Stratified by visit_bucket**: equal proportion of 1, 2–3, 4–6, 7–10, >10 visit patients in each split
- **Seed**: 42 (locked forever)

### Split Summary

| Split | Patients | All Visits | Supervised Visits (FULL+PARTIAL) |
|-------|----------|------------|----------------------------------|
| Train | 1,144 | 9,463 | 5,243 |
| Val | 143 | 1,199 | 699 |
| Test | 143 | 1,215 | 700 |

### Downstream Usage
- **Every phase**: filter `data_inventory.csv` by `patient_id in splits['train']`
- **Cross-validation**: never use `val` or `test` patients during training or hyperparameter search

---

## 6. `AUDIT_REPORT.md` — Human-Readable Audit Summary

Auto-generated markdown report containing:
- Dataset summary table (total patients, visits, files, follow-up stats)
- Usability label breakdown with per-label counts and use cases
- Longitudinal eligibility table (patients by visit count threshold)
- Visit count stratification table by split
- Train/val/test split summary
- File integrity check result (0 corrupt files)
- Scanner metadata coverage (100% of patients)
- Output files reference table

---

## 7. EDA Figures (`eda/figures/`)

| File | Description |
|------|-------------|
| `01_full_eda_dashboard.png` | 6-panel figure: (1) visits/patient histogram, (2) year distribution, (3) modality completeness, (4) follow-up duration, (5) inter-visit interval, (6) usability breakdown |
| `02_top20_patients.png` | Horizontal bar chart of top 20 patients by visit count |
| `03_modality_heatmap.png` | Modality availability heatmap (patients × modalities) |

---

## 📍 Where We Are in PHASE1_PLAN.md

### ✅ COMPLETED BLOCKS

| Block | Task | Status |
|-------|------|--------|
| **Block 1** (09:00–10:30) | Data Audit & Inventory → `data_inventory.csv` | ✅ DONE |
| **Block 2** (10:30–11:30) | Clinical Metadata Parsing → `metadata.json` | ✅ DONE |
| **Block 3** (11:30–12:30) | File Integrity Scan (NIfTI headers) | ✅ DONE — 0 corrupt files |
| **Block 4** (13:00–14:00) | EDA Figures | ✅ DONE — 3 figures generated |
| **Block 5** (14:00–15:00) | Train/Val/Test Split + Timelines → `splits.json`, `patient_timelines.json` | ✅ DONE |

**Extra work done beyond plan**:
- ✅ Parsed `image_acquisition_parameters` Excel sheet → `acquisition_parameters.csv` + enriched both `data_inventory.csv` and `metadata.json` with TR/TE/TI/slice thickness per modality
- ✅ Added `POST_ONLY` usability tier (not in original plan — improves pretraining data awareness)
- ✅ `acquisition_parameters.csv` (33,804 rows) — not in original plan but critical for ComBat harmonization

### 📊 End-of-Day Success Criteria Status

| Criterion | Status |
|-----------|--------|
| `data_inventory.csv` — 11,877 rows, all columns | ✅ 11,877 rows × 35 cols |
| `metadata.json` — clinical + scanner info exported | ✅ 1,430 patients, 100% coverage |
| File integrity check — all headers checked | ✅ 33,804 files, 0 corrupt |
| `splits.json` — locked 80/10/10 stratified split | ✅ Seed=42, stratified by visit_bucket |
| `patient_timelines.json` — per-patient visit sequences | ✅ 1,430 patients, with time deltas |
| 6 EDA figures saved | ⚠️ 3 figures saved (combined into fewer multi-panel plots) |
| `EDA_REPORT.md` written | ⚠️ Replaced by `AUDIT_REPORT.md` (more complete) |
| `scripts/01_preprocess.py` written | ❌ NOT STARTED |
| `scripts/02_cnn_baseline.py` written | ❌ NOT STARTED |
| BraTS Toolkit launched in background | ❌ NOT STARTED |
| Kaggle notebook for nnU-Net created | ❌ NOT STARTED |

---

## 🚀 NEXT STEPS

### Immediate Next Step: Block 6 — Write Preprocessing Pipeline

**Script**: `implementation/scripts/01_preprocess.py`

This is the **next task** from `PHASE1_PLAN.md` (Block 6, 15:00–16:30). It writes the full BraTS Toolkit preprocessing pipeline — do NOT run it today, just write and test on 1 patient.

**Pipeline steps to implement**:
```
Step A: N4 bias field correction (SimpleITK)
Step B: Skull stripping (HD-BET via BraTS Toolkit)
Step C: Intra-visit registration (POST/T2/FLAIR → PRE space, itk-elastix)
Step D: Spatial resampling to 1mm³ isotropic (SimpleITK)
Step E: Save to Data/processed/{patient_id}/{visit_date}/
```

**Requirements**:
- Reads patient list from `splits.json` (train set by default)
- Processes one visit at a time
- Logs to `outputs/preprocessing_log.csv`
- **Idempotent** — skips already-processed visits
- Handles missing modalities gracefully
- Test on 1 patient before launching overnight

### Then: Block 7 — CNN Baseline Script

**Script**: `implementation/scripts/02_cnn_baseline.py`

Single-timepoint tumor classification using ResNet-3D / MONAI:
- Input: POST scan (T1c), 128×128×64 crop
- Architecture: `monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)`
- DataLoader reads `data_inventory.csv` + `splits.json`
- Checkpointing + CSV logging
- **Do not run today** — needs Kaggle T4 GPU

### Then: Block 8 — Kaggle Setup

- Create Kaggle notebook for nnU-Net segmentation inference
- Upload processed data (post-BraTS Toolkit) as Kaggle Dataset
- Strategy: 6 × ~2,000 visit batches across 6 Kaggle sessions (~3 days total)

---

## ⚠️ Key Locked Decisions (Do Not Change)

| Decision | Value | Reason |
|----------|-------|--------|
| Split seed | `42` | Reproducibility across all papers |
| Minimum supervised modality | `POST` + `FLAIR` | Keeps 75.7% of visits usable |
| Split granularity | Patient-level | No data leakage |
| Baseline visit | First chronological visit | Clinical convention |
| Disk strategy | Process then compress (`gzip -9`) | Only 22 GB free |
