# Yale Brain Metastases — Data Audit Report
Generated: 2026-03-10 15:50:41

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total patients | 1,430 |
| Total visit folders | 11,877 |
| Total NIfTI files | 33,811 |
| Median visits per patient | 6.0 |
| Median follow-up | 288 days |
| Max follow-up | 5,061 days (13.9 years) |
| Top patient | YG_C9D2TEGNY08A (66 visits) |

## Visit Training Usability Labels
> Note: All visits are real MRI data. This label describes HOW each visit
> can be used in the training pipeline — not whether the data is valid.

| Label | Count | % | Use |
|-------|-------|---|-----|
| FULL (all 4 modalities) | 4,364 | 36.7% | Full supervised training |
| PARTIAL_GOOD (POST+FLAIR ± others) | 2,278 | 19.2% | Supervised w/ modality dropout |
| POST_ONLY (POST present, no FLAIR) | 2,352 | 19.8% | Self-supervised pretraining only |
| UNUSABLE (no POST) | 2,883 | 24.3% | Temporal context only |
| CORRUPT (header errors) | 0 | 0.0% | Excluded entirely |

**Supervised training pool (FULL + PARTIAL_GOOD)**: 6,642 visits (55.9%)
**Recoverable for pretraining (+ POST_ONLY)**: 8,994 visits (75.7%)

## Longitudinal Eligibility
| Criterion | Patients |
|-----------|----------|
| Only 1 visit (not usable for longitudinal learning) | 131 |
| ≥ 3 supervised visits | 1,126 |
| ≥ 5 supervised visits (high priority) | 842 |

## Visit Count Distribution by Split (Stratified)
```
split         train  val  test  TOTAL
visit_bucket                         
1               105   13    13    131
2–3             256   32    32    320
4–6             262   33    33    328
7–10            216   27    27    270
>10             305   38    38    381
```
> Rows = visit_bucket (number of visits per patient).
> Columns = patients in each split.
> Stratification ensures all temporal depths appear in train/val/test at the same ratio.

## Train/Val/Test Split
| Split | Patients | All Visits | Supervised Visits (FULL+PARTIAL) |
|-------|----------|------------|----------------------------------|
| Train | 1,144 | 9,463 | 5,243 |
| Val   | 143 | 1,199 | 699 |
| Test  | 143 | 1,215 | 700 |

Seed: 42 — **DO NOT REGENERATE**

## File Integrity
- Total files checked: 33,804
- Files with header errors: 0
- No corrupt files found ✅

## Scanner Metadata Coverage
- Patients with clinical + scanner info: 1,430 / 1,430 (100%)
- Fields available: age_at_imaging, sex, scanner vendor, model, field_strength (T), 2D/3D, scanner_site

## Output Files
| File | Location |
|------|----------|
| data_inventory.csv | `implementation/outputs/data_inventory.csv` |
| patient_timelines.json | `implementation/outputs/patient_timelines.json` |
| metadata.json | `implementation/outputs/metadata.json` |
| splits.json | `implementation/outputs/splits.json` |
| AUDIT_REPORT.md | `implementation/outputs/AUDIT_REPORT.md` |
| EDA figures (3 PNG) | `implementation/outputs/eda/figures/` |
