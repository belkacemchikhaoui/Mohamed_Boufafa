# Phase 1 — EDA Report: BraTS 2024 Post-Treatment Glioma

Generated: 2026-04-20 15:33

## Dataset Overview
- **Total training scans:** 1621
- **Unique patients:** 731
- **Longitudinal (≥2 timepoints):** 559 (76%)
- **Multi-timepoint (≥3):** 153
- **Validation scans (no labels):** 0
- **Modalities:** T1, T1ce, T2, FLAIR — ALL present in 100% of scans
- **Labels:** {0, 1, 2, 4} — confirmed BraTS glioma convention
- **Resolution:** 1mm³ isotropic, co-registered to SRI24

## Tumor Volume Statistics (mL)
| Region | Min | Max | Median | Mean |
|--------|-----|-----|--------|------|
| WT | 0.20 | 344.9 | 54.0 | 64.4 |
| TC | 0.00 | 165.8 | 7.5 | 14.6 |
| ET | 0.00 | 165.8 | 5.1 | 12.9 |

## Temporal Volume Changes
- **Longitudinal pairs analyzed:** 890
- **Growing (>1mL):** 492 (55%)
- **Stable (±1mL):** 99 (11%)
- **Shrinking (<-1mL):** 299 (34%)

## Figures Generated
- intensity_distributions.png
- inter_patient_variability.png
- temporal_volume_changes.png
- tumor_location_distribution.png
- tumor_location_distributions.png
- tumor_volume_distributions.png

## Data Files
- `scan_index.json` — all training scans with paths
- `longitudinal_index.json` — patient → ordered timepoints
- `tumor_volumes.csv` — WT/TC/ET volumes + centroids
- `temporal_changes.csv` — volume changes between timepoints