# MU-Glioma Treatment-Aware TaViT Implementation

Re-implementation of the full pipeline on MU-Glioma-Post dataset with treatment conditioning and counterfactual trajectory simulation.

## Structure

```
implementation_mu_glioma/
├── Phase_M1/   — Data pipeline (scan index, clinical parsing, volume extraction, treatment tokens)
├── Phase_M2/   — Fine-tune CNN (nnUNet) + ViT (SwinUNETR) on MU-Glioma
├── Phase_M3/   — Treatment-Aware TaViT V3 (time PE + treatment tokens)
├── Phase_M4/   — Counterfactual trajectory simulation
└── Phase_M5/   — Evaluation, 18-test battery, visualizations
```

## Data Paths

- **Imaging**: `/home/moamed/HDD/validation_glomia/PKG - MU-Glioma-Post/MU-Glioma-Post/`
- **Clinical XLSX**: `/home/moamed/HDD/validation_glomia/MU-Glioma-Post_ClinicalData-July2025.xlsx`
- **Segmentation Volumes XLSX**: `/home/moamed/HDD/validation_glomia/MU-Glioma-Post_Segmentation_Volumes.xlsx`

## Key Outputs (Phase M1)

- `mu_glioma_master.csv` — One row per scan, with treatment state and volume data
- `scan_index.json` — BraTS-compatible scan index
- `longitudinal_index.json` — Patient trajectory index with treatment phases
- Treatment timeline & EDA visualizations
