# Essential Papers — Reading Index

> 22 papers organized by objective. Print these and read them carefully.  
> **Naming**: `XX_ShortName_Author_Year.pdf`

---

## 📁 Objective 1 — Preprocessing & Data Pipeline (Papers 01–09)

| # | File | Paper Title | Why You Need It |
|---|---|---|---|
| 01 | `01_BraTS_Toolkit_Kofler_2020.pdf` | BraTS Toolkit | HD-BET skull stripping + z-score normalization protocol |
| 02 | `02_nnUNet_Isensee_2021.pdf` | nnU-Net | Self-configuring tumor segmentation (Dice ~0.908) |
| 03 | `03_ITK_Elastix_Niessen_2023.pdf` | itk-elastix | Temporal registration — pip install, brain-validated, MONAI-native |
| 04 | `04_Yale_Dataset_Ramakrishnan_2025.pdf` | Yale Brain Metastases | YOUR primary dataset: 11,884 scans, 1,430 patients |
| 05 | `05_Cyprus_Dataset_Trimithiotis_2025.pdf` | Cyprus Brain Metastases | Validation dataset: 744 scans, 40 patients, expert labels |
| 06 | `06_ComBat_Guide_Fortin_2022.pdf` | ComBat Harmonization Guide | Baseline method for fixing scanner differences |
| 07 | `07_ComBat_Validation_Moyer_2022.pdf` | ComBat Validation | Proves ComBat works + warns: test BEFORE applying |
| 08 | `08_Generalized_ComBat_Horng_2022.pdf` | Generalized ComBat | Nested multi-batch effects (scanner + year + field strength) |
| 09 | `09_Longitudinal_ComBat_Beer_2020.pdf` | Longitudinal ComBat | Preserves within-patient trajectories during harmonization |

**Reading order**: 04 (your data) → 01 (how to clean it) → 02 (segment tumors) → 03 (align over time) → 06 → 07 → 08 → 09 (harmonization stack)

---

## 📁 Objective 2 — ViT Representations (Papers 10–13)

| # | File | Paper Title | Why You Need It |
|---|---|---|---|
| 10 | `10_Swin_UNETR_Tang_2022.pdf` | Swin UNETR | 768-dim embeddings + segmentation (Dice 0.913), MONAI available |
| 11 | `11_TaViT_Time_Distance_ViT_Hager_2022.pdf` | Time-Distance ViT (TaViT) | Temporal attention for irregular scan intervals (AUC 0.786) |
| 12 | `12_TransXAI_Zeineldin_2024.pdf` | TransXAI | Grad-CAM explainability for transformer models |
| 13 | `13_CAFNet_Ahmed_2025.pdf` | CAFNet | Validates hybrid CNN+ViT design (87% → 96%) |

**Reading order**: 10 (your feature extractor) → 11 (temporal modeling) → 12 (explainability) → 13 (hybrid validation)

---

## 📁 Objective 3 — LLM Integration (Papers 14–15)

| # | File | Paper Title | Why You Need It |
|---|---|---|---|
| 14 | `14_RadFM_Wu_2025.pdf` | RadFM (Nature Comms 2025) | Perceiver + MedLLaMA-13B: your vision-to-language bridge |
| 15 | `15_MM_Embed_Lin_2025.pdf` | MM-Embed (ICLR 2025, NVIDIA) | Training methodology: modality bias, curriculum learning, hard negatives |

**Reading order**: 14 (your architecture) → 15 (how to train it properly)

---

## 📁 Objective 4 — Video Diffusion & Counterfactuals (Papers 16–22)

| # | File | Paper Title | Why You Need It |
|---|---|---|---|
| 16 | `16_DDPM_Ho_2020.pdf` | DDPM | Diffusion theory foundation: ε-prediction, L_simple |
| 17 | `17_LDM_Rombach_2022.pdf` | Latent Diffusion Models | VAE + latent diffusion + cross-attention conditioning |
| 18 | `18_Video_LDM_Blattmann_2023.pdf` | Video LDM | Temporal layer insertion into frozen image model |
| 19 | `19_TaDiff_Treatment_Aware_Liu_2025.pdf` | TaDiff (IEEE-TMI 2025) | Same domain! Treatment-conditioned glioma MRI generation |
| 20 | `20_MedEdit_BenAlaya_2024.pdf` | MedEdit (MICCAI 2024) | Counterfactual brain MRI editing, clinical validation |
| 21 | `21_EchoNet_Synthetic_Reynaud_2024.pdf` | EchoNet-Synthetic (MICCAI 2024) | Medical video diffusion — YOUR starting codebase (code available!) |
| 22 | `22_Counterfactual_Diff_AE_Atad_2024.pdf` | Counterfactual Diffusion AE | Latent space counterfactual manipulation |

**Reading order**: 16 (theory) → 17 (latent space) → 18 (video extension) → 19 (your domain!) → 21 (your codebase) → 20 (counterfactual method) → 22 (alternative counterfactual)

---

## 📊 Quick Stats

- **Total papers**: 22 PDFs
- **Total size**: ~130 MB
- **Objective 1**: 9 papers (preprocessing + harmonization)
- **Objective 2**: 4 papers (ViT + temporal + explainability)
- **Objective 3**: 2 papers (RadFM + MM-Embed)
- **Objective 4**: 7 papers (diffusion + video + counterfactual)

---

## 🔑 Top 5 Must-Read-First Papers

1. **04** — Yale Dataset (know your data)
2. **10** — Swin UNETR (your feature extractor)
3. **11** — TaViT (your temporal model)
4. **14** — RadFM (your LLM bridge)
5. **19** — TaDiff (same domain, treatment conditioning)
