# 📁 Final Papers — Print This Folder (15 PDFs)

> Reading order follows the pipeline: Data → Preprocessing → Features → LLM → Video → Evaluation

| # | File | Paper | Read For |
|---|---|---|---|
| 01 | `01_Yale_Dataset_Ramakrishnan_2025.pdf` | Yale Glioma MRI Dataset | Our main dataset: 11,884 scans, 1,430 patients |
| 02 | `02_Cyprus_Dataset_Trimithiotis_2025.pdf` | Cyprus Glioma Dataset | Validation: 40 patients with expert labels |
| 03 | `03_BraTS_Toolkit_Kofler_2020.pdf` | BraTS Toolkit | HD-BET skull stripping pipeline |
| 04 | `04_nnUNet_Isensee_2021.pdf` | nnU-Net | Self-configuring tumor segmentation |
| 05 | `05_ITK_Elastix_Niessen_2023.pdf` | ITK-Elastix | Longitudinal registration (Python API) |
| 06 | `06_Generalized_ComBat_Horng_2022.pdf` | Generalized ComBat | **Nested ComBat** for Yale's 5+ batch effects |
| 07 | `07_Longitudinal_ComBat_Beer_2020.pdf` | Longitudinal ComBat | Preserve within-patient temporal trajectories |
| 08 | `08_ComBat_Guide_Fortin_2022.pdf` | ComBat Guide | ComBat theory + feature-space application |
| 09 | `09_Swin_UNETR_Tang_2022.pdf` | Swin UNETR | 768-dim feature extractor (MONAI built-in) |
| 10 | `10_TaViT_Hager_2022.pdf` | Time-Aware ViT | Time-distance encoding (without it → AUC 0.50!) |
| 11 | `11_RadFM_Wu_2025.pdf` | RadFM | Perceiver (32 tokens) + MedLLaMA-13B |
| 12 | `12_MM_Embed_Lin_2025.pdf` | MM-Embed | Contrastive embedding alignment (supervisor's paper) |
| 13 | `13_LDM_Rombach_2022.pdf` | Latent Diffusion Models | VAE + latent diffusion + cross-attention |
| 14 | `14_TaDiff_Liu_2025.pdf` | Treatment-Aware Diffusion | Treatment-conditioned generation + counterfactuals |
| 15 | `15_EchoNet_Synthetic_Reynaud_2024.pdf` | EchoNet-Synthetic | Video diffusion pipeline (our starting code) |

**Pipeline doc**: See `PIPELINE_FINAL.md` for what idea we take from each paper and how we implement it.
