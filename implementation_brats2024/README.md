# Explainable Disease Progression — BraTS 2024 Post-Treatment Glioma

## Project Overview

Longitudinal tumor evolution analysis using deep embeddings from fine-tuned segmentation models,
integrated with LLMs for clinical reasoning and diffusion-based video generation.

**Training Dataset:** BraTS 2024 Post-Treatment Glioma
- Main Training: ~1,350 scans (has labels)
- Additional Training: 271 scans (has labels) → used as validation
- Official Validation: 188 scans (NO labels — challenge submission only)

**External Validation:** MU-Glioma-Post (203 patients, 617 timepoints)

## Architecture Comparison

### Phase 2 — CNN Baselines
| Model | Pretrained On | Params | Source |
|-------|--------------|--------|--------|
| **nnU-Net v2** | BraTS 2021 | ~31M | Zenodo |
| **SegResNet** | BraTS 2018 | ~15M | MONAI Model Zoo |

### Phase 3 — Vision Transformers
| Model | Pretrained On | Params | Source |
|-------|--------------|--------|--------|
| **Swin UNETR** | BraTS 2021 (SSL) | ~62M | MONAI Research / GitHub |
| **3D-TransUNet** | BraTS 2023 | ~90M | Published paper |

## Label Convention
```
BraTS Glioma: 0=Background, 1=NCR, 2=ED, 4=ET
WT = {1,2,4} | TC = {1,4} | ET = {4}
```

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | 🔄 EDA notebook ready, data downloading | Main training zip still downloading |
| Phase 2 | 📝 Strategy defined | Build notebooks after data ready |
| Phase 3 | 📝 Strategy defined | Build notebooks after Phase 2 |
| Phase 4 | 📋 Planned | LLM + Video generation |

See [STRATEGY.md](STRATEGY.md) for the full implementation plan.
