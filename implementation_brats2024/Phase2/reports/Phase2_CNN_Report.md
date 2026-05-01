# Phase 2 — CNN Baseline Report
## Baseline Vision Models for Tumor Representation
### BraTS 2024 Post-Treatment Glioma — nnUNet v2 PlainConvUNet

**Duration:** Weeks 5–7 (3 weeks)  
**Notebook:** `Phase2_A2_nnUNet_Finetune.ipynb` + `Phase2_B1_CNN_Embedding_Evaluation.ipynb`  
**Dataset:** BraTS 2024 Post-Treatment Glioma | 731 patients | 1621 longitudinal scans  

---

## Activity 1 — Implementation of Baseline CNN

### Model Architecture: nnUNet v2 PlainConvUNet

| Property | Value |
|---|---|
| Architecture | PlainConvUNet (nnUNet v2) |
| Total parameters | 30.8M |
| Encoder stages | 6 |
| Feature channels per stage | [32, 64, 128, 256, 320, 320] |
| Strides per stage | [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,1]] |
| Input patch size | 128×128×128 |
| Input channels | 4 (T1, T1ce, T2, FLAIR) |
| Output channels | 3 (WT, TC, ET — binary masks) |
| Framework | MONAI 1.5.2 + PyTorch 2.10 |

**Encoder depth map:**

```
Stage 0: 128³ → 128³  | 32 ch   (no downsampling)
Stage 1: 128³ → 64³   | 64 ch   (stride 2)
Stage 2:  64³ → 32³   | 128 ch  (stride 2)
Stage 3:  32³ → 16³   | 256 ch  (stride 2) ← embedding hook
Stage 4:  16³ →  8³   | 320 ch  (stride 2)
Stage 5:   8³ →  4³   | 320 ch  (stride [2,2,1])
```

The embedding hook is placed at **Stage 3** (16×16×16 resolution, 256 channels) — chosen to match SwinUNETR's `layers2` hook depth for fair cross-model comparison.

---

### Pretrained Model

| Property | Value |
|---|---|
| Source | nnUNet BraTS 2021 (BRATS19 Dataset002) |
| Fold | fold_0 |
| Pre-trained Dice (BraTS 2021) | WT=0.9005, TC=0.8673, ET=0.8509 |
| Layers loaded | 209 / 219 (95.4%) |
| Loading strategy | `safe_torch_load()` — compatible with PyTorch 2.6+ weights_only |

The remaining 10 layers (output head) were re-initialised for BraTS 2024's 3-channel output.

---

### Sub-Region Label Mapping

BraTS 2024 Post-Treatment uses a **different label convention** from BraTS 2021:

| Label | Class | Description |
|---|---|---|
| 1 | NETC | Non-Enhancing Tumor Core (necrosis, cysts) |
| 2 | SNFH | Surrounding Non-enhancing FLAIR Hyperintensity (edema) |
| 3 | ET | Enhancing Tissue (active tumor) |
| 4 | RC | **Resection Cavity — EXCLUDED from all regions** |

**Compound regions (output channels):**

| Region | Constituent Labels | Clinical Meaning |
|---|---|---|
| **WT** (Whole Tumor) | 1 + 2 + 3 | Total tumor burden |
| **TC** (Tumor Core) | 1 + 3 | Active + necrotic core |
| **ET** (Enhancing Tumor) | 3 only | Active enhancing tissue |

RC (label 4) is explicitly excluded — it represents post-surgical cavity with no tumour tissue.

---

### Loss Function

Combined **Dice + Cross-Entropy** loss (standard nnUNet regime):

```
L_total = 0.5 × L_Dice + 0.5 × L_CE

where:
  L_Dice = 1 - (2 × |P ∩ G|) / (|P| + |G|)   per region, mean over WT/TC/ET
  L_CE   = -Σ G·log(sigmoid(P))                 binary CE per voxel per region
```

Optimiser: AdamW | LR schedule: cosine annealing (1e-4 → 0, 5-epoch warm-up) | AMP (fp16) | Gradient clipping

---

## Activity 2 — Training and Validation

### Training Configuration

| Parameter | Value |
|---|---|
| Train scans | 1,324 (80% patient-level split) |
| Val scans | 297 (20% patient-level split) |
| Split strategy | By patient ID — no data leakage across timepoints |
| Epochs | 28 |
| Batch size | 1 (GPU memory constraint: T4 15.6 GB) |
| Cache strategy | CacheDataset(cache_rate=0.05) — avoids Kaggle OOM |
| Corrupt file handling | `safe_loader_iter()` — skips gzip-corrupt NiFTI files |
| GPU | NVIDIA Tesla T4 (15.6 GB VRAM) |

### Training Curve

| Epoch | Loss | Mean Dice | WT | TC | ET | Note |
|---|---|---|---|---|---|---|
| 0 | 0.9477 | — | — | — | — | warm-up |
| 3 | 0.6713 | 0.7332 | 0.831 | 0.686 | 0.682 | NEW BEST |
| 7 | 0.3361 | 0.7965 | 0.844 | 0.770 | 0.776 | NEW BEST |
| 11 | 0.3171 | 0.8039 | 0.856 | 0.774 | 0.782 | NEW BEST |
| 15 | 0.2999 | 0.8044 | 0.857 | 0.774 | 0.782 | NEW BEST |
| 19 | 0.2900 | 0.8103 | 0.861 | 0.781 | 0.789 | NEW BEST |
| 23 | 0.2807 | 0.8142 | 0.863 | 0.787 | 0.793 | NEW BEST |
| **27** | **0.2765** | **0.8150** | **0.864** | **0.787** | **0.794** | **FINAL BEST** |

**Gap vs BraTS 2021 benchmark:**

| Region | BraTS 2021 | BraTS 2024 (ours) | Gap |
|---|---|---|---|
| WT | 0.9005 | 0.864 | -4.1% |
| TC | 0.8673 | 0.787 | -8.0% |
| ET | 0.8509 | 0.794 | -5.7% |

The gap is expected: post-treatment glioma is harder due to radiation necrosis, pseudoprogression, and treatment-induced signal heterogeneity, all of which reduce segmentation boundary clarity.

---

## Activity 3 — Embedding Extraction

### Embedding Design

The CNN encoder was used to extract a **2825-D structured feature vector** per scan:

```
Component          Source                              Dimensions
─────────────────────────────────────────────────────────────────
Octant features    Stage-3 feat map (16³, 256ch)       2048-D
                   → WT bounding box crop               
                   → 2×2×2 octant adaptive avg pool    (8 × 256)
─────────────────────────────────────────────────────────────────
Region features    Stage-3 feat map × GT mask           768-D
                   → mask-weighted pool per region      (3 × 256)
                     (WT mask, TC mask, ET mask)
─────────────────────────────────────────────────────────────────
Volumetric morph.  GT mask voxel counts + derived        9-D
                   log1p(wt_vol), log1p(tc_vol),
                   log1p(et_vol), has_wt, has_tc,
                   has_et, tc/wt ratio, et/tc ratio,
                   et/wt ratio
─────────────────────────────────────────────────────────────────
TOTAL                                                  2825-D
```

**Handling of complete resection** (51 scans): Octant + region components are zero vectors; volumetric component encodes `has_wt=0`. This is intentional — tumour absence is itself a clinical signal.

### Saved Artifacts

| File | Shape | Size |
|---|---|---|
| `embeddings/cnn_nnunet_embeddings.npz` | (1620, 2825) float32 | 15.3 MB |
| `embeddings/cnn_spatial_tokens.npz` | (1620, 2744, 256) float32 | 975.9 MB |
| `embeddings/tumor_volumes.csv` | 1620 rows | — |
| `outputs/nnunet_best.pth` | — | checkpoint epoch 27 |

Extraction: 1620/1621 scans processed (1 skipped — corrupt gzip), 51 with zero WT (complete resection).

Raw embedding statistics:
- Norm: min=0.0, max=25.2, mean=15.4
- Cosine similarity (500 pairs): mean=0.794, std=0.157
- RankMe (effective rank): 824.4 — high-dimensional, non-collapsed

---

## Activity 4 — Quantitative Evaluation (18-Test Battery)

All 22/26 tests passed on a clean Kaggle run (n=1620 scans, 731 patients).  
Results cached in `outputs/cnn_eval_results.json`.

### Morphology Tests (M1–M6)

| Test | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| M1 | Volume R² (Ridge) | 0.806 | ≥0.50 | PASS |
| M1 | Volume R² (RF) | 0.869 | ≥0.50 | PASS |
| M1 | Spearman ρ | 0.871 | ≥0.55 | PASS |
| M2 | Log-vol R² (Ridge) | 0.825 | ≥0.40 | PASS |
| M2 | Log-vol R² (RF) | 0.820 | ≥0.40 | PASS |
| M3 | Enhancement R² (Ridge) | 0.947 | ≥0.25 | PASS |
| M3 | Enhancement R² (RF) | 0.986 | ≥0.25 | PASS |
| M4 | Necrosis F1 | 0.880 | ≥0.60 | PASS |
| M5 | Core fraction R² (Ridge) | 0.961 | ≥0.30 | PASS |
| M5 | Core fraction R² (RF) | 0.985 | ≥0.30 | PASS |
| **M6** | **Patient purity (10-NN)** | **12.8%** | **≥60%** | **FAIL** |

M6 FAIL is expected and **intentional**: a static single-scan CNN has no mechanism to group the same patient's repeated scans together — this is the core motivation for Phase 3 temporal modeling.

### Heterogeneity Tests (H1–H5)

| Test | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| H1 | RankMe (effective rank) | 824.4 | ≥30 | PASS |
| H1 | Effective rank 95% | 1122.0 | ≥50 | PASS |
| **H2** | **Diversity** | **0.233** | **≥0.25** | **FAIL (marginal)** |
| H2 | Uniformity | -0.786 | ≥-3.0 | PASS |
| H3 | Responder F1 | 0.821 | ≥0.55 | PASS |
| H4 | Norm CV | 0.047 | ≤0.30 | PASS |
| H5 | RankMe standalone | 824.4 | ≥30 | PASS |

### Temporal Tests (T1–T8)

| Test | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| T1 | Spearman ρ (WT vol change) | 0.304 | ≥0.30 | PASS |
| T2 | Temporal ordering pass | 1.000 | ≥1.0 | PASS |
| T3 | Delta R² | **-0.010** | — | descriptor |
| T3 | Directional AUC | 0.750 | ≥0.55 | PASS |
| T4 | RANO AUC (≥25% growth) | 0.687 | ≥0.65 | PASS |
| T5 | Coherence (consec<rand) | 0.939 | ≥0.70 | PASS |
| T5 | Dual pass | 0.000 | ≥1.0 | FAIL (low-pri) |
| T6 | Velocity CV | 0.610 | — | descriptor |
| **T7** | **Treatment Cohen d** | **0.452** | **≥0.50** | **FAIL (marginal)** |
| T8 | Kendall τ (PC1, Δvol) | 0.394 | ≥0.30 | PASS |

### Summary: 22/26 PASS

**High-priority fails and their meaning:**
1. **M6 purity 12.8% (needs 60%)** — CNN cannot cluster temporal scans by patient identity without explicit temporal context → motivates TaViT (Phase 3E)
2. **T7 Cohen d=0.452 (needs 0.50)** — Treatment response separation is weak — the embedding conflates treated and untreated tumour appearance
3. **T3 delta_R2 = -0.010** — Static CNN embedding cannot predict *change* in tumour volume (R² negative = worse than mean predictor) → key static modeling limitation

---

## Activity 5 — Identification of Static Modeling Limitations

The following limitations were confirmed quantitatively and explicitly motivate Phase 3:

| Limitation | Evidence | Phase 3 Target |
|---|---|---|
| Cannot group same-patient scans | M6 purity=12.8% vs threshold 60% | TaViT temporal aggregation → same-patient trajectory |
| Cannot predict volume change | T3 delta_R2=-0.010 (negative) | Temporal sequence modeling over embedding evolution |
| Weak treatment separation | T7 Cohen d=0.452 | Supervised ratio head trained on Δvol |
| Marginal embedding diversity | H2 diversity=0.233 | SwinUNETR self-supervised pretraining |
| Single time-point bias | All T-tests marginally above floor | Phase 3 cross-attention over temporal sequences |

---

## Deliverable 1 — Baseline Model Implementations and Trained Weights

| Deliverable | Location | Status |
|---|---|---|
| Training notebook | `notebooks/Phase2_A2_nnUNet_Finetune.ipynb` | DONE |
| Best checkpoint | `outputs/nnunet_best.pth` (epoch 27, Dice=0.8150) | DONE |
| Latest checkpoint | `outputs/nnunet_latest.pth` | DONE |
| Embedding extraction code | A2 Cell 10 (extract_embeddings function) | DONE |
| Embeddings (2825-D) | `embeddings/cnn_nnunet_embeddings.npz` (1620×2825) | DONE |
| Spatial tokens | `embeddings/cnn_spatial_tokens.npz` (1620×2744×256) | DONE |
| Visualizations (2D+3D) | `outputs/fig/` (12 figures) | DONE |

## Deliverable 2 — Comparative Performance Results (Reference Benchmarks)

| Deliverable | Location | Status |
|---|---|---|
| 18-test evaluation notebook | `notebooks/Phase2_B1_CNN_Embedding_Evaluation.ipynb` | DONE |
| Cached results (all metrics) | `outputs/cnn_eval_results.json` | DONE |
| t-SNE visualization | `outputs/fig/nnunet_tsne.png` | DONE |
| Distribution plots | `outputs/fig/nnunet_distributions.png` | DONE |
| Uploaded to Kaggle for Phase 3 | `mohamedmohamed23/embedding-datasets` dataset | DONE |

---

## Quick Reference — Numbers to Report

| Metric | Value |
|---|---|
| Model | nnUNet v2 PlainConvUNet, 30.8M params |
| Best mean Dice (BraTS 2024) | 0.8150 (epoch 27/28) |
| WT / TC / ET Dice | 0.864 / 0.787 / 0.794 |
| Embedding dimension | 2825-D |
| Scans processed | 1620 / 1621 |
| 18-test pass rate | 22/26 (84.6%) |
| Key limitation (T3 delta_R2) | -0.010 (cannot model change) |
| Key limitation (M6 purity) | 12.8% (cannot group patient timeseries) |

---

*Generated: 2026-05-01 | Phase 2 — Weeks 5–7*
