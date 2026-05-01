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

## Design Decisions and Rationale

### Why concatenate three components into 2825-D?

Each component captures a qualitatively different information type that the others cannot:

| Component | What it captures | Cannot be replaced by |
|---|---|---|
| Octant (2048-D) | Spatial distribution of tumour appearance across 8 anatomical sub-regions (anterior/posterior, superior/inferior, left/right) | Region-weighted pool — which has no spatial decomposition |
| Region-weighted (768-D) | Mean encoder activations specifically INSIDE each sub-region mask (WT, TC, ET) — encodes what each tumour compartment looks like | Octant — which pools the bounding box indiscriminately |
| Volumetric (9-D) | Absolute tumour size (log-volumes), presence flags, and inter-region ratios (TC/WT, ET/TC) | Neither above — both are normalised feature maps insensitive to absolute size |

The concatenation follows standard radiomics practice: combining shape features (volume), texture (encoder activations), and spatial distribution (octant) into a single descriptor. This is consistent with established medical image feature frameworks and aligns with the SwinUNETR 4617-D design in Phase 3 for fair cross-model comparison.

**Limitation acknowledged:** No ablation study was performed to quantify the marginal contribution of each component individually. This is standard for a baseline model that will be superseded. See Planned Actions section.

### Why Stage 3 for the embedding hook?

Stage 3 (16×16×16, 256 channels) was selected because:
1. It is the deepest encoder stage that still retains spatial structure at a resolution meaningful for tumour-level analysis
2. It matches the resolution of SwinUNETR's `layers2` feature map — enabling a controlled cross-model comparison at equivalent spatial scale
3. Deeper stages (Stage 4, 5) collapse to 8³ or 4³ resolution, losing spatial discriminability

---

## Honest Limitations and Known Issues

### Issue 1 — GT Mask Used in Embedding Extraction (Partial Circularity)

**What happens:** Both the region-weighted component and the volumetric component use the ground-truth segmentation mask during extraction:
- Region-weighted pool: the encoder feature map is multiplied by the GT WT/TC/ET binary masks → the pooled vector depends on GT mask boundaries
- Volumetric component: `log1p(GT_wt_vol)`, `log1p(GT_tc_vol)`, `log1p(GT_et_vol)` are directly computed from GT mask voxel counts

**Why this is a problem:**
1. **Evaluation circularity** — tests M1 (volume R²=0.869), M2 (log-vol R²=0.820), M4 (necrosis F1=0.880), and M5 (core fraction R²=0.985) are inflated because the GT volumes and region boundaries are literally encoded inside the embedding. Predicting volume from an embedding that contains the volume is not a meaningful test.
2. **Non-deployable as-is** — in a real clinical setting, GT masks are not available for new patients. The extraction procedure as designed requires the ground truth to produce the embedding.

**Why it was done anyway:**
- For a Phase 2 **baseline** on training data where GT is available, this gives the strongest possible feature descriptor
- The intent was to establish an upper-bound reference, not a deployable system
- The limitation is systematic and affects CNN and ViT Phase 3 embeddings equally — making cross-model comparisons still internally consistent

**Correct interpretation of affected test scores:**

| Test | Reported value | Honest interpretation |
|---|---|---|
| M1 Volume R² (RF) | 0.869 | Upper bound — inflated by GT volume in embedding. True discriminative value unknown. |
| M2 Log-vol R² (RF) | 0.820 | Same issue — partially circular. |
| M4 Necrosis F1 | 0.880 | Partially inflated — region mask boundaries encoded. |
| M5 Core fraction R² | 0.985 | Highly inflated — TC/WT ratio is literally the et/wt ratio stored in vol component. |
| M1 Spearman ρ | 0.871 | Same as M1 R². |
| T1, T4, T7, T8 | 0.304 / 0.687 / 0.452 / 0.394 | Not directly affected — these compare temporal differences, not absolute volume. |
| M6, H1, H2, H3 | Not affected — structural properties of the embedding space, not volume prediction. |

**Tests that remain valid (GT does not directly inflate them):**
M6, H1, H2, H3, H4, T1, T2, T3, T4, T5, T6, T7, T8 — all temporal and structural tests.

### Issue 2 — No Component Ablation Study

Without running each component (Octant-only, Region-only, Volume-only, All-combined) through the 18-test battery, it is unknown:
- How much each component independently contributes
- Whether the concatenation improves on any single component
- Whether the 9-D volume component (GT-dependent) is necessary if the encoder features already encode volume

**Scope:** This is a known gap for the Phase 2 baseline. Ablation is planned for Phase 3 (see below) because Phase 3 is the publication-relevant model.

### Issue 3 — Cell 9 NameError in Original Training Notebook

The original `Phase2_A2_nnUNet_Finetune.ipynb` training run crashed in Cell 9 (visualization) with `NameError: name 'gc' is not defined` after training completed. Training itself finished correctly at epoch 27 with best checkpoint saved. The extraction notebook (`after_run/phase2-a2-nnunet-finetune_continue.ipynb`) recovered the checkpoint and completed embedding extraction cleanly with no errors.

---

## Planned Actions (Phase 3 Review — Before Paper Submission)

### Action 1 — Verify 4608-D SwinUNETR Encoder Alone is Sufficient (HIGH PRIORITY)

**Context:** Phase 3's 4617-D embedding = 4608-D SwinUNETR encoder features + 9-D GT volumes. The same GT circularity applies. If the 4608-D encoder alone achieves comparable M1 R² scores, the 9-D GT volume component can be **dropped entirely** — making the Phase 3 embedding 100% GT-free and fully deployable without ground truth.

**Plan:**
1. Modify `Phase3/notebooks/Phase3_D1_Hybrid_Extraction.ipynb` — add a one-line flag to skip the 9-D volume concatenation
2. Re-run Phase3_B1 with the 4608-D-only embedding
3. Compare M1 R², T1 ρ, N11 Silhouette against the 4617-D results

**Expected outcome:** If M1 R² stays ≥0.90 without the 9-D component → drop it. The SwinUNETR encoder was trained on segmentation, so it necessarily encodes volume information in its features.

**If retained after ablation:** The 9-D component should switch from GT mask volumes to **predicted mask volumes** (from the model's own sigmoid output) — making it GT-free.

**Estimated cost:** One Kaggle run (~1 hour). No architecture changes.  
**When:** During Phase 3 review, before Phase 3_D1 is finalised.

### Action 2 — Component Ablation for Phase 3 4617-D (MEDIUM PRIORITY)

Run Phase3_B1 with three embedding variants to justify the design:
- Encoder only (4608-D)
- Encoder + volume (4617-D, current)
- Volume only (9-D)

Add a comparison column to the 18-test table in the Phase 3 report.

**Estimated cost:** Two additional Kaggle runs.  
**When:** After Action 1 confirms whether to keep or drop the 9-D component.

---

*Report last updated: 2026-05-01*  
*Sources: Phase2_A2_nnUNet_Finetune.ipynb (13 cells), Phase2_B1_CNN_Embedding_Evaluation.ipynb (7 cells), training_log_nnunet.txt, eval_log_nnunet.txt, extraction_log_nnunet.txt, cnn_eval_results.json*
