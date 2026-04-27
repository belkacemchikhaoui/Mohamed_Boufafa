# Phase 3 Complete Plan — All Research Consolidated

## Where We Are Now

### Phase 2 (DONE ✅) — CNN Baseline
| Component | Detail |
|---|---|
| Architecture | DynUNet / SegResNet (CNN) |
| Pretraining | BraTS-METS (brain metastasis specific) |
| Input | 4-channel (T1, T1ce, T2, FLAIR) |
| Output | 3-class segmentation (WT, TC, ET) + 256-dim embeddings |
| Results | 3-fold CV completed, Dice ~0.65-0.70 |
| Embeddings | Extracted for all 45 patients × ~4 timepoints |

### Phase 3 Activity 1 (IN PROGRESS 🔄) — Swin UNETR V1
| Component | Detail |
|---|---|
| Architecture | Swin UNETR V1 (Vision Transformer) |
| Pretraining | `model_swinvit.pt` — BraTS 2021 SSL (1,251 MRI volumes) |
| Input | 4-channel (T1, T1ce, T2, FLAIR) |
| Output | 3-class segmentation + 768-dim embeddings |
| Status | **Fold 0 training — Dice 0.43 at epoch 19/60, steadily improving** |

---

## 🏆 GPT Round 2 Analysis — The Best Response Yet

### Accuracy: ~90%. Three MAJOR verified discoveries:

### 1. BrainSegFounder — **THE BEST FIND ACROSS ALL RESPONSES**

> [!IMPORTANT]
> This is the optimal pretrained weight for Phase 3. It's a Swin UNETR pretrained on
> **42,470 brain MRI scans** with **4-channel BraTS input** — exactly our setup.

| Detail | Verified? |
|---|---|
| HuggingFace | ✅ `huggingface.co/smilelab/BrainSegFounder` |
| Architecture | ✅ MONAI SwinUNETR (same as ours!) |
| Pretraining data | ✅ 42,470 UK Biobank + BraTS scans |
| Input channels | ✅ **4-channel (T1, T1ce, T2, FLAIR)** — native! |
| BraTS fine-tuned version | ✅ Available separately |
| Embedding dim | ✅ 768-dim bottleneck |

**Available weight files:**
```
# SSL pretrained on 42k brain MRI (best for our embedding goal):
https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-pretrain.pt

# Fine-tuned on BraTS tumor segmentation (best for Dice comparison):
https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-finetune.pt
```

**Why this is the winner:**
- 42k scans vs our current 1.2k = **35x more pretraining data**
- Same architecture (Swin UNETR) = **drop-in weight replacement**
- Trained on brain MRI with 4 channels = **no modality mismatch**
- Both SSL and fine-tuned versions = **we can test both**

---

### 2. TaViT/TeViT Code — Ready for Activity 2

| Detail | Verified? |
|---|---|
| GitHub | ✅ `github.com/tom1193/time-distance-transformer` |
| Implements | ✅ TaViT + TeViT (Time-Distance ViT) |
| Input | Per-timepoint embeddings + time gaps |
| Results | AUC 0.786 vs 0.734 (non-temporal baseline) |
| Framework | ✅ PyTorch |

**This is exactly the temporal transformer we planned for Activity 2.** Public code exists — we don't need to build from scratch.

---

### 3. Triad — Verified and Available

| Detail | Verified? |
|---|---|
| GitHub | ✅ `github.com/wangshansong1/Triad` |
| Pretraining | ✅ 131k MRI volumes (brain + breast + prostate) |
| Weights | ✅ Google Drive links in repo |
| Gain | +6.9% Dice, +4.0% AUC over scratch |

Lower priority than BrainSegFounder (mixed-organ vs brain-specific), but available as a third option.

---

### GPT Round 2 also correctly flagged:
- ✅ UKBOB/Swin-BOB: "no public URL yet" — honest
- ✅ BEiT3/MAE: "not released for 4-channel brain MRI" — honest
- ✅ Evaluation metrics: Linear probing, temporal prediction, CKA — all valid

---

## Combined Findings: All Verified ViT Options for Phase 3

| Rank | Model | Pretraining | Data Size | 4-ch? | Availability | Drop-in? |
|---|---|---|---|---|---|---|
| 🥇 | **BrainSegFounder** | UK Biobank + BraTS SSL | **42,470** | ✅ | HuggingFace direct link | ✅ Yes |
| 🥈 | **Swin UNETR V2** (`use_v2=True`) | Same weights, better arch | 1,251 | ✅ | Built into MONAI | ✅ 1-line change |
| 🥉 | **SwinBrain** | 75k head MRI SSL | 75,861 | ⚠️ 3-ch | GitHub (check) | ⚠️ Needs adaptation |
| 4 | **Triad** | 131k mixed MRI | 131,170 | ⚠️ | GitHub + Google Drive | ⚠️ Different arch |
| 5 | UKBOB/Swin-BOB | 51k UK Biobank | 51,761 | ✅ | Not released yet | — |
| 6 | Current (`model_swinvit.pt`) | BraTS 2021 SSL | 1,251 | ✅ | MONAI | ✅ Running now |

---

## 🎯 OPTIMAL PHASE 3 PLAN

### Activity 1: Encoder Training & Embedding Extraction

#### Experiment A — Current (RUNNING NOW ✅)
```
Swin UNETR V1 + model_swinvit.pt (1.2k BraTS SSL)
→ Fold 0 in progress, Fold 1-2 to follow
→ Baseline ViT for comparison with Phase 2 CNN
```

#### Experiment B — BrainSegFounder (HIGHEST PRIORITY NEXT)
```
Swin UNETR V1 + BrainSegFounder BRATS-pretrain.pt (42k brain MRI SSL)
→ Download weights from HuggingFace
→ Replace model_swinvit.pt with BrainSegFounder weights
→ Same 3-fold training pipeline
→ Expected: significantly better Dice + richer embeddings
```

> [!TIP]
> BrainSegFounder may need different weight loading than `model.load_from()`.
> Check if it provides full SwinUNETR state_dict vs encoder-only weights.
> If full state_dict: use `model.load_state_dict(weights)`
> If encoder-only: use `model.load_from(weights=weight)` (same as current)

#### Experiment C — V2 Architecture (EASY ADD-ON)
```python
# In make_swinunetr_nb.py, change create_model():
model = SwinUNETR(
    in_channels=4, out_channels=3, feature_size=48,
    use_v2=True,  # ← One line change
    use_checkpoint=True, spatial_dims=3,
)
# Use BrainSegFounder weights with V2 architecture
# V2 adds ResConv blocks = better local features
```

### Priority Order:
1. **Complete Experiment A** (Fold 0 running → Folds 1, 2)
2. **Run Experiment B** (BrainSegFounder + V1) — highest expected gain
3. **Run Experiment C** (BrainSegFounder + V2) — if time permits

---

### Activity 2: TaViT Temporal Modeling

**Code:** `github.com/tom1193/time-distance-transformer`

| Step | Detail |
|---|---|
| Input | 768-dim embeddings per timepoint + Δt between scans |
| Architecture | TaViT: Time-encoding vectors + Temporal Emphasis scaling |
| Training | Predict tumor evolution (growth vs stable vs shrinkage) |
| Comparison | Train TaViT on CNN embeddings (Phase 2) vs ViT embeddings (Phase 3) |

Implementation plan:
1. Clone `time-distance-transformer` repo
2. Adapt the data loader for our embedding format
3. Train on Phase 2 CNN embeddings → baseline TaViT
4. Train on Phase 3 ViT embeddings → expected improvement
5. Statistical comparison (paired t-test)

---

### Activity 3: Evaluation Battery

Based on the research, implement these metrics:

#### Segmentation Metrics (standard):
- Dice score (WT, TC, ET) — 3-fold CV mean ± std
- HD95 — Hausdorff distance at 95th percentile

#### Embedding Quality Metrics (from research):
| Metric | What it measures | Implementation |
|---|---|---|
| **Linear Probing** | Task-relevant signal in frozen embeddings | Train linear head on frozen embeddings → predict tumor growth |
| **Temporal Consistency** | Smoothness across timepoints | Cosine similarity between consecutive visit embeddings |
| **Intra/Inter-class Ratio** | Patient-specific tracking quality | Within-patient vs between-patient embedding distances |
| **TaViT AUC** | Temporal prediction accuracy | AUC for progression prediction using TaViT |
| **t-SNE/UMAP Visualization** | Visual cluster quality | Color by patient, timepoint, growth status |

#### Comparison Table for Thesis:
| Metric | Phase 2 CNN | Phase 3a (V1+SSL) | Phase 3b (V1+BSF) | Phase 3c (V2+BSF) |
|---|---|---|---|---|
| Dice (WT) | X.XXX | X.XXX | X.XXX | X.XXX |
| Dice (TC) | X.XXX | X.XXX | X.XXX | X.XXX |
| Dice (ET) | X.XXX | X.XXX | X.XXX | X.XXX |
| Linear Probe Acc | X.XX% | X.XX% | X.XX% | X.XX% |
| Temporal Consistency | X.XXX | X.XXX | X.XXX | X.XXX |
| TaViT Progression AUC | X.XXX | X.XXX | X.XXX | X.XXX |

---

## Timeline (Remaining Weeks 8-11)

```
Week 8 (NOW):
  ├── Fold 0 training (running) ✅
  ├── Download BrainSegFounder weights
  └── Investigate weight loading format

Week 9:
  ├── Complete Folds 1-2 with current weights
  ├── Run BrainSegFounder experiment (all 3 folds)
  └── Clone time-distance-transformer repo

Week 10:
  ├── Extract all embeddings (4 encoder variants)
  ├── Implement TaViT temporal modeling
  └── Train TaViT on CNN vs ViT embeddings

Week 11:
  ├── Compute embedding quality metrics
  ├── Generate visualizations (t-SNE, training curves)
  ├── Statistical comparisons
  └── Write results section
```

---

## Summary: What Each AI Got Right

| Finding | DeepSeek R1 | DeepSeek R2 | GPT R1 | GPT R2 |
|---|---|---|---|---|
| BrainSegFounder (42k, 4-ch, HF) | ❌ | ❌ | ❌ | ✅ 🏆 |
| MONAI `use_v2=True` | ❌ | ✅ | ❌ | ❌ |
| TaViT code (GitHub) | ❌ | ❌ | ❌ | ✅ |
| SwinBrain (75k head MRI) | ❌ | ✅ | ❌ | ❌ |
| UKBOB/Swin-BOB | ❌ | ❌ | ✅ | ✅ |
| Triad (131k, GitHub) | ❌ | ❌ | ✅ | ✅ |
| BraTS-METS weights exist | ❌ | ❌ | ✅ | ✅ |
| Embedding evaluation metrics | ❌ | ✅ | ❌ | ✅ |

**No single AI found everything. The combination of all 4 responses gives the complete picture.**
