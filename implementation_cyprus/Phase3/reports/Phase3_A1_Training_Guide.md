# Phase 3 Activity 1 — Swin UNETR Training Guide

## Quick Start

### Step 1: Upload Pretrained Weights to Kaggle

```bash
cd implementation_cyprus/Phase3
python3 scripts/upload_swinunetr_weights.py
```

This uploads `model_swinvit.pt` (393 MB) to `mohamedmohamed23/swinunetr-pretrained-weights`.

### Step 2: Create Kaggle Notebook

1. Go to [kaggle.com/mohamedmohamed23](https://kaggle.com/mohamedmohamed23) → **New Notebook**
2. Upload `Phase3/notebooks/Phase3_A1_SwinUNETR_Training.ipynb`
3. **Add Data Sources** (right panel → +Add Data):
   - `boufafamoamed/cyprus-proteas-brain-mets` (same dataset from Phase 2)
   - `mohamedmohamed23/swinunetr-pretrained-weights` (just uploaded)
4. **Settings**:
   - Accelerator: **GPU T4 × 2**
   - Internet: **ON** (for pip install)
   - Persistence: **Files** (keeps outputs between sessions)

### Step 3: Run Each Fold

| Session | Fold | Account | Action |
|---|---|---|---|
| **Session 1** | Fold 0 | mohamedmohamed23 | Run All → wait ~5-6 hrs |
| **Session 2** | Fold 1 | mohamedmohamed23 | Relaunch → auto-detects fold 0 done → trains fold 1 |
| **Session 3** | Fold 2 | mohamedmohamed23 | Relaunch → auto-detects folds 0,1 done → trains fold 2 |

**Important:** The notebook auto-detects completed folds. Just relaunch — it picks up where it left off.

---

## Estimated Training Time

### Per-Epoch Timing (Swin UNETR vs Met-Seg)

| Component | Met-Seg (Phase 2) | Swin UNETR (Phase 3) |
|---|---|---|
| Model size | 28M params | 62M params |
| Batch size | 2 | 1 |
| Crops per volume | 3 | 2 |
| Forward pass | ~0.3 sec | ~0.8 sec |
| Backward pass | ~0.5 sec | ~1.2 sec (gradient checkpointing) |
| **Train epoch** | **~2.3 min** | **~4-5 min** |
| **Val epoch** | **~6 min** | **~8 min** |
| **Time per fold** | **217 min (~3.6 hrs)** | **~350 min (~5.8 hrs)** |

### Total Time Budget

| Phase | Sessions | Time per Session | Total GPU |
|---|---|---|---|
| Fold 0 | 1 | ~6 hrs (5.8 train + 0.4 embed) | 6 hrs |
| Fold 1 | 1 | ~6 hrs | 6 hrs |
| Fold 2 | 1 | ~6 hrs | 6 hrs |
| **Total** | **3** | | **~18 hrs** |

**Kaggle limit:** 30 hrs/week per account. All 3 folds fit within one account's weekly quota.

### Memory (VRAM) Estimates

| Component | VRAM Usage |
|---|---|
| Swin UNETR (62M params, fp32) | 248 MB |
| Swin UNETR (fp16 with AMP) | 124 MB |
| Forward + activation maps (96³) | ~4 GB |
| Gradient checkpointing savings | -2 GB |
| **Total training (batch=1)** | **~4-5 GB** |
| **Total validation (sliding window)** | **~6-8 GB** |
| **T4 available** | **15 GB** ✅ |

If you see OOM errors:
- Reduce `CONFIG['num_samples']` from 2 to 1
- Reduce `CONFIG['patch_size']` from 96 to 64 (last resort — changes comparison fairness)

---

## What the Notebook Does

### Architecture (Swin UNETR)

```
Input: 4ch × 96 × 96 × 96 (T1, T1c, T2, FLAIR)
  ↓
Swin Transformer Encoder (5 stages, shifted-window attention):
  Stage 1: 48ch × 48³   → Skip Connection
  Stage 2: 96ch × 24³   → Skip Connection
  Stage 3: 192ch × 12³  → Skip Connection
  Stage 4: 384ch × 6³   → Skip Connection
  Bottleneck: 768ch × 3³ → THIS IS THE EMBEDDING
  ↓
CNN Decoder (residual blocks + skip connections):
  → 384ch × 6³ → 192ch × 12³ → 96ch × 24³ → 48ch × 48³
  ↓
Output: 3ch × 96 × 96 × 96 (WT, TC, ET masks)
```

### LR Schedule

```
LR
1e-4 |    ╭────────╮
     |   /          ╲
     |  /            ╲──────╮
1e-5 | /              ╲     │────────
     |/                ╲    │
1e-6 |──                ╲──╯
     └────────────────────────────────→ Epoch
       0   5          50  55   60
     warmup   cosine decay   finetune
```

### Per Fold, the Notebook:

1. **Loads data** using Phase 2's `data_splits.json` (same train/val splits)
2. **Creates Swin UNETR** and loads BraTS-2021 pretrained encoder weights
3. **Trains** for 60 epochs with DiceCE loss, AMP, gradient checkpointing
4. **Validates** every 5 epochs with sliding window inference
5. **Saves** best + latest checkpoints (for resume on relaunch)
6. **Extracts embeddings** (768-dim per scan) after training
7. **Generates** training curve figures

### Outputs Per Fold

| File | Size | Description |
|---|---|---|
| `checkpoints/swinunetr_fold{N}_best.pth` | ~500 MB | Best model weights |
| `checkpoints/swinunetr_fold{N}_latest.pth` | ~500 MB | Latest checkpoint (resume) |
| `embeddings/swinunetr_embeddings_fold{N}.npz` | ~1 MB | 768-dim embeddings |
| `figures/swinunetr_fold{N}_training_curves.png` | ~0.1 MB | Loss/Dice/LR plots |
| `metrics/swinunetr_fold{N}_metrics.json` | <1 KB | All logged metrics |

**Total per fold:** ~1 GB. Kaggle allows 19 GB output → plenty of room.

---

## Key Differences from Phase 2 (Met-Seg)

| Feature | Phase 2 (Met-Seg) | Phase 3 (Swin UNETR) |
|---|---|---|
| Architecture | DynUNet + DenseNet121 detector | Swin UNETR (encoder-decoder) |
| Two-stage pipeline | Yes (detector → segmenter) | No (single model) |
| Detector unfreezing | At epoch 30 | N/A |
| Loss | Dice + BCE (deep supervision) | DiceCE (single output) |
| LR schedule | Warmup → flat → step drops | Warmup → cosine → fine-tune |
| Batch size | 2 | 1 |
| Crops per volume | 3 | 2 |
| Embedding dim | 1024 (DynUNet bottleneck) | 768 (Swin bottleneck) |
| Embedding extraction | Hook on last downsample | Direct swinViT forward |
| Output structure | Same | Same |
| Data splits | Same | **Same (critical!)** |
| Transforms | Same | Same + RandShiftIntensity |

---

## Troubleshooting

### "Swin UNETR weights NOT found"
→ Add `mohamedmohamed23/swinunetr-pretrained-weights` as a data source. The notebook will fallback to downloading from MONAI (slower, needs internet ON).

### CUDA Out of Memory
→ Set `CONFIG['num_samples'] = 1` in the config cell. This reduces memory by ~30%.

### "Cannot find fold data"
→ The Cyprus dataset must be added as a data source. It should appear under `/kaggle/input/` with `data_splits.json` inside.

### Notebook times out (12 hrs)
→ The notebook saves `latest.pth` every epoch. Relaunch and it resumes from the last epoch automatically.

### Dice is much lower than Met-Seg
→ This is **expected and OK**. The goal is embedding quality, not segmentation Dice. Swin UNETR is pretrained on glioma (not metastasis), so Dice may be lower. The embedding evaluation (Activity 3) is what matters.

---

## After All 3 Folds Complete

Download from Kaggle output:
1. `checkpoints/swinunetr_fold{0,1,2}_best.pth` → `Phase3/weights/`
2. `embeddings/swinunetr_embeddings_fold{0,1,2}.npz` → `Phase3/embeddings/`
3. `figures/*` → `Phase3/outputs/training_figures/`
4. `metrics/*` → `Phase3/training_logs/`

Then proceed to **Activity 2** (TaViT temporal modeling) and **Activity 3** (16-test evaluation battery).
