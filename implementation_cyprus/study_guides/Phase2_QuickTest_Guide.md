# Phase 2 — Activity 2: Quick Test Guide

**Date:** March 25, 2026  
**Status:** 🔜 READY TO RUN  
**Script:** `scripts/phase2_kaggle_training.py --mode quick_test --fold 0`  
**Duration:** ~2 hours on Kaggle T4

---

## Purpose

Before committing 16+ hours of GPU quota to full training, this 2-hour quick test validates that:
1. Data loads correctly from the Kaggle Dataset
2. The 3D U-Net model fits in T4 VRAM (15 GB)
3. The loss function decreases (model is learning)
4. Segmentation produces meaningful Dice scores
5. Embedding extraction from the bottleneck layer works

---

## How to Run on Kaggle

### Step 1: Create a New Kaggle Notebook

1. Go to kaggle.com → **New Notebook**
2. Under **Settings**:
   - Accelerator: **GPU T4 x2**
   - Language: **Python**
   - Internet: **On** (needed for pip install)
3. Under **Data** → **Add Data**:
   - Add your uploaded Cyprus PROTEAS dataset
   - Make sure `data_splits.json` is in the dataset root

### Step 2: Install Dependencies (First Cell)

```python
!pip install monai==1.3.0 nibabel scikit-learn --quiet
```

### Step 3: Upload and Run the Training Script

**Option A — Copy-paste into notebook cells:**
Copy the contents of `scripts/phase2_kaggle_training.py` into cells.

**Option B — Upload as a utility script:**
```python
# Upload phase2_kaggle_training.py to /kaggle/working/ first, then:
!python /kaggle/working/phase2_kaggle_training.py --mode quick_test --fold 0
```

**Option C — Clone from GitHub (if you push the scripts):**
```python
!git clone https://github.com/YOUR_REPO.git /kaggle/working/repo
!python /kaggle/working/repo/scripts/phase2_kaggle_training.py --mode quick_test --fold 0
```

### Step 4: Check Results

After ~2 hours, you should see output like:
```
[CHECK 1] Data loading...
  ✅ Image shape: torch.Size([2, 4, 96, 96, 96])
  ✅ Label shape: torch.Size([2, 1, 96, 96, 96])
  
[CHECK 2] Forward pass...
  ✅ Output shape: torch.Size([2, 4, 96, 96, 96])
  ✅ No OOM error

[CHECK 3] Training for 15 epochs...
  Epoch  0 | Loss: 1.2345 | Dice: 0.0231
  Epoch  5 | Loss: 0.8123 | Dice: 0.1540
  Epoch 14 | Loss: 0.5678 | Dice: 0.3200

[CHECK 4] Embedding extraction...
  ✅ Embedding dim: (256,)
  ✅ Total embeddings: 170

  ✅ PIPELINE VALIDATED — Proceed to full training
```

---

## Decision Matrix After Quick Test

| Observation | Action |
|-------------|--------|
| ✅ Loss drops steadily, Dice > 0.3 on at least 1 class | **Proceed to full training:** `--mode train --fold 0` |
| ⚠️ Loss drops slowly, Dice 0.1-0.3 | Try: `--lr 3e-4`, add more augmentation, or try `batch_size=4` |
| ❌ Loss flat or NaN | Debug: check label values (`torch.unique`), verify mask alignment with images |
| ❌ Dice stays at 0 for all classes | Check: are masks loading correctly? Are labels 0,1,2,3? Is loss function seeing one-hot? |
| 💥 OOM crash | Reduce: `resolution=[64,64,64]` or `batch_size=1` |
| 💥 Data loading crash | Fix: check NIfTI paths, verify `data_splits.json` is accessible |

---

## Quick Test Configuration

```python
config = {
    'resolution': [96, 96, 96],    # Standard BraTS resolution
    'batch_size': 2,                # Safe for T4 15GB
    'epochs': 15,                   # Just enough to see learning
    'lr': 1e-4,                     # Conservative learning rate
    'patience': 10,                 # Don't stop too early
    'val_interval': 1,              # Validate every epoch (quick test only)
    'in_channels': 4,               # T1, T1c, T2, FLAIR
    'out_channels': 4,              # BG, NCR, ET, ED
}
```

### Expected Timeline on T4

| Time | What Happens |
|------|-------------|
| 00:00 - 00:05 | Install MONAI, load data |
| 00:05 - 00:15 | Cache dataset transforms |
| 00:15 - 01:30 | Train 15 epochs (~5 min/epoch) |
| 01:30 - 01:45 | Validate on test fold |
| 01:45 - 02:00 | Extract embeddings, save results |

### Expected VRAM Usage

| Component | Memory |
|-----------|--------|
| 3D U-Net model (5.3M params) | ~21 MB |
| Feature maps (fwd + bwd) | ~300 MB |
| Batch of 2 × (4, 96, 96, 96) | ~270 MB |
| Optimizer states | ~42 MB |
| **Total** | **~1.0 GB** (of 15 GB available) |

---

## What Gets Saved

After quick test, these files are saved to `/kaggle/working/phase2_outputs/`:

```
phase2_outputs/
├── checkpoints/
│   ├── unet_fold0_best.pth     # Best model from 15 epochs
│   └── unet_fold0_last.pth     # Last epoch (for resume)
├── metrics/
│   └── fold0_metrics.json      # Loss curve + Dice per epoch
├── embeddings/
│   ├── cnn_embeddings_fold0.npz      # 256-dim vectors
│   └── cnn_embeddings_fold0_meta.json # Patient/visit metadata
└── figures/
    └── (empty, populated during full training)
```

**Important:** Download these files from Kaggle Output before the session ends! They are needed for:
- Resume training in the next session
- Comparing with Phase 3 (ViT) results
- Running the embedding quality tests

---

## Cross-Session / Cross-Account Resume

If you need to continue training on a **different Kaggle account** or **new session**:

1. Download `unet_fold0_last.pth` from previous session's output
2. Upload it as part of your new Dataset (or use Kaggle Dataset versioning)
3. Run:
   ```python
   !python phase2_kaggle_training.py --mode train --fold 0 \
       --resume /kaggle/input/YOUR-CHECKPOINT-DATASET/unet_fold0_last.pth
   ```

The checkpoint contains:
- Model weights
- Optimizer state (AdamW momentum)
- Scheduler state (cosine annealing position)
- Training history (loss curve, validation Dice)
- Best Dice so far (for early stopping comparison)

This means training resumes **exactly** where it left off.

---

## After Quick Test: Full Training Plan

```
Session 1:  python phase2_kaggle_training.py --mode train --fold 0
            python phase2_kaggle_training.py --mode train --fold 1
            (~8 hrs total, may need to split across sessions)

Session 2:  python phase2_kaggle_training.py --mode train --fold 2
            python phase2_kaggle_training.py --mode extract --fold all
            (~6 hrs total)
```
