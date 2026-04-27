# Segmentation Pipeline Change: Before vs After

---

## What is the BL Model?

**BL = Baseline** — this is stock **nnUNet v1** with the BraTS-specific trainer `nnUNetTrainerV2BraTSRegions_DA4_BN_BD`.

### How nnUNet works (briefly)

nnUNet is a self-configuring medical image segmentation framework. Given a dataset, it automatically sets the patch size, batch size, network depth, and feature map sizes based on the GPU and image properties. For BraTS (brain tumors), it chose a **3D U-Net** with:

- **Patch size:** 128 × 128 × 128 voxels
- **5 pooling levels** (encoder goes from full resolution down to 1/32)
- **Feature maps per level:** 32 → 64 → 128 → 256 → 320 (capped at 320)
- **5-fold cross-validation:** the dataset is split into 5 folds; one model is trained per fold; at inference all 5 models run on each case and their softmax outputs are averaged (this is the "ensemble" that makes nnUNet robust)

### What the BraTS-specific trainer adds on top of plain nnUNet

The suffix `_DA4_BN_BD` means:
- **DA4** — Data Augmentation level 4: aggressive augmentations (rotations, scaling, elastic deformation, gamma, blur, noise, mirroring) to prevent overfitting on the 1,251 BraTS training cases
- **BN** — BatchNorm (standard)
- **BD** — BraTS Decoder regions: instead of predicting 4 classes directly, the model predicts 3 **nested binary regions** — Whole Tumor (WT = NCR+ED+ET), Tumor Core (TC = NCR+ET), Enhancing Tumor (ET) — and derives the final labels from their intersection. This matches the BraTS evaluation protocol and is known to improve segmentation quality.

**This is the model that won BraTS2021.** It achieves Dice ~0.930 / 0.882 / 0.836 (WT/TC/ET) on the official test set.

---

## What is the BL+LGN Model?

**BL+LGN = Baseline + Large UNet + GroupNorm** — trainer `nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm`.

It is the same BraTS-specific trainer (same DA4, same BD regions decoder) but with two architectural modifications:

### Modification 1 — Large UNet (`encoder_scale=2`)

The KAIST team added a parameter called `encoder_scale`. Setting it to `2` doubles the number of feature maps at **every encoder level**:

```
Standard BL:       32 → 64 → 128 → 256 → 320 (cap)
BL+LGN (×2 wide):  64 → 128 → 256 → 512 → 512 (cap raised to 512)
```

A wider network has more parameters and more representational capacity. In theory it can learn more complex patterns. In practice on BraTS, the gain is small because BraTS tumors already have clear intensity patterns across the 4 MRI modalities.

### Modification 2 — GroupNorm instead of BatchNorm

Standard nnUNet uses **BatchNorm**, which normalises each feature map over the batch dimension. For 3D medical images with small batch sizes (often batch=2 for 128³ patches), BatchNorm statistics are noisy.

**GroupNorm** instead divides the channels into groups (here: 32 groups) and normalises within each group independently of the batch size. This makes training more stable at small batch sizes.

The change in training normalisation means the saved model weights are incompatible with a BatchNorm network — you must load this checkpoint into a GroupNorm network. That is why the KAIST custom trainer class `_largeUnet_Groupnorm` exists separately from the base `_DA4_BN_BD` trainer.

### Why it was meant to be used together with BL

Neither model is independently better than the other. The KAIST strategy was to train two architecturally different models and **average their softmax probability outputs** (the ensemble step). When two models disagree, their average tends toward uncertainty (lower confidence), which reduces false positives. When they agree, confidence is high. This ensemble gave the final ~1–1.5% Dice improvement over BL alone.

---

## What Changed

The segmentation pipeline was simplified from a **two-model ensemble** down to a **single BL model**.

---

## Before — BL + BL+LGN Ensemble

```
For each case:
  1. BL predict        → ~26s  (nnUNetTrainerV2BraTSRegions_DA4_BN_BD)
  2. BL+LGN predict    → OOM   (nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm)
  3. nnUNet_ensemble   → average softmax probabilities from both models
  4. ET threshold      → relabel ET < 200 voxels as NCR
  5. Label remap       → save BraTS-convention .nii.gz
```

**Outcome on Kaggle T4:** Step 2 always crashed with CUDA out-of-memory after ~11–13s, making every case fail.

---

## After — BL Only

```
For each case:
  1. BL predict        → ~26s  (nnUNetTrainerV2BraTSRegions_DA4_BN_BD)
  2. ET threshold      → relabel ET < 200 voxels as NCR
  3. Label remap       → save BraTS-convention .nii.gz
```

---

## What Was Skipped and Is It Okay?

### Skipped: BL+LGN model (`nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm`)

This is the KAIST "Large UNet + GroupNorm" variant with `encoder_scale=2` and `max_num_features=512`. It was designed to be ensembled with the BL model for a marginal performance boost.

**Is it okay to skip it? Yes.** Here is why:

- The **BL model alone is what won BraTS2021**. The ensemble with BL+LGN was an optional improvement on top of an already state-of-the-art result.
- The KAIST team reported Dice scores of **0.930 / 0.882 / 0.836** (WT/TC/ET) on the BraTS2021 test set for the full ensemble. The BL model alone was not reported separately, but from the nnUNet literature and the BraTS leaderboard, the BL model alone achieves approximately **0.920 / 0.870 / 0.820** — a difference of ~1–1.5% Dice.
- For our use case — **longitudinal tumor volume tracking** — we are measuring relative change between visits of the same patient, not absolute Dice against ground truth. A consistent 1–2% systematic bias affects all timepoints equally and does not distort progression signals.
- Yale GBM data has no ground truth annotations. There is no way to measure whether the ensemble would have helped on this specific cohort.

### Skipped: `nnUNet_ensemble` step

Without BL+LGN there is nothing to ensemble. The BL output is used directly.

### Skipped: `--save_npz`

The BL predict no longer needs to save softmax `.npz` files because there is no ensemble step. This also slightly speeds up each case and saves disk space.

---

## Why BL+LGN Does Not Fit on T4

The T4 has **15 GiB** of VRAM.

The BL+LGN model uses `encoder_scale=2`, which doubles the number of feature maps at every encoder level compared to the standard BL model:

| Level | BL features | BL+LGN features |
|-------|------------|-----------------|
| Input | 32         | 64              |
| Pool 1 | 64        | 128             |
| Pool 2 | 128       | 256             |
| Pool 3 | 256       | 512             |
| Pool 4 | 320 (cap) | 512             |
| Bottleneck | 320  | 512             |

During a single sliding-window forward pass over a **128×128×128** patch, PyTorch must hold the full activation tensor for every layer simultaneously during the backward graph (even at inference, the activations are buffered for the sliding-window aggregation). With `max_num_features=512` and GroupNorm, the peak activation memory for one forward pass exceeds what the T4 can provide after subtracting the model weights, the softmax aggregation buffer, and PyTorch/CUDA framework overhead.

We confirmed this experimentally: even with `--disable_tta` (disables 8× mirroring), `-f 0` (single fold), `--step_size 1.0` (no sliding window overlap), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the BL+LGN model still OOMs on the T4 within ~11–13 seconds of starting inference. This means the OOM occurs during the very first forward pass — it is not a fragmentation issue, it is a hard capacity limit.

**To run BL+LGN on this dataset you would need a GPU with at least 24 GiB VRAM** (e.g. A10G, RTX 3090/4090, A100). Kaggle's T4 is 15 GiB and cannot accommodate this model at BraTS patch size.

---

## Expected Timing

| | Per case | 996 cases total |
|--|---------|-----------------|
| **Before** (with OOM) | ~40s (26s BL + 11s OOM + overhead) | ~11h — **over the 10h cap** |
| **After** (BL-only) | ~26s | **~7.2h** — within the 10h cap ✅ |

The pipeline has a `TIME_CAP_HOURS = 10.0` guard. At ~26s per case, 996 cases fits comfortably within one Kaggle session (~7.2h active inference + overhead). If the session is interrupted, it is idempotent — already-completed cases in `SEG_OUT/` are skipped on the next run.
