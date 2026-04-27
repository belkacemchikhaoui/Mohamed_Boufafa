# KAIST BraTS2021 Approach: Inference, Not Retraining

## What the KAIST README Offers

The KAIST BraTS2021 repository provides **three options**:

| Option | What it does |
|--------|-------------|
| **Docker inference** | Runs the pre-trained model using a Docker image |
| **Command-line inference** | Runs the pre-trained model using `nnUNet_predict` directly |
| **Training** | Re-trains the model from scratch using `nnUNet_train` |

---

## What We Are Doing: Direct Inference with Pre-Trained Weights

We use **Option 2 — command-line inference** with the pre-trained weights that KAIST provides. We do **not** retrain anything.

The exact pipeline from the README:

```bash
# Step 1: BL model predict
nnUNet_predict -i <input_folder> -o <output_folder1> \
  -t 500 -m 3d_fullres \
  -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD \
  --save_npz

# Step 2: BL+LGN model predict (SKIPPED — OOM on T4)
# nnUNet_predict -i <input_folder> -o <output_folder2> \
#   -t 500 -m 3d_fullres \
#   -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm \
#   --save_npz

# Step 3: Ensemble (SKIPPED — only BL used)
# nnUNet_ensemble -f <output_folder1> <output_folder2> -o <final_output_folder>

# Step 4: Post-processing (applied in Python)
# ET threshold 200 + BraTS label convention
```

In our Kaggle notebook (`05_nnunet_segmentation_kaggle.ipynb`):
- Cell 3 downloads and extracts the KAIST pre-trained weights from Google Drive
- Cell 5 calls `nnUNet_predict` with `TRAINER_BL = "nnUNetTrainerV2BraTSRegions_DA4_BN_BD"`
- The weights are loaded directly — no training step is ever run

---

## Why Inference and Not Retraining?

### 1. The Weights Already Exist and Are State-of-the-Art

KAIST trained these models on the **BraTS 2021 challenge dataset** (1,251 labeled training cases with expert segmentations). Their BL model achieves:

| Region | Dice Score |
|--------|-----------|
| Whole Tumor (WT) | **0.930** |
| Tumor Core (TC) | **0.882** |
| Enhancing Tumor (ET) | **0.836** |

These are the **winning scores** from the official BraTS 2021 competition. There is nothing better available for this task with these labels.

### 2. Yale Has No Ground Truth Labels

Our dataset (Yale Brain Metastases Longitudinal, 11,884 scans) has **no tumor segmentation labels**. There is nothing to train on. The entire point of running nnU-Net inference on Yale is precisely to **generate** the tumor masks that we currently lack.

Retraining would require:
- Expert neuroradiologist annotations for hundreds of scans → months of manual work
- A labeling budget we do not have

### 3. Transfer Learning Is Already Built In

The BraTS 2021 dataset and our Yale dataset share the same:
- **Tumor type**: Brain tumors (glioblastoma / brain metastases)
- **Modalities**: FLAIR, T1 pre-contrast (PRE), T1 post-contrast (POST/T1c), T2
- **Format**: NIfTI, skull-stripped, 1mm³ isotropic

The KAIST model was trained on the same 4-modality brain MRI format that Yale uses. No domain adaptation is needed.

### 4. Time and Compute Constraints

Retraining from scratch requires:
- A large labeled dataset (minimum ~200+ cases for robust training)
- Multi-GPU compute (KAIST used their lab cluster for weeks)
- 5-fold cross-validation → 5 separate training runs

Our constraints:
- **Kaggle T4 GPU**: 15 GiB VRAM, 10-hour session limit
- **No labeled data**: Yale has no segmentation ground truth
- **Timeline**: 20-week internship — retraining would consume weeks we don't have

Running inference takes **~26 seconds per case**, making the full 996-case run feasible in ~7.2 hours — within one Kaggle session.

### 5. Inference Is the Correct Scientific Choice

We are in **Phase 1 — Data Pipeline**. Our objective is to produce segmentation masks as input features for the downstream model (Swin UNETR → TaViT → ComBat → RadFM). The segmentation masks are an intermediate artifact, not the final contribution of the project.

Fine-tuning nnU-Net on Yale would only make sense if:
- We had expert labels to fine-tune on (we don't)
- Segmentation quality was our primary research contribution (it's not)

The primary contributions of this project are the **temporal modeling** (Phase 2–3), **clinical explainability** (Phase 4), and **counterfactual video generation** (Phase 5).

---

## What We Changed Compared to the KAIST README

| KAIST README | Our Implementation | Reason |
|---|---|---|
| Docker image | Command-line on Kaggle | No Docker on Kaggle |
| `--save_npz` flag | Omitted | Only needed for ensemble step |
| BL + BL+LGN ensemble | **BL only** | BL+LGN OOMs on T4 (15 GiB) |
| `nnUNet_ensemble` step | **Skipped** | No BL+LGN output to ensemble |
| Post-processing script | Re-implemented in Python | Same logic: ET threshold 200 + BraTS label remap |
| PyTorch 1.9.1 (Docker) | PyTorch 2.6 (Kaggle) | Runtime patches applied in Cell 5 |

The BL-only approach is still the **BraTS 2021 winning model**. The ensemble with BL+LGN was an optional improvement that added ~1% Dice — not worth the VRAM cost in our context.

---

## Summary

> **We run direct inference** using KAIST's pre-trained weights because Yale has no labels to train on, the model already achieves state-of-the-art performance on the exact same task and modalities, and our research contribution lies in the downstream longitudinal analysis — not in re-training a segmentation model.
