# Kaggle Upload Guide — BraTS 2024

## What You Need to Upload

You need **3 Kaggle datasets** (uploaded separately):

---

### Dataset 1: `brats2024-training-data` (Main training NIfTI)
**Upload:** The training NIfTI folders (once main zip finishes downloading + unzipping)

```
Content:
  BraTS-GLI-00005-100/
    BraTS-GLI-00005-100-t1n.nii.gz
    BraTS-GLI-00005-100-t1c.nii.gz
    BraTS-GLI-00005-100-t2w.nii.gz
    BraTS-GLI-00005-100-t2f.nii.gz
    BraTS-GLI-00005-100-seg.nii.gz
  BraTS-GLI-00005-101/
    ...
  (all ~1350 training + 271 additional training folders)
```

**Size:** ~28 GB uncompressed (~23 GB as zip)
**How:** Upload the zip file directly to Kaggle — it auto-extracts

> ⚠️ If the main zip is too large for Kaggle (>20GB per file), split into 2 datasets:
> - `brats2024-training-main` (~23 GB zip)
> - `brats2024-training-additional` (~5 GB, the 271 folders already extracted)

---

### Dataset 2: `brats2024-metadata` (Small files)
**Upload these files together (~1 MB total):**

```
brats2024-metadata/
├── BraTS-PTG supplementary demographic information and metadata.xlsx
├── scan_index.json          ← from Phase 1 output
├── longitudinal_index.json  ← from Phase 1 output
├── tumor_volumes.csv        ← from Phase 1 output
└── CITATIONS.bib
```

**Source paths:**
- Excel: `/home/moamed/HDD/brats2024_posttreatment/BraTS-PTG supplementary...xlsx`
- Phase 1 outputs: `implementation_brats2024/Phase1/outputs/`

---

### Dataset 3: `brats2024-pretrained-weights` (Model weights)

The easiest way to get pretrained weights is by running a small script to download them via the `monai` package, then uploading those downloaded files to Kaggle. 

Run this in your terminal:

```bash
# 1. Download SegResNet weights
python3 -c "from monai.bundle import download; download('brats_mri_segmentation', bundle_dir='./models/segresnet')"

# 2. Download SwinUNETR weights
python3 -c "from monai.bundle import download; download('swin_unetr_btcv_segmentation', bundle_dir='./models/swinunetr')"
```

*Note: For the DynUNet/nnU-Net model, we will train it from scratch in the notebook since a "universal" pretrained weight file isn't reliably hosted on direct download links anymore.*

Zip up the resulting `./models` directory and upload that to Kaggle.

---

## Upload Summary

| Dataset | Size | Contents | Needed For |
|---------|------|----------|------------|
| `brats2024-training-data` | ~28 GB | NIfTI patient folders | Phase 2 + 3 training |
| `brats2024-metadata` | ~1 MB | Excel + scan_index + volumes | All phases |
| `brats2024-pretrained-weights` | ~300 MB | SegResNet + SwinUNETR | Phase 2 + 3 |

## How to Attach in Kaggle Notebook

In your Kaggle notebook, click **"Add Data"** → search for your datasets.
They'll appear at:
```
/kaggle/input/brats2024-training-data/
/kaggle/input/brats2024-metadata/
/kaggle/input/brats2024-pretrained-weights/
```

The notebooks already search these paths automatically.

---

## Step-by-Step Upload Order

1. **First:** Upload `brats2024-metadata` (tiny, instant)
2. **Second:** Upload `brats2024-pretrained-weights` (download via monai, ~300MB)
3. **Third:** Upload `brats2024-training-data` (wait for main zip to finish downloading)
4. **Then:** Run Phase 1 EDA locally or on Kaggle
5. **Then:** Upload Phase 1 outputs to metadata dataset (scan_index.json etc.)
6. **Then:** Run Phase 2 notebooks on Kaggle GPU

## Current Download Status

| File | Status |
|------|--------|
| Main Training zip (~23 GB) | 🔄 Still downloading (33 GB .crdownload) |
| Additional Training (271 folders) | ✅ Extracted (5.1 GB) |
| Validation (188 folders) | ⚠️ Partially extracted (7 folders/273 MB) |
| Metadata Excel | ✅ Available |
