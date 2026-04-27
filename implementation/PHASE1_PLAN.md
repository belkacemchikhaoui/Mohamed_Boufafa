# Phase 1 + 2 — One-Day Execution Plan
## Yale Brain Metastases Dataset · March 10, 2026

> **Goal**: Complete Phase 1 (data pipeline + EDA) AND Phase 2 (CNN baseline design) in a single working day.  
> **Constraint**: No GPU on local machine. No nnU-Net/BraTS Toolkit run today — those are scheduled as async background jobs.  
> **Deliverables by end of day**: Data inventory CSV, full EDA report with figures, preprocessing pipeline code (ready to run), CNN baseline code + Kaggle training plan ready to submit.

---

## 🖥️ Machine Reality Check

| Resource | Your Machine | Impact |
|----------|-------------|--------|
| **CPU** | Intel i7-1065G7 @ 1.3GHz · 4 cores / 8 threads | Can run analysis scripts, pandas, nibabel |
| **RAM** | 12 GB total · ~5.4 GB available | Can load single NIfTI files (~50–500MB each), NOT all 33k at once |
| **GPU** | ❌ None | Cannot train CNNs or run nnU-Net locally |
| **Disk** | 22 GB free · Data = 43 GB | Data already there. Processed outputs need ~97 GB (train) — **need to free ~75 GB** |
| **Python env** | `datapre` conda env | Has: nibabel, pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, scikit-image, itk-elastix, SimpleITK, **PyTorch 2.10**, tqdm, openpyxl |
| **Missing** | monai, torchvision, timm | Install in datapre if needed (pip, fast) |

### ⚠️ What This Means for Today

- ✅ **CAN do today (CPU-only)**: Data audit, EDA figures, metadata parsing, write all pipeline scripts, define splits, write CNN baseline code
- ⏳ **Cannot run today but START tonight**: BraTS Toolkit preprocessing (30–60h on CPU), nnU-Net inference (GPU needed → Kaggle)
- 🎯 **Strategy**: Write all code today → launch CPU jobs tonight as background → GPU jobs on Kaggle

---

## 🔍 What We Already Have (Ground Truth From Disk)

Before planning anything, here is the **exact state** of the data right now:

### ✅ Data Already Downloaded

| Fact | Value |
|------|-------|
| **Location** | `implementation/Data/Yale-Brain-Mets-Longitudinal/` |
| **Total patients** | **1,430** (confirmed: `1430` folders) |
| **Total visits** | **11,877** visit folders |
| **Total NIfTI files** | **33,811** `.nii.gz` files |
| **Clinical metadata** | `Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx` |
| **Date range** | 2004-07-08 → 2023-03-14 (19 years) |

### ✅ Folder Structure (Already Clean)

```
Yale-Brain-Mets-Longitudinal/
├── YG_<PATIENT_ID>/            ← 1,430 patients
│   ├── YYYY-MM-DD/             ← visit date folder
│   │   ├── YG_<ID>_<DATE>_<TIME>_FLAIR.nii.gz
│   │   ├── YG_<ID>_<DATE>_<TIME>_POST.nii.gz   ← T1 post-contrast (= T1c)
│   │   ├── YG_<ID>_<DATE>_<TIME>_PRE.nii.gz    ← T1 pre-contrast
│   │   └── YG_<ID>_<DATE>_<TIME>_T2.nii.gz
│   └── YYYY-MM-DD/
└── ...
```

> **Naming clarification**: `PRE` = T1 pre-contrast, `POST` = T1 post-contrast (= T1ce). These ARE the T1 and T1c sequences mentioned in papers.

### ✅ Modality Availability (Computed From Disk)

| Modality | Count | % of 11,877 visits |
|----------|-------|-------------------|
| FLAIR | 9,081 | **76.5%** |
| POST (T1c) | 8,994 | **75.7%** |
| PRE (T1) | 8,382 | **70.6%** |
| T2 | 7,408 | **62.4%** |
| **All 4 complete** | **4,405** | **37.1%** |

> ⚠️ **Critical finding**: Only 37% of visits have all 4 modalities. 63% are partial. You need a strategy for handling missing modalities.

### ✅ Visit Distribution (Computed From Disk)

| Category | Count |
|----------|-------|
| Patients with **1 visit only** | 131 (unusable for longitudinal) |
| Patients with **2 visits** | 173 |
| Patients with **≥3 visits** | **1,126** (core training set) |
| Patients with **≥5 visits** | 816 |
| Patients with **≥10 visits** | 271 |
| **Max visits** (YG_C9D2TEGNY08A) | **66 visits** over 11 years |

### ✅ Temporal Follow-up Statistics (Computed From Disk)

| Statistic | Value |
|-----------|-------|
| Median follow-up span | 345 days (~11 months) |
| 75th percentile | 857 days (~28 months) |
| Max follow-up | 5,061 days (13.9 years) |
| Min follow-up | 1 day (likely same-day repeat scan) |

### ✅ Temporal Density (Visits Per Year)

Data peaks 2013–2022. Very sparse before 2011. This matters for scanner harmonization — older scans (2004–2012) are predominantly 1.5T and will need more aggressive harmonization.

---

## ❌ What We DON'T Have (Must Build)

| Missing | Why It Matters | Planned for |
|---------|---------------|-------------|
| **No tumor segmentation masks** | Yale = raw scans only. nnU-Net must generate these | Kaggle overnight job |
| **No parsed clinical metadata** | Excel file exists but not linked to images | **Today morning** |
| **No inter-visit alignment** | Scans not registered across timepoints | After preprocessing completes |
| **No data split (train/val/test)** | Needed before any training | **Today afternoon** |
| **No missing-modality map** | Only 37% of visits are complete | **Today morning** |
| **No file integrity checks** | Possible corrupted NIfTIs | **Today morning** (header-only scan) |
| **No time-delta metadata** | Days between visits not computed | **Today morning** |

---

## ⚙️ Environment Setup (Do This First — 5 min)

Always use the `datapre` conda environment. It has PyTorch 2.10, nibabel, itk-elastix, SimpleITK, scikit-learn, scipy, pandas, seaborn.

```bash
conda activate datapre
pip install tqdm openpyxl  # if not already there (both are — just confirm)
# For CNN baseline later:
pip install torchvision timm monai
```

All scripts go in: `implementation/scripts/`  
All outputs go in: `implementation/outputs/`

---

## 📅 Today's Full Schedule (March 10, 2026)

| Time Block | Task | Machine | Output |
|-----------|------|---------|--------|
| ~~**09:00–10:30**~~ | ✅ Data audit + inventory script | Local CPU | `data_inventory.csv` |
| ~~**10:30–11:30**~~ | ✅ Clinical metadata parsing | Local CPU | `metadata.json` |
| ~~**11:30–12:30**~~ | ✅ File integrity scan (header-only) | Local CPU | `usability_report.csv` |
| ~~**12:30–13:00**~~ | ✅ ☕ Break | — | — |
| ~~**13:00–14:00**~~ | ✅ EDA figures (6 plots) | Local CPU | `eda/figures/*.png` |
| ~~**14:00–15:00**~~ | ✅ Train/val/test split + timelines | Local CPU | `splits.json`, `timelines.json` |
| ~~**15:00–16:30**~~ | ✅ Preprocessing pipeline notebook | Local CPU | `02_preprocessing_pipeline.ipynb` |
| **NEXT** | ⏳ Free disk (~75 GB) + run batch preprocessing | Local CPU overnight | `Data/processed/` (~97 GB) |
| **AFTER BATCH** | ⏳ Upload processed data to Kaggle | CLI upload | Kaggle Dataset |
| **AFTER UPLOAD** | ⏳ CNN baseline notebook on Kaggle | Kaggle T4 GPU | `02_cnn_baseline.ipynb` |

---

## 📋 Detailed Task Breakdown

### BLOCK 1 (09:00–10:30) — Data Audit & Inventory

**Script**: `implementation/scripts/00_data_audit.py`

**What it does**:
- Walks all 1,430 patient folders
- For each visit: records which of PRE/POST/T2/FLAIR are present
- Computes `days_since_baseline` per visit (visit date − patient's first visit date)
- Computes inter-visit interval in days
- Flags same-day visits (within 2 days → possible duplicate scan)
- Loads NIfTI header only (fast — no pixel data) for shape/voxel-size check
- Classifies each visit: `FULL` (4 modalities) | `PARTIAL_GOOD` (POST+FLAIR at minimum) | `UNUSABLE`

**Output**: `implementation/outputs/data_inventory.csv`
```
patient_id | visit_date | visit_idx | days_since_baseline | interval_days | has_PRE | has_POST | has_T2 | has_FLAIR | n_modalities | usability | shape_ok | voxel_size_mm
```

**Runtime estimate**: ~10–15 min on CPU (header-only reads are fast).

> ✅ **COMPLETED** — `data_inventory.csv` has **11,877 rows × 35 columns**. All visits scanned.

---

### BLOCK 2 (10:30–11:30) — Clinical Metadata Parsing

**Script**: `implementation/scripts/00b_parse_metadata.py`

**What it does**:
- Load `Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx` with pandas
- Print all column names first → understand what clinical variables exist
- Extract: patient_id, age, sex, and whatever treatment/scanner fields exist
- Join with `data_inventory.csv` on patient_id
- Export `metadata.json` keyed by patient_id

**Why this matters**: Phase 4 LLM needs age, sex, treatment info. If scanner manufacturer/field strength is in this Excel, it's needed for ComBat harmonization in Phase 3.

**Output**: `implementation/outputs/metadata.json`

> ✅ **COMPLETED** — `metadata.json` has **1,430 patients** with clinical fields linked.

---

### BLOCK 3 (11:30–12:30) — File Integrity Scan

**Script**: part of `00_data_audit.py` (already included above)

What "header-only check" means in nibabel:
```python
import nibabel as nib
img = nib.load(filepath)          # Loads header ONLY (fast, ~1ms per file)
shape = img.shape                  # e.g. (240, 240, 155) for a brain MRI
zooms = img.header.get_zooms()    # voxel spacing in mm
# Flag if: shape is not 3D, zooms are 0 or >10mm, file can't be opened
```

**Expected**: ~33,811 files × ~1ms = ~34 seconds total. Flag anything suspicious.

**Output**: `usability_report.csv` with per-file status (integrated into `data_inventory.csv`).

> ✅ **COMPLETED** — All 33,811 NIfTI headers checked. Usability classified per visit.

---

### BLOCK 4 (13:00–14:00) — EDA Figures (6 Plots)

**Script**: `implementation/scripts/00c_eda.py`

Generate and save these 6 figures to `implementation/outputs/eda/figures/`:

**Figure 1 — Visit Distribution**
Histogram: X = number of visits per patient, Y = patient count.  
Mark vertical lines at 1, 3, 5, 10 visits. Shows the usable pool clearly.

**Figure 2 — Year Distribution**
Bar chart: visits per year (2004–2023). Reveals the scanner technology shift era.

**Figure 3 — Modality Completeness Heatmap**
Matrix: rows = modality combinations (FULL / PRE+POST+FLAIR / POST+FLAIR / POST-only / etc.), columns = visit count and percentage. Visualizes the 37% full / 63% partial reality.

**Figure 4 — Follow-up Duration Histogram**
X = total follow-up span in months, Y = patient count. Show median (11 months) and 75th pct (28 months).

**Figure 5 — Inter-visit Interval Boxplot**
Distribution of days between consecutive visits across all patients. Shows that Yale has irregular, real-world sampling.

**Figure 6 — Longitudinal Patient Sample Plot**
For 10 randomly selected patients with ≥5 visits: plot their visit timeline on a horizontal axis (each dot = one visit, color = modality completeness). Gives intuition of what "longitudinal data" looks like here.

**Output**: `eda_report.md` with all 6 figures embedded + key numbers.

> ✅ **COMPLETED** — 6 EDA figures saved. `acquisition_parameters.csv` (33,804 × 11) generated.

---

### BLOCK 5 (14:00–15:00) — Train/Val/Test Split + Timelines JSON

**Script**: `implementation/scripts/00d_split.py`

**Split strategy**:
- Split **at patient level** (critical: never split one patient's visits across sets)
- Exclude patients with 0 usable visits
- Stratify by visit count tier: `[1-2, 3-5, 6-10, >10]` → equal representation of all temporal depths
- **Train 80% / Val 10% / Test 10%**
- Use `random_state=42` for reproducibility

**Patient pools**:
- 1,430 total → after removing 131 single-visit patients for longitudinal splits → ~1,299 for the main split
- (Single-visit patients go into a separate `single_visit` list — usable for Phase 2 CNN but not Phase 3+)

**Patient timelines JSON**:
```json
{
  "YG_04YGLO8ATWRL": {
    "visits": ["2013-07-25", "2013-08-02", "2013-09-07", ...],
    "days_from_baseline": [0, 8, 44, ...],
    "modalities": [["PRE","POST","T2","FLAIR"], ["POST","FLAIR"], ...],
    "n_visits": 14,
    "span_days": 1369
  }
}
```

**Output**: `splits.json` + `patient_timelines.json`

> ⚠️ Once saved: **do not regenerate splits**. All future scripts read `splits.json` and never re-split.

> ✅ **COMPLETED** — `splits.json` LOCKED: train=1,144 / val=143 / test=143 patients. `patient_timelines.json` done.

---

### BLOCK 6 (15:00–16:30) — Preprocessing Pipeline Script

**Script**: `implementation/scripts/01_preprocess.py`

Write the full pipeline script (do NOT run it today — just write + test on 1 patient):

```
Step A: Intensity normalization (SimpleITK N4 bias field correction)
Step B: Skull stripping (HD-BET via BraTS Toolkit)  
Step C: Intra-visit registration (align POST/T2/FLAIR → PRE within one visit, BraTS Toolkit)
Step D: Spatial resampling to 1mm³ isotropic
Step E: Save to Data/processed/
```

The script must:
- Accept a patient list (reads from `splits.json` train set by default)
- Process one visit at a time
- Log results to `outputs/preprocessing_log.csv`
- Skip already-processed visits (idempotent — can re-run safely)
- Handle missing modalities gracefully (skip T2 if absent, etc.)

**Test it on 1 patient (5 visits) before tonight's full launch.**

---

### BLOCK 7 — Phase 2 CNN Baseline on Kaggle

> ⏳ **STATUS: NEXT AFTER BATCH PREPROCESSING + KAGGLE UPLOAD**

**Notebook**: `implementation/notebooks/02_cnn_baseline.ipynb` (write locally, run on Kaggle T4)

#### Sub-step 7a — Free disk + run batch preprocessing locally
```bash
# 1. Free space first
conda clean --all          # ~2-5 GB
pip cache purge            # ~0.5 GB
# Move other large files to external drive if available

# 2. Open 02_preprocessing_pipeline.ipynb
# 3. Set DRY_RUN = False in the batch cell
# 4. Set PROCESS_SPLIT = 'train'  (already default)
# 5. Run the batch cell → launches ~39h overnight job
```

#### Sub-step 7b — Upload processed data to Kaggle
```bash
# Install Kaggle CLI
pip install kaggle

# Put your kaggle.json at ~/.kaggle/kaggle.json (from kaggle.com → Account → API)
chmod 600 ~/.kaggle/kaggle.json

# Initialize a new Kaggle dataset for your processed files
mkdir /tmp/kaggle_upload
cp implementation/outputs/data_inventory.csv /tmp/kaggle_upload/
cp implementation/outputs/splits.json /tmp/kaggle_upload/
cp implementation/outputs/metadata.json /tmp/kaggle_upload/
# (add processed NIfTIs after batch completes)

kaggle datasets init -p /tmp/kaggle_upload
# Edit the generated dataset-metadata.json with title + slug
kaggle datasets create -p /tmp/kaggle_upload --dir-mode tar

# To add processed NIfTIs later:
kaggle datasets version -p /tmp/kaggle_upload -m "Added processed train split"
```

#### Sub-step 7c — Kaggle notebook for CNN training
```python
# Kaggle notebook header (inside Kaggle):
DATA_ROOT = '/kaggle/input/yale-brain-mets-processed/'
INVENTORY = f'{DATA_ROOT}/data_inventory.csv'
SPLITS    = f'{DATA_ROOT}/splits.json'
PROCESSED = f'{DATA_ROOT}/processed/'

!pip install monai SimpleITK -q

# Then: write/paste the CNN training cells
# GPU: select T4 × 1 in Kaggle settings (free tier = 30h GPU/week)
# Expected training time: 50 epochs ≈ 2 hours on T4
```

**Task**: Single-timepoint tumor response prediction using CNN.

**Input**: One preprocessed visit (POST scan, 1mm³, skull-stripped) → predict "is there an active/growing tumor?" (binary, using automated nnU-Net masks as pseudo-labels).

**Architecture**: ResNet-3D (from torchvision or MONAI) — NOT training from scratch, fine-tuning from ImageNet-pretrained 2D weights adapted for 3D via MONAI.

```python
from monai.networks.nets import resnet50
model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
```

**Training details**:
- Input size: 128×128×64 (crop around brain region)
- Batch size: 4 (limited by GPU memory on Kaggle T4 = 16GB)
- Epochs: 50
- Optimizer: Adam lr=1e-4
- Loss: CrossEntropy
- Metrics: Accuracy, AUC, Dice (for segmentation extension)

**The script must have**:
1. `DataLoader` class — reads `data_inventory.csv` + `splits.json`
2. `train()` function
3. `validate()` function  
4. `main()` with argparse for config
5. Checkpointing (saves best model)
6. W&B logging (or CSV fallback)

Write the full script but **do not run it today** (needs Kaggle GPU).

---

### BLOCK 8 — Kaggle Setup + nnU-Net Launch Plan

**Set up the Kaggle notebook** for nnU-Net inference (this runs overnight/tomorrow):

#### Kaggle Notebook Plan — nnU-Net Segmentation

**Machine**: Kaggle T4 GPU (16GB VRAM) — Free tier: 30h/week GPU  
**Runtime per visit**: ~30–60s on T4 GPU → 11,877 visits = ~100–200h → need batched approach

**Strategy for Kaggle GPU limit**:
- Split all visits into 6 batches of ~2,000 visits each
- Run 1 batch per Kaggle session (≈ 3–4h each)
- Save outputs to Kaggle Dataset (persistent storage) between sessions
- Total: 6 sessions over ~3 days

**Kaggle notebook structure**:
```python
# 1. Mount the Yale data (upload as Kaggle Dataset — it's 43GB, use TCIA link or upload subset)
# 2. Install nnUNetv2
# 3. Download BraTS pretrained weights
# 4. Run inference on batch N (defined by BATCH_ID variable)
# 5. Save segmentation masks to output
```

> ⚠️ **Data upload constraint**: 43GB is too large to upload to Kaggle directly.  
> **Solution**: Upload ONLY the processed NIfTI stacks (after local BraTS Toolkit runs overnight), which will be smaller (~10–15GB compressed). OR use the TCIA direct download API inside the notebook.

#### Kaggle Notebook Plan — CNN Baseline Training

**Machine**: Kaggle T4 × 2 (recommended) or P100  
**Runtime**: ~50 epochs × 1,144 patients × avg 3 visits = ~3,400 samples → ~2–4h on T4

**Steps**:
1. Upload `data_inventory.csv`, `splits.json`, preprocessed data subset
2. Run `02_cnn_baseline.py` — train for 50 epochs
3. Save model weights + metrics CSV to output
4. Download results to local machine

#### Alternative: Google Colab Pro
If Kaggle quota runs out:
- Colab Pro = $10/month → A100 GPU (40GB) → nnU-Net inference 3× faster
- Same scripts work on Colab with minor path adjustments

---

## 🗂️ Final Folder Structure After Today

```
implementation/
├── notebooks/                        ← USED INSTEAD OF SCRIPTS (richer + visual)
│   ├── 01_data_audit_inventory.ipynb ← ✅ DONE — all 20 cells clean
│   ├── 02_preprocessing_pipeline.ipynb ← ✅ DONE — all 16 cells clean
│   └── 02_cnn_baseline.ipynb         ← ⏳ NEXT — write locally, run on Kaggle T4
│
├── Data/
│   ├── Yale-Brain-Mets-Longitudinal/ ← ✅ RAW — 43 GB, never touch
│   ├── processed/                    ← ⏳ FILLING — 1 test patient done
│   │   └── YG_01M98EKKAR50/         ← ✅ 4 NIfTIs, (160,230,184), 1mm³
│   └── segmentations/                ← ⏳ LATER — nnU-Net on Kaggle
│
└── outputs/
    ├── data_inventory.csv            ← ✅ 11,877 rows × 35 cols
    ├── metadata.json                 ← ✅ 1,430 patients
    ├── patient_timelines.json        ← ✅ all patients
    ├── acquisition_parameters.csv    ← ✅ 33,804 × 11 cols
    ├── splits.json                   ← ✅ LOCKED (train=1144/val=143/test=143)
    ├── AUDIT_REPORT.md               ← ✅ full data audit summary
    ├── OUTPUTS_GUIDE.md              ← ✅ schema for all output files
    ├── PREPROCESSING_EXPLAINED.md    ← ✅ full pipeline explanation + Kaggle guide
    └── eda/
        └── figures/
            ├── 01_visit_distribution.png    ← ✅
            ├── 02_year_distribution.png     ← ✅
            ├── 03_modality_heatmap.png      ← ✅
            ├── 04_followup_duration.png     ← ✅
            ├── 05_interval_boxplot.png      ← ✅
            ├── 06_patient_timeline.png      ← ✅
            ├── preproc_01_raw_data.png      ← ✅ (from preprocessing notebook)
            ├── preproc_02_registration.png  ← ✅
            ├── preproc_03_resampling.png    ← ✅
            ├── preproc_04_normalization.png ← ✅
            └── preproc_05_final_4ch.png     ← ✅
```

---

## 🚀 What Phase 1 → Phase 2 → Phase 3 Feeds

| Output | Used In | How |
|--------|---------|-----|
| `data_inventory.csv` | Phase 2 DataLoader | Knows which files exist and which are usable |
| `splits.json` | All phases | No data leakage — fixed train/val/test forever |
| `patient_timelines.json` | Phase 3 TaViT | Time-distance encoding: days between visits |
| `metadata.json` | Phase 4 LLM | Age, sex, treatment → LLM clinical narrative |
| `processed/` (skull-stripped) | Phase 2 CNN | Model input |
| `segmentations/` (nnU-Net masks) | Phase 2 + 3 | Pseudo-labels for CNN; volume extraction for Phase 3 |
| `registered/` (inter-visit) | Phase 3 + 5 | Swin UNETR extracts temporally consistent embeddings |

---

## ⚠️ Key Decisions (Locked Today)

### Decision 1: Minimum Modality for a Visit to Be "Usable"
→ **POST (T1c) + FLAIR** minimum = keeps 75.7% of visits  
→ All 4 = only 37.1% — too restrictive, drops too much data  
→ **For CNN baseline (Phase 2): use POST only** (always available) for simplicity. Extend to 4-channel later.

### Decision 2: Split Reference
→ First visit = baseline reference for longitudinal registration  
→ Clinically meaningful (all changes measured from pre-treatment baseline)

### Decision 3: Patients with 1 Visit
→ **131 patients** excluded from longitudinal split but included in Phase 2 CNN (single-timepoint task)

### Decision 4: Disk Space
→ Only 22GB free. Processed outputs for train split ≈ **97 GB** (9,463 visits × 2.1 avg mods × 4.9 MB).  
→ **Solution**: Free ~75 GB first (clear conda cache, move old files, external drive).  
→ Alternative: process POST-only first → ~47 GB → fits after freeing ~25 GB.  
→ ✅ Preprocessing pipeline `02_preprocessing_pipeline.ipynb` is **written, tested, and validated** on test patient.
→ ⏳ **Batch run is the next action** — see PREPROCESSING_EXPLAINED.md Part 9 for full timing/disk/Kaggle guide.

---

## 🔧 Conda Environment Command

```bash
conda activate datapre

# Missing packages to install (run once):
pip install monai torchvision timm

# Confirm everything:
python3 -c "import nibabel, pandas, numpy, scipy, matplotlib, seaborn, sklearn, itk, SimpleITK, torch, tqdm, openpyxl; print('ALL OK')"
```

---

## 📊 Progress Tracker

- [x] ✅ `conda activate datapre` + all packages confirmed working
- [x] ✅ `data_inventory.csv` — **11,877 rows × 35 columns**
- [x] ✅ `metadata.json` — **1,430 patients**, clinical fields linked
- [x] ✅ `usability_report.csv` — all 33,811 NIfTI headers checked
- [x] ✅ `splits.json` — **LOCKED** train=1,144 / val=143 / test=143
- [x] ✅ `patient_timelines.json` — all patients with time deltas
- [x] ✅ 6 EDA figures saved to `outputs/eda/figures/`
- [x] ✅ `AUDIT_REPORT.md` + `OUTPUTS_GUIDE.md` + `PREPROCESSING_EXPLAINED.md` written
- [x] ✅ `02_preprocessing_pipeline.ipynb` — written, tested on 1 patient, all 16 cells pass
- [ ] ⏳ **Free ~75 GB disk space** (conda clean + external drive)
- [ ] ⏳ **Run batch preprocessing** — set `DRY_RUN = False`, launch overnight (~39h)
- [ ] ⏳ **Upload to Kaggle** — metadata CSVs first, then processed NIfTIs via CLI
- [ ] ⏳ **CNN baseline on Kaggle T4** — `02_cnn_baseline.ipynb`, ~2h training
- [ ] ⏳ Kaggle notebook for nnU-Net segmentation (after CNN baseline confirmed)

---

## ⏱️ Estimated Times for Async Jobs

| Job | Machine | Time Estimate |
|-----|---------|--------------|
| BraTS Toolkit (11,877 visits) | Local CPU (8 threads) | **30–50 hours** (launch tonight) |
| nnU-Net inference (11,877 visits) | Kaggle T4 GPU | **~120 hours total** → 6 × 4h sessions over 3 days |
| itk-elastix registration (per-patient) | Local CPU | ~2 min/patient × 1,430 = ~48h (after preprocessing) |
| CNN baseline training (50 epochs) | Kaggle T4 | **~2–4 hours** |
| Swin UNETR feature extraction | Kaggle T4/P100 | **~6–8 hours** (Phase 3) |

