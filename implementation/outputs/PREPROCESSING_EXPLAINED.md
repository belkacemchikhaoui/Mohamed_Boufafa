# 🧠 Preprocessing Pipeline — Complete Explanation

> **Who is this for?** You looked at the images and asked: *"why are some blurry? why are coronal/sagittal views unstable? where is the 3D? can we use it?"*
> This document answers all of that, step by step.

---

## Part 1 — What IS the data, really?

### An MRI scan is a 3D volume, NOT a photo

An MRI scan is a **3-dimensional grid of numbers** (a numpy array with shape `[Z, Y, X]`).
Each number is called a **voxel** (volume pixel).

```
Raw scan of patient YG_01M98EKKAR50, visit 2016-11-13:

  PRE  → shape (32, 320, 256)  → 32 slices × 320 rows × 256 columns
  POST → shape (32, 320, 272)  → same Z, different X (different scanner FOV)
  T2   → shape (32, 320, 320)
  FLAIR→ shape (32, 320, 320)
```

When you see an image in the notebook, it is **one 2D slice** cut from that 3D volume, like slicing bread.

There are three ways to slice:
- **Axial (z)** — cut horizontally → top-down view of the brain ← the *clear* one
- **Coronal (y)** — cut front-to-back → face-on view
- **Sagittal (x)** — cut left-to-right → side view

---

## Part 2 — Why do the coronal and sagittal views look BLURRY / BLOCKY?

### The root cause: anisotropic voxels

This is the **single most important concept** to understand about medical imaging.

The scanner acquires slices **one at a time**. The **in-plane** resolution (within each axial slice) is high. The **through-plane** spacing between slices is much larger.

```
Raw voxel size for this dataset:
  In-plane (X, Y) : 0.72 mm × 0.72 mm  ← very fine
  Through-plane (Z):  5.0 mm            ← 7× coarser

This is ANISOTROPIC = unequal spacing in different directions.
```

When you view the **axial** slice (cut along Z), you look at the fine in-plane resolution → **sharp**.

When you view the **coronal** or **sagittal** slice, you are cutting **across** the thick 5mm gaps → **staircase effect / blur**.

```
Visual analogy:

  Axial cut (XY plane):            Sagittal cut (XZ plane):
  ████████████████████             █
  ████████████████████             █
  ████████████████████             █
  (fine 0.72mm steps)              (coarse 5mm steps → looks jagged)
```

**This is NOT a quality problem** — it is the physical reality of clinical MRI acquisition.
The scanner took ~5 minutes per patient; acquiring 160 thin slices would take 25+ minutes.

---

## Part 3 — Where is the "3D"? Can we use it?

### Yes — the 3D volume is exactly what we work with. It IS a 3D array.

When we load a NIfTI file:
```python
img = sitk.ReadImage("PRE.nii.gz")
arr = sitk.GetArrayFromImage(img)
# arr.shape = (32, 320, 256)  ← this IS the 3D volume
```

It is already 3D. The 2D images you see are just visualizations of one slice at a time.

### Can a 3D CNN use it? Yes, but with a caveat

**Before resampling** (raw data):
```
Shape: (32, 320, 256) — voxels are NOT equal size
Voxel: 0.72 × 0.72 × 5.0 mm

Problem: the Z-axis is 7× coarser than X,Y
→ A 3D CNN kernel of size 3×3×3 would "see"
  0.72mm × 0.72mm × 5.0mm  ← NOT a cube, a flat pancake
→ The model would learn wrong geometric relationships
→ A tumor 3mm tall (4 thin slices) would appear as zero slices
```

**After resampling to 1mm isotropic** (what our pipeline produces):
```
Shape: (160, 230, 184) — every voxel is 1mm × 1mm × 1mm
Voxel: 1.0 × 1.0 × 1.0 mm  ← perfect cube

Now a 3D CNN kernel of 3×3×3 sees a true 3mm × 3mm × 3mm region
→ All spatial relationships are geometrically correct
→ Ready for 3D ViT / 3D CNN
```

---

## Part 4 — The Full Pipeline, Step by Step

### Step 0 — Already Done by Yale: Skull Stripping (HD-BET)

**What it does:** Remove everything outside the brain (skull, skin, eyes, neck).

**Why:** The model should not learn "skull shape = disease progression". Also reduces volume size.

**How we know it's done:** 91.3% of voxels are exactly 0.0 — the background is zeroed out.

**What you see in the images:** Brain floating in black space. That black = zeroed skull region.

```
Before skull strip: brain + skull + neck
After skull strip:  brain only, rest = 0
```

> **Status:** ✅ Already in the Yale dataset. We do NOT redo this.

---

### Step 1 — Co-Registration (rigid alignment)

**Problem it solves:**

Each modality (PRE, POST, T2, FLAIR) is acquired in a separate scan. Between scans, the patient moves their head (even 1–2mm of movement changes which brain structure each voxel represents).

```
Before registration:
  PRE  shape: (256, 320, 32)  ← 256 columns
  POST shape: (272, 320, 32)  ← 272 columns! Different FOV + slight rotation
  T2   shape: (320, 320, 32)
  FLAIR shape: (320, 320, 32)

Pixel [100, 150, 16] in PRE ≠ same anatomical location as [100, 150, 16] in POST
→ You cannot stack them as a 4-channel volume — they don't align
```

**What registration does:** Find a rigid transform (rotation + translation, no stretching) that moves POST, T2, FLAIR to match PRE's coordinate frame.

```
After registration:
  PRE   shape: (256, 320, 32)  ← unchanged (it's the "fixed" reference)
  POST  shape: (256, 320, 32)  ← resampled into PRE's grid
  T2    shape: (256, 320, 32)
  FLAIR shape: (256, 320, 32)

Now pixel [100, 150, 16] in ALL 4 modalities = same brain location ✅
```

**Tool used:** `itk-elastix`, rigid parameter map, Mattes Mutual Information metric (works across modalities — PRE and T2 look completely different but both contain information about anatomy).

**What the before/after visualization shows:**
- Left: PRE (fixed reference)
- Middle: POST/T2/FLAIR *before* registration (may be rotated, shifted, different size)
- Right: POST/T2/FLAIR *after* registration (matches PRE's field of view)
- Heat map: bright = large remaining difference between PRE and registered modality (expected — different tissue contrasts)

**Time:** POST took 19s (more iteration needed due to large shape difference). T2/FLAIR took 3s each.

---

### Step 2 — Resampling to 1mm Isotropic

**Problem it solves:** The anisotropic voxel problem from Part 2.

```
Input:   (32, 320, 256) @ (0.72, 0.72, 5.0) mm  ← pancake voxels
Output:  (160, 230, 184) @ (1.0, 1.0, 1.0) mm   ← cubic voxels
```

**What happens geometrically:**
- Z dimension: 32 slices × 5.0mm = 160mm physical extent → at 1mm = 160 slices
- X dimension: 256 × 0.72mm = 184mm physical extent → at 1mm = 184 voxels
- Y dimension: 320 × 0.72mm = 230mm physical extent → at 1mm = 230 voxels

The total physical size of the brain (in mm) is **preserved**. We just sample it more uniformly.

**Interpolation used:** Linear (B-spline order 1) for MRI intensities. This fills in the "gaps" between thick slices using smooth interpolation — that's why the AFTER sagittal view looks *smoother* but *still a bit soft* (it's mathematically interpolated, not real acquired data).

**The blurriness in coronal/sagittal AFTER resampling is expected and correct** — it's the best possible reconstruction of what exists between the 5mm slices.

---

### Step 3 — Z-Score Normalization (Intensity Normalization)

**Problem it solves:** MRI intensities are **not standardized** across scanners, patients, or sessions.

```
Raw intensities from the notebook:
  PRE   → min=0, max=537,  mean=237
  POST  → min=0, max=1054, mean=246  ← 2× higher!
  T2    → min=0, max=1892, mean=503  ← 4× higher!
  FLAIR → min=0, max=561,  mean=220

A neural network cannot learn if the same tissue type has value 237 in
one scan and value 503 in another scan.
```

**What z-score does:**

$$\text{normalized} = \frac{x - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$

Where $\mu_{\text{brain}}$ and $\sigma_{\text{brain}}$ are computed **only from brain voxels** (non-zero mask), ignoring the zeroed background.

```
After normalization:
  PRE   → min=-0.46, max=3.19,  mean≈1.9, std≈0.84
  POST  → min=-0.46, max=3.13,  mean≈1.8, std≈0.96
  T2    → min=-0.44, max=5.37,  mean≈1.6, std≈1.20
  FLAIR → min=-0.44, max=3.95,  mean≈1.7, std≈1.06
```

All modalities now live in a **comparable range**. The neural network can now compare features across modalities and patients.

**Why mean≈1.8 and not ≈0:** The zero background voxels pull the mean up slightly — this is expected and fine.

---

## Part 5 — What Does the Final Output Look Like?

### File structure on disk

```
Data/processed/
└── YG_01M98EKKAR50/
    └── 2016-11-13/
        ├── PRE_processed.nii.gz    (4.7 MB)  shape=(160, 230, 184)  voxel=1mm³
        ├── POST_processed.nii.gz   (5.0 MB)
        ├── T2_processed.nii.gz     (5.3 MB)
        └── FLAIR_processed.nii.gz  (4.9 MB)
```

### What the model receives per patient-visit

```python
# For one visit, the model gets a 4-channel 3D tensor:
volume = np.stack([PRE, POST, T2, FLAIR], axis=0)
# Shape: (4, 160, 230, 184)  = [C, Z, Y, X]
#         ^--- 4 MRI modalities (like RGB channels, but 4D)
```

This is directly usable by:
- **3D ViT (Vision Transformer)**: splits into 3D patches, e.g. 16×16×16
- **3D CNN / U-Net**: applies 3D convolutions across all axes
- **2.5D CNN**: extracts 2D slices from each axis and processes them

---

## Part 6 — Summary: What Each Image in the Notebook Tells You

| Image | What it shows | Key thing to notice |
|---|---|---|
| **Raw data (Cell 3)** | All 4 modalities raw, before any processing | Different shapes (256 vs 272 vs 320), blurry coronal/sagittal |
| **Registration before/after (Cell 5)** | PRE vs POST/T2/FLAIR before + after rigid alignment | After reg: same field of view, brain in same position |
| **Registration heat map** | Pixel-wise absolute difference after registration | Bright edges = tissue contrast difference (expected), not misalignment |
| **Resampling before/after (Cell 7)** | Shape and view before/after 1mm resampling | Sagittal BEFORE = jagged stairs. AFTER = smoother (interpolated) |
| **Normalization before/after (Cell 9)** | Raw intensities vs z-score normalized, with histograms | BEFORE: wide spread different per modality. AFTER: all comparable range |
| **Final overview (Cell 10)** | All 4 modalities together, same slice index = same anatomy | All modalities now share same coordinate space ✅ |

---

## Part 7 — Common Questions

**Q: The sagittal view still looks blurry after resampling. Did it fail?**

No. There are only 32 original slices over 160mm. The new 160 slices are **interpolated** from 32 real measurements. The intermediate voxels are estimated, not acquired. This is the physical limit of the data — we cannot recover information that was never measured.

**Q: Why not acquire isotropic data in the first place?**

Clinical brain MRI protocols optimize for speed: 32 thick slices take ~5 min. Acquiring 160 thin slices (1mm each) would take ~25 min per modality. With 4 modalities that's 100 min — not feasible for patients with metastases.

**Q: Is the 3D volume usable for deep learning?**

Yes — after resampling to 1mm isotropic, the volume is fully 3D and geometrically correct. The final shape `(4, 160, 230, 184)` is exactly what a 3D model receives.

**Q: Why use 4 modalities (channels) instead of just one?**

Each modality highlights different tissue:
- **PRE (T1 pre-contrast)**: gray/white matter anatomy
- **POST (T1 post-contrast)**: blood-brain barrier breakdown → tumors enhance (brighten)
- **T2**: edema, fluid → perilesional swelling
- **FLAIR**: white matter lesions, suppresses normal fluid

Together they give the model complementary information — like having X-ray + CT + ultrasound simultaneously.

**Q: What is the "mean absolute diff = 40.94" in the registration visualization?**

It is the average intensity difference between PRE and registered POST, measured **only inside the brain mask**. It is high (40.94) because PRE and POST look intrinsically different (one is pre-contrast, one post-contrast, so tumor appears bright in POST but dark in PRE). This is **not** registration error — it is the expected biological difference between modalities.

---

## Part 8 — Where We Are in the Project

```
Phase 1: Data Understanding & Preprocessing
  ✅  Block 1-5:  Data audit → 01_data_audit_inventory.ipynb (DONE)
                  Outputs: data_inventory.csv, metadata.json, splits.json,
                           patient_timelines.json, AUDIT_REPORT.md, 6 EDA figures
  ✅  Block 6:    Preprocessing pipeline → 02_preprocessing_pipeline.ipynb (DONE)
                  - Test patient YG_01M98EKKAR50 fully processed & validated
                  - Output: (4, 160, 230, 184) float32 per visit, 1mm isotropic
                  - Batch: code ready, blocked by disk space (22GB free)

Phase 2: CNN Baseline (NEXT)
  ⏳  Block 7:    02_cnn_baseline.ipynb
                  - Run on Kaggle T4 GPU (16GB VRAM)
                  - Upload processed data + scripts via Kaggle Dataset
                  - Target: treatment response prediction at next visit
```

---

## Part 9 — Batch Preprocessing: Timing, Disk, and Where to Run

### How long will it take?

```
Numbers from our test patient:
  Registration (3 modalities):  ~26 seconds  (POST=19s, T2+FLAIR=3s each)
  Resampling (4 modalities):    ~0.5 seconds
  Normalization:                ~0.6 seconds
  Save (4 NIfTIs):              ~0.7 seconds
  Total per visit (4 mods):     ~28 seconds on 7 CPU cores
```

The dataset has **9,463 train visits** to process (batch dry-run confirmed this number).

```
Estimated total time on LOCAL CPU (7 cores):
  9,463 visits × 28s ≈ 264,964s ÷ 3600 ≈ 74 hours

  BUT: most visits have only 1–2 modalities (not all 4).
  Average modalities per visit ≈ 2.1 (from data audit)
  → Effective time ≈ 74h × (2.1/4) ≈ ~39 hours

  In practice: launch before bed → ready in ~2 nights running continuously.
```

> **Parallel opportunity**: `DRY_RUN = False` in the batch cell uses sequential processing.
> We can improve this later with `multiprocessing.Pool` (4 patients in parallel)
> → would cut time to ~10 hours. But for now, sequential is fine and safe.

---

### Can we do it with only 100 GB instead of 180 GB?

**Yes. Three strategies, pick one or combine:**

**Strategy A — Train split only (recommended first step)**
```
The batch cell already defaults to PROCESS_SPLIT = 'train'
Train split = 1,144 patients × avg 8.3 visits = 9,463 visits

Estimated output size:
  9,463 visits × avg 2.1 mods × 4.9 MB per file ≈ 97 GB

  ← This is already under 100 GB!
  But we only have 22 GB free right now. We need to free 75 GB first.
```

**Strategy B — Process only FULL visits (all 4 modalities)**
```
FULL visits in train = ~1,144 × 37.1% × 8.3 visits ≈ 3,510 visits
Output size: 3,510 × 4 mods × 4.9 MB ≈ 69 GB

Trade-off: We lose 63% of visits (only keep complete 4-modality ones).
Good enough for an initial CNN baseline.
```

**Strategy C — Process only POST modality (smallest possible)**
```
POST is the most clinically important (shows tumor enhancement).
Process POST only for all train visits.

Output size: 9,463 visits × 1 file × 5 MB ≈ 47 GB ← fits in 22 GB? NO
Still need to free ~25 GB. But after freeing:
  → 47 GB output is very manageable
  → CNN can train on single-channel POST input
```

**Recommended plan given 22 GB free:**
```
1. Free up ~80 GB (see below how)
2. Run batch with PROCESS_SPLIT='train' and all modalities → ~97 GB
   OR run with POST-only → 47 GB (for a fast first CNN run)
```

**How to free disk space:**
```bash
# Check what's using space:
du -sh /home/moamed/* | sort -rh | head -20

# Common space savers:
# - Clear conda package cache:  conda clean --all  (~2-5 GB)
# - Clear pip cache:            pip cache purge    (~0.5 GB)
# - Remove old conda envs you don't use
# - Move raw Yale data to external drive (43 GB!) and symlink it
#   ln -s /external_drive/Yale-Brain-Mets-Longitudinal ./Data/Yale-Brain-Mets-Longitudinal
```

---

### Should we do batch preprocessing locally or on Kaggle?

**Short answer: Do it LOCALLY. Here is why:**

| Factor | Local CPU | Kaggle |
|--------|-----------|--------|
| **Free disk** | 22 GB (need to free more) | 20 GB output limit per session |
| **Data access** | Data already here (43 GB) | Need to UPLOAD 43 GB first |
| **Upload time** | — | 43 GB @ ~10 Mbps = **10+ hours** |
| **GPU help?** | No — preprocessing is CPU-only | No — same CPU speed |
| **Session limit** | Unlimited | 12h per session, then restart |
| **Convenience** | Run overnight, unattended | Must re-upload data each new session |
| **Verdict** | ✅ **DO THIS LOCALLY** | ❌ Not worth it for preprocessing |

> **Key insight**: Registration and resampling (itk-elastix, SimpleITK) are **pure CPU operations**.
> A GPU gives ZERO speedup for preprocessing. The bottleneck is I/O and CPU math, not GPU compute.

**Kaggle IS the right place for:**
- CNN training (needs GPU)
- nnU-Net inference (needs GPU)
- Feature extraction (needs GPU)

---

### Would a GPU make preprocessing faster?

**No — and here is the detailed reason:**

```
Preprocessing steps and their compute type:

  Registration (itk-elastix):
    → Iterative optimizer running on CPU
    → Uses 7 CPU threads (N_THREADS = 7 in our config)
    → GPU version of elastix does NOT exist in itk-elastix 0.24
    → Cannot be accelerated with GPU

  Resampling (SimpleITK):
    → B-spline interpolation — CPU only in SimpleITK 2.5.3
    → GPU version requires special CUDA build of ITK
    → 0.1s per volume — already negligible

  Normalization (numpy):
    → Array arithmetic — already near-instant (0.6s for 4 volumes)
    → Moving to GPU would add overhead (copy to VRAM) with no benefit

  File I/O (NIfTI read/write):
    → Disk speed limited — no GPU helps here
```

**GPU becomes useful at Phase 2 (CNN training):**
```
CNN training comparison (estimated for our dataset):

  Local CPU only (i7-1065G7):
    50 epochs × 9,463 samples / batch_size=2 ≈ 236,575 forward passes
    ~0.5s per pass on CPU → 236,575 × 0.5s ÷ 3600 ≈ 33 hours

  Kaggle T4 GPU (16 GB):
    ~0.03s per pass on T4 → same calc → ~2 hours

  Kaggle T4 × 2 GPUs:
    → ~1 hour for 50 epochs

  Speedup: GPU is ~15–20× faster for CNN training.
  → ALWAYS use Kaggle T4 for training. Never train on local CPU.
```

---

## Part 10 — Next Step: CNN on Kaggle — What to Upload and How

### Step 1 — What you need to upload to Kaggle

You do NOT need to upload all 180 GB of processed data to train the first CNN baseline.
You only need:

```
Minimum upload for CNN baseline:

  implementation/outputs/data_inventory.csv     (~5 MB)
  implementation/outputs/splits.json            (~200 KB)
  implementation/outputs/metadata.json          (~800 KB)
  implementation/Data/processed/                (~97 GB for train, OR start with subset)

For a FAST first run (recommended before uploading everything):
  → Upload processed data for 100 patients only (~8 GB)
  → Train CNN on those 100 patients first to confirm everything works
  → Then upload full train split
```

### Step 2 — How to upload to Kaggle (step by step)

**Option A — Kaggle CLI (recommended, handles large uploads)**
```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Get your API key:
#    Go to kaggle.com → Your profile → Account → API → Create New Token
#    This downloads kaggle.json — put it at ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
# copy kaggle.json there, then:
chmod 600 ~/.kaggle/kaggle.json

# 3. Create a new Kaggle Dataset for your processed data
kaggle datasets init -p /path/to/upload/folder
# This creates a dataset-metadata.json — edit it with your dataset name

# 4. Upload (can resume if interrupted)
kaggle datasets create -p /path/to/upload/folder --dir-mode tar

# 5. To update later (add more processed files):
kaggle datasets version -p /path/to/upload/folder -m "Added 500 more patients"
```

**Option B — Direct Kaggle website upload (for small files only, <10 GB)**
```
kaggle.com → Datasets → New Dataset → Upload files via browser
Limited to ~10 GB per upload — OK for the CSV/JSON metadata files
NOT OK for the full processed NIfTI data
```

**Option C — Google Drive + gdown in Kaggle notebook (workaround for large data)**
```python
# Inside your Kaggle notebook:
!pip install gdown
import gdown
# Upload your processed data to Google Drive, get shareable link
gdown.download_folder('https://drive.google.com/drive/folders/YOUR_FOLDER_ID',
                      output='/kaggle/input/processed/', quiet=False)
```

### Step 3 — Kaggle notebook structure for CNN training

```python
# At top of Kaggle notebook:
import os

# Your uploaded dataset will be at:
DATA_ROOT = '/kaggle/input/yale-brain-mets-processed/'   # your dataset name
INVENTORY = f'{DATA_ROOT}/data_inventory.csv'
SPLITS    = f'{DATA_ROOT}/splits.json'
PROCESSED = f'{DATA_ROOT}/processed/'

# Install missing packages (Kaggle has torch pre-installed):
!pip install monai itk-elastix SimpleITK -q

# Then run your cnn_baseline script:
# (copy the script content directly into notebook cells, or upload as a dataset file)
```

### Step 4 — Recommended order of operations

```
TODAY / THIS WEEK:
  1. Free up disk space locally (target: 80 GB free)
  2. Run batch preprocessing locally overnight (DRY_RUN = False)
     → Expected time: ~39 hours for train split
     → Set PROCESS_SPLIT = 'train' in the batch cell

AFTER PREPROCESSING DONE:
  3. Upload processed data to Kaggle via CLI
     → Start with 100-patient subset for testing (~8 GB)
     → Then upload full train split (~97 GB) — takes several hours

  4. Create CNN baseline notebook on Kaggle
     → Use T4 GPU × 1 (free tier: 30h GPU per week)
     → First run: train for 10 epochs to confirm pipeline works
     → Full run: 50 epochs ≈ 2 hours

  5. Download model weights + metrics to local machine
```

---

*Generated: March 2026 | Patient cohort: Yale Brain Metastases Longitudinal | Pipeline: registration → resample → normalize*
