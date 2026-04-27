# Kaggle Upload & GPU Training Guide
## Yale Brain Mets — Phase 2 CNN Baseline

> **GPU choice:** P100 ≈ 2× faster than T4 for this workload (16 GB vs 16 GB VRAM, better FP32 throughput for 3D convolutions). Always pick **P100** if available. T4 × 2 is a fallback.

---

## Part A — Upload Processed Data to Kaggle

### A1. What to upload

You have 2 things to upload as separate Kaggle Datasets:

| Dataset name | What | Size | Where on disk |
|---|---|---|---|
| `yale-processed` | All processed `.nii.gz` folders | ~19 GB (grows as batch runs) | `/media/moamed/Data/yale-processed/` |
| `yale-processed-manifest` | One CSV file | < 1 MB | `implementation/outputs/processed_manifest.csv` |

---

### A2. Install Kaggle CLI

```bash
pip install kaggle
```

Get your API key:
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
2. Download `kaggle.json` → place it at `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

---

### A3. Upload the manifest CSV (small, fast)

```bash
# Create dataset folder
mkdir -p /tmp/yale-manifest
cp /home/moamed/canada_me/explainable_diseas/implementation/outputs/processed_manifest.csv \
   /tmp/yale-manifest/

# Create metadata
cat > /tmp/yale-manifest/dataset-metadata.json << 'EOF'
{
  "title": "Yale Brain Mets Processed Manifest",
  "id": "YOUR_KAGGLE_USERNAME/yale-processed-manifest",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload
kaggle datasets create -p /tmp/yale-manifest --dir-mode zip
```

---

### A4. Upload the processed NIfTI data (large — use zip or folder upload)

**Option 1 — Direct folder upload (recommended, resumable):**
```bash
# Create metadata inside the processed folder
cat > /media/moamed/Data/yale-processed/dataset-metadata.json << 'EOF'
{
  "title": "Yale Brain Mets Processed NIfTI",
  "id": "YOUR_KAGGLE_USERNAME/yale-processed",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload (--dir-mode tar = preserves folder structure, faster than zip)
kaggle datasets create -p /media/moamed/Data/yale-processed --dir-mode tar
```

**Option 2 — Zip and upload (if folder upload is slow):**
```bash
# Zip the processed data (takes ~10 min, produces ~12 GB zip)
cd /media/moamed/Data
zip -r yale-processed.zip yale-processed/

# Upload
kaggle datasets create -p /media/moamed/Data --dir-mode zip
```

> **Tip:** While 19 GB uploads, continue running the preprocessing batch locally. You can update the Kaggle dataset later with `kaggle datasets version -p <path> -m "Added more visits"`.

---

### A5. Update dataset when more visits are processed

```bash
# After another batch run — re-export manifest
# (run the last cell of 03_processed_eda.ipynb first)

# Update manifest dataset
kaggle datasets version -p /tmp/yale-manifest -m "Added more processed visits"

# Update NIfTI dataset  
kaggle datasets version -p /media/moamed/Data/yale-processed -m "Added batch 2"
```

---

## Part B — Run the CNN Notebook on Kaggle

### B1. Create the Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File → Import Notebook** → upload `04_cnn_baseline_kaggle.ipynb`

---

### B2. Choose your GPU — P100 vs T4

| | **P100** ✅ Recommended | **T4 × 2** |
|---|---|---|
| VRAM | 16 GB | 16 GB × 2 = 32 GB |
| FP32 TFLOPS | 9.3 | 8.1 (×2 = 16.2 but needs DDP code) |
| Speed for 3D CNN (single GPU) | **~2× faster** | Slower per GPU |
| Weekly quota | 30h (same) | 30h (same) |
| Best for | Single-GPU 3D CNN ← **use this** | Multi-GPU (requires DataParallel) |

**To select P100:**  
Settings (right panel) → Accelerator → **GPU P100** → Save

**To use T4 × 2** (only if you want to use both GPUs):  
Settings → Accelerator → **2× T4 GPU** → Save  
Then in Cell 2 of the notebook, add after `model = BrainMetsCNN(...)`:
```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(DEVICE)
```

---

### B3. Attach the datasets

In the right panel → **Data** → **Add Data**:
1. Search `yale-processed-manifest` → Add
2. Search `yale-processed` → Add

They will be mounted at:
- `/kaggle/input/yale-processed-manifest/processed_manifest.csv`
- `/kaggle/input/yale-processed/` (folder tree)

---

### B4. Fix paths in the notebook

In **Cell 2** of `04_cnn_baseline_kaggle.ipynb`, change the paths block to:

```python
# ── Kaggle paths ──────────────────────────────────────────────────────────────
MANIFEST = Path("/kaggle/input/yale-processed-manifest/processed_manifest.csv")
CKPT_DIR = Path("/kaggle/working/checkpoints")
```

Then run **Cell 12** (Kaggle path fixer) — it auto-remaps  
`/media/moamed/Data/yale-processed/` → `/kaggle/input/yale-processed/`  
in the manifest paths.

---

### B5. Recommended settings for Kaggle GPU run

In **Cell 2**, tune `CFG` for Kaggle:

```python
CFG = {
    "target_shape"   : (96, 96, 64),   # P100: fine at full res
    # "target_shape" : (64, 64, 48),   # T4: use smaller if OOM
    "batch_size"     : 6,              # P100 16 GB can handle 6
    # "batch_size"   : 4,              # T4: use 4
    "epochs"         : 50,
    "n_workers"      : 4,              # Kaggle has 4 CPU cores
    "mixed_precision": True,           # Always True on GPU
    ...
}
```

---

### B6. Expected training time on Kaggle

With 889 processed visits (760 train / 51 val / 78 test):

| Hardware | Batch size | Target shape | Time / epoch | 50 epochs |
|----------|-----------|--------------|--------------|-----------|
| **P100** | 6 | 96×96×64 | ~4–6 min | **~4–5 h** |
| T4 (×1) | 4 | 96×96×64 | ~7–10 min | ~7–8 h |
| T4 (×1) | 4 | 64×64×48 | ~3–4 min | ~3 h |
| CPU only | — | — | ~60 min | — |

> Kaggle session limit: **12 hours** (P100/T4). 50 epochs fits comfortably.  
> With early stopping (patience=8), training typically stops at epoch 20–30.

---

### B7. Run the notebook

```
Cell 1  → Install missing packages (nibabel, pandas, sklearn, scipy, tqdm)
Cell 2  → Imports + config
Cell 3  → Pseudo-label generation (~30s)
Cell 4  → Dataset + DataLoaders
Cell 5  → Build BrainMetsCNN
Cell 6  → Optimiser + loss
Cell 7  → TRAIN ← main cell (4–5h on P100)
Cell 8  → Training curves
Cell 9  → Test evaluation (AUC, confusion matrix, ROC)
Cell 10 → Grad-CAM visualisations
Cell 11 → Package outputs (best_model.pt + figures)
Cell 12 → Path fixer (Kaggle only)
```

**Run All:** Kernel menu → **Run All**

---

### B8. Save outputs from Kaggle

After training:

1. The `best_model.pt` and figures are in `/kaggle/working/checkpoints/`
2. Download via: **Output** tab (right panel) → **Download**
3. Or add to a new Kaggle dataset:
```python
# Add this at the end of Cell 11 in the Kaggle run
import subprocess
subprocess.run(["kaggle", "datasets", "create", "-p", "/kaggle/working/checkpoints",
                "--dir-mode", "zip"])
```

---

## Part C — Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `target_shape` to `(64,64,48)` and `batch_size` to 2 |
| `FileNotFoundError` on `.nii.gz` | Run Cell 12 (path fixer) before Cell 3 |
| `ModuleNotFoundError: nibabel` | Cell 1 auto-installs — re-run Cell 1 |
| Dataset not found on Kaggle | Check dataset is public or attached to notebook in Data panel |
| Upload stalls at 19 GB | Use `--dir-mode tar` instead of `zip`, or split into batches |
| P100 not available | It cycles: check back in a few hours, or use T4 with smaller batch |

---

## Part D — After Training: What to Report

Your **Phase 1 Intermediate Deliverable** is complete when you have:

- [x] `EDA_REPORT_PHASE1.md` — this file's companion
- [x] `processed_manifest.csv` — 889 complete visits, train/val/test annotated
- [x] `02_preprocessing_pipeline.ipynb` — reproducible pipeline
- [x] `03_processed_eda.ipynb` — executed with all outputs
- [ ] `04_cnn_baseline_kaggle.ipynb` — run on Kaggle, download `best_model.pt`
- [ ] Test AUC reported (target > 0.65 for pseudo-labelled baseline)
