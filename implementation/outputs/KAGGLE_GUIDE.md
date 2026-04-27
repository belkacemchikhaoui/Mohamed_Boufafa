# Kaggle Upload & CNN Run — Complete Guide

## Quick Status Check (run this first)

```bash
cd /home/moamed/canada_me/explainable_diseas
python3 implementation/outputs/kaggle_upload.py --check
```

---

## What We Have

| Item | Local Path | Kaggle ID | Status |
|------|-----------|-----------|--------|
| Manifest CSV | `implementation/outputs/processed_manifest.csv` | `mohamedmohamed23/yale-processed-manifest` | ✅ Uploaded (464 KB) |
| NIfTI data | `/media/moamed/Data/yale-processed/` | `mohamedmohamed23/yale-processed-nifti` | ❌ Not yet on Kaggle |
| CNN Notebook | `implementation/notebooks/04_cnn_baseline_kaggle.ipynb` | — | ✅ Ready to upload |

---

## Upload Commands

### Re-upload Manifest (if needed)
```bash
cd /home/moamed/canada_me/explainable_diseas
python3 implementation/outputs/kaggle_upload.py --manifest-only
```

### Upload NIfTI Data (main task, ~10 GB)
```bash
cd /home/moamed/canada_me/explainable_diseas
python3 implementation/outputs/kaggle_upload.py --nifti-only
```

> **Expected time**: 20–90 min depending on connection speed  
> **Progress file**: `/tmp/kaggle_nifti_progress.json`  
> **Log file**: `/tmp/kaggle_upload_python.log`

### Upload Both
```bash
python3 implementation/outputs/kaggle_upload.py
```

---

## How to Check Upload Progress

### While upload is running (in another terminal):
```bash
# Watch the log file live
tail -f /tmp/kaggle_upload_python.log

# Check progress JSON
cat /tmp/kaggle_nifti_progress.json | python3 -m json.tool

# Count tracked folders
python3 -c "
import json
p = json.load(open('/tmp/kaggle_nifti_progress.json'))
done = len(p['uploaded_folders'])
print(f'{done}/232 folders tracked ({done*100//232}%)')
"
```

### Check if upload actually landed on Kaggle:
```bash
export KAGGLE_API_TOKEN="KGAT_409fa7f5d6a6c6be92cc91e56d173d21"
/home/moamed/canada_me/explainable_diseas/.venv/bin/kaggle datasets files mohamedmohamed23/yale-processed-manifest
/home/moamed/canada_me/explainable_diseas/.venv/bin/kaggle datasets files mohamedmohamed23/yale-processed-nifti
```

---

## If Upload Stalls / Gets Killed

The script is **resumable** — just run it again:

```bash
python3 implementation/outputs/kaggle_upload.py --nifti-only
```

It reads `/tmp/kaggle_nifti_progress.json` to know what's already done.

### To start completely fresh:
```bash
rm /tmp/kaggle_nifti_progress.json
python3 implementation/outputs/kaggle_upload.py --nifti-only
```

---

## Running the CNN on Kaggle GPU

Once `yale-processed-nifti` exists on Kaggle:

### Step 1 — Upload the notebook
Go to **kaggle.com → Your Work → Notebooks → New Notebook → File → Import**  
Upload: `implementation/notebooks/04_cnn_baseline_kaggle.ipynb`

### Step 2 — Attach datasets
In the notebook editor, click **+ Add Data** (right sidebar):
1. Search for `yale-processed-nifti` → add it
2. Search for `yale-processed-manifest` → add it

Or paste these IDs directly:
- `mohamedmohamed23/yale-processed-nifti`
- `mohamedmohamed23/yale-processed-manifest`

### Step 3 — Set GPU accelerator
**Settings** (right sidebar) → **Accelerator** → **GPU T4 x2** or **GPU P100**

### Step 4 — Run
**Run All** → Expected runtime: ~4–6 hours on P100

### Step 5 — Outputs
After it finishes, outputs will be in `/kaggle/working/`:
- `cnn_baseline_results.zip` — model + plots + metrics

---

## Troubleshooting

### "401 Unauthorized" from kaggle CLI
The token format changed. Use `KAGGLE_API_TOKEN` env var, NOT the old `~/.kaggle/kaggle.json`:
```bash
export KAGGLE_API_TOKEN="KGAT_409fa7f5d6a6c6be92cc91e56d173d21"
```

### "Dataset not found" after uploading
Kaggle takes 5–10 min to index new datasets. Also, `kaggle datasets list --mine` shows cached results — use `datasets files <id>` to confirm the file is there.

### Notebook can't find the CSV
Make sure the path in the notebook is:
```python
MANIFEST = Path("/kaggle/input/yale-processed-manifest/processed_manifest.csv")
```

### NIfTI files not found on Kaggle notebook
The dataset slug must match exactly. The notebook looks for:
```
/kaggle/input/yale-processed-nifti/
```
Dataset ID: `mohamedmohamed23/yale-processed-nifti`

---

## API Token

```
KGAT_409fa7f5d6a6c6be92cc91e56d173d21
```
- Format: **new-style** Kaggle API token (starts with `KGAT_`)
- Set as environment variable `KAGGLE_API_TOKEN` — do NOT use the old `~/.kaggle/kaggle.json` `key:` field format with this

---

## File Sizes Reference

| File | Size |
|------|------|
| `processed_manifest.csv` | 454 KB (889 visits) |
| NIfTI data (uncompressed) | ~10 GB (232 patients) |
| NIfTI data (tar compressed) | ~7–8 GB |

---

## CNN Notebook Structure

`04_cnn_baseline_kaggle.ipynb` has 12 code cells:
1. Install packages
2. Imports + config (batch size: 4 on P100, 2 on T4)
3. Path fixer (local ↔ Kaggle path remapping)
4. Pseudo-labels (POST signal change >10% = progressive)
5. Dataset class (4 modalities, resize, augment)
6. Model (`BrainMetsCNN` — 3D ConvBlocks + GAP)
7. Optimizer setup (AdamW + CosineAnnealingLR)
8. Training loop (mixed precision, early stopping)
9. Training curves plot
10. Test evaluation (classification report, ROC-AUC)
11. Grad-CAM 3D visualization
12. Package outputs to zip
