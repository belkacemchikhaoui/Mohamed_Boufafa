#!/usr/bin/env python3
"""
Upload MetSeg (DynUNet/CNN) fold outputs to Kaggle under zinou123viva account.
Run this after each Kaggle session to persist checkpoints + embeddings.

Usage:
    python3 scripts/upload_cnn_fold_outputs_zinou123viva.py

Expected files in Phase2/cnn_fold_outputs/ (downloaded from Kaggle output):
    checkpoints/metseg_fold0_best.pth
    checkpoints/metseg_fold0_latest.pth
    embeddings/cnn_metseg_embeddings_fold0.npz
    figures/metseg_fold0_training_curves.png
    metrics/metseg_fold0_metrics.json

Dataset: zinou123viva/metseg-fold-outputs
"""

import os, json, subprocess, sys, tempfile, shutil
from pathlib import Path

# ── Account config ────────────────────────────────────────────────────
ACCOUNT = {
    "username": "zinou123viva",
    "key":      "KGAT_735ba951854a6e5a95241e016225e8cb",
}

DATASET_SLUG  = "zinou123viva/metseg-fold-outputs"
DATASET_TITLE = "MetSeg CNN Fold Outputs (Phase 2 — zinou123viva)"
FILE_PREFIX   = "metseg"

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# ── Find fold output files ────────────────────────────────────────────
search_dirs = [
    PROJECT_DIR / "cnn_fold_outputs",
    PROJECT_DIR / "fold_outputs",
    PROJECT_DIR / "notebooks" / "fold_outputs",
    Path.home() / "Downloads",
]

found_files = []
input_base  = None

for search_dir in search_dirs:
    if search_dir is None or not search_dir.exists():
        continue
    for pat in [
        f"{FILE_PREFIX}_fold*_best.pth",
        f"{FILE_PREFIX}_fold*_latest.pth",
        f"cnn_{FILE_PREFIX}_embeddings_fold*.npz",
        f"cnn_{FILE_PREFIX}_embeddings_fold*_meta.json",
        f"{FILE_PREFIX}_fold*_training_curves.png",
        f"{FILE_PREFIX}_fold*_metrics.json",
    ]:
        for f in search_dir.rglob(pat):
            if f not in found_files:
                found_files.append(f)
    if found_files:
        input_base = search_dir
        break

if not found_files:
    print("No fold output files found!")
    print("\nExpected files like:")
    print("  metseg_fold0_best.pth")
    print("  metseg_fold0_latest.pth")
    print("  cnn_metseg_embeddings_fold0.npz")
    print("\nSearched in:")
    for d in search_dirs:
        if d:
            print(f"  {d}")
    print("\nTo fix:")
    print("  1. Download Kaggle notebook output (version output tab)")
    print("  2. Extract to: Phase2/cnn_fold_outputs/")
    print("  3. Re-run this script")
    sys.exit(1)

print(f"Found {len(found_files)} files in {input_base}:")
for f in sorted(found_files):
    print(f"  {f.name}  ({f.stat().st_size / 1024**2:.1f} MB)")

# ── Find kaggle CLI ───────────────────────────────────────────────────
def find_kaggle_bin():
    for candidate in [
        shutil.which("kaggle"),
        os.path.expanduser("~/.local/bin/kaggle"),
        str(Path.home() / "canada_me" / "explainable_diseas" / ".venv" / "bin" / "kaggle"),
        "/home/moamed/miniconda3/bin/kaggle",
        "/home/moamed/miniconda3/envs/datapre/bin/kaggle",
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None

kaggle_bin = find_kaggle_bin()
if not kaggle_bin:
    print("kaggle CLI not found! Install with: pip install kaggle")
    sys.exit(1)
print(f"\nKaggle CLI: {kaggle_bin}")

# ── Auth setup ────────────────────────────────────────────────────────
kaggle_dir = Path.home() / '.kaggle'
creds_file = kaggle_dir / 'kaggle.json'
bak_file   = kaggle_dir / 'kaggle.json.bak'

kaggle_dir.mkdir(exist_ok=True)
if creds_file.exists():
    shutil.copy2(creds_file, bak_file)
    creds_file.rename(bak_file)
    print("Backed up kaggle.json -> kaggle.json.bak")

os.environ['KAGGLE_API_TOKEN'] = ACCOUNT['key']
cmd_env = os.environ.copy()
cmd_env['KAGGLE_API_TOKEN'] = ACCOUNT['key']
print(f"Using account: {ACCOUNT['username']}")

try:
    # ── Build staging directory ───────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="kaggle_cnn_fold_outputs_"))
    (staging / "checkpoints").mkdir()
    (staging / "embeddings").mkdir()
    (staging / "figures").mkdir()
    (staging / "metrics").mkdir()

    copied = 0
    for f in found_files:
        if f.suffix == '.pth':
            dest_dir = staging / "checkpoints"
        elif f.suffix == '.npz' or (f.suffix == '.json' and 'embedding' in f.name):
            dest_dir = staging / "embeddings"
        elif f.suffix == '.png':
            dest_dir = staging / "figures"
        elif f.suffix == '.json' and 'metrics' in f.name:
            dest_dir = staging / "metrics"
        else:
            dest_dir = staging
        shutil.copy2(str(f), str(dest_dir / f.name))
        copied += 1

    # Remove empty dirs
    for d in [staging / "checkpoints", staging / "embeddings",
              staging / "figures", staging / "metrics"]:
        if not any(d.iterdir()):
            d.rmdir()

    print(f"\nStaged {copied} files:")
    for p in sorted(staging.rglob('*')):
        if p.is_file():
            print(f"  {p.relative_to(staging)}  ({p.stat().st_size / 1024**2:.1f} MB)")

    # Dataset metadata
    metadata = {
        "title":    DATASET_TITLE,
        "id":       DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    # Count folds included
    folds_included = {i for f in found_files for i in range(3) if f"fold{i}" in f.name}
    fold_str = "+".join(str(f) for f in sorted(folds_included))
    total_mb = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) / 1024**2
    print(f"\nUploading to {DATASET_SLUG}  ({total_mb:.1f} MB)...")

    # Try create first, then version
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=1800
    )

    combined = (result.stdout + result.stderr).lower()
    if result.returncode != 0 or any(x in combined for x in ["already in use", "already exists", "error"]):
        print("Dataset already exists -> uploading as new version...")
        result = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging),
             "-m", f"Fold {fold_str} outputs (checkpoints + embeddings)",
             "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=1800
        )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    combined_result = (result.stdout + result.stderr).lower()
    if result.returncode == 0 and "error" not in combined_result:
        print(f"\nUpload complete!")
        print(f"  Dataset: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"  Wait ~5 min for Kaggle to process")
        print(f"\n  Add this dataset to your notebook before running the next fold:")
        print(f"  Add Data -> Your Datasets -> metseg-fold-outputs")
    else:
        print(f"\nUpload failed (exit {result.returncode})")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")

    shutil.rmtree(staging, ignore_errors=True)

finally:
    os.environ.pop('KAGGLE_API_TOKEN', None)
    if bak_file.exists():
        if creds_file.exists():
            creds_file.unlink()
        bak_file.rename(creds_file)
        print("\nRestored original kaggle.json")

print("\nDone!")
