#!/usr/bin/env python3
"""
Upload Swin UNETR fold outputs (checkpoints, embeddings, figures, metrics)
to Kaggle (boufafamoamed account) so they persist across notebook versions.

Usage:
  1. Download the fold output from the Kaggle notebook version output
  2. Place the files in Phase3/fold_outputs/ (or specify --input-dir)
  3. Run: python3 scripts/upload_fold_outputs.py

Expected structure in the input directory:
  checkpoints/swinunetr_fold0_best.pth
  checkpoints/swinunetr_fold0_latest.pth
  embeddings/swinunetr_embeddings_fold0.npz  (if generated)
  figures/swinunetr_fold0_training_curves.png (if generated)
  metrics/swinunetr_fold0_metrics.json        (if generated)
"""

import os, json, subprocess, sys, tempfile, shutil, argparse, glob
from pathlib import Path

# ── Account config ──
ACCOUNT = {
    "username": "mohamedmohamed23",
    "key": "KGAT_8111de93847141e6f93467431b823db0",
}

EXPERIMENTS = {
    "swinunetr": {
        "slug": "mohamedmohamed23/swinunetr-fold-outputs",
        "title": "Swin UNETR Fold Outputs (Phase 3)",
        "prefix": "swinunetr",
    },
    "bsf": {
        "slug": "mohamedmohamed23/brainsegfounder-fold-outputs",
        "title": "BrainSegFounder Fold Outputs (Phase 3)",
        "prefix": "bsf",
    },
}

# ── Parse args ──
parser = argparse.ArgumentParser(description="Upload fold outputs to Kaggle")
parser.add_argument("--input-dir", type=str, default=None,
                    help="Directory containing fold outputs to upload")
parser.add_argument("--experiment", type=str, default="swinunetr",
                    choices=["swinunetr", "bsf"],
                    help="Which experiment: 'swinunetr' (Exp A) or 'bsf' (Exp B BrainSegFounder)")
args = parser.parse_args()

exp = EXPERIMENTS[args.experiment]
DATASET_SLUG = exp["slug"]
DATASET_TITLE = exp["title"]
FILE_PREFIX = exp["prefix"]

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# ── Find input files ──
# Search in multiple places for fold outputs
search_dirs = [
    Path(args.input_dir) if args.input_dir else None,
    PROJECT_DIR / "bsf_fold_outputs" if args.experiment == "bsf" else None,
    PROJECT_DIR / "fold_outputs",
    PROJECT_DIR / "notebooks" / "fold_outputs",
    Path.home() / "Downloads",
]

found_files = []
input_base = None

for search_dir in search_dirs:
    if search_dir is None or not search_dir.exists():
        continue
    
    # Look for checkpoint/embedding files (supports both swinunetr_ and bsf_ prefixes)
    # For BrainSegFounder, include early stopping files
    patterns = [
        f"**/{FILE_PREFIX}_fold*_best.pth",
        f"**/{FILE_PREFIX}_fold*_latest.pth",
        f"**/{FILE_PREFIX}_fold*_early.pth",
        f"**/{FILE_PREFIX}_fold*_earlystopped.flag",
        f"**/{FILE_PREFIX}_embeddings_fold*.npz",
        f"**/{FILE_PREFIX}_embeddings_fold*_meta.json",
        f"**/{FILE_PREFIX}_fold*_training_curves.png",
        f"**/{FILE_PREFIX}_fold*_metrics.json"
    ]
    
    for pat in patterns:
        for f in search_dir.rglob(pat.replace("**/", "")):
            found_files.append(f)
    
    if found_files:
        input_base = search_dir
        break

if not found_files:
    print("❌ No fold output files found!")
    print("\nExpected files like:")
    if args.experiment == "bsf":
        print("  bsf_fold0_best.pth")
        print("  bsf_fold0_latest.pth")
        print("  bsf_fold0_early.pth")
        print("  bsf_fold0_earlystopped.flag")
        print("  bsf_fold0_training_curves.png")
        print("  bsf_fold0_metrics.json")
    else:
        print("  swinunetr_fold0_best.pth")
        print("  swinunetr_fold0_latest.pth")
        print("  swinunetr_embeddings_fold0.npz")
    print("\nSearched in:")
    for d in search_dirs:
        if d:
            print(f"  {d}")
    print("\nTo fix:")
    if args.experiment == "bsf":
        print("  1. Ensure BrainSegFounder training has completed")
        print("  2. Check that files are in: Phase3/bsf_fold_outputs/")
        print("  3. Re-run this script")
    else:
        print("  1. Download output from Kaggle notebook version")
        print("  2. Extract to: Phase3/fold_outputs/")
        print("  3. Re-run this script")
    sys.exit(1)

print(f"Found {len(found_files)} files in {input_base}:")
for f in sorted(found_files):
    size_mb = f.stat().st_size / 1024**2
    print(f"  {f.name} ({size_mb:.1f} MB)")

# ── Find kaggle CLI ──
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
    print("❌ kaggle CLI not found! Install with: pip install kaggle")
    sys.exit(1)
print(f"Kaggle CLI: {kaggle_bin}")

# ── Setup auth ──
kaggle_dir = Path.home() / '.kaggle'
creds_file = kaggle_dir / 'kaggle.json'
bak_file = kaggle_dir / 'kaggle.json.bak'

if creds_file.exists():
    shutil.copy2(creds_file, bak_file)
    creds_file.rename(bak_file)
    print(f"Backed up kaggle.json → kaggle.json.bak")

os.environ['KAGGLE_API_TOKEN'] = ACCOUNT['key']
cmd_env = os.environ.copy()
cmd_env['KAGGLE_API_TOKEN'] = ACCOUNT['key']
print(f"Using account: {ACCOUNT['username']} via KAGGLE_API_TOKEN")

try:
    # ── Create staging directory with proper structure ──
    staging = Path(tempfile.mkdtemp(prefix="kaggle_fold_outputs_"))
    
    # Create subdirectories
    (staging / "checkpoints").mkdir()
    (staging / "embeddings").mkdir()
    (staging / "figures").mkdir()
    (staging / "metrics").mkdir()
    
    # Copy files preserving directory structure
    copied = 0
    for f in found_files:
        # Determine target subdirectory
        if f.suffix == '.pth' or f.suffix == '.flag':
            dest_dir = staging / "checkpoints"
        elif f.suffix == '.npz' or (f.suffix == '.json' and 'embedding' in f.name):
            dest_dir = staging / "embeddings"
        elif f.suffix == '.png':
            dest_dir = staging / "figures"
        elif f.suffix == '.json' and 'metrics' in f.name:
            dest_dir = staging / "metrics"
        else:
            dest_dir = staging
        
        dest = dest_dir / f.name
        shutil.copy2(str(f), str(dest))
        copied += 1
    
    print(f"\nStaged {copied} files in {staging}")
    
    # Remove empty directories
    for d in [staging / "checkpoints", staging / "embeddings", staging / "figures", staging / "metrics"]:
        if not any(d.iterdir()):
            d.rmdir()
    
    # List what's being uploaded
    for p in sorted(staging.rglob('*')):
        if p.is_file():
            print(f"  {p.relative_to(staging)} ({p.stat().st_size / 1024**2:.1f} MB)")
    
    # Create metadata
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # ── Upload ──
    total_mb = sum(f.stat().st_size for f in staging.rglob('*') if f.is_file()) / 1024**2
    print(f"\nUploading to {DATASET_SLUG} ({total_mb:.1f} MB)...")
    print("This may take a few minutes...")
    
    # Count which folds are included (for version message)
    folds_included = set()
    for f in found_files:
        for i in range(3):
            if f"fold{i}" in f.name:
                folds_included.add(i)
    fold_str = "+".join(str(f) for f in sorted(folds_included))
    
    # Try create first
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=1800
    )
    
    combined_output = (result.stdout + result.stderr).lower()
    create_failed = (
        result.returncode != 0 or
        "already in use" in combined_output or
        "already exists" in combined_output or
        "error" in combined_output
    )
    
    if create_failed:
        print("Dataset already exists → uploading as new version...")
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
        print(f"\n✅ Upload complete!")
        print(f"   Dataset URL: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"   Wait ~5 min for Kaggle to process")
        print(f"\n   Add this dataset to your notebook before running the next fold.")
    else:
        print(f"\n❌ Upload failed (exit code {result.returncode})")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
    
    # Clean up staging
    shutil.rmtree(staging, ignore_errors=True)

finally:
    # Always restore original kaggle.json
    os.environ.pop('KAGGLE_API_TOKEN', None)
    if bak_file.exists():
        if creds_file.exists():
            creds_file.unlink()
        bak_file.rename(creds_file)
        print(f"\nRestored original kaggle.json")

print("\nDone!")
