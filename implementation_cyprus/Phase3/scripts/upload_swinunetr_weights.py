#!/usr/bin/env python3
"""
Upload Swin UNETR pretrained weights to Kaggle (boufafamoamed account).
Usage: python3 scripts/upload_swinunetr_weights.py
"""

import os, json, subprocess, sys, tempfile, shutil
from pathlib import Path

# ── Account config ──
ACCOUNT = {
    "username": "boufafamoamed",
    "key": "KGAT_ba154e8f08aaa10ac815ea34934226d6",
}

DATASET_SLUG = "boufafamoamed/swinunetr-pretrained-weights"
DATASET_TITLE = "Swin UNETR Pretrained Weights (BraTS 2021)"

# ── Find the weight file ──
SCRIPT_DIR = Path(__file__).parent
WEIGHT_FILE = SCRIPT_DIR.parent / "model_swinvit.pt"

if not WEIGHT_FILE.exists():
    for candidate in [
        SCRIPT_DIR / "model_swinvit.pt",
        SCRIPT_DIR.parent.parent / "model_swinvit.pt",
    ]:
        if candidate.exists():
            WEIGHT_FILE = candidate
            break

if not WEIGHT_FILE.exists():
    print(f"❌ model_swinvit.pt not found!")
    print(f"   Expected at: {WEIGHT_FILE}")
    print(f"   Download from: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt")
    sys.exit(1)

print(f"Weight file: {WEIGHT_FILE} ({WEIGHT_FILE.stat().st_size / 1024**2:.1f} MB)")

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

# ── Setup auth (KGAT_ token approach — same as upload_second_account.py) ──
kaggle_dir = Path.home() / '.kaggle'
creds_file = kaggle_dir / 'kaggle.json'
bak_file = kaggle_dir / 'kaggle.json.bak'

# Back up existing kaggle.json
if creds_file.exists():
    shutil.copy2(creds_file, bak_file)
    creds_file.rename(bak_file)  # Move aside so env var takes priority
    print(f"Backed up kaggle.json → kaggle.json.bak")

# Set KAGGLE_API_TOKEN for KGAT_ format tokens
os.environ['KAGGLE_API_TOKEN'] = ACCOUNT['key']
cmd_env = os.environ.copy()
cmd_env['KAGGLE_API_TOKEN'] = ACCOUNT['key']

print(f"Using account: {ACCOUNT['username']} via KAGGLE_API_TOKEN")

try:
    # ── Create staging directory ──
    staging = Path(tempfile.mkdtemp(prefix="kaggle_swinunetr_"))
    
    dst = staging / "model_swinvit.pt"
    print(f"Copying weight file to staging...")
    shutil.copy2(str(WEIGHT_FILE), str(dst))
    
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    print(f"Staging dir: {staging}")
    print(f"Files: {[f.name for f in staging.iterdir()]}")

    # ── Upload ──
    print(f"\nUploading to {DATASET_SLUG}...")
    print("This may take 5-10 minutes for 393 MB...")
    
    # Try create first
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=600
    )
    
    if result.returncode != 0 and "already exists" in (result.stderr + result.stdout).lower():
        print("Dataset exists, updating version...")
        result = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging),
             "-m", "Swin UNETR pretrained weights for Phase 3",
             "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=600
        )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print(f"✅ Upload complete!")
        print(f"   Dataset URL: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"   Wait ~5 min for Kaggle to process")
    else:
        print(f"❌ Upload failed (exit code {result.returncode})")

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
