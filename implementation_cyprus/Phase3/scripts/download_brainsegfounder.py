#!/usr/bin/env python3
"""
Download BrainSegFounder weights from HuggingFace and upload to Kaggle.
Usage: python3 scripts/download_brainsegfounder.py
"""

import os, json, subprocess, sys, tempfile, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
WEIGHT_DIR = PROJECT_DIR

# ── Download weights ──
WEIGHT_URL = "https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-finetune.pt"
WEIGHT_FILE = WEIGHT_DIR / "model_brainsegfounder_brats_finetune.pt"

if WEIGHT_FILE.exists():
    print(f"✅ Weight file already exists: {WEIGHT_FILE}")
    print(f"   Size: {WEIGHT_FILE.stat().st_size / 1024**2:.1f} MB")
else:
    print(f"Downloading BrainSegFounder BRATS-finetune weights...")
    print(f"  URL: {WEIGHT_URL}")
    print(f"  Saving to: {WEIGHT_FILE}")
    
    result = subprocess.run(
        ["wget", "-O", str(WEIGHT_FILE), WEIGHT_URL],
        timeout=600
    )
    
    if result.returncode != 0:
        # Try curl as fallback
        result = subprocess.run(
            ["curl", "-L", "-o", str(WEIGHT_FILE), WEIGHT_URL],
            timeout=600
        )
    
    if WEIGHT_FILE.exists() and WEIGHT_FILE.stat().st_size > 100_000_000:
        print(f"✅ Downloaded: {WEIGHT_FILE.stat().st_size / 1024**2:.1f} MB")
    else:
        print("❌ Download failed!")
        sys.exit(1)

# ── Verify weight format ──
print("\nVerifying weight format...")
try:
    import torch
    state_dict = torch.load(str(WEIGHT_FILE), map_location="cpu", weights_only=False)
    
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Check if it's a full SwinUNETR state dict
    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
    swinvit_keys = [k for k in keys if k.startswith("swinViT.")]
    encoder_keys = [k for k in keys if k.startswith("encoder")]
    decoder_keys = [k for k in keys if k.startswith("decoder")]
    
    print(f"  Total keys: {len(keys)}")
    print(f"  swinViT keys: {len(swinvit_keys)}")
    print(f"  encoder keys: {len(encoder_keys)}")
    print(f"  decoder keys: {len(decoder_keys)}")
    print(f"  First 10 keys: {keys[:10]}")
    
    # Check input channels from first conv
    for k in keys:
        if "patch_embed" in k and "weight" in k:
            shape = state_dict[k].shape
            print(f"  Patch embedding weight shape: {shape}")
            if len(shape) >= 2:
                print(f"  → Input channels: {shape[1]}")
            break
    
    print("\n✅ Weight file verified!")
except ImportError:
    print("  ⚠️ torch not available, skipping verification")
except Exception as e:
    print(f"  ⚠️ Verification error: {e}")

# ── Upload to Kaggle ──
print("\n" + "=" * 60)
print("Upload to Kaggle?")
print("=" * 60)

ACCOUNT = {
    "username": "boufafamoamed",
    "key": "KGAT_ba154e8f08aaa10ac815ea34934226d6",
}

DATASET_SLUG = "boufafamoamed/brainsegfounder-weights"
DATASET_TITLE = "BrainSegFounder Weights (BraTS Finetune)"

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
    print("❌ kaggle CLI not found! Upload manually.")
    sys.exit(0)

print(f"Kaggle CLI: {kaggle_bin}")

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

try:
    staging = Path(tempfile.mkdtemp(prefix="kaggle_bsf_"))
    shutil.copy2(str(WEIGHT_FILE), str(staging / WEIGHT_FILE.name))
    
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    total_mb = WEIGHT_FILE.stat().st_size / 1024**2
    print(f"\nUploading to {DATASET_SLUG} ({total_mb:.1f} MB)...")
    
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=1800
    )
    
    combined_output = (result.stdout + result.stderr).lower()
    if "already in use" in combined_output or "already exists" in combined_output or "error" in combined_output:
        print("Dataset exists → uploading new version...")
        result = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging),
             "-m", "BrainSegFounder BRATS-finetune weights",
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
    else:
        print(f"\n❌ Upload may have failed")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
    
    shutil.rmtree(staging, ignore_errors=True)

finally:
    os.environ.pop('KAGGLE_API_TOKEN', None)
    if bak_file.exists():
        if creds_file.exists():
            creds_file.unlink()
        bak_file.rename(creds_file)
        print(f"\nRestored original kaggle.json")

print("\nDone!")
