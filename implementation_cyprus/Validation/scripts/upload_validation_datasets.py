#!/usr/bin/env python3
"""
Upload assets required for Validation_Step2_BSF_Extraction.ipynb to Kaggle.

Two separate Kaggle datasets are managed:

  1. openbtai-preprocessed
     The 373 preprocessed OpenBTAI scans produced by Validation_Step1_Preprocessing.ipynb.
     Each scan: image_t1c.nii.gz (single-ch float32, ~12MB) + mask_subregions.nii.gz
     Plus the 4 CSV/JSON metadata files.
     Total: ~4.5 GB

  2. openbtai-bsf-assets
     Everything the BSF extractor needs that is NOT patient data:
       - bsf_fold0/1/2_best.pth          (3x frozen SwinUNETR checkpoints, 720 MB each)
       - shape_scaler.pkl                 (Cyprus-fitted scaler — must NOT be refit)
       - bsf_embeddings_averaged_v2.npz  (Cyprus BSF for dim detection in notebook)
       - OpenBTAI_Radiomic.xlsx
       - OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx
     Total: ~2.2 GB

Usage:
  # Upload preprocessed scans (run AFTER Validation_Step1_Preprocessing.ipynb finishes)
  python3 scripts/upload_validation_datasets.py --dataset preprocessed

  # Upload frozen model assets (checkpoints, scaler, xlsx)
  python3 scripts/upload_validation_datasets.py --dataset bsf-assets

  # By default both are uploaded sequentially
  python3 scripts/upload_validation_datasets.py

  # Override the preprocessed data directory
  python3 scripts/upload_validation_datasets.py --preprocess-dir /path/to/preprocessed_openbtai

Note: This script uses the same Kaggle API key as upload_fold_outputs.py.
"""

import os, json, subprocess, sys, tempfile, shutil, argparse
from pathlib import Path

# ── Account config (same as upload_fold_outputs.py) ─────────────────────────
ACCOUNT = {
    "username": "boufafamoamed",
    "key": "KGAT_ba154e8f08aaa10ac815ea34934226d6",
}

# ── Dataset slugs ─────────────────────────────────────────────────────────────
DATASETS = {
    "preprocessed": {
        "slug":  "boufafamoamed/openbtai-preprocessed",
        "title": "OpenBTAI Preprocessed Scans (Val 1)",
    },
    "bsf-assets": {
        "slug":  "boufafamoamed/openbtai-bsf-assets",
        "title": "OpenBTAI BSF Extraction Assets",
    },
}

# ── Local paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent                       # Validation/scripts/
VALID_DIR   = SCRIPT_DIR.parent                           # Validation/
PROJECT_DIR = VALID_DIR.parent                            # implementation_cyprus/

DEFAULT_PREPROCESS_DIR = Path("/home/moamed/HDD/validation_data/preprocessed_openbtai")

# BSF checkpoint files
CHECKPOINT_DIR = PROJECT_DIR / "Phase3" / "bsf_fold_outputs" / "checkpoints"

# shape_scaler.pkl — Cyprus-fitted, must NOT be refit on OpenBTAI
SHAPE_SCALER = PROJECT_DIR / "Phase3" / "bsf_fold_outputs" / "embeddings_hybrid" / "shape_scaler.pkl"

# Cyprus BSF embeddings (for dim detection inside notebook)
CYPRUS_BSF = PROJECT_DIR / "Phase3" / "bsf_v2_embeddings" / "bsf_embeddings_averaged_v2.npz"

# OpenBTAI radiomics xlsx files (used for shape feature extraction)
RADIO_XLSX_DIR = Path("/home/moamed/HDD/validation_data")

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Upload Validation Step 2 assets to Kaggle")
parser.add_argument(
    "--dataset", type=str, default="all",
    choices=["all", "preprocessed", "bsf-assets"],
    help="Which dataset to upload (default: all)"
)
parser.add_argument(
    "--preprocess-dir", type=str, default=None,
    help=f"Path to preprocessed_openbtai/ folder (default: {DEFAULT_PREPROCESS_DIR})"
)
args = parser.parse_args()

preprocess_dir = Path(args.preprocess_dir) if args.preprocess_dir else DEFAULT_PREPROCESS_DIR
upload_targets = ["preprocessed", "bsf-assets"] if args.dataset == "all" else [args.dataset]


# ── Kaggle CLI ─────────────────────────────────────────────────────────────────
def find_kaggle_bin():
    for candidate in [
        shutil.which("kaggle"),
        os.path.expanduser("~/.local/bin/kaggle"),
        "/home/moamed/miniconda3/bin/kaggle",
        "/home/moamed/miniconda3/envs/datapre/bin/kaggle",
        str(Path.home() / "canada_me" / "explainable_diseas" / ".venv" / "bin" / "kaggle"),
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None

kaggle_bin = find_kaggle_bin()
if not kaggle_bin:
    print("❌ kaggle CLI not found!  Install with:  pip install kaggle")
    sys.exit(1)
print(f"Kaggle CLI : {kaggle_bin}")


# ── Auth ────────────────────────────────────────────────────────────────────────
kaggle_dir = Path.home() / ".kaggle"
creds_file = kaggle_dir / "kaggle.json"
bak_file   = kaggle_dir / "kaggle.json.bak"

if creds_file.exists():
    shutil.copy2(creds_file, bak_file)
    creds_file.rename(bak_file)
    print("Backed up kaggle.json → kaggle.json.bak")

cmd_env = os.environ.copy()
cmd_env["KAGGLE_API_TOKEN"] = ACCOUNT["key"]
print(f"Using account: {ACCOUNT['username']} via KAGGLE_API_TOKEN\n")


# ── Helper: create or version a dataset ────────────────────────────────────────
def push_dataset(staging: Path, slug: str, version_msg: str) -> bool:
    """Try create first; fall back to version if dataset already exists."""
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=7200,
    )
    combined = (result.stdout + result.stderr).lower()
    already  = (result.returncode != 0 or
                "already in use" in combined or
                "already exists" in combined or
                ("error" in combined and "successfully created" not in combined))

    if already:
        print("Dataset already exists → uploading new version…")
        result = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging),
             "-m", version_msg, "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=7200,
        )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    ok = (result.returncode == 0 and
          "error" not in (result.stdout + result.stderr).lower())
    if ok:
        print(f"✅ Upload complete!  https://www.kaggle.com/datasets/{slug}")
        print(   "   Wait ~5 min for Kaggle to process the dataset.\n")
    else:
        print(f"❌ Upload failed (exit code {result.returncode})\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 — openbtai-preprocessed
# ─────────────────────────────────────────────────────────────────────────────
def upload_preprocessed():
    ds = DATASETS["preprocessed"]
    print("=" * 65)
    print(f"  DATASET: {ds['slug']}")
    print("=" * 65)

    if not preprocess_dir.exists():
        print(f"❌ Preprocessed data not found: {preprocess_dir}")
        print("   Run Validation_Step1_Preprocessing.ipynb first.")
        return False

    # Check for expected outputs
    img_files  = list(preprocess_dir.rglob("image_t1c.nii.gz"))
    msk_files  = list(preprocess_dir.rglob("mask_subregions.nii.gz"))
    meta_files = [preprocess_dir / f for f in [
        "openbtai_patient_timelines.csv",
        "openbtai_scan_volumes.csv",
        "openbtai_lesion_selection.csv",
        "preprocessing_metadata.json",
    ]]
    meta_ok = [f for f in meta_files if f.exists()]

    print(f"  image_t1c.nii.gz files     : {len(img_files)}")
    print(f"  mask_subregions.nii.gz     : {len(msk_files)}")
    print(f"  metadata CSV/JSON files    : {len(meta_ok)} / {len(meta_files)}")

    if len(img_files) == 0:
        print("❌ No preprocessed image files found — check preprocessing step.")
        return False

    if len(img_files) != len(msk_files):
        print(f"⚠ Mismatch: {len(img_files)} images vs {len(msk_files)} masks.")

    missing_meta = [f.name for f in meta_files if not f.exists()]
    if missing_meta:
        print(f"⚠ Missing metadata files: {missing_meta}")

    # Estimate total size
    total_bytes = sum(f.stat().st_size for f in preprocess_dir.rglob("*") if f.is_file())
    print(f"  Total size                 : {total_bytes/1e9:.2f} GB")
    print()

    # Build staging — symlink-free copy would take forever for 4.5GB;
    # instead write metadata.json into the actual preprocess_dir temporarily.
    staging = preprocess_dir
    meta_path = staging / "dataset-metadata.json"
    meta_path.write_text(json.dumps({
        "title": ds["title"],
        "id":    ds["slug"],
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    print(f"Uploading {total_bytes/1e9:.2f} GB from {staging} …")
    print("  (large upload — may take 30-60 min on a standard connection)\n")

    ok = push_dataset(staging, ds["slug"], f"{len(img_files)} preprocessed scans (image_t1c float32)")
    meta_path.unlink(missing_ok=True)   # clean up the metadata file we injected
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 — openbtai-bsf-assets
# ─────────────────────────────────────────────────────────────────────────────
def upload_bsf_assets():
    ds = DATASETS["bsf-assets"]
    print("=" * 65)
    print(f"  DATASET: {ds['slug']}")
    print("=" * 65)

    # Collect files
    required = {}

    # 3 fold checkpoints
    for fold in range(3):
        p = CHECKPOINT_DIR / f"bsf_fold{fold}_best.pth"
        required[f"bsf_fold{fold}_best.pth"] = p

    # shape_scaler.pkl
    required["shape_scaler.pkl"] = SHAPE_SCALER

    # Cyprus BSF embeddings (for dim detection)
    required["bsf_embeddings_averaged_v2.npz"] = CYPRUS_BSF

    # XLSX files (search in HDD validation data dir)
    for xlsx_name in ["OpenBTAI_Radiomic.xlsx", "OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx"]:
        p = RADIO_XLSX_DIR / xlsx_name
        if not p.exists():
            # fallback: search nearby
            for candidate in RADIO_XLSX_DIR.rglob(xlsx_name):
                p = candidate; break
        required[xlsx_name] = p

    # Report status
    ok_files    = {k: v for k, v in required.items() if v.exists()}
    missing     = {k: v for k, v in required.items() if not v.exists()}
    total_bytes = sum(v.stat().st_size for v in ok_files.values())

    print(f"  Files found    : {len(ok_files)} / {len(required)}")
    for name, path in sorted(ok_files.items()):
        print(f"    ✅  {name:50s} {path.stat().st_size/1e6:8.1f} MB")
    if missing:
        print(f"\n  Files NOT FOUND : {len(missing)}")
        for name, path in missing.items():
            print(f"    ❌  {name:50s}  expected: {path}")
        if len(missing) >= len(required):
            print("\n❌ Cannot upload — no files found.")
            return False
        print("\n⚠ Proceeding with partial upload (missing files will be absent on Kaggle).")

    print(f"\n  Total size     : {total_bytes/1e6:.1f} MB\n")

    # Build staging
    staging = Path(tempfile.mkdtemp(prefix="kaggle_bsf_assets_"))
    try:
        (staging / "checkpoints").mkdir()
        (staging / "assets").mkdir()

        for name, path in ok_files.items():
            if path.suffix == ".pth":
                shutil.copy2(str(path), staging / "checkpoints" / name)
            else:
                shutil.copy2(str(path), staging / "assets" / name)

        print("Staged files:")
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(staging)}  ({p.stat().st_size/1e6:.1f} MB)")

        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": ds["title"],
            "id":    ds["slug"],
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2))

        print(f"\nUploading {total_bytes/1e6:.1f} MB …\n")
        return push_dataset(staging, ds["slug"],
                            "Frozen BSF checkpoints + shape_scaler + XLSX (Validation Step 2)")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
results = {}
try:
    if "preprocessed" in upload_targets:
        results["preprocessed"] = upload_preprocessed()

    if "bsf-assets" in upload_targets:
        results["bsf-assets"] = upload_bsf_assets()

finally:
    # Always restore original kaggle.json
    os.environ.pop("KAGGLE_API_TOKEN", None)
    if bak_file.exists():
        if creds_file.exists():
            creds_file.unlink()
        bak_file.rename(creds_file)
        print("Restored original kaggle.json")

print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
for name, ok in results.items():
    status = "✅ OK" if ok else "❌ FAILED"
    print(f"  {name:20s} : {status}")

print("""
Next steps on Kaggle:
  1. Open Validation_Step2_BSF_Extraction.ipynb
  2. Add datasets:
       boufafamoamed/openbtai-preprocessed
       boufafamoamed/openbtai-bsf-assets
  3. Enable GPU (T4 x2 recommended)
  4. Run all cells  →  output: openbtai_hybrid_embeddings_v2.npz
  5. Download the output folder and upload for Step 3 (TaViT inference)
""")
