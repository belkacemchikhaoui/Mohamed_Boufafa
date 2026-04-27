#!/usr/bin/env python3
"""
Upload BraTS 2024 data to Kaggle.

TRICK: Rename .nii.gz → .nii_gz before upload.
  Kaggle auto-extracts .gz but NOT _gz.
  5 GB stays 5 GB instead of exploding to 50 GB.
  In the notebook, create symlinks: file.nii.gz → file.nii_gz

4 datasets, each with its own --dataset flag:
  metadata    → xlsx + Phase1 outputs (~1 MB)
  weights     → SegResNet + nnU-Net + SwinUNETR (~1.5 GB)
  validation  → 271 additional training folders (~5 GB as _gz)
  training    → main training folders (~28 GB as _gz)

Usage:
  python3 upload_brats2024_data.py --dataset metadata
  python3 upload_brats2024_data.py --dataset weights
  python3 upload_brats2024_data.py --dataset validation
  python3 upload_brats2024_data.py --dataset training
"""

import os, json, subprocess, sys, shutil, argparse, tempfile
from pathlib import Path

ACCOUNT = {
    "username": "zinou123viva",
    "key":      "KGAT_9811f85e7078d6bbb85aad26ae841bdc",
}

DATA_ROOT = Path("/home/moamed/HDD/brats2024_posttreatment")
MAIN_TRAINING = DATA_ROOT / "BraTS2024-BraTS-GLI-TrainingData" / "training_data1_v2"
ADDITIONAL_TRAINING = DATA_ROOT / "training_data_additional"
MODELS_DIR = DATA_ROOT / "models"
METADATA_XLSX = DATA_ROOT / "BraTS-PTG supplementary demographic information and metadata.xlsx"
CITATIONS_BIB = DATA_ROOT / "CITATIONS.bib"
PHASE1_OUTPUTS = Path("/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase1/outputs")

DATASETS = {
    "metadata":   {"slug": f"{ACCOUNT['username']}/brats2024-metadata",
                   "title": "BraTS 2024 Metadata"},
    "weights":    {"slug": f"{ACCOUNT['username']}/brats2024-pretrained-weights",
                   "title": "BraTS 2024 Pretrained Weights"},
    "validation": {"slug": f"{ACCOUNT['username']}/brats2024-validation-data",
                   "title": "BraTS 2024 Validation Data (Additional Training)"},
    "training":   {"slug": f"{ACCOUNT['username']}/brats2024-training-data",
                   "title": "BraTS 2024 Training Data"},
}


def find_kaggle_bin():
    for c in [shutil.which("kaggle"),
              os.path.expanduser("~/.local/bin/kaggle"),
              "/home/moamed/miniconda3/bin/kaggle",
              "/home/moamed/miniconda3/envs/datapre/bin/kaggle",
              str(Path.home() / "canada_me" / "explainable_diseas" / ".venv" / "bin" / "kaggle")]:
        if c and os.path.isfile(c): return c
    return None


def push_dataset(staging, slug, msg):
    r = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=14400)
    combo = (r.stdout + r.stderr).lower()
    if r.returncode != 0 or "already" in combo or ("error" in combo and "successfully" not in combo):
        print("  Exists → new version…")
        r = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging), "-m", msg, "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=14400)
    print(r.stdout)
    if r.stderr: print(r.stderr)
    ok = r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()
    print(f"  {'✅' if ok else '❌'} https://www.kaggle.com/datasets/{slug}")
    return ok


def link_patient_as_nii_gz_safe(patient_dir, dest_dir):
    """Hardlink a patient folder with .nii.gz → .nii_gz rename.
    Hardlinks are INSTANT and use ZERO extra disk space.
    Works only when source and dest are on the same filesystem (both on HDD here).
    """
    dst_patient = dest_dir / patient_dir.name
    dst_patient.mkdir(parents=True, exist_ok=True)
    for f in patient_dir.iterdir():
        if f.is_file():
            new_name = f.name.replace('.nii.gz', '.nii_gz')
            dst = dst_patient / new_name
            if not dst.exists():
                try:
                    os.link(str(f), str(dst))  # hardlink: instant, zero copy
                except OSError:
                    shutil.copy2(str(f), str(dst))  # fallback: different fs


# ═══════════ 1. METADATA ═══════════
def upload_metadata():
    ds = DATASETS["metadata"]
    print(f"\n{'='*60}\n  {ds['title']}\n{'='*60}")
    staging = Path(tempfile.mkdtemp(prefix="kg_meta_"))
    try:
        if METADATA_XLSX.exists():
            shutil.copy2(str(METADATA_XLSX), staging / METADATA_XLSX.name)
            print(f"  ✅ {METADATA_XLSX.name}")
        if CITATIONS_BIB.exists():
            shutil.copy2(str(CITATIONS_BIB), staging / CITATIONS_BIB.name)
        if PHASE1_OUTPUTS.exists():
            for f in PHASE1_OUTPUTS.iterdir():
                if f.is_file():
                    shutil.copy2(str(f), staging / f.name)
                    print(f"  ✅ {f.name}")
        else:
            print("  ⚠️  Phase1 outputs not yet generated")
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": ds["title"], "id": ds["slug"], "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        return push_dataset(staging, ds["slug"], "Metadata + Phase1 outputs")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ═══════════ 2. WEIGHTS ═══════════
def upload_weights():
    ds = DATASETS["weights"]
    print(f"\n{'='*60}\n  {ds['title']}\n{'='*60}")
    staging = Path(tempfile.mkdtemp(prefix="kg_wt_"))
    try:
        # SegResNet
        seg = MODELS_DIR / "segresnet" / "brats_mri_segmentation" / "models" / "model.pt"
        if seg.exists():
            shutil.copy2(str(seg), staging / "segresnet_model.pt")
            print(f"  ✅ SegResNet: {seg.stat().st_size/1e6:.0f} MB")

        # nnU-Net v2
        trainer = MODELS_DIR / "nnU-Net_v2 " / "Dataset002_BRATS19" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        if trainer.exists():
            dst_t = staging / "nnunet_v2" / "Dataset002_BRATS19" / trainer.name
            dst_t.mkdir(parents=True, exist_ok=True)
            for cfg in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
                src = trainer / cfg
                if src.exists(): shutil.copy2(str(src), dst_t / cfg)
            for fold in range(5):
                ckpt = trainer / f"fold_{fold}" / "checkpoint_final.pth"
                if ckpt.exists():
                    (dst_t / f"fold_{fold}").mkdir(exist_ok=True)
                    shutil.copy2(str(ckpt), dst_t / f"fold_{fold}" / "checkpoint_final.pth")
                    print(f"  ✅ nnU-Net fold_{fold}: {ckpt.stat().st_size/1e6:.0f} MB")

        # SwinUNETR
        swin = MODELS_DIR / "swinunetr" / "swin_unetr_btcv_segmentation" / "models" / "model.pt"
        if swin.exists():
            shutil.copy2(str(swin), staging / "swinunetr_model.pt")
            print(f"  ✅ SwinUNETR: {swin.stat().st_size/1e6:.0f} MB")

        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": ds["title"], "id": ds["slug"], "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        return push_dataset(staging, ds["slug"], "SegResNet + nnU-Net v2 5-fold + SwinUNETR")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ═══════════ 3. VALIDATION ═══════════
def upload_validation():
    ds = DATASETS["validation"]
    print(f"\n{'='*60}\n  {ds['title']}\n{'='*60}")
    if not ADDITIONAL_TRAINING.exists():
        print(f"  ❌ Not found: {ADDITIONAL_TRAINING}"); return False
    dirs = sorted([d for d in ADDITIONAL_TRAINING.iterdir() if d.is_dir() and "BraTS-GLI" in d.name])
    print(f"  {len(dirs)} patient folders")

    staging = Path(tempfile.mkdtemp(prefix="kg_val_"))
    try:
        val_dir = staging / "validation_data"
        val_dir.mkdir()
        for i, d in enumerate(dirs):
            link_patient_as_nii_gz_safe(d, val_dir)
            if (i+1) % 50 == 0: print(f"    Linked {i+1}/{len(dirs)}")
        print(f"  ✅ All {len(dirs)} patients linked with .nii_gz extension (instant)")

        total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1e9
        print(f"  Total: {total:.1f} GB (will stay this size on Kaggle)")

        (staging / "manifest.json").write_text(json.dumps({
            "n_patients": len(dirs), "patient_ids": [d.name for d in dirs],
            "note": "Files use .nii_gz extension to prevent Kaggle auto-extraction. Create symlinks in notebook."
        }, indent=2))
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": ds["title"], "id": ds["slug"], "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        return push_dataset(staging, ds["slug"], f"{len(dirs)} scans (.nii_gz safe)")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ═══════════ 4. TRAINING ═══════════
# Staged on HDD (not /tmp SSD) because training data is ~28 GB
HDD_STAGING = Path("/home/moamed/HDD/kaggle_staging")

def upload_training():
    ds = DATASETS["training"]
    print(f"\n{'='*60}\n  {ds['title']}\n{'='*60}")
    if not MAIN_TRAINING.exists():
        print(f"  ❌ Not found: {MAIN_TRAINING}"); return False
    dirs = sorted([d for d in MAIN_TRAINING.iterdir() if d.is_dir() and "BraTS-GLI" in d.name])
    print(f"  {len(dirs)} patient folders")
    if len(dirs) < 100:
        print(f"  ⚠️  Only {len(dirs)} — still unzipping? Wait."); return False

    staging = HDD_STAGING / "brats2024_training_upload"
    if staging.exists():
        print(f"  Cleaning previous staging: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    print(f"  Staging on HDD: {staging}")
    print(f"  Using hardlinks (instant, zero disk copy) — source & dest on same HDD")

    try:
        train_dir = staging / "training_data"
        train_dir.mkdir()
        for i, d in enumerate(dirs):
            link_patient_as_nii_gz_safe(d, train_dir)
            if (i+1) % 200 == 0: print(f"    Linked {i+1}/{len(dirs)}")
        print(f"  ✅ All {len(dirs)} patients linked with .nii_gz extension (instant)")

        (staging / "manifest.json").write_text(json.dumps({
            "n_patients": len(dirs), "patient_ids": [d.name for d in dirs],
            "note": ".nii_gz extension prevents Kaggle extraction"
        }, indent=2))
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": ds["title"], "id": ds["slug"], "licenses": [{"name": "CC0-1.0"}]}, indent=2))

        print(f"  Uploading to Kaggle (this will take a while for 28 GB)...")
        result = push_dataset(staging, ds["slug"], f"{len(dirs)} scans (.nii_gz)")
        return result
    finally:
        # Hardlinks: removing staging does NOT delete the original files
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  Staging cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["metadata", "weights", "validation", "training", "all"])
    args = parser.parse_args()

    kaggle_bin = find_kaggle_bin()
    if not kaggle_bin: print("❌ kaggle CLI not found"); sys.exit(1)
    print(f"Kaggle CLI: {kaggle_bin}")

    kaggle_dir = Path.home() / ".kaggle"
    creds_file = kaggle_dir / "kaggle.json"
    bak_file = kaggle_dir / "kaggle.json.bak"
    kaggle_dir.mkdir(exist_ok=True)
    if creds_file.exists():
        shutil.copy2(creds_file, bak_file); creds_file.rename(bak_file)
    # Point kaggle CLI temp dir to HDD — /tmp on SSD may be full for large uploads
    hdd_tmp = Path("/home/moamed/HDD/kaggle_staging/tmp")
    hdd_tmp.mkdir(parents=True, exist_ok=True)
    cmd_env = os.environ.copy()
    cmd_env["KAGGLE_API_TOKEN"] = ACCOUNT["key"]
    cmd_env["TMPDIR"] = str(hdd_tmp)
    cmd_env["TEMP"] = str(hdd_tmp)
    cmd_env["TMP"] = str(hdd_tmp)
    print(f"Account: {ACCOUNT['username']}")
    print(f"Kaggle TMPDIR: {hdd_tmp}\n")

    dispatch = {"metadata": upload_metadata, "weights": upload_weights,
                "validation": upload_validation, "training": upload_training}
    targets = list(dispatch.keys()) if args.dataset == "all" else [args.dataset]

    try:
        results = {}
        for t in targets: results[t] = dispatch[t]()
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        if bak_file.exists():
            if creds_file.exists(): creds_file.unlink()
            bak_file.rename(creds_file)

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for n, ok in results.items(): print(f"  {n:15s} {'✅' if ok else '❌'}")
    print("""
In notebook, add this cell BEFORE data loading:

    import os
    from pathlib import Path
    SYMLINK_DIR = Path('/kaggle/working/nifti_links')
    def resolve_nii_gz(path_str):
        p = Path(path_str)
        if p.exists(): return str(p)
        # Try .nii_gz version
        nii_gz = str(p).replace('.nii.gz', '.nii_gz')
        if Path(nii_gz).exists():
            link = SYMLINK_DIR / p.parent.name / p.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists(): os.symlink(nii_gz, str(link))
            return str(link)
        raise FileNotFoundError(path_str)
""")
