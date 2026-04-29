#!/usr/bin/env python3
"""
Upload MU-Glioma-Post validation data to Kaggle.

TRICK: Rename .nii.gz → .nii_gz before upload.
  Kaggle auto-extracts .gz but NOT _gz.
  11.9 GB stays 11.9 GB instead of exploding to ~60 GB.
  In the notebook, create symlinks: file.nii.gz → file.nii_gz

Structure on disk:
  validation_glomia/
  ├── MU-Glioma-Post_ClinicalData-July2025.xlsx
  ├── MU-Glioma-Post_Segmentation_Volumes.xlsx
  └── PKG - MU-Glioma-Post/MU-Glioma-Post/
      ├── PatientID_0003/
      │   ├── Timepoint_1/  (5 files: t1c, t1n, t2f, t2w, tumorMask)
      │   ├── Timepoint_2/
      │   └── Timepoint_5/
      ├── ...

Usage:
  python3 upload_mu_glioma_data.py
"""

import os, json, subprocess, sys, shutil, argparse
from pathlib import Path

ACCOUNT = {
    "username": "mohamedmohamed23",
    "key":      "KGAT_8111de93847141e6f93467431b823db0",
}

DATA_ROOT = Path("/home/moamed/HDD/validation_glomia")
NIFTI_ROOT = DATA_ROOT / "PKG - MU-Glioma-Post" / "MU-Glioma-Post"

DATASET = {
    "slug":  f"{ACCOUNT['username']}/mu-glioma-post-validation",
    "title": "MU-Glioma-Post External Validation",
}

# Staging on HDD (same filesystem as source → hardlinks work)
HDD_STAGING = Path("/home/moamed/HDD/kaggle_staging/mu_glioma_upload")


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
    """Hardlink a patient folder (with nested Timepoint subdirs) with .nii.gz → .nii_gz rename.
    
    MU-Glioma-Post structure:
      PatientID_XXXX/
        Timepoint_1/
          PatientID_XXXX_Timepoint_1_brain_t1c.nii.gz  → .nii_gz
          PatientID_XXXX_Timepoint_1_tumorMask.nii.gz   → .nii_gz
        Timepoint_2/
          ...
    
    Hardlinks are INSTANT and use ZERO extra disk space.
    Works only when source and dest are on the same filesystem.
    """
    dst_patient = dest_dir / patient_dir.name
    dst_patient.mkdir(parents=True, exist_ok=True)

    for tp_dir in sorted(patient_dir.iterdir()):
        if tp_dir.is_dir() and tp_dir.name.startswith("Timepoint"):
            dst_tp = dst_patient / tp_dir.name
            dst_tp.mkdir(parents=True, exist_ok=True)
            for f in tp_dir.iterdir():
                if f.is_file():
                    new_name = f.name.replace('.nii.gz', '.nii_gz')
                    dst = dst_tp / new_name
                    if not dst.exists():
                        try:
                            os.link(str(f), str(dst))  # hardlink: instant, zero copy
                        except OSError:
                            shutil.copy2(str(f), str(dst))  # fallback: different fs


def upload_mu_glioma():
    ds = DATASET
    print(f"\n{'='*60}")
    print(f"  {ds['title']}")
    print(f"{'='*60}")

    if not NIFTI_ROOT.exists():
        print(f"  ❌ Not found: {NIFTI_ROOT}")
        return False

    # Discover patients
    patients = sorted([d for d in NIFTI_ROOT.iterdir()
                       if d.is_dir() and d.name.startswith("PatientID")])
    total_scans = sum(
        1 for p in patients
        for tp in p.iterdir() if tp.is_dir() and tp.name.startswith("Timepoint")
    )
    print(f"  {len(patients)} patients, {total_scans} scans")

    # Clean previous staging
    if HDD_STAGING.exists():
        print(f"  Cleaning previous staging: {HDD_STAGING}")
        shutil.rmtree(HDD_STAGING)
    HDD_STAGING.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy Excel files (small, just copy) ──
    for xlsx in DATA_ROOT.glob("*.xlsx"):
        shutil.copy2(str(xlsx), HDD_STAGING / xlsx.name)
        print(f"  ✅ {xlsx.name} ({xlsx.stat().st_size/1e3:.0f} KB)")

    # ── 2. Hardlink NIfTI scans with .nii_gz trick ──
    scan_dir = HDD_STAGING / "MU-Glioma-Post"
    scan_dir.mkdir()

    print(f"  Linking {len(patients)} patients (hardlinks, instant)...")
    for i, p in enumerate(patients):
        link_patient_as_nii_gz_safe(p, scan_dir)
        if (i + 1) % 50 == 0:
            print(f"    Linked {i+1}/{len(patients)}")

    print(f"  ✅ All {len(patients)} patients linked with .nii_gz extension")

    # ── 3. Total size check ──
    total_size = sum(f.stat().st_size for f in HDD_STAGING.rglob("*") if f.is_file())
    print(f"  Total: {total_size/1e9:.1f} GB (will stay this size on Kaggle)")

    # ── 4. Write manifest ──
    patient_info = {}
    for p in patients:
        tps = sorted([d.name for d in p.iterdir()
                       if d.is_dir() and d.name.startswith("Timepoint")])
        patient_info[p.name] = tps

    (HDD_STAGING / "manifest.json").write_text(json.dumps({
        "n_patients": len(patients),
        "n_scans": total_scans,
        "patient_timepoints": patient_info,
        "note": "Files use .nii_gz extension to prevent Kaggle auto-extraction. "
                "Create symlinks in notebook: .nii_gz → .nii.gz"
    }, indent=2))

    # ── 5. Kaggle metadata ──
    (HDD_STAGING / "dataset-metadata.json").write_text(json.dumps({
        "title": ds["title"],
        "id": ds["slug"],
        "licenses": [{"name": "CC0-1.0"}]
    }, indent=2))

    # ── 6. Upload ──
    print(f"  Uploading to Kaggle (~12 GB, this will take a while)...")
    result = push_dataset(HDD_STAGING, ds["slug"],
                          f"{len(patients)} patients, {total_scans} scans (.nii_gz)")
    return result


if __name__ == "__main__":
    kaggle_bin = find_kaggle_bin()
    if not kaggle_bin:
        print("❌ kaggle CLI not found")
        sys.exit(1)
    print(f"Kaggle CLI: {kaggle_bin}")

    kaggle_dir = Path.home() / ".kaggle"
    creds_file = kaggle_dir / "kaggle.json"
    bak_file = kaggle_dir / "kaggle.json.bak"
    kaggle_dir.mkdir(exist_ok=True)
    if creds_file.exists():
        shutil.copy2(creds_file, bak_file)
        creds_file.rename(bak_file)

    # Point kaggle CLI temp dir to HDD
    hdd_tmp = Path("/home/moamed/HDD/kaggle_staging/tmp")
    hdd_tmp.mkdir(parents=True, exist_ok=True)
    cmd_env = os.environ.copy()
    cmd_env["KAGGLE_API_TOKEN"] = ACCOUNT["key"]
    cmd_env["TMPDIR"] = str(hdd_tmp)
    cmd_env["TEMP"] = str(hdd_tmp)
    cmd_env["TMP"] = str(hdd_tmp)
    print(f"Account: {ACCOUNT['username']}")
    print(f"Kaggle TMPDIR: {hdd_tmp}\n")

    try:
        ok = upload_mu_glioma()
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        if bak_file.exists():
            if creds_file.exists():
                creds_file.unlink()
            bak_file.rename(creds_file)

    print(f"\n{'='*60}")
    print(f"  RESULT: {'✅ SUCCESS' if ok else '❌ FAILED'}")
    print(f"{'='*60}")

    if ok:
        print("""
In notebook, add this cell BEFORE data loading:

    import os
    from pathlib import Path
    SYMLINK_DIR = Path('/kaggle/working/nifti_links')
    for nii_gz in Path('/kaggle/input').rglob('*.nii_gz'):
        real_name = nii_gz.name.replace('.nii_gz', '.nii.gz')
        link = SYMLINK_DIR / nii_gz.parent.name / real_name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(str(nii_gz), str(link))
    print(f"Symlinks created in {SYMLINK_DIR}")
""")
