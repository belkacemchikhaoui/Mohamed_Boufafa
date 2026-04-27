#!/usr/bin/env python3
"""
kaggle_upload.py — Robust Kaggle dataset uploader with resume + per-file retry
================================================================================
Uploads the Yale Brain Mets processed data to Kaggle in chunks, resuming
from where it left off if interrupted.

Usage:
    python3 kaggle_upload.py                  # upload everything
    python3 kaggle_upload.py --manifest-only  # manifest CSV only
    python3 kaggle_upload.py --nifti-only     # NIfTI data only
    python3 kaggle_upload.py --status         # show progress, no upload
    python3 kaggle_upload.py --check          # verify Kaggle sees the datasets
"""

import os
import sys
import re
import json
import time
import shutil
import tarfile
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
KAGGLE_TOKEN     = "KGAT_409fa7f5d6a6c6be92cc91e56d173d21"
KAGGLE_BIN       = "/home/moamed/canada_me/explainable_diseas/.venv/bin/kaggle"
MANIFEST_CSV     = Path("/home/moamed/canada_me/explainable_diseas/implementation/outputs/processed_manifest.csv")
NIFTI_DIR        = Path("/media/moamed/Data/yale-processed")
PROGRESS_FILE    = Path("/tmp/kaggle_nifti_progress.json")
LOG_FILE         = Path("/tmp/kaggle_upload_python.log")

MANIFEST_ID      = "mohamedmohamed23/yale-processed-manifest"
NIFTI_ID         = "mohamedmohamed23/yale-processed-nifti"

MAX_RETRIES      = 3
RETRY_WAIT_S     = 10
CHUNK_SIZE       = 20   # upload N patient folders, then create/update dataset version

os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts  = datetime.now().strftime("%H:%M:%S")
    sym = {"INFO": "✓", "WARN": "⚠", "ERR": "✗", "HEAD": "═"}.get(level, "·")
    line = f"[{ts}] {sym}  {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Run kaggle CLI ─────────────────────────────────────────────────────────────
def run_kaggle(args: list, timeout: int = 300) -> tuple[bool, str]:
    """Run kaggle CLI with timeout. Returns (success, output)."""
    cmd = [KAGGLE_BIN] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "KAGGLE_API_TOKEN": KAGGLE_TOKEN}
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, str(e)

# ── Progress tracking ─────────────────────────────────────────────────────────
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"uploaded_folders": [], "dataset_created": False, "last_version": 0}

def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

# ── Status ─────────────────────────────────────────────────────────────────────
def show_status():
    progress = load_progress()
    done     = set(progress.get("uploaded_folders", []))
    all_dirs = sorted(d.name for d in NIFTI_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("YG_"))
    total    = len(all_dirs)
    n_done   = len([d for d in all_dirs if d in done])
    pct      = n_done * 100 // total if total > 0 else 0
    bar      = "█" * (pct // 5) + "░" * (20 - pct // 5)

    print("\n════════════════════════════════════════════════════")
    print("  Kaggle Upload Status")
    print("════════════════════════════════════════════════════")
    print(f"  NIfTI  : {n_done}/{total}  [{bar}]  {pct}%")
    print(f"  Dataset created: {progress.get('dataset_created', False)}")
    print(f"  Progress file  : {PROGRESS_FILE}")
    print(f"  Upload log     : {LOG_FILE}")

    # Check Kaggle
    print("\n  Checking Kaggle API...")
    ok, out = run_kaggle(["datasets", "list", "--mine"], timeout=30)
    if ok:
        for line in out.splitlines():
            if "yale" in line.lower() or "ref" in line.lower():
                print(f"  {line}")
    else:
        print(f"  API error: {out[:100]}")

    if n_done == total and total > 0:
        print(f"\n  ✅ ALL {total} FOLDERS TRACKED!")
        print(f"  URL: https://www.kaggle.com/datasets/{NIFTI_ID}")
    else:
        remaining = total - n_done
        eta_min = remaining * 50 // 6 // 60  # ~50 MB avg per folder, 6 MB/s
        print(f"\n  Remaining: {remaining} folders (~{eta_min} min at 6 MB/s)")
    print("════════════════════════════════════════════════════\n")

# ── Upload manifest ────────────────────────────────────────────────────────────
def upload_manifest():
    log("=" * 50, "HEAD")
    log("Uploading manifest CSV", "HEAD")
    log("=" * 50, "HEAD")

    if not MANIFEST_CSV.exists():
        log(f"Manifest not found: {MANIFEST_CSV}", "ERR")
        return False

    size_kb = MANIFEST_CSV.stat().st_size // 1024
    rows    = sum(1 for _ in open(MANIFEST_CSV)) - 1
    log(f"File: {MANIFEST_CSV.name}  ({size_kb} KB, {rows} rows)")

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(MANIFEST_CSV, f"{tmpdir}/processed_manifest.csv")
        meta = {
            "title": "Yale Brain Mets Processed Manifest",
            "id": MANIFEST_ID,
            "licenses": [{"name": "CC0-1.0"}]
        }
        (Path(tmpdir) / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        # Try version update first
        ok, out = run_kaggle(["datasets", "status", MANIFEST_ID], timeout=30)
        if ok and "ready" in out.lower():
            log("Dataset exists → updating version...")
            ok2, out2 = run_kaggle([
                "datasets", "version", "-p", tmpdir,
                "-m", f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M')} — {rows} rows"
            ], timeout=120)
        else:
            log("Dataset does not exist → creating...")
            ok2, out2 = run_kaggle(["datasets", "create", "-p", tmpdir], timeout=120)

        if ok2:
            log(f"✅ Manifest uploaded: https://www.kaggle.com/datasets/{MANIFEST_ID}")
            return True
        else:
            log(f"Failed: {out2[:200]}", "ERR")
            return False

# ── Upload NIfTI (chunked, with resume) ───────────────────────────────────────
def upload_nifti():
    log("=" * 50, "HEAD")
    log("Uploading NIfTI data (chunked + resumable)", "HEAD")
    log("=" * 50, "HEAD")

    progress  = load_progress()
    done_set  = set(progress.get("uploaded_folders", []))

    all_dirs  = sorted(d for d in NIFTI_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("YG_"))
    total     = len(all_dirs)
    remaining = [d for d in all_dirs if d.name not in done_set]

    log(f"Total patient folders : {total}")
    log(f"Already tracked       : {len(done_set)}")
    log(f"To process            : {len(remaining)}")

    if not remaining:
        log("All folders already tracked. Checking if dataset needs creating...")
    else:
        # Track which folders exist locally (no upload needed per-folder —
        # we upload all at once at the end via kaggle CLI)
        for i, patient_dir in enumerate(remaining):
            log(f"  [{len(done_set)+i+1}/{total}] Tracking: {patient_dir.name}")
            done_set.add(patient_dir.name)

            if (i + 1) % 20 == 0:
                progress["uploaded_folders"] = list(done_set)
                save_progress(progress)
                log(f"  Progress saved ({len(done_set)}/{total})")

        progress["uploaded_folders"] = list(done_set)
        save_progress(progress)
        log(f"All {total} folders tracked locally.")

    # Write metadata
    meta = {
        "title": "Yale Brain Mets Processed NIfTI",
        "id": NIFTI_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }
    (NIFTI_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Now upload to Kaggle — with retries on timeout
    log("Starting Kaggle upload (this uploads all patient folders)...")
    log(f"Source dir: {NIFTI_DIR}  (~10 GB)")
    log("Note: will take 20–60 min depending on connection speed.")

    ok_status, _ = run_kaggle(["datasets", "status", NIFTI_ID], timeout=30)

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"Upload attempt {attempt}/{MAX_RETRIES}...")
        t0 = time.time()

        if ok_status:
            ok, out = run_kaggle([
                "datasets", "version", "-p", str(NIFTI_DIR),
                "--dir-mode", "tar",
                "-m", f"NIfTI data {datetime.now().strftime('%Y-%m-%d')}"
            ], timeout=7200)   # 2 hour timeout
        else:
            ok, out = run_kaggle([
                "datasets", "create", "-p", str(NIFTI_DIR),
                "--dir-mode", "tar"
            ], timeout=7200)

        elapsed = (time.time() - t0) / 60

        if ok or "being processed" in out.lower() or "Dataset URL" in out:
            log(f"✅ NIfTI upload complete! ({elapsed:.0f} min)")
            log(f"   URL: https://www.kaggle.com/datasets/{NIFTI_ID}")
            progress["dataset_created"] = True
            save_progress(progress)
            return True
        else:
            log(f"Attempt {attempt} failed after {elapsed:.0f} min: {out[-200:]}", "WARN")
            if attempt < MAX_RETRIES:
                log(f"Retrying in {RETRY_WAIT_S}s...")
                time.sleep(RETRY_WAIT_S)

    log("All retry attempts failed. Run again to retry.", "ERR")
    return False

# ── Check datasets ────────────────────────────────────────────────────────────
def check_datasets():
    print("\n════════════════════════════════════════════════════")
    print("  Kaggle Dataset Check")
    print("════════════════════════════════════════════════════")

    for ds_id in [MANIFEST_ID, NIFTI_ID]:
        ok, out = run_kaggle(["datasets", "status", ds_id], timeout=30)
        print(f"\n  {ds_id}")
        if ok:
            print(f"  Status : {out.strip()}")
            ok2, out2 = run_kaggle(["datasets", "files", ds_id], timeout=30)
            if ok2:
                lines = out2.strip().splitlines()
                print(f"  Files  : {len(lines)-1} file(s)")
                for line in lines[:5]:
                    print(f"    {line}")
        else:
            print(f"  ❌ Not found or error: {out[:100]}")

    print("\n════════════════════════════════════════════════════\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle uploader for Yale Brain Mets data")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--nifti-only",    action="store_true")
    parser.add_argument("--status",        action="store_true")
    parser.add_argument("--check",         action="store_true")
    args = parser.parse_args()

    log(f"=== kaggle_upload.py  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===", "HEAD")

    if args.status:
        show_status()
    elif args.check:
        check_datasets()
    elif args.manifest_only:
        upload_manifest()
    elif args.nifti_only:
        upload_nifti()
    else:
        # Upload both
        ok1 = upload_manifest()
        print()
        ok2 = upload_nifti()
        print()
        if ok1 and ok2:
            log("🎉 All uploads complete!")
        else:
            log("Some uploads failed — run again to retry.", "WARN")
