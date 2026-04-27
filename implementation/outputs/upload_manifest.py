#!/usr/bin/env python3
"""
upload_manifest.py — Upload processed_manifest.csv to Kaggle
=============================================================
Uploads ONLY the manifest CSV as a new version of the existing dataset.
Does NOT touch the NIfTI files.

Usage:
    python3 upload_manifest.py          # update existing dataset version
    python3 upload_manifest.py --check  # just print manifest info, no upload

Run from anywhere — all paths are absolute.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
MANIFEST_CSV  = Path("/home/moamed/canada_me/explainable_diseas/implementation/outputs/processed_manifest.csv")
DATASET_ID    = "mohamedmohamed23/yale-processed-manifest"
KAGGLE_BIN    = "/home/moamed/canada_me/explainable_diseas/.venv/bin/kaggle"
KAGGLE_TOKEN  = "KGAT_409fa7f5d6a6c6be92cc91e56d173d21"

os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN


def run_kaggle(args: list, timeout: int = 120) -> tuple[bool, str]:
    result = subprocess.run(
        [KAGGLE_BIN] + args,
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "KAGGLE_API_TOKEN": KAGGLE_TOKEN},
    )
    return result.returncode == 0, result.stdout + result.stderr


def print_manifest_info():
    """Print info about the manifest without uploading."""
    if not MANIFEST_CSV.exists():
        print(f"❌  Manifest not found: {MANIFEST_CSV}")
        return

    import csv
    with open(MANIFEST_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n  File   : {MANIFEST_CSV}")
    print(f"  Size   : {MANIFEST_CSV.stat().st_size // 1024} KB")
    print(f"  Rows   : {len(rows)}")
    if rows:
        print(f"  Columns: {list(rows[0].keys())}")

    # Count key columns
    has_seg    = sum(1 for r in rows if r.get("path_seg", ""))
    has_label  = sum(1 for r in rows if r.get("label_seg", ""))
    complete   = sum(1 for r in rows if r.get("complete", "").lower() == "true")
    print(f"  complete=True         : {complete}")
    print(f"  With path_seg         : {has_seg}")
    print(f"  With label_seg        : {has_label}")

    # Check Kaggle current version
    ok, out = run_kaggle(["datasets", "status", DATASET_ID], timeout=15)
    print(f"\n  Kaggle dataset ({DATASET_ID}):")
    print(f"  Status: {out.strip() if ok else 'error: ' + out[:100]}")


def upload_manifest(dry_run: bool = False) -> bool:
    """Upload processed_manifest.csv to Kaggle as a new version."""

    if not MANIFEST_CSV.exists():
        print(f"❌  Manifest not found: {MANIFEST_CSV}")
        return False

    size_kb = MANIFEST_CSV.stat().st_size // 1024
    import csv
    with open(MANIFEST_CSV) as f:
        rows = sum(1 for _ in f) - 1  # subtract header

    print(f"Manifest : {MANIFEST_CSV.name}  ({size_kb} KB, {rows} rows)")
    print(f"Target   : https://www.kaggle.com/datasets/{DATASET_ID}")

    if dry_run:
        print("\n[--check mode] No upload performed.")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Stage file
        shutil.copy(MANIFEST_CSV, f"{tmpdir}/processed_manifest.csv")

        # Kaggle metadata
        meta = {
            "title": "Yale Brain Mets Processed Manifest",
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (Path(tmpdir) / "dataset-metadata.json").write_text(
            json.dumps(meta, indent=2)
        )

        # Check if dataset already exists
        print("\nChecking Kaggle dataset status...")
        ok, out = run_kaggle(["datasets", "status", DATASET_ID], timeout=15)
        dataset_exists = ok and "ready" in out.lower()

        version_msg = (
            f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
            f"{rows} rows, columns: "
            f"{'path_seg ' if _col_exists('path_seg') else ''}"
            f"{'label_seg ' if _col_exists('label_seg') else ''}"
            f"{'et_volume_mm3 ' if _col_exists('et_volume_mm3') else ''}"
        ).strip()

        if dataset_exists:
            print("Dataset exists → uploading as new version...")
            ok2, out2 = run_kaggle([
                "datasets", "version",
                "-p", tmpdir,
                "-m", version_msg,
                "--dir-mode", "zip",
            ])
        else:
            print("Dataset does not exist → creating new dataset...")
            ok2, out2 = run_kaggle([
                "datasets", "create",
                "-p", tmpdir,
                "--dir-mode", "zip",
            ])

        if ok2:
            print(f"\n✅  Manifest uploaded successfully!")
            print(f"   URL: https://www.kaggle.com/datasets/{DATASET_ID}")
            return True
        else:
            print(f"\n❌  Upload failed:")
            print(f"   {out2[:400]}")
            return False


def _col_exists(colname: str) -> bool:
    """Check if a column exists in the manifest."""
    import csv
    try:
        with open(MANIFEST_CSV) as f:
            return colname in next(csv.reader(f))
    except Exception:
        return False


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload manifest CSV to Kaggle")
    parser.add_argument("--check", action="store_true",
                        help="Print manifest info only — no upload")
    args = parser.parse_args()

    print("══════════════════════════════════════════════")
    print("  Yale Brain Mets — Manifest Uploader")
    print("══════════════════════════════════════════════")

    print_manifest_info()

    if not args.check:
        print()
        success = upload_manifest()
        sys.exit(0 if success else 1)
