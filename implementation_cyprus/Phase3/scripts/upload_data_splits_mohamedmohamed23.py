#!/usr/bin/env python3
"""
Upload the new balanced data_splits.json to Kaggle as a standalone dataset.

This uploads ONLY the data_splits.json (502 KB) — NOT the full MRI dataset.
The BSF training notebook (Cell 5) automatically finds it by scanning all
/kaggle/input directories, so just attach this dataset to the notebook.

Usage:
    python3 scripts/upload_data_splits_mohamedmohamed23.py

After upload:
    1. Go to your Kaggle BSF training notebook
    2. Click Add Data → Your Datasets → cypress-proteas-splits
    3. Run the notebook — Cell 5 will auto-detect the new 4-variable splits

Dataset created: mohamedmohamed23/cyprus-proteas-splits
"""

import os, json, subprocess, sys, tempfile, shutil
from pathlib import Path

# ── Account config (same as upload_fold_outputs_mohamedmohamed23.py) ──
ACCOUNT = {
    "username": "mohamedmohamed23",
    "key":      "KGAT_8111de93847141e6f93467431b823db0",
}

DATASET_SLUG  = "mohamedmohamed23/cyprus-proteas-splits"
DATASET_TITLE = "Cyprus PROTEAS — Balanced Data Splits (4-variable stratification)"

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# ── Locate the new data_splits.json ──────────────────────────────────
SPLITS_CANDIDATES = [
    PROJECT_DIR.parent / "Data" / "Cyprus-PROTEAS-zips" / "data_splits.json",
    PROJECT_DIR.parent / "Phase2" / "outputs" / "data_splits.json",
    PROJECT_DIR.parent / "outputs" / "data_splits.json",
]

splits_path = None
for c in SPLITS_CANDIDATES:
    if c.exists():
        splits_path = c
        break

if splits_path is None:
    print("❌ data_splits.json not found! Searched:")
    for c in SPLITS_CANDIDATES:
        print(f"   {c}")
    print("\nRun Phase2_A1_Data_Preparation.ipynb first to generate the new splits.")
    sys.exit(1)

# ── Verify the splits have 4-variable stratification ─────────────────
with open(splits_path) as f:
    splits_data = json.load(f)

meta = splits_data.get('metadata', {})
strat_vars = meta.get('stratification_variables', [])
low_pats   = meta.get('extreme_responders_LOW', [])

print(f"Splits file:  {splits_path}")
print(f"Size:         {splits_path.stat().st_size / 1024:.0f} KB")
print(f"Strat vars:   {strat_vars}")
print(f"LOW patients: {low_pats}")

if len(strat_vars) < 4:
    print()
    print("⚠️  WARNING: This looks like the OLD splits (only 2-variable stratification).")
    print("   Please run Phase2_A1_Data_Preparation.ipynb with the new code first.")
    answer = input("Upload anyway? [y/N] ").strip().lower()
    if answer != 'y':
        sys.exit(0)
else:
    print("✅ New 4-variable balanced splits confirmed")
    fold_3 = splits_data.get('3fold', {})
    low_set = set(low_pats)
    for fk in sorted(fold_3.keys()):
        val_grps = fold_3[fk].get('test_groups', [])
        n_low = sum(1 for g in val_grps if g in low_set)
        ok = '✅' if n_low >= 2 else '❌'
        print(f"  {fk}: LOW in val = {n_low}  {ok}")

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
    print("❌ kaggle CLI not found! Install with: pip install kaggle")
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
    print("Backed up kaggle.json → kaggle.json.bak")

os.environ['KAGGLE_API_TOKEN'] = ACCOUNT['key']
cmd_env = os.environ.copy()
cmd_env['KAGGLE_API_TOKEN'] = ACCOUNT['key']
print(f"Using account: {ACCOUNT['username']}")

try:
    # ── Build staging directory ───────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="kaggle_splits_"))

    # Copy only the splits file
    shutil.copy2(str(splits_path), str(staging / "data_splits.json"))

    # Write dataset metadata
    metadata = {
        "title":    DATASET_TITLE,
        "id":       DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nStaging directory: {staging}")
    for p in sorted(staging.rglob('*')):
        if p.is_file():
            print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

    total_kb = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) / 1024
    print(f"\nUploading to {DATASET_SLUG} ({total_kb:.0f} KB)...")

    # Try create first, then version
    result = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=300
    )

    combined = (result.stdout + result.stderr).lower()
    if result.returncode != 0 or any(x in combined for x in ["already in use", "already exists", "error"]):
        print("Dataset already exists → uploading as new version...")
        result = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging),
             "-m", "Balanced 4-variable splits (treatment×histology×visit_depth×tumor_burden)",
             "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=300
        )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    combined_result = (result.stdout + result.stderr).lower()
    if result.returncode == 0 and "error" not in combined_result:
        print(f"\n✅ Upload complete!")
        print(f"   Dataset: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"   Wait ~2 min for Kaggle to process, then:")
        print(f"\n   In your BSF Training notebook:")
        print(f"   1. Click 'Add Data' → 'Your Datasets'")
        print(f"   2. Search: 'cyprus-proteas-splits'")
        print(f"   3. Add it — Cell 5 auto-finds it via /kaggle/input scan")
    else:
        print(f"\n❌ Upload failed (exit {result.returncode})")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")

    shutil.rmtree(staging, ignore_errors=True)

finally:
    os.environ.pop('KAGGLE_API_TOKEN', None)
    if bak_file.exists():
        if creds_file.exists():
            creds_file.unlink()
        bak_file.rename(creds_file)
        print("\nRestored original kaggle.json")

print("\nDone!")
