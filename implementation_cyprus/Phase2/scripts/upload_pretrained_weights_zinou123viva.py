#!/usr/bin/env python3
"""
Upload BraTS 2021 pretrained nnUNet v2 weights to zinou123viva Kaggle account.

Creates dataset: zinou123viva/brats2024-pretrained-weights
Contents:
  nnunet_v2/Dataset002_BRATS19/nnUNetTrainer__nnUNetPlans__3d_fullres/
    plans.json
    fold_0/checkpoint_final.pth

Usage:
  python3 upload_pretrained_weights_zinou123viva.py

What to attach to the Cyprus C2 notebook on Kaggle:
  Input dataset: zinou123viva/brats2024-pretrained-weights
  Input dataset: zinou123viva/cyprus-proteas-brain-mets  (already there)
"""

import os, json, shutil, subprocess, sys, tempfile
from pathlib import Path

# ── Credentials (from implementation_cyprus/.kaggle_accounts.json) ──
ACCOUNT = {
    "username": "zinou123viva",
    "key":      "KGAT_9811f85e7078d6bbb85aad26ae841bdc",
}

# ── Source: same pretrained weights used for BraTS 2024 training ──
MODELS_DIR = Path("/home/moamed/HDD/brats2024_posttreatment/models")
TRAINER_DIR = MODELS_DIR / "nnU-Net_v2 " / "Dataset002_BRATS19" / "nnUNetTrainer__nnUNetPlans__3d_fullres"

DATASET_SLUG  = f"{ACCOUNT['username']}/brats2024-pretrained-weights"
DATASET_TITLE = "BraTS 2024 Pretrained Weights"


def find_kaggle_bin():
    for c in [
        shutil.which("kaggle"),
        os.path.expanduser("~/.local/bin/kaggle"),
        "/home/moamed/miniconda3/bin/kaggle",
        "/home/moamed/miniconda3/envs/datapre/bin/kaggle",
    ]:
        if c and os.path.isfile(c):
            return c
    return None


def push_dataset(staging, slug, msg, cmd_env):
    kaggle_bin = find_kaggle_bin()
    if not kaggle_bin:
        print("❌ kaggle CLI not found"); sys.exit(1)

    r = subprocess.run(
        [kaggle_bin, "datasets", "create", "-p", str(staging), "--dir-mode", "zip"],
        capture_output=True, text=True, env=cmd_env, timeout=7200)
    combo = (r.stdout + r.stderr).lower()
    # Only fall back to 'version' if the dataset genuinely already exists
    if r.returncode != 0 and ("already exist" in combo or "403" in combo):
        print("  Dataset exists → pushing new version...")
        r = subprocess.run(
            [kaggle_bin, "datasets", "version", "-p", str(staging), "-m", msg, "--dir-mode", "zip"],
            capture_output=True, text=True, env=cmd_env, timeout=7200)
    print(r.stdout)
    if r.stderr: print(r.stderr)
    ok = r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()
    print(f"  {'✅' if ok else '❌'} https://www.kaggle.com/datasets/{slug}")
    return ok


def upload_weights():
    print(f"\n{'='*60}")
    print(f"  Uploading pretrained weights → {DATASET_SLUG}")
    print(f"{'='*60}")

    if not TRAINER_DIR.exists():
        print(f"❌ Trainer dir not found: {TRAINER_DIR}")
        print("   Check path to nnU-Net_v2 models directory.")
        sys.exit(1)

    staging = Path(tempfile.mkdtemp(prefix="kg_weights_zinou_"))
    try:
        dst = staging / "nnunet_v2" / "Dataset002_BRATS19" / TRAINER_DIR.name
        dst.mkdir(parents=True, exist_ok=True)

        # ── plans.json + dataset.json ──
        for fname in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
            src = TRAINER_DIR / fname
            if src.exists():
                shutil.copy2(str(src), dst / fname)
                print(f"  ✅ {fname}")
            else:
                print(f"  ⚠️  Missing {fname} (not critical)")

        # ── fold checkpoints ──
        n_folds = 0
        for fold in range(5):
            ckpt = TRAINER_DIR / f"fold_{fold}" / "checkpoint_final.pth"
            if ckpt.exists():
                fold_dst = dst / f"fold_{fold}"
                fold_dst.mkdir(exist_ok=True)
                shutil.copy2(str(ckpt), fold_dst / "checkpoint_final.pth")
                size_mb = ckpt.stat().st_size / 1e6
                print(f"  ✅ nnU-Net fold_{fold}: {size_mb:.0f} MB")
                n_folds += 1

        if n_folds == 0:
            print("❌ No checkpoint_final.pth files found!")
            sys.exit(1)

        # ── dataset-metadata.json ──
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "title": DATASET_TITLE,
            "id":    DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}]
        }, indent=2))

        total_gb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1e9
        print(f"\n  Total: {total_gb:.2f} GB | {n_folds} fold(s) + plans")

        # ── Auth setup exactly like upload_cnn_fold_outputs_zinou123viva.py ──
        kaggle_dir = Path.home() / ".kaggle"
        creds_file = kaggle_dir / "kaggle.json"
        bak_file   = kaggle_dir / "kaggle.json.bak"
        kaggle_dir.mkdir(exist_ok=True)
        if creds_file.exists():
            shutil.copy2(creds_file, bak_file)
            creds_file.rename(bak_file)
            print("  Auth: Backed up kaggle.json -> kaggle.json.bak")
        
        try:
            cmd_env = os.environ.copy()
            cmd_env["KAGGLE_API_TOKEN"] = ACCOUNT["key"]
            # Clear old variables to avoid conflicts
            cmd_env.pop("KAGGLE_USERNAME", None)
            cmd_env.pop("KAGGLE_KEY", None)

            hdd_tmp = Path("/home/moamed/HDD/kaggle_staging/tmp")
            hdd_tmp.mkdir(parents=True, exist_ok=True)
            cmd_env["TMPDIR"] = str(hdd_tmp)
            cmd_env["TEMP"]   = str(hdd_tmp)
            cmd_env["TMP"]    = str(hdd_tmp)

            print(f"\n  Uploading as: {ACCOUNT['username']} using KAGGLE_API_TOKEN")
            result = push_dataset(staging, DATASET_SLUG,
                                  f"nnUNet v2 {n_folds}-fold BraTS 2021", cmd_env)
        finally:
            if bak_file.exists():
                if creds_file.exists(): creds_file.unlink()
                bak_file.rename(creds_file)
                print(f"  Auth: restored original kaggle.json")
        return result

    finally:
        shutil.rmtree(staging, ignore_errors=True)
        print("  Staging cleaned up.")


if __name__ == "__main__":
    ok = upload_weights()

    if ok:
        print(f"""
{'='*60}
  ✅ Upload complete!
{'='*60}

  Dataset URL: https://www.kaggle.com/datasets/{DATASET_SLUG}

  To run Phase2_C2_nnUNet_Cyprus_Finetune.ipynb on Kaggle:
  ─────────────────────────────────────────────────────────
  1. Open the notebook on Kaggle (upload the .ipynb file)
  2. Click  Settings → Add Data  and attach:
       ✅ zinou123viva/brats2024-pretrained-weights   ← just uploaded
       ✅ zinou123viva/cyprus-proteas-brain-mets      ← already there
  3. Enable T4 GPU (Settings → Accelerator → GPU T4 x1)
  4. Enable Internet (Settings → Internet → On)
  5. Session type: 12 hours
  6. Click Run All
{'='*60}
""")
    else:
        print("❌ Upload failed — check errors above.")
