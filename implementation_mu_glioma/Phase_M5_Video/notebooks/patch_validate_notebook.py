"""
patch_validate_notebook.py
==========================
Applies the macro_block_size=1 fix to Video_D_BraTS_Validate.ipynb.

The validation notebook generates videos for 109 BraTS-PTG patients and has
the same FFmpeg 500×512 macroblock resize issue as the generation notebook.
Temporal smoothing is NOT applied here because validation evaluates single
triplets against ground truth — sequence-level smoothing would artificially
inflate Dice scores.
"""

import json
import shutil
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent.parent
    / "Phase_M5_Validation/notebooks/Video_D_BraTS_Validate.ipynb"
)
BACKUP_PATH = NOTEBOOK_PATH.with_suffix(".ipynb.bak")


def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"Backup saved → {BACKUP_PATH.name}")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    n_fixed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        new_src = []
        for line in src:
            if "imageio.mimsave(" in line and "macro_block_size" not in line:
                line = line.replace("fps=FPS)", "fps=FPS, macro_block_size=1)")
                n_fixed += 1
            new_src.append(line)
        cell["source"] = new_src

    if n_fixed == 0:
        print("No imageio.mimsave calls found (or already patched).")
        return

    # Clear stale outputs
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NOTEBOOK_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Fixed {n_fixed} imageio.mimsave call(s) with macro_block_size=1")
    print("\nNext steps:")
    print("  1. Upload patched Video_D_BraTS_Validate.ipynb to Kaggle")
    print("  2. Re-run the notebook")


if __name__ == "__main__":
    main()
