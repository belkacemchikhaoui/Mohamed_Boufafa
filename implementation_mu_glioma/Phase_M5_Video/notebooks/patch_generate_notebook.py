"""
patch_generate_notebook.py
==========================
Applies two targeted fixes to Video_C_Generate.ipynb to eliminate
pixel-gap / flickering artifacts in the generated tumour-evolution videos.

FIX 1 — macro_block_size=1
  imageio was silently resizing every 500×500 frame to 512×512 to satisfy
  the H.264 macro-block constraint.  That lossy up-sample introduces subtle
  blurring / compression rings visible as "pixel gaps" at tumour boundaries.
  Passing macro_block_size=1 tells imageio to skip the resize entirely.

FIX 2 — Temporal majority-vote smoothing
  DDPM generates each interpolated frame *independently*; after argmax the
  hard per-pixel class decisions can flip 1–2 classes between consecutive
  frames, making the tumour look like it is randomly shrinking / growing.
  We apply a 3-frame sliding-window majority vote after building the full
  sequence, which removes single-frame class flips without blurring the
  spatial boundaries.

Usage:
  python patch_generate_notebook.py
  (run from the notebooks/ directory, or adjust NOTEBOOK_PATH below)
"""

import json
import re
import shutil
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "Video_C_Generate.ipynb"
BACKUP_PATH   = NOTEBOOK_PATH.with_suffix(".ipynb.bak")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

TEMPORAL_SMOOTH_CODE = [
    "# === FIX: temporal majority-vote smoothing (removes frame-to-frame flicker) ===\n",
    "from scipy import stats as sp_stats\n",
    "\n",
    "def temporal_smooth_segs(seg_list, window=3):\n",
    "    \"\"\"Majority-vote across a sliding temporal window to kill per-frame pixel flicker.\n",
    "    Each pixel keeps the class that wins the majority vote over [t-1, t, t+1].\n",
    "    \"\"\"\n",
    "    if len(seg_list) <= 2:\n",
    "        return seg_list\n",
    "    import numpy as _np\n",
    "    arr = _np.stack([s[0] for s in seg_list], axis=0)  # (T, H, W)\n",
    "    out = arr.copy()\n",
    "    n_cls = int(arr.max()) + 1\n",
    "    for t in range(len(arr)):\n",
    "        lo = max(0, t - window // 2)\n",
    "        hi = min(len(arr), t + window // 2 + 1)\n",
    "        win = arr[lo:hi]  # (k, H, W)\n",
    "        votes = _np.zeros((n_cls,) + arr.shape[1:], dtype=_np.int32)\n",
    "        for c in range(n_cls):\n",
    "            votes[c] = (win == c).sum(axis=0)\n",
    "        out[t] = _np.argmax(votes, axis=0).astype(arr.dtype)\n",
    "    return [(out[i], seg_list[i][1]) for i in range(len(seg_list))]\n",
    "\n",
]

APPLY_SMOOTH_CODE = [
    "    # === FIX: apply temporal smoothing before rendering ===\n",
    "    all_segs_ddpm = temporal_smooth_segs(all_segs_ddpm, window=3)\n",
    "    \n",
]

def source_has(source_lines, needle):
    return any(needle in line for line in source_lines)

def patch_source(source_lines):
    """Return patched source lines (list of strings)."""
    patched = []
    smooth_fn_inserted = False
    smooth_apply_inserted = False

    for i, line in enumerate(source_lines):

        # ---------------------------------------------------------------
        # FIX 2a: insert temporal_smooth_segs definition just before the
        #          per-patient loop so it is defined before first call.
        # ---------------------------------------------------------------
        if (not smooth_fn_inserted
                and "for pi,pid in enumerate(selected_pids):" in line):
            patched.extend(TEMPORAL_SMOOTH_CODE)
            smooth_fn_inserted = True

        patched.append(line)

        # ---------------------------------------------------------------
        # FIX 2b: apply temporal smoothing right after the inner frame-
        #          collection loop ends (all_segs_ddpm is fully built).
        # ---------------------------------------------------------------
        if (not smooth_apply_inserted
                and "all_segs_linear.append((seg_l,lbl))" in line):
            patched.extend(APPLY_SMOOTH_CODE)
            smooth_apply_inserted = True

        # ---------------------------------------------------------------
        # FIX 1: add macro_block_size=1 to both mimsave calls so FFmpeg
        #         does NOT up-sample 500×500 → 512×512.
        # ---------------------------------------------------------------
        if "imageio.mimsave(" in line and "macro_block_size" not in line:
            patched[-1] = line.rstrip("\n").replace(
                "fps=FPS)", "fps=FPS, macro_block_size=1)"
            ) + "\n"

    return patched


def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    # back up original
    shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"Backup saved → {BACKUP_PATH.name}")

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    total_cells_patched = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])

        # Only patch the cell that contains the video generation loop
        if not source_has(src, "for pi,pid in enumerate(selected_pids):"):
            continue

        # Guard: skip if already patched
        if source_has(src, "temporal_smooth_segs"):
            print("Cell already patched – nothing to do.")
            return

        cell["source"] = patch_source(src)
        total_cells_patched += 1

    if total_cells_patched == 0:
        print("WARNING: Target cell not found. Notebook may have a different structure.")
        return

    # Clear stale execution outputs so Kaggle re-runs cleanly
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NOTEBOOK_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nPatched {total_cells_patched} cell(s) successfully.")
    print("Changes applied:")
    print("  [FIX 1] macro_block_size=1 added to imageio.mimsave (stops 500→512 resize)")
    print("  [FIX 2] temporal_smooth_segs() defined + applied to all_segs_ddpm")
    print("\nNext steps:")
    print("  1. Upload the patched Video_C_Generate.ipynb to Kaggle")
    print("  2. Re-run the notebook (Run All)")
    print("  3. The pixel-gap/flickering artifacts should be gone from the new videos")


if __name__ == "__main__":
    main()
