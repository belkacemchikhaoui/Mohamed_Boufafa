"""
GLCM Feature Extraction from PROTEAS-MRI_radiomics_data.xlsx
=============================================================
Run with:  conda run -n datapre python3 extract_glcm_from_xlsx.py

Extracts 3 GLCM features from mask_tumor × t1c modality for all patients:
  - glcm_DifferenceEntropy
  - glcm_Contrast
  - glcm_ClusterShade

Split patients (P04a/b, P07a/b, P17a/b, P20a/b, P23a/b) treated as
separate cases with their sheet name as the patient key.

Saves: Phase3/bsf_fold_outputs/embeddings_hybrid/glcm_features.npz
       Key format: {sheet_name}_{tp_normalized}
       e.g. P01_baseline, P01_fu1, P04a_fu2 ...
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path("/home/moamed/canada_me/explainable_diseas")
IMPL = ROOT / "implementation_cyprus"
XLS  = IMPL / "Data" / "Cyprus-PROTEAS-zips" / "PROTEAS-MRI_radiomics_data.xlsx"
OUT  = IMPL / "Phase3" / "bsf_fold_outputs" / "embeddings_hybrid"
OUT.mkdir(parents=True, exist_ok=True)

# ── Timepoint name normalisation ──────────────────────────────────────────────
TP_MAP = {
    'baseline':    'baseline',
    'Baseline':    'baseline',
    'follow_up_1': 'fu1',
    'follow_up_2': 'fu2',
    'follow_up_3': 'fu3',
    'follow_up_4': 'fu4',
    'follow_up_5': 'fu5',
    'follow_up_6': 'fu6',
}

# Target: mask_tumor, modality t1c, GLCM features
MASK = 'mask_tumor'
MOD  = 't1c'
GLCM_FEATS = ['DifferenceEntropy', 'Contrast', 'ClusterShade']

xl = pd.ExcelFile(XLS)
print(f"Excel has {len(xl.sheet_names)} sheets: {xl.sheet_names}")

results = {}
missing = []

for pid in xl.sheet_names:
    df = xl.parse(pid)
    if 'RadiomicsFeature' not in df.columns:
        print(f"  ⚠️  {pid}: unexpected columns {df.columns.tolist()}")
        continue

    # Parse feature components
    def parse_row(feat_str):
        parts = str(feat_str).split('__')
        if len(parts) < 4:
            return None, None, None, None
        return parts[0], parts[1], parts[2], '__'.join(parts[3:])

    df[['mask','mod','tp_raw','feat']] = pd.DataFrame(
        [parse_row(r) for r in df['RadiomicsFeature']],
        columns=['mask','mod','tp_raw','feat']
    )

    # Filter: mask_tumor + t1c
    sub = df[(df['mask'] == MASK) & (df['mod'] == MOD)].copy()
    if sub.empty:
        # Fallback to mask_all if mask_tumor not available
        sub = df[(df['mask'] == 'mask_all') & (df['mod'] == MOD)].copy()
        if sub.empty:
            print(f"  ⚠️  {pid}: no mask_tumor/mask_all + t1c rows found")
            continue
        print(f"  ⚠️  {pid}: using mask_all (mask_tumor not available)")

    # Get unique timepoints
    for tp_raw in sub['tp_raw'].unique():
        tp_norm = TP_MAP.get(tp_raw, tp_raw.replace('follow_up_', 'fu'))
        key = f"{pid}_{tp_norm}"

        tp_sub = sub[sub['tp_raw'] == tp_raw].copy()
        tp_sub = tp_sub.set_index('feat')['RadiomicsValue']

        row = []
        all_ok = True
        for gfeat in GLCM_FEATS:
            feat_key = f"original_glcm_{gfeat}"
            if feat_key in tp_sub.index:
                val = float(tp_sub[feat_key])
                if np.isnan(val):
                    all_ok = False
                    val = 0.0
                row.append(val)
            else:
                print(f"  ⚠️  {key}: missing {feat_key}")
                row.append(np.nan)
                all_ok = False

        results[key] = np.array(row, dtype=np.float32)
        status = '✅' if all_ok else '⚠️'
        print(f"  {status} {key}: DiffEnt={row[0]:.4f}  Contrast={row[1]:.4f}  ClusterShade={row[2]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT / "glcm_features.npz"
np.savez(out_path, **results)

print(f"\n{'='*60}")
print(f"✅ Saved {len(results)} GLCM feature vectors → {out_path}")
print(f"   Features: {GLCM_FEATS}")
print(f"   Mask: {MASK}  |  Modality: {MOD}")

# Show coverage vs timeline
try:
    df_tl = pd.read_csv(IMPL / "Phase1/outputs/cyprus_patient_timelines.csv")
    df_tl = df_tl[df_tl['has_mask'] == True]
    tl_keys = set(f"{r.patient_id}_{r.visit_name}" for _, r in df_tl.iterrows())
    glcm_keys = set(results.keys())
    matched = tl_keys & glcm_keys
    unmatched = tl_keys - glcm_keys
    print(f"\n   Timeline scans: {len(tl_keys)}")
    print(f"   GLCM matched:   {len(matched)}")
    print(f"   Unmatched:      {len(unmatched)} — {sorted(unmatched)[:5]}{'...' if len(unmatched)>5 else ''}")
except Exception as e:
    print(f"Coverage check failed: {e}")
