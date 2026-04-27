"""
compare_roi_impact.py
=====================
Run this AFTER Phase3_A4_Embedding_Eval.ipynb to compare
Old BSF (no ROI crop) vs New BSF (with ROI crop) across all 16 tests.

Usage:
    cd /home/moamed/canada_me/explainable_diseas
    python3 implementation_cyprus/Phase3/scripts/compare_roi_impact.py
"""

import json
import numpy as np
from pathlib import Path

PHASE3   = Path("/home/moamed/canada_me/explainable_diseas/implementation_cyprus/Phase3")
OUTPUTS  = PHASE3 / "outputs"

OLD_PATH = OUTPUTS / "phase3_a4_results_OLD_no_roi_crop.json"
NEW_PATH = OUTPUTS / "phase3_a4_results.json"

# ── Thresholds (from hard-commit table) ─────────────────────────────────────
THRESHOLDS = {
    'M1_volume_R2':          ('R²',  0.10),
    'M2_log_volume_R2':      ('R²',  0.10),
    'M3_svr_R2':             ('R²',  0.05),
    'M4_necrosis_F1':        ('F1',  0.50),
    'M5_elongation_R2':      ('R²',  0.10),
    'M6_nn_consistency_pct': ('%',   70.0),
    'H1_pca_residual_r':     ('r',   0.30),
    'H2_glcm_r':             ('r',   0.30),
    'H3_et_volume_r':        ('r',   0.30),
    'H4_texture_r':          ('r',   0.30),
    'T1_dist_vol_r':         ('r',   0.30),
    'T3_delta_r':            ('r',   0.20),
    'T4_response_auc':       ('AUC', 0.60),
    'T5_coherence_r':        ('r',   0.50),
    'T6_velocity_r':         ('r',   0.30),
    'T7_treatment_effect':   ('effect', 0.10),
}

LABELS = {
    'M1_volume_R2':          'M1  Volume R²',
    'M2_log_volume_R2':      'M2  Log-volume R²',
    'M3_svr_R2':             'M3  Surface-vol ratio R²',
    'M4_necrosis_F1':        'M4  Necrosis F1',
    'M5_elongation_R2':      'M5  Elongation R²',
    'M6_nn_consistency_pct': 'M6  NN Consistency %',
    'H1_pca_residual_r':     'H1  PCA Residual r',
    'H2_glcm_r':             'H2  GLCM r',
    'H3_et_volume_r':        'H3  ET Volume r',
    'H4_texture_r':          'H4  Texture r',
    'T1_dist_vol_r':         'T1  Dist-vol r',
    'T3_delta_r':            'T3  ΔVol r',
    'T4_response_auc':       'T4  Response AUC',
    'T5_coherence_r':        'T5  Coherence r',
    'T6_velocity_r':         'T6  Velocity r',
    'T7_treatment_effect':   'T7  Treatment effect',
}


def load(path):
    if not path.exists():
        return None
    return json.load(open(path))


def passes(val, key):
    if key not in THRESHOLDS:
        return None
    _, thr = THRESHOLDS[key]
    return val >= thr


def fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '   N/A '
    return f'{val:+.3f}'


old = load(OLD_PATH)
new = load(NEW_PATH)

if new is None:
    print("❌ New results not found. Run Phase3_A4_Embedding_Eval.ipynb first.")
    exit(1)

if old is None:
    print("⚠️  Old results not found — showing new results only.\n")

print()
print("=" * 72)
print("  BSF EMBEDDING EVALUATION: OLD (no ROI) vs NEW (ROI crop)")
print("=" * 72)
print(f"  {'Test':<28} {'OLD':>8} {'NEW':>8} {'Delta':>8}  {'New Pass?':>10}")
print("-" * 72)

total_old_pass = 0
total_new_pass = 0
n_tests = 0

groups = [
    ('── Morphology M1–M6 ──', [k for k in THRESHOLDS if k.startswith('M')]),
    ('── Heterogeneity H1–H4 ──', [k for k in THRESHOLDS if k.startswith('H')]),
    ('── Temporal T1–T7 ──', [k for k in THRESHOLDS if k.startswith('T')]),
]

for group_title, keys in groups:
    print(f"\n  {group_title}")
    for key in keys:
        label = LABELS.get(key, key)
        new_val = new.get(key)
        old_val = old.get(key) if old else None

        if new_val is None:
            continue
        n_tests += 1

        new_v = float(new_val) if new_val is not None else None
        old_v = float(old_val) if old_val is not None else None

        delta = (new_v - old_v) if (new_v is not None and old_v is not None) else None
        new_p = passes(new_v, key)
        old_p = passes(old_v, key) if old_v is not None else None

        if new_p:
            total_new_pass += 1
        if old_p:
            total_old_pass += 1

        new_str  = fmt(new_v)
        old_str  = fmt(old_v) if old else '   N/A '
        delta_str = f'{delta:+.3f}' if delta is not None else '    —  '

        pass_icon = '✅ PASS' if new_p else '❌ FAIL'
        # Highlight improvements
        improvement = ''
        if delta is not None and delta > 0.01:
            improvement = ' ↑'
        elif delta is not None and delta < -0.01:
            improvement = ' ↓'

        print(f"  {label:<28} {old_str:>8} {new_str:>8} {delta_str:>8}  {pass_icon}{improvement}")

print()
print("=" * 72)
if old:
    print(f"  OLD score: {total_old_pass}/{n_tests}  →  NEW score: {total_new_pass}/{n_tests}")
    gain = total_new_pass - total_old_pass
    sign = '+' if gain >= 0 else ''
    print(f"  Net change: {sign}{gain} tests ({sign}{gain/n_tests*100:.0f}%)")
else:
    print(f"  NEW score: {total_new_pass}/{n_tests}")
print("=" * 72)
print()

# Expected impact of ROI crop
if old:
    m4_old = old.get('M4_necrosis_F1')
    m4_new = new.get('M4_necrosis_F1')
    if m4_old and m4_new:
        print(f"  M4 necrosis F1:  {float(m4_old):.3f} → {float(m4_new):.3f}  (expected +0.10–0.20)")
    h3_old = old.get('H3_et_volume_r')
    h3_new = new.get('H3_et_volume_r')
    if h3_old and h3_new:
        print(f"  H3 ET volume r:  {float(h3_old):.3f} → {float(h3_new):.3f}  (expected +0.05–0.15)")
    h4_old = old.get('H4_texture_r')
    h4_new = new.get('H4_texture_r')
    if h4_old and h4_new:
        print(f"  H4 texture r:  {float(h4_old):.3f} → {float(h4_new):.3f}  (expected +0.05–0.15)")
    print()
