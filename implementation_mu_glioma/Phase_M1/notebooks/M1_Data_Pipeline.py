#!/usr/bin/env python3
"""
Phase M1 — MU-Glioma Data Pipeline
====================================
Builds the complete data infrastructure for treatment-aware TaViT V3.

Outputs (to Phase_M1/outputs/):
  - scan_index.json            — BraTS-compatible per-scan file index
  - longitudinal_index.json    — Per-patient trajectory with treatment phases
  - mu_glioma_master.csv       — Master table: 1 row per scan with volumes, treatment tokens, clinical data
  - tumor_volumes.csv          — BraTS-compatible volume CSV  
  - treatment_tokens.csv       — 15-D treatment context per scan
  - viz_treatment_timelines.png
  - viz_volume_distributions.png
  - viz_treatment_phase_eda.png
  - viz_trajectory_classes.png

Usage:
  python M1_Data_Pipeline.py
"""

import os, sys, json, re, warnings
import numpy as np
import pandas as pd
import nibabel as nib
import openpyxl
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
MU_ROOT     = "/home/moamed/HDD/validation_glomia/PKG - MU-Glioma-Post/MU-Glioma-Post"
CLINICAL_XL = "/home/moamed/HDD/validation_glomia/MU-Glioma-Post_ClinicalData-July2025.xlsx"
VOLUMES_XL  = "/home/moamed/HDD/validation_glomia/MU-Glioma-Post_Segmentation_Volumes.xlsx"
OUT_DIR     = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[M1] MU-Glioma Data Pipeline")
print(f"[M1] Imaging root: {MU_ROOT}")
print(f"[M1] Clinical data: {CLINICAL_XL}")
print(f"[M1] Output dir: {OUT_DIR}")
print()

# ============================================================
# STEP 1: BUILD SCAN INDEX
# ============================================================
print("=" * 70)
print("STEP 1: Building scan index")
print("=" * 70)

scan_records = []
patient_ids = sorted([d for d in os.listdir(MU_ROOT) 
                       if os.path.isdir(os.path.join(MU_ROOT, d))])

for pid in patient_ids:
    pdir = os.path.join(MU_ROOT, pid)
    timepoints = sorted([d for d in os.listdir(pdir) 
                         if os.path.isdir(os.path.join(pdir, d))])
    
    for tp in timepoints:
        tpdir = os.path.join(pdir, tp)
        tp_num = int(tp.replace('Timepoint_', ''))
        scan_id = f"{pid}_{tp}"
        
        # Find modality files
        files = os.listdir(tpdir)
        paths = {}
        for f in files:
            fpath = os.path.join(tpdir, f)
            if 'brain_t1c' in f:
                paths['t1c'] = fpath
            elif 'brain_t1n' in f:
                paths['t1n'] = fpath
            elif 'brain_t2f' in f:
                paths['t2f'] = fpath
            elif 'brain_t2w' in f:
                paths['t2w'] = fpath
            elif 'tumorMask' in f:
                paths['mask'] = fpath
        
        # Verify completeness
        expected = ['t1c', 't1n', 't2f', 't2w', 'mask']
        missing = [k for k in expected if k not in paths]
        
        scan_records.append({
            'scan_id': scan_id,
            'patient_id': pid,
            'timepoint': tp_num,
            'timepoint_str': tp,
            'scan_dir': tpdir,
            'path_t1c': paths.get('t1c', ''),
            'path_t1n': paths.get('t1n', ''),
            'path_t2f': paths.get('t2f', ''),
            'path_t2w': paths.get('t2w', ''),
            'path_mask': paths.get('mask', ''),
            'complete': len(missing) == 0,
            'missing_mods': ','.join(missing) if missing else '',
        })

df_scans = pd.DataFrame(scan_records)
print(f"  Total scans: {len(df_scans)}")
print(f"  Total patients: {df_scans['patient_id'].nunique()}")
print(f"  Complete scans: {df_scans['complete'].sum()}")
print(f"  Incomplete: {(~df_scans['complete']).sum()}")

# Save scan_index.json (BraTS-compatible format)
scan_index = {
    'dataset': 'MU-Glioma-Post',
    'label_convention': {
        '1': 'Necrotic Tumor Core (NETC)',
        '2': 'Tumor Infiltration and Edema (SNFH)',
        '3': 'Enhancing Tumor Core (ET)',
        '4': 'Resection Cavity (RC)',
    },
    'sub_regions': {
        'WT': 'labels 1+2+3 (Whole Tumor)',
        'TC': 'labels 1+3 (Tumor Core)',
        'ET': 'label 3 (Enhancing Tumor)',
        'RC': 'label 4 (Resection Cavity)',
    },
    'scans': {},
}
for _, row in df_scans.iterrows():
    scan_index['scans'][row['scan_id']] = {
        'patient_id': row['patient_id'],
        'timepoint': row['timepoint'],
        'paths': {
            't1c': row['path_t1c'],
            't1n': row['path_t1n'],
            't2f': row['path_t2f'],
            't2w': row['path_t2w'],
            'mask': row['path_mask'],
        },
        'complete': row['complete'],
    }

with open(OUT_DIR / 'scan_index.json', 'w') as f:
    json.dump(scan_index, f, indent=2)
print(f"  Saved: scan_index.json")

# ============================================================
# STEP 2: PARSE CLINICAL DATA
# ============================================================
print()
print("=" * 70)
print("STEP 2: Parsing clinical Excel data")
print("=" * 70)

wb = openpyxl.load_workbook(CLINICAL_XL, read_only=True)
ws = wb['MU Glioma Post']
rows = list(ws.iter_rows(values_only=True))
headers = rows[0]
data_rows = rows[1:]

# Column index mapping (explicit for reproducibility)
COL = {
    'patient_id':           0,
    'sex':                  1,
    'race':                 2,
    'age_at_diagnosis':     3,
    'primary_diagnosis':    4,
    'grade':                5,
    'stereotactic_biopsy':  6,
    'progression':          7,
    'days_to_1st_prog':     8,
    'type_1st_prog':        9,
    'second_prog':          10,
    'type_2nd_prog':        11,
    'multiple_surgeries':   12,
    'hospice':              13,
    'overall_survival':     14,
    'days_to_death':        15,
    'idh1':                 16,
    'idh2':                 17,
    '1p19q':                18,
    'atrx':                 19,
    'mgmt':                 20,
    'braf':                 21,
    'tert':                 22,
    'chr7_10':              23,
    'h3_3a':                24,
    'egfr':                 25,
    'pten':                 26,
    'cdkn2a_b':             27,
    'tp53':                 28,
    'other_mutations':      29,
    'days_to_surgery':      34,
    'chemo':                35,
    'chemo_name':           36,
    'days_to_chemo_start':  37,
    'days_to_chemo_end':    38,
    'radiation':            39,
    'days_to_radio_start':  40,
    'days_to_radio_end':    41,
    'radiation_dose':       42,
    'radiation_fractions':  43,
    'days_to_1st_prog_v2':  44,
    'days_to_further_prog': 45,
    'treat_after_2nd_prog': 46,
    'days_to_new_treat':    47,
    'additional_therapy':   48,
    'add_therapy_cycle_len':49,
    'days_to_add_start':    50,
    'days_to_add_end':      51,
    'n_add_cycles':         52,
    'add_therapy_2':        53,
    'add2_cycle_len':       54,
    'days_to_add2_start':   55,
    'days_to_add2_end':     56,
    'n_add2_cycles':        57,
    'immunotherapy':        58,
    'immuno_cycle_len':     59,
    'days_to_immuno_start': 60,
    'days_to_immuno_end':   61,
    'n_immuno_cycles':      62,
    'brachytherapy':        63,
    'days_to_brachy':       64,
    'other_therapy':        65,
    'days_to_other_start':  66,
    'days_to_other_end':    67,
    'days_to_mri_tp1':      68,
    'days_to_mri_tp2':      69,
    'days_to_mri_tp3':      70,
    'days_to_mri_tp4':      71,
    'days_to_mri_tp5':      72,
    'days_to_mri_tp6':      73,
}

def safe_val(row, key, default=None):
    """Extract a value from a row, handling NA/None."""
    idx = COL[key]
    if idx >= len(row):
        return default
    v = row[idx]
    if v is None or str(v).strip() in ['', 'NA', 'N/A', 'LTF', 'na']:
        return default
    return v

def safe_float(row, key, default=np.nan):
    v = safe_val(row, key)
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def safe_int(row, key, default=None):
    v = safe_val(row, key)
    if v is None:
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default

def parse_dose(row):
    """Parse radiation dose string like '60 Gy' → 60.0"""
    v = safe_val(row, 'radiation_dose')
    if v is None:
        return np.nan
    s = str(v).strip().lower().replace('gy', '').strip()
    try:
        return float(s)
    except:
        return np.nan

# Build clinical DataFrame
clinical_records = []
for row in data_rows:
    pid = safe_val(row, 'patient_id')
    if pid is None:
        continue
    
    # Parse grade
    grade_raw = safe_val(row, 'grade')
    try:
        grade = int(float(grade_raw)) if grade_raw is not None else np.nan
    except:
        grade = np.nan
    
    rec = {
        'patient_id':        pid,
        'sex':               safe_val(row, 'sex', ''),
        'race':              safe_val(row, 'race', ''),
        'age_at_diagnosis':  safe_float(row, 'age_at_diagnosis'),
        'primary_diagnosis': safe_val(row, 'primary_diagnosis', ''),
        'grade':             grade,
        # Molecular markers
        'idh1':              safe_int(row, 'idh1'),
        'idh2':              safe_int(row, 'idh2'),
        'mgmt':              safe_int(row, 'mgmt'),
        'atrx':              safe_int(row, 'atrx'),
        '1p19q':             safe_int(row, '1p19q'),
        'tert':              safe_int(row, 'tert'),
        'egfr':              safe_int(row, 'egfr'),
        'pten':              safe_int(row, 'pten'),
        'tp53':              safe_int(row, 'tp53'),
        # Clinical outcomes
        'progression':       safe_int(row, 'progression'),
        'days_to_1st_prog':  safe_float(row, 'days_to_1st_prog'),
        'overall_survival':  safe_int(row, 'overall_survival'),
        'days_to_death':     safe_float(row, 'days_to_death'),
        # Surgery
        'days_to_surgery':   safe_float(row, 'days_to_surgery'),
        'multiple_surgeries':safe_val(row, 'multiple_surgeries'),
        # Chemotherapy
        'chemo':             safe_val(row, 'chemo'),
        'chemo_name':        safe_val(row, 'chemo_name', ''),
        'days_to_chemo_start': safe_float(row, 'days_to_chemo_start'),
        'days_to_chemo_end':   safe_float(row, 'days_to_chemo_end'),
        # Radiation
        'radiation':         safe_val(row, 'radiation'),
        'days_to_radio_start': safe_float(row, 'days_to_radio_start'),
        'days_to_radio_end':   safe_float(row, 'days_to_radio_end'),
        'radiation_dose_gy':   parse_dose(row),
        'radiation_fractions': safe_float(row, 'radiation_fractions'),
        # Additional therapy
        'additional_therapy':  safe_val(row, 'additional_therapy', ''),
        'days_to_add_start':   safe_float(row, 'days_to_add_start'),
        'days_to_add_end':     safe_float(row, 'days_to_add_end'),
        # Immunotherapy (Avastin)
        'immunotherapy':       safe_val(row, 'immunotherapy', ''),
        'days_to_immuno_start': safe_float(row, 'days_to_immuno_start'),
        'days_to_immuno_end':   safe_float(row, 'days_to_immuno_end'),
        # MRI timing
        'days_to_mri_tp1':   safe_float(row, 'days_to_mri_tp1'),
        'days_to_mri_tp2':   safe_float(row, 'days_to_mri_tp2'),
        'days_to_mri_tp3':   safe_float(row, 'days_to_mri_tp3'),
        'days_to_mri_tp4':   safe_float(row, 'days_to_mri_tp4'),
        'days_to_mri_tp5':   safe_float(row, 'days_to_mri_tp5'),
        'days_to_mri_tp6':   safe_float(row, 'days_to_mri_tp6'),
    }
    clinical_records.append(rec)

df_clinical = pd.DataFrame(clinical_records)
# Filter to imaging patients only
imaging_pids = set(df_scans['patient_id'].unique())
df_clinical = df_clinical[df_clinical['patient_id'].isin(imaging_pids)].copy()
print(f"  Clinical records (imaging patients): {len(df_clinical)}")

# Save clinical CSV
df_clinical.to_csv(OUT_DIR / 'clinical_data.csv', index=False)
print(f"  Saved: clinical_data.csv")

# ============================================================
# STEP 3: EXTRACT TUMOR VOLUMES FROM GT MASKS
# ============================================================
print()
print("=" * 70)
print("STEP 3: Extracting tumor volumes from GT masks")
print("=" * 70)

volume_records = []
total = len(df_scans)
for i, (_, row) in enumerate(df_scans.iterrows()):
    if not row['complete']:
        volume_records.append({
            'scan_id': row['scan_id'],
            'patient_id': row['patient_id'],
            'timepoint': row['timepoint'],
            'wt_vol_ml': np.nan, 'tc_vol_ml': np.nan,
            'et_vol_ml': np.nan, 'rc_vol_ml': np.nan,
            'wt_voxels': np.nan, 'tc_voxels': np.nan,
            'et_voxels': np.nan, 'rc_voxels': np.nan,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
        })
        continue
    
    mask_path = row['path_mask']
    try:
        img = nib.load(mask_path)
        mask = img.get_fdata().astype(np.int16)
        voxel_vol = np.prod(img.header.get_zooms()[:3])  # mm³ per voxel
        
        # BraTS label convention for MU-Glioma:
        # 1 = Necrotic Tumor Core (NETC)
        # 2 = Tumor Infiltration & Edema (SNFH)
        # 3 = Enhancing Tumor Core (ET)
        # 4 = Resection Cavity (RC)
        netc = (mask == 1)
        snfh = (mask == 2)
        et   = (mask == 3)
        rc   = (mask == 4)
        
        wt = netc | snfh | et      # Whole Tumor (no RC)
        tc = netc | et             # Tumor Core
        
        wt_vox = int(wt.sum())
        tc_vox = int(tc.sum())
        et_vox = int(et.sum())
        rc_vox = int(rc.sum())
        
        # Centroid of whole tumor
        if wt_vox > 0:
            coords = np.array(np.where(wt))
            cx, cy, cz = coords.mean(axis=1)
        else:
            cx = cy = cz = np.nan
        
        volume_records.append({
            'scan_id': row['scan_id'],
            'patient_id': row['patient_id'],
            'timepoint': row['timepoint'],
            'wt_vol_ml': wt_vox * voxel_vol / 1000.0,  # mm³ → mL
            'tc_vol_ml': tc_vox * voxel_vol / 1000.0,
            'et_vol_ml': et_vox * voxel_vol / 1000.0,
            'rc_vol_ml': rc_vox * voxel_vol / 1000.0,
            'wt_voxels': wt_vox,
            'tc_voxels': tc_vox,
            'et_voxels': et_vox,
            'rc_voxels': rc_vox,
            'centroid_x': cx,
            'centroid_y': cy,
            'centroid_z': cz,
        })
    except Exception as e:
        print(f"  ⚠ CORRUPT FILE: {row['scan_id']} — {e}")
        volume_records.append({
            'scan_id': row['scan_id'],
            'patient_id': row['patient_id'],
            'timepoint': row['timepoint'],
            'wt_vol_ml': np.nan, 'tc_vol_ml': np.nan,
            'et_vol_ml': np.nan, 'rc_vol_ml': np.nan,
            'wt_voxels': np.nan, 'tc_voxels': np.nan,
            'et_voxels': np.nan, 'rc_voxels': np.nan,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'centroid_z': np.nan,
        })
    
    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"  [{i+1}/{total}] Processed {row['scan_id']}")

df_volumes = pd.DataFrame(volume_records)
df_volumes.to_csv(OUT_DIR / 'tumor_volumes.csv', index=False)
print(f"  Saved: tumor_volumes.csv")
print(f"  Median WT volume: {df_volumes['wt_vol_ml'].median():.2f} mL")
print(f"  Median ET volume: {df_volumes['et_vol_ml'].median():.2f} mL")

# ============================================================
# STEP 4: MERGE SCAN TIMING + CLINICAL + VOLUMES → MASTER CSV
# ============================================================
print()
print("=" * 70)
print("STEP 4: Building master CSV with treatment tokens")
print("=" * 70)

# Map timepoint number → MRI day column
tp_day_map = {
    1: 'days_to_mri_tp1', 2: 'days_to_mri_tp2', 3: 'days_to_mri_tp3',
    4: 'days_to_mri_tp4', 5: 'days_to_mri_tp5', 6: 'days_to_mri_tp6',
}

# Merge scan, volume, and clinical data
master_records = []
for _, scan_row in df_scans.iterrows():
    pid = scan_row['patient_id']
    tp = scan_row['timepoint']
    scan_id = scan_row['scan_id']
    
    # Get clinical data for this patient
    clin = df_clinical[df_clinical['patient_id'] == pid]
    if len(clin) == 0:
        continue
    clin = clin.iloc[0]
    
    # Get volume data for this scan
    vol = df_volumes[df_volumes['scan_id'] == scan_id]
    if len(vol) == 0:
        continue
    vol = vol.iloc[0]
    
    # ---- REAL TIME: days from diagnosis ----
    tp_col = tp_day_map.get(tp)
    days_from_diag = clin[tp_col] if tp_col and not pd.isna(clin.get(tp_col, np.nan)) else np.nan
    
    # ---- TREATMENT STATE AT THIS SCAN ----
    scan_day = days_from_diag
    
    # Surgery
    surgery_day = clin['days_to_surgery'] if not pd.isna(clin.get('days_to_surgery', np.nan)) else np.nan
    days_since_surgery = (scan_day - surgery_day) if (not pd.isna(scan_day) and not pd.isna(surgery_day)) else np.nan
    
    # Chemotherapy
    chemo_start = clin['days_to_chemo_start'] if not pd.isna(clin.get('days_to_chemo_start', np.nan)) else None
    chemo_end = clin['days_to_chemo_end'] if not pd.isna(clin.get('days_to_chemo_end', np.nan)) else None
    
    on_chemo = 0
    days_on_chemo = 0.0
    if not pd.isna(scan_day) and chemo_start is not None and chemo_end is not None:
        if chemo_start <= scan_day <= chemo_end:
            on_chemo = 1
            days_on_chemo = scan_day - chemo_start
    
    # Pre-chemo flag
    pre_chemo = 0
    if not pd.isna(scan_day) and chemo_start is not None:
        if scan_day < chemo_start:
            pre_chemo = 1
    
    # Radiation
    radio_start = clin['days_to_radio_start'] if not pd.isna(clin.get('days_to_radio_start', np.nan)) else None
    radio_end = clin['days_to_radio_end'] if not pd.isna(clin.get('days_to_radio_end', np.nan)) else None
    
    on_radiation = 0
    days_on_radiation = 0.0
    if not pd.isna(scan_day) and radio_start is not None and radio_end is not None:
        if radio_start <= scan_day <= radio_end:
            on_radiation = 1
            days_on_radiation = scan_day - radio_start
    
    pre_radiation = 0
    if not pd.isna(scan_day) and radio_start is not None:
        if scan_day < radio_start:
            pre_radiation = 1
    
    # Immunotherapy (Avastin)
    immuno_start = clin['days_to_immuno_start'] if not pd.isna(clin.get('days_to_immuno_start', np.nan)) else None
    immuno_end = clin['days_to_immuno_end'] if not pd.isna(clin.get('days_to_immuno_end', np.nan)) else None
    
    on_immunotherapy = 0
    days_on_immuno = 0.0
    if not pd.isna(scan_day) and immuno_start is not None and immuno_end is not None:
        if immuno_start <= scan_day <= immuno_end:
            on_immunotherapy = 1
            days_on_immuno = scan_day - immuno_start
    
    # ---- TREATMENT PHASE CLASSIFICATION ----
    has_treatment_info = (chemo_start is not None or radio_start is not None)
    if pd.isna(scan_day):
        treatment_phase = 'UNKNOWN_TIMING'
    elif not has_treatment_info:
        treatment_phase = 'UNKNOWN_TREATMENT'
    else:
        earliest_treatment = min(
            [t for t in [chemo_start, radio_start] if t is not None]
        )
        if scan_day < earliest_treatment:
            treatment_phase = 'PRE_TREATMENT'
        elif on_chemo or on_radiation:
            treatment_phase = 'ON_TREATMENT'
        else:
            treatment_phase = 'POST_TREATMENT'
    
    # ---- TREATMENT TOKEN (15-D) ----
    # Normalized time features (scale ~0-1 for most patients)
    def norm_days(d, scale=365.0):
        """Normalize days to ~[0, 1] range."""
        if pd.isna(d):
            return 0.0
        return float(d) / scale
    
    treatment_token = {
        'tt_days_since_surgery':     norm_days(days_since_surgery),
        'tt_on_chemo':               float(on_chemo),
        'tt_days_on_chemo':          norm_days(days_on_chemo, 200.0),
        'tt_pre_chemo':              float(pre_chemo),
        'tt_on_radiation':           float(on_radiation),
        'tt_days_on_radiation':      norm_days(days_on_radiation, 60.0),
        'tt_radiation_dose_norm':    (clin['radiation_dose_gy'] / 60.0) if not pd.isna(clin.get('radiation_dose_gy', np.nan)) else 0.0,
        'tt_on_immunotherapy':       float(on_immunotherapy),
        'tt_days_on_immuno':         norm_days(days_on_immuno, 200.0),
        'tt_idh1_mutated':           float(clin['idh1'] == 1) if clin.get('idh1') is not None else 0.0,
        'tt_mgmt_methylated':        float(clin['mgmt'] == 1) if clin.get('mgmt') is not None else 0.0,
        'tt_grade_norm':             (float(clin['grade']) / 4.0) if not pd.isna(clin.get('grade', np.nan)) else 0.0,
        'tt_progressed':             float(not pd.isna(clin.get('days_to_1st_prog', np.nan)) and 
                                          not pd.isna(scan_day) and scan_day >= clin['days_to_1st_prog']),
        'tt_days_since_prog':        norm_days(scan_day - clin['days_to_1st_prog']) 
                                     if (not pd.isna(scan_day) and not pd.isna(clin.get('days_to_1st_prog', np.nan)) 
                                         and scan_day >= clin['days_to_1st_prog']) else 0.0,
        'tt_age_norm':               (clin['age_at_diagnosis'] / 100.0) if not pd.isna(clin.get('age_at_diagnosis', np.nan)) else 0.0,
    }
    
    # ---- TRAJECTORY CLASS ----
    # We'll compute this per-patient after building all scans
    
    rec = {
        'scan_id': scan_id,
        'patient_id': pid,
        'timepoint': tp,
        # Paths
        'path_t1c': scan_row['path_t1c'],
        'path_t1n': scan_row['path_t1n'],
        'path_t2f': scan_row['path_t2f'],
        'path_t2w': scan_row['path_t2w'],
        'path_mask': scan_row['path_mask'],
        'complete': scan_row['complete'],
        # Real time
        'days_from_diagnosis': days_from_diag,
        'days_since_surgery': days_since_surgery,
        # Treatment state
        'on_chemo': on_chemo,
        'days_on_chemo': days_on_chemo,
        'pre_chemo': pre_chemo,
        'on_radiation': on_radiation,
        'days_on_radiation': days_on_radiation,
        'pre_radiation': pre_radiation,
        'on_immunotherapy': on_immunotherapy,
        'days_on_immuno': days_on_immuno,
        'treatment_phase': treatment_phase,
        # Clinical (static per patient)
        'sex': clin['sex'],
        'age_at_diagnosis': clin['age_at_diagnosis'],
        'grade': clin['grade'],
        'primary_diagnosis': clin['primary_diagnosis'],
        'idh1': clin['idh1'],
        'mgmt': clin['mgmt'],
        'radiation_dose_gy': clin['radiation_dose_gy'],
        'chemo_name': clin['chemo_name'],
        'immunotherapy_name': clin['immunotherapy'],
        'progression': clin['progression'],
        'days_to_1st_prog': clin['days_to_1st_prog'],
        'overall_survival': clin['overall_survival'],
        'days_to_death': clin['days_to_death'],
        # Treatment timing (patient-level, for reference)
        'pat_chemo_start': chemo_start if chemo_start is not None else np.nan,
        'pat_chemo_end': chemo_end if chemo_end is not None else np.nan,
        'pat_radio_start': radio_start if radio_start is not None else np.nan,
        'pat_radio_end': radio_end if radio_end is not None else np.nan,
        'pat_immuno_start': immuno_start if immuno_start is not None else np.nan,
        'pat_immuno_end': immuno_end if immuno_end is not None else np.nan,
        # Volumes
        'wt_vol_ml': vol['wt_vol_ml'],
        'tc_vol_ml': vol['tc_vol_ml'],
        'et_vol_ml': vol['et_vol_ml'],
        'rc_vol_ml': vol['rc_vol_ml'],
        'wt_voxels': vol['wt_voxels'],
        'tc_voxels': vol['tc_voxels'],
        'et_voxels': vol['et_voxels'],
        'rc_voxels': vol['rc_voxels'],
        'centroid_x': vol['centroid_x'],
        'centroid_y': vol['centroid_y'],
        'centroid_z': vol['centroid_z'],
    }
    # Add treatment token columns
    rec.update(treatment_token)
    master_records.append(rec)

df_master = pd.DataFrame(master_records)
print(f"  Master rows: {len(df_master)}")

# ---- Compute trajectory class per patient ----
print("  Computing trajectory classes...")
traj_classes = {}
for pid, grp in df_master.groupby('patient_id'):
    grp_sorted = grp.sort_values('timepoint')
    vols = grp_sorted['wt_vol_ml'].values
    
    if len(vols) < 2:
        traj_classes[pid] = 'SINGLE_SCAN'
        continue
    
    # Use first and last non-NaN volumes
    valid = vols[~np.isnan(vols)]
    if len(valid) < 2:
        traj_classes[pid] = 'INSUFFICIENT_DATA'
        continue
    
    first_vol = valid[0]
    last_vol = valid[-1]
    
    if first_vol == 0:
        ratio = 1.0 if last_vol == 0 else 999.0
    else:
        ratio = last_vol / first_vol
    
    if ratio > 1.25:
        traj_classes[pid] = 'PROGRESSIVE'
    elif ratio < 0.75:
        traj_classes[pid] = 'RESPONDING'
    else:
        traj_classes[pid] = 'STABLE'

df_master['trajectory_class'] = df_master['patient_id'].map(traj_classes)

# ---- Compute days_between_scans ----
df_master = df_master.sort_values(['patient_id', 'timepoint'])
df_master['days_between_scans'] = df_master.groupby('patient_id')['days_from_diagnosis'].diff()

# Save
df_master.to_csv(OUT_DIR / 'mu_glioma_master.csv', index=False)
print(f"  Saved: mu_glioma_master.csv ({len(df_master)} rows, {df_master.shape[1]} columns)")

# Save treatment tokens separately for easy loading
tt_cols = [c for c in df_master.columns if c.startswith('tt_')]
df_tt = df_master[['scan_id', 'patient_id', 'timepoint', 'days_from_diagnosis'] + tt_cols].copy()
df_tt.to_csv(OUT_DIR / 'treatment_tokens.csv', index=False)
print(f"  Saved: treatment_tokens.csv ({len(tt_cols)} token dimensions)")

# ============================================================
# STEP 5: BUILD LONGITUDINAL INDEX (BraTS-compatible)
# ============================================================
print()
print("=" * 70)
print("STEP 5: Building longitudinal index")
print("=" * 70)

longitudinal_index = {}
for pid, grp in df_master.groupby('patient_id'):
    grp_sorted = grp.sort_values('timepoint')
    
    # Determine treatment info
    phases = grp_sorted['treatment_phase'].unique().tolist()
    has_pre = 'PRE_TREATMENT' in phases
    has_on = 'ON_TREATMENT' in phases
    has_post = 'POST_TREATMENT' in phases
    is_transition = has_pre and (has_on or has_post)
    
    longitudinal_index[pid] = {
        'patient_id': pid,
        'n_timepoints': len(grp_sorted),
        'timepoints': grp_sorted['timepoint'].tolist(),
        'scan_ids': grp_sorted['scan_id'].tolist(),
        'days_from_diagnosis': grp_sorted['days_from_diagnosis'].tolist(),
        'trajectory_class': traj_classes.get(pid, 'UNKNOWN'),
        'treatment_phases': phases,
        'is_transition_patient': is_transition,
        'grade': int(grp_sorted.iloc[0]['grade']) if not pd.isna(grp_sorted.iloc[0]['grade']) else None,
        'idh1': int(grp_sorted.iloc[0]['idh1']) if grp_sorted.iloc[0]['idh1'] is not None else None,
        'mgmt': int(grp_sorted.iloc[0]['mgmt']) if grp_sorted.iloc[0]['mgmt'] is not None else None,
        'has_avastin': grp_sorted.iloc[0]['immunotherapy_name'] not in [None, '', np.nan],
    }

with open(OUT_DIR / 'longitudinal_index.json', 'w') as f:
    json.dump(longitudinal_index, f, indent=2, default=str)
print(f"  Saved: longitudinal_index.json ({len(longitudinal_index)} patients)")

# Summary stats
n_multi = sum(1 for v in longitudinal_index.values() if v['n_timepoints'] >= 2)
n_transition = sum(1 for v in longitudinal_index.values() if v['is_transition_patient'])
print(f"  Multi-timepoint patients: {n_multi}")
print(f"  Transition patients (pre→post): {n_transition}")

# ============================================================
# STEP 6: DATA SUMMARY & STATISTICS
# ============================================================
print()
print("=" * 70)
print("STEP 6: Summary Statistics")
print("=" * 70)

# Treatment phase distribution
phase_counts = df_master['treatment_phase'].value_counts()
print("\n  Treatment phases:")
for phase, count in phase_counts.items():
    print(f"    {phase:25s}: {count:4d} ({100*count/len(df_master):.1f}%)")

# Trajectory class distribution
traj_counts = df_master.drop_duplicates('patient_id')['trajectory_class'].value_counts()
print("\n  Trajectory classes (patients):")
for cls, count in traj_counts.items():
    print(f"    {cls:20s}: {count:4d}")

# Timing completeness
timed = df_master['days_from_diagnosis'].notna().sum()
print(f"\n  Scans with timing: {timed}/{len(df_master)} ({100*timed/len(df_master):.0f}%)")
print(f"  Median days from diagnosis: {df_master['days_from_diagnosis'].median():.0f}")
print(f"  Range: {df_master['days_from_diagnosis'].min():.0f} – {df_master['days_from_diagnosis'].max():.0f}")

# Volume stats
print(f"\n  Volume statistics (mL):")
for col in ['wt_vol_ml', 'tc_vol_ml', 'et_vol_ml', 'rc_vol_ml']:
    vals = df_master[col].dropna()
    name = col.replace('_vol_ml', '').upper()
    print(f"    {name}: median={vals.median():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}")

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================
print()
print("=" * 70)
print("STEP 7: Generating visualizations")
print("=" * 70)

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
})

# ---- VIZ 1: Treatment Timelines (top 12 transition patients) ----
print("  Generating treatment timelines...")

transition_pids = [pid for pid, info in longitudinal_index.items() 
                   if info['is_transition_patient'] and info['n_timepoints'] >= 3]
transition_pids = sorted(transition_pids, 
                         key=lambda p: longitudinal_index[p]['n_timepoints'], 
                         reverse=True)[:12]

if len(transition_pids) > 0:
    n_plots = min(len(transition_pids), 12)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 2.5 * n_plots), dpi=120)
    if n_plots == 1:
        axes = [axes]
    
    for idx, pid in enumerate(transition_pids[:n_plots]):
        ax = axes[idx]
        pat = df_master[df_master['patient_id'] == pid].sort_values('timepoint')
        clin_row = df_clinical[df_clinical['patient_id'] == pid].iloc[0]
        
        # Plot volume trajectory
        days = pat['days_from_diagnosis'].values
        wt = pat['wt_vol_ml'].values
        valid = ~np.isnan(days) & ~np.isnan(wt)
        
        if valid.sum() > 0:
            ax.plot(days[valid], wt[valid], 'o-', color='#58a6ff', lw=2, ms=8, 
                    zorder=10, label='WT Volume')
        
        # Treatment bars
        bar_y_max = np.nanmax(wt) * 1.15 if np.nanmax(wt) > 0 else 10
        bar_height = bar_y_max * 0.06
        
        cs = clin_row.get('days_to_chemo_start', np.nan)
        ce = clin_row.get('days_to_chemo_end', np.nan)
        if not pd.isna(cs) and not pd.isna(ce):
            ax.barh(bar_y_max * 0.92, ce - cs, left=cs, height=bar_height,
                    color='#f97316', alpha=0.7, label='Chemo')
        
        rs = clin_row.get('days_to_radio_start', np.nan)
        re = clin_row.get('days_to_radio_end', np.nan)
        if not pd.isna(rs) and not pd.isna(re):
            ax.barh(bar_y_max * 0.82, re - rs, left=rs, height=bar_height,
                    color='#22c55e', alpha=0.7, label='Radiation')
        
        ims = clin_row.get('days_to_immuno_start', np.nan)
        ime = clin_row.get('days_to_immuno_end', np.nan)
        if not pd.isna(ims) and not pd.isna(ime):
            ax.barh(bar_y_max * 0.72, ime - ims, left=ims, height=bar_height,
                    color='#a855f7', alpha=0.7, label='Avastin')
        
        # Scan markers colored by phase
        phase_colors = {
            'PRE_TREATMENT': '#facc15',
            'ON_TREATMENT': '#f97316',
            'POST_TREATMENT': '#22c55e',
            'UNKNOWN_TREATMENT': '#6b7280',
            'UNKNOWN_TIMING': '#6b7280',
        }
        for _, srow in pat.iterrows():
            if not pd.isna(srow['days_from_diagnosis']):
                color = phase_colors.get(srow['treatment_phase'], '#6b7280')
                ax.axvline(srow['days_from_diagnosis'], color=color, alpha=0.3, lw=1)
        
        # Progression marker
        prog_day = clin_row.get('days_to_1st_prog', np.nan)
        if not pd.isna(prog_day):
            ax.axvline(prog_day, color='#ef4444', ls='--', lw=1.5, alpha=0.8)
            ax.text(prog_day, bar_y_max * 0.6, ' ⚠ Progression', color='#ef4444',
                    fontsize=7, va='center')
        
        traj = longitudinal_index[pid]['trajectory_class']
        grade_str = f"G{int(clin_row['grade'])}" if not pd.isna(clin_row.get('grade', np.nan)) else "G?"
        ax.set_title(f"{pid}  ({pat['timepoint'].nunique()} scans)  [{traj}]  {grade_str}", 
                     fontsize=10, color='#58a6ff', loc='left')
        ax.set_ylabel('WT Vol (mL)', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.5)
        if idx == n_plots - 1:
            ax.set_xlabel('Days from Diagnosis', fontsize=9)
    
    plt.suptitle('Treatment Timelines — Transition Patients (Pre → Post Treatment)',
                 fontsize=14, color='#58a6ff', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'viz_treatment_timelines.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved: viz_treatment_timelines.png")

# ---- VIZ 2: Volume Distributions ----
print("  Generating volume distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
vol_cols = [('wt_vol_ml', 'Whole Tumor', '#58a6ff'),
            ('tc_vol_ml', 'Tumor Core', '#f97316'),
            ('et_vol_ml', 'Enhancing Tumor', '#22c55e'),
            ('rc_vol_ml', 'Resection Cavity', '#a855f7')]

for ax, (col, name, color) in zip(axes.flat, vol_cols):
    vals = df_master[col].dropna()
    ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='none')
    ax.axvline(vals.median(), color='white', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'{name} (median={vals.median():.2f} mL)', fontsize=11, color=color)
    ax.set_xlabel('Volume (mL)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('Tumor Volume Distributions — MU-Glioma-Post (597 scans)',
             fontsize=14, color='#c9d1d9')
plt.tight_layout()
plt.savefig(OUT_DIR / 'viz_volume_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"    Saved: viz_volume_distributions.png")

# ---- VIZ 3: Treatment Phase EDA ----
print("  Generating treatment phase EDA...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

# 3a: Phase distribution
ax = axes[0, 0]
phases = df_master['treatment_phase'].value_counts()
colors_map = {
    'PRE_TREATMENT': '#facc15', 'ON_TREATMENT': '#f97316',
    'POST_TREATMENT': '#22c55e', 'UNKNOWN_TREATMENT': '#6b7280',
    'UNKNOWN_TIMING': '#4b5563',
}
bars = ax.barh(range(len(phases)), phases.values, 
               color=[colors_map.get(p, '#6b7280') for p in phases.index])
ax.set_yticks(range(len(phases)))
ax.set_yticklabels(phases.index, fontsize=8)
for i, v in enumerate(phases.values):
    ax.text(v + 2, i, f'{v} ({100*v/len(df_master):.0f}%)', 
            va='center', fontsize=8, color='#c9d1d9')
ax.set_title('Treatment Phase Distribution', fontsize=11, color='#58a6ff')
ax.set_xlabel('Number of Scans')
ax.grid(True, alpha=0.2, axis='x')

# 3b: Trajectory class distribution
ax = axes[0, 1]
traj_patients = df_master.drop_duplicates('patient_id')
traj_cls = traj_patients['trajectory_class'].value_counts()
traj_colors = {'PROGRESSIVE': '#ef4444', 'STABLE': '#22c55e', 
               'RESPONDING': '#58a6ff', 'SINGLE_SCAN': '#6b7280',
               'INSUFFICIENT_DATA': '#4b5563'}
ax.bar(range(len(traj_cls)), traj_cls.values,
       color=[traj_colors.get(c, '#6b7280') for c in traj_cls.index])
ax.set_xticks(range(len(traj_cls)))
ax.set_xticklabels(traj_cls.index, fontsize=8, rotation=20)
for i, v in enumerate(traj_cls.values):
    ax.text(i, v + 1, str(v), ha='center', fontsize=9, color='#c9d1d9')
ax.set_title('Trajectory Class Distribution', fontsize=11, color='#58a6ff')
ax.set_ylabel('Patients')
ax.grid(True, alpha=0.2, axis='y')

# 3c: Volume by treatment phase
ax = axes[1, 0]
phase_order = ['PRE_TREATMENT', 'ON_TREATMENT', 'POST_TREATMENT']
phase_data = [df_master[df_master['treatment_phase'] == p]['wt_vol_ml'].dropna()
              for p in phase_order]
bp = ax.boxplot(phase_data, labels=['Pre-Treat\n(📊)', 'Active\n(💊)', 'Post-Treat\n(👁️)'],
                patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['#facc15', '#f97316', '#22c55e']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title('WT Volume by Treatment Phase', fontsize=11, color='#58a6ff')
ax.set_ylabel('WT Volume (mL)')
ax.grid(True, alpha=0.2, axis='y')

# 3d: Scans per patient distribution
ax = axes[1, 1]
scans_per_patient = df_master.groupby('patient_id').size()
ax.hist(scans_per_patient, bins=range(1, 8), color='#58a6ff', alpha=0.7, 
        edgecolor='#0d1117', rwidth=0.8, align='left')
ax.set_xlabel('Scans per Patient')
ax.set_ylabel('Count')
ax.set_title(f'Scans per Patient (median={scans_per_patient.median():.0f})', 
             fontsize=11, color='#58a6ff')
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('MU-Glioma-Post — Treatment Phase & Trajectory EDA',
             fontsize=14, color='#c9d1d9')
plt.tight_layout()
plt.savefig(OUT_DIR / 'viz_treatment_phase_eda.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"    Saved: viz_treatment_phase_eda.png")

# ---- VIZ 4: Trajectory Classes with Treatment Context ----
print("  Generating trajectory class visualization...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)

# Select representative patients for each class
for cls_idx, (cls, cls_color) in enumerate([
    ('PROGRESSIVE', '#ef4444'), ('STABLE', '#22c55e'), ('RESPONDING', '#58a6ff')
]):
    cls_patients = [pid for pid, info in longitudinal_index.items()
                    if info['n_timepoints'] >= 3 and traj_classes.get(pid) == cls]
    if len(cls_patients) < 2:
        continue
    # Pick best 2
    cls_patients = sorted(cls_patients, 
                          key=lambda p: longitudinal_index[p]['n_timepoints'],
                          reverse=True)[:2]
    
    for row_idx, pid in enumerate(cls_patients):
        ax = axes[row_idx, cls_idx]
        pat = df_master[df_master['patient_id'] == pid].sort_values('timepoint')
        
        days = pat['days_from_diagnosis'].values
        wt = pat['wt_vol_ml'].values
        valid_mask = ~np.isnan(days) & ~np.isnan(wt)
        
        if valid_mask.sum() > 0:
            ax.plot(days[valid_mask], wt[valid_mask], 'o-', color=cls_color, 
                    lw=2.5, ms=10, zorder=10)
        
        # Color background by phase
        for _, srow in pat.iterrows():
            if pd.isna(srow['days_from_diagnosis']):
                continue
            pc = {'PRE_TREATMENT': '#facc1520', 'ON_TREATMENT': '#f9731620',
                  'POST_TREATMENT': '#22c55e10'}.get(srow['treatment_phase'], '#ffffff05')
            ax.axvspan(srow['days_from_diagnosis'] - 15, 
                       srow['days_from_diagnosis'] + 15,
                       color=pc, zorder=1)
        
        n_scans = pat['timepoint'].nunique()
        ax.set_title(f"{pid} ({n_scans} scans) [{cls}]", fontsize=9, color=cls_color)
        ax.set_ylabel('WT Vol (mL)', fontsize=8)
        ax.set_xlabel('Days from Diagnosis', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

plt.suptitle('Trajectory Classes — Progressive vs Stable vs Responding',
             fontsize=14, color='#c9d1d9')
plt.tight_layout()
plt.savefig(OUT_DIR / 'viz_trajectory_classes.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"    Saved: viz_trajectory_classes.png")

# ============================================================
# DONE
# ============================================================
print()
print("=" * 70)
print("PHASE M1 COMPLETE")
print("=" * 70)
print(f"""
Outputs saved to: {OUT_DIR}

Files:
  scan_index.json          — {len(scan_index['scans'])} scans indexed
  longitudinal_index.json  — {len(longitudinal_index)} patients
  clinical_data.csv        — {len(df_clinical)} patients × {df_clinical.shape[1]} clinical columns
  tumor_volumes.csv        — {len(df_volumes)} scans × volumes (WT/TC/ET/RC)
  mu_glioma_master.csv     — {len(df_master)} scans × {df_master.shape[1]} columns (THE master file)
  treatment_tokens.csv     — {len(df_tt)} scans × {len(tt_cols)} treatment token dims

Visualizations:
  viz_treatment_timelines.png
  viz_volume_distributions.png
  viz_treatment_phase_eda.png
  viz_trajectory_classes.png

Key Stats:
  Patients: {df_master['patient_id'].nunique()}
  Scans: {len(df_master)}
  Transition patients: {n_transition}
  Trajectory classes: {dict(traj_counts)}
  Treatment phases: {dict(phase_counts)}
""")
