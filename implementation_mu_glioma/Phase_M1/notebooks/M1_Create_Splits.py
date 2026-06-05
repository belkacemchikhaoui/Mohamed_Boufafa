#!/usr/bin/env python3
"""
M1 — Create Stratified Train/Val/Test Split
=============================================
Patient-level split stratified by:
  - trajectory_class (PROGRESSIVE / STABLE / RESPONDING / SINGLE_SCAN)
  - idh1 status (0=wildtype, 1=mutant)
  - grade (2 vs 4)

Output: Phase_M1/outputs/data_splits.json
"""

import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MASTER  = OUT_DIR / "mu_glioma_master.csv"

df = pd.read_csv(MASTER)
print(f"Master CSV: {len(df)} scans, {df['patient_id'].nunique()} patients")

# ── Patient-level features for stratification ──
patients = df.drop_duplicates('patient_id')[['patient_id', 'trajectory_class', 'idh1', 'grade']].copy()

# Simplify trajectory for stratification
patients['traj_strat'] = patients['trajectory_class'].map({
    'PROGRESSIVE': 'PROG',
    'RESPONDING': 'RESP',
    'STABLE': 'STABLE',
    'SINGLE_SCAN': 'SINGLE',
    'INSUFFICIENT_DATA': 'SINGLE',
})

# Simplify grade for stratification
patients['grade_strat'] = patients['grade'].apply(lambda g: 'HGG' if g >= 3 else 'LGG')

# IDH stratification
patients['idh_strat'] = patients['idh1'].apply(lambda x: 'IDH_MUT' if x == 1 else 'IDH_WT')

# Combined stratification key
patients['strat_key'] = patients['traj_strat'] + '_' + patients['grade_strat'] + '_' + patients['idh_strat']

print("\nStratification group sizes:")
for key, count in patients['strat_key'].value_counts().items():
    print(f"  {key:35s}: {count}")

# ── Stratified split: 77% train, 10% val, 13% test ──
np.random.seed(42)

train_pids, val_pids, test_pids = [], [], []

for strat_key, group in patients.groupby('strat_key'):
    pids = group['patient_id'].values.tolist()
    np.random.shuffle(pids)
    n = len(pids)
    
    if n <= 2:
        # Too small to split — put all in train
        train_pids.extend(pids)
        continue
    
    n_test = max(1, round(n * 0.13))
    n_val  = max(1, round(n * 0.10))
    n_train = n - n_test - n_val
    
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - n_train - n_test)
    
    test_pids.extend(pids[:n_test])
    val_pids.extend(pids[n_test:n_test + n_val])
    train_pids.extend(pids[n_test + n_val:])

# Verify no overlap
assert len(set(train_pids) & set(val_pids)) == 0
assert len(set(train_pids) & set(test_pids)) == 0
assert len(set(val_pids) & set(test_pids)) == 0
total = len(train_pids) + len(val_pids) + len(test_pids)
assert total == len(patients), f"Split total {total} != {len(patients)}"

print(f"\n{'='*50}")
print(f"Split summary:")
print(f"  Train: {len(train_pids)} patients ({100*len(train_pids)/total:.0f}%)")
print(f"  Val:   {len(val_pids)} patients ({100*len(val_pids)/total:.0f}%)")
print(f"  Test:  {len(test_pids)} patients ({100*len(test_pids)/total:.0f}%)")

# Scan counts per split
train_scans = df[df['patient_id'].isin(train_pids)]
val_scans   = df[df['patient_id'].isin(val_pids)]
test_scans  = df[df['patient_id'].isin(test_pids)]
print(f"\n  Train scans: {len(train_scans)}")
print(f"  Val scans:   {len(val_scans)}")
print(f"  Test scans:  {len(test_scans)}")

# Show balance
for split_name, split_pids in [('Train', train_pids), ('Val', val_pids), ('Test', test_pids)]:
    split_df = patients[patients['patient_id'].isin(split_pids)]
    traj = split_df['traj_strat'].value_counts().to_dict()
    grade = split_df['grade_strat'].value_counts().to_dict()
    idh = split_df['idh_strat'].value_counts().to_dict()
    print(f"\n  {split_name}:")
    print(f"    Trajectory: {traj}")
    print(f"    Grade:      {grade}")
    print(f"    IDH:        {idh}")

# ── Save ──
splits = {
    'train': sorted(train_pids),
    'val':   sorted(val_pids),
    'test':  sorted(test_pids),
    'metadata': {
        'n_train': len(train_pids),
        'n_val':   len(val_pids),
        'n_test':  len(test_pids),
        'seed':    42,
        'stratification': 'trajectory_class + grade + idh1',
    }
}

with open(OUT_DIR / 'data_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'data_splits.json'}")
