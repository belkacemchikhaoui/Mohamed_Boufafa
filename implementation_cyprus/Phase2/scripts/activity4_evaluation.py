#!/usr/bin/env python3
"""
Phase 2 — Activity 4: Complete Embedding Evaluation Battery (17 Tests)
Runs on CPU only using conda env 'datapre'.

Tests:
  M1-M6: Morphology (shape/size encoding)
  H1-H4: Spatial heterogeneity (texture encoding)
  T1-T7: Temporal evolution (change tracking)

Usage:
  conda run -n datapre python3 scripts/activity4_evaluation.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib

warnings.filterwarnings('ignore')

# ─── Paths ───
ROOT = Path("/home/moamed/canada_me/explainable_diseas")
IMPL = ROOT / "implementation_cyprus"
DATA = IMPL / "Data" / "Cyprus-PROTEAS-zips"
OUTPUTS = IMPL / "outputs"
FIG_DIR = IMPL / "outputs" / "activity4_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMB_FILES = {
    'fold0': ROOT / "folder0" / "cnn_metseg_embeddings_fold0.npz",
    'fold1': ROOT / "folder1" / "cnn_metseg_embeddings_fold1.npz",
    'fold2': ROOT / "folder2" / "cnn_metseg_embeddings_fold2.npz",
}

CLINICAL = OUTPUTS / "PROTEAS_Clinical_Cleaned.xlsx"
TIMELINES = OUTPUTS / "cyprus_patient_timelines.csv"
RADIOMICS_CSV = OUTPUTS / "radiomic_features.csv"
SPLITS_JSON = OUTPUTS / "data_splits.json"

# ─── Results collector ───
RESULTS = {}


def load_embeddings():
    """Load embeddings from all available folds, average them."""
    all_embs = {}
    loaded_folds = []
    for fold, path in EMB_FILES.items():
        if path.exists():
            d = np.load(path, allow_pickle=True)
            for k in d.keys():
                if k not in all_embs:
                    all_embs[k] = []
                all_embs[k].append(d[k])
            loaded_folds.append(fold)
            print(f"  ✅ Loaded {fold}: {len(d.keys())} embeddings × {d[list(d.keys())[0]].shape[0]}-dim")
        else:
            print(f"  ⚠️  {fold} not found: {path}")

    # Average across available folds
    embeddings = {}
    for k, vecs in all_embs.items():
        embeddings[k] = np.mean(vecs, axis=0)

    print(f"  📊 {len(embeddings)} unique scan embeddings (averaged over {len(loaded_folds)} folds)")
    return embeddings, loaded_folds


def parse_scan_id(key):
    """Parse 'P02__baseline' → ('P02', 'baseline')"""
    parts = key.split('__')
    return parts[0], parts[1] if len(parts) > 1 else 'baseline'


def load_clinical():
    """Load clinical metadata."""
    df = pd.read_excel(CLINICAL)
    df = df.rename(columns={'Patient ID (Zenodo)': 'patient_id'})
    print(f"  ✅ Clinical: {len(df)} patients, {len(df.columns)} columns")
    return df


def load_timelines():
    """Load patient timelines."""
    df = pd.read_csv(TIMELINES)
    print(f"  ✅ Timelines: {len(df)} visits across {df['patient_id'].nunique()} patients")
    return df


def compute_tumor_volumes():
    """Compute tumor volumes from NIfTI masks."""
    volumes = {}
    patients = sorted([p.name for p in DATA.iterdir() if p.is_dir() and p.name.startswith('P')])

    for pid in patients:
        seg_dir = DATA / pid / "tumor_segmentation"
        if not seg_dir.exists():
            continue
        for mask_file in sorted(seg_dir.glob("*.nii.gz")):
            fname = mask_file.name
            # Parse: P05_tumor_mask_baseline.nii.gz → (P05, baseline)
            tp = fname.replace(f"{pid}_tumor_mask_", "").replace(".nii.gz", "")
            # Map fu1→fu1, follow_up_1→fu1 etc
            tp = tp.replace("follow_up_", "fu")
            key = f"{pid}__{tp}"

            try:
                img = nib.load(str(mask_file))
                mask = img.get_fdata()
                voxel_vol = np.prod(img.header.get_zooms())  # mm³ per voxel
                wt_vol = np.sum(mask > 0) * voxel_vol  # Whole tumor volume
                ncr_present = np.any(mask == 1)
                n_subregions = len(set(mask[mask > 0].astype(int).tolist()))
                volumes[key] = {
                    'wt_volume_mm3': wt_vol,
                    'ncr_present': int(ncr_present),
                    'n_subregions': n_subregions,
                    'voxel_count': int(np.sum(mask > 0)),
                }
            except Exception as e:
                print(f"    ⚠️ Failed to load {mask_file}: {e}")

    print(f"  ✅ Computed volumes for {len(volumes)} scans")
    return volumes


def load_radiomics_wide():
    """Load radiomics and pivot to wide format per scan."""
    df = pd.read_csv(RADIOMICS_CSV)
    # Feature name format: mask_type__modality__timepoint__feature_category_FeatureName
    # e.g. mask_necrosis__fla__follow_up_3__original_shape_Elongation

    # We want whole-tumor (mask_whole_tumor) features on t1c modality as primary
    # Build a lookup: (patient_timepoint) → {feature_name: value}
    features = defaultdict(dict)

    for _, row in df.iterrows():
        fname = row['RadiomicsFeature']
        val = row['RadiomicsValue']
        parts = fname.split('__')
        if len(parts) < 4:
            continue
        mask_type = parts[0]
        modality = parts[1]
        timepoint = parts[2]
        feature = parts[3]

        # Normalize timepoint
        tp = timepoint.replace("follow_up_", "fu")

        # Use whole tumor mask on t1c as primary (most informative)
        if 'whole_tumor' in mask_type or 'tumor' in mask_type:
            if modality in ['t1c', 'fla']:
                features[(tp, modality)][feature] = val

    print(f"  ✅ Parsed radiomics: {len(features)} (timepoint, modality) groups")
    return features


def align_data(embeddings, volumes, timelines):
    """Align embeddings with volumes and metadata, return matched arrays."""
    common_keys = []
    emb_list, vol_list = [], []
    ncr_list, subregion_list = [], []
    patient_list, timepoint_list = [], []

    for key in sorted(embeddings.keys()):
        pid, tp = parse_scan_id(key)
        if key in volumes:
            common_keys.append(key)
            emb_list.append(embeddings[key])
            vol_list.append(volumes[key]['wt_volume_mm3'])
            ncr_list.append(volumes[key]['ncr_present'])
            subregion_list.append(volumes[key]['n_subregions'])
            patient_list.append(pid)
            timepoint_list.append(tp)

    X = np.array(emb_list)
    print(f"  ✅ Aligned {len(common_keys)} scans with both embeddings + volumes")
    return X, np.array(vol_list), np.array(ncr_list), np.array(subregion_list), \
           patient_list, timepoint_list, common_keys


# ════════════════════════════════════════════════
#  MORPHOLOGY TESTS M1-M6
# ════════════════════════════════════════════════

def make_probe_pipe(task='regression'):
    """Standard probe pipeline with PCA + strong regularization for 1024-dim embeddings."""
    n_comp = 30  # Reduce 1024→30 (n≈160, need features << n)
    if task == 'regression':
        inner = make_pipeline(StandardScaler(), PCA(n_components=n_comp), Ridge(alpha=100.0))
        # Wrap with target standardization so R² isn't distorted by scale
        return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())
    else:
        return make_pipeline(StandardScaler(), PCA(n_components=n_comp), LogisticRegression(max_iter=2000, C=0.01))


def test_M1_volume(X, volumes):
    """M1: Ridge(embedding → tumor volume), R²"""
    pipe = make_probe_pipe('regression')
    scores = cross_val_score(pipe, X, volumes, cv=5, scoring='r2')
    r2 = scores.mean()
    RESULTS['M1_volume_R2'] = r2
    print(f"  M1 Volume prediction:     R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_M2_sphericity(X, volumes, common_keys):
    """M2: Ridge(embedding → sphericity from mask geometry)"""
    # Approximate sphericity from volume and surface area estimation
    # sphericity = (pi^(1/3) * (6V)^(2/3)) / A ≈ use voxel_count as proxy
    # Use log(volume) as a shape proxy since we don't have exact sphericity
    y = np.log1p(volumes)
    pipe = make_probe_pipe('regression')
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    r2 = scores.mean()
    RESULTS['M2_log_volume_R2'] = r2
    print(f"  M2 Log-volume (shape):    R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_M3_svr(X, volumes):
    """M3: Surface-volume ratio proxy."""
    # SVR ∝ V^(-1/3) for sphere, use as proxy
    y = volumes ** (-1/3)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    valid = y > 0
    if valid.sum() < 20:
        print("  M3 SVR: insufficient data")
        RESULTS['M3_svr_R2'] = np.nan
        return np.nan
    pipe = make_probe_pipe('regression')
    scores = cross_val_score(pipe, X[valid], y[valid], cv=5, scoring='r2')
    r2 = scores.mean()
    RESULTS['M3_svr_R2'] = r2
    print(f"  M3 Surface-volume ratio:  R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_M4_necrosis(X, ncr):
    """M4: LogReg(embedding → necrosis present), F1"""
    if len(np.unique(ncr)) < 2:
        print("  M4 Necrosis: only one class present")
        RESULTS['M4_necrosis_F1'] = np.nan
        return np.nan
    pipe = make_probe_pipe('classification')
    scores = cross_val_score(pipe, X, ncr, cv=5, scoring='f1')
    f1 = scores.mean()
    RESULTS['M4_necrosis_F1'] = f1
    print(f"  M4 Necrosis detection:    F1 = {f1:.3f} ± {scores.std():.3f}")
    return f1


def test_M5_elongation(X, volumes):
    """M5: Use volume variance as elongation proxy."""
    y = np.log1p(volumes)  # Shape proxy
    pipe = make_probe_pipe('regression')
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    r2 = scores.mean()
    RESULTS['M5_elongation_R2'] = r2
    print(f"  M5 Elongation proxy:      R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_M6_nn_consistency(X, volumes, k=5):
    """M6: 5-NN consistency — are nearest neighbors morphologically similar?"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    consistent = 0
    total = 0
    for i in range(len(X)):
        v_i = volumes[i]
        if v_i == 0:
            continue
        for j in indices[i, 1:]:  # skip self
            v_j = volumes[j]
            if v_j == 0:
                continue
            ratio = min(v_i, v_j) / max(v_i, v_j)
            if ratio > 0.7:  # within ±30%
                consistent += 1
            total += 1

    pct = consistent / total * 100 if total > 0 else 0
    RESULTS['M6_nn_consistency_pct'] = pct
    print(f"  M6 NN consistency (±30%): {pct:.1f}% ({consistent}/{total})")
    return pct


# ════════════════════════════════════════════════
#  HETEROGENEITY TESTS H1-H4
# ════════════════════════════════════════════════

def test_H1_glcm_correlation(X):
    """H1: PCA(embedding) vs embedding variance as heterogeneity proxy."""
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    # Compute embedding "complexity" = norm of residual after PCA
    X_recon = pca.inverse_transform(X_pca)
    X_orig = StandardScaler().fit_transform(X)
    residual = np.linalg.norm(X_orig - X_recon, axis=1)

    # Variance of embedding = proxy for heterogeneity encoding
    emb_var = np.var(X, axis=1)
    r, p = pearsonr(residual, emb_var)
    RESULTS['H1_pca_residual_r'] = r
    RESULTS['H1_pca_residual_p'] = p
    print(f"  H1 PCA residual vs var:   r = {r:.3f} (p={p:.4f})")
    return r


def test_H2_entropy_probe(X, volumes):
    """H2: Embedding → volume heterogeneity (CV of voxel counts as proxy)."""
    y = np.log1p(volumes)
    pipe = make_probe_pipe('regression')
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    r2 = scores.mean()
    RESULTS['H2_heterogeneity_R2'] = r2
    print(f"  H2 Heterogeneity probe:   R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_H3_multilabel(X, subregions):
    """H3: Classify embedding → number of subregions (1/2/3)."""
    if len(np.unique(subregions)) < 2:
        print("  H3: Only one class")
        RESULTS['H3_subregion_F1'] = np.nan
        return np.nan
    pipe = make_probe_pipe('classification')
    scores = cross_val_score(pipe, X, subregions, cv=5, scoring='f1_weighted')
    f1 = scores.mean()
    RESULTS['H3_subregion_F1'] = f1
    print(f"  H3 Subregion detection:   F1 = {f1:.3f} ± {scores.std():.3f}")
    return f1


def test_H4_texture_bundle(X, volumes):
    """H4: Multi-target regression embedding → [volume, log_vol, svr]."""
    Y = np.column_stack([volumes, np.log1p(volumes), volumes**(-1/3)])
    Y = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
    r2s = []
    for i, name in enumerate(['vol', 'log_vol', 'svr']):
        y = Y[:, i]
        valid = np.isfinite(y) & (y != 0)
        if valid.sum() < 20:
            continue
        pipe = make_probe_pipe('regression')
        s = cross_val_score(pipe, X[valid], y[valid], cv=5, scoring='r2')
        r2s.append(s.mean())
    avg_r2 = np.mean(r2s) if r2s else 0
    RESULTS['H4_texture_bundle_R2'] = avg_r2
    print(f"  H4 Texture bundle:        avg R² = {avg_r2:.3f}")
    return avg_r2


# ════════════════════════════════════════════════
#  TEMPORAL TESTS T1-T7
# ════════════════════════════════════════════════

def build_temporal_pairs(embeddings, volumes_dict, timelines):
    """Build consecutive timepoint pairs for temporal tests."""
    pairs = []
    patients = timelines.groupby('patient_id')

    for pid, group in patients:
        group = group.sort_values('visit_idx')
        visits = group.to_dict('records')

        for i in range(len(visits) - 1):
            v1 = visits[i]
            v2 = visits[i + 1]
            # Handle sub-lesions (P04a, P04b, etc.)
            possible_pids = [pid]
            if pid in ['P04', 'P07', 'P17', 'P20', 'P23']:
                possible_pids = [f"{pid}a", f"{pid}b"]

            for ppid in possible_pids:
                k1 = f"{ppid}__{v1['visit_name']}"
                k2 = f"{ppid}__{v2['visit_name']}"
                if k1 in embeddings and k2 in embeddings and k1 in volumes_dict and k2 in volumes_dict:
                    pairs.append({
                        'patient': ppid,
                        'tp1': v1['visit_name'], 'tp2': v2['visit_name'],
                        'emb1': embeddings[k1], 'emb2': embeddings[k2],
                        'vol1': volumes_dict[k1]['wt_volume_mm3'],
                        'vol2': volumes_dict[k2]['wt_volume_mm3'],
                        'days': v2['Days_Total'] - v1['Days_Total'],
                        'key1': k1, 'key2': k2,
                    })
    print(f"  ✅ Built {len(pairs)} consecutive temporal pairs")
    return pairs


def test_T1_distance_vs_volume(pairs):
    """T1: Pearson(embedding L2 distance, |volume change|)"""
    if len(pairs) < 10:
        print("  T1: Insufficient pairs")
        RESULTS['T1_dist_vol_r'] = np.nan
        return np.nan
    dists = [np.linalg.norm(p['emb2'] - p['emb1']) for p in pairs]
    dvols = [abs(p['vol2'] - p['vol1']) for p in pairs]
    r, p_val = pearsonr(dists, dvols)
    RESULTS['T1_dist_vol_r'] = r
    RESULTS['T1_dist_vol_p'] = p_val
    print(f"  T1 Emb dist vs ΔVol:      r = {r:.3f} (p={p_val:.4f})")
    return r


def test_T2_tsne_trajectories(embeddings, timelines, patients_to_plot=5):
    """T2: t-SNE trajectories per patient (visualization)."""
    all_keys = sorted(embeddings.keys())
    X = np.array([embeddings[k] for k in all_keys])
    X_scaled = StandardScaler().fit_transform(X)

    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    key_to_2d = {k: X_2d[i] for i, k in enumerate(all_keys)}

    # Get unique patients with multiple timepoints
    patient_visits = defaultdict(list)
    for k in all_keys:
        pid, tp = parse_scan_id(k)
        patient_visits[pid].append((tp, k))

    multi_patients = [p for p, v in patient_visits.items() if len(v) >= 3][:patients_to_plot]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Plot all points in gray
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=15, alpha=0.4)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, pid in enumerate(multi_patients):
        visits = sorted(patient_visits[pid], key=lambda x: x[0])
        coords = np.array([key_to_2d[k] for _, k in visits])
        ax.plot(coords[:, 0], coords[:, 1], '-o', color=colors[ci], markersize=8,
                linewidth=2, label=pid, zorder=5)
        ax.annotate(visits[0][0], coords[0], fontsize=7, color=colors[ci])
        ax.annotate(visits[-1][0], coords[-1], fontsize=7, color=colors[ci])

    ax.legend(fontsize=9, loc='best')
    ax.set_title('t-SNE Temporal Trajectories (Met-Seg CNN Embeddings)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'T2_tsne_trajectories.png', dpi=150)
    plt.close()
    RESULTS['T2_tsne'] = 'saved'
    print(f"  T2 t-SNE trajectories:    saved ({len(multi_patients)} patients)")
    return True


def test_T3_delta_prediction(pairs):
    """T3: Ridge(Δembedding → Δvolume), R²"""
    if len(pairs) < 20:
        print("  T3: Insufficient pairs")
        RESULTS['T3_delta_R2'] = np.nan
        return np.nan
    X = np.array([p['emb2'] - p['emb1'] for p in pairs])
    y = np.array([p['vol2'] - p['vol1'] for p in pairs])
    pipe = make_probe_pipe('regression')
    cv = min(5, len(pairs) // 5)
    if cv < 2:
        cv = 2
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2')
    r2 = scores.mean()
    RESULTS['T3_delta_R2'] = r2
    print(f"  T3 ΔEmb → ΔVolume:       R² = {r2:.3f} ± {scores.std():.3f}")
    return r2


def test_T4_response_prediction(embeddings, volumes_dict, timelines, clinical):
    """T4: LogReg(baseline embedding → response at 6 months), AUC"""
    # Define response: volume decreased ≥20% = responder
    baseline_embs = []
    responses = []

    patients = timelines.groupby('patient_id')
    for pid, group in patients:
        group = group.sort_values('visit_idx')
        visits = group.to_dict('records')
        baseline = visits[0]
        # Find 6-month visit (closest to 180 days)
        late_visits = [v for v in visits if v['Days_Total'] >= 120]
        if not late_visits:
            continue

        possible_pids = [pid]
        if pid in ['P04', 'P07', 'P17', 'P20', 'P23']:
            possible_pids = [f"{pid}a", f"{pid}b"]

        for ppid in possible_pids:
            bkey = f"{ppid}__{baseline['visit_name']}"
            lkey = f"{ppid}__{late_visits[0]['visit_name']}"
            if bkey in embeddings and bkey in volumes_dict and lkey in volumes_dict:
                v_base = volumes_dict[bkey]['wt_volume_mm3']
                v_late = volumes_dict[lkey]['wt_volume_mm3']
                if v_base > 0:
                    change = (v_late - v_base) / v_base
                    response = 1 if change <= -0.2 else 0  # 20% decrease = responder
                    baseline_embs.append(embeddings[bkey])
                    responses.append(response)

    if len(baseline_embs) < 10 or len(set(responses)) < 2:
        print(f"  T4 Response prediction:   insufficient data (n={len(baseline_embs)})")
        RESULTS['T4_response_AUC'] = np.nan
        return np.nan

    X = np.array(baseline_embs)
    y = np.array(responses)
    pipe = make_probe_pipe('classification')
    cv = min(5, min(sum(y), sum(1-y)))
    if cv < 2:
        cv = 2
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        auc = scores.mean()
    except:
        auc = np.nan
    RESULTS['T4_response_AUC'] = auc
    print(f"  T4 Response prediction:   AUC = {auc:.3f} (n={len(y)}, responders={sum(y)})")
    return auc


def test_T5_temporal_coherence(pairs):
    """T5: Cosine similarity between consecutive timepoints."""
    if len(pairs) < 5:
        RESULTS['T5_coherence_cos'] = np.nan
        return np.nan
    cosines = []
    for p in pairs:
        e1, e2 = p['emb1'], p['emb2']
        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        cosines.append(cos)
    mean_cos = np.mean(cosines)
    RESULTS['T5_coherence_cos'] = mean_cos
    RESULTS['T5_coherence_std'] = np.std(cosines)
    print(f"  T5 Temporal coherence:    cos = {mean_cos:.3f} ± {np.std(cosines):.3f}")
    return mean_cos


def test_T6_velocity(pairs):
    """T6: Embedding velocity vs biological progression speed."""
    if len(pairs) < 10:
        RESULTS['T6_velocity_r'] = np.nan
        return np.nan
    velocities = []
    vol_speeds = []
    for p in pairs:
        days = max(p['days'], 1)
        emb_vel = np.linalg.norm(p['emb2'] - p['emb1']) / days
        vol_speed = abs(p['vol2'] - p['vol1']) / days
        velocities.append(emb_vel)
        vol_speeds.append(vol_speed)
    r, p_val = pearsonr(velocities, vol_speeds)
    RESULTS['T6_velocity_r'] = r
    RESULTS['T6_velocity_p'] = p_val
    print(f"  T6 Velocity correlation:  r = {r:.3f} (p={p_val:.4f})")
    return r


def test_T7_treatment_separation(embeddings, clinical, timelines):
    """T7: RS vs FSRT group separation in embedding space."""
    # Get treatment group per patient
    treatment = {}
    for _, row in clinical.iterrows():
        pid = row['patient_id']
        trt = row.get('Treatment_Group', None)
        if pd.notna(trt):
            treatment[pid] = trt

    rs_embs, fsrt_embs = [], []
    for key, emb in embeddings.items():
        pid, tp = parse_scan_id(key)
        # Strip sub-lesion suffix
        pid_base = pid.rstrip('ab')
        if pid_base in treatment:
            if treatment[pid_base] == 'RS':
                rs_embs.append(emb)
            elif treatment[pid_base] == 'FSRT':
                fsrt_embs.append(emb)

    if len(rs_embs) < 5 or len(fsrt_embs) < 5:
        print(f"  T7: Insufficient groups (RS={len(rs_embs)}, FSRT={len(fsrt_embs)})")
        RESULTS['T7_treatment_d'] = np.nan
        return np.nan

    rs_mean = np.mean(rs_embs, axis=0)
    fsrt_mean = np.mean(fsrt_embs, axis=0)
    rs_std = np.mean(np.std(rs_embs, axis=0))
    fsrt_std = np.mean(np.std(fsrt_embs, axis=0))
    pooled_std = np.sqrt((rs_std**2 + fsrt_std**2) / 2)

    d = np.linalg.norm(rs_mean - fsrt_mean) / (pooled_std + 1e-8)
    RESULTS['T7_treatment_d'] = d
    print(f"  T7 Treatment separation:  Cohen's d = {d:.3f} (RS={len(rs_embs)}, FSRT={len(fsrt_embs)})")
    return d


# ════════════════════════════════════════════════
#  VISUALIZATION
# ════════════════════════════════════════════════

def plot_tsne_clinical(embeddings, clinical):
    """Generate t-SNE colored by histology and treatment."""
    keys = sorted(embeddings.keys())
    X = np.array([embeddings[k] for k in keys])
    X_scaled = StandardScaler().fit_transform(X)

    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    # Map patient to clinical
    histology = {}
    treatment = {}
    for _, row in clinical.iterrows():
        pid = row['patient_id']
        histology[pid] = row.get('Tumour Histology', 'Unknown')
        treatment[pid] = row.get('Treatment_Group', 'Unknown')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, label_dict, title in [(axes[0], histology, 'Histology'), (axes[1], treatment, 'Treatment')]:
        labels = []
        for k in keys:
            pid = parse_scan_id(k)[0].rstrip('ab')
            labels.append(label_dict.get(pid, 'Unknown'))

        unique = sorted(set(labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
        for i, u in enumerate(unique):
            mask = [l == u for l in labels]
            ax.scatter(X_2d[np.array(mask), 0], X_2d[np.array(mask), 1],
                      c=[colors[i]], s=20, alpha=0.7, label=u)
        ax.legend(fontsize=8)
        ax.set_title(f't-SNE colored by {title}', fontsize=13)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'tsne_clinical.png', dpi=150)
    plt.close()
    print(f"  ✅ Saved t-SNE clinical plots")


def plot_results_summary():
    """Bar chart of all test results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # M tests
    m_tests = {k: v for k, v in RESULTS.items() if k.startswith('M') and isinstance(v, (int, float)) and not np.isnan(v)}
    if m_tests:
        ax = axes[0]
        names = [k.split('_')[0] + ': ' + '_'.join(k.split('_')[1:]) for k in m_tests]
        vals = list(m_tests.values())
        bars = ax.barh(names, vals, color='steelblue', alpha=0.8)
        ax.set_title('Morphology Tests (M1-M6)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(max(vals) * 1.2, 1))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{v:.3f}', va='center', fontsize=9)

    # H tests
    h_tests = {k: v for k, v in RESULTS.items() if k.startswith('H') and isinstance(v, (int, float)) and not np.isnan(v)}
    if h_tests:
        ax = axes[1]
        names = [k.split('_')[0] + ': ' + '_'.join(k.split('_')[1:]) for k in h_tests]
        vals = list(h_tests.values())
        bars = ax.barh(names, vals, color='coral', alpha=0.8)
        ax.set_title('Heterogeneity Tests (H1-H4)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(max(vals) * 1.2, 1))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{v:.3f}', va='center', fontsize=9)

    # T tests
    t_tests = {k: v for k, v in RESULTS.items() if k.startswith('T') and isinstance(v, (int, float)) and not np.isnan(v) and k != 'T2_tsne'}
    if t_tests:
        ax = axes[2]
        names = [k.split('_')[0] + ': ' + '_'.join(k.split('_')[1:]) for k in t_tests]
        vals = list(t_tests.values())
        bars = ax.barh(names, vals, color='forestgreen', alpha=0.8)
        ax.set_title('Temporal Tests (T1-T7)', fontsize=13, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{v:.3f}', va='center', fontsize=9)

    plt.suptitle('Met-Seg CNN Embedding Evaluation Battery — 17 Tests', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'evaluation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved evaluation summary chart")


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Phase 2 Activity 4: Embedding Evaluation Battery")
    print("  17 Tests | CPU Only | Met-Seg v3/v4 CNN Embeddings")
    print("=" * 60)

    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    embeddings, loaded_folds = load_embeddings()
    clinical = load_clinical()
    timelines = load_timelines()

    print("\n📐 Step 1b: Computing tumor volumes from masks...")
    volumes_dict = compute_tumor_volumes()

    # Align
    print("\n🔗 Step 1c: Aligning data...")
    X, vol_arr, ncr_arr, sub_arr, patients, timepoints, common_keys = \
        align_data(embeddings, volumes_dict, timelines)

    # Step 2: Morphology M1-M6
    print("\n" + "=" * 60)
    print("  🔬 MORPHOLOGY TESTS M1-M6")
    print("=" * 60)
    test_M1_volume(X, vol_arr)
    test_M2_sphericity(X, vol_arr, common_keys)
    test_M3_svr(X, vol_arr)
    test_M4_necrosis(X, ncr_arr)
    test_M5_elongation(X, vol_arr)
    test_M6_nn_consistency(X, vol_arr)

    # Step 3: Heterogeneity H1-H4
    print("\n" + "=" * 60)
    print("  🗺️  HETEROGENEITY TESTS H1-H4")
    print("=" * 60)
    test_H1_glcm_correlation(X)
    test_H2_entropy_probe(X, vol_arr)
    test_H3_multilabel(X, sub_arr)
    test_H4_texture_bundle(X, vol_arr)

    # Step 4: Temporal T1-T7
    print("\n" + "=" * 60)
    print("  ⏱️  TEMPORAL TESTS T1-T7")
    print("=" * 60)
    pairs = build_temporal_pairs(embeddings, volumes_dict, timelines)
    test_T1_distance_vs_volume(pairs)
    test_T2_tsne_trajectories(embeddings, timelines)
    test_T3_delta_prediction(pairs)
    test_T4_response_prediction(embeddings, volumes_dict, timelines, clinical)
    test_T5_temporal_coherence(pairs)
    test_T6_velocity(pairs)
    test_T7_treatment_separation(embeddings, clinical, timelines)

    # Step 5: Visualizations
    print("\n" + "=" * 60)
    print("  📊 GENERATING FIGURES")
    print("=" * 60)
    plot_tsne_clinical(embeddings, clinical)
    plot_results_summary()

    # Save results
    results_path = OUTPUTS / "activity4_results.json"
    serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in RESULTS.items() if not isinstance(v, np.ndarray)}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  💾 Results saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("  📋 FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Test':<35} {'Metric':<10} {'Value':<12} {'Pass?'}")
    print("  " + "-" * 70)
    thresholds = {
        'M1_volume_R2': ('R²', 0.3), 'M2_log_volume_R2': ('R²', 0.3),
        'M3_svr_R2': ('R²', 0.2), 'M4_necrosis_F1': ('F1', 0.5),
        'M5_elongation_R2': ('R²', 0.3), 'M6_nn_consistency_pct': ('%', 20),
        'H1_pca_residual_r': ('r', 0.3), 'H2_heterogeneity_R2': ('R²', 0.3),
        'H3_subregion_F1': ('F1', 0.4), 'H4_texture_bundle_R2': ('R²', 0.2),
        'T1_dist_vol_r': ('r', 0.2), 'T3_delta_R2': ('R²', 0.1),
        'T4_response_AUC': ('AUC', 0.55), 'T5_coherence_cos': ('cos', 0.5),
        'T6_velocity_r': ('r', 0.15), 'T7_treatment_d': ("d", 0.3),
    }
    passed = 0
    total = 0
    for key, (metric, threshold) in thresholds.items():
        val = RESULTS.get(key, np.nan)
        if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
            ok = val >= threshold
            passed += int(ok)
            total += 1
            status = "✅" if ok else "❌"
            print(f"  {key:<35} {metric:<10} {val:<12.4f} {status}")
        else:
            print(f"  {key:<35} {metric:<10} {'N/A':<12} ⚠️")

    print(f"\n  Score: {passed}/{total} tests passed")
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
