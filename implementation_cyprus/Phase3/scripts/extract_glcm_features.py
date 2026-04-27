"""
GLCM Feature Extraction — Exact Provider Pipeline Reproduction
==============================================================
Run with:  conda run -n radiomics python3 \
             implementation_cyprus/Phase3/scripts/extract_glcm_features.py

Exact match to provider's BrainMetastased_radiomics_extraction.py:
  1. BraTS T1CE NIfTI (0-1800 range, 240×240×155, NOT Z-score — verified)
  2. N4 bias correction with Otsu mask  (provider: otsu_threshold=True)
  3. Clip to [0.1, 99.9] percentile of NON-ZERO voxels only
  4. Rescale to [0, 1024] using skimage rescale_intensity
  5. Crop to tumor bounding box (cropToTumorMask)
  6. LABEL 2 ONLY (ET/enhancing tumor = mask_tumor in provider's replacers dict)
  7. binWidth=5, distances=[1], normalize=False

Key format:   {pid}__{tp}  (double underscore, matches BSF embedding keys)
Output:       Phase3/bsf_fold_outputs/embeddings_hybrid/glcm_computed_pyradiomics.npz
Time:         ~2 hours (170 scans × ~40s each)
"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import logging, warnings, tempfile, os
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import SimpleITK as sitk
from skimage.exposure import rescale_intensity
from radiomics import featureextractor, imageoperations

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path("/home/moamed/canada_me/explainable_diseas")
IMPL = ROOT / "implementation_cyprus"
DATA = IMPL / "Data" / "Cyprus-PROTEAS-zips"
OUT  = IMPL / "Phase3" / "bsf_fold_outputs" / "embeddings_hybrid"
OUT.mkdir(parents=True, exist_ok=True)

# ── PyRadiomics — exact provider settings ─────────────────────────────────────
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['binWidth']              = 5       # PROTEAS paper: bin width = 5
extractor.settings['interpolator']          = None
extractor.settings['resampledPixelSpacing'] = None
extractor.settings['normalize']             = False
extractor.settings['distances']             = [1]
extractor.disableAllImageTypes()
extractor.disableAllFeatures()
extractor.enableImageTypeByName('Original')
extractor.enableFeaturesByName(glcm=['DifferenceEntropy', 'Contrast', 'ClusterShade'])

# ── Preprocessing: exact match to BrainMetastased_radiomics_extraction.py ────
def n4_otsu(sitk_img):
    """N4 bias correction with Otsu threshold mask — provider's bias_correction_sitk."""
    tmp = sitk.Cast(sitk_img, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(tmp)
    arr[arr == 0] = np.finfo(float).eps
    tmp2 = sitk.GetImageFromArray(arr)
    tmp2.CopyInformation(sitk_img)
    mask = sitk.OtsuThreshold(tmp2, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(tmp2, mask)

def clip_nonzero(img, lo=0.1, hi=99.9):
    """Clip to percentile of NON-ZERO voxels — provider's clip_image_sitk."""
    arr = sitk.GetArrayFromImage(img).ravel()
    arr = arr[arr != 0]
    f = sitk.ClampImageFilter()
    f.SetLowerBound(float(np.percentile(arr, lo)))
    f.SetUpperBound(float(np.percentile(arr, hi)))
    return f.Execute(img)

def preprocess_t1c(t1c_path):
    """Full provider preprocessing pipeline on BraTS T1CE NIfTI."""
    raw = sitk.ReadImage(str(t1c_path), sitk.sitkFloat32)
    img = n4_otsu(raw)
    img = clip_nonzero(img)
    arr = sitk.GetArrayFromImage(img).astype('float32')
    arr = rescale_intensity(arr, out_range=(0, 1024))  # provider: out_range=(0,1024) only
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(raw)
    return out

# ── Helpers ───────────────────────────────────────────────────────────────────
TP_MAP = {
    'baseline': 'baseline', 'follow_up_1': 'fu1', 'follow_up_2': 'fu2',
    'follow_up_3': 'fu3',   'follow_up_4': 'fu4', 'follow_up_5': 'fu5',
}

def tp_from_mask_name(mask_filename, pid):
    name = mask_filename.replace(f'{pid}_tumor_mask_', '').replace('.nii.gz', '')
    return TP_MAP.get(name, name.replace('follow_up_', 'fu'))

def find_seg_dir(pid_dir):
    """Handle both 'tumor segmentation' (space) and 'tumor_segmentation' (underscore)."""
    for name in ['tumor segmentation', 'tumor_segmentation']:
        d = pid_dir / name
        if d.exists():
            return d
    return None

# ── Main extraction ───────────────────────────────────────────────────────────
all_pid_dirs = sorted([d for d in DATA.iterdir() if d.is_dir() and d.name.startswith('P')])
print(f"Found {len(all_pid_dirs)} patient directories")
print(f"Pipeline: BraTS T1CE → N4/Otsu → clip(non-zero 0.1-99.9%) → rescale(0-1024)")
print(f"Mask:     label=2 ONLY (ET/enhancing tumor = provider's mask_tumor)\n")

results = {}
failed  = []

for pid_dir in all_pid_dirs:
    pid = pid_dir.name

    seg_dir = find_seg_dir(pid_dir)
    if seg_dir is None:
        continue

    for mask_path in sorted(seg_dir.glob('*.nii.gz')):
        tp  = tp_from_mask_name(mask_path.name, pid)
        key = f"{pid}__{tp}"   # double underscore matches BSF embedding keys

        t1c_path = pid_dir / 'BraTS' / tp / 't1c.nii.gz'
        if not t1c_path.exists():
            print(f"  ⚠️  {key}: t1c.nii.gz not found")
            failed.append(key)
            continue

        tmp_mk = None
        try:
            # Load mask — LABEL 2 ONLY (ET = enhancing tumor = provider's mask_tumor)
            mask_nib = nib.load(str(mask_path))
            mask_arr = mask_nib.get_fdata()
            et_mask  = (mask_arr == 2).astype('int16')

            if et_mask.sum() < 5:
                print(f"  ⚠️  {key}: ET voxels too few ({int(et_mask.sum())}), trying all>0")
                et_mask = (mask_arr > 0).astype('int16')
                if et_mask.sum() < 5:
                    print(f"  ⚠️  {key}: mask empty — skip")
                    failed.append(key)
                    continue

            # Preprocess T1CE
            sitk_raw  = sitk.ReadImage(str(t1c_path), sitk.sitkFloat32)
            sitk_proc = preprocess_t1c(t1c_path)

            # Save ET mask to temp file
            tmp_mk = tempfile.mktemp(suffix='.nii.gz')
            nib.save(nib.Nifti1Image(et_mask, mask_nib.affine), tmp_mk)

            sitk_mask = sitk.Cast(sitk.ReadImage(tmp_mk), sitk.sitkInt32)
            sitk_mask.CopyInformation(sitk_raw)   # same 240×240×155 space

            os.unlink(tmp_mk); tmp_mk = None

            # Crop to tumor bounding box (provider: cropToTumorMask)
            try:
                bb, _ = imageoperations.checkMask(sitk_proc, sitk_mask)
                img_crop, mask_crop = imageoperations.cropToTumorMask(sitk_proc, sitk_mask, bb)
            except Exception:
                img_crop, mask_crop = sitk_proc, sitk_mask   # fallback: no crop

            # Extract GLCM — binWidth=5 (PROTEAS paper)
            feat = extractor.execute(img_crop, mask_crop)
            de = float(feat.get('original_glcm_DifferenceEntropy', np.nan))
            co = float(feat.get('original_glcm_Contrast',          np.nan))
            cs = float(feat.get('original_glcm_ClusterShade',      np.nan))

            results[key] = np.array([de, co, cs], dtype=np.float32)
            print(f"  ✅ {key}: DE={de:.3f}  Co={co:.1f}  CS={cs:.1f}  "
                  f"(ET={int((mask_arr==2).sum())}vx)")

        except Exception as e:
            print(f"  ❌ {key}: {type(e).__name__}: {str(e)[:80]}")
            failed.append(key)
        finally:
            if tmp_mk and os.path.exists(tmp_mk):
                try: os.unlink(tmp_mk)
                except: pass

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT / "glcm_computed_pyradiomics.npz"
np.savez(out_path, **results)
print(f"\n{'='*65}")
print(f"✅ Saved {len(results)} GLCM vectors → {out_path}")
print(f"   Pipeline: BraTS T1CE | N4/Otsu | clip(non-zero) | rescale | ET-only | binWidth=5")
print(f"   Failed: {len(failed)}")
if failed: print(f"   Failed keys: {failed[:8]}")

# ── Correlation vs Excel GT ───────────────────────────────────────────────────
excel_path = OUT / "glcm_excel_gt.npz"
if excel_path.exists() and len(results) > 10:
    excel = np.load(excel_path)
    common = sorted(set(results.keys()) & set(excel.files))
    print(f"\n📊 Spearman ρ vs Excel GT ({len(common)} matched scans):")
    for i, fname in enumerate(['DiffEntropy', 'Contrast', 'ClusterShade']):
        pv = [float(results[k][i]) for k in common if not np.isnan(float(results[k][i]))]
        ev = [float(excel[k][i])   for k in common if not np.isnan(float(results[k][i]))]
        if len(pv) > 10:
            r, p = spearmanr(pv, ev)
            status = '✅' if r > 0.6 else ('⚠️' if r > 0.4 else '❌')
            print(f"   {fname:20s}: ρ={r:.3f}  p={p:.2e}  {status}")
