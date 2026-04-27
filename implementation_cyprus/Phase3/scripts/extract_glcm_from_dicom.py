"""
GLCM Extraction from Raw DICOM — Exact Provider Pipeline Reproduction
======================================================================
Matches the provider's BrainMetastased_radiomics_extraction.py exactly:
  1. Read raw DICOM T1CE (not BraTS preprocessed)
  2. N4 bias correction with Otsu mask
  3. Clip to [0.1, 99.9] percentile of NON-ZERO voxels
  4. Rescale to [0, 1024]
  5. Crop to tumor bounding box
  6. Extract GLCM on LABEL 2 ONLY (mask_tumor = ET/enhancing tumor)
  7. binWidth=5 (paper setting)

DICOM folder naming:
  - Standard:     {pid}/DICOM/T1C_YYYY-MM-DD/
  - High-res:     {pid}/DICOM/T1C_HR_YYYY-MM-DD/ (P29, P33, P38)

Timepoint mapping: chronological order of DICOM dates within patient
  → date_0 = baseline, date_1 = fu1, date_2 = fu2, ...

Run: conda run -n radiomics python3 implementation_cyprus/Phase3/scripts/extract_glcm_from_dicom.py
Time: ~3 hours (170 scans × ~60s each, N4 Otsu is accurate but slower)
Output: embeddings_hybrid/glcm_computed_pyradiomics.npz  (170 keys, double underscore)
"""

import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
import logging, warnings, tempfile, os, re
from datetime import datetime

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import SimpleITK as sitk
from skimage.exposure import rescale_intensity
from radiomics import featureextractor, imageoperations
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path("/home/moamed/canada_me/explainable_diseas")
IMPL  = ROOT / "implementation_cyprus"
DATA  = IMPL / "Data" / "Cyprus-PROTEAS-zips"
OUT   = IMPL / "Phase3" / "bsf_fold_outputs" / "embeddings_hybrid"
OUT.mkdir(parents=True, exist_ok=True)

# ── PyRadiomics — binWidth=5, all features disabled except GLCM ───────────────
ext = featureextractor.RadiomicsFeatureExtractor()
ext.settings['binWidth']              = 5
ext.settings['interpolator']          = None
ext.settings['resampledPixelSpacing'] = None
ext.settings['normalize']             = False
ext.settings['distances']             = [1]
ext.disableAllImageTypes()
ext.disableAllFeatures()
ext.enableImageTypeByName('Original')
ext.enableFeaturesByName(glcm=['DifferenceEntropy', 'Contrast', 'ClusterShade'])

# ── Helper: read DICOM directory → SimpleITK image ───────────────────────────
def dicom_dir_to_sitk(dicom_dir):
    """Read a DICOM series directory → SimpleITK 3D image."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()

# ── Provider preprocessing (exact match to BrainMetastased_radiomics_extraction.py) ──
def n4_otsu(sitk_img):
    """N4 bias correction with Otsu mask — matches provider's bias_correction_sitk."""
    tmp = sitk.Cast(sitk_img, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(tmp)
    arr[arr == 0] = np.finfo(float).eps
    tmp2 = sitk.GetImageFromArray(arr)
    tmp2.CopyInformation(sitk_img)
    mask = sitk.OtsuThreshold(tmp2, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(tmp2, mask)

def clip_nonzero(sitk_img, lo=0.1, hi=99.9):
    """Clip to percentile of NON-ZERO voxels — matches provider's clip_image_sitk."""
    arr = sitk.GetArrayFromImage(sitk_img).ravel()
    arr = arr[arr != 0]
    lb = float(np.percentile(arr, lo))
    ub = float(np.percentile(arr, hi))
    f = sitk.ClampImageFilter()
    f.SetLowerBound(lb)
    f.SetUpperBound(ub)
    return f.Execute(sitk_img)

def preprocess(dicom_dir):
    """Full provider preprocessing pipeline."""
    raw = dicom_dir_to_sitk(dicom_dir)
    img = n4_otsu(raw)                    # Step 1: N4 with Otsu
    img = clip_nonzero(img)               # Step 2: Clip non-zero percentiles
    arr = sitk.GetArrayFromImage(img).astype('float32')
    arr = rescale_intensity(arr, out_range=(0, 1024))  # Step 3: Rescale 0-1024
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(raw)              # Preserve spatial metadata from raw
    return out, raw

# ── Helper: find T1C DICOM dir for a patient + date ──────────────────────────
def find_t1c_dirs(pid_dir):
    """
    Find all T1C DICOM directories sorted by date.
    Handles:  T1C_YYYY-MM-DD  and  T1C_HR_YYYY-MM-DD
    Returns: list of (datetime, Path) sorted chronologically
    """
    dicom_dir = pid_dir / 'DICOM'
    if not dicom_dir.exists():
        return []
    date_pat = re.compile(r'T1C(?:_HR)?_(\d{4}-\d{2}-\d{2})$', re.IGNORECASE)
    hits = []
    for d in dicom_dir.iterdir():
        m = date_pat.match(d.name)
        if m and d.is_dir():
            dt = datetime.strptime(m.group(1), '%Y-%m-%d')
            hits.append((dt, d))
    return sorted(hits, key=lambda x: x[0])  # chronological

TP_NAMES = ['baseline', 'fu1', 'fu2', 'fu3', 'fu4', 'fu5']

# ── Main extraction loop ──────────────────────────────────────────────────────
all_pid_dirs = sorted([d for d in DATA.iterdir() if d.is_dir() and d.name.startswith('P')])
print(f"Found {len(all_pid_dirs)} patient directories")
print(f"Settings: DICOM raw T1CE | N4+Otsu | clip(0.1-99.9% non-zero) | rescale(0-1024)")
print(f"Mask: label=2 ONLY (ET/enhancing tumor = mask_tumor in provider's code)\n")

# Load BSF embedding keys to know which scans we need
bsf = np.load(OUT / 'bsf_hybrid_embeddings.npz')
bsf_keys = set(bsf.files)

results = {}
failed  = []
skipped = []  # scans where DICOM date mapping couldn't be determined

for pid_dir in all_pid_dirs:
    pid = pid_dir.name
    t1c_dirs = find_t1c_dirs(pid_dir)

    if not t1c_dirs:
        print(f"  ⚠️  {pid}: no T1C DICOM dirs found")
        continue

    for idx, (dt, t1c_dir) in enumerate(t1c_dirs):
        if idx >= len(TP_NAMES):
            print(f"  ⚠️  {pid}: more T1C dirs ({len(t1c_dirs)}) than TP names")
            break

        tp  = TP_NAMES[idx]
        key = f"{pid}__{tp}"

        # Only process scans that are in our BSF embedding
        if key not in bsf_keys:
            skipped.append(key)
            continue

        # Find mask — label=2 only (ET = enhancing tumor)
        seg_dir = pid_dir / 'tumor segmentation'
        if not seg_dir.exists(): seg_dir = pid_dir / 'tumor_segmentation'
        mask_file = next((f for f in seg_dir.glob('*.nii.gz') if tp in f.name), None)
        if mask_file is None:
            print(f"  ⚠️  {key}: mask file not found")
            failed.append(key)
            continue

        tmp_t1c = tmp_mk = None
        try:
            # Load mask — label 2 ONLY (ET, = mask_tumor in provider's code)
            mask_nib = nib.load(str(mask_file))
            mask_arr = mask_nib.get_fdata()
            et_mask  = (mask_arr == 2).astype('int16')

            if et_mask.sum() < 5:
                print(f"  ⚠️  {key}: ET voxels too few ({et_mask.sum()})")
                failed.append(key)
                continue

            # Preprocess T1CE from raw DICOM
            sitk_proc, sitk_raw = preprocess(t1c_dir)

            # Save preprocessed image temp
            tmp_t1c = tempfile.mktemp(suffix='.nii.gz')
            sitk.WriteImage(sitk_proc, tmp_t1c)

            # Save ET mask temp
            tmp_mk = tempfile.mktemp(suffix='.nii.gz')
            nib.save(nib.Nifti1Image(et_mask, mask_nib.affine), tmp_mk)

            # Load into sitk and crop to bounding box (provider does cropToTumorMask)
            sitk_t1c_f = sitk.ReadImage(tmp_t1c)
            sitk_mask_f = sitk.Cast(sitk.ReadImage(tmp_mk), sitk.sitkInt32)

            # Align mask spacing/origin to image (in case of minor mismatch)
            sitk_mask_f.CopyInformation(sitk_t1c_f)

            os.unlink(tmp_t1c); tmp_t1c = None
            os.unlink(tmp_mk);  tmp_mk  = None

            # Crop to tumor bounding box (matches provider's cropToTumorMask)
            try:
                bb, correctedMask = imageoperations.checkMask(sitk_t1c_f, sitk_mask_f)
                img_crop, mask_crop = imageoperations.cropToTumorMask(sitk_t1c_f, sitk_mask_f, bb)
            except Exception:
                img_crop, mask_crop = sitk_t1c_f, sitk_mask_f  # fallback: no crop

            # Extract GLCM features
            feat = ext.execute(img_crop, mask_crop)
            de = float(feat.get('original_glcm_DifferenceEntropy', np.nan))
            co = float(feat.get('original_glcm_Contrast',          np.nan))
            cs = float(feat.get('original_glcm_ClusterShade',      np.nan))

            results[key] = np.array([de, co, cs], dtype=np.float32)
            print(f"  ✅ {key} [{dt.date()}] ET={int(et_mask.sum())}vx: "
                  f"DE={de:.3f}  Co={co:.1f}  CS={cs:.1f}")

        except Exception as e:
            print(f"  ❌ {key}: {type(e).__name__}: {str(e)[:80]}")
            failed.append(key)
        finally:
            for f in [tmp_t1c, tmp_mk]:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT / "glcm_computed_pyradiomics.npz"
np.savez(out_path, **results)

print(f"\n{'='*65}")
print(f"✅ Saved {len(results)} GLCM vectors → {out_path}")
print(f"   Method: DICOM raw T1CE | N4+Otsu | clip | rescale | ET-only | binWidth=5")
print(f"   Failed: {len(failed)}  |  Skipped (not in BSF): {len(skipped)}")
if failed: print(f"   Failed: {failed[:8]}")

# ── Correlation vs Excel GT ───────────────────────────────────────────────────
excel_path = OUT / "glcm_excel_gt.npz"
if excel_path.exists() and len(results) > 10:
    excel = np.load(excel_path)
    common = sorted(set(results.keys()) & set(excel.files))
    print(f"\n📊 Spearman ρ vs Excel GT ({len(common)} matched scans):")
    for i, fname in enumerate(['DiffEntropy', 'Contrast', 'ClusterShade']):
        pv = [float(results[k][i]) for k in common if not np.isnan(results[k][i])]
        ev = [float(excel[k][i])   for k in common if not np.isnan(results[k][i])]
        if len(pv) > 10:
            r, _ = spearmanr(pv, ev)
            status = '✅ GOOD' if r > 0.6 else ('⚠️ OK' if r > 0.4 else '❌ POOR')
            print(f"   {fname:20s}: ρ={r:.3f}  {status}")
