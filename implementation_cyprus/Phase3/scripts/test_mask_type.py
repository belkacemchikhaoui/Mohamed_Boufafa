"""
Diagnostic test — BraTS T1CE + provider exact preprocessing + label=2 (ET only)
=================================================================================
Run: conda run -n radiomics python3 implementation_cyprus/Phase3/scripts/test_mask_type.py
Time: ~5 minutes (5 scans × ~60s each for N4/Otsu)

Why BraTS NIfTI works (NOT Z-score normalized):
  Checked actual range: P01 min=-1, max=1834  P06 min=0, max=1425
  These are original MRI intensity units — same as provider's images.

Validates that BraTS T1CE + N4/Otsu + clip(non-zero) + rescale + label=2 + binWidth=5
reproduces the Excel GT GLCM values (Spearman ρ should be > 0.6).
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging, warnings, tempfile, os

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import SimpleITK as sitk
from skimage.exposure import rescale_intensity
from radiomics import featureextractor, imageoperations

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA  = Path("/home/moamed/canada_me/explainable_diseas/implementation_cyprus/Data/Cyprus-PROTEAS-zips")
EMBS  = Path("/home/moamed/canada_me/explainable_diseas/implementation_cyprus/Phase3/bsf_fold_outputs/embeddings_hybrid")

# ── PyRadiomics — exact provider settings ─────────────────────────────────────
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

# ── Provider preprocessing functions ──────────────────────────────────────────
def n4_otsu(sitk_img):
    """N4 with Otsu mask — exact match to provider's bias_correction_sitk."""
    tmp = sitk.Cast(sitk_img, sitk.sitkFloat64)
    arr = sitk.GetArrayFromImage(tmp)
    arr[arr == 0] = np.finfo(float).eps
    tmp2 = sitk.GetImageFromArray(arr)
    tmp2.CopyInformation(sitk_img)
    mask = sitk.OtsuThreshold(tmp2, 0, 1)
    return sitk.N4BiasFieldCorrectionImageFilter().Execute(tmp2, mask)

def clip_nonzero(img, lo=0.1, hi=99.9):
    """Clip to percentile of NON-ZERO voxels — exact match to provider's clip_image_sitk."""
    arr = sitk.GetArrayFromImage(img).ravel()
    arr = arr[arr != 0]
    f = sitk.ClampImageFilter()
    f.SetLowerBound(float(np.percentile(arr, lo)))
    f.SetUpperBound(float(np.percentile(arr, hi)))
    return f.Execute(img)

def preprocess_brats(t1c_path):
    """Full provider pipeline on BraTS NIfTI (0-1800 range, 240×240×155 space)."""
    raw = sitk.ReadImage(str(t1c_path), sitk.sitkFloat32)
    img = n4_otsu(raw)                     # N4 with Otsu mask
    img = clip_nonzero(img)                # clip non-zero percentiles
    arr = sitk.GetArrayFromImage(img).astype('float32')
    arr = rescale_intensity(arr, out_range=(0, 1024))  # rescale to 0-1024
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(raw)
    return out

def find_seg_dir(pid_dir):
    """Handle both 'tumor segmentation' (space) and 'tumor_segmentation' (underscore)."""
    for name in ['tumor segmentation', 'tumor_segmentation']:
        d = pid_dir / name
        if d.exists():
            return d
    return None

def find_mask(pid_dir, tp):
    """Find mask file for a given timepoint."""
    seg_dir = find_seg_dir(pid_dir)
    if seg_dir is None:
        return None
    return next((f for f in seg_dir.glob('*.nii.gz') if tp in f.name), None)

def extract_glcm_for_label(sitk_proc, mask_arr, mask_affine, sitk_ref, label_value):
    """Extract GLCM using a specific label from the mask array."""
    if label_value == 0:
        binary = (mask_arr > 0).astype('int16')   # all labels
    else:
        binary = (mask_arr == label_value).astype('int16')
    if binary.sum() < 5:
        return (np.nan,)*3

    tmp_mk = tempfile.mktemp(suffix='.nii.gz')
    try:
        nib.save(nib.Nifti1Image(binary, mask_affine), tmp_mk)
        mask_s = sitk.Cast(sitk.ReadImage(tmp_mk), sitk.sitkInt32)
        mask_s.CopyInformation(sitk_ref)   # same space as BraTS T1CE

        try:
            bb, _ = imageoperations.checkMask(sitk_proc, mask_s)
            img_c, mask_c = imageoperations.cropToTumorMask(sitk_proc, mask_s, bb)
        except Exception:
            img_c, mask_c = sitk_proc, mask_s

        f = ext.execute(img_c, mask_c)
        return (float(f.get('original_glcm_DifferenceEntropy', np.nan)),
                float(f.get('original_glcm_Contrast',          np.nan)),
                float(f.get('original_glcm_ClusterShade',      np.nan)))
    finally:
        if os.path.exists(tmp_mk): os.unlink(tmp_mk)

# ── Test cases ────────────────────────────────────────────────────────────────
test_cases = [
    ('P01', 'baseline'),  # space folder, 5 timepoints
    ('P06', 'baseline'),  # underscore folder
    ('P29', 'baseline'),  # T1C_HR DICOM (but we use BraTS NIfTI)
    ('P33', 'baseline'),  # T1C_HR DICOM (but we use BraTS NIfTI)
    ('P12', 'fu2'),       # follow-up
]

excel = np.load(EMBS / 'glcm_excel_gt.npz')

print("="*95)
print(f"{'Scan':<18} | {'ExcelDE':>7} {'AllDE':>7} {'ETonlyDE':>9} | "
      f"{'ExcelCo':>9} {'AllCo':>7} {'ETonlyCo':>9} | ET/All voxels")
print("-"*95)

for pid, tp in test_cases:
    key = f'{pid}__{tp}'
    pid_dir   = DATA / pid
    t1c_path  = pid_dir / 'BraTS' / tp / 't1c.nii.gz'
    mask_file = find_mask(pid_dir, tp)

    if not t1c_path.exists():
        print(f"{key:<18} | t1c.nii.gz not found"); continue
    if mask_file is None:
        print(f"{key:<18} | mask not found in {find_seg_dir(pid_dir) or 'NO SEG DIR'}"); continue

    mask_nib = nib.load(str(mask_file))
    mask_arr = mask_nib.get_fdata()
    et_vox   = int((mask_arr == 2).sum())
    all_vox  = int((mask_arr > 0).sum())

    t1c_nib  = nib.load(str(t1c_path))
    print(f"  ⏳ {key}: ET={et_vox}vx  All={all_vox}vx  "
          f"T1C range=[{t1c_nib.get_fdata().min():.0f}, {t1c_nib.get_fdata().max():.0f}]")

    try:
        sitk_raw  = sitk.ReadImage(str(t1c_path), sitk.sitkFloat32)
        sitk_proc = preprocess_brats(t1c_path)

        res_all = extract_glcm_for_label(sitk_proc, mask_arr, mask_nib.affine, sitk_raw, 0)
        res_et  = extract_glcm_for_label(sitk_proc, mask_arr, mask_nib.affine, sitk_raw, 2)

        ex_de = float(excel[key][0]) if key in excel.files else np.nan
        ex_co = float(excel[key][1]) if key in excel.files else np.nan

        print(f"{key:<18} | {ex_de:>7.3f} {res_all[0]:>7.3f} {res_et[0]:>9.3f} | "
              f"{ex_co:>9.1f} {res_all[1]:>7.1f} {res_et[1]:>9.1f} | {et_vox}/{all_vox}")
    except Exception as e:
        print(f"{key:<18} | ERROR: {str(e)[:80]}")

print("="*95)
print("\n  Interpretation:")
print("  ET-only (label=2) values should be close to Excel GT")
print("  DiffEntropy target: within ±0.5  |  Contrast target: within 2×")
