# Objective 1 Implementation Guide: Paper-by-Paper Roadmap
## Developing a Robust Longitudinal Cancer Imaging Pipeline

---

## 📋 TABLE OF CONTENTS

1. [Overview & Learning Path](#overview--learning-path)
2. [Phase 1: Foundation Papers (Week 1)](#phase-1-foundation-papers-week-1)
3. [Phase 2: Longitudinal Registration Papers (Week 2-3)](#phase-2-longitudinal-registration-papers-week-2-3)
4. [Phase 3: Harmonization Papers (Week 3-4)](#phase-3-harmonization-papers-week-3-4)
5. [Phase 4: Advanced Topics (Optional)](#phase-4-advanced-topics-optional)
6. [Complete Resource Links](#complete-resource-links)
7. [Weekly Implementation Checklist](#weekly-implementation-checklist)

---

## 🎯 OVERVIEW & LEARNING PATH

### Objective 1 Goal Reminder
```
Build a reproducible and clinically meaningful data pipeline for processing 
and organizing longitudinal cancer imaging data (MRI or CT).

Sub-objectives:
✓ Acquire and curate public datasets (BraTS, Yale-Brain-Mets, UCSF, LUMIERE)
✓ Standardized preprocessing (normalization, spatial alignment, temporal consistency)
✓ Structure data for temporal and generative modeling (Objectives 2-5)
```

### The Learning Journey
```
Week 1: FOUNDATION
├─ Understand what "good" preprocessing looks like (BraTS standard)
├─ Learn automated preprocessing philosophy (nnU-Net)
└─ Get familiar with medical imaging libraries (MONAI, SimpleITK)

Week 2-3: TEMPORAL ALIGNMENT
├─ Learn longitudinal registration techniques
├─ Understand tumor-preserving registration
└─ Implement temporal consistency checks

Week 3-4: HARMONIZATION
├─ Handle multi-scanner variability
├─ Preserve temporal trajectories
└─ Finalize preprocessing pipeline

Week 4: VALIDATION & ORGANIZATION
├─ Quality control implementation
├─ Data structure for Objectives 2-5
└─ Documentation and deliverables
```

---

## 📚 PHASE 1: FOUNDATION PAPERS (WEEK 1)

### Paper 1: BraTS Toolkit - Your Preprocessing Bible

#### 📄 Paper Details
- **Title:** "BraTS Toolkit: Translating BraTS Brain Tumor Segmentation Algorithms Into Clinical Practice"
- **Year:** 2020
- **Authors:** Kofler et al.
- **Journal:** Frontiers in Neuroscience
- **Link:** https://www.frontiersin.org/articles/10.3389/fnins.2020.00125/full
- **GitHub:** https://github.com/neuronflow/BraTS-Toolkit
- **Status:** ✅ Open Access, ✅ Public Code

#### 🎯 What This Paper Does for You

**For Objective 1 (Primary Use):**
- Provides a **complete reference implementation** of medical image preprocessing
- Shows the **exact steps** needed: DICOM→NIfTI, co-registration, skull-stripping, normalization
- Gives you **validated code** that you can compare your implementation against
- Documents **quality control procedures** used in BraTS challenge

**Alignment with Other Objectives:**
- **Objective 2 (ViT Training):** Preprocessing quality directly affects ViT performance
- **Objective 3 (LLM):** Standardized format enables consistent feature extraction
- **Objective 4 (Video):** Spatial alignment is prerequisite for temporal consistency
- **Objective 5 (Evaluation):** BraTS provides benchmark metrics for comparison

#### 📖 What to Read (Priority Sections)

**Essential Sections (Read First):**
1. **Introduction** - Understand why preprocessing matters
2. **Section 2.2: Preprocessing Pipeline** - The core workflow you'll implement
3. **Figure 2** - Visual diagram of the complete pipeline
4. **Section 2.4: Quality Control** - How to validate your outputs
5. **Supplementary Material** - Detailed parameter settings

**Skim/Reference:**
- Section 3: Results (you'll refer to this when validating)
- Section 4: Discussion (helpful for understanding limitations)

**Reading Time:** 2-3 hours (deep read of sections 2.2, 2.4)

#### 💡 Key Takeaways & Implementation

**Core Preprocessing Steps You'll Implement:**

```python
# Based on BraTS Toolkit workflow

preprocessing_pipeline = {
    "step_1_format_conversion": {
        "action": "Convert DICOM to NIfTI (if needed)",
        "note": "Yale data already in NIfTI - skip this!",
        "code": "dcm2niix or nibabel"
    },
    
    "step_2_coregistration": {
        "action": "Register all modalities (T1, T1CE, T2, FLAIR) to same space",
        "reference": "Use T1CE as reference (best contrast)",
        "algorithm": "Rigid registration (6 DOF)",
        "tool": "SimpleITK or ANTs"
    },
    
    "step_3_skull_stripping": {
        "action": "Remove non-brain tissue",
        "method": "HD-BET (deep learning) or FSL BET",
        "why": "Focus on brain, reduce computation",
        "code": "HD-BET via BraTS Toolkit"
    },
    
    "step_4_resampling": {
        "action": "Standardize to 1mm³ isotropic resolution",
        "target": "(1.0, 1.0, 1.0) mm spacing",
        "interpolation": "Linear for images, Nearest for masks",
        "tool": "SimpleITK.Resample or MONAI Spacingd"
    },
    
    "step_5_normalization": {
        "action": "Standardize intensity values",
        "method": "Z-score normalization OR min-max to [0,1]",
        "important": "Apply per-scan, not across dataset",
        "code": "ScaleIntensityRanged from MONAI"
    },
    
    "step_6_cropping": {
        "action": "Remove empty background voxels",
        "method": "Bounding box around brain",
        "benefit": "Reduces file size, speeds training",
        "code": "CropForegroundd from MONAI"
    }
}
```

**What You'll Implement This Week:**

```python
# Week 1 Implementation: BraTS preprocessing pipeline

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, 
    CropForegroundd, Compose
)
import SimpleITK as sitk

# 1. Define preprocessing transform
brats_preprocessing = Compose([
    LoadImaged(keys=["T1", "T1CE", "T2", "FLAIR", "seg"]),
    EnsureChannelFirstd(keys=["T1", "T1CE", "T2", "FLAIR", "seg"]),
    Orientationd(keys=["T1", "T1CE", "T2", "FLAIR", "seg"], axcodes="RAS"),
    Spacingd(
        keys=["T1", "T1CE", "T2", "FLAIR", "seg"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")
    ),
    ScaleIntensityRanged(
        keys=["T1", "T1CE", "T2", "FLAIR"],
        a_min=0, a_max=None,  # Computed per-channel
        b_min=0, b_max=1,
        clip=True
    ),
    CropForegroundd(
        keys=["T1", "T1CE", "T2", "FLAIR", "seg"],
        source_key="T1CE"  # Use T1CE for foreground detection
    )
])

# 2. Test on one BraTS case
test_case = {
    "T1": "path/to/BraTS_001_t1.nii.gz",
    "T1CE": "path/to/BraTS_001_t1ce.nii.gz",
    "T2": "path/to/BraTS_001_t2.nii.gz",
    "FLAIR": "path/to/BraTS_001_flair.nii.gz",
    "seg": "path/to/BraTS_001_seg.nii.gz"
}

preprocessed = brats_preprocessing(test_case)

# 3. Validate against BraTS Toolkit output
# Download official BraTS preprocessed data and compare
```

#### 🔧 What You Can Improve

**BraTS Toolkit Limitations → Your Improvements:**

1. **Limitation:** BraTS Toolkit requires manual configuration
   - **Your Improvement:** Automate parameter selection using nnU-Net fingerprinting (see Paper 2)

2. **Limitation:** No temporal registration (BraTS is single timepoint)
   - **Your Improvement:** Add longitudinal registration for Yale data (see Phase 2 papers)

3. **Limitation:** CPU-only implementation (slow)
   - **Your Improvement:** Use GPU-accelerated MONAI transforms (10-50× faster)

4. **Limitation:** Requires Docker setup
   - **Your Improvement:** Pure Python implementation using MONAI + SimpleITK

5. **Limitation:** Limited to brain imaging
   - **Your Improvement:** Generalize pipeline to work with other anatomies (if expanding to ISPY1 breast data)

#### ✅ Success Criteria (By End of Week 1)

```python
# You should be able to:

def validate_brats_preprocessing(your_output, official_brats_output):
    """
    Compare your preprocessing to BraTS Toolkit
    """
    checks = {
        "resolution": check_spacing(your_output) == (1.0, 1.0, 1.0),
        "orientation": check_orientation(your_output) == "RAS",
        "intensity_range": check_intensity_range(your_output, min=0, max=1),
        "spatial_alignment": dice_score(your_output, official_brats_output) > 0.99,
        "file_size": check_cropped_properly(your_output)
    }
    
    return all(checks.values())
```

#### 🔗 Resources & Links

- **Paper PDF:** https://www.frontiersin.org/articles/10.3389/fnins.2020.00125/full
- **GitHub Code:** https://github.com/neuronflow/BraTS-Toolkit
- **BraTS Dataset:** http://braintumorsegmentation.org/
- **HD-BET (Skull-stripping):** https://github.com/MIC-DKFZ/HD-BET
- **MONAI Documentation:** https://docs.monai.io/en/stable/transforms.html
- **SimpleITK Tutorials:** https://simpleitk.readthedocs.io/en/master/examples.html

---

### Paper 2: nnU-Net - The Automated Preprocessing Philosophy

#### 📄 Paper Details
- **Title:** "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
- **Year:** 2020 (updated 2021)
- **Authors:** Isensee et al.
- **Journal:** Nature Methods
- **Link:** https://www.nature.com/articles/s41592-020-01008-z
- **GitHub:** https://github.com/MIC-DKFZ/nnUNet
- **Status:** ✅ Open Access (preprint), ✅ Public Code

#### 🎯 What This Paper Does for You

**For Objective 1 (Primary Use):**
- Teaches you **automated preprocessing** - no manual parameter tuning!
- Introduces **dataset fingerprinting** - analyze your data to determine optimal settings
- Provides **state-of-the-art preprocessing strategies** validated across 50+ datasets
- Shows how to handle **different image properties** automatically

**Alignment with Other Objectives:**
- **Objective 2 (ViT Training):** nnU-Net preprocessing is proven optimal for medical image analysis
- **Objective 3 (LLM):** Consistent preprocessing enables reliable feature extraction
- **Objective 4 (Video):** Standardization crucial for temporal consistency
- **Objective 5 (Evaluation):** nnU-Net provides benchmark results for comparison

#### 📖 What to Read (Priority Sections)

**Essential Sections:**
1. **Box 1: The nnU-Net pipeline** - 3-step overview (fingerprint → configure → train)
2. **Figure 1** - Complete workflow diagram
3. **Section: Automated preprocessing** - How it determines optimal settings
4. **Section: Resampling strategy** - Critical for your spatial standardization
5. **Supplementary Note 1** - Detailed preprocessing algorithm

**Advanced (Optional for Week 1):**
- Network architecture details (you'll use ViT, not U-Net)
- Training strategies (relevant for Objective 2)

**Reading Time:** 2-3 hours (focus on preprocessing sections)

#### 💡 Key Takeaways & Implementation

**The "Dataset Fingerprint" Concept:**

```python
# nnU-Net analyzes your dataset to automatically determine settings

def compute_dataset_fingerprint(dataset):
    """
    Analyze dataset properties to guide preprocessing
    """
    fingerprint = {
        "spacing": {
            "median": np.median([scan.spacing for scan in dataset]),
            "min": np.min([scan.spacing for scan in dataset]),
            "max": np.max([scan.spacing for scan in dataset])
        },
        "sizes": {
            "median_shape": np.median([scan.shape for scan in dataset], axis=0),
            "size_reduction_needed": check_if_too_large(dataset)
        },
        "intensities": {
            "modality": detect_modality(dataset),  # CT vs MRI
            "percentiles": compute_percentiles(dataset)
        },
        "anisotropy": {
            "is_anisotropic": check_anisotropy(dataset),
            "anisotropy_axis": identify_anisotropic_axis(dataset)
        }
    }
    
    return fingerprint

# Example: Yale-Brain-Mets fingerprint
yale_fingerprint = {
    "spacing": {"median": (1.0, 1.0, 3.0)},  # Anisotropic!
    "sizes": {"median_shape": (256, 256, 60)},
    "intensities": {"modality": "MRI_T1CE"},
    "anisotropy": {"is_anisotropic": True, "axis": 2}
}

# nnU-Net decision: Resample to isotropic 1mm³? 
# OR keep anisotropic and use 3D low-res + 2D high-res approach?
```

**What You'll Implement:**

```python
# Week 1: Add nnU-Net-inspired automatic configuration to your pipeline

def configure_preprocessing_automatically(dataset_path):
    """
    Automatically determine optimal preprocessing settings
    """
    # 1. Compute dataset fingerprint
    fingerprint = analyze_dataset(dataset_path)
    
    # 2. Determine resampling target
    if fingerprint['anisotropy']['is_anisotropic']:
        # Yale data is anisotropic (1mm × 1mm × 3mm typical)
        target_spacing = determine_target_spacing(fingerprint)
        # Decision: Resample to isotropic (1, 1, 1) for your ViT
    else:
        target_spacing = (1.0, 1.0, 1.0)
    
    # 3. Determine normalization strategy
    if fingerprint['intensities']['modality'] == 'CT':
        normalization = 'clip_to_HU_range'
    else:  # MRI
        normalization = 'z_score_per_channel'
    
    # 4. Determine if cropping needed
    crop_to_nonzero = fingerprint['has_large_background']
    
    return {
        'target_spacing': target_spacing,
        'normalization': normalization,
        'crop': crop_to_nonzero
    }
```

#### 🔧 What You Can Improve Based on nnU-Net

**nnU-Net Insights → Your Improvements:**

1. **Insight:** Automatic parameter selection beats manual tuning
   - **Your Implementation:** Create dataset analyzer for Yale/BraTS
   
2. **Insight:** Different modalities need different normalization
   - **Your Implementation:** Auto-detect modality and apply appropriate normalization

3. **Insight:** Patch-based training for large images
   - **Your Implementation:** Use for ViT training (Objective 2) if images too large

4. **Insight:** Anisotropic data needs special handling
   - **Your Implementation:** Resample Yale's anisotropic scans to isotropic for ViT

5. **Insight:** Foreground cropping reduces memory 10-100×
   - **Your Implementation:** Always crop to brain region

**Example: Handling Anisotropic Yale Data**

```python
# Yale scans often: 0.5mm × 0.5mm × 3mm (anisotropic in z-axis)

# Option 1: Resample to isotropic (nnU-Net recommendation for 3D models)
target_spacing = (1.0, 1.0, 1.0)  # Isotropic
# Pro: ViT can see all directions equally
# Con: Upsampling z-axis by 3× (may introduce artifacts)

# Option 2: Keep anisotropic
target_spacing = (1.0, 1.0, 3.0)  # Keep original z-spacing
# Pro: No artificial upsampling
# Con: ViT sees lower resolution in z

# YOUR DECISION for ViT (Objective 2):
# → Choose Option 1 (isotropic) because ViT benefits from consistent spatial resolution
# → Validate by checking if tumor features preserved after resampling
```

#### ✅ Success Criteria

By end of Week 1, you should have:
- [ ] Automated configuration script that analyzes dataset properties
- [ ] Preprocessing pipeline that adapts to different spacing/modalities
- [ ] Validation that shows your pipeline matches nnU-Net quality

#### 🔗 Resources & Links

- **Paper (Nature Methods):** https://www.nature.com/articles/s41592-020-01008-z
- **arXiv Preprint (Free):** https://arxiv.org/abs/1809.10486
- **GitHub:** https://github.com/MIC-DKFZ/nnUNet
- **Documentation:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
- **Tutorial Videos:** Search "nnU-Net tutorial" on YouTube

---

### Paper 3: BraTS 2021-2025 Evolution - Understanding Standards

#### 📄 Paper Details
- **Title:** "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification"
- **Year:** 2021 (updated annually)
- **Authors:** Baid et al.
- **Journal:** arXiv (conference proceedings)
- **Link:** https://arxiv.org/abs/2107.02314
- **Status:** ✅ Open Access

#### 🎯 What This Paper Does for You

**For Objective 1 (Primary Use):**
- Documents **quality control procedures** for medical imaging datasets
- Shows **evolution of preprocessing standards** (2012→2025)
- Explains **multi-institutional data challenges** and solutions
- Provides **validation metrics** you should track

**Alignment with Other Objectives:**
- **Objective 2:** Dataset size evolution (660→2,040 cases) shows scale needed for deep learning
- **Objective 5:** Evaluation metrics used in BraTS challenge

#### 📖 What to Read

**Essential Sections:**
1. **Section 2: Data** - Dataset composition and preprocessing
2. **Section 3.1: Preprocessing Pipeline** - Current BraTS standard
3. **Table 1** - Dataset statistics across years
4. **Section 5: Quality Control** - How to validate imaging data

**Reading Time:** 1-2 hours

#### 💡 Key Takeaways

**Quality Control Checklist (Implement in Week 4):**

```python
# Based on BraTS QC procedures

def quality_control_pipeline(preprocessed_scan):
    """
    Comprehensive QC following BraTS standards
    """
    qc_report = {}
    
    # 1. Resolution check
    qc_report['resolution_correct'] = (
        preprocessed_scan.spacing == (1.0, 1.0, 1.0)
    )
    
    # 2. Orientation check
    qc_report['orientation_correct'] = (
        preprocessed_scan.direction == RAS_DIRECTION
    )
    
    # 3. Intensity range check
    qc_report['intensity_normalized'] = (
        0 <= preprocessed_scan.min() <= preprocessed_scan.max() <= 1
    )
    
    # 4. Artifact detection
    qc_report['no_motion_artifacts'] = detect_motion_artifacts(preprocessed_scan)
    qc_report['no_intensity_artifacts'] = detect_intensity_outliers(preprocessed_scan)
    
    # 5. Alignment check (for longitudinal data)
    if preprocessed_scan.has_previous_timepoint:
        qc_report['temporal_alignment_quality'] = (
            compute_mutual_information(current, previous) > threshold
        )
    
    # 6. Completeness check
    qc_report['all_modalities_present'] = (
        has_T1 and has_T1CE and has_T2 and has_FLAIR
    )
    
    # 7. File size check (indicates proper cropping)
    qc_report['properly_cropped'] = (
        10_000_000 < preprocessed_scan.file_size < 100_000_000  # 10-100 MB
    )
    
    return qc_report
```

#### 🔧 What You Can Improve

**BraTS Evolution Shows:**
- Quality control became more rigorous over time
- Multi-institutional data requires harmonization (see Phase 3)
- Dataset size matters (2,040 cases for robust models)

**Your Improvements:**
- Implement automated QC from day 1 (don't wait for errors)
- Track QC metrics across all patients
- Flag outliers for manual review

#### 🔗 Resources

- **BraTS 2021 Paper:** https://arxiv.org/abs/2107.02314
- **BraTS Website:** http://braintumorsegmentation.org/
- **BraTS Challenge Papers:** Search "BraTS MICCAI" for yearly updates

---

## 📚 PHASE 2: LONGITUDINAL REGISTRATION PAPERS (WEEK 2-3)

### Paper 4: Treatment-Aware Longitudinal Registration (CRITICAL!)

#### 📄 Paper Details
- **Title:** "Treatment-Aware Longitudinal Registration for Breast DCE-MRI"
- **Year:** 2024
- **Authors:** Wang et al.
- **Conference:** MICCAI 2024
- **Link:** https://link.springer.com/chapter/10.1007/978-3-031-72104-5_60
- **GitHub:** ✅ https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration
- **Status:** ✅ Public Code Available

#### 🎯 What This Paper Does for You

**For Objective 1 (CRITICAL for Longitudinal Data):**
- Solves the **tumor-preserving registration problem**
- Prevents registration from **"hiding" real tumor changes**
- Provides **conditional pyramid registration** framework
- Shows how to **track treatment response** accurately

**This is THE MOST IMPORTANT paper for your longitudinal Yale data!**

**Alignment with Other Objectives:**
- **Objective 2:** Proper registration ensures ViT learns biological changes, not artifacts
- **Objective 3:** Accurate change quantification for LLM narratives
- **Objective 4:** Temporal consistency in generated videos depends on registration quality
- **Objective 5:** Ground truth for evaluating progression predictions

#### 📖 What to Read (Priority Sections)

**MUST READ (Critical Understanding):**
1. **Abstract** - The core problem and solution
2. **Figure 1** - Visual explanation of the problem
3. **Section 2.1: Problem Formulation** - Why standard registration fails
4. **Section 2.2: Treatment-Aware Registration** - The solution
5. **Figure 2** - Network architecture (you'll adapt this)
6. **Section 3: Experiments** - Validation approach

**Reading Time:** 3-4 hours (this is critical, read carefully!)

#### 💡 Key Takeaways & Implementation

**THE PROBLEM: Standard Registration Hides Real Tumor Changes**

```python
# Why standard registration FAILS for longitudinal tumor imaging:

# Scenario: Patient with breast tumor
baseline_scan = patient.scan_at_day_0      # Tumor is 20mm
followup_scan = patient.scan_at_day_42     # After chemo, tumor is 10mm (responding!)

# Standard deformable registration:
registered_followup = standard_registration(baseline_scan, followup_scan)

# What happens:
# Registration algorithm tries to make followup_scan LOOK LIKE baseline_scan
# It literally WARPS the 10mm tumor to appear 20mm!
# Result: You measure 20mm at both timepoints → "no change" (WRONG!)

# The tumor DID shrink, but registration hid it!
```

**Visualization of the Problem:**

```
TIME 0 (Baseline):        TIME 1 (After Treatment):
┌─────────────┐           ┌─────────────┐
│             │           │             │
│   ████      │  ───→     │   ██        │  (Tumor shrunk!)
│   ████      │           │   ██        │
│             │           │             │
└─────────────┘           └─────────────┘
   20mm tumor                10mm tumor

After STANDARD registration:
┌─────────────┐
│             │
│   ████      │  ← Registration WARPED it back to 20mm!
│   ████      │     (Algorithm doesn't know tumor changes are REAL)
│             │
└─────────────┘

After TREATMENT-AWARE registration:
┌─────────────┐
│             │
│   ██        │  ← Preserved the 10mm size (correct!)
│   ██        │     (Algorithm knows tumor changes are biological)
│             │
└─────────────┘
```

**THE SOLUTION: Conditional Pyramid Registration**

```python
# Treatment-aware registration approach

def treatment_aware_registration(baseline, followup, treatment_info):
    """
    Preserves tumor volume changes while aligning anatomy
    """
    
    # 1. Predict tumor location in both scans
    tumor_mask_baseline = segment_tumor(baseline)
    tumor_mask_followup = segment_tumor(followup)
    
    # 2. Extract treatment information
    treatment_type = treatment_info['type']  # 'chemotherapy', 'surgery', 'radiation'
    expected_response = treatment_info['expected_response']  # 'shrinkage', 'growth', 'stable'
    
    # 3. Conditional pyramid registration
    # Key idea: Register AROUND the tumor, not THROUGH it
    
    # 3a. Coarse level: Align brain anatomy (excluding tumor)
    brain_mask = (1 - tumor_mask_baseline) * (1 - tumor_mask_followup)
    coarse_transform = register(
        baseline * brain_mask,
        followup * brain_mask,
        level='coarse'
    )
    
    # 3b. Fine level: Refine alignment, still avoiding tumor
    fine_transform = register(
        baseline * brain_mask,
        followup * brain_mask,
        level='fine',
        initial=coarse_transform
    )
    
    # 3c. Apply transform to whole image
    registered_followup = apply_transform(followup, fine_transform)
    
    # 4. Verify tumor volume preserved
    original_tumor_volume = compute_volume(followup, tumor_mask_followup)
    registered_tumor_volume = compute_volume(registered_followup, tumor_mask_followup)
    
    assert abs(original_tumor_volume - registered_tumor_volume) < 0.5  # <0.5cc error
    
    return registered_followup, fine_transform
```

**What You'll Implement (Week 2-3):**

```python
# Simplified version for Yale-Brain-Mets data

import SimpleITK as sitk
import numpy as np

def yale_longitudinal_registration(baseline_scan, followup_scan, 
                                   baseline_tumor_mask=None,
                                   followup_tumor_mask=None):
    """
    Register Yale longitudinal scans while preserving tumor changes
    
    Args:
        baseline_scan: T0 (pre-treatment)
        followup_scan: T1 (post-treatment)
        baseline_tumor_mask: Tumor segmentation at T0 (optional, will auto-detect if None)
        followup_tumor_mask: Tumor segmentation at T1 (optional)
    """
    
    # Step 1: Detect tumors if masks not provided
    if baseline_tumor_mask is None:
        baseline_tumor_mask = auto_detect_tumor(baseline_scan)
    if followup_tumor_mask is None:
        followup_tumor_mask = auto_detect_tumor(followup_scan)
    
    # Step 2: Create brain-only masks (exclude tumor regions)
    baseline_brain_only = baseline_scan * (1 - baseline_tumor_mask)
    followup_brain_only = followup_scan * (1 - followup_tumor_mask)
    
    # Step 3: Setup registration (AFFINE only - preserves volumes!)
    registration = sitk.ImageRegistrationMethod()
    
    # Metric: Mutual Information (works for MRI)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    
    # Optimizer: LBFGS-B (fast, reliable)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100
    )
    
    # Transform: AFFINE (rigid + scaling, NO deformable warping)
    initial_transform = sitk.CenteredTransformInitializer(
        baseline_brain_only, 
        followup_brain_only,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform)
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Step 4: Execute registration on brain-only images
    print("Registering brain anatomy (excluding tumors)...")
    final_transform = registration.Execute(
        sitk.Cast(baseline_brain_only, sitk.sitkFloat32),
        sitk.Cast(followup_brain_only, sitk.sitkFloat32)
    )
    
    print(f"Registration metric value: {registration.GetMetricValue()}")
    print(f"Optimizer iterations: {registration.GetOptimizerIteration()}")
    
    # Step 5: Apply transform to FULL followup scan (including tumor)
    registered_followup = sitk.Resample(
        followup_scan,
        baseline_scan,  # Reference image
        final_transform,
        sitk.sitkLinear,
        0.0,  # Default pixel value
        followup_scan.GetPixelID()
    )
    
    # Step 6: Verify tumor volume preservation
    original_volume = compute_tumor_volume(followup_scan, followup_tumor_mask)
    registered_volume = compute_tumor_volume(registered_followup, followup_tumor_mask)
    volume_error = abs(original_volume - registered_volume) / original_volume
    
    print(f"Original tumor volume: {original_volume:.2f} cc")
    print(f"Registered tumor volume: {registered_volume:.2f} cc")
    print(f"Volume preservation error: {volume_error*100:.2f}%")
    
    if volume_error > 0.05:  # >5% error is concerning
        print("⚠️ WARNING: Large volume change detected. Check registration quality.")
    
    return registered_followup, final_transform


def compute_tumor_volume(scan, mask):
    """Calculate tumor volume in cubic centimeters"""
    scan_array = sitk.GetArrayFromImage(scan)
    mask_array = sitk.GetArrayFromImage(mask)
    spacing = scan.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
    num_voxels = np.sum(mask_array > 0)
    volume_mm3 = num_voxels * voxel_volume
    volume_cc = volume_mm3 / 1000.0  # Convert mm³ to cc
    return volume_cc
```

#### 🔧 What You Can Improve

**Paper's Approach → Your Enhancements:**

1. **Paper uses:** Deep learning registration network
   - **Your approach:** Start with SimpleITK (simpler, more reliable for smaller datasets)
   - **Future improvement:** Implement deep learning version if SimpleITK insufficient

2. **Paper focuses on:** Breast cancer
   - **Your adaptation:** Brain metastases (different anatomy, but same principles)

3. **Paper uses:** Manual tumor annotations
   - **Your improvement:** Auto-detect tumors using pre-trained segmentation model

4. **Paper validates on:** 314 patients
   - **Your scale:** 1,430 patients (Yale dataset)

5. **Paper's limitation:** Requires paired scans
   - **Your handling:** Yale provides multiple timepoints, process sequentially

#### ✅ Success Criteria (Week 2-3)

By end of Week 3, you should have:
- [ ] Implemented tumor-preserving registration
- [ ] Validated on 10 Yale patients with known treatment response
- [ ] Verified tumor volumes preserved within 5% error
- [ ] Documented registration quality metrics

#### 🔗 Resources & Links

- **Paper:** https://link.springer.com/chapter/10.1007/978-3-031-72104-5_60
- **GitHub Code:** https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration
- **SimpleITK Registration:** https://simpleitk.readthedocs.io/en/master/registrationOverview.html
- **Tutorial on Medical Image Registration:** https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks

---

### Paper 5: FLIRE - Fast Longitudinal Registration

#### 📄 Paper Details
- **Title:** "FLIRE: Fast Longitudinal Image Registration for Breast MRI"
- **Year:** 2024
- **Authors:** Tong et al.
- **Conference:** Medical Imaging Meets NeurIPS Workshop
- **GitHub:** ✅ https://github.com/michelle-tong18/FLIRE-MRI-registration
- **Status:** ✅ Public Code

#### 🎯 What This Paper Does for You

**For Objective 1 (Speed Optimization):**
- Achieves **9× faster registration** than standard methods
- Maintains **high accuracy** (Pearson correlation 0.98)
- Provides **efficient implementation** for large datasets

**Alignment with Other Objectives:**
- **Objective 4:** Fast registration enables processing thousands of scans for video model training

#### 📖 What to Read

**Essential Sections:**
1. **Abstract** - Speed vs accuracy tradeoff
2. **Section 2: Method** - Fast registration algorithm
3. **Table 1** - Runtime comparisons
4. **GitHub README** - Implementation details

**Reading Time:** 1-2 hours

#### 💡 Key Takeaways

**Speed Optimization Techniques:**

```python
# FLIRE speed optimizations you can adopt

# 1. Multi-resolution pyramid (coarse-to-fine)
def fast_registration_pyramid(baseline, followup):
    """
    Register at multiple resolutions for speed
    """
    # Level 1: 8× downsampled (very fast, rough alignment)
    coarse_transform = register_at_resolution(baseline, followup, factor=8)
    
    # Level 2: 4× downsampled (refine)
    medium_transform = register_at_resolution(
        baseline, followup, factor=4, 
        initial=coarse_transform
    )
    
    # Level 3: 2× downsampled (fine-tune)
    fine_transform = register_at_resolution(
        baseline, followup, factor=2,
        initial=medium_transform
    )
    
    # Final: Full resolution (only if needed)
    if needs_full_resolution:
        final_transform = register_at_resolution(
            baseline, followup, factor=1,
            initial=fine_transform
        )
    else:
        final_transform = fine_transform
    
    return final_transform

# 2. GPU acceleration
# Use CuPy or PyTorch for GPU-accelerated registration

# 3. Batch processing
# Process multiple patients in parallel
from multiprocessing import Pool

def register_patient(patient_id):
    # Register all timepoints for one patient
    return registered_sequence

with Pool(processes=8) as pool:
    results = pool.map(register_patient, patient_ids)
```

#### 🔧 Your Implementation Strategy

**Week 3: Optimize your registration pipeline**

```python
# Start: Slow registration (SimpleITK, CPU-only)
# Runtime: ~5 minutes per patient × 1,430 patients = 119 hours!

# Optimization 1: Multi-resolution pyramid
# Runtime: ~2 minutes per patient × 1,430 = 47 hours (2.5× faster)

# Optimization 2: Parallel processing (8 cores)
# Runtime: 47 hours / 8 = 6 hours (20× faster than original!)

# Optimization 3: GPU acceleration (if available)
# Runtime: ~30 seconds per patient × 1,430 / 8 cores = 1.5 hours
```

#### 🔗 Resources

- **GitHub:** https://github.com/michelle-tong18/FLIRE-MRI-registration
- **Paper:** Search "FLIRE Medical Imaging Meets NeurIPS 2024"

---

### Paper 6: Longitudinal Brain Metastases Dataset

#### 📄 Paper Details
- **Title:** "A longitudinal MRI dataset of brain metastases"
- **Year:** 2025
- **Authors:** Shapey et al.
- **Journal:** Scientific Data
- **Link:** https://www.nature.com/articles/s41597-024-04186-3
- **Dataset:** https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/
- **Status:** ✅ Open Access Paper & Dataset

#### 🎯 What This Paper Does for You

**For Objective 1 (Data Understanding):**
- Describes **Yale dataset structure** (your primary longitudinal dataset!)
- Documents **clinical metadata** available
- Explains **temporal sampling** strategy
- Provides **data organization** blueprint

**This paper describes YOUR PRIMARY DATASET!**

**Alignment with Other Objectives:**
- **Objective 3:** Clinical metadata structure for LLM integration
- **Objective 4:** Temporal intervals inform video generation
- **Objective 5:** Validation approach and metrics

#### 📖 What to Read (MUST READ!)

**ESSENTIAL (Read Completely):**
1. **Abstract** - Dataset overview
2. **Background & Summary** - Why this dataset was created
3. **Methods** - Data acquisition and organization
4. **Data Records** - File structure and naming conventions
5. **Table 1** - Patient demographics and scan statistics
6. **Figure 2** - Example longitudinal sequence
7. **Technical Validation** - Quality control procedures

**Reading Time:** 2-3 hours (read thoroughly!)

#### 💡 Key Takeaways & Implementation

**Understanding Yale Dataset Structure:**

```python
# Yale-Brain-Mets-Longitudinal file naming convention

# Format: {PatientID}_{DateTime}_{Sequence}.nii.gz
# Example: YALE_0001_2015-03-15-14-30_T1CE.nii.gz

# Breaking it down:
file_structure = {
    "patient_id": "YALE_0001",           # Unique patient identifier
    "datetime": "2015-03-15-14-30",      # Scan date and time
    "sequence": "T1CE",                   # MRI sequence type
    "format": ".nii.gz"                   # Compressed NIfTI
}

# Temporal organization:
patient_timeline = {
    "YALE_0001": {
        "timepoint_0": {
            "date": "2015-03-15",
            "scans": ["T1", "T1CE", "T2", "FLAIR"],
            "treatment_status": "pre-treatment",
            "days_from_baseline": 0
        },
        "timepoint_1": {
            "date": "2015-04-26",
            "scans": ["T1", "T1CE", "T2", "FLAIR"],
            "treatment_status": "6-weeks-post-SRS",
            "days_from_baseline": 42
        },
        "timepoint_2": {
            "date": "2015-07-20",
            "scans": ["T1", "T1CE", "T2", "FLAIR"],
            "treatment_status": "follow-up",
            "days_from_baseline": 127
        }
    }
}
```

**Implementing Yale Data Loader:**

```python
import os
import glob
from datetime import datetime
from collections import defaultdict
import nibabel as nib

def load_yale_dataset(dataset_root):
    """
    Load and organize Yale-Brain-Mets-Longitudinal dataset
    """
    # 1. Scan all files
    all_files = glob.glob(os.path.join(dataset_root, "*.nii.gz"))
    
    # 2. Organize by patient
    patients = defaultdict(lambda: defaultdict(dict))
    
    for filepath in all_files:
        # Parse filename
        filename = os.path.basename(filepath)
        parts = filename.replace('.nii.gz', '').split('_')
        
        patient_id = f"{parts[0]}_{parts[1]}"  # e.g., YALE_0001
        datetime_str = parts[2]  # e.g., 2015-03-15-14-30
        sequence = parts[3]  # e.g., T1CE
        
        # Convert datetime string to date object
        date = datetime.strptime(datetime_str.split('-')[0:3], '%Y-%m-%d').date()
        
        # Store
        patients[patient_id][date][sequence] = filepath
    
    # 3. Sort timepoints for each patient
    organized_dataset = {}
    for patient_id, timepoints in patients.items():
        sorted_timepoints = sorted(timepoints.items(), key=lambda x: x[0])
        
        # Calculate days from baseline
        baseline_date = sorted_timepoints[0][0]
        
        patient_data = []
        for date, scans in sorted_timepoints:
            days_from_baseline = (date - baseline_date).days
            
            timepoint = {
                'date': date,
                'days_from_baseline': days_from_baseline,
                'scans': scans,
                'num_modalities': len(scans)
            }
            patient_data.append(timepoint)
        
        organized_dataset[patient_id] = patient_data
    
    return organized_dataset


# Usage:
dataset = load_yale_dataset('/data/Yale-Brain-Mets-Longitudinal')

# Filter patients with ≥3 timepoints (good for longitudinal modeling)
longitudinal_patients = {
    pid: data for pid, data in dataset.items()
    if len(data) >= 3
}

print(f"Total patients: {len(dataset)}")
print(f"Patients with ≥3 timepoints: {len(longitudinal_patients)}")
```

**Clinical Metadata Integration:**

```python
# Yale provides clinical metadata (download separately from TCIA)

def integrate_clinical_metadata(imaging_data, clinical_csv):
    """
    Merge imaging data with clinical information
    """
    import pandas as pd
    
    clinical_df = pd.read_csv(clinical_csv)
    
    for patient_id in imaging_data.keys():
        # Get clinical info
        patient_clinical = clinical_df[clinical_df['PatientID'] == patient_id]
        
        # Add to imaging data structure
        imaging_data[patient_id]['clinical'] = {
            'age': patient_clinical['Age'].values[0],
            'sex': patient_clinical['Sex'].values[0],
            'primary_cancer': patient_clinical['PrimaryCancer'].values[0],
            'treatment_history': patient_clinical['TreatmentHistory'].values[0]
        }
    
    return imaging_data
```

#### 🔧 What You Can Improve

**Dataset Paper → Your Enhancements:**

1. **Paper provides:** Raw NIfTI files
   - **Your addition:** Preprocessed, registered versions

2. **Paper provides:** Clinical metadata in CSV
   - **Your addition:** Integrated JSON structure for LLM

3. **Paper provides:** Multiple timepoints
   - **Your analysis:** Compute temporal statistics (intervals, missing data patterns)

4. **Paper limitation:** No segmentation masks
   - **Your solution:** Auto-segment tumors using nnU-Net or similar

#### ✅ Success Criteria

By end of Week 2, you should:
- [ ] Have Yale dataset downloaded and organized
- [ ] Understand file naming and structure
- [ ] Know which patients have ≥3 timepoints
- [ ] Have clinical metadata integrated

#### 🔗 Resources

- **Paper:** https://www.nature.com/articles/s41597-024-04186-3
- **Dataset:** https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/
- **TCIA Download Tool:** https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever

---

## 📚 PHASE 3: HARMONIZATION PAPERS (WEEK 3-4)

### Paper 7: ComBat Harmonization

#### 📄 Paper Details
- **Title:** "Harmonization of multi-site diffusion tensor imaging data"
- **Year:** 2017 (foundational), updated implementations 2020-2022
- **Authors:** Fortin et al.
- **Journal:** NeuroImage
- **Link:** https://www.sciencedirect.com/science/article/pii/S1053811917306948
- **GitHub:** ✅ https://github.com/Jfortin1/neuroCombat (R version)
- **Python:** ✅ `pip install neuroCombat`
- **Status:** ✅ Open Access, ✅ Public Code

#### 🎯 What This Paper Does for You

**For Objective 1 (Multi-Scanner Harmonization):**
- Removes **scanner-specific artifacts** from imaging features
- Enables **multi-institutional data** combination
- Preserves **biological variability** while removing technical artifacts

**Alignment with Other Objectives:**
- **Objective 2:** Harmonized features improve ViT training (model learns biology, not scanner differences)
- **Objective 3:** Consistent features for LLM reasoning
- **Objective 5:** Fair comparison across institutions

#### 📖 What to Read

**Essential Sections:**
1. **Abstract** - Problem and solution overview
2. **Section 2.2: ComBat harmonization** - The algorithm
3. **Figure 1** - Before/after harmonization visualization
4. **Section 3: Results** - Effectiveness demonstration

**Reading Time:** 2-3 hours

#### 💡 Key Takeaways & Implementation

**The Scanner Harmonization Problem:**

```python
# Problem: Same patient scanned on different scanners → different values!

# Patient scanned on Siemens 3T:
tumor_intensity_siemens = 850

# Same patient, same tumor, on GE 3T:
tumor_intensity_ge = 650

# Difference is SCANNER, not biology!
# This confuses AI models
```

**ComBat Solution:**

```python
# ComBat removes scanner-specific "batch effects"

from neuroCombat import neuroCombat
import pandas as pd
import numpy as np

def harmonize_yale_dataset(features, metadata):
    """
    Apply ComBat harmonization to Yale multi-scanner data
    
    Args:
        features: [n_samples, n_features] array of imaging features
        metadata: DataFrame with columns ['scanner', 'site', 'age', 'sex']
    
    Returns:
        Harmonized features
    """
    
    # Convert features to neuroCombat format
    # Rows = features, Columns = samples (neuroCombat convention)
    data = features.T
    
    # Prepare covariates dataframe
    covars = pd.DataFrame({
        'batch': metadata['scanner'],  # Scanner type (Siemens, GE, etc.)
        'age': metadata['age'],
        'sex': metadata['sex']
    })
    
    # Apply ComBat
    # - batch_col: Column identifying scanner/batch
    # - categorical_cols: Categorical variables to preserve
    # - continuous_cols: Continuous variables to preserve
    
    harmonized_data = neuroCombat(
        dat=data,
        covars=covars,
        batch_col='batch',
        categorical_cols=['sex'],
        continuous_cols=['age']
    )
    
    # Convert back to [n_samples, n_features]
    harmonized_features = harmonized_data['data'].T
    
    return harmonized_features


# Example usage with Yale dataset:
def extract_and_harmonize_yale_features():
    """
    Complete pipeline: extract features → harmonize
    """
    all_features = []
    all_metadata = []
    
    for patient in yale_dataset:
        for timepoint in patient.timepoints:
            # Extract radiomics features
            features = extract_radiomics(timepoint.scan)
            
            # Get scanner info from DICOM metadata
            scanner = get_scanner_info(timepoint)
            
            all_features.append(features)
            all_metadata.append({
                'scanner': scanner,
                'patient_id': patient.id,
                'timepoint': timepoint.date
            })
    
    # Convert to arrays
    features_array = np.array(all_features)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Harmonize
    harmonized = harmonize_yale_dataset(features_array, metadata_df)
    
    return harmonized
```

#### 🔧 What to Apply to Your Project

**ComBat for Yale Dataset:**

```python
# Yale data spans 2000-2019 → multiple scanner upgrades!

# Identify scanner variability in your dataset:
def analyze_scanner_variability(yale_dataset):
    """
    Check how many scanners/protocols in Yale data
    """
    scanners = defaultdict(int)
    field_strengths = defaultdict(int)
    
    for patient in yale_dataset:
        for timepoint in patient.timepoints:
            metadata = read_dicom_metadata(timepoint)
            scanners[metadata['Manufacturer']] += 1
            field_strengths[metadata['MagneticFieldStrength']] += 1
    
    print("Scanners in dataset:")
    for scanner, count in scanners.items():
        print(f"  {scanner}: {count} scans")
    
    print("\nField strengths:")
    for field, count in field_strengths.items():
        print(f"  {field}T: {count} scans")
    
    # If multiple scanners → harmonization needed!
    return len(scanners) > 1

# When to harmonize:
harmonization_decision = """
Harmonize if:
✓ Multiple scanner manufacturers (Siemens + GE + Philips)
✓ Multiple field strengths (1.5T + 3T)
✓ Multiple sites (even same manufacturer)
✓ Scanner upgrades over time (2000→2019 definitely has this!)

Don't harmonize if:
✗ Single scanner, single protocol
✗ Already very homogeneous data
"""
```

#### 🔗 Resources

- **Original Paper:** https://www.sciencedirect.com/science/article/pii/S1053811917306948
- **R Package:** https://github.com/Jfortin1/neuroCombat
- **Python Package:** `pip install neuroCombat`
- **Tutorial:** https://github.com/Jfortin1/neuroCombat_Tutorials

---

### Paper 8: LongComBat - Longitudinal Harmonization

#### 📄 Paper Details
- **Title:** "Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data"
- **Year:** 2022
- **Authors:** Beer et al.
- **Journal:** NeuroImage
- **Link:** https://www.sciencedirect.com/science/article/pii/S1053811922001136
- **Code:** ✅ Available in R `neuroCombat` package
- **Status:** ✅ Open Access

#### 🎯 What This Paper Does for You

**For Objective 1 (CRITICAL for Longitudinal Data):**
- Harmonizes **multi-scanner data** while **preserving temporal trajectories**
- Solves: "How to remove scanner effects WITHOUT hiding real tumor changes?"
- Enables **Yale dataset** multi-scanner harmonization correctly

**This is THE harmonization method for longitudinal imaging!**

**Alignment with Other Objectives:**
- **Objective 4:** Preserving real temporal changes essential for video generation

#### 📖 What to Read

**MUST READ:**
1. **Abstract** - Why standard ComBat fails for longitudinal data
2. **Figure 1** - Problem visualization
3. **Section 2.2: Longitudinal ComBat** - The solution
4. **Section 3: Simulations** - Validation

**Reading Time:** 2-3 hours

#### 💡 Key Takeaways

**Why Standard ComBat FAILS for Longitudinal Data:**

```python
# Problem: Standard ComBat removes BOTH scanner effects AND real changes!

# Patient longitudinal data:
T0_scan = {'scanner': 'Siemens_old', 'tumor_volume': 20}
T1_scan = {'scanner': 'Siemens_new', 'tumor_volume': 10}  # Tumor shrank!

# Standard ComBat:
# Sees scanner difference (old → new) and "corrects" it
# But also sees volume difference and treats it as scanner artifact!
# Result: Harmonized T0 = 20, Harmonized T1 = 20 (WRONG!)

# LongComBat:
# Knows patient is SAME across timepoints
# Removes scanner effect while preserving real 20→10 change
# Result: Harmonized T0 = 20, Harmonized T1 = 10 (CORRECT!)
```

**LongComBat Implementation:**

```python
# LongComBat requires patient ID to link timepoints

def longcombat_harmonize(features, metadata):
    """
    Harmonize longitudinal data while preserving temporal changes
    """
    from neuroCombat import neuroCombat
    
    # CRITICAL: Include patient_id in metadata
    covars = pd.DataFrame({
        'batch': metadata['scanner'],
        'patient_id': metadata['patient_id'],  # Links timepoints!
        'timepoint': metadata['days_from_baseline'],
        'age': metadata['age']
    })
    
    # Enable longitudinal mode
    harmonized = neuroCombat(
        dat=features.T,
        covars=covars,
        batch_col='batch',
        categorical_cols=[],
        continuous_cols=['age'],
        # IMPORTANT: Longitudinal parameters
        longitudinal=True,
        patient_id_col='patient_id'
    )
    
    return harmonized['data'].T
```

#### 🔧 Your Implementation (Week 4)

```python
# Apply to Yale dataset

def harmonize_yale_longitudinal():
    """
    Full pipeline: Extract features → LongComBat → Save
    """
    all_data = []
    
    # 1. Extract features from all scans
    for patient in yale_dataset:
        patient_id = patient.id
        
        for idx, timepoint in enumerate(patient.timepoints):
            # Extract radiomics features
            features = extract_radiomics(timepoint.scan)
            
            # Get metadata
            scanner = get_scanner_from_dicom(timepoint.scan)
            days = timepoint.days_from_baseline
            
            all_data.append({
                'features': features,
                'patient_id': patient_id,
                'timepoint_idx': idx,
                'days_from_baseline': days,
                'scanner': scanner,
                'age': patient.age
            })
    
    # 2. Organize into arrays
    features_matrix = np.array([d['features'] for d in all_data])
    metadata_df = pd.DataFrame([
        {k: v for k, v in d.items() if k != 'features'}
        for d in all_data
    ])
    
    # 3. Apply LongComBat
    harmonized_features = longcombat_harmonize(features_matrix, metadata_df)
    
    # 4. Validate: Check that temporal trajectories preserved
    validate_temporal_preservation(
        original=features_matrix,
        harmonized=harmonized_features,
        metadata=metadata_df
    )
    
    return harmonized_features


def validate_temporal_preservation(original, harmonized, metadata):
    """
    Verify that patient-specific temporal trends preserved
    """
    for patient_id in metadata['patient_id'].unique():
        # Get patient's timepoints
        patient_mask = metadata['patient_id'] == patient_id
        patient_timepoints = metadata[patient_mask]['days_from_baseline']
        
        # Get features at each timepoint
        original_trajectory = original[patient_mask, :]
        harmonized_trajectory = harmonized[patient_mask, :]
        
        # Compute temporal correlation
        # (Should be preserved after harmonization)
        for feature_idx in range(original.shape[1]):
            original_trend = np.corrcoef(
                patient_timepoints,
                original_trajectory[:, feature_idx]
            )[0, 1]
            
            harmonized_trend = np.corrcoef(
                patient_timepoints,
                harmonized_trajectory[:, feature_idx]
            )[0, 1]
            
            # Trends should be similar
            assert abs(original_trend - harmonized_trend) < 0.1, \
                f"Temporal trend changed for patient {patient_id}!"
```

#### ✅ Success Criteria

Week 4: You should have:
- [ ] Identified scanner variability in Yale dataset
- [ ] Applied LongComBat harmonization
- [ ] Validated temporal trajectories preserved
- [ ] Documented harmonization parameters

#### 🔗 Resources

- **Paper:** https://www.sciencedirect.com/science/article/pii/S1053811922001136
- **R Implementation:** https://github.com/Jfortin1/neuroCombat
- **Tutorial:** https://rpubs.com/jfortin/ComBat

---

## 📚 PHASE 4: ADVANCED TOPICS (OPTIONAL)

### Paper 9: nnU-Net for Automatic Segmentation

If Yale dataset lacks tumor segmentations (which it does), you'll need to create them.

**Paper:** "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" (2020)
**Use:** Auto-segment tumors in Yale data for registration

**Quick Implementation:**
```bash
# Use pretrained nnU-Net model
pip install nnunet

# Download pretrained BraTS model
# Apply to Yale data to get tumor masks
```

---

### Paper 10: MONAI Framework Documentation

**Link:** https://docs.monai.io/
**Use:** Complete medical imaging preprocessing library

**Key MONAI Components:**

```python
from monai.transforms import (
    LoadImaged,           # Load NIfTI/DICOM
    EnsureChannelFirstd,  # Add channel dimension
    Spacingd,             # Resample to target spacing
    Orientationd,         # Standardize orientation
    ScaleIntensityRanged, # Normalize intensities
    CropForegroundd,      # Remove background
    RandAffined,          # Data augmentation (for training)
    Compose               # Chain transforms
)

from monai.data import (
    Dataset,              # Basic dataset
    DataLoader,           # Batch loading
    CacheDataset          # Cache preprocessed data
)
```

---

## 🔗 COMPLETE RESOURCE LINKS

### 📥 Datasets

| Dataset | Link | Size | Priority |
|---------|------|------|----------|
| **BraTS 2023** | http://braintumorsegmentation.org/ | ~100 GB | ✅ High |
| **Yale-Brain-Mets** | https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/ | ~200 GB | ✅ Critical |
| **UCSF Post-Glioma** | https://www.cancerimagingarchive.net/collection/ucsf-post-treatment-glioma/ | ~50 GB | ⚙️ Medium |
| **LUMIERE** | https://www.cancerimagingarchive.net/collection/lumiere/ | ~20 GB | ⚙️ Optional |

### 💻 Code Repositories

| Tool | GitHub | Purpose |
|------|--------|---------|
| **BraTS Toolkit** | https://github.com/neuronflow/BraTS-Toolkit | Reference preprocessing |
| **nnU-Net** | https://github.com/MIC-DKFZ/nnUNet | Auto preprocessing + segmentation |
| **Treatment-Aware Registration** | https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration | Tumor-preserving registration |
| **FLIRE** | https://github.com/michelle-tong18/FLIRE-MRI-registration | Fast longitudinal registration |
| **neuroCombat** | https://github.com/Jfortin1/neuroCombat | Harmonization |

### 📚 Python Libraries

```bash
# Install all required libraries:

pip install monai
pip install SimpleITK
pip install nibabel
pip install pydicom
pip install neuroCombat
pip install nnunet
pip install pandas numpy scipy scikit-image
```

### 📖 Tutorials & Documentation

| Resource | Link | Topic |
|----------|------|-------|
| **MONAI Tutorials** | https://github.com/Project-MONAI/tutorials | Medical imaging in PyTorch |
| **SimpleITK Notebooks** | https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks | Registration, segmentation |
| **TCIA User Guide** | https://wiki.cancerimagingarchive.net/ | Dataset download and usage |
| **Medical Image Analysis Course** | https://www.coursera.org/learn/medical-image-analysis | Fundamentals |

---

## ✅ WEEKLY IMPLEMENTATION CHECKLIST

### Week 1: Foundation & BraTS

**Monday-Tuesday: Setup**
- [ ] Install all Python libraries (MONAI, SimpleITK, neuroCombat)
- [ ] Download BraTS 2023 dataset (~100 GB)
- [ ] Download BraTS Toolkit: `git clone https://github.com/neuronflow/BraTS-Toolkit`
- [ ] Read BraTS Toolkit paper (Section 2.2, 2.4)
- [ ] Read nnU-Net paper (preprocessing sections)

**Wednesday-Thursday: Implementation**
- [ ] Implement basic preprocessing pipeline (6 steps from BraTS Toolkit)
- [ ] Test on 5 BraTS cases
- [ ] Compare your output to official BraTS preprocessing
- [ ] Validate: Dice score > 0.99 with official data

**Friday: BraTS Processing**
- [ ] Process all BraTS cases (2,040 scans)
- [ ] Implement automated QC checks
- [ ] Document any issues/failures
- [ ] **Deliverable:** Preprocessed BraTS dataset

**Weekend: Yale Download**
- [ ] Download Yale-Brain-Mets-Longitudinal (~200 GB, can start with 100-patient subset)
- [ ] Explore file structure
- [ ] Read Longitudinal Brain Metastases dataset paper

---

### Week 2: Longitudinal Registration

**Monday: Yale Data Exploration**
- [ ] Implement Yale data loader
- [ ] Identify patients with ≥3 timepoints
- [ ] Visualize example temporal sequences
- [ ] Extract clinical metadata (if available)

**Tuesday-Wednesday: Registration Theory**
- [ ] Read Treatment-Aware Registration paper (CRITICAL!)
- [ ] Read FLIRE paper
- [ ] Clone GitHub repos:
  - `git clone https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration`
  - `git clone https://github.com/michelle-tong18/FLIRE-MRI-registration`

**Thursday-Friday: Registration Implementation**
- [ ] Implement tumor-preserving registration (SimpleITK)
- [ ] Test on 5 Yale patients with known progression
- [ ] Validate: Tumor volume preserved within 5% error
- [ ] Implement registration quality checks

**Weekend: Scale Up**
- [ ] Register 50 Yale patients (pilot)
- [ ] Analyze registration quality metrics
- [ ] Identify cases needing manual review

---

### Week 3: Harmonization & Finalization

**Monday-Tuesday: Harmonization Theory**
- [ ] Read ComBat paper
- [ ] Read LongComBat paper
- [ ] Install neuroCombat: `pip install neuroCombat`
- [ ] Understand when to apply harmonization

**Wednesday: Scanner Analysis**
- [ ] Analyze Yale dataset for scanner variability
- [ ] Extract scanner metadata from DICOM headers
- [ ] Decide: Does Yale need harmonization? (Answer: Probably yes!)

**Thursday-Friday: Harmonization Implementation**
- [ ] Extract radiomics features from Yale data
- [ ] Apply LongComBat harmonization
- [ ] Validate temporal trajectories preserved
- [ ] Document harmonization parameters

**Weekend: Full Yale Processing**
- [ ] Process all Yale patients
- [ ] Complete registration + harmonization
- [ ] Run comprehensive QC

---

### Week 4: Quality Control & Organization

**Monday-Tuesday: Quality Control**
- [ ] Implement automated QC pipeline (from BraTS standards)
- [ ] Run QC on all processed data
- [ ] Flag outliers for manual review
- [ ] Fix any issues

**Wednesday: Data Organization for Objectives 2-5**
- [ ] Structure data for ViT training (Objective 2)
- [ ] Create JSON metadata for LLM (Objective 3)
- [ ] Organize temporal sequences for video (Objective 4)
- [ ] Split train/val/test sets (Objective 5)

**Thursday: Documentation**
- [ ] Document preprocessing pipeline (code + parameters)
- [ ] Create preprocessing report (data quality, statistics)
- [ ] Document limitations and challenges
- [ ] Create README for dataset

**Friday: Deliverables**
- [ ] **Deliverable 1:** Preprocessed BraTS + Yale datasets
- [ ] **Deliverable 2:** Data loader for Objective 2
- [ ] **Deliverable 3:** Metadata JSON for Objective 3
- [ ] **Deliverable 4:** Temporal sequences for Objective 4
- [ ] **Deliverable 5:** Technical report

**Weekend: Prepare for Objective 2**
- [ ] Review ViT papers (for next phase)
- [ ] Install PyTorch/TensorFlow
- [ ] Familiarize with vision transformer architectures

---

## 🎯 ALIGNMENT WITH OTHER OBJECTIVES

### How Objective 1 Enables Objective 2 (ViT Training)

```
Objective 1 Output:          Enables Objective 2:
├─ Standardized images   →   ViT input requirements met
├─ Isotropic resolution  →   Consistent patch extraction
├─ Normalized intensity  →   Stable training convergence
├─ Registered sequences  →   Temporal pattern learning
└─ Harmonized features   →   Model learns biology, not scanners
```

### How Objective 1 Enables Objective 3 (LLM Integration)

```
Objective 1 Output:          Enables Objective 3:
├─ Clinical metadata JSON →  LLM context for narratives
├─ Temporal timestamps   →   Timeline construction
├─ Volume measurements   →   Quantitative facts for LLM
└─ Treatment alignment   →   Response explanation
```

### How Objective 1 Enables Objective 4 (Video Generation)

```
Objective 1 Output:          Enables Objective 4:
├─ Registered sequences  →   Spatially consistent frames
├─ Temporal ordering     →   Correct progression direction
├─ Harmonized scans      →   Temporally consistent appearance
└─ Quality-controlled data → Artifact-free training
```

### How Objective 1 Enables Objective 5 (Evaluation)

```
Objective 1 Output:          Enables Objective 5:
├─ Validated preprocessing → Fair baseline comparison
├─ Volume measurements   →   Ground truth for predictions
├─ Multi-dataset QC      →   Cross-validation possible
└─ Documentation         →   Reproducible evaluation
```

---

## 🎓 FINAL TIPS

### Reading Strategy
1. **Don't read papers linearly** - start with abstract, figures, methods relevant to you
2. **Focus on implementation** - code is often clearer than text
3. **Take notes** - document what you'll implement from each paper
4. **Keep a questions list** - revisit papers when implementing

### Implementation Strategy
1. **Start simple** - basic pipeline first, optimizations later
2. **Test on small data** - don't process 1,430 patients until pilot works
3. **Validate at each step** - catch errors early
4. **Document everything** - future you will thank present you

### Time Management
- **Don't get stuck on one paper** - set time limits (2-3 hours max per paper)
- **Prioritize implementation** - reading without coding wastes time
- **Iterate** - you'll re-read papers when implementing anyway

### Success Metrics
By end of Objective 1:
- ✅ Reproducible preprocessing pipeline
- ✅ Validated on BraTS + Yale data
- ✅ Quality metrics documented
- ✅ Data ready for Objectives 2-5
- ✅ Technical report completed

---

**You've got this! This roadmap gives you everything needed to succeed in Objective 1 while setting yourself up perfectly for Objectives 2-5. Start with BraTS (Week 1), add Yale longitudinal processing (Weeks 2-3), finalize with harmonization and QC (Week 4), and you'll have a publication-quality preprocessing pipeline!** 🚀
