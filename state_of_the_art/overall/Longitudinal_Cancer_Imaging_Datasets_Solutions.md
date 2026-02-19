# Longitudinal Cancer Imaging Datasets: Solutions for Your Research Project

## 🚨 CRITICAL ISSUE: BraTS is NOT Longitudinal

### The Problem
**BraTS (Brain Tumor Segmentation Challenge) contains ONLY pre-operative, single-timepoint MRI scans.** This is a fundamental incompatibility with your research objectives which require:
- Temporal tumor evolution modeling
- Treatment response tracking
- Video generation of progression
- Counterfactual scenario simulation

### Why BraTS Was Designed This Way
- BraTS focuses on **segmentation quality** at diagnosis
- Pre-operative scans avoid complications from surgery/treatment artifacts
- Standardized across multiple institutions (2012-2025)
- 2,040+ cases (BraTS 2021), but all **single snapshots**

---

## 💡 SOLUTION 1: Use BraTS as Baseline + Add True Longitudinal Datasets

### Strategy: Hybrid Approach

```
┌─────────────────────────────────────────────────┐
│ Phase 1: Objective 1 (Preprocessing Pipeline)  │
│ • Use BraTS for pipeline development           │
│ • Learn preprocessing on large, clean dataset  │
│ • Validate registration/normalization methods  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Phase 2: Objectives 2-3 (ViT + LLM)            │
│ • Train ViT on BraTS single-timepoint features │
│ • Transfer learning to longitudinal data       │
│ • Use BraTS for augmentation/pre-training      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Phase 3: Objectives 4-5 (Video + Evaluation)   │
│ • Switch to TRUE longitudinal datasets         │
│ • Generate progression videos                  │
│ • Validate on real temporal sequences          │
└─────────────────────────────────────────────────┘
```

**Justification:**
- BraTS provides clean, standardized data for pipeline development
- Large sample size enables robust ViT training
- Real longitudinal data reserved for temporal modeling where it's essential

---

## 💡 SOLUTION 2: Simulate Longitudinal Sequences from BraTS

### Synthetic Temporal Data Generation

#### Method A: Cross-Patient Pseudo-Progression
```python
# Create synthetic temporal sequences by ordering different patients
# Example: Simulate progression from Grade II → Grade III → Grade IV

def create_synthetic_progression(brats_dataset):
    """
    Order patients by tumor characteristics to simulate progression
    """
    sequences = []
    
    # Group patients by tumor grade
    grade_II = filter_by_grade(brats_dataset, "LGG")
    grade_III = filter_by_grade(brats_dataset, "GBM_small")  # Early GBM
    grade_IV = filter_by_grade(brats_dataset, "GBM_large")   # Advanced GBM
    
    # Create pseudo-longitudinal sequences
    for idx in range(min(len(grade_II), len(grade_III), len(grade_IV))):
        synthetic_sequence = {
            'T0_baseline': grade_II[idx],      # Small, low-grade tumor
            'T1_progression': grade_III[idx],   # Growing, mid-grade
            'T2_advanced': grade_IV[idx]        # Large, high-grade
        }
        sequences.append(synthetic_sequence)
    
    return sequences
```

#### Method B: GAN-Based Temporal Interpolation
```python
# Train a GAN to interpolate between two BraTS cases
# Simulate intermediate timepoints

def train_temporal_gan(brats_small_tumors, brats_large_tumors):
    """
    Learn morphological transitions
    Generate intermediate tumor states
    """
    # Implementation using StyleGAN or similar
    pass
```

**Limitations:**
- ❌ Not clinically realistic progression patterns
- ❌ Lacks patient-specific evolution
- ❌ Cannot capture treatment effects
- ⚠️ **USE ONLY FOR OBJECTIVE 2 (ViT training), NOT Objectives 4-5**

---

## ✅ SOLUTION 3: Public Longitudinal Cancer Imaging Datasets (RECOMMENDED)

### 🏆 TOP TIER: True Longitudinal Brain Cancer Datasets

| Dataset Name | Patients | Scans | Timepoints | Cancer Type | Access | GitHub/Code |
|-------------|----------|-------|------------|-------------|--------|-------------|
| **Yale-Brain-Mets-Longitudinal** | 1,430 | 11,892 | Multi (pre/post-treatment) | Brain metastases | TCIA | ❌ |
| **UCSF Longitudinal Post-Treatment Glioma** | 298 | 596 | 2 consecutive follow-ups | Diffuse gliomas | TCIA | ❌ |
| **LUMIERE (Glioblastoma)** | 25 | 375 | Up to 15 per patient | Glioblastoma | TCIA | ❌ |
| **Burdenko-GBM-Progression** | ~100 | Multi | Longitudinal progression | Glioblastoma | TCIA (Restricted) | ✅ [GitHub](https://github.com/kurmukovai/burdenko_glioma_progression) |
| **MU-Glioma-Post** | 203 | 617 | Multiple post-treatment | Post-op gliomas | TCIA | ❌ |

### 🥇 **RECOMMENDED PRIMARY DATASET: Yale-Brain-Mets-Longitudinal**

**Why This is Your Best Option:**

✅ **Massive Scale:**
- 11,892 MRI studies (largest public longitudinal brain cancer dataset)
- 1,430 patients with brain metastases
- Spans nearly 20 years (2000-2019)

✅ **True Longitudinal Design:**
- Pre-treatment scans (within 30 days before treatment)
- All follow-up imaging after treatment
- Enables treatment response modeling (critical for Objective 4)

✅ **Multi-Modal MRI:**
- T1-weighted pre-contrast
- T1-weighted post-contrast (T1CE)
- T2-weighted
- FLAIR
- **Exactly matches BraTS modalities!**

✅ **Clinical Metadata:**
- Treatment types (stereotactic radiosurgery, whole-brain radiotherapy, surgical resection)
- Patient demographics
- Scanner details (Siemens/GE, 1.5T/3T)
- Perfect for LLM integration (Objective 3)

✅ **Pre-processed Format:**
- NIfTI format (not DICOM)
- Standardized filenames: `caseID_date-time_sequence.nii.gz`
- Already partially preprocessed

✅ **Open Access:**
- Available on TCIA immediately
- No restricted access requirements
- CC BY license

**Download:**
```bash
# Access via TCIA
https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/

# Direct download using NBIA Data Retriever
# Dataset size: ~100-500 GB (estimate)
```

---

### 🥈 **SECONDARY DATASET: UCSF Longitudinal Post-Treatment Glioma**

**Use Case:** More detailed annotations, smaller but higher quality

✅ **Expert Annotations:**
- Voxelwise segmentation by board-certified neuroradiologists
- Tumor subregions: Enhancing tumor (ET), non-enhancing FLAIR hyperintensity (SNFH), non-enhancing tumor core (NETC), resection cavity (RC)
- **Longitudinal change annotations** (tracks what changed between timepoints!)

✅ **True Temporal Design:**
- 298 patients × 2 consecutive follow-ups = 596 scans
- Post-treatment monitoring (clinically realistic)
- Treatment response patterns

✅ **Radiology: Artificial Intelligence Publication:**
- High-quality dataset with validation
- Published 2023 (recent, well-curated)

❌ **Limitations:**
- Only 2 timepoints per patient (vs. Yale's multiple)
- Smaller sample size (298 vs 1,430 patients)
- Post-treatment only (no pre-treatment baseline)

**Download:**
```bash
https://www.cancerimagingarchive.net/collection/ucsf-post-treatment-glioma/
```

---

### 🥉 **THIRD OPTION: LUMIERE Dataset**

**Use Case:** Dense longitudinal sampling (many timepoints per patient)

✅ **Dense Temporal Sampling:**
- 25 patients with glioblastoma
- Up to 15 MRI scans per patient over treatment course
- Total: 375 scans
- Ideal for learning temporal dynamics

✅ **Expert RANO Evaluation:**
- Response Assessment in Neuro-Oncology criteria
- Clinical assessment labels (progression, stable, response)
- Excellent for LLM reasoning (Objective 3)

✅ **Published in Scientific Data (2022):**
- Well-documented, high-quality annotations
- Validation included

❌ **Limitations:**
- Very small patient cohort (25 patients)
- Limited diversity
- May not generalize well

**Download:**
```bash
https://www.cancerimagingarchive.net/collection/lumiere/
```

---

## 🩺 BREAST CANCER: Alternative Longitudinal Options

### If You Want to Expand Beyond Brain Cancer

| Dataset | Patients | Description | Longitudinal? | Access |
|---------|----------|-------------|---------------|--------|
| **ISPY1 (I-SPY TRIAL)** | 222 | Neoadjuvant breast cancer MRI | ✅ Pre/post treatment | TCIA |
| **RIDER Breast MRI** | 19 | Test-retest reliability | ✅ 1-2 days apart | TCIA |
| **QIN-Breast** | Multi | Treatment response PET/CT + MRI | ✅ Longitudinal | TCIA |
| **QIN-Breast-02** | Multi | Multi-parametric MRI | ✅ Multi-timepoint | TCIA (Limited) |

**ISPY1 Recommended for Breast Cancer:**
- 222 patients with invasive breast cancer
- Pre-treatment + during neoadjuvant therapy + post-treatment
- Clinical outcomes data (pathological complete response)
- Multi-institutional (ACRIN 6657/CALGB 150007 trial)

---

## 🫁 LUNG CANCER: LIDC-IDRI Situation

### ⚠️ LIDC-IDRI is ALSO NOT Truly Longitudinal

**Common Misconception:**
- LIDC-IDRI = 1,018 patients with thoracic CT scans
- Contains lung nodule annotations by 4 radiologists
- **BUT: Designed for nodule detection, not progression tracking**

**Reality:**
- Most patients have single screening timepoint
- Some incidental longitudinal data exists (screening follow-ups)
- Not standardized temporal intervals
- Annotations don't track same nodules over time

### **Alternative: NLST-New-Lesion-LongCT**

✅ **True Longitudinal Lung Cancer Dataset:**
- 152 lung lesions in 122 participants
- Derived from National Lung Screening Trial (NLST)
- **Baseline CT + follow-up CT with new lesions**
- Point annotations for lesion detection
- Registered baseline images for comparison

**Download:**
```bash
https://www.cancerimagingarchive.net/analysis-result/nlst-new-lesion-longct/
```

---

## 📊 TCGA: Genomic Data, Limited Imaging Longitudinal

### TCGA Imaging Collections

**Available:**
- TCGA-GBM: 135 pre-operative glioblastoma scans
- TCGA-LGG: 108 pre-operative low-grade glioma scans
- Multiple other cancer types (lung, breast, kidney, etc.)

**Longitudinal Status:**
- ❌ **Radiology images are predominantly single-timepoint**
- ✅ **Clinical follow-up data IS longitudinal** (survival, treatment, outcomes)
- ✅ **Genomic/proteomic data available**

**Use Case for Your Project:**
- Use TCGA for multi-modal integration (imaging + genomics + clinical)
- NOT for temporal imaging modeling
- Good for Objective 3 (LLM integration of multi-modal data)

---

## 🎯 RECOMMENDED DATASET STRATEGY FOR YOUR PROJECT

### **Optimal Configuration: Multi-Dataset Approach**

```
┌──────────────────────────────────────────────────────────┐
│ OBJECTIVE 1: Preprocessing Pipeline Development         │
├──────────────────────────────────────────────────────────┤
│ Primary: BraTS 2021/2023 (2,040 cases)                  │
│   → Large, clean, standardized                           │
│   → Develop/validate preprocessing methods               │
│   → Quality control benchmarks                           │
│                                                          │
│ Secondary: Yale-Brain-Mets (sample subset)               │
│   → Test longitudinal registration                       │
│   → Validate temporal consistency checks                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ OBJECTIVE 2: ViT Representation Learning                │
├──────────────────────────────────────────────────────────┤
│ Pre-training: BraTS (all 2,040 cases)                    │
│   → Learn tumor morphology features                      │
│   → Transfer learning baseline                           │
│                                                          │
│ Fine-tuning: Yale-Brain-Mets (1,430 patients)            │
│   → Temporal embedding extraction                        │
│   → Patient-specific progression patterns                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ OBJECTIVE 3: LLM Clinical Reasoning                     │
├──────────────────────────────────────────────────────────┤
│ Primary: Yale-Brain-Mets                                 │
│   → Rich clinical metadata (treatment types, dates)      │
│   → Multi-timepoint narratives                           │
│                                                          │
│ Augmentation: LUMIERE                                    │
│   → RANO criteria labels                                 │
│   → Clinical assessment language                         │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ OBJECTIVE 4: Video Generation of Progression            │
├──────────────────────────────────────────────────────────┤
│ PRIMARY: Yale-Brain-Mets (ESSENTIAL)                     │
│   → 11,892 scans = sufficient for video model training   │
│   → Real temporal sequences                              │
│   → Treatment vs. no-treatment comparisons               │
│                                                          │
│ Validation: UCSF Post-Treatment Glioma                   │
│   → Test generalization                                  │
│   → Expert annotation comparison                         │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ OBJECTIVE 5: Evaluation & Validation                    │
├──────────────────────────────────────────────────────────┤
│ Quantitative:                                            │
│   → Yale-Brain-Mets (hold-out test set)                  │
│   → UCSF (external validation)                           │
│   → LUMIERE (clinical plausibility - RANO alignment)     │
│                                                          │
│ Qualitative:                                             │
│   → All three longitudinal datasets                      │
│   → Compare generated vs. real progression               │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ PRACTICAL IMPLEMENTATION PLAN

### Week 1-2: Dataset Acquisition & Exploration

#### Step 1: Download Datasets
```bash
# Install NBIA Data Retriever (TCIA download tool)
# Download from: https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever

# Priority 1: Yale-Brain-Mets-Longitudinal
# - Access: https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/
# - Size: ~150-300 GB (estimated)
# - Format: NIfTI (.nii.gz)

# Priority 2: BraTS 2023 (for pipeline development)
# - Access: http://braintumorsegmentation.org/
# - Size: ~100 GB
# - Format: NIfTI

# Priority 3: UCSF Longitudinal Glioma (validation)
# - Access: https://www.cancerimagingarchive.net/collection/ucsf-post-treatment-glioma/
# - Size: ~50 GB
```

#### Step 2: Initial Exploration
```python
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# Yale-Brain-Mets structure
dataset_root = "/data/Yale-Brain-Mets"

# Files are named: caseID_date-time_sequence.nii.gz
# Example: patient001_2015-03-15-14-30_T1CE.nii.gz

def explore_longitudinal_structure(dataset_path):
    """
    Analyze temporal structure of Yale dataset
    """
    import os
    from collections import defaultdict
    
    patient_scans = defaultdict(list)
    
    for filename in os.listdir(dataset_path):
        if filename.endswith('.nii.gz'):
            parts = filename.split('_')
            patient_id = parts[0]
            date_time = parts[1]
            sequence = parts[2].replace('.nii.gz', '')
            
            patient_scans[patient_id].append({
                'date': date_time,
                'sequence': sequence,
                'filename': filename
            })
    
    # Analyze temporal spacing
    for patient_id, scans in patient_scans.items():
        scans_sorted = sorted(scans, key=lambda x: x['date'])
        num_timepoints = len(set([s['date'] for s in scans_sorted]))
        print(f"Patient {patient_id}: {num_timepoints} timepoints")
    
    return patient_scans

# Identify patients with ≥3 timepoints (ideal for progression modeling)
longitudinal_patients = {
    pid: scans for pid, scans in patient_scans.items()
    if len(set([s['date'] for s in scans])) >= 3
}

print(f"Patients with ≥3 timepoints: {len(longitudinal_patients)}")
```

---

### Week 2-3: Preprocessing Pipeline (Modified Objective 1)

#### Adapt BraTS Toolkit for Yale-Brain-Mets

```python
# Yale data is already in NIfTI format - no DICOM conversion needed!

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd
)
from monai.data import Dataset, DataLoader
import SimpleITK as sitk

# Step 1: Load Yale-Brain-Mets data
def load_yale_patient_sequences(patient_id, dataset_path):
    """
    Load all timepoints for a single patient
    Returns ordered temporal sequence
    """
    patient_files = glob.glob(f"{dataset_path}/{patient_id}_*_*.nii.gz")
    
    # Group by date
    timepoints = defaultdict(dict)
    for filepath in patient_files:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        date = parts[1]
        sequence = parts[2].replace('.nii.gz', '')
        
        timepoints[date][sequence] = filepath
    
    # Sort by date
    sorted_timepoints = sorted(timepoints.items(), key=lambda x: x[0])
    
    return sorted_timepoints

# Step 2: Preprocessing transforms (same as BraTS)
preprocessing = Compose([
    LoadImaged(keys=["T1", "T1CE", "T2", "FLAIR"]),
    EnsureChannelFirstd(keys=["T1", "T1CE", "T2", "FLAIR"]),
    Orientationd(keys=["T1", "T1CE", "T2", "FLAIR"], axcodes="RAS"),
    Spacingd(
        keys=["T1", "T1CE", "T2", "FLAIR"],
        pixdim=(1.0, 1.0, 1.0),  # Same as BraTS
        mode=("bilinear", "bilinear", "bilinear", "bilinear")
    ),
    ScaleIntensityRanged(
        keys=["T1", "T1CE", "T2", "FLAIR"],
        a_min=0, a_max="percentile_99.5",
        b_min=0, b_max=1,
        clip=True
    ),
    CropForegroundd(keys=["T1", "T1CE", "T2", "FLAIR"])
])

# Step 3: Longitudinal registration (critical!)
def register_patient_timepoints(baseline_scan, followup_scans):
    """
    Register all follow-up scans to baseline
    Preserve tumor morphology changes
    """
    registered_sequence = [baseline_scan]
    
    for followup in followup_scans:
        # Use SimpleITK for registration
        registration = sitk.ImageRegistrationMethod()
        
        # Affine registration (preserves tumor volume changes)
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetOptimizerAsLBFGSB()
        
        initial_transform = sitk.CenteredTransformInitializer(
            baseline_scan, followup,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        registration.SetInitialTransform(initial_transform)
        final_transform = registration.Execute(baseline_scan, followup)
        
        registered = sitk.Resample(
            followup, baseline_scan, final_transform,
            sitk.sitkLinear, 0.0, followup.GetPixelID()
        )
        
        registered_sequence.append(registered)
    
    return registered_sequence
```

---

### Week 3-4: Data Organization for Objectives 2-5

#### Final Dataset Structure

```python
# Organize into standardized format for all objectives

final_dataset_structure = {
    'patient_001': {
        'demographics': {
            'age': 65,
            'sex': 'M',
            'primary_cancer': 'lung',  # Source of brain metastasis
        },
        'timepoints': [
            {
                'id': 'T0',
                'date': '2015-03-15',
                'days_from_baseline': 0,
                'treatment_status': 'pre-treatment',
                'scans': {
                    'T1': 'path/to/T1.nii.gz',
                    'T1CE': 'path/to/T1CE.nii.gz',
                    'T2': 'path/to/T2.nii.gz',
                    'FLAIR': 'path/to/FLAIR.nii.gz'
                },
                'clinical': {
                    'num_metastases': 3,
                    'largest_diameter_mm': 15.2,
                    'edema_present': True,
                    'karnofsky_score': 80
                }
            },
            {
                'id': 'T1',
                'date': '2015-04-26',  # 6 weeks post-treatment
                'days_from_baseline': 42,
                'treatment_status': 'post-SRS',  # Stereotactic radiosurgery
                'scans': {...},
                'clinical': {
                    'num_metastases': 3,
                    'largest_diameter_mm': 12.1,  # Shrinking
                    'treatment_response': 'partial_response',
                    'radiation_necrosis': False
                }
            },
            {
                'id': 'T2',
                'date': '2015-07-15',  # 4 months from baseline
                'days_from_baseline': 122,
                'treatment_status': 'follow-up',
                'scans': {...},
                'clinical': {
                    'num_metastases': 4,  # New lesion!
                    'largest_diameter_mm': 18.5,  # Growing
                    'treatment_response': 'progressive_disease',
                    'new_lesion_location': 'occipital_lobe'
                }
            }
        ],
        'outcomes': {
            'overall_survival_days': 365,
            'progression_free_survival_days': 122,
            'neurological_death': False
        }
    },
    # ... more patients
}

# Save as JSON for LLM integration (Objective 3)
import json
with open('yale_dataset_structured.json', 'w') as f:
    json.dump(final_dataset_structure, f, indent=2)
```

---

## 📋 CHECKLIST: Adapting Your Research Plan

### ✅ Updated Timeline for Longitudinal Data

| Week | Original Plan (BraTS) | **UPDATED Plan (Yale + BraTS)** |
|------|----------------------|-----------------------------------|
| 1-2 | Download BraTS only | Download Yale + BraTS + UCSF |
| 2-3 | BraTS preprocessing | Adapt BraTS pipeline for Yale NIfTI format |
| 3-4 | BraTS QC | **Add longitudinal registration** + Yale QC |
| 5-7 | CNN baseline | CNN baseline on BraTS |
| 8-11 | ViT on BraTS | **ViT pre-train (BraTS) → fine-tune (Yale)** |
| 12-14 | LLM integration | **LLM with Yale clinical timelines** |
| 15-18 | Video generation | **CRITICAL: Use Yale longitudinal sequences** |
| 19-20 | Evaluation | Validate on Yale + UCSF + LUMIERE |

### ✅ Required Modifications to Research Proposal

**Section 2.1 - Background:**
```
ADD:
"While the BraTS challenge has provided standardized pre-operative brain tumor 
segmentation benchmarks, it lacks the longitudinal follow-up data necessary for 
modeling disease progression. To address this, we will leverage the Yale Brain 
Metastases Longitudinal dataset (11,892 MRI studies from 1,430 patients) which 
captures pre-treatment and multiple post-treatment timepoints, enabling true 
temporal modeling of cancer evolution and treatment response."
```

**Section 2.2 - Objective 1:**
```
MODIFY:
"Acquire and curate publicly available oncology datasets (e.g., TCGA, BraTS, LIDC-IDRI)."

TO:
"Acquire and curate publicly available oncology datasets:
• BraTS 2023 for preprocessing pipeline development (2,040 pre-operative cases)
• Yale-Brain-Mets-Longitudinal for true longitudinal modeling (11,892 scans, 1,430 patients)
• UCSF Post-Treatment Glioma for external validation (596 scans, 298 patients)
• LUMIERE for dense temporal sampling and clinical assessment validation (375 scans, 25 patients)"
```

**Section 2.4 - Timeline:**
```
UPDATE Phase 1 (Weeks 1-4):
• Week 1: Download Yale-Brain-Mets-Longitudinal, BraTS 2023, UCSF datasets
• Week 2: Explore temporal structure, identify patients with ≥3 timepoints
• Week 3: Implement longitudinal registration pipeline
• Week 4: Apply multi-scanner harmonization (LongComBat) to Yale data
```

---

## 🎓 LEARNING RESOURCES FOR LONGITUDINAL DATA

### Papers to Read (in order):

1. **Yale Brain Mets paper** (2025) - Dataset description
   - Understand data collection, temporal intervals, treatment protocols

2. **Treatment-Aware Longitudinal Registration** (2024) - From earlier table
   - Learn tumor-preserving registration techniques
   - Critical for preserving biological changes

3. **LongComBat** (2022) - From earlier table
   - Multi-scanner harmonization that preserves temporal trajectories

4. **LUMIERE Dataset** paper (2022, Scientific Data)
   - RANO criteria for clinical assessment
   - Dense longitudinal sampling strategies

### Code Repositories to Study:

```bash
# 1. Burdenko GBM Progression preprocessing
git clone https://github.com/kurmukovai/burdenko_glioma_progression

# 2. FLIRE longitudinal breast MRI registration
git clone https://github.com/michelle-tong18/FLIRE-MRI-registration

# 3. Treatment-aware registration
git clone https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration

# 4. BraTS Toolkit (for baseline preprocessing)
git clone https://github.com/neuronflow/BraTS-Toolkit
```

---

## ⚠️ COMMON PITFALLS TO AVOID

### 1. **Don't Assume All Public Datasets Are Longitudinal**
- ❌ BraTS = single timepoint
- ❌ LIDC-IDRI = mostly single timepoint
- ❌ TCGA imaging = mostly single timepoint
- ✅ Always verify temporal structure before committing

### 2. **Registration Must Preserve Biological Changes**
- ❌ Don't use deformable registration (will make tumors "disappear")
- ✅ Use affine/rigid registration only
- ✅ Validate that tumor volumes remain accurate post-registration

### 3. **Temporal Intervals Matter**
- ❌ Don't mix irregular intervals without handling explicitly
- ✅ Document actual time gaps in your data structure
- ✅ Your video generation model should condition on time deltas

### 4. **Clinical Metadata Is Essential**
- ❌ Don't treat this as pure image-to-image problem
- ✅ Extract treatment dates, types, responses from TCIA metadata
- ✅ Align imaging timepoints with treatment interventions

### 5. **Dataset Size Requirements Differ by Objective**
```
Objective 1 (Preprocessing): 50-100 patients sufficient for validation
Objective 2 (ViT Training): 500-2,000 patients ideal (use BraTS + Yale)
Objective 3 (LLM Integration): 100-500 patients sufficient
Objective 4 (Video Generation): 1,000+ scans minimum (Yale provides this!)
Objective 5 (Evaluation): 100-300 held-out patients for validation
```

---

## 🏁 FINAL RECOMMENDATION

### **Primary Dataset: Yale-Brain-Mets-Longitudinal**

**Why it's perfect for your project:**

1. ✅ **Scale**: 11,892 scans = largest public longitudinal brain cancer dataset
2. ✅ **True temporal design**: Pre/post treatment with multiple follow-ups
3. ✅ **Multi-modal**: T1, T1CE, T2, FLAIR (matches BraTS)
4. ✅ **Clinical metadata**: Treatment types, dates (for LLM narratives)
5. ✅ **Open access**: No restrictions, immediate download
6. ✅ **Pre-processed**: Already in NIfTI format
7. ✅ **Recent**: Published 2025, high quality curation
8. ✅ **Treatment diversity**: SRS, WBRT, surgery (for counterfactual modeling)

**Use BraTS as:**
- Preprocessing pipeline development (Weeks 1-4)
- ViT pre-training (Objective 2)
- Augmentation data
- Baseline comparisons

**Use Yale as:**
- Primary longitudinal dataset (Objectives 3-5)
- Temporal modeling
- Video generation training
- Clinical reasoning with LLM

**Use UCSF + LUMIERE as:**
- External validation
- Generalization testing
- Clinical plausibility assessment

---

## 📞 NEXT STEPS

1. **Download Yale-Brain-Mets-Longitudinal** (this week)
2. **Explore temporal structure** (identify 3+ timepoint patients)
3. **Adapt preprocessing pipeline** (BraTS Toolkit → Yale NIfTI)
4. **Implement longitudinal registration** (read Treatment-Aware paper first)
5. **Extract clinical metadata** (build JSON structure for LLM)
6. **Update research proposal** (add Yale as primary dataset)

Your research project is **feasible and exciting** with these longitudinal datasets! The Yale dataset in particular is a game-changer for temporal cancer imaging AI.
