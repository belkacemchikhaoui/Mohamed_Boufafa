# Phase 1: Oncology Medical Imaging Preparation and Exploration
## Detailed Step-by-Step Implementation Guide

**Duration:** 4 weeks (Weeks 1–4)  
**Objective:** Develop a Robust Pipeline for Longitudinal Cancer Imaging Analysis

---

## Table of Contents
1. [Week-by-Week Overview](#week-by-week-overview)
2. [Activity 1: Dataset Selection and Acquisition](#activity-1-dataset-selection-and-acquisition)
3. [Activity 2: Preprocessing Pipeline](#activity-2-preprocessing-pipeline)
4. [Activity 3: Tumor Annotation Processing](#activity-3-tumor-annotation-processing)
5. [Activity 4: Exploratory Data Analysis](#activity-4-exploratory-data-analysis)
6. [Activity 5: Longitudinal Organization](#activity-5-longitudinal-organization)
7. [Deliverables Checklist](#deliverables-checklist)
8. [Code Templates](#code-templates)

---

## Week-by-Week Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WEEK 1: Dataset Acquisition & Environment Setup                        │
│  ├── Day 1-2: Environment setup, account registrations                  │
│  ├── Day 3-4: Download BraTS dataset                                    │
│  └── Day 5-7: Download TCGA/LIDC-IDRI, initial exploration              │
├─────────────────────────────────────────────────────────────────────────┤
│  WEEK 2: Preprocessing Pipeline Development                             │
│  ├── Day 1-2: Intensity normalization implementation                    │
│  ├── Day 3-4: Spatial resampling implementation                         │
│  └── Day 5-7: Registration/alignment implementation                     │
├─────────────────────────────────────────────────────────────────────────┤
│  WEEK 3: Annotation Processing & Initial EDA                            │
│  ├── Day 1-2: Segmentation mask processing                              │
│  ├── Day 3-4: Tumor statistics extraction                               │
│  └── Day 5-7: Begin EDA (distributions, variability)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  WEEK 4: EDA Completion & Longitudinal Organization                     │
│  ├── Day 1-2: Complete EDA analysis                                     │
│  ├── Day 3-4: Organize longitudinal sequences                           │
│  └── Day 5-7: Documentation, report writing, deliverables               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Activity 1: Dataset Selection and Acquisition

### 1.1 Environment Setup (Day 1-2)

#### Step 1: Create Project Structure
```bash
mkdir -p ~/mitacs_project/{data,code,notebooks,reports,models,configs}
mkdir -p ~/mitacs_project/data/{raw,processed,interim}
mkdir -p ~/mitacs_project/data/raw/{brats,tcga,lidc}
mkdir -p ~/mitacs_project/code/{preprocessing,eda,utils}
cd ~/mitacs_project
```

#### Step 2: Setup Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai[all]
pip install nibabel SimpleITK pydicom
pip install numpy pandas matplotlib seaborn
pip install scikit-image scikit-learn scipy
pip install tqdm jupyter jupyterlab
pip install tensorboard wandb
pip install albumentations

# Save requirements
pip freeze > requirements.txt
```

#### Step 3: Create Configuration File
```yaml
# configs/data_config.yaml
datasets:
  brats:
    path: "data/raw/brats"
    modalities: ["t1", "t1ce", "t2", "flair"]
    target_spacing: [1.0, 1.0, 1.0]
    
  tcga:
    path: "data/raw/tcga"
    cancer_types: ["GBM", "LGG"]  # Glioblastoma, Low-grade glioma
    
  lidc:
    path: "data/raw/lidc"
    min_nodule_size: 3  # mm

preprocessing:
  intensity_normalization: "zscore"  # Options: zscore, minmax, histogram
  target_spacing: [1.0, 1.0, 1.0]  # mm
  crop_to_brain: true
  
output:
  processed_path: "data/processed"
  format: "nifti"  # Options: nifti, numpy
```

### 1.2 Dataset Registration & Download (Day 3-7)

#### BraTS Dataset (Primary - Brain Tumors)
**Registration Steps:**
1. Go to: https://www.synapse.org/#!Synapse:syn51156910/wiki/622351 (BraTS 2023)
2. Create Synapse account
3. Accept data use agreement
4. Download using Synapse client

```bash
# Install Synapse client
pip install synapseclient

# Download BraTS (after authentication)
synapse get -r syn51156910 --downloadLocation data/raw/brats
```

**BraTS Data Structure:**
```
brats/
├── BraTS2023_00000/
│   ├── BraTS2023_00000-t1n.nii.gz      # Native T1
│   ├── BraTS2023_00000-t1c.nii.gz      # Contrast-enhanced T1
│   ├── BraTS2023_00000-t2w.nii.gz      # T2-weighted
│   ├── BraTS2023_00000-t2f.nii.gz      # T2-FLAIR
│   └── BraTS2023_00000-seg.nii.gz      # Segmentation mask
├── BraTS2023_00001/
...
```

#### TCGA Dataset (Secondary - Multi-cancer)
**Registration Steps:**
1. Go to: https://portal.gdc.cancer.gov/
2. Create account and login
3. Navigate to Repository → Cases → Primary Site → Brain
4. Download manifest file
5. Use GDC Data Transfer Tool

```bash
# Install GDC client
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip

# Download using manifest
./gdc-client download -m manifest.txt -d data/raw/tcga
```

#### LIDC-IDRI Dataset (Secondary - Lung)
**Registration Steps:**
1. Go to: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. Download using NBIA Data Retriever
3. Or use `pylidc` library

```bash
pip install pylidc

# Configure pylidc
python -c "
import pylidc as pl
# First run will prompt for data path configuration
# Set path to: data/raw/lidc
"
```

### 1.3 Initial Data Exploration

```python
# code/utils/data_exploration.py
import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd

def explore_brats_dataset(data_path):
    """Initial exploration of BraTS dataset."""
    data_path = Path(data_path)
    subjects = list(data_path.glob("BraTS*"))
    
    stats = []
    for subject in subjects[:10]:  # Sample first 10
        # Load T1 contrast-enhanced
        t1ce_path = list(subject.glob("*t1c*.nii.gz"))[0]
        img = nib.load(t1ce_path)
        data = img.get_fdata()
        
        # Load segmentation
        seg_path = list(subject.glob("*seg*.nii.gz"))[0]
        seg = nib.load(seg_path).get_fdata()
        
        stats.append({
            'subject': subject.name,
            'shape': data.shape,
            'spacing': img.header.get_zooms(),
            'intensity_min': data.min(),
            'intensity_max': data.max(),
            'intensity_mean': data.mean(),
            'tumor_voxels': (seg > 0).sum(),
            'enhancing_tumor': (seg == 4).sum(),
            'edema': (seg == 2).sum(),
            'necrotic_core': (seg == 1).sum(),
        })
    
    return pd.DataFrame(stats)

if __name__ == "__main__":
    df = explore_brats_dataset("data/raw/brats")
    print(df.describe())
    df.to_csv("reports/initial_exploration.csv", index=False)
```

---

## Activity 2: Preprocessing Pipeline

### 2.1 Intensity Normalization (Week 2, Day 1-2)

#### Understanding the Options

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Z-score** | `(x - μ) / σ` | General purpose, per-volume |
| **Min-Max** | `(x - min) / (max - min)` | Fixed range needed |
| **Percentile** | Clip to [1%, 99%], then normalize | Handle outliers |
| **Histogram Matching** | Match to reference | Cross-site standardization |

#### Implementation

```python
# code/preprocessing/intensity_normalization.py
import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib

class IntensityNormalizer:
    """Intensity normalization for medical images."""
    
    def __init__(self, method='zscore', clip_percentile=(1, 99)):
        """
        Args:
            method: 'zscore', 'minmax', 'percentile', or 'histogram'
            clip_percentile: Tuple of (low, high) percentiles for clipping
        """
        self.method = method
        self.clip_percentile = clip_percentile
    
    def normalize(self, image, mask=None):
        """
        Normalize image intensity.
        
        Args:
            image: 3D numpy array
            mask: Optional brain mask (normalize only within mask)
        
        Returns:
            Normalized image
        """
        if mask is None:
            # Create simple foreground mask
            mask = image > np.percentile(image, 1)
        
        # Extract foreground values
        foreground = image[mask]
        
        if self.method == 'zscore':
            return self._zscore_normalize(image, foreground, mask)
        elif self.method == 'minmax':
            return self._minmax_normalize(image, foreground, mask)
        elif self.method == 'percentile':
            return self._percentile_normalize(image, foreground, mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _zscore_normalize(self, image, foreground, mask):
        """Z-score normalization."""
        mean = foreground.mean()
        std = foreground.std()
        
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[mask] = (image[mask] - mean) / (std + 1e-8)
        return normalized
    
    def _minmax_normalize(self, image, foreground, mask):
        """Min-max normalization to [0, 1]."""
        min_val = foreground.min()
        max_val = foreground.max()
        
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[mask] = (image[mask] - min_val) / (max_val - min_val + 1e-8)
        return normalized
    
    def _percentile_normalize(self, image, foreground, mask):
        """Percentile-based normalization (robust to outliers)."""
        low = np.percentile(foreground, self.clip_percentile[0])
        high = np.percentile(foreground, self.clip_percentile[1])
        
        # Clip values
        clipped = np.clip(image, low, high)
        
        # Normalize to [0, 1]
        normalized = (clipped - low) / (high - low + 1e-8)
        normalized[~mask] = 0
        return normalized


def normalize_brats_subject(subject_path, output_path, normalizer):
    """Normalize all modalities for a BraTS subject."""
    import os
    from pathlib import Path
    
    subject_path = Path(subject_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    modalities = ['t1n', 't1c', 't2w', 't2f']
    
    for mod in modalities:
        # Find modality file
        mod_files = list(subject_path.glob(f"*{mod}*.nii.gz"))
        if not mod_files:
            continue
            
        # Load image
        img = nib.load(mod_files[0])
        data = img.get_fdata().astype(np.float32)
        
        # Normalize
        normalized = normalizer.normalize(data)
        
        # Save
        out_img = nib.Nifti1Image(normalized, img.affine, img.header)
        out_path = output_path / f"{subject_path.name}_{mod}_norm.nii.gz"
        nib.save(out_img, out_path)
        
    # Copy segmentation without normalization
    seg_files = list(subject_path.glob("*seg*.nii.gz"))
    if seg_files:
        import shutil
        shutil.copy(seg_files[0], output_path / f"{subject_path.name}_seg.nii.gz")


# Example usage
if __name__ == "__main__":
    normalizer = IntensityNormalizer(method='zscore')
    normalize_brats_subject(
        "data/raw/brats/BraTS2023_00000",
        "data/processed/BraTS2023_00000",
        normalizer
    )
```

### 2.2 Spatial Resampling (Week 2, Day 3-4)

#### Understanding Resampling

**Why resample?**
- Different scanners have different voxel sizes
- Models need consistent input dimensions
- Standard spacing enables fair comparison

**Target spacing for medical imaging:**
- **BraTS default:** 1mm × 1mm × 1mm (isotropic)
- **Alternative:** 0.5mm × 0.5mm × 1mm (higher in-plane resolution)

#### Implementation

```python
# code/preprocessing/spatial_resampling.py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import nibabel as nib

class SpatialResampler:
    """Resample medical images to target spacing."""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), interpolation='linear'):
        """
        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
            interpolation: 'linear', 'nearest', or 'bspline'
        """
        self.target_spacing = np.array(target_spacing)
        self.interpolation = interpolation
    
    def resample_sitk(self, image_path, output_path):
        """Resample using SimpleITK (recommended for medical images)."""
        # Load image
        image = sitk.ReadImage(str(image_path))
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        # Calculate new size
        new_size = np.round(
            original_size * original_spacing / self.target_spacing
        ).astype(int).tolist()
        
        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing.tolist())
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        
        # Set interpolation method
        if self.interpolation == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif self.interpolation == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif self.interpolation == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        
        # Resample
        resampled = resampler.Execute(image)
        
        # Save
        sitk.WriteImage(resampled, str(output_path))
        
        return {
            'original_spacing': original_spacing,
            'original_size': original_size,
            'new_spacing': self.target_spacing,
            'new_size': new_size
        }
    
    def resample_scipy(self, data, original_spacing):
        """
        Resample using scipy (faster but less precise).
        
        Args:
            data: 3D numpy array
            original_spacing: Original voxel spacing
        
        Returns:
            Resampled data
        """
        # Calculate zoom factors
        zoom_factors = np.array(original_spacing) / self.target_spacing
        
        # Choose order based on interpolation
        if self.interpolation == 'nearest':
            order = 0
        elif self.interpolation == 'linear':
            order = 1
        else:
            order = 3
        
        # Resample
        resampled = zoom(data, zoom_factors, order=order)
        
        return resampled


def resample_brats_subject(subject_path, output_path, target_spacing=(1.0, 1.0, 1.0)):
    """Resample all modalities for a BraTS subject."""
    from pathlib import Path
    
    subject_path = Path(subject_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    resampler_linear = SpatialResampler(target_spacing, interpolation='linear')
    resampler_nearest = SpatialResampler(target_spacing, interpolation='nearest')
    
    # Resample each file
    for nii_file in subject_path.glob("*.nii.gz"):
        out_file = output_path / nii_file.name.replace('.nii.gz', '_resampled.nii.gz')
        
        # Use nearest neighbor for segmentation masks
        if 'seg' in nii_file.name.lower():
            resampler_nearest.resample_sitk(nii_file, out_file)
        else:
            resampler_linear.resample_sitk(nii_file, out_file)
        
        print(f"Resampled: {nii_file.name}")


# Example usage
if __name__ == "__main__":
    resample_brats_subject(
        "data/processed/BraTS2023_00000",
        "data/processed/BraTS2023_00000_resampled",
        target_spacing=(1.0, 1.0, 1.0)
    )
```

### 2.3 Longitudinal Image Alignment (Week 2, Day 5-7)

#### Understanding Registration

**Types of Registration:**
| Type | Degrees of Freedom | Use Case |
|------|-------------------|----------|
| **Rigid** | 6 (3 rotation + 3 translation) | Same patient, minor movement |
| **Affine** | 12 (rigid + scale + shear) | Same patient, different scanners |
| **Deformable** | Many (B-spline, demons) | Significant anatomical changes |

**For longitudinal tumor imaging:** Start with **rigid registration**, then consider affine if needed.

#### Implementation

```python
# code/preprocessing/registration.py
import SimpleITK as sitk
import numpy as np
from pathlib import Path

class LongitudinalRegistration:
    """Register longitudinal images to a reference (usually first timepoint)."""
    
    def __init__(self, registration_type='rigid', metric='mi'):
        """
        Args:
            registration_type: 'rigid', 'affine', or 'deformable'
            metric: 'mi' (mutual information), 'mse', or 'ncc'
        """
        self.registration_type = registration_type
        self.metric = metric
    
    def register(self, fixed_path, moving_path, output_path):
        """
        Register moving image to fixed image.
        
        Args:
            fixed_path: Path to reference image
            moving_path: Path to image to be registered
            output_path: Path to save registered image
        
        Returns:
            Transform parameters and metrics
        """
        # Load images
        fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
        
        # Initialize registration
        registration = sitk.ImageRegistrationMethod()
        
        # Set metric
        if self.metric == 'mi':
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif self.metric == 'mse':
            registration.SetMetricAsMeanSquares()
        elif self.metric == 'ncc':
            registration.SetMetricAsCorrelation()
        
        # Set optimizer
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set transform
        if self.registration_type == 'rigid':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.registration_type == 'affine':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        
        registration.SetInitialTransform(initial_transform, inPlace=False)
        
        # Multi-resolution strategy
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Set interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Execute registration
        final_transform = registration.Execute(fixed, moving)
        
        # Apply transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        
        registered = resampler.Execute(moving)
        
        # Save result
        sitk.WriteImage(registered, str(output_path))
        
        return {
            'final_metric': registration.GetMetricValue(),
            'transform_parameters': final_transform.GetParameters(),
            'stop_condition': registration.GetOptimizerStopConditionDescription()
        }
    
    def register_segmentation(self, seg_path, transform, reference_path, output_path):
        """Apply saved transform to segmentation mask (nearest neighbor)."""
        fixed = sitk.ReadImage(str(reference_path))
        seg = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(transform)
        
        registered_seg = resampler.Execute(seg)
        sitk.WriteImage(registered_seg, str(output_path))


def register_longitudinal_series(subject_timepoints, output_dir):
    """
    Register a series of timepoints to the first one.
    
    Args:
        subject_timepoints: List of paths to timepoint directories
        output_dir: Output directory for registered images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registrator = LongitudinalRegistration(registration_type='rigid')
    
    # Use first timepoint as reference
    reference_dir = Path(subject_timepoints[0])
    reference_t1 = list(reference_dir.glob("*t1c*.nii.gz"))[0]
    
    # Copy reference to output (no registration needed)
    import shutil
    for f in reference_dir.glob("*.nii.gz"):
        shutil.copy(f, output_dir / f"tp0_{f.name}")
    
    # Register subsequent timepoints
    for i, tp_dir in enumerate(subject_timepoints[1:], start=1):
        tp_dir = Path(tp_dir)
        moving_t1 = list(tp_dir.glob("*t1c*.nii.gz"))[0]
        
        # Register T1ce (main modality)
        result = registrator.register(
            reference_t1,
            moving_t1,
            output_dir / f"tp{i}_{moving_t1.name}"
        )
        print(f"Timepoint {i} registered: metric = {result['final_metric']:.4f}")
        
        # TODO: Apply same transform to other modalities and segmentation


# Example usage
if __name__ == "__main__":
    # For demonstration - adjust paths as needed
    timepoints = [
        "data/processed/patient001_tp0",
        "data/processed/patient001_tp1",
        "data/processed/patient001_tp2"
    ]
    register_longitudinal_series(timepoints, "data/processed/patient001_registered")
```

---

## Activity 3: Tumor Annotation Processing

### 3.1 Segmentation Mask Processing (Week 3, Day 1-2)

#### BraTS Label Convention
| Label | Structure | Description |
|-------|-----------|-------------|
| 0 | Background | Non-tumor region |
| 1 | NCR/NET | Necrotic and Non-Enhancing Tumor core |
| 2 | ED | Peritumoral Edema |
| 4 | ET | Enhancing Tumor |

**Note:** Label 3 is not used in BraTS.

#### Implementation

```python
# code/preprocessing/annotation_processing.py
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path

class TumorAnnotationProcessor:
    """Process and analyze tumor segmentation masks."""
    
    # BraTS label mapping
    LABELS = {
        0: 'background',
        1: 'necrotic_core',  # NCR/NET
        2: 'edema',          # ED
        4: 'enhancing'       # ET
    }
    
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        """
        Args:
            spacing: Voxel spacing in mm for volume calculations
        """
        self.spacing = np.array(spacing)
        self.voxel_volume = np.prod(self.spacing)  # mm³
    
    def load_segmentation(self, seg_path):
        """Load segmentation mask."""
        img = nib.load(seg_path)
        return img.get_fdata().astype(np.uint8), img.affine
    
    def get_tumor_regions(self, segmentation):
        """
        Extract different tumor regions from segmentation.
        
        Returns:
            Dictionary of boolean masks for each region
        """
        return {
            'whole_tumor': segmentation > 0,
            'tumor_core': (segmentation == 1) | (segmentation == 4),
            'enhancing': segmentation == 4,
            'necrotic': segmentation == 1,
            'edema': segmentation == 2
        }
    
    def compute_tumor_statistics(self, segmentation):
        """
        Compute comprehensive tumor statistics.
        
        Returns:
            Dictionary of statistics
        """
        regions = self.get_tumor_regions(segmentation)
        stats = {}
        
        for name, mask in regions.items():
            if mask.sum() == 0:
                stats[name] = {
                    'volume_mm3': 0,
                    'volume_ml': 0,
                    'voxel_count': 0,
                    'centroid': None,
                    'bbox': None
                }
                continue
            
            # Volume
            voxel_count = mask.sum()
            volume_mm3 = voxel_count * self.voxel_volume
            volume_ml = volume_mm3 / 1000  # Convert to ml
            
            # Centroid (center of mass)
            centroid = ndimage.center_of_mass(mask)
            
            # Bounding box
            coords = np.where(mask)
            bbox = {
                'min': [c.min() for c in coords],
                'max': [c.max() for c in coords],
                'size': [c.max() - c.min() + 1 for c in coords]
            }
            
            stats[name] = {
                'volume_mm3': volume_mm3,
                'volume_ml': volume_ml,
                'voxel_count': int(voxel_count),
                'centroid': [float(c) for c in centroid],
                'bbox': bbox
            }
        
        # Compute ratios
        if stats['whole_tumor']['voxel_count'] > 0:
            wt_voxels = stats['whole_tumor']['voxel_count']
            stats['ratios'] = {
                'enhancing_to_whole': stats['enhancing']['voxel_count'] / wt_voxels,
                'necrotic_to_whole': stats['necrotic']['voxel_count'] / wt_voxels,
                'edema_to_whole': stats['edema']['voxel_count'] / wt_voxels,
                'core_to_whole': stats['tumor_core']['voxel_count'] / wt_voxels
            }
        
        return stats
    
    def validate_segmentation(self, segmentation):
        """
        Validate segmentation mask for common issues.
        
        Returns:
            List of warnings/errors
        """
        issues = []
        
        # Check for valid labels
        unique_labels = np.unique(segmentation)
        invalid_labels = set(unique_labels) - {0, 1, 2, 4}
        if invalid_labels:
            issues.append(f"Invalid labels found: {invalid_labels}")
        
        # Check for empty segmentation
        if (segmentation > 0).sum() == 0:
            issues.append("Empty segmentation (no tumor)")
        
        # Check for disconnected components
        regions = self.get_tumor_regions(segmentation)
        for name, mask in regions.items():
            if mask.sum() > 0:
                labeled, num_components = ndimage.label(mask)
                if num_components > 1:
                    issues.append(f"{name} has {num_components} disconnected components")
        
        # Check for holes in tumor core
        whole_tumor = regions['whole_tumor']
        filled = ndimage.binary_fill_holes(whole_tumor)
        if filled.sum() != whole_tumor.sum():
            issues.append("Whole tumor mask has holes")
        
        return issues


def process_dataset_annotations(data_dir, output_csv):
    """Process all annotations in a dataset."""
    import pandas as pd
    from tqdm import tqdm
    
    data_dir = Path(data_dir)
    processor = TumorAnnotationProcessor()
    
    all_stats = []
    
    for subject_dir in tqdm(list(data_dir.glob("BraTS*"))):
        seg_files = list(subject_dir.glob("*seg*.nii.gz"))
        if not seg_files:
            continue
        
        seg, _ = processor.load_segmentation(seg_files[0])
        stats = processor.compute_tumor_statistics(seg)
        issues = processor.validate_segmentation(seg)
        
        # Flatten for DataFrame
        row = {
            'subject': subject_dir.name,
            'whole_tumor_ml': stats['whole_tumor']['volume_ml'],
            'enhancing_ml': stats['enhancing']['volume_ml'],
            'necrotic_ml': stats['necrotic']['volume_ml'],
            'edema_ml': stats['edema']['volume_ml'],
            'issues': '; '.join(issues) if issues else 'None'
        }
        
        if 'ratios' in stats:
            row.update({
                'enhancing_ratio': stats['ratios']['enhancing_to_whole'],
                'necrotic_ratio': stats['ratios']['necrotic_to_whole'],
                'edema_ratio': stats['ratios']['edema_to_whole']
            })
        
        all_stats.append(row)
    
    df = pd.DataFrame(all_stats)
    df.to_csv(output_csv, index=False)
    print(f"Saved annotation statistics to {output_csv}")
    return df


# Example usage
if __name__ == "__main__":
    df = process_dataset_annotations(
        "data/raw/brats",
        "reports/tumor_statistics.csv"
    )
    print(df.describe())
```

---

## Activity 4: Exploratory Data Analysis

### 4.1 Distribution Analysis (Week 3, Day 3-7)

```python
# code/eda/exploratory_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

class MedicalImagingEDA:
    """Exploratory Data Analysis for Medical Imaging Datasets."""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_tumor_size_distribution(self, stats_df):
        """Analyze and visualize tumor size distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Whole tumor volume distribution
        ax = axes[0, 0]
        sns.histplot(stats_df['whole_tumor_ml'], kde=True, ax=ax)
        ax.set_xlabel('Volume (ml)')
        ax.set_title('Whole Tumor Volume Distribution')
        ax.axvline(stats_df['whole_tumor_ml'].median(), color='r', linestyle='--', 
                   label=f'Median: {stats_df["whole_tumor_ml"].median():.1f} ml')
        ax.legend()
        
        # Enhancing tumor volume
        ax = axes[0, 1]
        sns.histplot(stats_df['enhancing_ml'], kde=True, ax=ax, color='orange')
        ax.set_xlabel('Volume (ml)')
        ax.set_title('Enhancing Tumor Volume Distribution')
        
        # Volume boxplots
        ax = axes[1, 0]
        volume_cols = ['whole_tumor_ml', 'enhancing_ml', 'necrotic_ml', 'edema_ml']
        stats_df[volume_cols].boxplot(ax=ax)
        ax.set_ylabel('Volume (ml)')
        ax.set_title('Tumor Component Volume Comparison')
        ax.set_xticklabels(['Whole', 'Enhancing', 'Necrotic', 'Edema'])
        
        # Component ratios
        ax = axes[1, 1]
        ratio_cols = ['enhancing_ratio', 'necrotic_ratio', 'edema_ratio']
        if all(col in stats_df.columns for col in ratio_cols):
            stats_df[ratio_cols].boxplot(ax=ax)
            ax.set_ylabel('Ratio')
            ax.set_title('Tumor Component Ratios')
            ax.set_xticklabels(['Enhancing', 'Necrotic', 'Edema'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tumor_size_distribution.png', dpi=150)
        plt.close()
        
        return {
            'whole_tumor_stats': stats_df['whole_tumor_ml'].describe(),
            'enhancing_stats': stats_df['enhancing_ml'].describe()
        }
    
    def analyze_tumor_location(self, data_dir, sample_size=50):
        """Analyze tumor location distribution in brain coordinates."""
        centroids = []
        
        subjects = list(Path(data_dir).glob("BraTS*"))[:sample_size]
        
        for subject in tqdm(subjects, desc="Analyzing locations"):
            seg_file = list(subject.glob("*seg*.nii.gz"))
            if not seg_file:
                continue
            
            img = nib.load(seg_file[0])
            seg = img.get_fdata()
            
            # Get centroid of whole tumor
            tumor_mask = seg > 0
            if tumor_mask.sum() > 0:
                from scipy import ndimage
                centroid = ndimage.center_of_mass(tumor_mask)
                
                # Normalize to image dimensions
                shape = seg.shape
                normalized_centroid = [c / s for c, s in zip(centroid, shape)]
                centroids.append({
                    'subject': subject.name,
                    'x_norm': normalized_centroid[0],
                    'y_norm': normalized_centroid[1],
                    'z_norm': normalized_centroid[2]
                })
        
        centroid_df = pd.DataFrame(centroids)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # X-Y projection (axial)
        ax = axes[0]
        ax.scatter(centroid_df['x_norm'], centroid_df['y_norm'], alpha=0.5)
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_title('Tumor Location - Axial View')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # X-Z projection (coronal)
        ax = axes[1]
        ax.scatter(centroid_df['x_norm'], centroid_df['z_norm'], alpha=0.5)
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Z (normalized)')
        ax.set_title('Tumor Location - Coronal View')
        
        # Y-Z projection (sagittal)
        ax = axes[2]
        ax.scatter(centroid_df['y_norm'], centroid_df['z_norm'], alpha=0.5)
        ax.set_xlabel('Y (normalized)')
        ax.set_ylabel('Z (normalized)')
        ax.set_title('Tumor Location - Sagittal View')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tumor_location_distribution.png', dpi=150)
        plt.close()
        
        return centroid_df
    
    def analyze_intensity_distributions(self, sample_subjects=10):
        """Analyze intensity distributions across modalities."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        modalities = ['t1n', 't1c', 't2w', 't2f']
        colors = ['blue', 'orange', 'green', 'red']
        
        for ax, mod, color in zip(axes.flat, modalities, colors):
            all_values = []
            
            subjects = list(self.data_dir.glob("BraTS*"))[:sample_subjects]
            for subject in subjects:
                mod_file = list(subject.glob(f"*{mod}*.nii.gz"))
                if mod_file:
                    data = nib.load(mod_file[0]).get_fdata()
                    # Sample values (foreground only)
                    mask = data > np.percentile(data, 5)
                    sampled = np.random.choice(data[mask].flatten(), size=10000)
                    all_values.extend(sampled)
            
            ax.hist(all_values, bins=100, color=color, alpha=0.7)
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{mod.upper()} Intensity Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'intensity_distributions.png', dpi=150)
        plt.close()
    
    def analyze_inter_patient_variability(self, stats_df):
        """Analyze variability between patients."""
        # Coefficient of variation for volumes
        cv_stats = {}
        for col in ['whole_tumor_ml', 'enhancing_ml', 'necrotic_ml', 'edema_ml']:
            mean = stats_df[col].mean()
            std = stats_df[col].std()
            cv = (std / mean) * 100 if mean > 0 else 0
            cv_stats[col] = {'mean': mean, 'std': std, 'cv_percent': cv}
        
        # Create summary table
        cv_df = pd.DataFrame(cv_stats).T
        cv_df.to_csv(self.output_dir / 'variability_statistics.csv')
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        cv_df['cv_percent'].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Inter-Patient Variability by Tumor Component')
        ax.set_xticklabels(['Whole Tumor', 'Enhancing', 'Necrotic', 'Edema'], rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'inter_patient_variability.png', dpi=150)
        plt.close()
        
        return cv_df
    
    def generate_eda_report(self, stats_df):
        """Generate comprehensive EDA report."""
        report = []
        report.append("# Exploratory Data Analysis Report")
        report.append(f"## Dataset: BraTS")
        report.append(f"**Total Subjects:** {len(stats_df)}\n")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("### Tumor Volumes (ml)")
        report.append(stats_df[['whole_tumor_ml', 'enhancing_ml', 'necrotic_ml', 'edema_ml']].describe().to_markdown())
        
        # Key findings
        report.append("\n## Key Findings")
        report.append(f"- Median whole tumor volume: {stats_df['whole_tumor_ml'].median():.1f} ml")
        report.append(f"- Volume range: {stats_df['whole_tumor_ml'].min():.1f} - {stats_df['whole_tumor_ml'].max():.1f} ml")
        report.append(f"- Subjects with large tumors (>100ml): {(stats_df['whole_tumor_ml'] > 100).sum()}")
        report.append(f"- Subjects with small tumors (<10ml): {(stats_df['whole_tumor_ml'] < 10).sum()}")
        
        # Issues found
        issues_count = (stats_df['issues'] != 'None').sum()
        report.append(f"\n## Data Quality")
        report.append(f"- Subjects with annotation issues: {issues_count}")
        
        # Save report
        with open(self.output_dir / 'eda_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)


# Main EDA workflow
def run_full_eda(data_dir, output_dir):
    """Run complete EDA pipeline."""
    from annotation_processing import process_dataset_annotations
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process annotations
    print("Step 1: Processing annotations...")
    stats_df = process_dataset_annotations(data_dir, output_dir / 'tumor_statistics.csv')
    
    # Step 2: Run EDA
    print("Step 2: Running EDA...")
    eda = MedicalImagingEDA(data_dir, output_dir)
    
    print("  - Analyzing tumor size distribution...")
    eda.analyze_tumor_size_distribution(stats_df)
    
    print("  - Analyzing tumor locations...")
    eda.analyze_tumor_location(data_dir)
    
    print("  - Analyzing intensity distributions...")
    eda.analyze_intensity_distributions()
    
    print("  - Analyzing inter-patient variability...")
    eda.analyze_inter_patient_variability(stats_df)
    
    # Step 3: Generate report
    print("Step 3: Generating report...")
    eda.generate_eda_report(stats_df)
    
    print(f"\nEDA complete! Results saved to {output_dir}")


if __name__ == "__main__":
    run_full_eda("data/raw/brats", "reports/eda")
```

---

## Activity 5: Longitudinal Organization

### 5.1 Dataset Organization Structure (Week 4, Day 3-4)

```python
# code/preprocessing/organize_longitudinal.py
import json
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

class LongitudinalDataOrganizer:
    """Organize processed data into longitudinal patient sequences."""
    
    def __init__(self, processed_dir, output_dir):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_patient_manifest(self, patient_id, timepoints):
        """
        Create a manifest file for a patient's longitudinal data.
        
        Args:
            patient_id: Unique patient identifier
            timepoints: List of dicts with timepoint info
        
        Returns:
            Path to manifest file
        """
        manifest = {
            'patient_id': patient_id,
            'num_timepoints': len(timepoints),
            'created': datetime.now().isoformat(),
            'timepoints': timepoints
        }
        
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        manifest_path = patient_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
    
    def organize_brats_longitudinal(self, metadata_csv=None):
        """
        Organize BraTS data into longitudinal structure.
        
        Note: BraTS 2023 has limited longitudinal data. This creates
        a structure that can accommodate longitudinal cases when available.
        """
        # Create directory structure
        structure = {
            'train': self.output_dir / 'train',
            'val': self.output_dir / 'val',
            'test': self.output_dir / 'test'
        }
        
        for split_dir in structure.values():
            split_dir.mkdir(exist_ok=True)
        
        # Get all subjects
        subjects = list(self.processed_dir.glob("BraTS*"))
        
        # For BraTS, most cases are single timepoint
        # Create pseudo-longitudinal structure for consistency
        for i, subject in enumerate(subjects):
            patient_id = subject.name
            
            # Create patient directory
            # 80/10/10 split
            if i < len(subjects) * 0.8:
                split = 'train'
            elif i < len(subjects) * 0.9:
                split = 'val'
            else:
                split = 'test'
            
            patient_dir = structure[split] / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            # Create timepoint directory (tp0 for single timepoint)
            tp_dir = patient_dir / 'tp0'
            tp_dir.mkdir(exist_ok=True)
            
            # Copy/link processed files
            for nii_file in subject.glob("*.nii.gz"):
                dst = tp_dir / nii_file.name
                if not dst.exists():
                    shutil.copy(nii_file, dst)
            
            # Create manifest
            timepoint_info = [{
                'timepoint_id': 'tp0',
                'relative_day': 0,
                'modalities': ['t1n', 't1c', 't2w', 't2f'],
                'has_segmentation': True
            }]
            
            self.create_patient_manifest(patient_id, timepoint_info)
        
        # Create dataset summary
        summary = {
            'dataset': 'BraTS2023',
            'total_patients': len(subjects),
            'splits': {
                'train': len(list(structure['train'].glob("*"))),
                'val': len(list(structure['val'].glob("*"))),
                'test': len(list(structure['test'].glob("*")))
            },
            'longitudinal_cases': 0,  # Update when longitudinal data available
            'created': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def create_pytorch_dataset_config(data_dir, config_path):
    """Create configuration for PyTorch DataLoader."""
    config = {
        'data_root': str(data_dir),
        'modalities': ['t1n', 't1c', 't2w', 't2f'],
        'target': 'seg',
        'splits': {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        },
        'preprocessing': {
            'normalize': True,
            'crop_to_nonzero': True,
            'target_spacing': [1.0, 1.0, 1.0],
            'patch_size': [128, 128, 128]
        },
        'augmentation': {
            'flip_prob': 0.5,
            'rotation_range': [-15, 15],
            'scale_range': [0.85, 1.15],
            'intensity_shift': 0.1,
            'intensity_scale': 0.1
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


# Example usage
if __name__ == "__main__":
    organizer = LongitudinalDataOrganizer(
        "data/processed",
        "data/organized"
    )
    
    summary = organizer.organize_brats_longitudinal()
    print(f"Organized {summary['total_patients']} patients")
    
    create_pytorch_dataset_config(
        "data/organized",
        "configs/dataset_config.json"
    )
```

---

## Deliverables Checklist

### Week 1 Deliverables
- [ ] Development environment set up
- [ ] All required accounts created (Synapse, GDC, TCIA)
- [ ] BraTS dataset downloaded
- [ ] Initial data exploration notebook

### Week 2 Deliverables
- [ ] `intensity_normalization.py` - Tested and documented
- [ ] `spatial_resampling.py` - Tested and documented
- [ ] `registration.py` - Tested and documented
- [ ] Preprocessing pipeline integration tests

### Week 3 Deliverables
- [ ] `annotation_processing.py` - Complete with statistics
- [ ] `tumor_statistics.csv` - For all subjects
- [ ] Initial EDA visualizations
- [ ] Data quality report

### Week 4 Deliverables
- [ ] Complete EDA report (`reports/eda/eda_report.md`)
- [ ] Organized dataset structure
- [ ] Dataset manifest and configuration files
- [ ] PyTorch DataLoader configuration
- [ ] Phase 1 summary document

---

## Code Templates

### Quick Start: Complete Preprocessing Pipeline

```python
# code/preprocessing/pipeline.py
"""
Complete preprocessing pipeline for Phase 1.
Run this after downloading data to process entire dataset.
"""

from pathlib import Path
from tqdm import tqdm
import argparse

from intensity_normalization import IntensityNormalizer, normalize_brats_subject
from spatial_resampling import resample_brats_subject
from annotation_processing import TumorAnnotationProcessor, process_dataset_annotations

def run_preprocessing_pipeline(raw_dir, processed_dir, target_spacing=(1.0, 1.0, 1.0)):
    """
    Run complete preprocessing pipeline on raw data.
    
    Args:
        raw_dir: Directory containing raw BraTS data
        processed_dir: Output directory for processed data
        target_spacing: Target voxel spacing
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Initialize processors
    normalizer = IntensityNormalizer(method='zscore')
    
    # Get all subjects
    subjects = list(raw_dir.glob("BraTS*"))
    print(f"Found {len(subjects)} subjects")
    
    for subject in tqdm(subjects, desc="Preprocessing"):
        subject_name = subject.name
        
        # Step 1: Create output directory
        out_dir = processed_dir / subject_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Normalize intensities
        normalize_brats_subject(subject, out_dir / "normalized", normalizer)
        
        # Step 3: Resample to target spacing
        resample_brats_subject(
            out_dir / "normalized",
            out_dir / "final",
            target_spacing
        )
    
    # Step 4: Process annotations and generate statistics
    print("\nGenerating annotation statistics...")
    process_dataset_annotations(processed_dir, processed_dir / "tumor_statistics.csv")
    
    print(f"\nPreprocessing complete! Output saved to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--raw", type=str, default="data/raw/brats",
                       help="Raw data directory")
    parser.add_argument("--processed", type=str, default="data/processed",
                       help="Output directory")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Target spacing in mm")
    
    args = parser.parse_args()
    run_preprocessing_pipeline(args.raw, args.processed, tuple(args.spacing))
```

### Quick Start: Run EDA

```bash
# Run complete EDA after preprocessing
cd ~/mitacs_project
python code/eda/exploratory_analysis.py
```

---

## Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| `nibabel` can't read file | Corrupted download | Re-download file, verify checksum |
| Memory error during processing | Large 3D volumes | Process slice-by-slice or use memory mapping |
| Registration fails to converge | Very different images | Try multi-resolution, adjust parameters |
| Inconsistent spacing | Different scanner protocols | Verify original spacing, document variations |
| Missing segmentation | Annotation not available | Skip or flag for manual annotation |

---

## Next Steps After Phase 1

Upon completing Phase 1, you will have:
1. ✅ Clean, normalized, and standardized medical imaging data
2. ✅ Comprehensive understanding of tumor characteristics in your dataset
3. ✅ Organized data structure ready for model training
4. ✅ Baseline statistics for evaluating model performance

**Phase 2 Preview:**
- Implement CNN baselines (ResNet, EfficientNet)
- Begin Vision Transformer adaptation
- Compare baseline vs. transformer representations

---

*Phase 1 Implementation Guide - Mitacs Globalink Research Award at TÉLUQ University*
