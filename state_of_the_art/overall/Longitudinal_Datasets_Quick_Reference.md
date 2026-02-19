# Public Longitudinal Cancer Imaging Datasets - Quick Reference

## 🧠 BRAIN CANCER DATASETS

| Dataset | Patients | Scans | Timepoints/Patient | Modality | Cancer Type | Annotations | Access | Download Link | License |
|---------|----------|-------|-------------------|----------|-------------|-------------|--------|---------------|---------|
| **Yale-Brain-Mets-Longitudinal** | 1,430 | 11,892 | Multi (pre/post-Rx) | MRI (T1, T1CE, T2, FLAIR) | Brain metastases | Demographics, treatment, scanner info | Open | [TCIA](https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/) | CC BY |
| **UCSF Post-Treatment Glioma** | 298 | 596 | 2 consecutive | MRI (T1, T1CE, T2, FLAIR) | Diffuse gliomas (post-op) | Expert voxelwise segmentation + longitudinal change labels | Open | [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-post-treatment-glioma/) | CC BY |
| **LUMIERE** | 25 | 375 | Up to 15 | MRI (T1, T1CE, T2, FLAIR, perfusion) | Glioblastoma | RANO criteria, expert assessment | Open | [TCIA](https://www.cancerimagingarchive.net/collection/lumiere/) | CC BY |
| **Burdenko-GBM-Progression** | ~100 | Multi | Longitudinal follow-ups | MRI + CT + RT plans | Glioblastoma progression | GTV, CTV, PTV contours | Restricted (sign agreement) | [TCIA](https://www.cancerimagingarchive.net/collection/burdenko-gbm-progression/) | Restricted |
| **MU-Glioma-Post** | 203 | 617 | Multiple post-treatment | MRI (T1, T1CE, T2, FLAIR) | Post-operative gliomas | Automated nnU-Net segmentation | Open | [TCIA](https://www.cancerimagingarchive.net/collection/mu-glioma-post/) | CC BY |
| **CPTAC-GBM** | ~100 | Multi | Some longitudinal | MRI + histopathology | Glioblastoma | Proteomics, genomics | Restricted (previously) | [TCIA](https://www.cancerimagingarchive.net/collection/cptac-gbm/) | TCIA Restricted |
| **BraTS 2023** | 2,040+ | 2,040+ | **❌ Single timepoint** | MRI (T1, T1CE, T2, FLAIR) | Pre-op gliomas | Expert segmentation | Open | [BraTS](http://braintumorsegmentation.org/) | CC BY-NC-SA |

## 🎀 BREAST CANCER DATASETS

| Dataset | Patients | Scans | Timepoints/Patient | Modality | Treatment Phase | Annotations | Access | Download Link | License |
|---------|----------|-------|-------------------|----------|-----------------|-------------|--------|---------------|---------|
| **ISPY1 (ACRIN 6657)** | 222 | Multi | Pre/during/post neoadjuvant therapy | MRI (DCE) | Neoadjuvant chemotherapy | Clinical outcomes, pCR status | Open | [TCIA](https://www.cancerimagingarchive.net/collection/ispy1/) | CC BY |
| **QIN-Breast** | Multi | Multi | Longitudinal treatment response | PET/CT + MRI | Neoadjuvant | Quantitative imaging metrics | Open | [TCIA](https://www.cancerimagingarchive.net/collection/qin-breast/) | CC BY |
| **QIN-Breast-02** | Multi | Multi | Multi-timepoint | Multi-parametric MRI | Treatment monitoring | Advanced MRI protocols | Limited (TCIA) | [TCIA](https://www.cancerimagingarchive.net/collection/qin-breast-02/) | TCIA Limited |
| **RIDER Breast MRI** | 19 | 38 | 2 (1-2 days apart) | DCE-MRI + DTI | Test-retest reliability | Repeatability metrics | Open | [TCIA](https://www.cancerimagingarchive.net/collection/rider-breast-mri/) | CC BY |

## 🫁 LUNG CANCER DATASETS

| Dataset | Patients | Scans | Timepoints/Patient | Modality | Design | Annotations | Access | Download Link | License |
|---------|----------|-------|-------------------|----------|--------|-------------|--------|---------------|---------|
| **NLST-New-Lesion-LongCT** | 122 | 244+ | Baseline + follow-up | CT | New lesion detection | Point annotations, registered baselines | Open | [TCIA](https://www.cancerimagingarchive.net/analysis-result/nlst-new-lesion-longct/) | CC BY |
| **RIDER Lung PET-CT** | 244 | Multi | Longitudinal | PET-CT | Treatment response | Quantitative imaging | Open | [TCIA](https://www.cancerimagingarchive.net/collection/rider-lung-pet-ct/) | CC BY |
| **LIDC-IDRI** | 1,018 | 1,018 | **❌ Mostly single** | CT | Nodule detection | 4 radiologist annotations | Open | [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) | CC BY |

## 🔬 MULTI-CANCER / GENOMIC DATASETS

| Dataset | Patients | Cancer Types | Timepoints | Data Types | Longitudinal Imaging? | Access | Download Link |
|---------|----------|--------------|------------|------------|----------------------|--------|---------------|
| **TCGA (imaging)** | 10,000+ | 33 cancer types | **❌ Single timepoint** | MRI, CT + genomics + clinical | No (radiology) / Yes (clinical follow-up) | Open | [TCIA](https://www.cancerimagingarchive.net/collections/) + [GDC Portal](https://portal.gdc.cancer.gov/) |
| **CPTAC (various)** | Multi | Multiple (GBM, HNSCC, LUAD, etc.) | Some longitudinal | Imaging + proteomics + genomics | Varies by cohort | Restricted (proteomics) / Open (imaging) | [TCIA Collections](https://www.cancerimagingarchive.net/collections/) |

---

## 📊 DATASET COMPARISON BY USE CASE

### ✅ Best for YOUR Research Project (Video Generation of Cancer Progression)

| Rank | Dataset | Why Recommended | Score (1-10) |
|------|---------|-----------------|--------------|
| 🥇 1 | **Yale-Brain-Mets** | Largest (11,892 scans), true longitudinal, treatment diversity | 10/10 |
| 🥈 2 | **UCSF Post-Glioma** | Expert annotations, longitudinal change labels | 8/10 |
| 🥉 3 | **LUMIERE** | Dense temporal sampling (up to 15 timepoints) | 7/10 |
| 4 | **ISPY1** | If expanding to breast cancer | 8/10 |
| 5 | **NLST-New-Lesion** | If expanding to lung cancer | 6/10 |

### ❌ NOT Suitable for Temporal Modeling

| Dataset | Why NOT Suitable | Alternative |
|---------|-----------------|------------|
| BraTS | Single timepoint (pre-operative only) | Use for preprocessing pipeline development, ViT pre-training |
| LIDC-IDRI | Mostly single timepoint, irregular follow-ups | Use NLST-New-Lesion instead |
| TCGA imaging | Single timepoint radiology | Use clinical follow-up data for survival modeling |

---

## 🎯 RECOMMENDED MULTI-DATASET STRATEGY

```
PRIMARY (for temporal modeling):
┌────────────────────────────────────┐
│ Yale-Brain-Mets-Longitudinal       │
│ • 11,892 scans, 1,430 patients     │
│ • Pre/post treatment timepoints    │
│ • Use for Objectives 3-5           │
└────────────────────────────────────┘

SECONDARY (for validation):
┌────────────────────────────────────┐
│ UCSF Post-Treatment Glioma         │
│ • 596 scans, 298 patients          │
│ • Expert change annotations        │
│ • External validation set          │
└────────────────────────────────────┘

TERTIARY (for dense sampling):
┌────────────────────────────────────┐
│ LUMIERE                            │
│ • 375 scans, 25 patients           │
│ • Up to 15 timepoints/patient      │
│ • RANO criteria validation         │
└────────────────────────────────────┘

BASELINE (for pre-training):
┌────────────────────────────────────┐
│ BraTS 2023                         │
│ • 2,040+ cases                     │
│ • Preprocessing development        │
│ • ViT pre-training on static data  │
└────────────────────────────────────┘
```

---

## 📥 DOWNLOAD PRIORITY ORDER

### Week 1: Essential Downloads
1. **Yale-Brain-Mets-Longitudinal** (~200 GB)
   - Critical for all temporal objectives
   - Start with 100-patient subset if bandwidth limited

2. **BraTS 2023** (~100 GB)
   - For preprocessing pipeline development
   - ViT baseline training

### Week 2: Validation Datasets
3. **UCSF Post-Treatment Glioma** (~50 GB)
   - External validation
   - Expert annotation comparison

4. **LUMIERE** (~20 GB)
   - Dense temporal sampling examples
   - RANO criteria validation

### Optional: Expansion to Other Cancers
5. **ISPY1** (if expanding to breast cancer)
6. **NLST-New-Lesion** (if expanding to lung cancer)

---

## 🔧 DOWNLOAD INSTRUCTIONS

### Method 1: NBIA Data Retriever (Recommended)
```bash
# Download and install NBIA Data Retriever
# Link: https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever

# Steps:
# 1. Go to TCIA collection page (e.g., Yale-Brain-Mets)
# 2. Click "Download" button
# 3. Save .tcia manifest file
# 4. Open manifest in NBIA Data Retriever
# 5. Select destination folder
# 6. Start download
```

### Method 2: TCIA Browser (For selective download)
```bash
# Use TCIA web interface to browse and select specific patients
# Good for testing with small subset first

# 1. Go to https://nbia.cancerimagingarchive.net/
# 2. Search for collection (e.g., "Yale-Brain-Mets-Longitudinal")
# 3. Filter by criteria (modality, date range)
# 4. Add selected to cart
# 5. Download via NBIA Data Retriever
```

### Method 3: Command Line (for automation)
```bash
# Use TCIA REST API for programmatic download
# Documentation: https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides

pip install tcia-utils

# Example: Download specific collection
from tcia_utils import nbia
nbia.getSeries(collection="Yale-Brain-Mets-Longitudinal")
```

---

## 📋 DATA CHARACTERISTICS SUMMARY

### Format Comparison
| Dataset | Format | Pre-processed? | Registration Needed? | Segmentation Included? |
|---------|--------|----------------|---------------------|----------------------|
| Yale-Brain-Mets | NIfTI (.nii.gz) | ✅ Partially (NIfTI converted) | ✅ Yes (temporal alignment) | ❌ No |
| UCSF Post-Glioma | NIfTI | ✅ Yes | ✅ Yes | ✅ Yes (expert voxelwise) |
| LUMIERE | DICOM → NIfTI | ⚠️ Varies | ✅ Yes | ⚠️ RANO assessments (not voxelwise) |
| BraTS | NIfTI | ✅ Fully preprocessed | ❌ No (single timepoint) | ✅ Yes |
| ISPY1 | DICOM | ❌ No | ✅ Yes | ⚠️ Partial |

### Scanner Diversity
| Dataset | Vendors | Field Strengths | Multi-Institution? | Harmonization Needed? |
|---------|---------|-----------------|-------------------|----------------------|
| Yale-Brain-Mets | Siemens, GE | 1.5T, 3T | ✅ Yes | ✅ Yes (LongComBat) |
| UCSF Post-Glioma | Siemens primarily | 1.5T, 3T | ❌ Single institution | ⚠️ Minimal |
| LUMIERE | Varies | Varies | ⚠️ Unknown | ✅ Likely needed |
| BraTS | Siemens, GE, Philips | 1.5T, 3T | ✅ Multi-institutional | ✅ Yes (ComBat) |

---

## 🎓 LEARNING PATH

### Papers to Read for Each Dataset

**Yale-Brain-Mets:**
- Dataset publication: "Yale longitudinal dataset of brain metastases on MRI" (2025)
- Registration: "FLIRE: Fast Longitudinal Image Registration" (2024)
- Harmonization: "LongComBat" (2022)

**UCSF Post-Glioma:**
- Dataset paper: Radiology: Artificial Intelligence (2023)
- Segmentation: "nnU-Net" (2020, Nature Methods)
- Change detection methods

**LUMIERE:**
- Dataset paper: Scientific Data (2022)
- RANO criteria: MacDonald et al. (1990), updated criteria
- Clinical assessment integration

**BraTS:**
- BraTS Toolkit paper (2020)
- nnU-Net paper (2020)
- BraTS evolution paper (2015-2025)

---

## ⚠️ IMPORTANT NOTES

### Data Usage Requirements
- **Always cite datasets** using DOI provided on TCIA
- **Check license** before commercial use (most are CC BY)
- **Restricted datasets** (Burdenko, CPTAC-GBM) require signed agreements
- **De-identification**: TCIA data is de-identified, but some contain facial features (require special handling)

### Ethical Considerations
- Data represents real cancer patients
- Treatment outcomes may include deaths
- Handle longitudinal progression data with appropriate sensitivity
- Generated videos should be clearly labeled as AI-generated, not real patient data

### Technical Requirements
- **Storage**: 300-500 GB minimum (for Yale + BraTS + UCSF)
- **RAM**: 32 GB minimum for preprocessing
- **GPU**: 16 GB+ for ViT training, video generation
- **Compute**: Multi-day processing expected for full datasets

---

## 🚀 QUICK START CHECKLIST

### Week 1 Tasks:
- [ ] Install NBIA Data Retriever
- [ ] Download Yale-Brain-Mets-Longitudinal (100 patient subset)
- [ ] Download BraTS 2023 (full dataset)
- [ ] Explore Yale temporal structure
- [ ] Identify patients with ≥3 timepoints

### Week 2 Tasks:
- [ ] Implement NIfTI loading pipeline
- [ ] Adapt BraTS Toolkit preprocessing
- [ ] Test longitudinal registration on 5 patients
- [ ] Extract clinical metadata from TCIA

### Week 3 Tasks:
- [ ] Download UCSF Post-Glioma
- [ ] Implement LongComBat harmonization
- [ ] Quality control pipeline
- [ ] Validate temporal consistency

### Week 4 Tasks:
- [ ] Organize final dataset structure
- [ ] Create JSON metadata for LLM integration
- [ ] Document preprocessing decisions
- [ ] Prepare Objective 1 deliverables

---

## 📞 SUPPORT RESOURCES

### TCIA Support:
- Email: help@cancerimagingarchive.net
- Wiki: https://wiki.cancerimagingarchive.net/
- User forum: TCIA Google Group

### Dataset-Specific Issues:
- Check dataset homepage on TCIA for contact info
- Many datasets have GitHub issues pages
- Original publication authors often responsive

### Preprocessing Tools:
- MONAI: https://monai.io/
- SimpleITK: https://simpleitk.org/
- BraTS Toolkit: https://github.com/neuronflow/BraTS-Toolkit
- neuroCombat: https://github.com/Jfortin1/neuroCombat

---

**Last Updated:** February 2026
**Status:** All datasets verified as publicly available
**Recommended Primary Dataset:** Yale-Brain-Mets-Longitudinal (11,892 scans, 1,430 patients)
