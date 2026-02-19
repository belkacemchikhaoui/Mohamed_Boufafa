# Objective 1 Quick Reference: 4-Week Implementation Plan

## 🎯 GOAL: Robust Longitudinal Cancer Imaging Pipeline

```
INPUT:  Raw BraTS + Yale-Brain-Mets datasets
OUTPUT: Preprocessed, registered, harmonized data → Ready for Objectives 2-5
TIME:   4 weeks
```

---

## 📅 WEEK-BY-WEEK VISUAL ROADMAP

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          WEEK 1: FOUNDATION                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📚 READ (8 hours):                    💻 IMPLEMENT (32 hours):             │
│  ├─ BraTS Toolkit paper                ├─ Install: MONAI, SimpleITK         │
│  │  Focus: Section 2.2, 2.4            ├─ Download BraTS (100 GB)           │
│  ├─ nnU-Net paper                      ├─ Basic preprocessing pipeline:     │
│  │  Focus: Preprocessing sections      │  1. Load NIfTI                     │
│  └─ BraTS 2021-2025 evolution          │  2. Orientation → RAS              │
│                                         │  3. Resample → 1mm³                │
│  📊 DATASET:                            │  4. Normalize → [0,1]              │
│  └─ BraTS 2023 (2,040 cases)           │  5. Crop foreground                │
│                                         │  6. QC validation                  │
│  🎯 VALIDATION:                         └─ Process all 2,040 BraTS cases    │
│  └─ Compare to official BraTS preprocessing (Dice > 0.99)                   │
│                                                                              │
│  ✅ DELIVERABLE: Validated preprocessing pipeline + Preprocessed BraTS      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    WEEK 2: LONGITUDINAL REGISTRATION                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📚 READ (12 hours):                   💻 IMPLEMENT (28 hours):             │
│  ├─ Treatment-Aware Registration       ├─ Download Yale (~200 GB)           │
│  │  ⚠️ CRITICAL - Read thoroughly!     ├─ Yale data loader                  │
│  ├─ FLIRE fast registration            ├─ Tumor-preserving registration:    │
│  └─ Yale dataset paper                 │  • Register brain (not tumor)      │
│     (YOUR primary dataset!)            │  • Affine only (preserve volumes)  │
│                                         │  • Validate <5% volume error       │
│  📊 DATASET:                            └─ Test on 50 Yale patients (pilot) │
│  └─ Yale-Brain-Mets (1,430 patients)                                        │
│                                                                              │
│  🎯 KEY CONCEPT:                                                             │
│  Standard registration HIDES tumor changes by warping them away!            │
│  Solution: Register AROUND tumor, preserve biological changes               │
│                                                                              │
│  ✅ DELIVERABLE: Tumor-preserving registration code + 50 registered cases   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                 WEEK 3: HARMONIZATION & SCALE-UP                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📚 READ (8 hours):                    💻 IMPLEMENT (32 hours):             │
│  ├─ ComBat harmonization               ├─ Install neuroCombat               │
│  └─ LongComBat (longitudinal version)  ├─ Analyze Yale scanner variability  │
│     ⚠️ CRITICAL for Yale multi-scanner │ ├─ Extract radiomics features      │
│                                         │ └─ Apply LongComBat                │
│  🔍 SCANNER ANALYSIS:                   │                                    │
│  Yale = 2000-2019 data                  └─ Register ALL 1,430 Yale patients │
│  → Multiple scanner upgrades!                                               │
│  → Harmonization REQUIRED                                                   │
│                                                                              │
│  🎯 KEY CONCEPT:                                                             │
│  LongComBat removes scanner effects WHILE preserving temporal trajectories  │
│                                                                              │
│  ✅ DELIVERABLE: Harmonized Yale dataset + Registration quality report      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                  WEEK 4: QUALITY CONTROL & ORGANIZATION                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  💻 IMPLEMENT (40 hours):                                                    │
│  ├─ Comprehensive QC pipeline:                                              │
│  │  • Resolution check (1mm³?)                                              │
│  │  • Orientation check (RAS?)                                              │
│  │  • Intensity range check ([0,1]?)                                        │
│  │  • Registration quality (MI > threshold?)                                │
│  │  • Artifact detection                                                    │
│  │  • Temporal consistency                                                  │
│  │                                                                           │
│  ├─ Organize for downstream objectives:                                     │
│  │  • ViT training format (Objective 2)                                     │
│  │  • LLM metadata JSON (Objective 3)                                       │
│  │  • Video sequences (Objective 4)                                         │
│  │  • Train/val/test splits (Objective 5)                                   │
│  │                                                                           │
│  └─ Documentation:                                                           │
│     • Technical report                                                       │
│     • Code documentation                                                     │
│     • Data quality metrics                                                  │
│     • Preprocessing decisions log                                           │
│                                                                              │
│  ✅ DELIVERABLE: Complete preprocessed dataset + Technical report           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 PAPERS TO READ (In Order)

### Essential (MUST READ)

| # | Paper | Hours | Why Critical | Link |
|---|-------|-------|--------------|------|
| 1 | **BraTS Toolkit** | 3h | Learn preprocessing workflow | [Paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.00125/full) + [GitHub](https://github.com/neuronflow/BraTS-Toolkit) |
| 2 | **nnU-Net** | 3h | Automated preprocessing philosophy | [Paper](https://www.nature.com/articles/s41592-020-01008-z) + [GitHub](https://github.com/MIC-DKFZ/nnUNet) |
| 3 | **Treatment-Aware Registration** | 4h | ⚠️ CRITICAL for longitudinal data | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72104-5_60) + [GitHub](https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration) |
| 4 | **Yale Dataset Paper** | 3h | YOUR primary dataset description | [Paper](https://www.nature.com/articles/s41597-024-04186-3) |
| 5 | **LongComBat** | 3h | Harmonize while preserving temporal changes | [Paper](https://www.sciencedirect.com/science/article/pii/S1053811922001136) |

**Total Reading Time: ~16 hours**

### Optional (If Time Permits)

| Paper | Hours | Purpose |
|-------|-------|---------|
| FLIRE Registration | 2h | Speed optimization |
| ComBat Original | 2h | Understanding harmonization theory |
| BraTS Evolution 2021-2025 | 2h | Quality control standards |

---

## 💻 CODE STRUCTURE YOU'LL BUILD

```python
preprocessing_project/
│
├── data/
│   ├── raw/
│   │   ├── BraTS_2023/           # Downloaded Week 1
│   │   └── Yale_Brain_Mets/      # Downloaded Week 2
│   └── preprocessed/
│       ├── BraTS_preprocessed/   # Week 1 output
│       └── Yale_preprocessed/    # Week 2-4 output
│
├── src/
│   ├── preprocessing/
│   │   ├── brats_pipeline.py     # Week 1
│   │   ├── registration.py       # Week 2-3
│   │   └── harmonization.py      # Week 3
│   ├── quality_control/
│   │   └── qc_checks.py          # Week 4
│   └── data_organization/
│       └── organize_for_objectives.py  # Week 4
│
├── notebooks/
│   ├── week1_brats_exploration.ipynb
│   ├── week2_yale_registration.ipynb
│   ├── week3_harmonization.ipynb
│   └── week4_final_validation.ipynb
│
├── configs/
│   ├── preprocessing_config.yaml
│   └── dataset_splits.json
│
├── outputs/
│   ├── objective2_vit_data/       # For ViT training
│   ├── objective3_llm_metadata/   # For LLM integration
│   ├── objective4_video_sequences/ # For video generation
│   └── objective5_evaluation/     # For validation
│
└── docs/
    ├── technical_report.md        # Week 4 deliverable
    └── preprocessing_log.md
```

---

## 🔧 CORE IMPLEMENTATION SNIPPETS

### Week 1: Basic Preprocessing

```python
from monai.transforms import Compose, LoadImaged, Spacingd, Orientationd

preprocessing = Compose([
    LoadImaged(keys=["T1", "T1CE", "T2", "FLAIR"]),
    Orientationd(keys=["T1", "T1CE", "T2", "FLAIR"], axcodes="RAS"),
    Spacingd(keys=["T1", "T1CE", "T2", "FLAIR"], pixdim=(1.0, 1.0, 1.0)),
    ScaleIntensityRanged(keys=["T1", "T1CE", "T2", "FLAIR"], 
                        a_min=0, a_max=None, b_min=0, b_max=1),
    CropForegroundd(keys=["T1", "T1CE", "T2", "FLAIR"])
])
```

### Week 2: Tumor-Preserving Registration

```python
import SimpleITK as sitk

def register_preserving_tumor(baseline, followup):
    # Register brain EXCLUDING tumor region
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation()
    registration.SetOptimizerAsLBFGSB()
    
    # Affine only (preserves volumes!)
    transform = sitk.AffineTransform(3)
    registration.SetInitialTransform(transform)
    
    # Execute
    final_transform = registration.Execute(baseline, followup)
    registered = sitk.Resample(followup, baseline, final_transform)
    
    return registered
```

### Week 3: Longitudinal Harmonization

```python
from neuroCombat import neuroCombat

harmonized = neuroCombat(
    dat=features.T,
    covars=metadata,
    batch_col='scanner',
    longitudinal=True,           # KEY: Preserve temporal changes
    patient_id_col='patient_id'  # Link timepoints
)
```

---

## ✅ SUCCESS METRICS

### Week 1
- [ ] Preprocessing pipeline complete
- [ ] Validated: Dice score > 0.99 vs official BraTS
- [ ] All 2,040 BraTS cases processed
- [ ] QC report generated

### Week 2
- [ ] Yale data loader implemented
- [ ] 50 patients registered (pilot)
- [ ] Tumor volume preserved <5% error
- [ ] Registration quality > 0.8 (mutual information)

### Week 3
- [ ] Scanner variability analyzed
- [ ] Harmonization applied
- [ ] Temporal trajectories preserved
- [ ] All 1,430 Yale patients processed

### Week 4
- [ ] Comprehensive QC passed
- [ ] Data organized for Objectives 2-5
- [ ] Technical report complete
- [ ] Code documented

---

## 🎯 ALIGNMENT WITH OTHER OBJECTIVES

```
OBJECTIVE 1 → OBJECTIVE 2 (ViT Training)
├─ Standardized images (1mm³) → Consistent ViT input
├─ Normalized intensity → Stable training
├─ Registered sequences → Learn temporal patterns
└─ Harmonized features → Learn biology, not scanners

OBJECTIVE 1 → OBJECTIVE 3 (LLM Integration)
├─ Clinical metadata JSON → LLM context
├─ Temporal timestamps → Timeline narratives
├─ Volume measurements → Quantitative facts
└─ Treatment alignment → Response explanations

OBJECTIVE 1 → OBJECTIVE 4 (Video Generation)
├─ Registered sequences → Spatially consistent frames
├─ Temporal ordering → Correct progression
├─ Harmonized scans → Temporally consistent appearance
└─ QC data → Artifact-free training

OBJECTIVE 1 → OBJECTIVE 5 (Evaluation)
├─ Validated preprocessing → Fair baselines
├─ Volume ground truth → Prediction validation
├─ Multi-dataset QC → Cross-validation
└─ Documentation → Reproducible evaluation
```

---

## 🚨 CRITICAL WARNINGS

### ⚠️ Don't Skip These Steps:

1. **Week 1: Validation against BraTS Toolkit**
   - If your preprocessing differs from official → debug NOW
   - Don't proceed to Week 2 without validation

2. **Week 2: Tumor Volume Preservation**
   - If registration changes volumes >5% → registration is WRONG
   - This will ruin Objectives 4-5 (video generation, evaluation)

3. **Week 3: Temporal Trajectory Preservation**
   - Standard ComBat will DESTROY temporal patterns
   - MUST use LongComBat for longitudinal data

4. **Week 4: Quality Control**
   - Don't skip QC - bad data = bad models
   - Flag and manually review outliers

---

## 📊 EXPECTED FINAL OUTPUT

### Preprocessed Dataset Statistics

```
BraTS 2023 Preprocessed:
├─ Cases: 2,040
├─ Resolution: 1mm³ isotropic
├─ Format: NIfTI
├─ Size: ~100 GB preprocessed
└─ QC Pass Rate: >95%

Yale-Brain-Mets Preprocessed:
├─ Patients: 1,430
├─ Total Scans: 11,892
├─ Patients with ≥3 timepoints: ~800
├─ Resolution: 1mm³ isotropic
├─ Registered: Yes (tumor-preserving)
├─ Harmonized: Yes (LongComBat)
└─ QC Pass Rate: >90%

Ready for Objectives 2-5:
├─ ViT training data: ✅
├─ LLM metadata: ✅
├─ Video sequences: ✅
└─ Evaluation splits: ✅
```

---

## 🔗 QUICK LINKS REFERENCE

### Download Tools
- **NBIA Data Retriever:** https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever

### Essential GitHub Repos
- **BraTS Toolkit:** https://github.com/neuronflow/BraTS-Toolkit
- **nnU-Net:** https://github.com/MIC-DKFZ/nnUNet
- **Treatment-Aware Registration:** https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration
- **FLIRE:** https://github.com/michelle-tong18/FLIRE-MRI-registration
- **neuroCombat:** https://github.com/Jfortin1/neuroCombat

### Documentation
- **MONAI:** https://docs.monai.io/
- **SimpleITK:** https://simpleitk.readthedocs.io/
- **TCIA:** https://www.cancerimagingarchive.net/

---

## 💡 PRO TIPS

### Time Management
- **Don't over-read:** 2-3 hours max per paper, focus on implementation sections
- **Test on small data first:** Don't process 1,430 patients until pilot (10-50 patients) works
- **Document as you go:** Future you will thank present you

### Common Pitfalls to Avoid
1. **Skipping validation** → You won't know if your pipeline is correct
2. **Using deformable registration** → Will hide real tumor changes
3. **Using standard ComBat on longitudinal data** → Will destroy temporal patterns
4. **Not running QC** → Bad data = bad models downstream

### When to Ask for Help
- Registration quality consistently <0.7 (should be >0.8)
- Tumor volumes changing >10% after registration (should be <5%)
- >10% of data failing QC (should be <5%)
- Implementation taking 2× longer than estimated

---

**🚀 START HERE: Week 1, Day 1 → Install libraries + Download BraTS + Read BraTS Toolkit paper**

**By Week 4, you'll have a publication-quality preprocessing pipeline and data ready for cutting-edge ViT training, LLM integration, and video generation!**
