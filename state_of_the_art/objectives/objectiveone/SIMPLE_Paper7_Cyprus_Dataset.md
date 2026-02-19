# Paper 7: Cyprus Brain Metastases Dataset (2025) - PROTEAS Dataset

## 1. ONE-SENTENCE SUMMARY
A **smaller but MORE detailed** longitudinal brain metastases dataset from Cyprus with **744 MRI scans from 40 patients** including tumor segmentations, radiotherapy plans, CT scans, clinical data, AND radiomic features—perfect for **validating models trained on Yale** or combining datasets for better diversity!

---

## 2. KEY RESULTS (The Numbers That Matter!)

### Dataset Size Comparison

| Feature | **Yale Dataset (Paper 3)** | **Cyprus Dataset (Paper 7)** |
|---------|---------------------------|------------------------------|
| **Patients** | 1,430 | 40 |
| **Total scans** | 11,884 | 744 |
| **Scans/patient** | 8.3 average | **18.6 average** 🌟 |
| **Tumors** | Unknown total | 65 (labeled!) |
| **Time span** | 2004-2023 (20 years) | 2019-2024 (5 years) |
| **Scanner types** | Multiple (need harmonization) | 3T (68.5%) + 1.5T (31.5%) |
| **Cost** | FREE | FREE |

### What Cyprus Dataset ADDS (Yale Doesn't Have!)

✅ **Detailed Tumor Subregions** (Yale doesn't provide this!)
- Enhancing tumor (yellow in paper)
- Necrotic core (red)
- Peritumoral edema (green) - swelling around tumor
- **All manually verified by neuroradiologist with 10+ years experience!**

✅ **Radiotherapy Plans (45 RTP files)**
- Shows EXACT radiation dose distribution
- Includes CT scans for treatment planning
- 5 patients had TWO separate treatments (new metastases appeared)

✅ **Clinical Data Excel Files**
- Demographics (age, gender, ethnicity)
- Primary tumor origins: Lung 67.5%, Breast 32.5%
- Karnofsky performance status (how functional is patient)
- Neurocognitive status (brain function)
- Chemotherapy details
- Survival data (time to death or last follow-up)

✅ **Radiomic Features (110 per scan!)**
- 19 first-order statistics (intensity, mean, variance)
- 16 3D shape descriptors (volume, surface area, sphericity)
- 75 textural features (patterns, heterogeneity)
- Pre-extracted for ALL scans! (saves processing time)

### Follow-up Schedule (More Frequent Than Yale!)

**Cyprus intervals**: 6 weeks, 3 months, 6 months, 9 months, 12 months
- Average **18.6 scans per patient** (more than double Yale's 8.3!)
- All patients had **at least 1 follow-up** (many had 5-6!)

**Yale**: Variable intervals, average 8.3 scans

### Segmentation Quality

**Method**: Hybrid approach (better than single method!)
1. **Automatic**: DeepBraTumIA + nnU-Net models
2. **Semi-automatic**: Imalytics Preclinical (for missed lesions)
3. **Manual correction**: Expert neuroradiologist verified ALL segmentations
4. **Format**: BraTS-compliant (same as Yale preprocessing target!)

**Result**: High-quality, reliable tumor labels (ready to use!)

---

## 3. WHAT'S NEW? (Innovation!)

### 🎯 **Post-Treatment Longitudinal Data** (Most Datasets Only Have Pre-Treatment!)
- **Most BM datasets**: Only baseline scans before treatment
- **Cyprus dataset**: Baseline + 1-6 follow-ups AFTER radiotherapy
- **Why groundbreaking**: Can track HOW tumors respond to treatment over time!
  - Tumor shrinking? (good response)
  - Tumor growing? (treatment failure)
  - New metastases appearing? (disease progression)

### 📊 **Pre-Extracted Radiomic Features** (Saves Researchers MONTHS!)
- **Typical workflow**: Download scans → preprocess → segment → extract features (weeks of work!)
- **Cyprus dataset**: Features ALREADY extracted and provided in Excel file!
- **110 features per scan** × 744 scans = **81,840 features ready to use!**

**What are radiomic features?**
Think of them as "tumor personality traits":
- **Shape**: Is tumor round? Spiky? Irregular?
- **Intensity**: Is tumor bright? Dark? Uniform?
- **Texture**: Is tumor smooth? Rough? Heterogeneous (mixed)?

**Example use**: Train machine learning to predict "Will this tumor respond to treatment?" using these features

### 🏥 **Radiotherapy Plans Included** (Unique!)
- **What it is**: 3D map showing where radiation beams hit the brain
- **Why valuable**: Can correlate radiation dose with tumor response
- **Research questions enabled**:
  - "Did high-dose radiation shrink tumor more?"
  - "Did radiation cause side effects in surrounding tissue?"
  - "Can we optimize radiation planning using AI?"

### 🔬 **Multi-Modal Validation** (Trust the Data!)
- 3 different segmentation algorithms cross-checked
- Expert radiologist manually verified EVERY tumor
- Quality control: excluded scans with motion artifacts (blurry images)

### 🌍 **Geographic Diversity** (Cyprus Population)
- Yale dataset: USA population
- Cyprus dataset: Mediterranean/Middle Eastern population
- **Why matters**: Different genetics, different tumor patterns
- **Combined use**: More diverse training data → better AI generalization!

---

## 4. LIMITATIONS (What's Missing?)

### ❌ **MUCH Smaller Sample Size**
- **40 patients** vs Yale's **1,430 patients** (35× smaller!)
- **65 tumors** total vs Yale's thousands
- **Risk**: Models trained ONLY on Cyprus → might not generalize
- **Solution**: Use Cyprus for validation, not primary training

### ❌ **Limited Primary Tumor Types**
- **Cyprus**: 67.5% lung, 32.5% breast (only 2 types!)
- **Yale**: More diverse primary cancers
- **Missing**: Melanoma, colorectal, renal (common BM sources)
- **Impact**: Can't study how tumor origin affects metastasis patterns

### ❌ **Shorter Time Span (2019-2024)**
- Cyprus: 5 years of data
- Yale: 20 years (2004-2023)
- **Missing**: Long-term survival data (10+ years)
- **Missing**: Evolution of treatment techniques over decades

### ❌ **Scanner Variability Less Severe**
- Cyprus: Only 3T + 1.5T, 3 manufacturers (Philips 68.5%, Siemens 29.9%, Toshiba 1.6%)
- Yale: 20 years of scanner evolution (bigger harmonization challenge)
- **Pro**: Easier to harmonize Cyprus data
- **Con**: Doesn't test AI robustness to extreme scanner differences

### ❌ **Only Stereotactic Radiotherapy Patients**
- All patients got SRS/FSRT (focused radiation)
- **Missing**: Whole-brain radiotherapy (WBRT) patients
- **Missing**: Surgery-only patients
- **Missing**: Chemotherapy-only patients
- **Impact**: Can't compare different treatment modalities

### ❌ **Small Lesions Excluded**
- Tumors < 3mm ignored
- **Why**: Hard to segment accurately at that size
- **Problem**: Early detection models can't train on tiny tumors
- **Yale advantage**: Likely includes smaller lesions

---

## 5. METHODS (How It Works - Simple Steps!)

### **Overview: Multi-Stage Data Collection + Processing**

### STEP 1: Patient Selection (Inclusion Criteria)
**Who qualified:**
- ✅ Confirmed brain metastases (biopsy or clinical diagnosis)
- ✅ All 4 MRI sequences available: T1, T2, T1-Contrast, FLAIR
- ✅ Good image quality (no motion blur/ghosting)
- ✅ Signed informed consent
- ✅ Received stereotactic radiotherapy at BoCOC Cyprus (2019-2024)

**Primary cancers:**
- Non-small cell lung cancer: 65%
- Breast cancer: 32.5%
- Small cell lung cancer: 2.5%

---

### STEP 2: MRI Acquisition (Clinical Scanners)
**Equipment:**
- 3T scanners: 126 scans (68.5%) - higher resolution
- 1.5T scanners: 58 scans (31.5%) - standard clinical
- Manufacturers: Philips (68.5%), Siemens (29.9%), Toshiba (1.6%)

**Sequences acquired:**
1. **T1-weighted**: Shows brain anatomy (gray matter vs white matter)
2. **T2-weighted**: Shows edema (swelling) around tumors
3. **T1-Contrast**: GOLD STANDARD for tumor detection (tumors light up bright!)
   - Used Gadovist contrast agent (0.1 mmol/kg body weight)
4. **FLAIR**: Suppresses normal fluid, highlights abnormal fluid (edema)

**Typical parameters** (3T scanner):
- Slice thickness: 4mm (thin slices = better detail)
- Voxel size: ~0.5 × 0.5 × 4mm
- Repetition time (TR): 322-5530ms (depends on sequence)
- Echo time (TE): 2.5-186ms

---

### STEP 3: Data Preprocessing (Make It Standard!)

**Sub-step 3.1: Format Conversion**
- **Input**: DICOM files (hospital format, complex metadata)
- **Tool**: dcm2niix
- **Output**: NIfTI format (research standard, easier to work with)
- **Why**: NIfTI preserves spatial info, works with analysis tools

**Sub-step 3.2: BraTS Standardization**
- **Tool**: BraTS-Preprocessor
- **What it does**:
  1. Resample all images to **1mm³ isotropic voxels** (same in all directions)
  2. Register all sequences to common space (T1, T2, T1c, FLAIR aligned)
  3. Skull-strip (remove non-brain tissue)
  4. Normalize intensities (same brightness scale)
- **Output**: All scans in BraTS space (matches Yale preprocessing target!)

**Sub-step 3.3: Align CT and Radiotherapy Plans**
- **Tool**: Non-rigid registration
- **What it does**: Align CT + RTP to MRI T1 image
- **Why**: CT shows bone (for radiation planning), MRI shows soft tissue
- **Result**: All modalities in same coordinate system

---

### STEP 4: Tumor Segmentation (Find the Tumors!)

**Multi-Method Approach** (Use 3 methods, pick best):

**Method 1: DeepBraTumIA (Deep Learning)**
- **Type**: Automatic CNN-based segmentation
- **Trained on**: BraTS brain tumor data
- **Outputs**: 3 regions (enhancing tumor, edema, necrosis)
- **Limitation**: Sometimes misses small or multiple tumors

**Method 2: nnU-Net (Deep Learning)**
- **Type**: Self-configuring neural network (from Paper 2!)
- **Trained on**: BraTS 2020 dataset
- **Outputs**: 3 regions (enhancing tumor, edema, necrosis)
- **Limitation**: Same as DeepBraTumIA—sometimes misses lesions

**Method 3: Imalytics Preclinical (Semi-Automatic)**
- **Type**: Manual + automatic hybrid
- **Process**:
  1. Radiologist manually outlines tumor on central slices
  2. Software auto-detects tumor on remaining slices
  3. Radiologist manually segments edema + necrosis
- **Use case**: When Methods 1-2 missed tumors

**Final Step: Expert Verification**
- **Who**: Certified neuroradiologist (L.S.) with 10+ years experience
- **Tool**: ITK-SNAP 4.0.2 (visualization software)
- **What**: Manually reviewed and corrected ALL segmentations
- **Result**: High-quality, clinically reliable labels

**Segmentation Labels** (BraTS standard):
- Label 1: Necrotic core (dead tissue in tumor center)
- Label 2: Enhancing tumor (active tumor that lights up with contrast)
- Label 3: Edema (swelling around tumor)
- Label 10: Brain ventricles
- Label 30: White matter
- Label 40: Gray matter
- Label 50: Cerebrospinal fluid

---

### STEP 5: Radiomic Feature Extraction (Quantify Tumor Traits!)

**Pre-processing for radiomics:**
1. **N4 bias field correction** (fix brightness non-uniformity from scanner)
   - Tool: SimpleITK Python package
2. **Intensity clipping** (remove outlier pixel values)
   - Clip at 0.1 and 99.9 percentile
3. **Image rescaling** (normalize to 0-1024 range)
   - Tool: scikit-image

**Feature extraction:**
- **Tool**: Pyradiomics Python package (v3.1.0)
- **Total features per scan**: 110

**Feature categories:**
1. **First-order statistics (19 features)**
   - Mean, median, standard deviation, skewness, kurtosis
   - Example: "Is tumor uniformly bright or variable?"

2. **3D shape features (16 features)**
   - Volume, surface area, sphericity, compactness
   - Example: "Is tumor round like a ball or irregular like a starfish?"

3. **Texture features (75 features)**
   - Gray Level Co-occurrence Matrix (GLCM): 24 features
     - Measures spatial relationships between pixels
     - Example: "Are bright pixels next to bright pixels (smooth) or next to dark pixels (rough)?"
   - Gray Level Run Length Matrix (GLRLM): 16 features
     - Measures consecutive pixels with same intensity
   - Neighboring Gray Tone Difference Matrix (NGTDM): 5 features
   - Gray Level Size Zone Matrix (GLSZM): 16 features
   - Gray Level Dependence Matrix (GLDM): 14 features

**Why radiomics matter:**
- Capture tumor "personality" beyond size
- Two tumors same size can have different textures → different prognosis
- Machine learning can find patterns humans can't see

---

### STEP 6: Clinical Data Collection

**Source**: BoCOC electronic medical records

**Data collected:**
- **Demographics**: Age, gender, ethnicity
- **Cancer info**: Primary tumor type, location, diagnosis date
- **Treatment details**:
  - Chemotherapy drugs, doses, dates
  - Radiotherapy type (SRS/FSRT), fractions, doses
- **Functional status**:
  - Karnofsky performance score (0-100, higher = more functional)
  - Neurocognitive status (memory, thinking ability)
- **Outcome**: Death date or last follow-up (as of Dec 2024)
- **Per lesion**: Unique ID, brain location, detection date, treatment

**Anonymization:**
- Tool: Eclipse Radiotherapy Planning System built-in anonymizer
- Removed: Names, dates of birth, medical record numbers
- Process: On-site at research data server (never exposed protected info)

---

### STEP 7: Data Packaging & Sharing

**Format:**
- **Images**: DICOM (raw) + NIfTI (preprocessed)
- **Segmentations**: NIfTI (BraTS space)
- **Clinical data**: Excel spreadsheet
- **Radiomic data**: Excel spreadsheet
- **Repository**: Zenodo (DOI: 10.5281/zenodo.17253793)

**File structure** (Example: Patient 01):
```
P01/
├── BraTS/
│   ├── baseline/
│   │   ├── t1.nii.gz       (T1-weighted)
│   │   ├── t2.nii.gz       (T2-weighted)
│   │   ├── t1c.nii.gz      (T1-Contrast)
│   │   └── fla.nii.gz      (FLAIR)
│   ├── fu1/ (follow-up 1)
│   ├── fu2/ (follow-up 2)
│   └── ...
├── P01_brain_mask.nii.gz   (healthy brain regions)
├── P01_CT.nii.gz           (CT scan)
├── P01_RTP.nii.gz          (radiotherapy plan)
└── tumor_segmentation/
    ├── P01_tumor_mask_baseline.nii.gz
    ├── P01_tumor_mask_fu1.nii.gz
    └── ...
```

---

## 6. CODE/RESOURCES (Where to Find It!)

### **Dataset Download**
- **Repository**: Zenodo (permanent archive, DOI assigned)
- **URL**: https://doi.org/10.5281/zenodo.17253793
- **Cost**: FREE (open access)
- **License**: Open data (cite paper if you use)
- **Size**: ~100GB estimate (744 scans + segmentations + CT + RTP)

### **Radiomic Code (Open Source!)**
- **GitHub**: https://github.com/InSilicoModellingGroup/pyRadiomicBM
- **Language**: Python
- **Features**:
  - Reproduces paper's radiomic extraction
  - Comprehensive README
  - Well-documented code
  - Customizable for your own analysis

### **Software Used (All Free!)**

**Preprocessing:**
- dcm2niix - DICOM to NIfTI conversion
- BraTS-Preprocessor - Standardize to BraTS space
- SimpleITK (Python) - Bias field correction

**Segmentation:**
- DeepBraTumIA - Deep learning auto-segmentation
- nnU-Net - Self-configuring segmentation (Paper 2!)
- Imalytics Preclinical - Semi-automatic segmentation
- ITK-SNAP 4.0.2 - Manual correction/visualization

**Radiomic extraction:**
- Pyradiomics (Python v3.1.0) - Feature extraction
- scikit-image (Python) - Image preprocessing

**Visualization:**
- FSLeyes v1.12.6 - View NIfTI files
- ITK-SNAP 4.2.2 - 3D medical image viewer
- 3D Slicer v5.8.1 - Advanced visualization
- Horos v4.01 - DICOM viewer (Mac only)

### **Related Datasets (Comparisons)**
- **Yale dataset** (Paper 3): 1,430 patients, 11,884 scans - main dataset
- **BraTS-METS 2023**: Pre-treatment only, limited subregions
- **BrainMetShare**: Another public dataset
- **Paper 28 (referenced)**: Spanish BM dataset

---

## 7. HOW DOES THIS CONNECT TO YALE DATASET?

### **Complementary Roles: Yale = Training, Cyprus = Validation**

**Strategy 1: Train on Yale (Large), Validate on Cyprus (Detailed)**
```
Yale (1,430 patients) → Train AI models
Cyprus (40 patients) → Test generalization

Why: Large dataset needed for deep learning training
      Small diverse dataset tests if model works on new population
```

**Strategy 2: Combine Datasets for More Diversity**
```
Yale (USA, 20 years, 1,430 patients)
    +
Cyprus (Mediterranean, 5 years, 40 patients)
    =
More robust AI (works across populations & time periods)
```

**Strategy 3: Use Cyprus for Detailed Analysis Yale Can't Do**

| Analysis Type | Yale | Cyprus | Winner |
|---------------|------|--------|--------|
| **Large-scale training** | ✅ 11,884 scans | ❌ 744 scans | Yale |
| **Tumor subregion analysis** | ❌ No labels | ✅ 3 subregions labeled | Cyprus |
| **Radiomic features** | ❌ Must extract | ✅ Pre-extracted | Cyprus |
| **Treatment response** | ❌ Limited details | ✅ RT plans + outcomes | Cyprus |
| **Long-term survival** | ✅ 20 years | ❌ 5 years | Yale |
| **Geographic diversity** | USA only | Cyprus only | **Combine!** |

### **Practical Workflow:**

**PHASE 1: Preprocessing (Both Datasets)**
```
Yale raw scans → BraTS Preprocessor → Standardized
Cyprus raw scans → BraTS Preprocessor → Standardized
✅ BOTH in same format now! Can combine!
```

**PHASE 2: Training (Primarily Yale)**
```
Train ViT model on Yale (11,884 scans)
Why: Need large dataset for deep learning
```

**PHASE 3: Validation (Cyprus as Test Set)**
```
Test Yale-trained model on Cyprus (744 scans)
Measure: Does model work on Mediterranean population?
Measure: Does model detect 3 tumor subregions accurately?
```

**PHASE 4: Tumor Subregion Analysis (Cyprus)**
```
Cyprus has labels for: enhancing tumor, edema, necrosis
Analyze separately: Which subregion predicts treatment response?
Apply insights back to Yale dataset
```

**PHASE 5: Radiomic-Enhanced Predictions (Cyprus)**
```
Cyprus provides 110 radiomic features per scan
Test: Do radiomic features improve predictions?
If yes → Extract same features from Yale dataset
```

### **Specific Yale-Cyprus Synergies:**

**1. Scanner Harmonization Testing**
- Cyprus: 3T (68.5%) + 1.5T (31.5%)
- Yale: 20 years of scanner evolution
- **Test**: Does ComBat harmonization (Paper 9) work on BOTH datasets?

**2. Segmentation Validation**
- Yale: No tumor labels (must generate with nnU-Net)
- Cyprus: Expert-verified labels
- **Validation**: Run nnU-Net on Cyprus → compare to ground truth labels → measure accuracy
- **Result**: Know how accurate Yale's generated labels are!

**3. Treatment Response Prediction**
- Cyprus: Has radiotherapy plans + follow-up scans showing response
- Yale: Has treatment info but less detail
- **Strategy**: Train response predictor on Cyprus (small but detailed) → Apply to Yale (large scale)

**4. Small vs Large Tumor Handling**
- Cyprus: Excluded tumors < 3mm
- Yale: Likely includes small tumors
- **Research question**: Does model trained on larger tumors (Cyprus) detect small ones (Yale)?

---

## 8. HOW DOES THIS CONNECT TO OUR 5 OBJECTIVES?

### **OBJECTIVE 1: Automatic Tumor Tracking**
**Cyprus Role**: ✅ **VALIDATION DATASET**
- Yale-trained tracker → Test on Cyprus 40 patients
- Cyprus has GROUND TRUTH labels → Measure accuracy precisely
- **Advantage**: Expert-verified segmentations = trust the evaluation

**Specific use:**
```
ViT trained on Yale → Predict tumor on Cyprus baseline
Compare to Cyprus expert segmentation → Calculate Dice score
If Dice > 0.85 → Model works across populations ✅
```

### **OBJECTIVE 2: Predict Future Tumor Changes**
**Cyprus Role**: ✅ **TREATMENT RESPONSE VALIDATION**
- Cyprus has MORE follow-ups per patient (18.6 vs 8.3)
- Can test: "After seeing Months 0-3, predict Month 6"
- Cyprus has radiotherapy plans → Can factor in "patient got radiation at Month 1"

**Example:**
```
Input: Baseline + Month 1.5 + Month 3 + RT plan
Predict: Month 6 tumor size
Ground truth: Cyprus Month 6 scan
Error: |Predicted - Actual| < 5mm = good prediction
```

**Advantage over Yale**: More frequent follow-ups = better temporal resolution for prediction

### **OBJECTIVE 3: LLM-Generated Radiology Reports**
**Cyprus Role**: ✅ **CLINICAL CONTEXT TRAINING DATA**
- Cyprus provides actual treatment details (chemotherapy drugs, radiation doses)
- LLM can learn: "Patient received 25 Gy radiation → Tumor shrank 40% in 3 months"
- **Template reports** based on Cyprus data

**Example LLM input:**
```
Patient: 55F, breast cancer primary
Baseline: 2 metastases (frontal lobe 12mm, parietal lobe 8mm)
Treatment: SRS 20 Gy single fraction
Follow-up (3 months): Frontal reduced to 7mm, parietal stable
Radiomic change: Sphericity increased (less irregular)

LLM generates: "Partial response to stereotactic radiosurgery observed 
in dominant frontal lesion with 42% volume reduction. Parietal lesion 
demonstrates stable disease. Morphologic changes suggest treatment 
effect with decreased irregularity."
```

**Advantage**: Cyprus clinical data provides training examples for LLM to learn medical reasoning

### **OBJECTIVE 4: Video Generation (Treatment Scenarios)**
**Cyprus Role**: ✅ **FINE-GRAINED TEMPORAL DATA**
- More timepoints (18.6 avg) = smoother video interpolation
- Can generate: "Show me what would happen if patient got NO treatment vs WITH treatment"
- Radiotherapy plans provide "intervention" variable

**Video scenario 1: Treatment response**
```
Baseline → 6 weeks → 3 months → 6 months
Show tumor shrinking after radiation
Cyprus data trains model on ACTUAL treatment responses
```

**Video scenario 2: Counterfactual**
```
"What if patient didn't get radiation?"
Cyprus provides radiotherapy plans + outcomes
Model learns: "Radiation reduces tumor 30-50% typically"
Generate video: Alternative timeline without treatment (tumor grows)
```

**Advantage**: Radiotherapy plans = causal information (treatment → response) for better counterfactuals

### **OBJECTIVE 5: Clinical Decision Support**
**Cyprus Role**: ✅ **TREATMENT OUTCOME VALIDATION**
- Cyprus has EXACT radiation doses + tumor responses
- Can train: "Given tumor size X and location Y, will patient respond to Z Gy radiation?"
- Has survival data → can predict: "Patient with these tumor features lived N months"

**Clinical prediction examples:**

**1. Treatment Selection**
```
Input: Tumor 15mm, frontal lobe, spherical, homogeneous
Cyprus data: 80% of similar tumors responded to SRS
Recommendation: "SRS is appropriate first-line treatment"
```

**2. Dose Optimization**
```
Input: 3 metastases, largest 22mm
Cyprus data: Lesions >20mm needed 25 Gy (not 20 Gy) for control
Recommendation: "Consider 25 Gy for largest lesion"
```

**3. Follow-up Planning**
```
Input: Post-treatment, tumor reduced 40% at 3 months
Cyprus data: Similar responses stabilized by 6 months
Recommendation: "Next MRI at 6 months, then annual if stable"
```

**Advantage**: Real treatment outcomes data grounds AI recommendations in clinical reality

---

## 9. HOW DOES CYPRUS DATASET COMBINE WITH OTHER PAPERS?

### **With Paper 1 (BraTS Toolkit)**
**Already Done!** Cyprus used BraTS-Preprocessor
```
Cyprus raw scans → BraTS-Preprocessor → Standardized format
✅ Cyprus data is BraTS-compliant out of the box!
```
**Advantage**: Can combine Cyprus + Yale seamlessly (both BraTS format)

### **With Paper 2 (nnU-Net)**
**Cyprus Used nnU-Net for Segmentation!**
```
Cyprus applied: DeepBraTumIA + nnU-Net → Initial segmentations
Then: Expert radiologist verified/corrected
✅ Cyprus validates nnU-Net accuracy!
```
**Key finding**: nnU-Net sometimes MISSED lesions (why semi-automatic method needed)
**Lesson for Yale**: Will need quality checking after nnU-Net, can't trust 100%

### **With Paper 3 (Yale Dataset) - MAIN COMPARISON**
**Complementary Datasets!**

| Use Case | Yale | Cyprus | Best Approach |
|----------|------|--------|---------------|
| **Training large models** | ✅ 1,430 patients | ❌ 40 patients | Use Yale |
| **Validating models** | ❌ No ground truth labels | ✅ Expert labels | Use Cyprus |
| **Treatment response** | Limited detail | ✅ RT plans + outcomes | Use Cyprus |
| **Long-term survival** | ✅ 20 years | 5 years | Use Yale |
| **Subregion analysis** | No labels | ✅ 3 subregions | Use Cyprus |
| **Geographic diversity** | USA | Mediterranean | **Combine!** |

**Combined workflow:**
1. Train on Yale (large scale)
2. Validate on Cyprus (ground truth)
3. Combine for diversity (USA + Mediterranean)

### **With Papers 4 & 5 (Registration)**
**Cyprus Needs Registration Too!**
```
Cyprus: 40 patients × 18.6 scans = 744 registrations needed
Apply: FLIRE (fast) or Paper 4 method (tumor-preserving)
```
**Smaller dataset = Perfect for testing registration methods!**
- Test FLIRE on Cyprus first (744 scans faster than Yale's 11,884)
- If works well → apply to Yale
- If problems → debug on smaller dataset first

### **With Paper 9 (ComBat - NEXT PAPER)**
**Cyprus Harmonization Simpler Than Yale**
```
Cyprus: 3T + 1.5T, 3 manufacturers, 5 years
Yale: 20 years of scanner evolution
```
**Strategy**: Test ComBat on Cyprus first (easier), then Yale (harder)

### **With Future ViT Papers (Phase 2)**
**Cyprus = Ideal ViT Validation Set**
```
ViT trained on Yale (11,884 scans)
↓
Validate on Cyprus (744 scans, expert labels)
↓
Measure: Dice score, sensitivity, specificity
```
**Why Cyprus good for validation:**
- Ground truth labels (know correct answer)
- Different population (tests generalization)
- Smaller (faster to validate)

### **With Future LLM Papers (Phase 3)**
**Cyprus Provides Training Examples**
```
LLM needs: "Given tumor features + treatment → Generate report"
Cyprus provides: Actual treatment details + outcomes
```
**Example training data:**
```
Input: [MRI features] + [20 Gy SRS] + [3-month follow-up showing shrinkage]
Output: "Excellent response to stereotactic radiosurgery with 50% volume 
        reduction at first follow-up. Treatment effect confirmed by 
        decreased enhancement and increased necrosis."
```

### **With Future Diffusion Papers (Phase 4)**
**Cyprus = High-Quality Video Training Data**
```
18.6 timepoints/patient = Smooth temporal sequences
Perfect for training video diffusion models!
```
**Advantage**: More frames per patient = better video quality

---

## 10. CITATION EXAMPLE

**APA Format:**
```
Flouri, D., Papanikas, C.-P., Manikis, G. C., Ioannou, E., Karaoli, G., 
Papageorgiou, E., Constantinidou, A., Siakallis, L., Theodorou, M., & 
Vavourakis, V. (2025). A longitudinal MRI dataset of brain metastases 
with tumor segmentations, clinical & radiomic data. Scientific Data, 
12(1), 1828. https://doi.org/10.1038/s41597-025-06131-0
```

**Quick Reference:**
- **Authors**: Flouri et al.
- **Institution**: University of Cyprus + Bank of Cyprus Oncology Centre
- **Journal**: Scientific Data (Nature Publishing Group)
- **Year**: 2025 (VERY NEW!)
- **Dataset name**: PROTEAS (not explicitly stated but inferred from files)
- **DOI**: 10.1038/s41597-025-06131-0
- **Data DOI**: 10.5281/zenodo.17253793

**When to Cite This Paper:**
- Discussing dataset comparison (Yale vs Cyprus)
- Explaining validation strategy
- Reporting tumor subregion analysis
- Using radiomic features
- Discussing treatment response prediction
- Justifying multi-dataset approach

---

## 🎯 BOTTOM LINE FOR OUR RESEARCH

**Cyprus Dataset = VALIDATION + DETAILED ANALYSIS (not replacement for Yale!)**

### **Role in Our Project:**

✅ **PRIMARY USE: Validation set for Yale-trained models**
- Train on Yale (1,430 patients) → Test on Cyprus (40 patients)
- Cyprus has ground truth labels → Know true accuracy
- Different population → Tests generalization

✅ **SECONDARY USE: Detailed subregion analysis**
- Cyprus has 3 tumor subregions labeled (enhancing, edema, necrosis)
- Yale doesn't provide this
- Learn from Cyprus which subregions predict treatment response
- Apply insights to Yale

✅ **TERTIARY USE: Combined training for diversity**
- Combine Yale (11,884 scans) + Cyprus (744 scans) = 12,628 total
- USA + Mediterranean populations
- More robust AI (works across demographics)

### **When to Use Cyprus vs Yale:**

| Task | Dataset Choice | Reasoning |
|------|----------------|-----------|
| **Train Vision Transformer** | Yale | Need 11,884 scans (large dataset) |
| **Validate ViT accuracy** | Cyprus | Has ground truth expert labels |
| **Subregion analysis** | Cyprus | Only Cyprus has 3-part labels |
| **Radiomic features** | Cyprus | Pre-extracted, saves time |
| **Treatment response** | Cyprus | Has RT plans + detailed outcomes |
| **Long-term survival** | Yale | 20 years vs 5 years |
| **Large-scale study** | Yale | 1,430 patients vs 40 |
| **Method testing** | Cyprus | Smaller = faster iteration |

### **Immediate Actions:**

**NOW (Phase 1):**
1. ❌ Don't download Cyprus yet (focus on Yale first)
2. ✅ Remember Cyprus exists for validation later
3. ✅ Plan validation strategy (test Yale models on Cyprus)

**LATER (Phase 2-3):**
1. Download Cyprus dataset (~100GB)
2. Run Yale-trained ViT on Cyprus
3. Measure accuracy against expert labels
4. If poor → debug model
5. If good → publish! ("Model generalizes across populations")

**PHASE 4-5:**
1. Extract same radiomic features from Yale as Cyprus provides
2. Test if radiomic features improve predictions
3. Use Cyprus treatment data to train clinical decision support

### **Key Insight:**
"Cyprus is NOT a competitor to Yale—it's a PARTNER! Yale gives you scale (1,430 patients for training), Cyprus gives you validation (expert labels) and detail (3 tumor subregions, radiomic features, RT plans). Use BOTH together for better research!" 🤝

**Current Status After Paper 7:**
- ✅ 6 papers analyzed (BraTS, nnU-Net, Yale, Registration, FLIRE, Cyprus)
- ✅ Phase 1: 83% complete (need Paper 9 - ComBat)
- ✅ TWO longitudinal brain metastases datasets identified (Yale + Cyprus)
- ✅ Clear validation strategy: Train on Yale, test on Cyprus
