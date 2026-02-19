# 📊 DATASET OVERVIEW — Yale + Cyprus Brain Metastases MRI

> **Purpose**: Understand what data we have, where to get it, and how to use both datasets together

---

## 🎯 Quick Comparison

| Feature | **Yale Dataset** | **Cyprus Dataset (PROTEAS)** |
|---------|-----------------|------------------------------|
| **Role in project** | **PRIMARY TRAINING DATA** | **VALIDATION DATA** |
| **Patients** | 1,430 | 40 |
| **Total scans** | 11,884 | 744 |
| **Scans per patient** | 8.3 average | 18.6 average (2× more!) |
| **Time span** | 2004-2023 (20 years) | 2019-2024 (5 years) |
| **Tumor labels** | ❌ NO (we must generate) | ✅ YES (expert-verified 3-subregion) |
| **MRI sequences** | T1, T1c, T2, FLAIR | T1, T1c, T2, FLAIR |
| **Scanner types** | 3T (41%), 1.5T (58%) | 3T (68.5%), 1.5T (31.5%) |
| **Manufacturers** | Siemens (86%), GE (13%) | Philips (68.5%), Siemens (29.9%), Toshiba (1.6%) |
| **Clinical metadata** | ✅ Age, sex, treatment dates | ✅ Age, sex, treatment, survival, neurocognitive status |
| **Radiomic features** | ❌ NO (we extract) | ✅ YES (110 features pre-extracted!) |
| **Radiotherapy plans** | ❌ NO | ✅ YES (45 RTP files with CT scans) |
| **Cost** | **FREE** | **FREE** |
| **Download link** | [TCIA](https://www.cancerimagingarchive.net/) (DOI: 10.7937/3YAT-E768) | [Zenodo](https://doi.org/10.5281/zenodo.17253793) |

---

## 📦 Dataset #1: Yale Longitudinal Brain Metastases (Main Training)

### What You Get
- **11,884 MRI studies** from **1,430 patients** (2004-2023)
- **4 MRI sequences**: T1-weighted, T1-contrast-enhanced (T1c), T2-weighted, FLAIR
- **Longitudinal**: Average 8.3 scans per patient tracked over months/years
- **Treatment stages**: 15% pre-treatment baseline, 85% post-treatment follow-ups
- **Clinical metadata**: Age (median 64), sex (57% female), treatment dates

### Patient Demographics
- **Age**: Median 64 years (IQR: 56-71)
- **Sex**: 57% female, 43% male
- **Ethnic diversity**: 93% from Yale (USA), 7% from external sites

### Technical Specs
- **Scanners**: 
  - Field strength: 41% at 3T, 58% at 1.5T
  - Vendors: Siemens (86%), GE (13%), other (1%)
- **Sequences available** (% of studies):
  - T1W: 71% of studies
  - T1CE (contrast): 76%
  - T2W: 62%
  - FLAIR: 76%
- **Format**: NIfTI (`.nii.gz`), skull-stripped with HD-BET
- **Resolution**: Variable (real-world clinical data — need resampling to 1mm³ isotropic)

### What's MISSING (Important!)
❌ **No tumor segmentation masks** — you must label tumors yourself using nnU-Net  
❌ **No tumor locations** — no bounding boxes or coordinates  
❌ **No survival data** — time-to-death not provided  
❌ **No treatment response labels** — "did treatment work?" not annotated  
❌ **Variable quality** — 20 years = old 1.5T scanners (2004) mixed with modern 3T (2023)

### How to Download
1. Go to [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
2. Search for **"Yale Brain Metastases"** or use DOI: `10.7937/3YAT-E768`
3. Download using TCIA's download manager (free account required)
4. Expected size: ~1-2 TB (11,884 scans)

### Preprocessing Pipeline (from BraTS Toolkit paper)
```bash
# 1. Install BraTS Toolkit
pip install BraTS-Toolkit

# 2. Run standardized preprocessing
brats_preprocessor \
  --input_dir /path/to/yale_dicoms \
  --output_dir /path/to/preprocessed \
  --modalities t1 t1ce t2 flair \
  --skull_strip HD-BET \
  --resample 1mm

# Output: All scans resampled to 1mm³, co-registered (T1/T1c/T2/FLAIR aligned within each visit), skull-stripped
```

### Why Yale is Our PRIMARY Dataset
✅ **Scale**: 1,430 patients = enough to train deep learning without overfitting  
✅ **Longitudinal**: 8.3 scans per patient = can learn temporal patterns  
✅ **Real-world**: Clinical data (not research-controlled) = robust models  
✅ **Treatment diversity**: Pre/post surgery, radiation, chemo — covers all scenarios  
✅ **Public & free**: No IRB, no data use agreements, instant access

---

## 📦 Dataset #2: Cyprus PROTEAS (Validation)

### What You Get
- **744 MRI studies** from **40 patients** (2019-2024)
- **65 brain metastases** with **expert-verified 3-subregion segmentations**:
  - Enhancing tumor (active cancer)
  - Necrotic core (dead tissue)
  - Peritumoral edema (swelling around tumor)
- **Longitudinal**: Average 18.6 scans per patient (2× more than Yale!)
- **Follow-up schedule**: 6 weeks, 3 months, 6 months, 9 months, 12 months post-radiotherapy
- **BONUS**: 45 radiotherapy treatment plans (RTP) + CT scans + 110 radiomic features per scan

### Patient Demographics
- **Age**: Not specified in dataset (check metadata file)
- **Primary cancer origins**: 
  - Lung: 67.5% (27 patients)
  - Breast: 32.5% (13 patients)
- **Treatment**: All received stereotactic radiosurgery (SRS) or fractionated stereotactic radiotherapy (FSRT)

### Technical Specs
- **Scanners**:
  - Field strength: 68.5% at 3T, 31.5% at 1.5T
  - Vendors: Philips (68.5%), Siemens (29.9%), Toshiba (1.6%)
- **Sequences**: T1, T1c, T2, FLAIR (all 4 available for all scans)
- **Format**: NIfTI (`.nii.gz`), BraTS-standardized
- **Resolution**: 1mm³ isotropic (already resampled!)

### What's INCLUDED (Cyprus Advantage!)
✅ **Expert tumor segmentations** — 3 subregions manually verified by neuroradiologist (10+ years experience)  
✅ **Radiomic features** — 110 features pre-extracted (19 first-order, 16 shape, 75 texture)  
✅ **Clinical data** — age, sex, Karnofsky performance status, neurocognitive assessments  
✅ **Survival data** — time to death or last follow-up  
✅ **Radiotherapy plans** — 45 RTP files showing exact radiation dose distribution  
✅ **CT scans** — for radiation planning (45 CT studies)  
✅ **Already preprocessed** — skull-stripped, co-registered, BraTS-format

### How to Download
1. Go to [Zenodo](https://doi.org/10.5281/zenodo.17253793)
2. Click **"Download"** (no account needed)
3. Expected size: ~100 GB (744 MRI + 45 CT + segmentations + metadata)
4. Files included:
   - `MRI_scans/` — NIfTI files organized by patient and timepoint
   - `Segmentations/` — Tumor masks (3 labels)
   - `Radiotherapy/` — RTP files and CT scans
   - `Clinical_data.xlsx` — Patient demographics, treatment, survival
   - `Radiomic_features.xlsx` — 110 features per scan (ready to use!)

### File Structure (Cyprus)
```
PROTEAS/
├── MRI_scans/
│   ├── Patient_001/
│   │   ├── Timepoint_0_baseline/
│   │   │   ├── T1.nii.gz
│   │   │   ├── T1c.nii.gz
│   │   │   ├── T2.nii.gz
│   │   │   └── FLAIR.nii.gz
│   │   ├── Timepoint_1_6weeks/
│   │   └── ...
│   └── Patient_040/
├── Segmentations/
│   ├── Patient_001/
│   │   ├── Timepoint_0_seg.nii.gz  # 3 labels: 1=necrosis, 2=edema, 4=enhancing
│   └── ...
├── Radiotherapy/
│   ├── Patient_001_RTP.dcm
│   ├── Patient_001_CT.nii.gz
│   └── ...
├── Clinical_data.xlsx
└── Radiomic_features.xlsx
```

### Why Cyprus is Our VALIDATION Dataset
✅ **Ground truth labels** — Compare our nnU-Net segmentations (on Yale) to expert labels (on Cyprus)  
✅ **External validation** — Different hospital, different population → tests generalizability  
✅ **Different scanners** — Philips-dominated (Cyprus) vs Siemens-dominated (Yale) → tests harmonization  
✅ **Higher temporal density** — 18.6 scans per patient → better test for video generation continuity  
✅ **Treatment response** — All post-radiotherapy → validate counterfactual generation (treatment effects)

---

## 🤔 Can We Really Use Cyprus as Validation?

### ✅ YES — Here's Why:

| Validation Criterion | Cyprus Dataset | Status |
|---------------------|----------------|--------|
| **External data** (not from Yale) | Different hospital, country, population | ✅ PASS |
| **Same task** (brain metastases longitudinal MRI) | Same disease, same imaging | ✅ PASS |
| **Same modalities** (T1/T1c/T2/FLAIR) | All 4 sequences present | ✅ PASS |
| **Sufficient size** | 40 patients, 744 scans, 65 tumors | ✅ PASS (for validation) |
| **Ground truth** | Expert-verified 3-subregion segmentations | ✅ PASS |
| **Never seen in training** | Completely separate from Yale | ✅ PASS |

### 📊 Statistical Power Analysis

**For segmentation validation** (comparing our nnU-Net to expert labels):
- Sample size: 65 tumors with 3-subregion labels
- Metrics: Dice score (overlap), Hausdorff distance (boundary error)
- **Verdict**: ✅ Sufficient for statistically significant Dice comparison (need >30 samples, we have 65)

**For video generation validation** (temporal consistency):
- Sample size: 40 patients × avg 18.6 scans = 744 timepoint pairs
- Metrics: Frame-to-frame SSIM, temporal coherence
- **Verdict**: ✅ Sufficient for visual quality assessment

**For treatment effect modeling**:
- All 40 patients received radiotherapy
- Can test: "Does our model correctly predict tumor shrinkage after radiation?"
- **Verdict**: ✅ Valid for treatment-conditioned generation validation

### ⚠️ Limitations as Validation Set

❌ **Small patient count** (40 vs Yale's 1,430) — can't train on it alone  
❌ **Only 2 primary cancer types** (lung, breast) — Yale likely has more diversity  
❌ **Only radiotherapy patients** — Yale includes surgery, chemo (broader treatment validation)  
❌ **Shorter time span** (2019-2024) — Yale's 20-year span tests robustness better

### 💡 Best Use Strategy

**DO use Cyprus for**:
✅ Validating nnU-Net segmentation accuracy (Dice score on 65 expert-labeled tumors)  
✅ Testing harmonization success (different scanner vendors than Yale)  
✅ Validating temporal video consistency (18.6 scans per patient = dense sequences)  
✅ Testing treatment effect modeling (all post-radiotherapy)  
✅ Confirming LLM explanations match clinical data (rich metadata available)

**DON'T use Cyprus for**:
❌ Primary training (too small — would overfit)  
❌ Hyperparameter tuning (use Yale's validation split instead)  
❌ Pre-training ViT (insufficient scale)

---

## 🔄 Combined Usage Strategy (Yale + Cyprus)

### Phase 1 (Weeks 1-4): Preprocessing

**Yale**:
1. Download all 11,884 scans from TCIA
2. Apply BraTS Toolkit → skull-strip, resample to 1mm³, co-register T1/T1c/T2/FLAIR
3. Run nnU-Net (BraTS-pretrained) → generate 3-class tumor segmentations
4. **Validate**: NO ground truth available — visually inspect 50 random cases

**Cyprus**:
1. Download 744 scans from Zenodo (already preprocessed!)
2. **Validate our pipeline**: Run same nnU-Net on Cyprus scans
3. Compare: Our segmentations vs. expert labels → compute Dice scores
4. **Target**: Dice > 0.85 on all 3 subregions (enhancing, necrosis, edema)
5. **If Dice < 0.85**: Fine-tune nnU-Net on Cyprus, then re-run on Yale

### Phase 2-3 (Weeks 5-11): ViT Training

**Yale**:
1. Train Swin UNETR encoder on 1,430 patients (11,884 scans)
2. Extract 768-dim embeddings per scan
3. Apply TaViT time-distance encoding
4. Harmonize using Nested ComBat + Longitudinal ComBat

**Cyprus**:
1. Extract embeddings using trained Swin UNETR (inference only)
2. **Do NOT harmonize Cyprus with Yale** — keep separate for external validation
3. Use Cyprus embeddings to test: "Do embeddings cluster by tumor type (lung vs breast)?"

### Phase 4 (Weeks 12-14): LLM Integration

**Yale**:
1. Train RadFM Perceiver + MedLLaMA on Yale clinical narratives
2. Generate explanations: "Patient X, tumor grew 15% over 6 months..."

**Cyprus**:
1. Generate explanations for all 40 Cyprus patients
2. **Validate**: Compare LLM outputs to clinical data Excel file
3. Check: Does LLM correctly describe treatment response? Tumor size changes?

### Phase 5 (Weeks 15-18): Video Generation

**Yale**:
1. Train TaDiff (treatment-aware diffusion) on 1,430 patients
2. Train EchoNet-Synthetic video pipeline

**Cyprus**:
1. **Test generalization**: Generate progression videos for 40 Cyprus patients
2. **Validate**:
   - Visual quality: Do generated scans look realistic?
   - Temporal consistency: Do frames flow smoothly?
   - Treatment effect: Does radiation → tumor shrinkage match real follow-ups?
3. **Quantitative**: SSIM, PSNR between generated frames and real Cyprus follow-ups
4. **Qualitative**: Show videos to radiologist — "Does this look clinically plausible?"

### Phase 6 (Weeks 19-20): Final Evaluation

**Segmentation validation**:
- Report: Dice scores on Cyprus (nnU-Net vs. expert)
- Compare: Yale automated labels (no ground truth) vs. Cyprus (verified ground truth)

**Video validation**:
- Quantitative: SSIM, PSNR, FID on Cyprus held-out sequences
- Qualitative: Clinician review of 10 Cyprus progression videos

**Harmonization validation**:
- Before ComBat: Features cluster by scanner → plot PCA, color by vendor
- After ComBat: Features cluster by tumor type → no scanner fingerprint visible

---

## 📥 Download Commands (Quick Start)

### Yale Dataset (TCIA)
```bash
# 1. Install TCIA Download Manager
pip install tcia-utils

# 2. Download Yale dataset
from tcia_utils import nbia

# Get all series for Yale collection
series = nbia.getSeries(collection="Yale_Brain_Metastases")

# Download (this will take hours — ~1-2 TB)
nbia.downloadSeries(series, input_type="list", path="/data/yale")
```

### Cyprus Dataset (Zenodo)
```bash
# Direct download (100 GB)
wget https://zenodo.org/record/17253793/files/PROTEAS_Dataset.zip

# Unzip
unzip PROTEAS_Dataset.zip -d /data/cyprus
```

---

## 🔍 Quality Checks Before Starting

### For Yale:
- [ ] All 11,884 scans downloaded?
- [ ] Each scan has at least 3 of 4 modalities (T1, T1c, T2, FLAIR)?
- [ ] Skull-stripping successful? (check 10 random scans visually)
- [ ] NIfTI format readable by MONAI/ITK?

### For Cyprus:
- [ ] All 744 scans downloaded?
- [ ] Segmentation masks present for all 65 tumors?
- [ ] Clinical_data.xlsx and Radiomic_features.xlsx present?
- [ ] Can read NIfTI files without errors?

---

## 📚 Citation Requirements

**If you use Yale dataset**:
```
Ramakrishnan et al. (2025). "Yale Longitudinal Brain Metastases Dataset."
DOI: 10.7937/3YAT-E768
```

**If you use Cyprus dataset**:
```
Trimithiotis et al. (2025). "PROTEAS: A Public Longitudinal Brain Metastases Dataset."
DOI: 10.5281/zenodo.17253793
```

---

## ❓ FAQ

**Q: Why not combine Yale + Cyprus into one training set?**  
A: Cyprus is too small (40 patients). Adding it to Yale (1,430 patients) would barely change the model but would invalidate external validation. Keep Cyprus separate as held-out test set.

**Q: Can I train ONLY on Cyprus?**  
A: No — 40 patients is too small for deep learning. You'd overfit heavily. Always use Yale for training.

**Q: Cyprus has radiomic features — should I use them?**  
A: For our project (ViT + LLM + video generation), we extract ViT embeddings instead. Cyprus radiomics are useful if you want to compare: "Do hand-crafted radiomics or ViT embeddings predict treatment response better?" (side experiment).

**Q: What if Dice score on Cyprus is low (<0.80)?**  
A: Fine-tune nnU-Net on Cyprus expert labels, then re-run on all Yale scans. This improves Yale's automated labels.

**Q: Do I need IRB approval?**  
A: No — both datasets are publicly released with de-identified data. No IRB needed.

---

*Last updated: Feb 14, 2026 | See `PIPELINE_FINAL.md` for full methodology*
