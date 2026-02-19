# Paper 3: Yale Longitudinal Brain Metastases Dataset (2025)

## 1️⃣ ONE-SENTENCE SUMMARY
Largest public dataset of 11,884 brain MRI scans from 1,430 brain metastasis patients tracked over time.

---

## 2️⃣ KEY RESULTS

**Dataset Size**:
- **11,884 MRI studies** total
- **1,430 unique patients**
- **Average: 8.3 scans per patient** (11,884 ÷ 1,430)
- **Time span**: 2004-2023 (nearly 20 years of data)

**Patient Demographics**:
- Age: Median 64 years (range 56-71)
- Sex: 57% female, 43% male
- Treatment status: 15% pre-treatment, 85% follow-up scans

**Imaging Details**:
- **4 MRI sequence types**: T1W (71%), T1CE (76%), T2W (62%), FLAIR (76%)
- **Scanners**: 41% at 3T, 58% at 1.5T
- **Vendors**: Siemens (86%), GE (13%)
- **Public access**: Available on TCIA (The Cancer Imaging Archive)

**Temporal Coverage**:
- Most patients have 3-10 follow-up scans
- Follow-up intervals: Varied (real-world clinical data)
- Covers pre-treatment and post-treatment (surgery, radiation, chemo)

---

## 3️⃣ WHAT'S NEW

**This is THE FIRST**:
1. **Largest public brain metastasis dataset** ever released
2. **Only longitudinal dataset** with multiple timepoints per patient
3. **Real-world clinical data** (not just research scans)
4. **Public and free** - Available to anyone on TCIA
5. **Includes clinical metadata** (age, sex, treatment dates)

**Why it's special**:
- Previous datasets: Single timepoints, small samples (<100 patients)
- Yale: 1,430 patients × 8 scans each = Can train AI to understand tumor changes over time

---

## 4️⃣ LIMITATIONS

**What's missing**:
1. ❌ **No tumor segmentation masks** - Images only, no labeled tumors
2. ❌ **No tumor locations** - You have to find tumors yourself
3. ❌ **No survival data** - Don't know how long patients lived
4. ❌ **No treatment response labels** - Don't know if treatment worked
5. ❌ **Variable quality** - 20 years of data = old scanners mixed with new

**Technical limitations**:
- Different scanners = Images look slightly different
- Irregular follow-ups = Not every 6 weeks, varies by patient
- Missing sequences = Not all scans have all 4 image types
- Only 7% from outside Yale = Mostly one hospital's data

**What you need to do**:
- Label tumors yourself (or use nnU-Net to segment them)
- Handle scanner differences (need harmonization - ComBat papers)
- Create your own treatment response labels

---

## 5️⃣ METHODS (How they created the dataset)

**Step 1: Find patients (2004-2023)**
- Started with 46,364 scans from 7,111 patients
- Manually checked medical records
- Kept only confirmed brain metastasis patients
- Final: 1,430 patients

**Step 2: Select imaging**
- Only kept 4 essential MRI types: T1W, T1CE, T2, FLAIR
- Removed incomplete scans
- For treated patients: included pre-treatment + all follow-ups
- Final: 11,884 studies

**Step 3: Clean and anonymize**
- Removed patient names, dates, faces
- Used HD-BET to remove skull (protect privacy + focus on brain)
- Standardized sequence names (different scanners label things differently)
- Converted to NIfTI format (standard AI format)

**Step 4: Quality check**
- Manually reviewed 200 random scans
- Verified brain extraction worked correctly
- Checked acquisition parameters (slice thickness, etc.)

**Step 5: Upload to TCIA**
- Made publicly available (free download)
- Included Excel file with patient info and scan parameters

---

## 6️⃣ CODE & RESOURCES

**Code available**: ✅ Upon request (brain extraction, sequence standardization)

**Dataset access**:
- ✅ **FREE and PUBLIC**
- Location: The Cancer Imaging Archive (TCIA)
- Link: https://www.cancerimagingarchive.net/
- DOI: 10.7937/3YAT-E768
- Format: NIfTI files (standard for AI)

**Can reproduce**: ✅ Yes! Everything is publicly available

**Tools used**:
- HD-BET: Brain extraction (removes skull)
- Visage PACS: Medical image management system
- NIfTI format: Standard medical imaging format

---

## 7️⃣ CONNECTION TO YALE DATASET

**THIS IS YOUR MAIN DATASET!**

This paper describes the exact dataset you'll use for your entire project!

**What you get**:
- ✅ 1,430 patients with longitudinal scans (perfect for temporal ViT)
- ✅ Average 8 timepoints per patient (enough to train video generation)
- ✅ Pre and post-treatment scans (can model treatment effects)
- ✅ Clinical metadata (age, sex, treatment dates for LLM explanations)
- ✅ Free download (no approval needed)

**How you'll use it**:

### For Preprocessing (Objective 1):
- Download all 11,884 scans
- Apply BraTS Toolkit preprocessing methods
- Handle scanner differences (ComBat harmonization)

### For Vision Transformer (Objective 2):
- Train ViT on 1,430 patients
- Each patient = temporal sequence of 3-10 scans
- Learn: tumor growth patterns, treatment response, progression

### For LLM (Objective 3):
- Use clinical metadata (treatment dates, demographics)
- Generate explanations: "Patient 67yo female, tumor grew 15% after radiation..."

### For Video Generation (Objective 4):
- Use temporal sequences (T0 → T1 → T2 → T3...)
- Train diffusion model to predict T4, T5, T6...
- Generate counterfactual: "What if different treatment?"

### For Validation (Objective 5):
- Hold out 20-30% for testing
- Measure: prediction accuracy, temporal consistency
- Validate with radiologists

---

## 8️⃣ CONNECTION TO OUR OBJECTIVES

**Which objectives does this paper help?**
- ✅ Objective 1: Preprocessing (provides the data to preprocess!)
- ✅ Objective 2: Vision Transformer (temporal sequences to train on)
- ✅ Objective 3: LLM (clinical metadata for explanations)
- ✅ Objective 4: Video generation (longitudinal sequences needed)
- ✅ Objective 5: Validation (large test set available)

**ALL OBJECTIVES!** This dataset is the foundation of your entire project.

**Specifically how**:
This dataset provides the ONLY large-scale longitudinal brain metastasis data publicly available. Without this, you couldn't train temporal models (no other dataset has multiple timepoints). It gives you:
1. Scale (1,430 patients - enough for deep learning)
2. Temporal coverage (8 scans × 1,430 = 11,884 data points)
3. Real clinical variability (20 years, multiple scanners, real treatments)

---

## 9️⃣ HOW IT COMBINES WITH OTHER PAPERS

**Papers we already have**:
1. BraTS Toolkit - Preprocessing methods
2. nnU-Net - Tumor segmentation baseline

**How Yale fits**:
- **Yale NEEDS BraTS preprocessing** → Yale data is messier (20 years, multiple scanners)
- **Yale NEEDS nnU-Net segmentation** → Yale has no tumor labels, use nnU-Net to create them
- **Yale PROVIDES what BraTS lacks** → Temporal data (BraTS = single timepoint)

**The workflow**:
```
Step 1: Download Yale dataset (this paper)
        ↓
Step 2: Apply BraTS Toolkit preprocessing to clean Yale scans
        ↓
Step 3: Use nnU-Net to segment tumors in Yale scans
        ↓
Step 4: NOW you have: Clean Yale data + Tumor labels
        ↓
Step 5: Train your temporal ViT on Yale temporal sequences
        ↓
Step 6: Add LLM using Yale clinical metadata
        ↓
Step 7: Generate videos from Yale progression sequences
```

**No overlap**: Yale is DATA, BraTS/nnU-Net are METHODS you apply TO Yale data.

---

## 🔟 CITATION EXAMPLE

"We trained our temporal Vision Transformer on the Yale Longitudinal Brain Metastases Dataset (Chadha et al., 2025), which contains 11,884 MRI studies from 1,430 patients with brain metastases tracked over time."

---

## 💡 KEY TAKEAWAYS

**For your project**:
1. ✅ **This IS your dataset** - Download it immediately!
2. ✅ **Largest temporal brain data** - Perfect for your longitudinal analysis
3. ⚠️ **No labels included** - Must generate tumor segmentations (use nnU-Net)
4. ⚠️ **Scanner variability** - Must harmonize (use ComBat methods)
5. ✅ **Public and free** - No approval delays

**Next steps**:
1. Download from TCIA (may take 1-2 days, ~200GB)
2. Explore the data structure
3. Run BraTS preprocessing pipeline
4. Generate tumor segmentations with nnU-Net
5. Start training temporal models!

**Impact on timeline**:
- Week 1: Download Yale dataset
- Week 2-4: Preprocess all 11,884 scans
- Week 5+: Begin training models

This paper just described your MAIN dataset. Everything else you read will be about METHODS to apply TO this data!
