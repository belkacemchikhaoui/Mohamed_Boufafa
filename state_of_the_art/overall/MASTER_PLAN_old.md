# Master Research Plan - Simple & Logical

## 🎯 Your Main Goal

Build an AI system that:
1. Takes brain cancer scans from the same patient over months/years
2. Tracks how tumors change
3. Explains what's happening in plain language
4. Predicts what might happen next with different treatments

---

## 📊 Your Datasets

### PRIMARY: Yale Longitudinal Brain Metastases (2025) - For TRAINING

**What it is**: 11,884 brain scans from 1,430 patients tracked over time

**Key facts**:
- ✅ Average 8 scans per patient (enough for temporal analysis!)
- ✅ Covers 2004-2023 (20 years of real clinical data)
- ✅ Has treatment information (pre/post surgery, radiation, chemo)
- ✅ FREE and publicly available on TCIA
- ⚠️ No tumor labels (you'll generate them with nnU-Net)
- ⚠️ Multiple scanners (need harmonization)

### SECONDARY: Cyprus Brain Metastases (2025) - For VALIDATION (Optional)

**What it is**: 744 brain scans from 40 patients with expert-verified labels

**Key facts**:
- ✅ Average 18.6 scans per patient (more temporal detail!)
- ✅ Expert radiologist verified ALL tumor segmentations
- ✅ 3 tumor subregions labeled (enhancing, necrosis, edema)
- ✅ Radiotherapy plans + clinical data included
- ✅ FREE and publicly available on Zenodo
- 📝 Mediterranean population (different from Yale's USA)

**Strategy**: 
- **Focus on Yale FIRST** (larger scale for training)
- **Cyprus for external validation LATER** (optional - test if model generalizes to different population)
- **"Combining" datasets**: Means using Cyprus as independent validation set (Option A), NOT mixing them together
- **Decision**: Will be made with supervisor after Phase 1 complete

---

## 🏗️ BUILD YOUR RESEARCH IN THIS ORDER

### PHASE 1: Get Data Working (MUST DO FIRST - Weeks 1-4)

**Why first**: Without clean, aligned data, NOTHING else works!

#### Step 1.1: Download and Clean Data
**Papers analyzed**: 
- ✅ Yale Dataset (Paper 3) - Main training data
- ✅ Cyprus Dataset (Paper 7) - Validation data (optional for now)
- ✅ BraTS Toolkit (Paper 1) - How to clean brain scans

**What to do**:
1. Download Yale from TCIA (~200GB) - **PRIORITY**
2. (Optional) Download Cyprus from Zenodo (~100GB) - for validation later
3. Convert to standard format (NIfTI)
4. Remove skull using HD-BET
5. Normalize intensities

**Note**: Cyprus already in BraTS format, so less preprocessing needed!

**Time**: Week 1

---

#### Step 1.2: Find Tumors in Every Scan
**Papers analyzed**:
- ✅ nnU-Net (Paper 2) - Automatic tumor segmentation
- ✅ Cyprus Dataset (Paper 7) - Already has expert labels!

**What to do**:
1. **Yale**: Run nnU-Net on all 11,884 scans (generate labels)
2. **Yale**: Get tumor locations and sizes
3. **Yale**: Quality check (manually review 50-100 cases)
4. **Cyprus** (if downloaded): Already has 65 tumors labeled by expert radiologist - can use to validate nnU-Net accuracy!

**Why critical**: You need tumor locations for everything else!

**Validation trick**: Run nnU-Net on Cyprus → Compare to expert labels → Know your accuracy!

**Time**: Week 2

---

#### Step 1.3: Align Scans Over Time
**Papers analyzed**:
- ✅ Registration (Paper 4) - Tumor-preserving alignment
- ✅ FLIRE (Paper 5) - Super-fast registration (9× faster!)

**What to do**:
1. For each patient, pick baseline scan (T0)
2. Align all other scans to T0 (T1→T0, T2→T0, etc.)
3. **Choose registration method**:
   - **FLIRE**: 10 mins/scan, 0.98 accuracy (if available/implemented)
   - **Paper 4 method**: Proven tumor volume preservation (94.7%)
   - **Elastix**: Publicly available, 14.5 mins/scan, 0.95 accuracy
4. Preserve tumor volumes (track real changes!)

**Speed comparison for Yale (11,884 scans)**:
- FLIRE: ~80 days total processing ✅
- Paper 4 method: Unknown runtime, but proven accuracy
- Elastix: ~116 days (1.5× slower than FLIRE)
- DRAMMS: ~716 days (TOO SLOW!) ❌

**Why critical**: Can't compare T0 vs T1 if they're not aligned!

**Time**: Week 3 (first half)

---

#### ~~Step 1.4: Extract Tumor Features~~ ❌ REMOVED — ViT DOES THIS!
> **Why removed**: The supervisor confirmed that feature extraction is Objective 2's job (ViT).
> Extracting hand-crafted radiomic features here would be redundant — ViT learns **richer** 
> representations automatically. ComBat harmonization moves to Phase 2 (after ViT extracts embeddings).
> 
> **Old plan**: Extract 110+ radiomic features → Harmonize with ComBat → Feed to ViT
> **New plan**: Feed clean scans + masks to ViT → ViT extracts embeddings → Harmonize embeddings with ComBat
>
> This is simpler, avoids double work, and lets ViT learn what matters instead of us guessing.

---

#### ~~Step 1.5: Comprehensive Scanner Harmonization~~ → MOVED TO PHASE 2
> **Why moved**: ComBat harmonizes FEATURES. Since ViT now extracts features (not us manually),
> harmonization logically comes AFTER ViT produces embeddings in Phase 2.
> See Phase 2, Step 2.2 for the full harmonization strategy.

---

**✅ CHECKPOINT END OF PHASE 1**: 
- You have 1,430 Yale patients ready
- Each patient has ~8 clean, skull-stripped, aligned scans
- Tumor segmentation masks for all 11,884 scans (nnU-Net)
- (Optional) Cyprus 40 patients ready as independent test set
- **Ready for Phase 2: ViT will extract features + harmonize them!**

---

### COMPLETE PHASE 1 PIPELINE SUMMARY 🎯

**Step 1: Download & Clean**
- Papers: BraTS Toolkit, Yale Dataset, Cyprus Dataset
- Output: 11,884 Yale scans + 744 Cyprus scans in standard format

**Step 2: Segment Tumors**
- Papers: nnU-Net
- Output: Tumor masks for all scans (Yale generated, Cyprus expert-verified)

**Step 3: Align Temporal Sequences**
- Papers: Registration, FLIRE
- Output: All patient timepoints aligned to baseline (T1→T0, T2→T0, etc.)

**Output**: Clean, segmented, aligned longitudinal brain MRI data → Ready for ViT!

**Why this order?**
```
Raw scans → Clean (BraTS Toolkit)
Clean scans → Segment tumors (nnU-Net) 
Segmented scans → Align over time (FLIRE/Elastix)
Aligned + segmented scans → DONE! Feed to ViT in Phase 2 ✅
```

**What about feature extraction & harmonization?**
- ViT extracts features (Phase 2) — richer than hand-crafted radiomics
- ComBat harmonizes ViT embeddings (Phase 2) — after ViT, not before
- This avoids redundant work and keeps the pipeline simple

**Decision point (discuss with supervisor)**: Should we use Cyprus for external validation?
- **Option A**: Train on Yale only (1,430 patients) → Test on Cyprus (40 patients) = Proves model works on different population ✅
- **Option B**: Use Yale only, ignore Cyprus = Simpler but can't prove generalization
- **NOT recommended**: Mixing Yale + Cyprus in training = Loses independent validation
- **Your decision**: Will be made with supervisor after seeing Phase 1 results

---

### PHASE 2: Build Temporal Tumor Tracker (Weeks 5-11)

**Why second**: This is your CORE innovation - tracking changes over time

#### Step 2.1: ViT Papers (Week 5) ✅ 6 PAPERS AVAILABLE!
**Papers found** (6 public, prioritized):
- ✅ **Paper 11**: Time-distance ViT (2022) — ⭐⭐⭐ TEMPORAL BLUEPRINT! Encodes time intervals into ViT
- ✅ **Paper 13**: Swin UNETR (2022) — ⭐⭐⭐ 3D brain tumor segmentation + embedding extraction
- ✅ **TransXAI**: Explainable hybrid transformer (2024) — ⭐⭐ Explainability for Phase 3
- ✅ **CAFNet**: CNN+ViT cross-attention fusion (2025) — ⭐ Proves ViT+CNN >> pure ViT
- ✅ **ResAttU-Net-Swin**: Attention + Swin hybrid (2025) — ⭐ Backup if Swin UNETR underperforms
- ✅ **BRAIN-META**: CNN-ViT ensemble (2025) — ⭐ Ensemble strategy reference

**Core architecture** (from Papers 11 + 13) — NOW WITH CONCRETE DETAILS:
- **Swin UNETR encoder** (Paper 13): 4-stage hierarchical Swin Transformer, 61.98M params
  - Input: 128×128×128×4 (T1, T1c, T2, FLAIR) — same as Yale!
  - Stages: [48, 96, 192, 384, 768] channels, window 7×7×7, heads [3,6,12,24]
  - Bottleneck: 4×4×4×768 → Global Avg Pool → **768-dim embedding per scan**
  - Pretrained weights: `model_swinvit.pt` (MONAI)
  - Dice: 0.913 avg (beats nnU-Net 0.908!) — also upgrades Phase 1 segmentation!
- **TaViT temporal model** (Paper 11): Learnable sigmoid scales attention by time distance
  - Temporal Emphasis: `f(R) = 1/(1+exp(a×R - c))` → recent scans get higher weight
  - 8 attention heads, 8 encoder layers, masked autoencoder pretraining
  - AUC: 0.786 (beats single-scan CNN 0.734, p<0.05)
  - **CRITICAL**: Without time encoding → 0.50 AUC on irregular data (random chance!)
- Combined pipeline: Swin UNETR extracts 768-dim → ComBat harmonizes → TaViT temporal model

**Detailed analyses**: 
- `objectivetwo/SIMPLE_Paper11_Time_Distance_ViT.md` ✅
- `objectivetwo/SIMPLE_Paper13_Swin_UNETR.md` ✅
- `PHASE2_VIT_PAPERS_ANALYSIS.md` (overview)

---

#### Step 2.2: ViT Feature Extraction + Harmonization (Week 6-7)

**This is where feature extraction AND harmonization happen!**

The ViT takes clean, segmented, aligned scans (Phase 1 output) and learns **rich embeddings** 
that capture tumor morphology, texture, spatial patterns — far more powerful than hand-crafted radiomics.

**Concrete Architecture (from Paper 13 — Swin UNETR)**:
```
EMBEDDING EXTRACTION — Using Swin UNETR Encoder:

from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,        # T1, T1c, T2, FLAIR (same as Yale!)
    out_channels=3,       # ET, WT, TC
    feature_size=48,      # C = 48 initial embedding
)

# Load NVIDIA pretrained weights (5,050 CT volumes + BraTS fine-tuning)
weights = torch.load("model_swinvit.pt")
model.load_from(weights=weights)

# Fine-tune on Yale → then extract embeddings:
# Encoder bottleneck: 4×4×4×768 → Global Avg Pool → 768-dim per scan
```

**Pipeline**:
```
Phase 1 output (clean + segmented + aligned scans)
    ↓
Swin UNETR encoder (Paper 13) processes each scan
    → Output: 768-dim embedding vector per scan
    → For each patient: 8 embeddings × 768 dimensions = (8, 768) matrix
    ↓
TEST for scanner effects in embeddings (Paper 10++ method)
    ↓  (if effects found)
Nested ComBat on embeddings (Paper 10+) → Remove batch effects (year/field/manufacturer)
    ↓
Longitudinal ComBat on embeddings (Paper 10) → Preserve patient trajectories
    ↓
Harmonized ViT embeddings → Ready for temporal modeling! ✅
```

**Why this works better than the old plan**:
- ViT learns what matters (not us guessing which 110 features to extract)
- ComBat works on ANY features — hand-crafted OR learned embeddings
- One pipeline instead of two redundant feature extraction steps
- ViT + ComBat = richer features with scanner effects removed

**Papers used**:
- **Paper 13 (Swin UNETR)** — embedding extraction (768-dim from encoder bottleneck)
- ComBat Validation (Paper 10++) — test before harmonizing
- Generalized ComBat (Paper 10+) — Nested for multi-batch effects
- Longitudinal ComBat (Paper 10) — preserve trajectories
- Note: Skip basic ComBat (Paper 9) — Nested does everything it does + more

**nnU-Net's role here**: Can be REPLACED by Swin UNETR (Paper 13) — same model does BOTH segmentation (decoder) AND embedding extraction (encoder). This means one model for two jobs!

**BONUS**: Swin UNETR Dice 0.913 > nnU-Net 0.908 → Free segmentation upgrade!

---

#### Step 2.3: Design Temporal Attention (Week 8-9)
**Concrete Architecture (from Paper 11 — TaViT)**:

**Input**: Sequence of 8 harmonized Swin UNETR embeddings per patient + time distances

**TaViT Temporal Emphasis Model (TEM)**:
```
For each pair of scans (i, j):
  R_i,j = |time_i - time_j| in days
  Emphasis = 1 / (1 + exp(a × R_i,j - c))
  
  Where a, c are LEARNABLE parameters:
  → a controls how fast emphasis drops with time distance
  → c controls where the transition happens
  
  This SCALES the self-attention weights:
  Attention_temporal = softmax(Q × K^T / √d) × TEM(R)
  
  → Recent scan pairs: TEM ≈ 1.0 (full attention)
  → Old scan pairs: TEM ≈ 0.0 (minimal attention)
  → Model LEARNS the optimal decay curve!
```

**Architecture specs** (from Paper 11, scaled for Swin UNETR embeddings):
```
Input: (batch, 8 timepoints, 768 dims) + time distances per patient
Projection: Linear(768, D_model)  [if needed to reduce dimensionality]
+ [CLS] token prepended
+ TaViT time-distance attention scaling
Transformer: 8 layers, 8 attention heads
Output: [CLS] token → 768-dim temporal representation
```

**Pretraining (Masked Autoencoding — no labels needed!)**:
```
1. Take patient's 8 embeddings
2. Mask 75% of them (keep 2 visible)
3. Model predicts masked embeddings from visible ones + time info
4. Pretrain on ALL 11,884 Yale scans (self-supervised!)
5. Then fine-tune on downstream tasks
```

**What it learns**:
- How fast tumors grow (embedding distance ∝ volume change)
- Where tumors spread (spatial features in 768-dim capture location shifts)
- Response to treatment (shrinking vs growing patterns)
- Scanner-invariant progression (ComBat already removed scanner effects!)

**Why TaViT over TeViT?** (Paper 11 comparison):
- TaViT: 0.786 AUC, learnable emphasis, recent scans weighted higher → BEST for clinical use
- TeViT: 0.785 AUC, fixed sinusoidal, equal weighting → simpler but less adaptive
- Critical: WITHOUT time encoding → 0.50 AUC on irregular data (confirmed in paper!)
- Yale has irregular intervals → TaViT is essential, not optional!

---

#### Step 2.4: Train Your Model (Week 9-10)

**PRIMARY APPROACH: Yale Only**
- Training: 1,000 Yale patients (70%)
- Validation: 215 Yale patients (15%)
- Testing: 215 Yale patients (15%)

**OPTIONAL: External Validation on Cyprus (discuss with supervisor)**
- After training on Yale, test same model on Cyprus 40 patients
- **Purpose**: Proves model works on Mediterranean population (not just USA)
- **Note**: Cyprus has same disease (brain metastases) but different:
  - Primary tumors (mostly lung/breast in Cyprus vs diverse in Yale)
  - Population genetics (Mediterranean vs USA)
  - Scanner protocols (2019-2024 vs 2004-2023)
- **Decision**: Supervisor will decide if this validation is valuable for your thesis

**Baseline comparison**: Must beat nnU-Net (Paper 2)

---

#### Step 2.5: Evaluate Representations (Week 11)

**A) Downstream Prediction** ("Are the embeddings useful?"):
Feed ViT embeddings to a SIMPLE classifier (logistic regression / small MLP) to predict:
- Tumor growth classification: Growing vs Stable vs Shrinking
- Treatment response: Responder vs Non-responder
- Tumor volume at next timepoint (regression task)
- Compare: ViT embeddings vs hand-crafted radiomics vs CNN features
- Tool: `sklearn.linear_model.LogisticRegression` on frozen embeddings
- If a simple classifier works well → ViT learned rich representations!

**B) Clustering** ("Do similar tumors group together?"):
Visualize all embeddings (one per scan) in 2D/3D:
- Use t-SNE or UMAP on all embeddings → plot colored by:
  - Tumor stage (early vs advanced) → should cluster separately
  - Patient ID (same patient's T0-T7) → should form trajectories
  - Scanner/year (2004 vs 2023) → should be mixed (if ComBat worked!)
- Tool: `umap-learn`, `sklearn.manifold.TSNE`
- No labels needed — just check if groupings make biological sense

**C) Temporal Consistency**:
- Do predictions make sense T0→T1→T2? (monotonic growth/shrinkage?)
- Is embedding distance proportional to time distance?
- Can T0+T1+T2 embeddings predict T3 tumor characteristics?

**Metrics**:
- Downstream: AUC, accuracy, F1 for classification; RMSE for volume prediction
- Clustering: Silhouette score, adjusted Rand index
- Temporal: Pearson correlation between embedding distance and time distance

**✅ CHECKPOINT WEEK 11**: 
- ViT embeddings outperform hand-crafted radiomics on downstream tasks
- Clustering shows biologically meaningful groupings
- Temporal embeddings capture disease progression

---

### PHASE 3: Add Explanations with LLM (Weeks 12-14)

**Why third**: Once you have accurate tracking, ADD explanations

**Architecture validation** (from CAFNet analysis):
> CAFNet (2025) proved hybrid CNN+ViT with cross-attention fusion achieves 96.41% vs 87.34% pure ViT (+9%).
> Our Swin UNETR IS a hybrid (Transformer encoder + CNN decoder) — this validates our design choice.
> CAFNet's ablation: simple concat 92.20% vs cross-attention 96.41% — fusion METHOD matters!
> See: `objectivetwo/SIMPLE_Paper_CAFNet.md`

#### Step 3.1: Generate Explainability Maps (Week 12) ✅ METHOD FOUND!
**Explainability approach** (from TransXAI — Zeineldin et al., 2024):
- ✅ **Grad-CAM on Swin UNETR decoder layers** → spatial heatmaps showing WHERE model detects tumors
- ✅ **Per-MRI-modality analysis** → feed T1/T1c/T2/FLAIR separately → learn which modality detects which sub-region
  - TransXAI found: T1Gd → ET detection, FLAIR → edema/WT, T1 → minimal contribution
- ✅ **Internal layer visualization** → verify encoder learns meaningful features (white/gray matter, tumor boundaries)
- ✅ **TaViT temporal emphasis weights** → built-in temporal explainability (which timepoints model focuses on)
- ✅ **Tools**: NeuroXAI framework (https://github.com/razeineldin/NeuroXAI) + PyTorch Grad-CAM
- **Key advantage**: Post-hoc explainability → NO accuracy tradeoff, NO architecture modifications!
- **Clinical validation protocol**: TransXAI had 2 neurosurgeons evaluate heatmaps → confirmed consistency with clinical knowledge

**Papers analyzed**: See `objectivetwo/SIMPLE_Paper_TransXAI.md`

**Papers still needed**:
- ⏳ LLMs for radiology reports
- ⏳ Vision-language models (CLIP, BLIP, etc.)
- ⏳ Medical reasoning with LLMs
- ⏳ Multimodal fusion

---

#### Step 3.2: Connect ViT to LLM (Week 13)
**Architecture** (now with concrete explainability inputs):
```
Swin UNETR + TaViT pipeline
    ↓
Explainability layer (TransXAI approach):
  → Grad-CAM heatmaps: WHERE tumors detected per sub-region
  → Modality maps: WHICH MRI sequence contributed most
  → TaViT weights: WHICH timepoints model focused on
  → Embedding distances: HOW MUCH tumor changed between scans
    ↓
Structured features → LLM
    ↓
Generated report
```

**Example output**:
> "Patient 67yo female with lung cancer metastasis. Swin UNETR detected 15mm enhancing lesion in right frontal lobe (T1Gd primary contributor, confidence 92%). Temporal analysis shows 20% growth over 6 months (T0→T1), stabilization after radiosurgery (T1→T2). TaViT attention focused 94% on post-treatment scans. Recommend: continued surveillance, next scan in 3 months."

---

#### Step 3.3: Train and Validate (Week 14)
**Training**:
- Use Yale clinical metadata
- Fine-tune LLM on medical language
- Feed Grad-CAM heatmaps + temporal weights as structured input
- Validate: Do radiologists agree with explanations?

**Clinical validation** (following TransXAI protocol):
- Structured interviews with medical experts
- Evaluate: accuracy of heatmaps, clarity of explanations, alignment with clinical practice
- TransXAI showed this INCREASES clinician trust in AI systems

**✅ CHECKPOINT WEEK 14**:
- LLM generates accurate explanations with visual evidence
- Grad-CAM heatmaps highlight correct tumor regions
- Radiologists rate explanations as clinically useful

---

### PHASE 4: Generate Future Prediction Videos (Weeks 15-18)

**Why fourth**: Most advanced feature, requires everything else working

#### Step 4.1: Find Diffusion Papers (Week 15)
**Papers needed** (not found yet):
- ⏳ Diffusion models for medical imaging
- ⏳ Video generation models
- ⏳ Temporal prediction
- ⏳ Counterfactual generation

---

#### Step 4.2: Build Video Generator (Week 16-17)
**Input**: Patient's past scans (T0, T1, T2)
**Output**: Predicted future scans (T3, T4, T5)

**Two modes**:
1. Natural progression (no treatment)
2. Counterfactual (what if surgery? what if radiation?)

---

#### Step 4.3: Train on Yale Sequences (Week 18)
**Training**:
- Use real Yale progressions as ground truth
- Learn: tumor growth patterns, treatment responses

**✅ CHECKPOINT WEEK 18**:
- Can generate realistic future progressions
- Videos match real outcomes (validate on test set)

---

### PHASE 5: Prove It Works Clinically (Weeks 19-20)

**Why last**: Final validation before publication

#### Step 5.1: Quantitative Evaluation (Week 19)
**Metrics**:
- Accuracy vs ground truth (Yale test set)
- Temporal consistency
- Volume predictions
- Compare to baselines (nnU-Net, other methods)

---

#### Step 5.2: Clinical Validation (Week 20)
**Papers needed** (find later):
- ⏳ RANO criteria for brain tumors
- ⏳ Clinical evaluation methods

**What to do**:
1. Radiologist review (do they trust it?)
2. Clinical utility (does it help decisions?)
3. Compare to current clinical practice

**✅ FINAL RESULT**: Publishable research!

---

## 📝 Simple Paper Analysis Template

For EACH paper you mention, I'll extract:

### 1. One-Sentence Summary
What does this paper do?

### 2. Key Results (Numbers)
- What accuracy did they achieve?
- What datasets did they use?
- How well did it work?

### 3. What's New
What did they add that nobody else did?

### 4. Limitations
What doesn't work? What's missing?

### 5. Methods (Simple)
How did they do it? (explained simply)

### 6. Code/Resources
- Is code available?
- What datasets?
- Can you reproduce it?

### 7. Connection to Yale Dataset
How does this help you work with Yale Brain Mets data?

### 8. Connection to Your Objectives
Which of your 5 objectives does this help?

---

## 🔄 How Papers Combine

Think of it like building a house with Yale dataset as your land:

```
FOUNDATION (Weeks 1-4):
Papers: BraTS, registration, harmonization
→ Clean up Yale dataset
→ Align all 11,892 scans properly

WALLS (Weeks 5-11):
Papers: Vision Transformer papers
→ Build ViT that learns from Yale temporal sequences
→ Train on 1,430 patients x 8 scans each

ELECTRICAL (Weeks 12-14):
Papers: LLM papers
→ Connect ViT to language model
→ Use Yale clinical metadata to train explanations

WINDOWS (Weeks 15-18):
Papers: Diffusion/video papers
→ Generate future progression videos
→ Trained on Yale temporal sequences

INSPECTION (Weeks 19-20):
Papers: Validation papers
→ Test on Yale test set
→ Prove it works
```

---

## 📋 Current Status

### Papers Analyzed (2):
1. ✅ BraTS Toolkit - Preprocessing foundation
2. ✅ nnU-Net - Baseline for comparison

---

## 📊 WHAT WE'VE LEARNED SO FAR (4 Papers Analyzed)

### ✅ Papers That Help Phase 1 (Must Do First):

**Paper 1 - BraTS Toolkit**: How to clean brain scans
- Remove skull (HD-BET)
- Normalize intensities
- Convert formats
- **Use in**: Week 1 (Step 1.1)

**Paper 2 - nnU-Net**: How to find tumors automatically
- Self-configuring segmentation
- 67-95% accuracy on medical images
- **Use in**: Week 2 (Step 1.2)
- **Also**: Your baseline to beat!

**Paper 3 - Yale Dataset**: Your main data
- 11,884 scans, 1,430 patients
- Download from TCIA (free!)
- **Use in**: Week 1 (download immediately!)

**Paper 4 - Registration**: How to align temporal sequences
- Unsupervised keypoint detection
- Preserves tumor volumes
- 94.7% alignment accuracy
- **Use in**: Week 3 (Step 1.3)

---

### ⏳ Papers Still Needed (In Priority Order):

#### URGENT (Need for Phase 1 - Weeks 1-4):
1. **ComBat Harmonization** (Paper 9) - ANALYZE NEXT!
   - Fix scanner differences
   - Yale has 20 years of different scanners
   - **Need for**: Week 4 (Step 1.4)

#### HIGH PRIORITY (Need for Phase 2 - Weeks 5-11):
2. **Swin Transformer** or similar ViT paper
   - Learn how to build temporal Vision Transformer
   - **Need before**: Week 6

3. **Temporal/Longitudinal ViT** paper
   - How to handle sequences (T0→T1→T2...)
   - **Need before**: Week 6

#### MEDIUM PRIORITY (Need for Phase 3 - Weeks 12-14):
4. **LLM for Radiology** paper
   - How to generate medical explanations
   - **Need before**: Week 12

5. **Vision-Language Model** paper (CLIP, BLIP, etc.)
   - How to connect images to text
   - **Need before**: Week 13

#### LOWER PRIORITY (Need for Phase 4-5 - Weeks 15-20):
6. **Diffusion Model** for medical imaging
   - **Need before**: Week 16

7. **Video Generation** paper
   - **Need before**: Week 16

8. **Clinical Validation** papers
   - **Need before**: Week 19

---

## 🎯 PAPER READING STRATEGY

### NOW (Week 1-4 preparation):
✅ Analyze **Paper 9 (ComBat Harmonization)** immediately
- Last piece of Phase 1!
- Can't start coding without this

### SOON (Week 5-6 preparation):
⏳ Find and analyze **2-3 Vision Transformer papers**
- Search: "Swin Transformer medical imaging"
- Search: "Temporal Vision Transformer"
- Search: "3D Vision Transformer volumetric"

### LATER (Week 12+ preparation):
⏳ Find and analyze **LLM + Diffusion papers**
- Only need these after Phase 1-2 working
- Don't rush into these yet!

---

## 💡 KEY INSIGHTS FROM PAPERS SO FAR

### From BraTS Toolkit (Paper 1):
- ✅ HD-BET is standard for skull removal (Yale paper uses it too!)
- ✅ Multiple AI models can be combined (ensemble)
- ⚠️ Only handles single timepoints (you'll add temporal)

### From nnU-Net (Paper 2):
- ✅ Self-configuring = don't waste time tuning hyperparameters
- ✅ Preprocessing matters MORE than fancy architectures
- ✅ This is your BASELINE - must beat it!
- 💡 Insight: Good data preparation > complex models

### From Yale Dataset (Paper 3):
- ✅ Big enough for deep learning (1,430 patients)
- ✅ Real clinical variability (good for generalization)
- ⚠️ No labels (use nnU-Net to create them)
- ⚠️ Scanner differences (need ComBat harmonization)

### From Registration (Paper 4):
- ✅ Unsupervised keypoints = no manual labels needed!
- ✅ Volume preservation = critical for tracking real changes
- 💡 Insight: Where to deform vs where NOT to deform
- 🔧 Need to adapt: Breast → Brain (different anatomy)

---

## 🚨 CRITICAL DEPENDENCIES

**Can't start Phase 2 until Phase 1 complete!**

```
Phase 1 (Data Ready) → Phase 2 (ViT) → Phase 3 (LLM) → Phase 4 (Video) → Phase 5 (Validation)
        ↑
   MUST FINISH FIRST!
```

**Why**:
- Dirty data = bad AI models (garbage in, garbage out)
- Misaligned scans = can't track temporal changes
- Scanner differences = model learns scanner artifacts, not biology

---

## 📝 CURRENT STATUS

**Phase 1 Progress**: 80% Complete
- ✅ Data identified (Yale)
- ✅ Cleaning method (BraTS)
- ✅ Segmentation method (nnU-Net)
- ✅ Registration method (Paper 4)
- ⏳ Harmonization method (need Paper 9)

**Ready to code**: ⏳ After analyzing Paper 9 (1 more paper!)

**Papers analyzed**: 4/~20 (20% complete)

---

## 🎯 IMMEDIATE NEXT STEPS

1. **Analyze Paper 9** (ComBat Harmonization)
   - Finish Phase 1 preparation
   - Then start downloading Yale and coding!

2. **Download Yale dataset** (can do in parallel)
   - Go to TCIA
   - Search "Yale Brain Metastases"
   - ~200GB download

3. **Find ViT papers** (while Phase 1 downloads)
   - Search for 2-3 papers
   - Read before Week 5

4. **Set up coding environment**
   - Install: PyTorch, nnU-Net, HD-BET
   - Get GPU access
   - Prepare storage (1TB recommended)

---

**Remember**: BUILD IN ORDER! Finish Phase 1 before Phase 2. Each phase depends on the previous one working correctly!
