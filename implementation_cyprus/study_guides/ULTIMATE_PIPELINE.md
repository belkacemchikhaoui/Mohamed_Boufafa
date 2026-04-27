# Ultimate Research Pipeline

**Multimodal AI for Explainable Cancer Progression and Counterfactual Video Generation**

> **Intern**: Mohamed Boufafa | **Supervisor**: Dr. Belkacem Chikhaoui  
> **Institution**: TELUQ University, Montreal | **Program**: Mitacs Globalink Research Internship  
> **Duration**: 20 weeks (April-August 2026) | **Dataset**: Cyprus PROTEAS (primary) + Yale (training)

---

## How to Read This Document

This document maps **5 Objectives** to **6 Phases** with **22 papers**, showing exactly what we do, why, and what the CNN vs ViT comparison means at every step.

```
OBJECTIVES (WHAT)                    PHASES (WHEN)
─────────────────                    ─────────────
Obj 1: Data Pipeline           ───→  Phase 1 (Weeks 1-4)
Obj 2: ViT Representations     ───→  Phase 2 (Weeks 5-7) + Phase 3 (Weeks 8-11)
Obj 3: LLM Integration         ───→  Phase 4 (Weeks 12-14)
Obj 4: Video Generation         ───→  Phase 5 (Weeks 15-18)
Obj 5: Evaluation               ───→  Phase 6 (Weeks 19-20)
```

---

## The Pipeline in One Diagram

```
Raw MRI → HD-BET → nnU-Net → itk-elastix → [CNN Baseline] → Swin UNETR → ComBat → TaViT → RadFM → Video Diffusion → Evaluation
   │          │         │          │              │                │           │         │        │           │              │
 Week 1    Week 1    Week 2    Week 3         Week 5            Week 8      Week 8   Week 9  Week 12     Week 15        Week 19
   └──────────── Objective 1 ────────┘    └─── Obj 2 (CNN) ──┘ └──── Obj 2 (ViT) ────┘   └─ Obj 3 ─┘  └── Obj 4 ──┘    └ Obj 5 ┘
                                          └──── COMPARE ─────────────────────────────┘
```

---

## Core Research Questions

1. Can **ViT-based models** capture richer tumor representations than CNNs?
2. Can **temporal modeling** (TaViT) improve prediction over single-timepoint analysis?
3. Can **LLMs** generate clinically coherent progression narratives from imaging embeddings?
4. Can **diffusion models** generate temporally consistent cancer progression videos?
5. Can the system support **counterfactual analysis** (treatment A vs B)?

---

## Datasets

| | Cyprus PROTEAS (Primary) | Yale (Optional Scale-Up) |
|---|---|---|
| Patients | 40 (45 directories) | 1,430 |
| Total MRI scans | 744 (186 timepoints x 4 modalities) | 11,884 |
| Avg scans/patient | 18.6 | 8.3 |
| Tumor labels | **Expert-verified** (NCR, ET, ED) | None (need nnU-Net) |
| Radiomic features | 7,980 pre-extracted | Must extract |
| RT dose plans | 45 RTP files | Some available |
| Clinical metadata | Rich (KPS, treatment, histology) | Basic |
| Preprocessing | Already BraTS-formatted | Needs full pipeline |
| Size | 2.65 GB (NIfTI only) | ~200 GB |

**Strategy**: Cyprus is our primary dataset (expert labels, rich metadata, manageable size). Yale can be used for scale-up or cross-population validation.

---

# OBJECTIVE 1 -- Robust Pipeline for Longitudinal Cancer Imaging Analysis

> *"Build a reproducible and clinically meaningful data pipeline for processing and organizing longitudinal cancer imaging data."*

## Phase 1: Oncology Medical Imaging Preparation (Weeks 1-4)

### Step 1.1: Data Acquisition and Preprocessing

| Task | Tool | Paper | Status (Cyprus) |
|------|------|-------|-----------------|
| Dataset acquisition | Zenodo download | Flouri et al., 2025 | **DONE** |
| Skull stripping | HD-BET (BraTS Toolkit) | Kofler et al., 2020 | **DONE** (P28-P39 corrected) |
| Intensity normalization | Z-score per modality | BraTS protocol | **DONE** (validated: error=0.0) |
| Spatial resampling | Already 1mm isotropic | -- | **DONE** (240x240x155) |

### Step 1.2: Tumor Segmentation Verification

| Task | Tool | Paper | Status (Cyprus) |
|------|------|-------|-----------------|
| Expert mask validation | Custom pipeline | -- | **DONE** (171 masks, 0 empty) |
| Label distribution | Pandas/matplotlib | -- | **DONE** ({0,1,2,3}=84, {0,2,3}=72) |
| Missing mask documentation | -- | -- | **DONE** (15/186 visits, likely resolved) |

For Yale: Run **nnU-Net** (Isensee et al., 2021, Dice ~0.908) on all 11,884 scans. Validate against Cyprus expert labels.

### Step 1.3: Longitudinal Alignment

| Task | Tool | Paper | Status (Cyprus) |
|------|------|-------|-----------------|
| Intra-visit alignment | BraTS Toolkit | Kofler et al., 2020 | **DONE** (176/176 = 100%) |
| Inter-visit alignment | itk-elastix | Niessen et al., 2023 | **DONE** (552/552 = 100%) |
| CoM shift analysis | numpy | -- | **DONE** (~2mm mean, ~6mm max) |

For Yale: Apply itk-elastix rigid + B-spline deformable registration to all patient sequences.

### Step 1.4: Exploratory Data Analysis

| Task | Status (Cyprus) |
|------|-----------------|
| Tumor volume distributions (NCR, ET, ED, WT) | **DONE** |
| Spatial distributions + standardized locations | **DONE** (37 labels → 7 lobes) |
| Temporal trajectories + RANO response | **DONE** |
| Radiomic feature analysis (7 categories, PCA) | **DONE** |
| CT/RTP analysis + clinical cross-reference | **DONE** |
| Clinical data cleaning (5 fixes, 28 columns) | **DONE** |
| Follow-up retention curve (100% → 23%) | **DONE** |

### Phase 1 Deliverables

| Deliverable | Status |
|-------------|--------|
| Reproducible preprocessing pipeline | **DONE** |
| Cleaned, standardized, temporally organized datasets | **DONE** |
| Exploratory data analysis report | **DONE** |
| `PROTEAS_Clinical_Cleaned.xlsx` (28 columns) | **DONE** |
| `cyprus_patient_timelines.csv` | **DONE** |
| Phase1_Complete_EDA.ipynb (40 cells) | **DONE** |

> **Phase 1 Status: COMPLETE for Cyprus**

---

# OBJECTIVE 2 -- Learn High-Level Tumor Representations Using Vision Transformers

> *"Extract semantically rich and temporally informative representations of tumors using state-of-the-art Vision Transformer architectures."*

This objective spans **two phases** and contains the **core scientific comparison** of the project.

## Understanding the Role of Each Model

```
                     Phase 2                          Phase 3
                  ┌─────────┐                     ┌───────────┐
                  │ 3D U-Net │                     │ Swin UNETR│
  MRI Volume ──→  │  (CNN)   │──→ Segmentation     │   (ViT)   │──→ Segmentation (better?)
  (4ch, 96³)      │         │──→ CNN Embedding     │           │──→ ViT Embedding (768-dim)
                  └─────────┘    (256-dim)         └───────────┘          │
                       │                                │                  │
                       │         COMPARE AT             │                  ▼
                       └──────── EVERY LEVEL ───────────┘             ┌────────┐
                                                                      │ TaViT  │──→ Temporal Embedding
                                                                      │(Time)  │    (768-dim, time-aware)
                                                                      └────────┘
                                                                           │
                                                              ┌────────────┴────────────┐
                                                              │                         │
                                                         Phase 4                   Phase 5
                                                     ┌───────────┐            ┌───────────┐
                                                     │   RadFM   │            │  TaDiff   │
                                                     │   (LLM)   │            │  (Video)  │
                                                     └───────────┘            └───────────┘
                                                    Clinical narrative    Progression video
```

### Why CNN is the Baseline and ViT is the Experiment

The supervisor asks to:
1. **"Compare Transformer-based representations with CNN baselines"**
2. **"Learn embeddings that capture tumor morphology, spatial heterogeneity, and temporal evolution"**

The CNN (Phase 2) can do 1 and partially 2.  
The ViT (Phase 3) must do ALL of 2, and do it **measurably better**.  
Without the CNN baseline, we cannot claim ViT adds value.

---

## Phase 2: CNN Baseline (Weeks 5-7)

### What CNN Does

| Supervisor Activity | CNN Approach | What CNN Can Measure | What CNN CANNOT Do |
|---------------------|-------------|---------------------|-------------------|
| "Train on single time-point tasks" | 3D U-Net on one MRI volume → segmentation | Dice score per tumor subregion | Cannot see temporal patterns |
| "Quantitative evaluation" | Dice, HD95, sensitivity, specificity | Segmentation quality | Cannot predict future |
| "Identify limitations of static modeling" | Document failures | Which cases CNN gets wrong | Cannot explain WHY tumor changed |

### CNN Embedding Extraction

After training segmentation, we extract the **bottleneck features** as the "CNN representation":

```python
# 3D U-Net: encoder produces feature map at bottleneck
# Global Average Pool → 256-dim vector = "CNN embedding"
model = UNet(in_channels=4, out_channels=4, channels=(32,64,128,256))
# Hook into bottleneck layer
bottleneck_features = model.encoder[-1](x)  # shape: (batch, 256, 12, 12, 12)
cnn_embedding = bottleneck_features.mean(dim=(-3,-2,-1))  # shape: (batch, 256)
```

This embedding captures what the CNN learned about the tumor at a single timepoint.

### Evidence: Why CNN is Necessary

| Paper | Finding | Implication |
|-------|---------|-------------|
| Swin UNETR (Tang et al., 2022) | Swin gets Dice 0.913 vs nnU-Net 0.908 | ViT is only slightly better at segmentation |
| CAFNet (Ahmed et al., 2025) | Pure ViT: 87.3% → hybrid CNN+ViT: 96.4% | CNN still contributes to hybrid approaches |
| TaViT (Hager et al., 2022) | Without time encoding: AUC = 0.50 → with: 0.786 | Temporal modeling is the real value-add, not just ViT |

**Key insight**: ViT's advantage is NOT in segmentation (where CNN and ViT are close). ViT's advantage is in **embedding quality** and **temporal modeling** — things CNN cannot do by design.

---

## Phase 3: Vision Transformer + Temporal Modeling (Weeks 8-11)

### Step 3.1: Swin UNETR Feature Extraction (Week 8)

**Paper**: Swin UNETR (Tang et al., 2022) -- 61.98M params, pre-trained on 5,050 CT + BraTS

```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(img_size=(96,96,96), in_channels=4, out_channels=4)
model.load_state_dict(torch.load("swin_unetr_btcv.pth"))

# Extract 768-dim embedding (encoder only)
features = model.swinViT(x)[-1]  # bottleneck: (batch, 768, 3, 3, 3)
vit_embedding = features.mean(dim=(-3,-2,-1))  # (batch, 768)
```

**Why Swin UNETR, not plain ViT?**
- Plain ViT: flat 16x16 patches, no hierarchy → misses multi-scale tumor features
- Swin UNETR: shifted windows at 4 scales → captures both 5mm micro-lesions and 50mm tumors
- 768-dim output matches RadFM input exactly → no adapter needed

### Step 3.2: ComBat Harmonization (Week 8)

**Why**: Multi-scanner data contains batch effects. Without harmonization, embeddings encode scanner type, not just tumor biology.

**Papers**:
- Generalized ComBat (Horng et al., 2022): Nested ComBat for multiple batch effects
- Longitudinal ComBat (Beer et al., 2020): Preserve within-patient temporal trajectories
- ComBat Guide (Fortin et al., 2022): Feature-space harmonization theory

**Flow**:
```
Raw MRI → Swin UNETR → 768-dim embedding → Nested ComBat → Longitudinal ComBat → Clean embedding
```

**Why harmonize embeddings, not raw images?**
- ViT learns from scanner diversity (data augmentation effect)
- ComBat assumes Gaussian features (embeddings are; raw voxels aren't)
- 768 dims vs 9 million voxels — computationally tractable

### Step 3.3: Temporal Modeling with TaViT (Week 9)

**Paper**: TaViT -- Time-Distance Vision Transformer (Hager et al., 2022)

**The critical problem**: Yale/Cyprus have irregular scan intervals (6 weeks to 12 months between visits). Standard transformers treat all inputs as equidistant → fail on irregular temporal data.

**TaViT solution**: Learnable Temporal Emphasis Model (TEM) scales attention by time distance:
- Recent scans: high attention (TEM ≈ 1.0)
- Distant scans: low attention (TEM ≈ 0.0)

```python
# Input: patient's scan sequence
embeddings = [swin_unetr(scan) for scan in patient_scans]  # list of 768-dim vectors
time_gaps = [0, 42, 90, 180, 365]  # days since baseline

# TaViT adds time-aware positional encoding
temporal_output = tavit(embeddings, time_gaps)  # → 768-dim per visit + CLS summary
```

**Evidence of impact**:

| Model | AUC | Time-aware? |
|-------|-----|-------------|
| Single-scan CNN | 0.734 | No |
| Multi-scan ViT (no time) | 0.500 | No (random chance!) |
| TaViT (with time encoding) | **0.786** | Yes |

Without time encoding, a temporal model is WORSE than a single-scan CNN. TaViT's time awareness is essential.

### Step 3.4: Self-Supervised Pre-training (Week 9-10)

Mask 75% of patient scan sequence → predict masked embeddings from visible ones + time info.

What it learns: tumor growth rates, spread patterns, treatment response signatures — all WITHOUT labeled data.

### Step 3.5: Evaluation + CNN vs ViT Comparison (Week 10-11)

---

## How We Compare CNN vs ViT — Step by Step, with Evidence

This is the heart of Objective 2. The supervisor asks us to **"Compare Transformer-based representations with CNN baselines."** Below is exactly HOW we do that comparison at every level, explained verbally, so you understand the logic behind each experiment.

### Comparison 1: Segmentation — "Which model draws tumor boundaries closer to the expert?"

**Your intuition is exactly right**: If the ViT extracts better features than the CNN, then the segmentation it produces should be closer to the professional expert annotations that we already have from Cyprus.

**What we physically do:**
1. Take the same 171 MRI scans with expert tumor masks
2. Use the same 3-fold data split (same patients in train/test)
3. Train 3D U-Net (CNN) on folds → predict segmentation masks
4. Train Swin UNETR (ViT) on folds → predict segmentation masks
5. Compare BOTH predictions against the expert masks

**What we measure:**
- **Dice score**: overlap between predicted mask and expert mask (0 = no overlap, 1 = perfect)
- **Hausdorff distance 95**: worst-case boundary error in millimeters

**What result means what:**

| Result | What It Tells Us |
|--------|-----------------|
| ViT Dice > CNN Dice | ViT features understand tumor boundaries better (captures fine details the CNN misses) |
| ViT Dice ≈ CNN Dice | Both architectures are equally good at segmentation specifically (but ViT may still win on embedding quality — see Comparison 2) |
| ViT Dice < CNN Dice | Would be surprising; would mean Swin UNETR overfits or needs more data |

**Expected outcome and evidence**: Tang et al. (2022) showed Swin UNETR gets Dice 0.913 vs nnU-Net (CNN) 0.908 on BraTS — only +0.5% better. So we expect them to be **close**. This is important because it shows the ViT advantage is NOT just about segmentation — it's about what happens AFTER segmentation (embedding quality, temporal modeling).

**Why this matters**: Even if segmentation Dice is similar, it does NOT mean the features are equally good. A CNN can produce a good mask by learning local edge patterns, while a ViT produces the same mask by understanding the GLOBAL context of the tumor. The ViT's understanding is richer — and that richness shows up in the embedding quality.

| Metric | CNN (3D U-Net) | ViT (Swin UNETR) |
|--------|---------------|-------------------|
| Dice NCR | \_\_ ± \_\_ | \_\_ ± \_\_ |
| Dice ET | \_\_ ± \_\_ | \_\_ ± \_\_ |
| Dice ED | \_\_ ± \_\_ | \_\_ ± \_\_ |
| Mean Dice | \_\_ ± \_\_ | \_\_ ± \_\_ |
| Hausdorff 95 (mm) | \_\_ ± \_\_ | \_\_ ± \_\_ |

---

### Comparison 2: Embedding Quality — "Which model learns a better UNDERSTANDING of the tumor?"

This is **the most important comparison** and the core scientific claim of our project.

**The idea**: Both models (CNN and ViT) produce a segmentation mask AND internal feature vectors (embeddings). The segmentation mask is the "answer," but the embedding is the "understanding." Two students can get the same exam grade but have very different levels of understanding — and that deeper understanding matters when you face a harder problem.

**What we physically do:**
1. After training both models on segmentation (Comparison 1), we extract embeddings:
   - **CNN**: Take the bottleneck layer → Global Average Pool → 256-dim vector per scan
   - **ViT**: Take the Swin encoder output → Global Average Pool → 768-dim vector per scan
2. We now have 171 embedding vectors from CNN and 171 from ViT (one per timepoint)
3. We test these embeddings on tasks the models were NEVER trained for:

**Test A — Clustering**: "Do embeddings naturally group tumors by type?"
- Take all 171 CNN embeddings → run t-SNE → color by histology (NSCLC vs SCLC vs Breast)
- Take all 171 ViT embeddings → run t-SNE → color by histology
- **If ViT embeddings form tighter, more separated clusters**: the ViT has learned that NSCLC tumors "look" different from Breast tumors at a deep level, even though we never trained it to classify histology
- Measure with **Silhouette score** (higher = better separation)

**Test B — Linear probe**: "Can a simple classifier use the embeddings to predict something new?"
- Take CNN embeddings → train a tiny logistic regression → predict treatment response (progressive vs stable)
- Take ViT embeddings → train the SAME logistic regression → predict treatment response
- **If ViT gives higher F1 score**: the ViT embedding contains more useful information about what will happen to the tumor, even though both models were trained only on segmentation
- We use a LINEAR classifier deliberately: it cannot learn complex patterns. If the information is in the embedding, a linear model finds it. If not, even a perfect classifier cannot extract it.

**Test C — t-SNE visualization**: "What does the embedding space look like?"
- Plot CNN embeddings → do we see meaningful structure?
- Plot ViT embeddings → do we see meaningful structure?
- The ViT plot should show: tumors of the same type cluster together, different treatment responses separate, and IMPORTANTLY, the same patient's timepoints should form a trajectory (nearby in embedding space)

**Why the ViT should win (evidence)**:
- CNN convolution kernels see a 3x3x3 or 5x5x5 local patch — they learn local textures (edges, intensity gradients)
- Swin UNETR's self-attention sees the ENTIRE brain volume at multiple scales — it learns that "this tumor in the frontal lobe has edema pushing against the midline" or "this tumor's enhancing rim is thin and irregular"
- This global context gives richer embeddings that capture tumor morphology AND spatial relationships
- CAFNet (Ahmed et al., 2025) showed: pure ViT achieves 87.3% accuracy, but when you add CNN features to ViT (hybrid), it jumps to 96.4% — proving ViT captures DIFFERENT information than CNN

| Metric | CNN Embedding (256-dim) | ViT Embedding (768-dim) |
|--------|------------------------|------------------------|
| Silhouette score (by histology) | \_\_ | \_\_ |
| Linear probe: response (F1) | \_\_ | \_\_ |
| Linear probe: histology (F1) | \_\_ | \_\_ |
| t-SNE visualization | (figure) | (figure) |

---

### Comparison 3: Temporal Modeling — "Can the model understand how tumors CHANGE over time?"

This is where the **CNN completely fails** and the ViT shines.

**The fundamental limitation of CNN**: A CNN takes ONE MRI volume and produces ONE output. It sees a single snapshot. It's like looking at one photo of a child and trying to guess how fast they're growing — impossible. You need multiple photos taken at different times.

**What we physically do:**
1. For each patient, we have 3-6 MRI scans taken months apart (baseline, follow-up 1, follow-up 2, etc.)
2. **CNN approach**: Extract CNN embedding for each scan independently →   
   `[embedding_baseline, embedding_fu1, embedding_fu2]` — but these embeddings don't know about each other
3. **ViT + TaViT approach**: Feed ALL of a patient's scan embeddings + the actual time gaps (e.g., 42 days, 90 days, 365 days) into TaViT →  
   TaViT produces time-aware embeddings where each embedding KNOWS what came before and after it

**Test A — Temporal coherence**: "Do nearby-in-time embeddings look similar?"
- For the CNN: compute cosine similarity between consecutive embeddings (baseline↔fu1, fu1↔fu2)
- For ViT+TaViT: compute the same
- **If ViT+TaViT similarity is higher and smoother**: the model understands that a tumor doesn't change dramatically between visits 6 weeks apart, but can change greatly over 2 years
- CNN embeddings have NO temporal information, so consecutive embeddings may jump around randomly

**Test B — Delta-embedding vs volume change**: "Does embedding change predict real tumor change?"
- Compute: Δ_embedding = embedding_fu1 - embedding_baseline
- Compute: Δ_volume = tumor_volume_fu1 - tumor_volume_baseline
- Train a regression: Δ_embedding → Δ_volume
- **If R² is high**: the embedding change captures real biological change (tumor growth or shrinkage)
- CNN cannot do this meaningfully because its embeddings from different timepoints are independent

**Test C — Outcome prediction**: "Can the baseline embedding predict what happens in 12 months?"
- Take TaViT's [CLS] token (patient-level summary of entire trajectory)
- Predict: will the tumor progress, stable, or respond?
- Measure AUC
- **CNN has no equivalent** — it can only classify one scan at a time, not predict future from past

**Evidence (Hager et al., 2022)**:

| Model | AUC on temporal prediction |
|-------|---------------------------|
| Single-scan CNN (no temporal info) | 0.734 |
| Multi-scan ViT WITHOUT time encoding | 0.500 (random chance!) |
| Multi-scan ViT WITH TaViT time encoding | **0.786** |

The crucial finding: Without time encoding, even a ViT watching multiple scans is WORSE than a single-scan CNN. This proves that time-awareness is not optional — it's essential.

---

### Comparison 4: Downstream Impact — "Do better features produce better clinical outputs?"

This comparison proves that the feature quality difference affects the REAL outputs of our system (clinical narratives and videos).

**What we physically do:**

**For Phase 4 (LLM narratives)**:
1. Take 10 test patients
2. Run A: Feed CNN embeddings → RadFM → generate clinical narrative
3. Run B: Feed ViT embeddings → RadFM → generate clinical narrative
4. Compare both narratives against a reference clinical report
5. **If ViT narrative is more accurate and specific**: the ViT embedding gave the LLM more information to work with

Example of what we expect to see:
```
CNN embedding → RadFM output:
  "The patient has a brain tumor. There is some change over time."
  (vague, because CNN embedding only captures local texture)

ViT embedding → RadFM output:  
  "15mm enhancing lesion in right frontal lobe showing 20% volume increase
   over 6 months, with new peritumoral edema suggesting progression."
  (specific, because ViT embedding captures size, location, context, AND temporal trends)
```

**For Phase 5 (Video generation)**:
1. Generate future MRI scan conditioned on CNN embeddings → measure SSIM and tumor Dice
2. Generate future MRI scan conditioned on ViT embeddings → measure SSIM and tumor Dice
3. **If ViT-conditioned generation has higher SSIM and Dice**: the ViT embedding provides better conditioning, making the generated scan more realistic and anatomically correct

**For counterfactual analysis (ONLY ViT)**:
- Counterfactuals require temporal understanding: "this patient has been on chemotherapy for 6 months, what if we switch to radiation?"
- CNN cannot represent "6 months of chemotherapy" — it only sees one scan
- ViT + TaViT encodes the FULL treatment history + temporal trajectory → meaningful counterfactuals

| Downstream Task | CNN → Output Quality | ViT → Output Quality | ViT Advantage |
|-----------------|--------------------|--------------------|---------------|
| LLM narrative (BLEU) | \_\_ | \_\_ | More specific, temporally coherent |
| Video generation (SSIM) | \_\_ | \_\_ | More anatomically realistic |
| Counterfactual (Dice) | Cannot do | \_\_ | Only ViT can model "what if" |

---

### Summary: The Three Roles of ViT in Our Project

```
Role 1: SEGMENTATION
─────────────────────
CNN: draws tumor boundary using local patterns (edges, textures)
ViT: draws tumor boundary using global context (whole-brain understanding)
How we prove it: Compare Dice scores against expert masks
Expected: Close (~0.5% ViT advantage). This alone does NOT justify ViT.

Role 2: FEATURE EXTRACTION (for LLM + Video)  ← THIS IS THE MAIN POINT
────────────────────────────────────────────
CNN: produces 256-dim embedding capturing local texture only
ViT: produces 768-dim embedding capturing morphology + spatial context + heterogeneity
How we prove it: Clustering, linear probes, t-SNE on tasks NEVER seen during training
Expected: ViT embeddings cluster better, predict better, carry more information
Why it matters: These embeddings are the INPUT to Phase 4 (LLM) and Phase 5 (Video).
               Better features in → better outputs out. Garbage in, garbage out.

Role 3: TEMPORAL EVOLUTION  ← ONLY ViT CAN DO THIS
────────────────────────────
CNN: single snapshot, no concept of time
ViT + TaViT: models how embeddings change over time with irregular intervals
How we prove it: Temporal coherence, delta-prediction, outcome forecasting
Expected: AUC 0.786 (TaViT) vs 0.734 (single CNN) vs 0.500 (ViT without time)
Why it matters: Temporal understanding is REQUIRED for:
  - LLM to describe progression ("tumor grew over 6 months")
  - Video to show realistic evolution
  - Counterfactuals ("what if different treatment?")
```

This is why Phase 2 (CNN baseline) exists: **without it, we cannot claim any of the above.** The CNN is the control experiment that makes the ViT results scientifically meaningful.

---

## Complete Embedding Quality Test Battery (17 Tests)

To prove the supervisor's requirement: *"Learn embeddings that capture tumor morphology, spatial heterogeneity, and temporal evolution"*, we run the following tests on BOTH CNN and ViT embeddings. All tests use our Cyprus data as ground truth and cost **0 GPU time** (run on CPU after embedding extraction).

### 🔬 Morphology Tests (M1-M6) — Shape, Size, Structure

Ground truth: 14 shape radiomic features + 171 tumor masks (84 with NCR, 87 without)

| # | Test | Target | Pass Criterion |
| - | ---- | ------ | -------------- |
| M1 | Volume prediction | Mask volume (mm³) | R² > 0.6 |
| M2 | Sphericity prediction | `shape_Sphericity` | R² > 0.6 |
| M3 | Surface-volume ratio | `shape_SurfaceVolumeRatio` | R² > 0.6 |
| M4 | Necrosis detection | NCR present (84) vs absent (87) | F1 > 0.7 |
| M5 | Elongation prediction | `shape_Elongation` | R² > 0.5 |
| M6 | NN morphology consistency | 5-NN volume agreement (±30%) | >60% |

### 🗺️ Spatial Heterogeneity Tests (H1-H5) — Internal Texture

Ground truth: 1,824 GLCM + 1,064 GLDM features

| # | Test | Target | Pass Criterion |
| - | ---- | ------ | -------------- |
| H1 | GLCM entropy correlation | PCA vs `glcm_Contrast` | r > 0.5 |
| H2 | Heterogeneity probe | `glcm_DifferenceEntropy` | R² > 0.5 |
| H3 | Multi-label detection | {1,2,3} subregions present | F1 > 0.6 |
| H4 | Texture feature bundle | [Contrast, RunLengthVar, ZoneEntropy] | avg R² > 0.4 |
| H5 | Attention maps | Highlight heterogeneous regions | ⚠️ Phase 3 only |

### ⏱️ Temporal Evolution Tests (T1-T7) — Change Over Time

Ground truth: 45 patients × 3-7 timepoints (mean 5.2), timestamps, volume changes

| # | Test | Target | Pass Criterion |
| - | ---- | ------ | -------------- |
| T1 | Embedding dist vs ΔVolume | Pearson r | r > 0.4 |
| T2 | Temporal t-SNE trajectories | Visual smoothness | Qualitative |
| T3 | ΔEmbedding → ΔVolume | Regression R² | R² > 0.3 |
| T4 | Response classification | AUC (n=29 at 6 months) | AUC > 0.6 |
| T5 | Temporal coherence | Consecutive cosine similarity | > 0.85 |
| T6 | Embedding velocity vs progression | Rate vs clinical speed | r > 0.3 |
| T7 | Treatment group separation | RS vs FSRT (36 vs 11) | Cohen's d > 0.3 |

### Time Impact

| What | GPU Time | CPU Time |
| ---- | -------- | -------- |
| Embedding extraction (171 scans) | ~1 hr (already in plan) | 0 |
| All 17 tests | **0** | **~60 min** |
| Figures and tables | 0 | ~30 min |
| **Total added time** | **0** | **~90 min** |

The test battery adds zero GPU cost — all tests run on the extracted embeddings using scikit-learn on CPU.

---

# OBJECTIVE 3 -- Integrate Imaging and Clinical Context Using LLMs

> *"Enable multimodal clinical reasoning by combining visual tumor representations with structured clinical metadata using LLMs."*

## Phase 4: Multimodal Integration (Weeks 12-14)

### Step 4.1: Design Multimodal Representations (Week 12)

**Inputs to fuse**:
1. **Imaging**: Swin UNETR 768-dim embeddings per scan + TaViT temporal summary
2. **Clinical metadata**: Age, sex, histology, treatment type, KPS, lesion location, RT dose

### Step 4.2: Vision-to-Language Bridge (Week 13)

**Paper**: RadFM (Wu et al., Nature Communications 2025)

```
Swin UNETR (768-dim) → Perceiver (32 tokens per scan) → MedLLaMA-13B → Clinical narrative
```

**Key**: Our Swin UNETR outputs 768-dim, which matches RadFM's ViT output exactly. No adapter needed.

**Novel contribution**: No existing paper feeds temporal 3D MRI sequences → LLM for longitudinal progression reports. We extend RadFM's single-scan input to multi-visit sequences:
```
[VISIT_1] 32 tokens [TIME: +6 months] [VISIT_2] 32 tokens → LLM generates temporal narrative
```

### Step 4.3: Embedding Alignment (Week 13)

**Paper**: MM-Embed (Lin et al., ICLR 2025) -- contrastive alignment for multimodal embeddings.

Training awareness: watch for text bias (LLM ignoring visual features), use curriculum learning, use hard negatives.

### Step 4.4: Train and Validate (Week 14)

**Training**: LoRA fine-tuning on MedLLaMA-13B (~4M trainable params)  
**Compute**: 1x A100 40-80GB  
**Validation**: BLEU, ROUGE, UMLS precision/recall, expert review

### Phase 4 Deliverables

- Multimodal embedding representation (imaging + clinical)
- LLM prompt templates for brain tumor progression
- Generated clinical narratives
- Qualitative validation of plausibility

---

# OBJECTIVE 4 -- Generate and Analyze Cancer Progression Videos

> *"Model and visualize cancer progression through AI-generated video sequences conditioned on multimodal inputs."*

## Phase 5: Video Generation (Weeks 15-18)

### Step 5.1: Latent Diffusion Foundation (Week 15)

**Paper**: LDM (Rombach et al., CVPR 2022)

```
3D VAE Encoder → Latent space (512x compression) → Diffusion UNet → 3D VAE Decoder → Generated MRI
```

Why latent, not pixel: 240x240x155 = 9 million voxels per scan → computationally impossible.  
Latent: 30x30x20 = 18,000 → tractable.

### Step 5.2: Treatment-Conditioned Generation (Weeks 16-17)

**Paper**: TaDiff (Liu et al., IEEE-TMI 2025) -- treatment-aware diffusion for brain tumors

**Conditioning inputs**:
1. **Patient history**: Swin UNETR embeddings of past visits (via cross-attention)
2. **Treatment type**: Chemotherapy, radiation, surgery → embedding table → MLP
3. **Time gap**: Days since last scan → sinusoidal encoding → MLP
4. **TaViT summary**: 768-dim temporal trajectory representation

**Key innovation**: Joint generation + segmentation  
Model predicts noise AND tumor mask simultaneously → anatomically correct tumors.

**Counterfactual mode**: Same patient, different treatment vectors:
- "What if surgery?" → generate post-surgical progression
- "What if radiation?" → generate radiation response
- "What if no treatment?" → generate natural growth baseline

**Evidence**: TaDiff results: SSIM 0.919, PSNR 27.97 dB. Without treatment conditioning: SSIM drops to 0.882, Dice drops from 0.719 to 0.556.

### Step 5.3: Video Generation (Week 17-18)

**Paper**: EchoNet-Synthetic (Reynaud et al., MICCAI 2024)

3-stage pipeline: VAE → LIDM (frame generation) → LVDM (temporal consistency)

**Our adaptation**: Replace regular frame intervals (30fps) with irregular time gaps using TaViT's time-distance encoding in the temporal attention layers.

### Phase 5 Deliverables

- 3D VAE for brain MRI encoding/decoding
- Treatment-conditioned future scan generation
- Counterfactual trajectory visualizations
- Quantitative metrics (FID, SSIM, PSNR, tumor DSC)

---

# OBJECTIVE 5 -- Evaluate Explainability, Clinical Plausibility, and Scientific Impact

## Phase 6: Evaluation and Final Deliverables (Weeks 19-20)

### Quantitative Evaluation

| Objective | Metric | Target | Evidence |
|-----------|--------|--------|----------|
| Segmentation (Obj 1) | Dice score | >0.85 | nnU-Net: 0.908, Swin UNETR: 0.913 |
| Representation (Obj 2) | AUC, clustering | AUC >0.75 | TaViT: 0.786 |
| Harmonization | Silhouette by scanner | ~0 | ComBat validation (Moyer 2022) |
| LLM narratives (Obj 3) | BLEU, ROUGE, UMLS | BLEU >0.3 | RadFM benchmarks |
| Video quality (Obj 4) | SSIM, PSNR, FID | SSIM >0.85 | TaDiff: 0.919 |
| Tumor fidelity (Obj 4) | Tumor DSC | >0.65 | TaDiff: 0.719 |
| Clinical realism | Expert rating | >3.0/5 | MedEdit: 3.20/5 |

### Qualitative Evaluation

- Grad-CAM on Swin UNETR → heatmaps showing which brain regions drive predictions (TransXAI, Zeineldin 2024)
- Expert review of LLM narratives (medical accuracy)
- Blind assessment of generated videos (1-5 scale)
- Counterfactual plausibility (radiation → shrinkage, no treatment → growth)

### Final Deliverables

- Fully implemented multimodal AI framework
- Explainable cancer progression and counterfactual videos
- Final technical report
- Publication-ready research draft

---

## Complete Paper Reference (22 Papers)

### Objective 1 -- Data Pipeline (5 papers)

| # | Paper | Year | Role |
|---|-------|------|------|
| 01 | Cyprus PROTEAS Dataset (Flouri et al.) | 2025 | Primary dataset: 40 patients, expert labels |
| 02 | Yale Glioma Dataset (Ramakrishnan et al.) | 2025 | Scale-up dataset: 1,430 patients |
| 03 | BraTS Toolkit (Kofler et al.) | 2020 | HD-BET skull stripping + normalization |
| 04 | nnU-Net (Isensee et al.) | 2021 | Automated tumor segmentation (Dice 0.908) |
| 05 | itk-elastix (Niessen et al.) | 2023 | Longitudinal registration |

### Objective 2 -- ViT Representations (8 papers)

| # | Paper | Year | Role |
|---|-------|------|------|
| 06 | Swin UNETR (Tang et al.) | 2022 | 768-dim feature extractor (Dice 0.913) |
| 07 | TaViT (Hager et al.) | 2022 | Time-distance temporal encoding (AUC 0.786) |
| 08 | Generalized ComBat (Horng et al.) | 2022 | Nested multi-batch harmonization |
| 09 | Longitudinal ComBat (Beer et al.) | 2020 | Preserve temporal trajectories |
| 10 | ComBat Guide (Fortin et al.) | 2022 | Feature-space harmonization theory |
| 11 | TransXAI (Zeineldin et al.) | 2024 | Grad-CAM explainability for ViT |
| 12 | CAFNet (Ahmed et al.) | 2025 | Validates hybrid CNN+ViT design |
| -- | 3D U-Net (Cicek et al.) | 2016 | CNN baseline architecture (textbook) |

### Objective 3 -- LLM Integration (2 papers)

| # | Paper | Year | Role |
|---|-------|------|------|
| 13 | RadFM (Wu et al.) | 2025 | Perceiver + MedLLaMA-13B (Nature Comms) |
| 14 | MM-Embed (Lin et al.) | 2025 | Contrastive embedding alignment (ICLR) |

### Objective 4 -- Video Generation (5 papers)

| # | Paper | Year | Role |
|---|-------|------|------|
| 15 | DDPM (Ho et al.) | 2020 | Diffusion theory foundation |
| 16 | LDM (Rombach et al.) | 2022 | Latent diffusion + cross-attention conditioning |
| 17 | Video LDM (Blattmann et al.) | 2023 | Temporal layer insertion strategy |
| 18 | TaDiff (Liu et al.) | 2025 | Treatment-conditioned brain MRI generation |
| 19 | EchoNet-Synthetic (Reynaud et al.) | 2024 | Medical video diffusion (starting codebase) |

### Additional (3 papers)

| # | Paper | Year | Role |
|---|-------|------|------|
| 20 | MedEdit (Ben Alaya et al.) | 2024 | Counterfactual brain MRI editing |
| 21 | Counterfactual Diff. AE (Atad et al.) | 2024 | Latent counterfactual manipulation |
| 22 | ComBat Validation (Moyer et al.) | 2022 | "Test before harmonizing" methodology |

---

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Primary dataset | Cyprus PROTEAS | Expert labels, rich metadata, manageable size |
| Skull stripping | HD-BET | Best for tumor-distorted brains |
| Segmentation (label generation) | nnU-Net | Self-configuring, Dice 0.908 |
| Registration | itk-elastix | Python, pip-install, brain-validated |
| **CNN baseline** | **3D U-Net** | **Standard, fair comparison for ViT** |
| **Feature extractor** | **Swin UNETR** | **768-dim, hierarchical, pre-trained, MONAI built-in** |
| Time encoding | TaViT | Without it: AUC = 0.50 (random chance) |
| Harmonization target | Embeddings (not raw images) | ViT learns from diversity; ComBat cleans features |
| Multi-batch harmonization | Nested ComBat | Multiple batch effects require sequential correction |
| Temporal harmonization | Longitudinal ComBat | Preserves within-patient trajectories |
| Vision→Language bridge | RadFM Perceiver (32 tokens) | 768-dim match, no adapter needed |
| LLM backbone | MedLLaMA-13B + LoRA | Open-source, medical-tuned, memory-efficient |
| Image generation | Latent Diffusion | 512x compression makes 3D volumes tractable |
| Treatment conditioning | TaDiff | +3.7% SSIM, +16.3% DSC with treatment info |
| Video generation | EchoNet-Synthetic pipeline | Full code + weights, 3-stage VAE→LIDM→LVDM |
| Explainability | Grad-CAM on Swin UNETR | Post-hoc, no architecture modification |

---

## Timeline Summary

| Week | Phase | Obj | Activity | CNN/ViT |
|------|-------|-----|----------|---------|
| 1 | Phase 1 | 1 | Download, preprocess (HD-BET, normalize) | -- |
| 2 | Phase 1 | 1 | Segmentation verification (nnU-Net for Yale) | -- |
| 3 | Phase 1 | 1 | Longitudinal alignment (itk-elastix) | -- |
| 4 | Phase 1 | 1 | EDA + clinical data cleaning + reports | -- |
| 5 | Phase 2 | 2 | **3D U-Net baseline: data splits, pipeline, quick test** | **CNN** |
| 6 | Phase 2 | 2 | **3D U-Net training (3-fold CV)** | **CNN** |
| 7 | Phase 2 | 2 | **CNN evaluation + embedding extraction + limitations** | **CNN** |
| 8 | Phase 3 | 2 | **Swin UNETR embeddings + ComBat harmonization** | **ViT** |
| 9 | Phase 3 | 2 | **TaViT temporal modeling + self-supervised pretraining** | **ViT** |
| 10 | Phase 3 | 2 | CNN vs ViT comparison at all levels | Both |
| 11 | Phase 3 | 2 | Representation evaluation + Phase 2-3 report | Both |
| 12 | Phase 4 | 3 | Multimodal representation design | -- |
| 13 | Phase 4 | 3 | RadFM + MM-Embed pipeline | -- |
| 14 | Phase 4 | 3 | LLM training + narrative generation | -- |
| 15 | Phase 5 | 4 | Adapt video diffusion model (3D VAE + LDM) | -- |
| 16-17 | Phase 5 | 4 | Generate progression + counterfactual videos | -- |
| 18 | Phase 5 | 4 | Train + validate on patient sequences | -- |
| 19 | Phase 6 | 5 | Quantitative + qualitative evaluation | -- |
| 20 | Phase 6 | 5 | Final report + publication draft | -- |
