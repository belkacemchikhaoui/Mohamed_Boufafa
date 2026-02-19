# Master Research Plan — Explainable Disease Progression & Counterfactual Video Generation

> **Intern**: Mohamed | **Supervisor**: Dr. Belkacem Chikhaoui | **Institution**: TÉLUQ University, Montreal  
> **Duration**: 20 weeks (April–August 2026) | **Program**: Mitacs Globalink Research Internship

---

## Overall Objective

Design, implement, and evaluate a multimodal AI framework that integrates Vision Transformers (ViTs), Large Language Models (LLMs), and generative video models to **analyze, explain, and simulate cancer progression** from longitudinal brain MRI data.

---

## Datasets

### Yale Longitudinal Brain Metastases (Primary — Training)

| Property | Detail |
|---|---|
| Scans | 11,884 brain MRIs |
| Patients | 1,430 |
| Avg scans/patient | ~8 (longitudinal) |
| Period | 2004–2023 (20 years) |
| Modalities | T1, T1c, T2, FLAIR |
| Labels | None (generate with nnU-Net) |
| Treatment info | Pre/post surgery, radiation, chemo |
| Access | TCIA (free, public) |
| Challenge | Multiple scanners → needs harmonization |

### Cyprus Brain Metastases (Secondary — External Validation, Optional)

| Property | Detail |
|---|---|
| Scans | 744 brain MRIs |
| Patients | 40 |
| Avg scans/patient | ~18.6 |
| Period | 2019–2024 |
| Labels | Expert radiologist verified (3 sub-regions) |
| Access | Zenodo (free, public) |
| Format | Already BraTS-compatible |
| Purpose | Independent validation on Mediterranean population |

**Dataset strategy** (decide with supervisor after Phase 1):
- **Option A (recommended)**: Train on Yale → test on Cyprus = proves cross-population generalization
- **Option B**: Yale only = simpler but no external validation
- **NOT recommended**: Mixing Yale + Cyprus in training = loses independent validation

---

## Structure: 5 Objectives × 6 Phases

| Objective | Phase(s) | Weeks | Core Task |
|---|---|---|---|
| **Obj 1** — Robust Imaging Pipeline | Phase 1 | 1–4 | Preprocessing, segmentation, alignment |
| **Obj 2** — ViT Tumor Representations | Phase 2 + Phase 3 | 5–11 | CNN baseline → ViT embeddings + temporal modeling |
| **Obj 3** — LLM Clinical Reasoning | Phase 4 | 12–14 | Explainability + LLM-generated reports |
| **Obj 4** — Progression Video Generation | Phase 5 | 15–18 | Diffusion-based counterfactual videos |
| **Obj 5** — Evaluation & Scientific Impact | Phase 6 | 19–20 | Quantitative + qualitative validation |

---

## Objective 1 — Robust Pipeline for Longitudinal Cancer Imaging Analysis

> **Goal**: Build a reproducible pipeline that takes raw multi-scanner brain MRIs and outputs clean, segmented, temporally aligned longitudinal sequences ready for deep learning.

### Phase 1 — Oncology Medical Imaging Preparation (Weeks 1–4)

#### Step 1.1: Download & Preprocess (Week 1)

**Papers used**:
- **Paper 1 (BraTS Toolkit)**: Preprocessing pipeline for brain tumor MRI
  - **What we take**: HD-BET for skull stripping, z-score normalization per modality
  - **How to adapt**: Apply to Yale's 4 modalities (T1, T1c, T2, FLAIR)
  - **See**: `objectiveone/SIMPLE_Paper1_BraTS_Toolkit.md`
- **Paper 3 (Yale Dataset)**: Our primary training data source
  - **What we take**: 11,884 scans, 1,430 patients, treatment metadata
  - **Challenge**: No tumor labels → generate with nnU-Net
  - **See**: `objectiveone/SIMPLE_Paper3_Yale_Dataset.md`
- **Paper 7 (Cyprus Dataset)**: Optional external validation
  - **What we take**: 744 expert-labeled scans, 40 patients
  - **How to adapt**: Already BraTS-formatted, use for validation only
  - **See**: `objectiveone/SIMPLE_Paper7_Cyprus.md`

**Tasks**:
1. Download Yale from TCIA (~200 GB)
2. (Optional) Download Cyprus from Zenodo (~100 GB)
3. Convert to NIfTI standard format
4. Skull stripping with HD-BET (from Paper 1)
5. Intensity normalization: z-score per modality (from Paper 1)
6. Spatial resampling to uniform resolution (128×128×128 for Swin UNETR)

**Output**: 11,884 Yale scans (+ 744 Cyprus) in standardized NIfTI format, skull-stripped, normalized.

**Tools**: `HD-BET`, `BraTS Toolkit`, `SimpleITK`, `MONAI`

---

#### Step 1.2: Tumor Segmentation (Week 2)

**Papers used**:
- **Paper 2 (nnU-Net)**: Self-configuring segmentation baseline
  - **What we take**: Automated tumor segmentation for 3 sub-regions (ET, WT, TC)
  - **How to adapt**: Use BraTS pretrained weights, fine-tune on Yale if needed
  - **Performance**: Dice ~0.908 on brain tumors
  - **See**: `objectiveone/SIMPLE_Paper2_nnUNet.md`
- **Paper 7 (Cyprus Dataset)**: Validation reference
  - **What we take**: Expert-verified labels to test nnU-Net accuracy
  - **How to use**: Run nnU-Net on Cyprus → compare to expert labels → know your error rate

**Tasks**:
1. Run nnU-Net on all 11,884 Yale scans → generate tumor masks (ET, WT, TC)
2. Quality check: manually review 50–100 random cases
3. (Optional) Run nnU-Net on Cyprus → compare to expert labels → measure segmentation accuracy

**Output**: Tumor segmentation masks for all scans.

**Note**: Swin UNETR (Phase 3, Step 3.1) achieves Dice 0.913 > nnU-Net 0.908. This is a **bonus side effect** — when you train Swin UNETR for embeddings (Step 3.1), its decoder also outputs segmentation masks. You can optionally replace nnU-Net masks with Swin UNETR masks later, but for Phase 1, nnU-Net masks are sufficient.

---

#### Step 1.3: Temporal Registration (Week 3)

**Method: itk-elastix** — THE gold standard for brain tumor longitudinal registration

**Why itk-elastix is the clear winner**:
1. ✅ **Available NOW**: `pip install itk-elastix` → working in 30 seconds
2. ✅ **Proven for brain tumors**: 100+ papers, BraTS standard, Yale validated (0.95 correlation)
3. ✅ **Multi-parametric native**: Mutual Information metric handles T1/T1c/T2/FLAIR out-of-box
4. ✅ **Python ecosystem**: Seamless NumPy/PyTorch/MONAI/Swin UNETR integration
5. ✅ **Ready parameter files**: Brain tumor configs from model zoo (BraTS-optimized)
6. ✅ **Production-ready**: 15 years mature, GitHub forum, 100+ Jupyter examples, hundreds of CI tests
7. ✅ **Scalable**: 14.5 min/case × 11,884 scans = ~2,900 GPU-hours (manageable on cluster)

**FLIRE dismissed**: Code NOT released (can't use!), only breast MRI (no brain validation), MATLAB-only (no Python integration), no community. Speed advantage (4.5 min/case) irrelevant when FLIRE is unusable.

**Implementation** (Python API):
```python
import itk
from monai.data import itk_torch_bridge

# 1. Load baseline and moving scans
baseline = itk.imread('patient_T0_FLAIR.nii.gz', itk.F)
moving = itk.imread('patient_T1_FLAIR.nii.gz', itk.F)

# 2. Load brain tumor parameter file from model zoo
params = itk.ParameterObject.New()
params.ReadParameterFile('BraTS_Bspline_4modalities.txt')  # Brain-optimized!

# 3. Register (one line!)
registered, transform = itk.elastix_registration_method(
    baseline, moving, parameter_object=params
)

# 4. Apply same transform to tumor mask (nearest neighbor for labels)
transform.SetParameter(0, 'ResampleInterpolator', 
                       'FinalNearestNeighborInterpolator')
mask_moving = itk.imread('patient_T1_tumor_mask.nii.gz', itk.UC)
mask_registered = itk.transformix_filter(mask_moving, transform)

# 5. Convert to MONAI MetaTensor for Swin UNETR
metatensor = itk_torch_bridge.itk_image_to_metatensor(registered)
```

**Multi-stage registration** (handled automatically by parameter file):
- Stage 1: Rigid (6 params) → fix patient positioning
- Stage 2: Affine (12 params) → global brain shape
- Stage 3: B-spline deformable (1000s of params) → tumor growth/edema/atrophy

**Tasks**:
1. For each patient, designate baseline scan (T0 — earliest timepoint)
2. Register all subsequent scans to T0: T1→T0, T2→T0, ..., T7→T0
3. Apply same transform to all 4 modalities (T1, T1c, T2, FLAIR) and tumor masks
4. Validate on 50-100 patients: correlation >0.90, tumor Dice >0.85, Jacobian stable
5. Batch process all 11,884 scans in parallel on cluster

**Resources**:
- Jupyter examples: https://github.com/InsightSoftwareConsortium/ITKElastix/tree/main/examples
- Parameter file zoo: https://elastix.lumc.nl/modelzoo/ (filter: brain + MRI + 3D)
- See: `objectiveone/SIMPLE_Paper_ITK_Elastix.md` for full analysis

**Output**: All patient timepoints spatially aligned to their baseline scan.

---

#### Step 1.4: Exploratory Data Analysis (Week 4)

**Tasks**:
1. Distribution of tumor sizes, locations, and counts per patient
2. Inter-patient variability analysis
3. Temporal variability: scan intervals, number of timepoints per patient
4. Scanner/protocol diversity analysis (manufacturers, field strengths, years)
5. Document data quality issues and limitations

**Output**: EDA report with statistics and visualizations.

---

#### ✅ Phase 1 Checkpoint (End of Week 4)
- [ ] 1,430 Yale patients with ~8 clean, skull-stripped, aligned scans each
- [ ] Tumor segmentation masks for all 11,884 scans
- [ ] (Optional) Cyprus 40 patients preprocessed as independent test set
- [ ] EDA report documenting dataset characteristics
- [ ] **Ready for Objective 2: ViT feature extraction**

---

## Objective 2 — Learn High-Level Tumor Representations Using Vision Transformers

> **Goal**: Extract semantically rich, temporally informative tumor embeddings using ViT architectures, compare to CNN baselines, and model temporal tumor evolution.

### Phase 2 — Baseline Vision Models (Weeks 5–7)

#### Step 2.1: CNN Baseline Implementation (Week 5–6)

**Tasks**:
1. Implement CNN baseline (e.g., 3D ResNet, nnU-Net features) for single-timepoint tumor analysis
2. Extract CNN feature representations from tumor regions
3. Train on classification/regression tasks (tumor growth, treatment response)
4. Establish baseline performance metrics

**Output**: CNN baseline accuracy, feature representations, trained weights.

**Purpose**: Reference benchmark — ViT must outperform this.

---

#### Step 2.2: Baseline Evaluation (Week 7)

**Tasks**:
1. Evaluate CNN on downstream prediction tasks (growth classification, response prediction)
2. Identify limitations of static single-timepoint modeling
3. Document where temporal information is needed

**Output**: Baseline performance report documenting static model limitations.

---

#### ✅ Phase 2 Checkpoint (End of Week 7)
- [ ] CNN baseline trained and evaluated
- [ ] Performance benchmarks established
- [ ] Static modeling limitations documented
- [ ] **Ready for Phase 3: ViT temporal modeling**

---

### Phase 3 — Vision Transformer-Based Longitudinal Representation Learning (Weeks 8–11)

**Papers analyzed**:
- Paper 11: Time-Distance ViT / TaViT → `objectivetwo/SIMPLE_Paper11_Time_Distance_ViT.md`
- Paper 13: Swin UNETR → `objectivetwo/SIMPLE_Paper13_Swin_UNETR.md`
- TransXAI: Explainable Transformer → `objectivetwo/SIMPLE_Paper_TransXAI.md`
- CAFNet: CNN+ViT Cross-Attention → `objectivetwo/SIMPLE_Paper_CAFNet.md`

**Architecture validation** (from CAFNet):
> Pure ViT: 87.34% → CNN+ViT with cross-attention: 96.41% (+9%).  
> Our Swin UNETR is inherently a hybrid (Transformer encoder + CNN decoder) — validated design choice.

#### Step 3.1: Swin UNETR Embedding Extraction (Week 8)

**Architecture** (from Paper 13 — Swin UNETR, 61.98M params):

**Input**: 128×128×128×4 volumes (T1, T1c, T2, FLAIR)
**Encoder**: 4-stage hierarchical Swin Transformer with [48, 96, 192, 384, 768] channels
**Bottleneck**: 4×4×4×768 → Global Average Pool → **768-dim embedding per scan**
**Pretrained weights**: NVIDIA model trained on 5,050 CT volumes + BraTS fine-tuning
**Tools**: MONAI library, pretrained model: `model_swinvit.pt`

**Pipeline**:
```
Phase 1 scans (clean + segmented + aligned)
    ↓
Swin UNETR encoder → 768-dim embedding per scan
    ↓
Per patient: 8 embeddings × 768 dims = (8, 768) matrix
    ↓
[HARMONIZATION STARTS HERE — 3-step process]
    ↓
Step 1: ComBat Validation (Paper 10++)
    ↓
Step 2: Nested ComBat (Paper 10+)
    ↓
Step 3: Longitudinal ComBat (Paper 10)
    ↓
Harmonized embeddings → ready for temporal modeling
```

**WHY HARMONIZATION IS ESSENTIAL** (The Core Problem):

Yale has 11,884 scans from 2004–2023 with different scanners. **Without harmonization**:

❌ **Problem 1 — Scanner bias**: A 10mm tumor on Siemens 3T looks different from same 10mm tumor on GE 1.5T
- Swin UNETR embeddings will encode scanner type, not just tumor biology
- Model learns: "Siemens scans = aggressive tumors" (because newer scanners used for sicker patients)
- **Result**: Wrong predictions on different scanners

❌ **Problem 2 — Time bias**: 2004 protocols differ from 2023 protocols
- Model thinks tumors were "different" in 2004 vs 2023 (protocol changed, not biology)
- **Result**: Can't track real progression over 20 years

❌ **Problem 3 — Temporal tracking breaks**: Patient scanned on Scanner A (2015) then Scanner B (2018)
- Model sees huge embedding change — but is it tumor growth OR scanner change?
- **Result**: Can't tell real progression from technical artifacts

**HOW THIS BREAKS FUTURE OBJECTIVES**:

**Objective 3 (LLM explanations)**:
- LLM generates: "Tumor grew 40% between T0 and T1"
- Reality: Tumor grew 10%, scanner difference caused other 30%
- **Clinicians lose trust** — AI gave wrong information!

**Objective 4 (Video generation)**:
- Diffusion model learns scanner artifacts as "progression patterns"
- Generated videos show scanner-switch jumps, not smooth tumor growth
- **Videos look fake** — can't use clinically

**Objective 5 (Evaluation)**:
- External validation on Cyprus fails — different scanners than Yale
- Model can't generalize because it learned scanner patterns, not tumor patterns
- **Paper gets rejected** — reviewers say "not clinically useful"

---

**HARMONIZATION DETAILS** (3 ComBat papers, applied sequentially):

**Step 1 — ComBat Validation (Paper 10++)**: Test if scanner effects exist
- **What we take**: Statistical test for batch effects in embeddings
- **How to adapt**: Test Yale embeddings for effects from scanner/year/field strength
- **Decision**: If no effects (p > 0.05) → skip. If effects found → proceed to Step 2
- **See**: `objectiveone/SIMPLE_Paper10++_ComBat_Validation.md`

**Step 2 — Nested ComBat (Paper 10+)**: Remove multi-batch effects
- **What we take**: Generalized ComBat for nested batches
- **Yale's 3 nested batches**:
  - Scanner manufacturer (GE, Siemens, Philips)
**Temporal Emphasis Model (TEM)** — learnable sigmoid that scales attention by time distance:

**Formula**: TEM(R) = 1 / (1 + exp(a × R - c))
- **R** = time distance between two scans (in days)
- **a, c** = LEARNABLE parameters (model learns optimal values during training)
  - **a**: controls how fast attention drops with time distance
  - **c**: controls where the transition happens
- **Effect**: Recent scan pairs get high attention (TEM ≈ 1.0), distant pairs get low attention (TEM ≈ 0.0)
- **Applied to**: Self-attention mechanism — scales attention weights by time distance

**Full architecture**:
- **Input**: 8 timepoints × 768 Swin UNETR embeddings + time distances between scans
- **Projection**: Linear layer maps 768 dims to model dimension (if needed)
- **CLS token**: Added to beginning of sequence (standard Transformer approach)
- **TaViT attention**: 8 layers, 8 heads, with time-distance scaling
- **Output**: CLS token → 768-dim temporal representation (summarizes all 8 timepoints)onus**: Swin UNETR Dice 0.913 > nnU-Net 0.908 — when you train Swin UNETR for embeddings, its decoder also outputs better segmentation masks. You DON'T segment twice; you can optionally upgrade Phase 1 nnU-Net masks with Swin UNETR masks later.

---

#### Step 3.2: Temporal Attention with TaViT (Week 9)

**Paper used**:
- **Paper 11 (Time-Distance ViT / TaViT)**: Temporal emphasis for irregular scan intervals
  - **What we take**: Learnable sigmoid function that scales attention by time distance
  - **How to adapt**: 
    - Input: 8 Yale timepoints × 768 Swin UNETR embeddings (not Paper 11's patches)
    - Keep: TEM formula, masked autoencoder pretraining strategy
    - Change: Input dimensionality (Paper 11 used image patches, we use Swin embeddings)
  - **Why essential**: Yale has irregular intervals. Paper 11 proved: no time encoding → 0.50 AUC (random chance!)
  - **Performance**: TaViT 0.786 AUC > TeViT 0.785 > Single-scan CNN 0.734
  - **See**: `objectivetwo/SIMPLE_Paper11_Time_Distance_ViT.md`

**Architecture** (from Paper 11 — TaViT):

**Temporal Emphasis Model (TEM)** — learnable sigmoid that scales attention by time distance:
```
For scan pair (i, j):
  R_i,j = |time_i - time_j| in days
  TEM(R) = 1 / (1 + exp(a × R - c))

  a, c are LEARNABLE parameters:
  → a: controls decay speed with time distance
  → c: controls transition point
  
  Applied to self-attention:
  Attention_temporal = softmax(Q × K^T / √d) × TEM(R)
  
  Recent scans: TEM ≈ 1.0 (full attention)
  Distant scans: TEM ≈ 0.0 (minimal attention)
```

**Full architecture**:
```
Input: (batch, 8 timepoints, 768 dims) + time distances
    ↓
Linear projection (768 → D_model, if needed)
    ↓
+ [CLS] token prepended
+ TaViT time-distance attention scaling
    ↓
Transformer: 8 layers, 8 attention heads
    ↓
Output: [CLS] token → 768-dim temporal representation
```

**Why TaViT over alternatives?**

| Model | AUC | Time Handling | Notes |
|---|---|---|---|
| TaViT | 0.786 | Learnable emphasis | Recent scans weighted higher → best for clinical use |
| TeViT | 0.785 | Fixed sinusoidal | Simpler but less adaptive |
| No time encoding | 0.500 | None | Random chance on irregular data! |
| Single-scan CNN | 0.734 | N/A | No temporal modeling |

**Critical finding**: Without time encoding → 0.50 AUC on irregular intervals. Yale has irregular intervals → TaViT is essential.

---

#### Step 3.3: Self-Supervised Pretraining (Week 9–10)

**Masked Autoencoder Pretraining** (no labels needed):
1. Take a patient's 8 embeddings
2. Mask 75% (keep 2 visible)
3. Model predicts masked embeddings from visible ones + time info
4. Pretrain on ALL 11,884 Yale scans (self-supervised)
5. Fine-tune on downstream tasks

**What it learns**:
- Tumor growth rates (embedding distance ∝ volume change)
- Spread patterns (spatial features in 768-dim capture location shifts)
- Treatment response (shrinking vs. growing trajectories)
- Scanner-invariant progression (ComBat already removed scanner effects)

---

#### Step 3.4: Training (Week 10)

**Data split** (Yale only):
- Training: 1,000 patients (70%)
- Validation: 215 patients (15%)
- Testing: 215 patients (15%)

**Optional external validation** (discuss with supervisor):
- Test trained model on Cyprus 40 patients
- Proves cross-population generalization (Mediterranean vs. USA)
- Different primary tumors, population genetics, scanner protocols

---

#### Step 3.5: Representation Evaluation (Week 11)

**A) Downstream Prediction** — "Are the embeddings useful?"
- Feed frozen ViT embeddings to simple classifiers (logistic regression / small MLP)
- Tasks: tumor growth (growing/stable/shrinking), treatment response, volume prediction
- Compare: ViT embeddings vs. CNN features vs. hand-crafted radiomics
- Tool: `sklearn.linear_model.LogisticRegression`

**B) Clustering** — "Do similar tumors group together?"
- t-SNE / UMAP visualization of all embeddings, colored by:
  - Tumor stage → should cluster separately
  - Patient ID → should form trajectories over time
  - Scanner/year → should be mixed (if ComBat worked!)
- Tool: `umap-learn`, `sklearn.manifold.TSNE`

**C) Temporal Consistency** — "Do trajectories make sense?"
- Are predictions monotonic (consistent growth/shrinkage)?
- Is embedding distance proportional to time distance?
- Can T0+T1+T2 embeddings predict T3 characteristics?

**Metrics**:
- Downstream: AUC, accuracy, F1 (classification); RMSE (volume prediction)
- Clustering: silhouette score, adjusted Rand index
- Temporal: Pearson correlation (embedding distance vs. time distance)

---

#### ✅ Phase 3 Checkpoint (End of Week 11)
- [ ] Swin UNETR extracts 768-dim embeddings for all scans
- [ ] ComBat harmonization removes scanner effects from embeddings
- [ ] TaViT captures temporal tumor evolution with learnable time encoding
- [ ] ViT embeddings outperform CNN baseline on downstream tasks
- [ ] Clustering shows biologically meaningful groupings
- [ ] **Ready for Objective 3: Explainability + LLM integration**

---

## Objective 3 — Integrate Imaging and Clinical Context Using Large Language Models

> **Goal**: Enable multimodal clinical reasoning by combining visual tumor representations with structured clinical metadata using LLMs.

**Sub-objectives**:
- Design multimodal representations that fuse imaging embeddings with clinical variables (cancer stage, treatment type)
- Develop prompt-based strategies to guide LLMs in generating clinically coherent explanations of disease progression
- Align visual evidence with textual medical reasoning for interpretability

**Deliverables**:
- Multimodal embedding representation
- LLM prompt templates and generated clinical narratives
- Qualitative validation of clinical plausibility

---

### Phase 4 — LLM Integration (Weeks 12–14)

**Papers used**:
- **RadFM** (Wu et al., Nature Communications 2025) → `objectivethree/SIMPLE_Paper_RadFM.md`
- **MM-Embed** (ICLR 2025, NVIDIA) → `objectivethree/SIMPLE_Paper_MM_Embed.md` (training methodology)

---

#### Step 4.1: Design Multimodal Representations (Week 12)

**Activity**: Fuse imaging embeddings with structured clinical metadata

**Inputs to combine**:
1. **Imaging embeddings** (from Phase 3):
   - Swin UNETR embeddings: 768-dim per scan
   - TaViT temporal summary: captures progression trajectory
   
2. **Clinical metadata** (from Yale dataset):
   - Patient demographics: age, sex
   - Cancer information: primary cancer type, stage
   - Treatment history: surgery dates, radiation, chemotherapy

**Fusion approach** (from RadFM):
```
Temporal sequence: T0, T1, T2 (each = 32 tokens via Perceiver)
+ TaViT temporal summary (32 tokens)
+ Clinical metadata (text tokens)
→ Combined multimodal representation
```

**Tasks**:
1. Extract clinical metadata from Yale dataset
2. Format as structured text prompts
3. Design token interleaving strategy (imaging + clinical)
4. Verify dimension compatibility with RadFM architecture

**Output**: Multimodal embedding representation combining imaging + clinical data

---

#### Step 4.2: Build LLM Pipeline (Week 13)

**Activity**: Design prompt strategies and integrate RadFM architecture

**Architecture** (using RadFM):
```
Swin UNETR (768-dim) → Perceiver (32 tokens) → MedLLaMA-13B → Clinical narrative
```

**Prompt design** (prompt-based strategies):
```
Instruction: "Based on this temporal sequence of brain MRIs, describe the disease progression."

Input:
- Imaging: T0, T1, T2 scans (32 tokens each via Perceiver)
- Clinical: "Patient: 67yo female. Primary: lung adenocarcinoma. Treatment: SRS 2024-03."
- Temporal: TaViT summary (progression trajectory)

Output: "15mm enhancing lesion in right frontal lobe. 20% growth T0→T1 (6 months). 
Stabilized T1→T2 after radiosurgery. Recommend: surveillance, next scan 3 months."
```

**What we use from RadFM**:
- Perceiver: compresses 768-dim embeddings → 32 fixed tokens
- MedLLaMA-13B: generates clinical text
- Two-stage training: freeze LLM first, then fine-tune
- UMLS-weighted loss: medical terms weighted 3×

**Tasks**:
1. Set up RadFM code and weights
2. Replace RadFM's ViT with our Swin UNETR encoder
3. Design prompt templates for brain tumor progression
4. Test inference on sample Yale scans

**Code**: [RadFM GitHub](https://github.com/chaoyi-wu/RadFM) | [Weights](https://huggingface.co/chaoyi-wu/RadFM)

**Output**: LLM prompt templates ready for training

---

#### Step 4.3: Train & Generate Narratives (Week 14)

**Activity**: Train RadFM and generate medically coherent disease progression narratives

**Training** (two-stage, from RadFM + MM-Embed insights):

**Stage 1 — Alignment** (1-2 epochs):
- Freeze MedLLaMA-13B
- Train: Perceiver (~50M params) + projection layer
- Learn: map Swin UNETR embeddings → LLM space

**Stage 2 — Instruction Tuning**:
- Add LoRA adapters (~4M params) OR full fine-tune (if compute available)
- Train: generate brain tumor progression reports
- Data: Yale clinical metadata formatted as instruction-response pairs
- Loss: UMLS-weighted (medical terms = 3×, regular text = 1×)

**Training insights from MM-Embed**:
- Train stages **sequentially** (curriculum learning)
- Watch for **text bias** (model ignoring visual features)
- Use hard negatives (wrong reports that look plausible)

**Compute options**:
- Full: 32× A100 80GB (unlikely)
- LoRA: 1× A100 80GB (realistic)
- Adapter-only: 1× A100 40GB (most practical, ~54M trainable params)

**Tasks**:
1. Prepare Yale instruction tuning dataset (scan sequences + metadata → clinical reports)
2. Run Stage 1 training (alignment)
3. Run Stage 2 training (instruction tuning)
4. Generate narratives for validation set

**Output**: Generated clinical narratives for Yale patients

---

#### Step 4.4: Validate Clinical Plausibility (Week 14)

**Activity**: Qualitative validation of generated reports

**Validation methods**:
1. **Temporal alignment**: Do generated narratives match imaging timelines?
2. **Clinical coherence**: Are reports medically plausible?
3. **Visual grounding**: Do reports reference actual scan findings?

**Metrics**:
- BLEU, ROUGE (automatic text quality)
- UMLS precision/recall (medical terminology)
- Expert review (5-10 sample cases with radiologists)

**Tasks**:
1. Generate reports for test set
2. Compute automatic metrics
3. Select diverse cases for expert review
4. Document qualitative feedback

**Output**: Validated clinical narratives + qualitative assessment report

---

#### ✅ Phase 4 Checkpoint (End of Week 14)
- [ ] Multimodal representations designed (imaging + clinical metadata)
- [ ] LLM prompt templates created
- [ ] RadFM integrated with Swin UNETR embeddings
- [ ] Clinical narratives generated for Yale patients
- [ ] Qualitative validation completed
- [ ] **Ready for Objective 4: Video generation**
---

## Objective 4 — Generate and Analyze Cancer Progression Videos

> **Goal**: To model and visualize cancer progression through AI-generated video sequences conditioned on multimodal inputs. *(project.txt, Objective 4)*

### Phase 5 — Generative Video Modeling of Cancer Progression (Weeks 15–18)

**Key papers analyzed** (full details → `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md`):
- ⭐⭐⭐ **Treatment-aware Diffusion for Glioma** (Liu et al., IEEE-TMI 2025) — same domain, same task, treatment-conditioned MRI generation
- ⭐⭐⭐ **MedEdit** (Ben Alaya et al., MICCAI24 SASHIMI) — counterfactual brain MRI editing, beats Palette by 45%
- ⭐⭐⭐ **EchoNet-Synthetic** (Reynaud et al., MICCAI 2024) — LVDM for medical video, **full code+weights available**
- ⭐⭐ **Video LDM** (Blattmann et al., CVPR 2023) — temporal layer insertion into frozen image LDM
- ⭐⭐ **Counterfactual Diff. AE** — latent space manipulation for counterfactuals
- ⚙️ **LDM** (Rombach et al., 2022) + **DDPM** (Ho et al., 2020) — foundation building blocks

#### Step 5.1: Adapt Video Diffusion Model to Brain MRI (Week 15)

**What we do**: Select and adapt a diffusion-based video generation model for 3D brain MRI sequences.

**Architecture** (built from deep paper reading):
```
3D VAE Encoder (from LDM, adapted per EchoNet-Synthetic) → Latent space
→ Temporal Video Diffusion UNet:
    - Spatial layers: frozen (from pre-trained LDM)             ← Video LDM strategy
    - Temporal attention layers: trainable (reshape trick)       ← Video LDM: (b t) c h w ↔ b c t h w
    - Learnable merge parameter α (blend spatial/temporal)       ← Video LDM Eq. 3
    - Treatment conditioning: sinusoidal + MLP for ⟨τ,d⟩ pairs  ← TaDiff design
    - Vision conditioning: Swin UNETR 768-dim → cross-attention  ← LDM: Q=W_Q·φ(z), K=W_K·τ_θ(y)
    - Text conditioning: RadFM narratives → cross-attention       ← Our Obj 3 output
    - Joint output: noise prediction ε̃ + tumor segmentation m̃   ← TaDiff joint learning
→ 3D VAE Decoder → Realistic MRI volumes
```

**VAE design** (adapted from EchoNet-Synthetic + LDM paper insights):
- EchoNet-Synthetic VAE compresses 3×112×112 → 4×14×14 (8× downsampling, 48× compression)
- LDM paper shows f=4 to f=8 gives best quality-compute tradeoff (Table 8, Fig. 6)
- **Our 3D VAE**: 128³×4 (4 modalities) → 16³×4 latent (8× per dim, 512× total compression)
- Use KL-regularization variant (slight KL penalty toward N(0,1)) — LDM shows this is more flexible than VQ-reg
- Train with perceptual loss + adversarial loss + KL-reg (LDM recipe)
- **Target**: SSIM ≥ 0.78, PSNR ≥ 24.9 dB for reconstruction (EchoNet-Synthetic baseline)

**Starting point**: EchoNet-Synthetic codebase (only medical video diffusion with full code+weights)
- Clone [github.com/HReynaud/EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic)
- Their 3-model pipeline: (1) VAE, (2) LIDM (unconditional image), (3) LVDM (conditional video)
- We adapt: VAE (2D→3D), skip LIDM, heavily modify LVDM with our conditioning
- Replace echo-specific components with brain MRI pipeline

**Diffusion process specifics** (from DDPM + TaDiff papers):
- T=600 diffusion steps (TaDiff shows 600 is sufficient for medical MRI, vs. DDPM's 1000)
- Linear noise schedule: β₁=10⁻⁴ to β_T=0.02
- ε-prediction parameterization (DDPM Eq. 11): predict noise, not image directly
- Training loss: L_simple = E[‖ε − ε_θ(x_t, t)‖²] — unweighted MSE is simplest and best (DDPM finding)
- Can use DDIM for faster inference (50-100 steps instead of 600)

**Tasks**:
1. Train 3D VAE on Yale brain MRI volumes (encode/decode MRI to/from latent space)
   - All 4 modalities as channels: T1, T1c, T2, FLAIR → 4-channel input
   - Validate reconstruction quality on held-out scans (SSIM, PSNR)
2. Build UNet backbone with temporal layer insertion (Video LDM approach):
   - Spatial blocks: channel dims 64→128→256→512 (TaDiff UNet design)
   - Insert temporal attention + 3D conv residual blocks after each spatial block
   - Merge parameter αᵢ: learnable blend, initialized to 1 (=identity, recovers image model)
3. Design multi-modal conditioning:
   - Treatment: sinusoidal embedding for day + learned embedding for treatment type → MLP → sum (TaDiff)
   - Vision: Swin UNETR 768-dim → projection → cross-attention K,V (LDM cross-attention: Attention(Q,K,V) = softmax(QK^T/√d)·V)
   - Text: RadFM narrative tokens → cross-attention K,V
4. Test basic unconditional generation on Yale data (sanity check before conditioning)

---

#### Step 5.2: Generate Progression & Counterfactual Videos (Weeks 16–17)

**Input**: Patient's past scans (T0, T1, T2) + Swin UNETR embeddings + RadFM narratives + treatment metadata

**Output**: Two types of generated sequences:

**Mode 1 — Natural Progression** (predict future tumor evolution):
- Feed patient history → diffusion model generates T3, T4, T5
- TaViT temporal patterns guide generation (learned tumor growth dynamics)
- Temporal attention ensures frame-to-frame consistency
- **Video stitching** (from EchoNet-Synthetic): generate overlapping chunks of 4-8 time-points, overlap by half, stitch for long sequences

**Mode 2 — Counterfactual Scenarios** (treatment what-if):
- "What if surgery?" → generate progression under surgical conditioning
- "What if radiation?" → generate progression under radiation conditioning  
- "What if no treatment?" → generate natural progression baseline
- **Method (from TaDiff)**: Encode each treatment-day pair ⟨τ,d⟩ separately:
  - Treatment type τ → embedding table → learned vector
  - Day number d → sinusoidal embedding → learned vector
  - Final conditioning = treatment_emb + day_emb (per time-point)
  - Keep patient Swin UNETR embeddings FIXED, change ONLY treatment vectors
  - TaDiff showed: with treatment → SSIM 0.919, DSC 0.719; without → SSIM 0.882, DSC 0.556 (treatment info adds 3.7% SSIM, 16.3% DSC!)

- **Anatomical constraints (from MedEdit)**:
  - Mask dilation (kernel k=25): edit dilated region around tumor to capture indirect effects (edema, mass effect, brain shift)
  - RePaint-based inpainting: keep known anatomy, regenerate only target region
  - 4 resampling steps for harmonization between edited and preserved regions
  - MedEdit showed clinical realism = 3.20/5 (matching real samples!)

- **Joint generation + segmentation (from TaDiff)**:
  - Model predicts BOTH noise ε̃ AND tumor masks m̃ simultaneously
  - Joint loss: L = ‖ω(ε − ε̃)‖² + λ·ℓ_seg
  - Weighting mechanism ω: extra weight on tumor boundary region → better tumor prediction
  - ω = m̂·exp(−m̂ ∗ f_{k×k}) + 1 → weights range [1.886, 5.451] for tumor, 1 elsewhere
  - Segmentation loss: ℓ_seg = ℓ_dice(m̃_S, m_S) + √ᾱ_t·ℓ_dice(m̃_f, m_f)
  - √ᾱ_t scaling: segmentation loss weighted by noise level (more weight at less noisy steps)

**Uncertainty estimation** (from TaDiff Algorithm 2):
- Generate 5 stochastic samples per condition (different random seeds)
- Mean = prediction, Variance = uncertainty map
- Mask fusion over last T_m=10 denoising steps → cleaner tumor boundaries

**Cascaded generation for 3D feasibility** (from EchoDiffusion):
- Stage 1: Generate 64³ low-resolution volumes (coarse structure)
- Stage 2: Refine to 128³+ high-resolution (fine detail)
- Makes 3D video generation tractable on available GPU hardware

**Tasks**:
1. Implement treatment conditioning (TaDiff-style: sinusoidal day embedding + learned treatment embedding → MLP → sum)
2. Add conditioning pathway for Swin UNETR embeddings (768-dim → linear projection → cross-attention Q,K,V)
3. Add conditioning pathway for RadFM narratives (text tokens → cross-attention)
4. Implement joint noise+segmentation prediction with weighted loss ω
5. Train counterfactual generation: same patient, different treatment → different progression
6. Add uncertainty estimation via 5-sample stochastic sampling
7. Implement MedEdit-style mask dilation for indirect effect modeling
8. Implement cascaded refinement if full-resolution is too expensive

---

#### Step 5.3: Train & Validate on Yale Sequences (Week 18)

**Training** (informed by TaDiff + EchoNet-Synthetic training details):
- Use real Yale longitudinal progressions as ground truth (1,430 patients, 11,884 scans)
- Train on known trajectories: given (T0, T1, T2), predict (T3) and compare to real T3
- Treatment labels from Yale metadata → learn treatment-specific progression patterns
- **Training recipe** (adapted from TaDiff: 350 GPU-hours on V100 32GB):
  - 2D/2.5D approach: 192×192 slices (if GPU-limited) or full 3D 128³ volumes
  - Optimizer: Adam, lr=2.5×10⁻⁴, cosine decay
  - Batch size: 32 (with gradient accumulation ×2 if needed)
  - Training iterations: ~2–5M steps (TaDiff used 5M on 23 patients → we scale accordingly)
  - T=600 diffusion steps, linear β schedule
  - EchoNet-Synthetic reference: VAE=5 days on 8×A100; LVDM=2 days on 8×A100
- **Compute estimate**: With 1,430 patients (60× more than TaDiff's 23), expect 500-1000 GPU-hours total
  - VAE training: ~100-200 GPU-hours
  - Video diffusion training: ~300-500 GPU-hours
  - Counterfactual fine-tuning: ~100-200 GPU-hours

**Validation**:
- Hold out ~20% of patients for testing (286 patients, ~2,000+ scans)
- **Quantitative** (target metrics from paper benchmarks):

  | Metric | Target | Reference |
  |---|---|---|
  | SSIM | ≥ 0.85 | TaDiff external validation: 0.848 |
  | PSNR | ≥ 25 dB | TaDiff: 27.97 dB (local), ~24 dB (external) |
  | FID | ≤ 30 | EchoNet-Synthetic: 28.8 |
  | Tumor DSC | ≥ 0.65 | TaDiff: 0.719 (local), ~0.55 (external) |
  | FVD (temporal) | TBD | Video LDM: 389 on RDS driving |

- **Qualitative**: Visual inspection — do generated progressions look realistic?
- **Clinical plausibility** (MedEdit validation protocol):
  - Blind evaluation by neuroradiologist: rate realism on 1-5 scale
  - Target: ≥ 3.0/5 (MedEdit achieved 3.20/5 = indistinguishable from real)
  - Counterfactual scenarios: verify treatment effects match clinical expectations
    - Radiation → tumor shrinkage expected
    - No treatment → tumor growth expected
    - Different chemotherapy → different response patterns

**Deliverables**:
- Generated cancer progression video sequences (natural + counterfactual)
- Counterfactual disease trajectory visualizations with uncertainty maps
- Analysis of video realism, temporal coherence, and clinical plausibility
- Comparison: our pipeline (1,430 patients, multimodal conditioning) vs. TaDiff baseline (23 patients, treatment-only)

---

#### ✅ Phase 5 Checkpoint (End of Week 18)
- [ ] 3D VAE encodes/decodes Yale brain MRI volumes
- [ ] Video diffusion model generates temporally consistent future scans
- [ ] Conditioning works: embeddings + narratives + treatment metadata guide generation
- [ ] Counterfactual trajectories show clinically plausible treatment effects
- [ ] Quantitative metrics computed (FID, SSIM, LPIPS)
- [ ] **Ready for Objective 5: Final evaluation**

---

## Objective 5 — Evaluate Explainability, Clinical Plausibility, and Scientific Impact

> **Goal**: Rigorously assess the full multimodal framework's performance, interpretability, and medical relevance, then prepare publication-ready documentation.

### Phase 6 — Evaluation and Final Deliverables (Weeks 19–20)

**Papers needed**: RANO criteria for brain tumors, clinical evaluation methods

#### Step 6.1: Quantitative Evaluation (Week 19)

**Metrics**:
## Full Pipeline Summary

**OBJECTIVE 1 — Phase 1 (Weeks 1–4): DATA PIPELINE**
- Raw MRI scans → HD-BET skull strip → Normalize → nnU-Net segment → FLIRE/Elastix register
- **Output**: Clean, segmented, aligned longitudinal sequences

**OBJECTIVE 2 — Phase 2 (Weeks 5–7): CNN BASELINE**
- Aligned scans → CNN feature extraction → Baseline metrics
- **Output**: Performance benchmarks, static model limitations

**OBJECTIVE 2 — Phase 3 (Weeks 8–11): ViT + HARMONIZATION + TEMPORAL MODELING**
- Aligned scans → Swin UNETR encoder → 768-dim embeddings
- → ComBat harmonization (remove scanner effects, preserve biology)
- → TaViT temporal model (learn progression patterns)
- **Output**: Harmonized temporal embeddings capturing tumor evolution
- **Why harmonization here**: Ensures Objectives 3, 4, 5 work on biology not scanner artifacts

**OBJECTIVE 3 — Phase 4 (Weeks 12–14): EXPLAINABILITY + LLM**
- Embeddings → Grad-CAM heatmaps + TaViT weights → LLM → Clinical narratives
- **Output**: Explainable reports linked to visual evidence
- **Benefits from harmonization**: LLM explains real biology, not scanner differences

**OBJECTIVE 4 — Phase 5 (Weeks 15–18): VIDEO GENERATION**
- Embeddings + LLM reasoning → Diffusion model → Progression + counterfactual videos
- Architecture: 3D VAE (f=8, KL-reg) + Temporal Video Diffusion UNet (frozen spatial + trainable temporal layers + merge α)
- Conditioning: Swin UNETR 768-dim → cross-attention + RadFM narratives → cross-attention + treatment-day pairs → sinusoidal+MLP embedding (TaDiff design)
- Joint prediction: noise ε̃ + tumor segmentation m̃ with weighted loss ω focusing on tumor boundaries
- Counterfactuals: same patient, different treatment vector → different progression; MedEdit mask dilation (k=25) for indirect effects
- Training: T=600 steps, Adam lr=2.5e-4, ~500-1000 GPU-hours total
- Targets: SSIM≥0.85, PSNR≥25dB, tumor DSC≥0.65, clinical realism≥3.0/5
- **Output**: Temporally coherent disease trajectory videos + counterfactual comparisons + uncertainty maps
- Key papers: Treatment-aware Glioma/TaDiff (IEEE-TMI 2025), MedEdit (MICCAI24), EchoNet-Synthetic (MICCAI24, code available)
- **Benefits from harmonization**: Smooth videos, no scanner-switch jumps

**OBJECTIVE 5 — Phase 6 (Weeks 19–20): EVALUATION**
- Quantitative metrics + clinical validation + publication
- **Output**: Validated framework + publication-ready manuscript
- **Benefits from harmonization**: Works on Cyprus despite different scanners
---

#### ✅ Phase 6 Checkpoint (End of Week 20)
- [ ] All quantitative metrics computed and compared to baselines
- [ ] Clinical validation completed with domain experts
- [ ] Technical report finalized
- [ ] Publication draft ready
- [ ] **Project complete!**

---

## Full Pipeline Summary

```
OBJECTIVE 1 — Phase 1 (Weeks 1–4): DATA PIPELINE
Raw MRI scans → HD-BET skull strip → Normalize → nnU-Net segment → FLIRE/Elastix register
Output: Clean, segmented, aligned longitudinal sequences

OBJECTIVE 2 — Phase 2 (Weeks 5–7): CNN BASELINE
Aligned scans → CNN feature extraction → Baseline metrics
Output: Performance benchmarks, static model limitations

OBJECTIVE 2 — Phase 3 (Weeks 8–11): ViT + TEMPORAL MODELING  
Aligned scans → Swin UNETR encoder → 768-dim embeddings → ComBat harmonize → TaViT temporal model
Output: Harmonized temporal embeddings capturing tumor evolution

OBJECTIVE 3 — Phase 4 (Weeks 12–14): EXPLAINABILITY + LLM (RadFM)
Embeddings → Grad-CAM heatmaps + TaViT weights → RadFM Perceiver (32 tokens) → MedLLaMA-13B → Clinical narratives
Output: Explainable reports linked to visual evidence
Key paper: RadFM (Nature Communications 2025) — 768-dim perfect match with Swin UNETR

OBJECTIVE 4 — Phase 5 (Weeks 15–18): VIDEO GENERATION
Embeddings + RadFM narratives + treatment metadata → 3D VAE (f=8) + Temporal Video Diffusion UNet (frozen spatial + trainable temporal + merge α) → Progression videos + counterfactual scenarios
Treatment conditioning: sinusoidal day embedding + learned treatment embedding → MLP (from TaDiff)
Joint output: noise ε̃ + tumor segmentation m̃, weighted loss ω on tumor boundaries
Counterfactuals: fix patient embeddings, vary treatment vector; MedEdit mask dilation for indirect effects
Uncertainty: 5 stochastic samples → mean prediction + variance map
Key papers: TaDiff (IEEE-TMI 2025), MedEdit (MICCAI24), EchoNet-Synthetic (MICCAI24, full code)
Output: Temporally coherent disease trajectory videos + "what-if" treatment comparisons + uncertainty maps

OBJECTIVE 5 — Phase 6 (Weeks 19–20): EVALUATION
Quantitative metrics + clinical validation + publication
Output: Validated framework + publication-ready manuscript
```

---

## Papers Analyzed — Quick Reference

### Objective 1 (Phase 1 — Preprocessing)

| # | Paper | Role | SIMPLE Doc |
|---|---|---|---|
| 1 | BraTS Toolkit | Skull strip, normalize | `objectiveone/SIMPLE_Paper1_BraTS_Toolkit.md` |
| 2 | nnU-Net | Tumor segmentation baseline | `objectiveone/SIMPLE_Paper2_nnUNet.md` |
| 3 | Yale Dataset | Primary training data | `objectiveone/SIMPLE_Paper3_Yale_Dataset.md` |
| 4 | Registration (Keypoints) | Tumor-preserving alignment (reference) | `objectiveone/SIMPLE_Paper4_Registration.md` |
| 5 | FLIRE | ❌ SKIP: Code unavailable, breast-only, MATLAB | `objectiveone/SIMPLE_Paper5_FLIRE.md` |
| 6 | itk-elastix | **PRIMARY**: Python, brain-proven, multi-parametric | `objectiveone/SIMPLE_Paper_ITK_Elastix.md` |
| 7 | Cyprus Dataset | External validation data | `objectiveone/SIMPLE_Paper7_Cyprus.md` |
| 8 | MONAI | Medical imaging framework | `objectiveone/SIMPLE_Paper8_MONAI.md` |
| 9 | ComBat | Scanner harmonization | `objectiveone/SIMPLE_Paper9_ComBat.md` |
| 10 | Longitudinal ComBat | Preserve trajectories | `objectiveone/SIMPLE_Paper10_LongComBat.md` |
| 10+ | Generalized ComBat | Nested multi-batch effects | `objectiveone/SIMPLE_Paper10+_Generalized_ComBat.md` |

### Objective 2 (Phase 3 — ViT Representations)

| # | Paper | Role | SIMPLE Doc |
|---|---|---|---|
| 11 | Time-Distance ViT (TaViT) | Temporal attention model | `objectivetwo/SIMPLE_Paper11_Time_Distance_ViT.md` |
| 13 | Swin UNETR | 3D segmentation + 768-dim embeddings | `objectivetwo/SIMPLE_Paper13_Swin_UNETR.md` |
| — | TransXAI | Grad-CAM explainability (→ Phase 4) | `objectivetwo/SIMPLE_Paper_TransXAI.md` |
| — | CAFNet | CNN+ViT hybrid validation | `objectivetwo/SIMPLE_Paper_CAFNet.md` |

### Objective 3 (Phase 4 — LLM Integration)

| # | Paper | Role | SIMPLE Doc |
|---|---|---|---|
| — | **RadFM** (Nature Comms 2025) | 3D vision-language model: Perceiver + MedLLaMA-13B | `objectivethree/SIMPLE_Paper_RadFM.md` |
| — | **MM-Embed** (ICLR 2025, NVIDIA) | Multimodal retrieval training methodology (modality bias, curriculum learning) | `objectivethree/SIMPLE_Paper_MM_Embed.md` |

Additional papers identified (see `objectivethree/PHASE4_LLM_PAPERS_ANALYSIS.md`):
- LLaVA-Med: curriculum learning strategy
- RaDialog: conversational interface + structured findings
- R2GenGPT: frozen LLM + tiny adapter approach

### Objective 4–5 (Phases 5–6 — Video, Evaluation)

#### Objective 4 (Phase 5 — Video Generation & Counterfactual)

| # | Paper | Role | Analysis |
|---|---|---|---|
| — | **DDPM** (Ho et al., NeurIPS 2020) | Foundation: forward/reverse process, ε-prediction, L_simple | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **LDM / Stable Diffusion** (Rombach et al., CVPR 2022) | Foundation: VAE + latent diffusion + cross-attention conditioning | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **Video LDM** (Blattmann et al., CVPR 2023) | Temporal layer insertion: frozen spatial + trainable temporal, merge α | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **Treatment-aware Diffusion / TaDiff** (Liu et al., IEEE-TMI 2025) | Same domain! Treatment-conditioned glioma MRI generation, joint seg+gen | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **MedEdit** (Ben Alaya et al., MICCAI24 SASHIMI) | Counterfactual brain MRI editing, mask dilation, clinical validation | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **EchoNet-Synthetic** (Reynaud et al., MICCAI 2024) | LVDM medical video, **full code+weights** (starting point) | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | **Counterfactual Diffusion AE** (Atad et al., JMLBI 2024) | Latent space counterfactuals via hyperplane reflection | `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` |
| — | — | **ELI-style diffusion explainer document** | `objectivefour/DIFFUSION_MODELS_EXPLAINED.md` |

Additional papers reviewed (see `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md`):
- Counterfactual Diffusion Autoencoder: latent space counterfactual manipulation
- CLIMATv2: multi-agent trajectory forecasting
- EchoDiffusion (Cascaded): low→high resolution medical video generation
- MVG (CVPR 2024): LLM-guided medical video generation
- HeartBeat: multi-modal controllable synthesis
- SurGen: text-to-surgical-video (less relevant, different domain)
- Counterfactual MRI Data Augmentation, Counterfactual AD Effect: alternative counterfactual approaches

#### Objective 5 (Phase 6 — Evaluation)

Papers not yet analyzed — to be found during respective phase.

---

## Literature Review Status

| Phase | Papers Analyzed | Status |
|---|---|---|
| Phase 1 (Preprocessing) | 10/10 | ✅ Complete |
| Phase 2–3 (CNN + ViT) | 4/6 (2 skipped as lower priority) | ✅ Complete |
| Phase 4 (LLM) | 2 (RadFM + MM-Embed) + 7 surveyed | ✅ RadFM deep-analyzed, MM-Embed analyzed, others identified |
| Phase 5 (Video) | 7 key papers deep-analyzed (from PDFs) + 10 surveyed (17 total) | ✅ Complete — DDPM, LDM, Video LDM, TaDiff, MedEdit, EchoNet-Synthetic, CF-DAE deep-read; Explainer doc created |
| Phase 6 (Evaluation) | 0 | ⏳ Not started |

**Total**: 22 papers analyzed (10 P1 + 4 P2-3 + 2 P4 + 6 P5), 18 additional surveyed. Phases 1–5 literature review complete.

---

## Key Technical Decisions

1. **ViT extracts features, not manual radiomics** — Swin UNETR learns richer 768-dim representations than hand-crafted 110-feature radiomics. ComBat harmonizes embeddings in Phase 3, not raw features in Phase 1.

2. **Swin UNETR for dual-purpose** — Same model handles segmentation (decoder) AND embedding extraction (encoder). Dice 0.913 > nnU-Net 0.908.

3. **TaViT for irregular temporal data** — Yale has irregular scan intervals. Without time encoding → 0.50 AUC (random chance). TaViT's learnable emphasis is essential.

4. **itk-elastix for registration (FLIRE irrelevant)** — itk-elastix is pip-installable Python package, THE gold standard for brain tumor registration (100+ papers, BraTS benchmark, Yale validated 0.95 correlation, 14.5 min/case). Multi-parametric MRI native (Mutual Information handles T1/T1c/T2/FLAIR). Seamless NumPy/PyTorch/MONAI integration. Brain-optimized parameter files ready from model zoo. 15 years mature: GitHub forum, 100+ Jupyter examples, hundreds of CI tests. FLIRE dismissed: code not released (unusable!), only breast MRI (no brain validation), MATLAB-only (no Python), no community. Speed difference (4.5 min/case) irrelevant when FLIRE can't be used.

5. **Post-hoc explainability (TransXAI approach)** — Grad-CAM on existing architecture. No accuracy tradeoff, no architecture modifications needed.

5. **Post-hoc explainability (TransXAI approach)** — Grad-CAM on existing architecture. No accuracy tradeoff, no architecture modifications needed.

6. **ComBat on embeddings** — Harmonize after ViT extraction. Nested ComBat for multi-batch effects (scanner, year, field strength), Longitudinal ComBat to preserve patient trajectories.

7. **RadFM Perceiver for vision-to-LLM bridge** — RadFM's Perceiver compresses any-size 3D features to fixed 32 tokens. Our Swin UNETR output (768 dims) matches RadFM's ViT output (768 dims) exactly — no adapter needed. Two-stage training (freeze LLM first, then LoRA fine-tune) keeps compute tractable.

8. **Temporal multi-image input for longitudinal reports** — RadFM supports interleaved multi-image input. We feed patient's scan sequence as multiple images (T0, T1, T2) + TaViT temporal summary, enabling longitudinal progression reports. This is our novel contribution — no existing paper does temporal 3D MRI → LLM.

9. **Modality bias awareness (from MM-Embed, ICLR 2025)** — NVIDIA showed that MLLMs inherently prefer text over visual features. During RadFM fine-tuning, we must monitor for text bias (generating generic reports that ignore actual scan content) and use hard negative training to ensure visual grounding. Curriculum (sequential) training validated as superior to joint training.

10. **EchoNet-Synthetic as starting codebase for video generation** — Only medical video diffusion paper with full code+weights+dataset. Their 3-model pipeline (VAE→LIDM→LVDM) provides proven architecture. Adapt LVDM from 2D echo → 3D brain MRI. Insert temporal layers (Video LDM strategy: frozen spatial + trainable temporal = only ~20% params to train; reshape trick `(b t) c h w ↔ b c t h w`; learnable merge parameter α). Treatment conditioning from TaDiff paper (same domain — brain tumors, same modalities — T1/T1c/T2/FLAIR): sinusoidal day embedding + learned treatment embedding → MLP → sum per time-point. Joint noise+segmentation prediction with weighted loss ω focusing on tumor boundaries (TaDiff showed +16.3% DSC with treatment conditioning). Counterfactual formulation from MedEdit (mask dilation k=25 for indirect effects, 4 resampling steps, clinical realism matching real samples at 3.20/5). Cascaded generation (64³ → 128³) from EchoDiffusion for 3D feasibility. LDM cross-attention mechanism for Swin UNETR + RadFM conditioning injection: Q from latent features, K/V from conditioning tokens.

11. **Diffusion training specifics from paper analysis** — T=600 steps (TaDiff validated sufficient for medical MRI), linear β schedule (10⁻⁴ to 0.02), ε-prediction parameterization (DDPM), Adam optimizer lr=2.5×10⁻⁴ with cosine decay (TaDiff), batch=32 with gradient accumulation ×2. VAE uses KL-regularization (LDM shows more flexible than VQ-reg), f=4-8 optimal compression (LDM Table 8). EchoNet uses v-prediction variant for LVDM. Target metrics from paper benchmarks: SSIM≥0.85, PSNR≥25dB, tumor DSC≥0.65, FID≤30. Compute estimate: ~500-1000 GPU-hours total (TaDiff used 350h for 23 patients on V100; EchoNet used 5 days×8 A100s for VAE + 2 days×8 A100s for LVDM).
