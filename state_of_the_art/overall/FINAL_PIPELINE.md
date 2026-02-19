# Final Research Pipeline — Complete & Self-Explanatory

> **Intern**: Mohamed | **Supervisor**: Dr. Belkacem Chikhaoui | **Institution**: TÉLUQ University, Montreal  
> **Duration**: 20 weeks (April–August 2026) | **Program**: Mitacs Globalink Research Internship  
> **This document**: The single source of truth for the entire project pipeline.

---

## Overall Objective

Design, implement, and evaluate a multimodal AI framework that integrates Vision Transformers (ViTs), Large Language Models (LLMs), and generative video models to **analyze, explain, and simulate cancer progression** from longitudinal brain MRI data.

### Research Questions (from project.txt)
1. Can ViT-based models capture meaningful representations of tumor morphology and its evolution from longitudinal medical imaging?
2. Can LLMs integrate imaging-derived features with clinical metadata to provide coherent, medically grounded reasoning about disease progression?
3. Is it possible to generate temporally consistent and clinically plausible videos that visualize cancer progression over time?
4. Can such a system support counterfactual analysis by simulating alternative tumor trajectories under different treatment scenarios?

---

## Dataset

| | **Yale (Primary — Training)** | **Cyprus (Optional — Validation)** |
|---|---|---|
| Scans | 11,884 brain MRIs | 744 brain MRIs |
| Patients | 1,430 | 40 |
| Avg scans/patient | ~8 | ~18.6 |
| Period | 2004–2023 | 2019–2024 |
| Modalities | T1, T1c, T2, FLAIR | T1, T1c, T2, FLAIR |
| Labels | None (generate with nnU-Net) | Expert-verified (3 sub-regions) |
| Treatment info | Pre/post surgery, radiation, chemo | Available |
| Access | TCIA (free) | Zenodo (free) |

**Strategy**: Train on Yale → optionally validate on Cyprus (proves cross-population generalization).

**Papers**: Yale Dataset (Ramakrishnan et al., 2025), Cyprus Dataset (Trimithiotis et al., 2025)

---

## Complete Pipeline at a Glance

```
Raw MRI → HD-BET → nnU-Net → itk-elastix → Swin UNETR → ComBat → TaViT → RadFM → Video Diffusion → Evaluation
   │          │         │          │             │           │         │        │           │              │
 Week 1    Week 1    Week 2    Week 3       Week 8      Week 8   Week 9  Week 12     Week 15        Week 19
   └──────── Objective 1 ────────┘    └───── Objective 2 ──────┘   └─ Obj 3 ─┘  └── Obj 4 ──┘    └─ Obj 5 ─┘
```

---

## Objective 1 — Robust Pipeline for Longitudinal Cancer Imaging Analysis

> **project.txt**: "Build a reproducible and clinically meaningful data pipeline for processing and organizing longitudinal cancer imaging data."

### Phase 1 (Weeks 1–4)

#### Step 1.1: Download & Preprocess (Week 1)

**What**: Take raw multi-scanner brain MRIs and standardize them.

**Tasks**:
1. Download Yale from TCIA (~200 GB)
2. Convert to NIfTI format
3. Skull stripping with **HD-BET** — removes non-brain tissue
4. Intensity normalization: z-score per modality (T1, T1c, T2, FLAIR)
5. Spatial resampling to 128×128×128 (uniform resolution for Swin UNETR)

**Tools**: HD-BET, BraTS Toolkit, SimpleITK, MONAI  
**Paper**: BraTS Toolkit (Kofler et al., 2020) — provides HD-BET skull stripping + z-score normalization protocol  
**Output**: Standardized NIfTI scans, skull-stripped, normalized.

---

#### Step 1.2: Tumor Segmentation (Week 2)

**What**: Generate tumor masks (no labels exist for Yale).

**Tasks**:
1. Run **nnU-Net** on all 11,884 Yale scans → tumor masks (ET, WT, TC sub-regions)
2. Use BraTS pretrained weights, fine-tune on Yale if needed
3. Quality check: manually review 50–100 random cases
4. (Optional) Validate on Cyprus dataset — compare nnU-Net output to expert labels

**Expected performance**: Dice ~0.908 on brain tumors  
**Paper**: nnU-Net (Isensee et al., 2021) — self-configuring segmentation  
**Output**: Tumor segmentation masks for all scans.

**Note**: Swin UNETR (Step 3.1) achieves Dice 0.913 > nnU-Net's 0.908. You can optionally upgrade these masks later — but for Phase 1, nnU-Net is sufficient.

---

#### Step 1.3: Temporal Registration (Week 3)

**What**: Align each patient's scans over time so same brain regions overlap across visits.

**Method**: **itk-elastix** — the gold standard for brain MRI registration.

**Why itk-elastix**:
- `pip install itk-elastix` → working immediately
- 100+ brain tumor papers, BraTS benchmark validated (0.95 correlation)
- Handles T1/T1c/T2/FLAIR natively (Mutual Information metric)
- Seamless Python/MONAI integration
- Ready brain-optimized parameter files from model zoo

**Multi-stage registration** (automatic via parameter file):
1. Rigid (6 params) → fix patient positioning
2. Affine (12 params) → global brain shape
3. B-spline deformable (1000s of params) → tumor growth/edema/atrophy

**Tasks**:
1. For each patient, designate baseline scan (T0 = earliest timepoint)
2. Register all subsequent scans to T0: T1→T0, T2→T0, ..., T7→T0
3. Apply same transform to all 4 modalities AND tumor masks
4. Validate on 50–100 patients: correlation >0.90, tumor Dice >0.85

**Paper**: itk-elastix (Niessen et al., 2023) — Python medical image registration  
**Resources**: [GitHub](https://github.com/InsightSoftwareConsortium/ITKElastix), [Parameter zoo](https://elastix.lumc.nl/modelzoo/)  
**Output**: All patient timepoints spatially aligned to their baseline scan.

---

#### Step 1.4: Exploratory Data Analysis (Week 4)

**Tasks**:
1. Distribution of tumor sizes, locations, and counts per patient
2. Inter-patient and temporal variability analysis
3. Scan intervals, number of timepoints per patient
4. Scanner/protocol diversity (manufacturers, field strengths, years)
5. Document data quality issues

**Output**: EDA report with statistics and visualizations.

---

#### ✅ Phase 1 Deliverables
- 1,430 patients with ~8 clean, skull-stripped, aligned scans each
- Tumor segmentation masks for all 11,884 scans
- EDA report documenting dataset characteristics
- Reproducible preprocessing pipeline

---

## Objective 2 — Learn High-Level Tumor Representations Using Vision Transformers

> **project.txt**: "Extract semantically rich and temporally informative representations of tumors using state-of-the-art Vision Transformer architectures."

### Phase 2 — CNN Baseline (Weeks 5–7)

#### Step 2.1–2.2: CNN Baseline

**What**: Establish a performance baseline using traditional CNNs before moving to ViTs.

**Tasks**:
1. Implement 3D ResNet or extract nnU-Net features for single-timepoint tumor analysis
2. Train on classification/regression (tumor growth, treatment response)
3. Document performance metrics and limitations of static modeling

**Purpose**: Reference benchmark — ViT must outperform this.  
**Output**: Baseline accuracy, feature representations, documented static limitations.

---

### Phase 3 — ViT + Harmonization + Temporal Modeling (Weeks 8–11)

#### Step 3.1: Swin UNETR Embedding Extraction (Week 8)

**What**: Extract rich 768-dimensional feature vectors from each MRI scan using a pre-trained Vision Transformer.

**Architecture** (Swin UNETR, 61.98M params):
- **Input**: 128×128×128×4 volumes (T1, T1c, T2, FLAIR)
- **Encoder**: 4-stage hierarchical Swin Transformer → channels [48, 96, 192, 384, 768]
- **Output**: Bottleneck 4×4×4×768 → Global Average Pool → **768-dim embedding per scan**
- **Pre-trained weights**: NVIDIA model from 5,050 CT volumes + BraTS fine-tuning

**Per patient**: ~8 scans → 8 embeddings × 768 dims = (8, 768) matrix representing tumor evolution.

**Paper**: Swin UNETR (Tang et al., 2022) — 3D medical image segmentation + embeddings  
**Tools**: `pip install monai` → `SwinUNETR()`  
**Output**: 768-dim embedding per scan for all 11,884 scans.

---

#### Step 3.1b: ComBat Harmonization (Week 8)

**What**: Remove scanner/protocol bias from embeddings while preserving real biological signal.

**Why this is critical**: Yale spans 2004–2023 with multiple scanners. Without harmonization:
- Embeddings encode scanner type, not just tumor biology
- A patient switching scanners (2015→2018) produces an embedding jump that looks like tumor growth
- Downstream LLM would generate false clinical narratives
- Generated videos would show scanner-switch artifacts instead of smooth progression

**3-step harmonization** (applied sequentially to embeddings):

1. **Test for effects** (ComBat Validation) — Don't blindly harmonize. Test if scanner effects exist (p < 0.05). If no effects → skip.
2. **Nested ComBat** (Generalized ComBat) — Handle multiple batch effects: scanner manufacturer, field strength (1.5T vs 3T), acquisition year, protocol version.
3. **Longitudinal ComBat** — Preserve within-patient trajectories while removing between-scanner artifacts.

**Papers**:
- ComBat Harmonization (Fortin et al., 2022) — baseline scanner correction
- ComBat Validation (Moyer et al., 2022) — proves it works + warns about over-harmonization
- Generalized ComBat (Horng et al., 2022) — nested multi-batch effects
- Longitudinal ComBat (Beer et al., 2020) — preserves temporal trajectories

**Output**: Harmonized embeddings — scanner-invariant, biology-preserved.

---

#### Step 3.2: Temporal Modeling with TaViT (Week 9)

**What**: Model how tumors evolve over time, handling Yale's irregular scan intervals.

**Why TaViT**: Yale patients have irregular scan intervals (some 3 months apart, some 2 years). Without time encoding → 0.50 AUC (random chance) on irregular data. TaViT's learnable time emphasis is essential.

**Architecture** (Time-Distance Vision Transformer):
- **Input**: Patient's 8 embeddings (8 × 768) + time distances between scans (in days)
- **Temporal Emphasis Model (TEM)**: Learnable sigmoid that scales attention by time distance
  - Recent scan pairs → high attention (TEM ≈ 1.0)
  - Distant scan pairs → low attention (TEM ≈ 0.0)
  - Parameters a, c are learned during training
- **Transformer**: 8 layers, 8 attention heads, with TEM-scaled attention
- **Output**: CLS token → 768-dim temporal representation (summarizes entire patient trajectory)

| Model | AUC | Notes |
|---|---|---|
| TaViT | 0.786 | Learnable time emphasis — best |
| No time encoding | 0.500 | Random chance on irregular data |
| Single-scan CNN | 0.734 | No temporal modeling |

**Paper**: Time-Distance ViT (Hager et al., 2022) — temporal emphasis for irregular intervals  
**Code**: [GitHub](https://github.com/tom1193/time-distance-transformer) (public)

---

#### Step 3.3: Self-Supervised Pretraining (Week 9–10)

**What**: Pre-train TaViT on Yale without labels using masked autoencoder strategy.
1. Take a patient's 8 embeddings, mask 75% (keep 2 visible)
2. Model predicts masked embeddings from visible ones + time info
3. Pre-train on ALL 11,884 scans → fine-tune on downstream tasks

**What it learns**: Tumor growth rates, spread patterns, treatment response signatures.

---

#### Step 3.4–3.5: Training & Evaluation (Week 10–11)

**Data split**: 1,000 train (70%) / 215 validation (15%) / 215 test (15%) patients.

**Evaluation**:
- **Downstream prediction**: Tumor growth classification, treatment response (AUC, F1)
- **Clustering**: t-SNE/UMAP colored by tumor stage (should cluster), scanner (should mix if ComBat worked)
- **Temporal consistency**: Is embedding distance proportional to time distance?

**Supporting papers**:
- TransXAI (Zeineldin et al., 2024) — Grad-CAM explainability for ViT models (applied in Phase 4)
- CAFNet (Ahmed et al., 2025) — validates hybrid CNN+ViT design (pure ViT: 87.3% → hybrid: 96.4%)

---

#### ✅ Phase 2–3 Deliverables
- CNN baseline benchmarks
- 768-dim harmonized embeddings for all scans
- TaViT temporal model capturing tumor evolution
- ViT outperforms CNN baseline on downstream tasks
- Clustering shows biologically meaningful groupings

---

## Objective 3 — Integrate Imaging and Clinical Context Using LLMs

> **project.txt**: "Enable multimodal clinical reasoning by combining visual tumor representations with structured clinical metadata using LLMs."

### Phase 4 — LLM Integration (Weeks 12–14)

#### Step 4.1: Design Multimodal Representations (Week 12)

**What**: Fuse imaging embeddings with Yale's clinical metadata.

**Inputs to combine**:
1. **Imaging**: Swin UNETR 768-dim embeddings per scan + TaViT temporal summary
2. **Clinical metadata**: Age, sex, primary cancer type, stage, treatment history (surgery, radiation, chemo dates)

**Fusion**: RadFM's Perceiver compresses each scan's 768-dim embedding to 32 fixed tokens. Multiple scans (T0, T1, T2) are interleaved with clinical metadata tokens.

---

#### Step 4.2: Build LLM Pipeline (Week 13)

**What**: Connect vision features to a medical language model for clinical narrative generation.

**Architecture**:
```
Swin UNETR (768-dim) → Perceiver (32 tokens per scan) → MedLLaMA-13B → Clinical narrative
```

**Key design**: Our Swin UNETR outputs 768-dim — which matches RadFM's ViT output exactly. No adapter needed.

**Prompt design**:
```
Instruction: "Based on this temporal sequence of brain MRIs, describe the disease progression."
Input: [T0 scan tokens] + [T1 scan tokens] + [T2 scan tokens] + [TaViT summary] + [clinical metadata text]
Output: "15mm enhancing lesion in right frontal lobe. 20% growth T0→T1 (6 months). 
         Stabilized after radiosurgery. Recommend: surveillance, next scan 3 months."
```

**Novel contribution**: No existing paper feeds temporal 3D MRI sequences → LLM for longitudinal progression reports.

---

#### Step 4.3: Train & Generate (Week 14)

**Two-stage training** (from RadFM):
1. **Stage 1 — Alignment** (1–2 epochs): Freeze MedLLaMA-13B, train Perceiver + projection (~50M params). Learn to map Swin UNETR embeddings → LLM input space.
2. **Stage 2 — Instruction Tuning**: Add LoRA adapters (~4M params). Fine-tune on brain tumor progression reports. Use UMLS-weighted loss (medical terms = 3× weight).

**Training awareness** (from MM-Embed, ICLR 2025):
- Watch for **text bias** — LLM ignoring visual features, generating generic reports
- Use **sequential training** (curriculum learning > joint training)
- Use **hard negatives** — wrong reports that look plausible

**Compute**: Realistically 1× A100 40–80GB with LoRA (~54M trainable params).

---

#### Step 4.4: Validate (Week 14)

**Validation**: BLEU, ROUGE (text quality), UMLS precision/recall (medical terms), expert review of 5–10 cases.

**Papers**:
- RadFM (Wu et al., Nature Communications 2025) — 3D vision-language model, Perceiver + MedLLaMA-13B
- MM-Embed (Lin et al., ICLR 2025, NVIDIA) — training methodology: modality bias, curriculum learning

**Explainability** (from TransXAI): Apply Grad-CAM to Swin UNETR decoder layers → heatmaps showing which brain regions drive LLM's narrative. Post-hoc method — no architecture modification, no accuracy tradeoff.

---

#### ✅ Phase 4 Deliverables
- Multimodal representations (imaging + clinical metadata)
- LLM prompt templates for brain tumor progression
- Generated clinical narratives for Yale patients
- Qualitative validation of clinical plausibility

---

## Objective 4 — Generate and Analyze Cancer Progression Videos

> **project.txt**: "Model and visualize cancer progression through AI-generated video sequences conditioned on multimodal inputs."

### Phase 5 — Video Generation (Weeks 15–18)

#### Step 5.1: Adapt Video Diffusion Model (Week 15)

**What**: Build a diffusion model that generates future brain MRI volumes conditioned on patient history and treatment.

**Architecture** (assembled from multiple papers):
```
3D VAE Encoder → Latent space → Temporal Video Diffusion UNet → 3D VAE Decoder → Generated MRI
```

**Components**:

| Component | What it does | From paper |
|---|---|---|
| **3D VAE** | Compresses 128³×4 MRI → 16³×4 latent (512× compression) | LDM (Rombach 2022) + EchoNet-Synthetic |
| **Spatial UNet layers** | Process individual frames — frozen from pre-trained image model | Video LDM (Blattmann 2023) |
| **Temporal attention layers** | Ensure frame-to-frame consistency — trainable (~20% of params) | Video LDM |
| **Treatment conditioning** | Sinusoidal day embedding + learned treatment type → MLP | TaDiff (Liu 2025) |
| **Vision conditioning** | Swin UNETR 768-dim → cross-attention (Q from latent, K/V from embedding) | LDM cross-attention |
| **Text conditioning** | RadFM narrative tokens → cross-attention | Our Objective 3 output |
| **Joint segmentation** | Predicts noise ε̃ AND tumor mask m̃ simultaneously | TaDiff |

**Diffusion process**:
- T=600 steps (sufficient for medical MRI — TaDiff validated)
- Linear noise schedule: β₁=10⁻⁴ to β_T=0.02
- ε-prediction: predict noise, not image (DDPM)
- Training loss: L = E[‖ε − ε_θ(x_t, t)‖²] (simple MSE)
- Inference: DDIM sampler (50–100 steps instead of 600)

**Starting codebase**: EchoNet-Synthetic — the only medical video diffusion paper with full code + weights.
- Clone from GitHub, adapt from 2D echo → 3D brain MRI
- Their 3-model pipeline: VAE → LIDM → LVDM (we skip LIDM, modify LVDM)

---

#### Step 5.2: Generate Progression & Counterfactual Videos (Weeks 16–17)

**Mode 1 — Natural Progression**: Given (T0, T1, T2) → generate (T3, T4, T5). TaViT temporal patterns guide generation.

**Mode 2 — Counterfactual "What-If"**: Fix patient embeddings, change ONLY treatment vector:
- "What if surgery?" → generate post-surgical progression
- "What if radiation?" → generate radiation response
- "What if no treatment?" → generate natural growth baseline

**Key techniques**:
- **Treatment conditioning** (from TaDiff): Each treatment-day pair ⟨τ,d⟩ encoded separately. Treatment adds +3.7% SSIM, +16.3% DSC.
- **Mask dilation** (from MedEdit): Edit dilated region (kernel k=25) around tumor to capture indirect effects (edema, mass effect). Clinical realism = 3.20/5 (matches real samples).
- **Joint prediction** (from TaDiff): Model outputs both noise and tumor masks. Weighted loss ω focuses on tumor boundary regions.
- **Uncertainty**: Generate 5 stochastic samples per condition → mean = prediction, variance = uncertainty map.
- **Video stitching** (from EchoNet-Synthetic): Overlapping chunks for long sequences.

---

#### Step 5.3: Train & Validate (Week 18)

**Training**:
- Train on real Yale longitudinal progressions (1,430 patients, 11,884 scans)
- Given (T0, T1, T2), predict T3 and compare to real T3
- Adam optimizer, lr=2.5×10⁻⁴, cosine decay, batch=32
- Compute: ~500–1000 GPU-hours total

**Validation targets** (from paper benchmarks):
| Metric | Target | Reference |
|---|---|---|
| SSIM | ≥ 0.85 | TaDiff: 0.848 |
| PSNR | ≥ 25 dB | TaDiff: ~24 dB external |
| FID | ≤ 30 | EchoNet-Synthetic: 28.8 |
| Tumor DSC | ≥ 0.65 | TaDiff: 0.719 |
| Clinical realism | ≥ 3.0/5 | MedEdit: 3.20/5 |

**Papers**:
- DDPM (Ho et al., NeurIPS 2020) — diffusion theory foundation
- LDM (Rombach et al., CVPR 2022) — latent diffusion + cross-attention conditioning
- Video LDM (Blattmann et al., CVPR 2023) — temporal layer insertion strategy
- TaDiff (Liu et al., IEEE-TMI 2025) — treatment-conditioned glioma MRI generation (same domain!)
- MedEdit (Ben Alaya et al., MICCAI 2024) — counterfactual brain MRI editing
- EchoNet-Synthetic (Reynaud et al., MICCAI 2024) — medical video diffusion (starting codebase, code available)
- Counterfactual Diffusion AE (Atad et al., 2024) — latent space counterfactual manipulation

---

#### ✅ Phase 5 Deliverables
- 3D VAE encodes/decodes brain MRI volumes
- Video diffusion generates temporally consistent future scans
- Counterfactual trajectories show plausible treatment effects
- Quantitative metrics (FID, SSIM, PSNR, DSC) computed

---

## Objective 5 — Evaluate Explainability, Clinical Plausibility, and Scientific Impact

> **project.txt**: "Rigorously assess the performance, interpretability, and medical relevance of the proposed multimodal framework."

### Phase 6 — Evaluation & Final Deliverables (Weeks 19–20)

#### Step 6.1: Quantitative Evaluation (Week 19)

**Metrics across all objectives**:
- **Segmentation**: Dice score, Hausdorff distance (Obj 1)
- **Representation quality**: AUC, accuracy, clustering quality (Obj 2)
- **Report quality**: BLEU, ROUGE, UMLS precision (Obj 3)
- **Video quality**: FID, SSIM, PSNR, FVD, tumor DSC (Obj 4)
- **Temporal coherence**: Frame-to-frame consistency measures (Obj 4)
- **Optional external validation**: Test on Cyprus (different population, scanners)

#### Step 6.2: Qualitative Evaluation (Week 19–20)

- Visual plausibility of generated videos (blind assessment, 1–5 scale)
- Clinical coherence of LLM narratives
- Counterfactual scenarios match clinical expectations (radiation → shrinkage, no treatment → growth)
- Explainability: Grad-CAM heatmaps align with clinical reasoning

#### Step 6.3: Documentation & Publication (Week 20)

- Final technical report
- Publication-ready manuscript draft
- Knowledge transfer presentation

---

#### ✅ Phase 6 Deliverables
- Quantitative evaluation report
- Clinical validation results
- Publication-ready manuscript
- Final presentation

---

## Essential Papers Summary (No Repetition)

### Objective 1 — Preprocessing (5 essential papers)

| Paper | Year | Role in Pipeline |
|---|---|---|
| BraTS Toolkit (Kofler et al.) | 2020 | HD-BET skull stripping, z-score normalization |
| nnU-Net (Isensee et al.) | 2021 | Tumor segmentation (Dice ~0.908) |
| itk-elastix (Niessen et al.) | 2023 | Temporal registration (0.95 correlation, brain-validated) |
| Yale Dataset (Ramakrishnan et al.) | 2025 | Primary dataset: 11,884 scans, 1,430 patients |
| Cyprus Dataset (Trimithiotis et al.) | 2025 | Validation dataset: 744 scans, 40 patients, expert labels |

### Objective 1 — Harmonization (4 essential papers)

| Paper | Year | Role in Pipeline |
|---|---|---|
| ComBat Guide (Fortin et al.) | 2022 | Baseline scanner harmonization method |
| ComBat Validation (Moyer et al.) | 2022 | Proves ComBat works; warns: test before applying |
| Generalized ComBat (Horng et al.) | 2022 | Nested multi-batch effects (scanner + year + field strength) |
| Longitudinal ComBat (Beer et al.) | 2020 | Preserves within-patient trajectories |

### Objective 2 — ViT Representations (4 essential papers)

| Paper | Year | Role in Pipeline |
|---|---|---|
| Swin UNETR (Tang et al.) | 2022 | 768-dim embeddings + segmentation (Dice 0.913) |
| Time-Distance ViT / TaViT (Hager et al.) | 2022 | Temporal attention for irregular intervals (AUC 0.786) |
| TransXAI (Zeineldin et al.) | 2024 | Grad-CAM explainability for ViT (used in Obj 3) |
| CAFNet (Ahmed et al.) | 2025 | Validates hybrid CNN+ViT design (87% → 96%) |

### Objective 3 — LLM Integration (2 essential papers)

| Paper | Year | Role in Pipeline |
|---|---|---|
| RadFM (Wu et al.) | 2025 | Perceiver + MedLLaMA-13B architecture (Nature Comms) |
| MM-Embed (Lin et al.) | 2025 | Training methodology: modality bias, curriculum learning (ICLR) |

### Objective 4 — Video Generation (7 essential papers)

| Paper | Year | Role in Pipeline |
|---|---|---|
| DDPM (Ho et al.) | 2020 | Diffusion theory: ε-prediction, L_simple |
| LDM (Rombach et al.) | 2022 | Latent diffusion + cross-attention conditioning |
| Video LDM (Blattmann et al.) | 2023 | Temporal layer insertion into frozen image model |
| TaDiff (Liu et al.) | 2025 | Treatment-conditioned glioma generation (same domain!) |
| MedEdit (Ben Alaya et al.) | 2024 | Counterfactual brain MRI editing, clinical validation |
| EchoNet-Synthetic (Reynaud et al.) | 2024 | Medical video diffusion codebase (code available) |
| Counterfactual Diff. AE (Atad et al.) | 2024 | Latent space counterfactual manipulation |

**Total: 22 essential papers across 5 objectives.**

---

## Key Technical Decisions

1. **ViT extracts features, not manual radiomics** — Swin UNETR learns richer 768-dim representations than hand-crafted features.

2. **Swin UNETR for dual purpose** — Same model handles segmentation (decoder, Dice 0.913) AND embedding extraction (encoder, 768-dim).

3. **TaViT for irregular temporal data** — Yale has irregular scan intervals. Without time encoding → 0.50 AUC. TaViT's learnable emphasis is essential.

4. **itk-elastix for registration** — Pip-installable, brain-validated (0.95 correlation), Python/MONAI native, ready parameter files. FLIRE dismissed (code not released, breast-only, MATLAB).

5. **ComBat on embeddings, not raw images** — Harmonize after ViT extraction. Nested ComBat for multi-batch, Longitudinal ComBat for temporal preservation.

6. **Post-hoc Grad-CAM explainability** — No architecture modification, no accuracy tradeoff.

7. **RadFM Perceiver for vision-to-LLM bridge** — 768-dim Swin UNETR → 32 fixed tokens → MedLLaMA-13B. No adapter needed (dimension match).

8. **EchoNet-Synthetic as starting codebase** — Only medical video diffusion with full code + weights. Adapt from 2D echo → 3D brain MRI.

9. **TaDiff treatment conditioning** — Sinusoidal day embedding + learned treatment type → MLP. Adds +3.7% SSIM, +16.3% DSC.

10. **MedEdit counterfactual formulation** — Mask dilation (k=25) for indirect effects. Clinical realism matching real samples (3.20/5).

---

## Timeline Summary (Gantt-style)

| Week | Phase | Objective | Key Activity |
|---|---|---|---|
| 1 | Phase 1 | Obj 1 | Download, preprocess (HD-BET, normalize) |
| 2 | Phase 1 | Obj 1 | nnU-Net tumor segmentation |
| 3 | Phase 1 | Obj 1 | itk-elastix temporal registration |
| 4 | Phase 1 | Obj 1 | EDA + quality validation |
| 5–6 | Phase 2 | Obj 2 | CNN baseline implementation |
| 7 | Phase 2 | Obj 2 | Baseline evaluation + limitations |
| 8 | Phase 3 | Obj 2 | Swin UNETR embeddings + ComBat harmonization |
| 9 | Phase 3 | Obj 2 | TaViT temporal modeling + pretraining |
| 10 | Phase 3 | Obj 2 | Training |
| 11 | Phase 3 | Obj 2 | Representation evaluation |
| 12 | Phase 4 | Obj 3 | Multimodal representation design |
| 13 | Phase 4 | Obj 3 | RadFM pipeline integration |
| 14 | Phase 4 | Obj 3 | Training + narrative generation + validation |
| 15 | Phase 5 | Obj 4 | Adapt video diffusion model |
| 16–17 | Phase 5 | Obj 4 | Generate progression + counterfactual videos |
| 18 | Phase 5 | Obj 4 | Train + validate on Yale sequences |
| 19 | Phase 6 | Obj 5 | Quantitative + qualitative evaluation |
| 20 | Phase 6 | Obj 5 | Final report + publication draft |
