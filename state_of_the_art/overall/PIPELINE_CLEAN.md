# 🧠 The Pipeline — Explained Like You're 10

> **Mohamed** | **Supervisor**: Dr. Belkacem Chikhaoui | **TÉLUQ University, Montreal**  
> **20 weeks** (April–August 2026) | Mitacs Globalink Research Internship  
> **14 papers. 5 objectives. 6 phases. No fluff.**

---

## What Are We Building?

Imagine a doctor has brain scans of a patient taken every few months for years. Right now, the doctor looks at each scan one by one and tries to guess: "Is the tumor growing? Shrinking? What if we tried radiation instead of surgery?"

**We're building an AI that**:
1. **Looks** at ALL the scans together (not one at a time)
2. **Understands** the tumor's journey over time
3. **Explains** what's happening in plain English ("The tumor grew 15% after stopping chemo")
4. **Imagines** what WOULD have happened under different treatments (like a crystal ball)
5. **Shows** this as a video — smooth tumor evolution, not static pictures

```
The whole thing in one line:

Brain MRI → Clean it → Segment tumor → Align over time → ViT understands it → LLM explains it → Diffusion imagines the future
```

---

## Our Data

We use **one main dataset** and **one optional checker**:

| | **Yale (Training)** | **Cyprus (Validation)** |
|---|---|---|
| What | 11,884 brain MRI scans | 744 brain MRI scans |
| Patients | 1,430 | 40 |
| Scans/patient | ~8 over time | ~18.6 over time |
| Years | 2004–2023 | 2019–2024 |
| Modalities | T1, T1c, T2, FLAIR | T1, T1c, T2, FLAIR |
| Tumor labels | ❌ None (we generate them) | ✅ Expert-drawn |
| Treatment info | ✅ Surgery, radiation, chemo | ✅ Available |
| Cost | Free (TCIA) | Free (Zenodo) |

**Strategy**: Train everything on Yale. If time permits, test on Cyprus to prove it works on totally different patients/scanners.

📄 **Papers**: `04_Yale_Dataset` + `05_Cyprus_Dataset`

---

## Objective 1 — Build a Clean Data Pipeline

> *"Get the messy hospital scans into a clean, organized format the AI can actually use."*

### Phase 1 (Weeks 1–4)

Think of it like cleaning a messy room before you can study in it.

#### Step 1: Download & Clean (Week 1)

**Problem**: Hospital scans come from different machines, different years, different settings. Some have skulls, some are bright, some are dark.

**What we do**:
1. Download Yale from TCIA (~200 GB)
2. **Skull stripping** with HD-BET — remove everything that's not brain (skull, skin, eyes). Like cropping a photo to just the face.
3. **Normalize intensity** — make all scans have the same brightness scale (z-score per modality). Like adjusting all photos to the same brightness.
4. **Resample** to 128×128×128 — uniform size for the AI.

📄 **Paper**: `01_BraTS_Toolkit` — gives us HD-BET + normalization recipe  
🔧 **Tools**: HD-BET, BraTS Toolkit, MONAI, SimpleITK

**Output**: 11,884 clean, skull-stripped, normalized brain scans.

---

#### Step 2: Find the Tumors (Week 2)

**Problem**: Yale scans have NO tumor labels. We need to know where tumors are.

**What we do**:
1. Run **nnU-Net** — an AI that automatically finds and outlines tumors into 3 regions:
   - **ET** (Enhancing Tumor) — the active, dangerous part
   - **WT** (Whole Tumor) — everything including swelling
   - **TC** (Tumor Core) — the solid center
2. Quality check: manually look at 50–100 cases to make sure it's reasonable
3. (Optional) Run on Cyprus too → compare to expert labels → know your error rate

**Expected accuracy**: Dice ~0.908 (pretty good!)

📄 **Paper**: `02_nnUNet` — self-configuring segmentation  
**Output**: Tumor masks for all 11,884 scans.

---

#### Step 3: Align Scans Over Time (Week 3)

**Problem**: When a patient comes back 6 months later, they lie slightly differently in the scanner. Their brain is in a different position. You can't compare pixel-by-pixel.

**What we do**: **Registration** — warp each later scan to match the first scan, so the same brain spot is always at the same pixel.

**Tool**: **itk-elastix** — pip install, one-line Python, brain-validated.

How it works (3 automatic stages):
1. **Rigid** — fix head rotation/tilt (6 parameters)
2. **Affine** — fix brain size differences (12 parameters)
3. **B-spline deformable** — handle local changes like tumor growth, swelling (thousands of parameters)

For each patient: pick the first scan as "home base" (T0), then warp T1→T0, T2→T0, ..., T7→T0. Apply the same warp to tumor masks too.

📄 **Paper**: `03_ITK_Elastix` — Python medical image registration  
**Output**: All scans aligned. Same brain spot = same pixel across all visits.

---

#### Step 4: Explore the Data (Week 4)

Look at what we have: How big are tumors? How often do patients get scanned? How many different scanners were used? Document everything.

**Output**: EDA report with stats and plots.

---

#### ✅ Phase 1 Done!
You now have 1,430 patients × ~8 clean, segmented, aligned scans = ready for AI.

---

## Objective 2 — Teach the AI to Understand Tumors

> *"Turn brain scans into numbers the AI can reason about, and teach it that TIME matters."*

### Phase 2 — CNN Baseline (Weeks 5–7)

Before using fancy ViTs, build a simple CNN baseline (3D ResNet or nnU-Net features). Train it on single scans. Measure how well it does.

**Purpose**: You need this baseline to PROVE that your ViT is better. Your paper needs to say "ViT beats CNN by X%."

**Output**: Baseline accuracy numbers + documented limitations of static (one-scan-at-a-time) models.

---

### Phase 3 — ViT + Temporal Modeling (Weeks 8–11)

#### Step 1: Extract Smart Features with Swin UNETR (Week 8)

**What**: Instead of manually measuring tumor size/shape, let a pre-trained Vision Transformer look at each scan and produce a **768-number fingerprint** (called an "embedding") that captures EVERYTHING about that scan — tumor size, shape, location, texture, surrounding tissue, all of it.

**Architecture**: Swin UNETR (61.98M parameters)
- **Input**: 128×128×128 brain scan with 4 channels (T1, T1c, T2, FLAIR)
- **Encoder**: Swin Transformer — processes the scan in patches, builds understanding layer by layer
- **Output**: A single vector of **768 numbers** per scan — the scan's "fingerprint"
- **Bonus**: The decoder also segments tumors (Dice 0.913 — even better than nnU-Net's 0.908!)

Per patient: 8 scans → 8 fingerprints → an (8 × 768) table tracking how the tumor changes.

📄 **Paper**: `08_Swin_UNETR` — available in MONAI: `pip install monai`  
🔧 **Pre-trained weights**: Download `model_swinvit.pt` from MONAI GitHub

---

#### Step 2: Harmonize — Remove Scanner Bias (Week 8)

**Problem**: Yale spans 2004–2023 with different scanners. The 768-number fingerprints accidentally encode "which scanner brand was used" along with the real tumor info. If we don't fix this:
- The AI thinks a patient switching scanners = tumor changed dramatically
- The LLM writes "tumor grew 90%" when it only grew 10%
- Generated videos show weird jumps at scanner-switch points

**Solution**: **ComBat harmonization** on the embeddings (not the raw scans — see `WHY_HARMONIZE_AFTER_VIT.md`).

**Two-step process**:
1. **ComBat** (Fortin 2022) — removes systematic differences between scanners/years/field strengths from the 768-dim embeddings. Think: "adjust all thermometers to read the same temperature."
2. **Longitudinal ComBat** (Beer 2020) — extends ComBat to respect time. It knows that scans from the SAME patient should change smoothly, not jump around. It removes scanner-switch artifacts while preserving real tumor trajectories.

**Why only 2 papers, not 4?** ComBat Guide teaches the method. Longitudinal ComBat fixes its biggest limitation (it didn't handle repeated measurements). The "Validation" and "Generalized" papers are nice-to-know but don't add a new tool — you just apply ComBat and then Longitudinal ComBat on top. Done.

📄 **Papers**: `06_ComBat_Guide` (the method) + `07_Longitudinal_ComBat` (the temporal fix)

**Output**: Harmonized embeddings — scanner effects removed, real biology preserved.

---

#### Step 3: Teach Time with TaViT (Week 9–10)

**Problem**: You have 8 fingerprints per patient, but scans aren't evenly spaced. One gap might be 3 months, another 2 years. A normal AI treats them all equally — bad idea!

**Solution**: **TaViT** (Time-distance Vision Transformer) — a Transformer that KNOWS how far apart scans are in time.

**How it works**: 
- It adds a "time emphasis" to its attention: recent scan pairs get full attention, distant pairs get less
- The formula: TEM(R) = 1 / (1 + exp(a × R - c)), where R = days between scans, a and c are LEARNED
- **Critical finding**: Without time encoding → **0.50 AUC** (= coin flip!) on irregular data. With TaViT → **0.786 AUC**

**Self-supervised pretraining** (no labels needed):
1. Take a patient's 8 fingerprints, hide 75% of them
2. Make the model predict the hidden ones from the visible ones + time info
3. This teaches it tumor growth patterns, treatment responses, spread dynamics — all for free

**Output**: One **768-dim temporal summary** per patient — captures the entire tumor story.

📄 **Paper**: `09_TaViT` — code available on GitHub  

---

#### Step 4: Evaluate (Week 11)

- **Downstream tasks**: Can a simple classifier using ViT embeddings predict growth/shrinkage? (Compare to CNN baseline)
- **Clustering**: Do similar tumors group together in t-SNE/UMAP plots? Do different scanners mix? (If yes → ComBat worked!)
- **Temporal consistency**: Does embedding distance grow with time distance?

**Data split**: 1,000 train / 215 val / 215 test patients.

---

#### ✅ Phases 2–3 Done!
You now have harmonized, time-aware 768-dim representations for every patient. The AI understands tumor trajectories.

---

## Objective 3 — Make the AI Explain Itself

> *"Connect the visual understanding to a language model so it can write clinical reports."*

### Phase 4 — LLM Integration (Weeks 12–14)

#### Step 1: Design the Bridge (Week 12)

**Problem**: ViT outputs 768 numbers. An LLM expects text tokens. How to connect them?

**Solution**: **RadFM's Perceiver** — a module that compresses any-size visual features into exactly **32 tokens** that the LLM can read. Lucky coincidence: Swin UNETR outputs 768-dim, RadFM's ViT outputs 768-dim — perfect match, no adapter needed!

**What goes into the LLM**:
- Patient's scans: T0, T1, T2 (each → 32 tokens via Perceiver) = vision input
- TaViT temporal summary (32 tokens) = how the tumor changed
- Clinical metadata as text: "67yo female, lung adenocarcinoma, SRS March 2024" = context

#### Step 2: Build & Train (Week 13)

**Architecture**:
```
Swin UNETR 768-dim → Perceiver (32 tokens) → MedLLaMA-13B → Clinical narrative
```

**Two-stage training**:
1. **Alignment** (1–2 epochs): Freeze the LLM, only train the Perceiver bridge. Teach it to translate scan fingerprints into something the LLM understands.
2. **Instruction tuning**: Add LoRA adapters to the LLM (~4M trainable params). Fine-tune on generating brain tumor reports from scan sequences + metadata.

**Critical warning from MM-Embed**: LLMs are LAZY — they prefer reading text and ignoring images! You MUST:
- Train in stages (images first, text second) — curriculum learning
- Use hard negatives (show wrong-but-plausible reports) to force the LLM to actually look at scans
- Monitor for text bias: if it generates the same report for different scans → it's ignoring vision

📄 **Papers**: `10_RadFM` (architecture) + `11_MM_Embed` (training methodology)

**Explainability**: Apply Grad-CAM to Swin UNETR layers → heatmaps showing WHICH brain regions drove the LLM's report. This is standard technique, no extra paper needed.

#### Step 3: Validate (Week 14)

Generate reports for test patients. Check with automatic metrics (BLEU, ROUGE) and have experts review 5–10 cases.

---

#### ✅ Phase 4 Done!
The AI now writes clinical narratives like "15mm lesion in right frontal lobe, 20% growth over 6 months, stabilized after radiosurgery."

---

## Objective 4 — Generate "What-If" Videos

> *"Build a video generator that shows how a tumor WOULD evolve under different treatments."*

### Phase 5 — Video Diffusion (Weeks 15–18)

#### The Big Idea

A **diffusion model** learns to generate realistic images by:
1. **Forward**: Take a real scan → add noise gradually until it's pure static
2. **Backward**: Train a neural network to REMOVE noise step by step → creates realistic scans from noise

**But we don't want random scans**. We want: "Given THIS patient's history and THIS treatment, show me what happens NEXT." So we **condition** the generation on our ViT embeddings + LLM reports + treatment info.

#### Step 1: Build the Generator (Week 15)

**Architecture** (3 key components):

**Component 1 — 3D VAE** (compress brain scans):
- Takes 128³×4 MRI volume → compresses to 16³×4 latent code (512× smaller)
- Diffusion works in this small space (way faster than on full scans)
- From LDM paper — the "latent" in "Latent Diffusion Model"

**Component 2 — Temporal Diffusion UNet** (the brain):
- Spatial layers (frozen from pre-trained LDM) — already know what brain tissue looks like
- Temporal layers (we train these) — learn how brains change over time
- Only ~20% of parameters need training!

**Component 3 — Conditioning** (tell the model WHAT to generate):
- **Treatment**: Each treatment-day pair encoded as a vector using sinusoidal embedding + MLP (from TaDiff — same domain, brain tumors!)
- **Vision**: Swin UNETR 768-dim embedding → cross-attention (model "looks at" the patient's history)
- **Text**: RadFM narrative → cross-attention (model "reads" the clinical context)

**Starting codebase**: **EchoNet-Synthetic** — the only medical video diffusion with full code + weights on GitHub. We clone it and adapt from 2D heart echo → 3D brain MRI.

📄 **Papers**: `12_LDM` (latent diffusion foundation) + `13_TaDiff` (treatment conditioning, same domain!) + `14_EchoNet_Synthetic` (starting code)

---

#### Step 2: Generate Videos (Weeks 16–17)

**Mode 1 — Predict the future**:
- Feed patient history (T0, T1, T2) → model generates T3, T4, T5
- TaViT temporal patterns guide the generation

**Mode 2 — Counterfactual "what-if"**:
- Same patient, but change the treatment vector:
  - "What if surgery?" → generates post-surgical trajectory
  - "What if radiation?" → generates radiation response  
  - "What if no treatment?" → generates natural growth
- Keep everything FIXED except the treatment encoding → see the difference

**How treatment conditioning works** (from TaDiff):
- Each treatment-day pair ⟨treatment_type, day_number⟩ gets its own encoding
- Treatment type → learned embedding table
- Day number → sinusoidal positional encoding
- Combined with MLP → injected into the UNet
- TaDiff proved: with treatment info → SSIM 0.919; without → SSIM 0.882 (+3.7%!)

**Joint prediction** (also from TaDiff): The model predicts BOTH the scan AND the tumor mask at the same time. A special weighting (ω) focuses extra attention on tumor boundaries — the most important region.

---

#### Step 3: Train & Validate (Week 18)

**Training data**: Real Yale progressions. Given (T0, T1, T2), predict T3 and compare to the real T3.

**Training recipe** (from TaDiff):
- Optimizer: Adam, lr=2.5×10⁻⁴, cosine decay
- Diffusion steps: T=600 (TaDiff showed this is enough for medical MRI)
- Noise schedule: linear β from 10⁻⁴ to 0.02
- Loss: simple MSE on predicted noise (ε-prediction)
- Compute: ~500–1000 GPU-hours total

**Targets**:
| Metric | Target | What It Measures |
|---|---|---|
| SSIM | ≥ 0.85 | Structural similarity to real scan |
| PSNR | ≥ 25 dB | Signal quality |
| Tumor DSC | ≥ 0.65 | Tumor shape accuracy |
| Clinical realism | ≥ 3/5 | Does a doctor think it's real? |

---

#### ✅ Phase 5 Done!
You can now generate future brain scans and "what-if" treatment scenarios as video sequences.

---

## Objective 5 — Prove It Works

> *"Measure everything, get expert opinions, write the paper."*

### Phase 6 — Evaluation & Publication (Weeks 19–20)

#### Quantitative (Week 19)
- **Segmentation quality**: Dice score, Hausdorff distance
- **Representation quality**: AUC, accuracy, clustering metrics (ViT vs CNN)
- **Report quality**: BLEU, ROUGE, medical term precision
- **Video quality**: FID, SSIM, PSNR, temporal coherence
- (Optional) Test on Cyprus — different population, different scanners → proves generalization

#### Qualitative (Week 19–20)
- Blind evaluation: show generated scans to expert, rate 1–5
- Counterfactuals: does "radiation → shrinkage" and "no treatment → growth" make clinical sense?
- Grad-CAM heatmaps: does the AI focus on the right brain regions?

#### Publication (Week 20)
- Final technical report
- Publication-ready manuscript draft
- Presentation

---

## The Complete Pipeline — One Picture

```
PHASE 1 (Weeks 1–4): CLEAN THE DATA
  Raw MRI ──→ HD-BET skull strip ──→ z-score normalize ──→ nnU-Net segment tumors ──→ itk-elastix align over time
  📄 BraTS Toolkit         📄 nnU-Net               📄 itk-elastix              📄 Yale + Cyprus datasets

PHASE 2 (Weeks 5–7): CNN BASELINE  
  Aligned scans ──→ 3D ResNet ──→ Single-scan features ──→ Baseline metrics
  (No specific paper — standard practice)

PHASE 3 (Weeks 8–11): ViT UNDERSTANDS TUMORS
  Aligned scans ──→ Swin UNETR ──→ 768-dim embeddings ──→ ComBat + LongComBat harmonize ──→ TaViT temporal model ──→ 768-dim patient summary
  📄 Swin UNETR         📄 ComBat + Longitudinal ComBat          📄 TaViT

PHASE 4 (Weeks 12–14): LLM EXPLAINS IT
  Embeddings + metadata ──→ Perceiver (32 tokens) ──→ MedLLaMA-13B ──→ "Tumor grew 15%, stabilized after radiation"
  📄 RadFM              📄 MM-Embed (training tips)

PHASE 5 (Weeks 15–18): DIFFUSION IMAGINES THE FUTURE
  Embeddings + narratives + treatment ──→ 3D VAE ──→ Temporal Diffusion UNet ──→ Progression videos + counterfactual "what-if"
  📄 LDM (foundation)   📄 TaDiff (treatment conditioning)   📄 EchoNet-Synthetic (starting code)

PHASE 6 (Weeks 19–20): PROVE IT
  Metrics + expert review + publication
```

---

## All 14 Papers — One Table, No Repetition

| # | Paper | Year | Venue | What We Take From It | Phase |
|---|---|---|---|---|---|
| 01 | BraTS Toolkit (Kofler) | 2020 | Frontiers | HD-BET skull stripping + z-score normalization | 1 |
| 02 | nnU-Net (Isensee) | 2021 | Nature Methods | Self-configuring tumor segmentation | 1 |
| 03 | itk-elastix (Niessen) | 2023 | JOSS | Temporal registration — Python, brain-validated | 1 |
| 04 | Yale Dataset (Ramakrishnan) | 2025 | TCIA | Primary dataset: 11,884 scans, 1,430 patients | 1 |
| 05 | Cyprus Dataset (Trimithiotis) | 2025 | Zenodo | Validation: 744 scans, 40 patients, expert labels | 1 |
| 06 | ComBat Guide (Fortin) | 2022 | NeuroImage | Scanner harmonization method for embeddings | 3 |
| 07 | Longitudinal ComBat (Beer) | 2020 | NeuroImage | Temporal extension — preserves patient trajectories | 3 |
| 08 | Swin UNETR (Tang) | 2022 | CVPR | 768-dim embeddings + segmentation (Dice 0.913) | 3 |
| 09 | TaViT (Hager) | 2022 | arXiv | Temporal attention for irregular scan intervals | 3 |
| 10 | RadFM (Wu) | 2025 | Nature Comms | Perceiver + MedLLaMA-13B vision-language bridge | 4 |
| 11 | MM-Embed (Lin) | 2025 | ICLR | Training methodology: modality bias, curriculum learning | 4 |
| 12 | LDM (Rombach) | 2022 | CVPR | Latent diffusion + cross-attention conditioning | 5 |
| 13 | TaDiff (Liu) | 2025 | IEEE-TMI | Treatment-conditioned brain tumor MRI generation | 5 |
| 14 | EchoNet-Synthetic (Reynaud) | 2024 | MICCAI | Medical video diffusion codebase (code available!) | 5 |

---

## What We Cut and Why

| Removed Paper | Why |
|---|---|
| ComBat Validation (Moyer 2022) | Nice-to-know, but doesn't add a new tool. ComBat Guide is enough. |
| Generalized ComBat (Horng 2022) | Nested batches handled by applying ComBat + Longitudinal ComBat sequentially. |
| TransXAI (Zeineldin 2024) | We use Grad-CAM on Swin UNETR — standard technique, no paper needed. |
| CAFNet (Ahmed 2025) | Proves hybrid CNN+ViT works, but Swin UNETR IS already a hybrid. Redundant. |
| DDPM (Ho 2020) | Foundational theory, but LDM builds on it. We implement LDM, not DDPM. |
| Video LDM (Blattmann 2023) | Temporal layer insertion idea already absorbed into TaDiff (more recent, same domain). |
| MedEdit (Ben Alaya 2024) | Counterfactual approach — but TaDiff handles counterfactuals via treatment swapping. Simpler. |
| Counterfactual Diff AE (Atad 2024) | Alternative counterfactual method — TaDiff's approach (swap treatment vector) is cleaner for our use case. |

---

## Key Decisions (Short Version)

1. **Swin UNETR** for features — not manual radiomics. It gives 768 smart numbers per scan.
2. **ComBat + Longitudinal ComBat** on embeddings — not on raw scans. Safer, preserves biology.
3. **TaViT** for time — without it, irregular intervals = coin-flip accuracy.
4. **itk-elastix** for registration — pip install, brain-proven, Python native. 
5. **RadFM architecture** — Perceiver + MedLLaMA-13B. 768-dim match with Swin UNETR = no adapter.
6. **EchoNet-Synthetic codebase** — only medical video diffusion with full code. Adapt, don't build from scratch.
7. **TaDiff treatment conditioning** — same domain (brain tumors!), same modalities (T1/T1c/T2/FLAIR). Proven to add +3.7% SSIM.
