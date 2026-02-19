# 🧠 FINAL PIPELINE — Multimodal AI for Longitudinal Cancer Progression

> **Mitacs Globalink Research Internship — TÉLUQ University, Montreal**
> **Supervisor**: Dr. Belkacem Chikhaoui | **Duration**: 20 weeks (April–August 2026)
> **Dataset**: Yale Glioma MRI (11,884 scans, 1,430 patients) + Cyprus Validation (744 scans, 40 patients)

---

## How to Read This Document

This document follows **exactly** the structure from `project.txt`:
- **5 Objectives** (what we want to achieve)
- **6 Phases** (how we get there, week by week)

For every step, you'll find:
- 🎯 **What we do** — the task in plain English
- 📄 **Which paper** — the one paper that teaches us this (mentioned ONCE in the entire document)
- 💡 **What idea we take** — the specific technique from that paper
- 🔧 **How we implement it** — tools, libraries, concrete steps
- 🚀 **Our improvement** — what we do better or differently

**Total papers: 15** (see folder `final_papers/` for the PDFs to print)

---

## The Big Picture (Explain Like I'm 10)

Imagine you have thousands of brain scan photos of cancer patients, taken over many years. We want to:

1. **Clean up** all the photos so they look consistent (Phase 1)
2. **Teach a computer** to understand what tumors look like (Phase 2-3)
3. **Have AI explain** in words what's happening to the tumor (Phase 4)
4. **Make a video** showing how the tumor might grow or shrink — and show "what if we tried a different medicine?" (Phase 5)
5. **Check our work** — is the video realistic? Are the explanations correct? (Phase 6)

The flow looks like this:

```
Raw MRI Scans
    │
    ▼
[Phase 1] Clean + Align + Remove Scanner Differences
    │
    ▼
[Phase 2] Basic CNN Baseline (benchmark)
    │
    ▼
[Phase 3] Smart ViT Features (768-dim per scan) + Time Encoding
    │
    ▼
[Phase 4] LLM Reads Features → Generates Clinical Explanation
    │
    ▼
[Phase 5] Diffusion Model → Cancer Progression Video + Counterfactuals
    │
    ▼
[Phase 6] Evaluate Everything (metrics + visual + clinical plausibility)
```

---

## OBJECTIVE 1 — Robust Pipeline for Longitudinal Cancer Imaging Analysis

### Phase 1: Oncology Medical Imaging Preparation (Weeks 1–4)

---

#### Step 1.1 — Get the Data

**🎯 What we do**: Acquire two brain tumor MRI datasets — one huge for training, one small for validation.

**📄 Paper 01 — Yale Glioma MRI Dataset** (Ramakrishnan et al., 2025)
- 💡 **Idea**: Largest longitudinal glioma MRI collection available. 11,884 scans from 1,430 patients, spanning 2004–2023, with 4 MRI modalities (T1, T1c, T2, FLAIR). Average 8.3 scans per patient — perfect for learning how tumors change over time.
- 🔧 **Implementation**: Download from TCIA. Each patient has multiple visits over years. Load using PyDICOM → convert to NIfTI. Organize as: `data/patient_id/visit_date/modality.nii.gz`
- 🚀 **Our improvement**: Yale has NO tumor segmentation labels (only raw scans). We solve this in Step 1.3 using automated segmentation. Yale also has 5+ scanner batch effects (see Phase 3, Step 3.3 for harmonization) — we handle that in the feature space, not on raw images.

**📄 Paper 02 — Cyprus Glioma Dataset** (Trimithiotis et al., 2025)
- 💡 **Idea**: Small but gold-standard validation set. 40 patients, 744 scans (avg 18.6 per patient), with **expert-verified 3-subregion tumor labels** (enhancing tumor, non-enhancing, edema) + 110 pre-computed radiomic features.
- 🔧 **Implementation**: Use as held-out test set. Never train on it. Compare our automated segmentation (from Step 1.3) against their expert labels to validate our pipeline.
- 🚀 **Our improvement**: Cyprus is tiny (40 patients) so we can't train on it alone. But its rich labels let us measure exactly how good our automated pipeline is.

---

#### Step 1.2 — Skull Stripping (Remove Non-Brain Tissue)

**🎯 What we do**: Remove the skull, skin, and eyes from every MRI scan so the model only sees brain tissue.

**📄 Paper 03 — BraTS Toolkit** (Kofler et al., 2020)
- 💡 **Idea**: BraTS Toolkit bundles **HD-BET** (AI-based skull stripping) into a single standardized pipeline. HD-BET uses a deep learning model trained specifically on brain MRIs and achieves near-perfect skull removal even on tumor-distorted brains.
- 🔧 **Implementation**: `pip install BraTS-Toolkit`, then run `brats_preprocessor` on every scan. It handles: DICOM→NIfTI conversion, **co-registration of all 4 modalities to a common space (T1→T1c→T2→FLAIR alignment within ONE visit)**, skull stripping via HD-BET, and resampling to 1mm³ isotropic.
- 🚀 **Our improvement**: None needed — HD-BET is already the gold standard. We just use it as-is.

> **Note**: BraTS Toolkit co-registers the 4 modalities (T1/T1c/T2/FLAIR) **within a single visit**. It does NOT align scans **across different visits** (that's Step 1.4 — longitudinal registration with itk-elastix).

---

#### Step 1.3 — Tumor Segmentation (Find the Tumor)

**🎯 What we do**: Automatically draw tumor boundaries on every Yale scan (since Yale has no labels).

**📄 Paper 04 — nnU-Net** (Isensee et al., 2021)
- 💡 **Idea**: nnU-Net is a "self-configuring" segmentation network — you give it data and it automatically picks the best architecture, preprocessing, training schedule, and post-processing. It won BraTS 2020 and 2021 challenges. We use the BraTS-pretrained version to segment tumors into 3 regions: enhancing tumor (ET), tumor core (TC), whole tumor (WT).
- 🔧 **Implementation**: Download BraTS-pretrained nnU-Net weights. Run inference on all 11,884 Yale scans. For each scan: input = 4 modalities (T1/T1c/T2/FLAIR) stacked → output = 3-class segmentation mask.
- 🚀 **Our improvement**: We validate these automated labels against Cyprus's expert labels. If Dice score is below 0.85 for any subregion, we'll fine-tune nnU-Net on a small manually-corrected subset.

> **Why nnU-Net instead of training from scratch?** Because it already achieves Dice ~0.91 on BraTS. Re-inventing the wheel would waste weeks. We use it as a tool, not as our research contribution.

---

#### Step 1.4 — Longitudinal Registration (Align Scans Over Time)

**🎯 What we do**: For each patient, align all their scans to the same coordinate system so we can track exactly which voxel moved where between visits.

**📄 Paper 05 — itk-elastix** (Niessen et al., 2023)
- 💡 **Idea**: ITK-Elastix provides both **rigid** (correct head position) and **deformable** (correct brain shift) registration with full Python API. It produces a deformation field that maps every voxel from scan A to scan B, enabling precise longitudinal comparison.
- 🔧 **Implementation**: For each patient, pick one scan as reference (usually first visit). Register all other visits to it using: (1) Rigid registration first (correct rotation/translation), then (2) B-spline deformable registration (correct brain deformation). Use mutual information as similarity metric (works across different MRI contrasts).
  ```python
  import itk
  fixed = itk.imread("visit1.nii.gz")
  moving = itk.imread("visit3.nii.gz")
  registered, params = itk.elastix_registration_method(fixed, moving,
      parameter_object=itk.ParameterObject.New())
  ```
- 🚀 **Our improvement**: We chose itk-elastix over FLIRE (the other option we studied) because: (1) FLIRE is MATLAB-based with no Python API — would require wrapping. (2) itk-elastix is pure Python, pip-installable, and actively maintained. (3) FLIRE's main advantage (longitudinal-specific registration) can be replicated in itk-elastix by using the first visit as a fixed reference for all subsequent visits.



---

## OBJECTIVE 2 — Learn High-Level Tumor Representations Using Vision Transformers

### Phase 2: Baseline Vision Models (Weeks 5–7)

---

#### Step 2.1 — CNN Baseline

**🎯 What we do**: Train a simple CNN (like ResNet-50) on single time-point tumor classification. This gives us a **baseline** to prove that our ViT approach (Phase 3) is actually better.

- 🔧 **Implementation**: Use MONAI's built-in ResNet-50 or DenseNet-121. Train on Cyprus labels (3-class segmentation). Report Dice scores. This is NOT our research contribution — it's just a benchmark number to beat.
- No paper needed — ResNet is textbook. Every AI paper uses it as a baseline.

---

### Phase 3: Vision Transformer-Based Longitudinal Representation Learning (Weeks 8–11)

---

#### Step 3.1 — Extract Rich Features from Each Scan

**📄 Paper 09 — Swin UNETR** (Tang et al., 2022)
- 💡 **Idea**: Swin UNETR is a 3D medical image segmentation model that combines Swin Transformer (hierarchical, shifted-window attention) with a U-Net decoder. The key for us: its **encoder** produces a **768-dimensional embedding** per scan that captures tumor shape, texture, location, and surrounding tissue context. It's pre-trained on 5,050 CT scans via self-supervised learning, achieving Dice 0.913 on BraTS.
- 🔧 **Implementation**: Load Swin UNETR from MONAI (`monai.networks.nets.SwinUNETR`). Use BraTS-pretrained weights. For each of our 11,884 Yale scans: input = 4 modalities (T1/T1c/T2/FLAIR) → pass through encoder only → output = **768-dim embedding vector**. Store all embeddings in a lookup table: `{patient_id, visit_date} → 768-dim vector`.
  ```python
  from monai.networks.nets import SwinUNETR
  model = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3)
  model.load_state_dict(torch.load("swin_unetr_btcv.pth"))
  # Extract features (no decoder needed)
  features = model.swinViT(x)  # → 768-dim
  ```
- 🚀 **Our improvement**: We use Swin UNETR purely as a **feature extractor**, not for segmentation (nnU-Net already does that in Step 1.3). This lets us leverage its powerful pre-trained representations without re-training. If embeddings don't capture tumor-specific features well enough, we'll fine-tune the encoder on our nnU-Net segmentation masks.

> **Why Swin UNETR, not plain ViT?** Plain ViT processes flat 16×16 patches with no hierarchy. Swin UNETR uses shifted windows at multiple scales (like a pyramid), which is critical for 3D medical images where tumors can be 5mm or 50mm. Also: 768-dim output matches what RadFM expects in Phase 4 — no adapter needed.

---

#### Step 3.2 — Add Time Awareness (Turn Snapshots into a Story)

**📄 Paper 10 — TaViT (Time-Aware Vision Transformer)** (Hager et al., 2022)
- 💡 **Idea**: Standard ViT treats each scan independently — it has no concept of "this scan is from 2 months after the previous one." TaViT adds **time-distance positional embeddings**: instead of just position (patch 1, 2, 3...), it encodes the actual time gap between visits using sinusoidal functions. Critical finding: **without time encoding, AUC drops to 0.50 (random chance)**. With it: 0.786 AUC.
- 🔧 **Implementation**: Take our 768-dim Swin UNETR embeddings. For each patient's scan sequence, add time-distance encoding:
  ```python
  # Time between visits in days
  time_gaps = [0, 90, 180, 365, 730]  # days since first visit
  
  # Sinusoidal time encoding (same idea as positional encoding in Transformers)
  time_embed = sinusoidal_encoding(time_gaps, dim=768)
  
  # Add to visual features
  enriched = swin_embedding + time_embed  # still 768-dim
  
  # Pass through TaViT's temporal transformer layers
  # This learns HOW the tumor changed over time
  temporal_output = tavit_encoder(enriched)  # → 768-dim per visit
  ```
- 🚀 **Our improvement**: TaViT was designed for lung CT (detecting malignancy from temporal patterns). We adapt it to brain MRI gliomas. The core idea (time-aware positional encoding) transfers directly, but we'll need to adjust the time scales (lung screening = annual; glioma monitoring = every 2-3 months).

> **This is where the magic happens**: After this step, each patient has a SEQUENCE of 768-dim vectors, one per visit, where each vector knows "what the tumor looks like" AND "how much time has passed." This temporal story is what Phase 4 (LLM) and Phase 5 (video) will consume.

---

#### Step 3.3 — Harmonization (Remove Scanner Differences from Embeddings)

**🎯 What we do**: Yale data was collected on different scanners, at different hospitals, over 20 years. The same tumor looks different depending on whether it was scanned on a Siemens 3T in 2020 vs. a GE 1.5T in 2008. We need to remove these **batch effects** while keeping the real biological signal (tumor changes).

**This is a TWO-PART process applied to our 768-dim embeddings:**

##### Part A: Nested ComBat (Remove Multiple Batch Effects)

**📄 Paper 06 — Generalized ComBat** (Horng et al., 2022)
- 💡 **Idea**: Standard ComBat can only handle ONE batch variable at a time (e.g., just "scanner manufacturer"). But Yale has **5+ simultaneous batch effects**:
  1. **Acquisition year** (2004–2023 — protocols changed drastically)
  2. **Field strength** (41% at 3T, 58% at 1.5T — huge intensity difference)
  3. **Scanner manufacturer** (Siemens 86%, GE 13% — different reconstruction algorithms)
  4. **Imaging site** (multi-site acquisition)
  5. **Unknown effects** (discovered via GMM — Gaussian Mixture Model clustering)

  Generalized ComBat introduces **Nested ComBat**: apply ComBat sequentially, one batch effect at a time, in order from largest effect to smallest. It also uses **GMM** to discover hidden batch effects that you didn't even know existed.
  
- 🔧 **Implementation**:
  ```python
  # Input: 11,884 scans × 768-dim embeddings (from Swin UNETR + TaViT)
  # Metadata: Year, Field, Manufacturer, Site for each scan
  
  from neuroCombat import neuroCombat
  
  embeddings = np.array(all_embeddings)  # Shape: (11884, 768)
  
  # Step 1: Remove Year effect (2004-2023 protocol drift)
  embeddings = neuroCombat(embeddings, batch=year_labels)
  
  # Step 2: Remove Field Strength effect (1.5T vs 3T)
  embeddings = neuroCombat(embeddings, batch=field_strength)
  
  # Step 3: Remove Manufacturer effect (GE vs Siemens)
  embeddings = neuroCombat(embeddings, batch=manufacturer)
  
  # Step 4: Remove Site effect
  embeddings = neuroCombat(embeddings, batch=site)
  
  # Step 5: Discover + remove hidden batches using GMM
  from sklearn.mixture import GaussianMixture
  gmm = GaussianMixture(n_components=3)
  hidden_batches = gmm.fit_predict(embeddings)
  embeddings = neuroCombat(embeddings, batch=hidden_batches)
  ```
  Order matters — Year first because protocol changes between 2004–2023 are the biggest source of variation.
  
- 🚀 **Our improvement**: Before EACH ComBat step, we **test whether that batch effect actually exists** using Kruskal-Wallis test (p < 0.05). This prevents harming data by "correcting" something that isn't there. This lesson comes from the ComBat Validation literature (Moyer et al., 2022): harmonizing when there's no scanner effect can actually REDUCE data quality.

> **Why not just use regular ComBat?** Because if you only correct for "manufacturer" but ignore that 3T vs 1.5T gives completely different intensities, you've removed 1 of 5 problems. The remaining 4 still contaminate your features.

##### Part B: Longitudinal ComBat (Preserve Temporal Trajectories)

**📄 Paper 07 — Longitudinal ComBat** (Beer et al., 2020)
- 💡 **Idea**: Standard ComBat treats every scan as independent — it doesn't know that scan #1 and scan #5 come from the same patient over time. Longitudinal ComBat adds a **random intercept model** that says "each patient has a baseline that should stay consistent." This preserves within-patient trajectories (the tumor is genuinely getting bigger) while removing scanner effects.
- 🔧 **Implementation**: After Nested ComBat removes cross-sectional batch effects, apply Longitudinal ComBat on the result. It uses patient ID as the random effect, preserving the temporal signal. Uses the `longCombat` R package (we'll call it from Python via `rpy2`, or rewrite the key function in Python).
  ```python
  # After nested ComBat, apply longitudinal ComBat
  from rpy2.robjects.packages import importr
  longCombat = importr('longCombat')
  
  # Input: embeddings + patient IDs + timepoints
  harmonized = longCombat.longCombat(
      dat=embeddings,
      batch=scanner_batch,
      subject=patient_ids,
      time=timepoints
  )
  ```
- 🚀 **Our improvement**: We combine Nested + Longitudinal (no existing paper does both). Nested ComBat cleans cross-sectional batch effects (scanner, year, field). Longitudinal ComBat then preserves within-patient time trajectories. Together, they give us features that are clean AND temporally meaningful.

##### Part C: Why Harmonize Embeddings, Not Raw Images?

**📄 Paper 08 — ComBat Harmonization Guide** (Fortin et al., 2022)
- 💡 **Idea**: ComBat was originally designed for genomics (removing lab batch effects from gene expression). In neuroimaging, it works on extracted features (radiomic features, ViT embeddings) rather than raw voxel intensities. The key insight: harmonize the **feature space**, not the raw images.
- 🔧 **Implementation**: We apply ComBat on ViT embeddings (768-dim vectors), not on raw MRIs. 
  
  **Why this order?**
  
  | If you harmonize... | Problem |
  |---|---|
  | **Raw images** (before ViT) | ❌ ComBat might remove real biological signal! Different scanners capture different tissue contrasts → ComBat could erase tumor heterogeneity thinking it's scanner noise |
  | **ViT embeddings** (after ViT) | ✅ ViT already learned what's biology vs. scanner artifact → ComBat only removes the "scanner fingerprint" from the 768-dim vector, preserving tumor info |
  
  **Technical reasons**:
  1. **Dimensionality**: ViT embeddings = 768 dimensions (tractable). Raw voxels = 240×240×155 = 9 million per scan (computationally impossible)
  2. **Distributional assumptions**: ComBat assumes features are roughly Gaussian. ViT embeddings tend to be Gaussian. Raw voxel intensities are NOT.
  3. **Robustness**: ViT learns from diverse scanners (data diversity = better features). Then ComBat cleans the output.
  
- 🚀 **Our improvement**: By harmonizing in feature space, we get the best of both worlds: ViT learns from raw image diversity (seeing all scanner types makes it more robust), and then ComBat cleans the resulting features for downstream tasks.

> **Complete harmonization flow**:
> ```
> Raw MRI (with scanner noise) → Swin UNETR → 768-dim embedding → TaViT time encoding
>     → Nested ComBat (Year→Field→Manufacturer→Site→GMM)
>     → Longitudinal ComBat (preserve patient trajectories)
>     → Clean, temporally-consistent 768-dim embedding → Feed to RadFM
> ```
>
> **ELI10**: Imagine you have 1,000 photos of the same person taken with 10 different cameras. Some cameras make skin look redder, some bluer. If you try to "fix" the colors BEFORE teaching AI to recognize faces, you might accidentally erase freckles or birthmarks. Instead: let AI learn faces from all camera types (it learns "real face features"), THEN remove the "camera fingerprint" from AI's summary (the 768 numbers), keeping the real face info.

---

## OBJECTIVE 3 — Integrate Imaging and Clinical Context Using LLMs

### Phase 4: Multimodal Integration and Clinical Reasoning (Weeks 12–14)

---

#### Step 4.1 — Bridge Vision and Language (Compress Visual Features for the LLM)

**📄 Paper 11 — RadFM** (Wu et al., 2025)
- 💡 **Idea**: RadFM is the first **radiology foundation model** that takes 3D medical images + text and does question answering. The brilliant part for us: its **Perceiver module** compresses ANY medical image into exactly **32 tokens** that an LLM can understand. It uses cross-attention — 32 learnable query tokens attend to the input features and extract the most important information. The LLM backbone is **MedLLaMA-13B** (fine-tuned on medical text).
- 🔧 **Implementation**: 
  1. Take our harmonized 768-dim embeddings from Phase 3
  2. Feed them into RadFM's Perceiver: `768-dim embedding → cross-attention with 32 queries → 32 tokens`
  3. Concatenate these 32 visual tokens with text tokens (patient history, treatment info)
  4. Feed everything into MedLLaMA-13B
  5. Generate clinical narrative: "The tumor shows 15% volume increase in the enhancing region over 6 months, consistent with progression under TMZ treatment..."
  ```
  Input:  [32 visual tokens] + [text: "Patient 47, GBM Grade IV, TMZ since Jan 2020"]
  Output: "The T1c-enhancing component has grown from 12cm³ to 18cm³ (+50%)
           over the past 3 visits, suggesting treatment resistance..."
  ```
- 🚀 **Our improvement**: RadFM uses a single 3D scan as input. We extend it to handle a **sequence** of scans (the patient's full longitudinal history). We do this by feeding each visit's 32-token representation as a separate segment, with time markers between them: `[VISIT_1] 32 tokens [TIME: +6 months] [VISIT_2] 32 tokens [TIME: +12 months] [VISIT_3] 32 tokens → LLM generates temporal narrative`.

> **Why RadFM and not just GPT-4V?** GPT-4V is a black box API — we can't modify it, can't train it on our data, can't access intermediate representations. RadFM is open-source (code + weights), uses MedLLaMA-13B which fits on 2× A100 GPUs, and its Perceiver module is modular — we can plug in our own visual features.

> **Key number**: RadFM's input expects **768-dim** features. Our Swin UNETR produces exactly 768-dim. No adapter layer needed — they match perfectly by design.

---

#### Step 4.2 — Embedding Alignment (Supervisor's Paper)

**📄 Paper 12 — MM-Embed** (Lin et al., 2025 — from Dr. Chikhaoui's research group)
- 💡 **Idea**: MM-Embed proposes a framework for aligning embeddings from different modalities (vision, text, structured data) into a shared representation space. It uses contrastive learning to ensure that a scan showing "tumor growing" is close in embedding space to the text "disease progression."
- 🔧 **Implementation**: Use MM-Embed's contrastive alignment strategy to train a projection head that maps our ViT embeddings and text embeddings into the same space. This improves RadFM's ability to connect visual evidence with textual explanations.
- 🚀 **Our improvement**: We apply this alignment specifically for longitudinal sequences — aligning the CHANGE between two consecutive ViT embeddings with the corresponding textual description of what changed. This is more nuanced than aligning single images to single descriptions.

---

## OBJECTIVE 4 — Generate and Analyze Cancer Progression Videos

### Phase 5: Generative Video Modeling of Cancer Progression (Weeks 15–18)

---

#### Step 5.1 — Foundation: Latent Diffusion (Don't Generate in Pixel Space)

**📄 Paper 13 — Latent Diffusion Models (LDM)** (Rombach et al., 2022)
- 💡 **Idea**: Instead of running the noisy diffusion process on full-resolution images (computationally insane), first compress images into a small **latent space** using a VAE (Variational Autoencoder), run diffusion there, then decode back. A 256×256 image becomes a 32×32 latent — **64× fewer pixels** to denoise. LDM also introduces **cross-attention conditioning**: inject any external signal (text, class label, embeddings) into the denoising process via cross-attention layers.
- 🔧 **Implementation**: Train a VAE on our brain MRI slices: `MRI slice (240×240) → Encoder → latent (30×30×4) → Decoder → MRI slice`. Then train the diffusion UNet in this latent space. The compression factor `f=4-8` is optimal (from the paper's ablation).
- 🚀 **Our improvement**: We condition the LDM on our **harmonized ViT embeddings** (from Phase 3) via cross-attention. This means the diffusion model doesn't just generate random brain MRIs — it generates one that matches the patient's tumor state at a specific time point.

> **Why LDM and not raw DDPM?** DDPM works directly on pixels — for 3D medical volumes, this is computationally impossible (240×240×155 = 9 million voxels per scan). LDM compresses to latent space first, making the problem tractable. We DON'T need the DDPM paper separately because LDM includes and extends DDPM's theory.

---

#### Step 5.2 — Treatment-Conditioned Temporal Generation (The Core Contribution)

**📄 Paper 14 — TaDiff (Treatment-Aware Diffusion)** (Liu et al., 2025)
- 💡 **Idea**: This paper does **exactly our project** for brain tumors. It generates future MRI scans conditioned on (1) current scan + (2) treatment type + (3) time gap. Key innovations:
  - **Treatment conditioning**: encodes treatment type (chemoradiation, TMZ, surgery) via embedding table → MLP, injected into UNet
  - **Time conditioning**: encodes days-since-treatment via sinusoidal embedding → MLP, injected into UNet
  - **Joint generation + segmentation**: simultaneously generates the future MRI AND predicts the tumor mask — this keeps the generated tumors anatomically correct
  - **Tumor-focused loss weighting**: weights the loss function 5× higher on tumor boundary voxels, so the model pays extra attention to getting the tumor right
  - **Counterfactual generation**: give it treatment A → see one future; give it treatment B → see a different future for THE SAME patient. This is the "what if" capability.

- 🔧 **Implementation**:
  1. Train TaDiff's modified UNet on our Yale data:
     - Input: source MRI sequence (3 past visits concatenated) + noise
     - Conditioning: treatment type embedding + time gap embedding + ViT embedding (from Phase 3)
     - Output: predicted noise (for generation) + predicted tumor mask (for segmentation)
  2. Training: T=600 diffusion steps, linear β schedule, ~5M iterations on V100
  3. Inference: given a patient's history + chosen treatment → generate the next scan
  4. Counterfactual: generate two futures (treatment A vs. treatment B) for same patient → show both as video

- 🚀 **Our improvement**:
  1. **3D instead of 2D**: TaDiff processes 2D slices. We extend to 3D volumes (or pseudo-3D: generate slice-by-slice with 3D consistency loss)
  2. **Richer conditioning**: TaDiff uses only treatment + time. We add our **full ViT embedding** (768-dim) as additional conditioning via cross-attention, giving the model much more context about tumor state
  3. **Longitudinal conditioning**: TaDiff uses 3 past visits. With Yale's avg 8.3 visits per patient, we can condition on MORE history for better predictions

> **Key results from TaDiff**: SSIM 0.919, PSNR 27.97 dB. Without treatment conditioning, SSIM drops to 0.882 and tumor Dice drops from 0.719 to 0.556 — proving treatment info is critical, not just noise.

---

#### Step 5.3 — Video Generation Architecture (Turn Frames into Video)

**📄 Paper 15 — EchoNet-Synthetic** (Reynaud et al., 2024)
- 💡 **Idea**: EchoNet-Synthetic generates **realistic medical videos** (echocardiograms) using a 3-stage pipeline: (1) VAE compresses frames to latent space, (2) Latent Image Diffusion Model (LIDM) generates individual frames, (3) **Latent Video Diffusion Model (LVDM)** ensures temporal consistency across frames. The LVDM adds **temporal attention layers** inside the UNet — after each spatial attention block, there's a temporal attention block that looks across frames to ensure smooth motion.
- 🔧 **Implementation**: Adapt EchoNet-Synthetic's 3-stage pipeline to brain MRI:
  1. **VAE**: train on our brain MRI slices (reuse LDM's VAE from Step 5.1)
  2. **LIDM**: generate individual future brain scan frames (reuse TaDiff from Step 5.2)
  3. **LVDM**: add temporal attention to ensure frames flow smoothly as a video
  - Key adaptation: EchoNet generates cardiac cycles (periodic motion). Brain tumors have NON-periodic progression (growth, shrinkage, recurrence). We modify the temporal attention to handle irregular time gaps using TaViT's time-distance encoding (from Step 3.2).
- 🚀 **Our improvement**: 
  1. **Full code + weights available** — EchoNet-Synthetic is our starting codebase. We adapt it rather than building from scratch.
  2. **Irregular temporal spacing**: EchoNet assumes regular frame intervals (30fps video). Our visits are months apart with irregular gaps. We inject time-distance encoding into the temporal attention layers.
  3. **Multimodal conditioning**: EchoNet generates unconditionally. We condition on treatment + time + ViT features + LLM narration (from Phase 4).

---

## OBJECTIVE 5 — Evaluate Explainability, Clinical Plausibility, and Scientific Impact

### Phase 6: Evaluation, Explainability, and Final Deliverables (Weeks 19–20)

---

#### Step 6.1 — Quantitative Evaluation

**🎯 What we do**: Measure everything with numbers.

| What We Measure | Metric | Target | How |
|---|---|---|---|
| Generated image quality | SSIM, PSNR, FID | SSIM > 0.85 | Compare generated future scans to real future scans |
| Tumor segmentation accuracy | Dice score | > 0.85 | Compare our masks to Cyprus expert labels |
| Temporal consistency | Frame-to-frame SSIM | > 0.90 | Consecutive frames shouldn't jump wildly |
| Treatment effect | Δ between counterfactuals | Visible difference | Treatment A video vs. Treatment B video |
| Feature harmonization quality | Silhouette score by scanner | ~0 (no scanner clustering) | After ComBat, features shouldn't cluster by scanner |
| Time encoding importance | AUC with vs. without | +0.15 improvement | Ablation: remove time encoding, measure drop |

---

#### Step 6.2 — Qualitative Evaluation

**🎯 What we do**: Have humans look at our outputs and judge them.

- **LLM narratives**: Are the generated clinical explanations medically accurate? Do they correctly describe what's happening in the images?
- **Generated videos**: Do they look like real brain MRI progression? Are tumors anatomically plausible?
- **Counterfactuals**: Do the two treatment scenarios show meaningfully different trajectories?
- Use **Grad-CAM** on our ViT to visualize which brain regions the model focuses on. T1c (contrast-enhanced) should dominate attention for enhancing tumors.

---

#### Step 6.3 — Publication and Deliverables

- Technical report with full methodology and results
- Publication-ready manuscript draft
- Open-source code + trained model weights
- Generated video examples as supplementary material

---

## 📁 Paper Folder Summary (`final_papers/`)

| # | Paper | What We Take | Phase |
|---|---|---|---|
| 01 | Yale Glioma Dataset (Ramakrishnan 2025) | 11,884 scans, our main training data | 1 |
| 02 | Cyprus Glioma Dataset (Trimithiotis 2025) | 744 labeled scans, validation ground truth | 1 |
| 03 | BraTS Toolkit (Kofler 2020) | HD-BET skull stripping pipeline | 1 |
| 04 | nnU-Net (Isensee 2021) | BraTS-pretrained tumor segmentation | 1 |
| 05 | itk-elastix (Niessen 2023) | Longitudinal registration (rigid + deformable) | 1 |
| 06 | Generalized ComBat (Horng 2022) | Nested ComBat for 5+ Yale batch effects | 1→3 |
| 07 | Longitudinal ComBat (Beer 2020) | Preserve within-patient temporal trajectories | 1→3 |
| 08 | ComBat Guide (Fortin 2022) | ComBat theory + feature-space application | 1→3 |
| 09 | Swin UNETR (Tang 2022) | 768-dim feature extractor, pre-trained encoder | 3 |
| 10 | TaViT (Hager 2022) | Time-distance positional encoding for temporal awareness | 3 |
| 11 | RadFM (Wu 2025) | Perceiver (32 tokens) + MedLLaMA-13B for clinical reasoning | 4 |
| 12 | MM-Embed (Lin 2025) | Contrastive embedding alignment (supervisor's paper) | 4 |
| 13 | LDM (Rombach 2022) | Latent diffusion + VAE + cross-attention conditioning | 5 |
| 14 | TaDiff (Liu 2025) | Treatment-aware generation + joint seg + counterfactuals | 5 |
| 15 | EchoNet-Synthetic (Reynaud 2024) | Video diffusion pipeline (VAE→LIDM→LVDM), starting code | 5 |

---

## Key Technical Decisions (Quick Reference)

| Decision | Choice | Why |
|---|---|---|
| Skull stripping | HD-BET (via BraTS Toolkit) | Best for tumor-distorted brains |
| Segmentation | nnU-Net (BraTS pretrained) | Self-configuring, Dice ~0.91, no manual tuning |
| Registration | itk-elastix | Python API, pip-installable (FLIRE is MATLAB) |
| Feature extractor | Swin UNETR encoder | 768-dim, hierarchical, pre-trained on 5K CTs, MONAI built-in |
| Time encoding | TaViT sinusoidal | Without it: AUC = 0.50 (random chance!) |
| Multi-batch harmonization | Nested ComBat | Yale has 5+ batch effects, regular ComBat handles only 1 |
| Temporal harmonization | Longitudinal ComBat | Preserves within-patient trajectories over time |
| Harmonization timing | After ViT (feature space) | ViT learns from diversity; ComBat cleans features |
| Vision→Language bridge | RadFM Perceiver (32 tokens) | Compresses any image to fixed-size LLM input |
| LLM backbone | MedLLaMA-13B | Open-source, medical-tuned, fits on 2× A100 |
| Image generation | Latent Diffusion (not pixel DDPM) | 64× fewer computations in latent space |
| Tumor generation | TaDiff conditioning | Treatment + time + ViT features as conditioning |
| Video generation | EchoNet-Synthetic pipeline | Full code + weights, 3-stage (VAE→LIDM→LVDM) |
| Embedding alignment | MM-Embed contrastive | Supervisor's paper, aligns vision ↔ text |

---

## Papers We Studied But Don't Include (and Why)

| Paper | Why Excluded |
|---|---|
| TransXAI (Zeineldin 2024) | Grad-CAM is a standard technique — don't need a paper for it |
| CAFNet (Ahmed 2025) | Validates hybrid CNN-ViT — but Swin UNETR IS already hybrid |
| DDPM (Ho 2020) | LDM paper includes and extends DDPM theory |
| Video LDM (Blattmann 2023) | Architecture absorbed into our TaDiff + EchoNet approach |
| MedEdit (BenAlaya 2024) | TaDiff already does counterfactuals by swapping treatment vectors |
| Counterfactual Diff AE (Atad 2024) | Same — TaDiff approach is cleaner and domain-specific |
| ComBat Validation (Moyer 2022) | Key lesson ("test first with Kruskal-Wallis before harmonizing") incorporated into Phase 3 Step 3.3, don't need the full paper |

---

*Last updated: pipeline v3 (final). Previous versions: FINAL_PIPELINE.md (v1, 22 papers), PIPELINE_CLEAN.md (v2, 14 papers).*
