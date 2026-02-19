# Diffusion Models Explained Simply — And How We Use Them

> This document explains diffusion models from scratch (assuming zero knowledge), then connects everything to our project pipeline. Read this before reading any of the papers.

---

## 🎯 The Big Picture: What Are We Trying to Do?

We want to **generate future brain MRI scans** showing how a patient's tumor will look:
- 6 months from now? (natural progression)
- After surgery? (counterfactual)
- After radiation? (counterfactual)
- With no treatment? (counterfactual)

Diffusion models are the AI technique that makes this possible. They're currently the **best image generators in the world** — better than GANs (the previous king), and they power Stable Diffusion, DALL-E, Midjourney, etc.

---

## 🧱 Part 1: DDPM — The Foundation (Ho et al., 2020)

> 📄 Paper: "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
> Code: [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

### The Core Idea — Explained Like You're 10

Imagine you have a **clear photo** of a brain MRI. Now imagine slowly adding TV static (random noise) to it, one tiny bit at a time, until after 1000 steps, the photo is **pure noise** — you can't see the brain anymore.

**That's the forward process** — it's easy and doesn't need any AI.

Now here's the magic: what if a neural network could learn to **reverse** this? Starting from pure noise, remove the static one tiny bit at a time, until you get back a clear brain MRI?

**That's the reverse process** — and THAT is what diffusion models learn.

```
FORWARD (easy, no AI needed):
Clear Brain MRI → add noise → add noise → ... → add noise → Pure Random Noise
   x₀            x₁           x₂          ...     x₁₀₀₀

REVERSE (this is what the AI learns):
Pure Random Noise → remove noise → remove noise → ... → Clear Brain MRI
   x₁₀₀₀             x₉₉₉          x₉₉₈        ...      x₀
```

### The Math (Simplified)

**Forward process** (adding noise):
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

- $x_0$ = original clean image
- $x_t$ = noisy image at step $t$
- $\epsilon$ = random noise (from normal distribution)
- $\bar{\alpha}_t$ = how much of the original image is left (decreases from 1→0 as $t$ goes 1→1000)

At $t=0$: image is 100% original. At $t=1000$: image is 100% noise.

**What the neural network learns**: Given a noisy image $x_t$ and the timestep $t$, predict the noise $\epsilon$ that was added.

**Training loss** (beautifully simple):
$$L = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Translation: "How different is the noise the network predicted ($\epsilon_\theta$) from the actual noise ($\epsilon$)?" Just mean squared error!

**Generating new images** (sampling):
1. Start with pure random noise $x_T$
2. For each step $t = T, T{-}1, ..., 1$:
   - Network predicts noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
   - Remove a bit of noise: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}\right) + \sigma_t z$
3. Result: a brand new image $x_0$!

### Key Numbers from Paper
- $T = 1000$ diffusion steps
- Uses a **UNet** architecture (encoder-decoder with skip connections — you know this from nnU-Net!)
- Noise schedule: $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ (linear increase)
- **FID 3.17** on CIFAR-10 (state-of-the-art at the time)

### Why DDPM Alone Isn't Enough for Us
- Works on small images (32×32, 64×64)
- Operates in **pixel space** — huge memory for 3D MRI volumes
- No conditioning — generates random images, can't say "generate a brain with THIS tumor"
- Slow: needs all 1000 steps to generate one image

> **What we take from DDPM**: The fundamental math, training procedure, and UNet architecture. Everything else builds on this.

---

## 🚀 Part 2: LDM — Making It Practical (Rombach et al., 2022)

> 📄 Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
> Code: [github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
> This became **Stable Diffusion** — yes, the famous one.

### The Problem LDM Solves

DDPM works in **pixel space** — directly on the image. For a 256×256 image, the UNet processes 256×256×3 = 196,608 values. For our 3D brain MRI (256×256×256), that's **16.7 million values per scan**. Impossible.

### The Solution: Work in a Compressed Space

LDM adds a **two-stage** approach:

```
STAGE 1: Autoencoder (train once, reuse forever)
┌─────────────────────────────────────────────┐
│  Encoder (E)              Decoder (D)        │
│  Full image → compress → Latent → decompress → Full image  │
│  256×256×3       →        32×32×4       →    256×256×3      │
│  (196K values)           (4K values)         (196K values)  │
│                                                              │
│  Compression ratio: 48×!                                     │
└─────────────────────────────────────────────┘

STAGE 2: Diffusion in Latent Space (the actual generation)
┌─────────────────────────────────────────────┐
│  DDPM but on 32×32×4 instead of 256×256×3    │
│  48× less compute!                            │
│  Same quality!                                │
└─────────────────────────────────────────────┘
```

### How the Autoencoder Works

It's a **VAE** (Variational Autoencoder) with:
- **Encoder E**: compresses image $x$ into latent code $z = E(x)$
- **Decoder D**: reconstructs image $\tilde{x} = D(z) = D(E(x))$
- **Downsampling factor** $f$: image shrinks by this factor. LDM-4 means 256→64, LDM-8 means 256→32

The autoencoder is trained with perceptual loss + adversarial loss + KL regularization, so the latent space captures **meaningful image features**, not just pixel values.

### The Magic: Conditioning via Cross-Attention

This is how you tell the model WHAT to generate. LDM introduces **cross-attention layers** in the UNet:

```
UNet processes latent z_t (the noisy latent)
    ↓
At each layer, CROSS-ATTENTION with conditioning signal:
    Q = features from z_t (what the model is generating)
    K = features from conditioning y (what you WANT it to generate)
    V = features from conditioning y

    Attention(Q, K, V) = softmax(QK^T/√d) · V
```

The conditioning $y$ can be **anything** — a text prompt, a class label, an image embedding, a bounding box. You just need an encoder $\tau_\theta$ that converts your conditioning into tokens.

**For text**: $\tau_\theta$ = BERT/CLIP text encoder → tokens → cross-attention
**For us**: $\tau_\theta$ = Swin UNETR embeddings (768-dim) + RadFM narratives → tokens → cross-attention

### Training LDM with Conditioning

$$L_{LDM} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2\right]$$

Same as DDPM, but:
- $z_t$ instead of $x_t$ (latent space, not pixel space)
- $\tau_\theta(y)$ = conditioning signal injected via cross-attention

### Key Results
- **48× less compute** than pixel-space diffusion (same quality!)
- **f=4 or f=8** gives best tradeoff (LDM-4 and LDM-8)
- Training: single A100 GPU (vs. hundreds of GPU-days for pixel diffusion)
- Supports: text-to-image, inpainting, super-resolution, layout-to-image

### What We Take from LDM
1. **VAE encoder/decoder** — we'll train one for 3D brain MRI volumes
2. **Diffusion in latent space** — makes 3D MRI generation feasible
3. **Cross-attention conditioning** — how we inject our Swin UNETR embeddings + RadFM text
4. **Classifier-free guidance** — control the strength of conditioning

---

## 🎬 Part 3: Video LDM — Adding Time (Blattmann et al., 2023)

> 📄 Paper: "Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models" (CVPR 2023)
> By NVIDIA — same team that made LDM/Stable Diffusion

### The Problem

LDM generates one image at a time. We need **sequences** — a series of MRI scans that change consistently over time (tumor grows, shrinks, etc.). If you just generate each frame independently, they'll look like different patients!

### The Brilliant Solution: Temporal Layers

Instead of training a video model from scratch, Video LDM **inserts temporal layers** into an already-trained image LDM:

```
Pre-trained Image LDM UNet (FROZEN — don't change):
┌────────────────────────────────────────────┐
│  Spatial Block 1 → Spatial Block 2 → ...    │
│  (processes each frame independently)        │
└────────────────────────────────────────────┘

Video LDM UNet (add temporal layers, ONLY train these):
┌────────────────────────────────────────────┐
│  Spatial Block 1 → TEMPORAL LAYER 1 →       │
│  Spatial Block 2 → TEMPORAL LAYER 2 →       │
│  Spatial Block 3 → TEMPORAL LAYER 3 → ...   │
│                                              │
│  Spatial layers: FROZEN (from image model)   │
│  Temporal layers: TRAINABLE (new, for video) │
│                                              │
│  Only ~20% of parameters need training!      │
└────────────────────────────────────────────┘
```

### How Temporal Layers Work

The spatial layers process each video frame as if it's a separate image in a batch. The temporal layers then **look across frames** to ensure consistency:

```python
# Spatial layers see: batch of independent images
z.shape = (Batch × Time, Channels, Height, Width)
# → treats each frame separately

# Temporal layers see: video sequences
z = rearrange(z, '(b t) c h w -> b c t h w')
# → temporal attention across the t dimension
# → ensures frame-to-frame consistency
z = rearrange(z, 'b c t h w -> (b t) c h w')
# → back to batch format for next spatial layer
```

The temporal layers use:
1. **Temporal attention**: each frame attends to all other frames → learns what changes and what stays the same
2. **3D convolutions**: capture local temporal patterns (gradual changes between adjacent frames)
3. **Learnable merge parameter** $\alpha$: controls blend between spatial and temporal outputs. When $\alpha = 1$, temporal layers are skipped entirely (recovers original image model)

### Key Innovation: Temporal Fine-Tuning

Only the temporal layers are trained. The spatial layers remain frozen. This means:
- **Leverages all the knowledge** of the pre-trained image model
- **Way less training data/time** needed
- **Temporal layers transfer** across different image model checkpoints (e.g., DreamBooth)

### Video Stitching for Long Videos

For long videos, they generate overlapping chunks and stitch them:
- Generate frames 1-64, then frames 33-96 (overlap at 33-64), then 65-128...
- Overlapping regions ensure temporal continuity
- Can generate **5+ minute videos** this way

### Key Results
- Real driving videos: 512×1024 resolution
- Text-to-video: up to 1280×2048
- Beats previous methods on FVD (Fréchet Video Distance)
- **2.4 seconds** to generate 64 frames (vs. 279s for previous methods!)

### What We Take from Video LDM
1. **Frozen spatial + trainable temporal** strategy — efficient, practical
2. **Temporal attention** mechanism — ensures our generated MRI sequences are consistent
3. **Video stitching** — if we need to generate long progression sequences
4. The idea that you can **turn any image model into a video model**

---

## 🧠 Part 4: Treatment-aware Diffusion (TaDiff) — Our Closest Paper (Liu et al., IEEE-TMI 2025)

> 📄 Paper: "Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction"
> arXiv: [2309.05406](https://arxiv.org/abs/2309.05406)
> This paper does **almost exactly** what we want to do!

### What They Do

Given a patient's **past MRI scans** (e.g., 3 time-points) + **treatment information** (chemoradiation, temozolomide, etc.), the model:
1. **Generates future MRI scans** — what will the brain look like at a future date?
2. **Predicts tumor masks** — where will the tumor be?
3. **Provides uncertainty maps** — how confident is the model?

All conditioned on the **treatment type** and **target time-point**.

### Architecture: Modified DDPM with Treatment Conditioning

```
INPUTS:
├── Source MRI sequence: [s₁, s₂, s₃] (3 past scans)
├── Treatment-day pairs: [(CRT, day 36), (TMZ, day 64), (TMZ, day 127)]
└── Target treatment-day: (TMZ, day 225)  ← "what happens at day 225?"

CONDITIONING:
Treatment & Day → Embedding MLPs → 4 treatment vectors
(each treatment-day pair gets its own learned embedding)

DIFFUSION PROCESS:
Source MRIs ∪ noisy target → UNet → [predicted noise, predicted tumor masks]
                              ↑
                    Treatment embeddings injected via addition
```

### Key Design Choices

1. **Treatment conditioning**: Each (treatment, day) pair gets:
   - Treatment type → embedding table → learned vector
   - Day number → sinusoidal embedding → learned vector
   - Final: treatment_embedding + day_embedding = conditioning vector
   
2. **Joint learning**: The UNet predicts TWO things simultaneously:
   - $\tilde{\epsilon}$ = noise prediction (for image generation)
   - $\tilde{m}$ = tumor segmentation masks (for both source AND future scans)
   
3. **Weighted loss focusing on tumor region**:
   $$\omega = \hat{m} \cdot e^{-\hat{m} * f_{k \times k}} + 1$$
   Gives extra weight to the tumor boundary region (where changes happen most)

4. **Joint loss**:
   $$L = \|\omega(\epsilon - \tilde{\epsilon})\|^2 + \lambda \cdot \ell_{seg}$$
   Combines image generation loss (MSE) with segmentation loss (Dice)

### Results
| Metric | Value | What It Means |
|---|---|---|
| SSIM | 0.919 | Generated MRIs are structurally very similar to real ones |
| PSNR | 27.97 dB | Good image quality |
| DSC (future tumor) | 0.719 | Tumor prediction accuracy |
| DSC (source tumor) | 0.849 | Can segment tumors in input scans too |

**Treatment conditioning improves everything**:
- Without treatment info: SSIM 0.882, DSC 0.556
- With treatment info: SSIM 0.919 (+3.7%), DSC 0.719 (+16.3%)

### Limitations
- Only **23 patients** in local test, 37 in external test
- Only **2 treatment types** (CRT, TMZ) — second surgery breaks the model
- Works on **2D slices**, not full 3D volumes
- No video generation — predicts one future time-point at a time

### What We Take from TaDiff
1. **Treatment conditioning mechanism** — encoding treatment type + time as conditioning
2. **Joint generation + segmentation** — predict tumor masks alongside images
3. **Tumor-focused loss weighting** — extra attention to tumor boundaries
4. **Proof it works** — validates that our Objective 4 is achievable
5. **What to improve**: more patients (we have 1,430 vs. their 23), 3D (vs. their 2D), actual video (vs. their single frame)

---

## ✏️ Part 5: MedEdit — Counterfactual Brain Editing (Ben Alaya et al., MICCAI24)

> 📄 Paper: "MedEdit: Counterfactual Diffusion-based Image Editing on Brain MRI"
> arXiv: [2407.15270](https://arxiv.org/abs/2407.15270)
> MICCAI 2024, SASHIMI Workshop

### What They Do

Takes a **healthy brain MRI** and edits it to show **"what if this patient had a stroke?"** The generated stroke scans are so realistic that a **board-certified neuroradiologist couldn't tell them from real ones**.

### Key Innovation: Modeling INDIRECT Effects

Most image editing methods only change the target region (paint a lesion). But in reality, a stroke causes:
- **Direct effect**: the lesion itself
- **Indirect effects**: brain atrophy, ventricle enlargement on the affected side

MedEdit captures BOTH:

```
Input: healthy brain scan + pathology mask (where to put the stroke)

Step 1: Dilate the pathology mask
   m = dilate(pathology_mask, kernel_size=25)
   → allows editing a BIGGER region around the lesion

Step 2: RePaint-style inpainting
   - Known region (outside dilated mask): keep original
   - Unknown region (inside dilated mask): generate with diffusion
   - Resample 4 times for harmonization between regions

Step 3: Condition on brain mask + pathology mask
   UNet input = [noisy_image, brain_mask, pathology_mask]
   (concatenated as input channels)
```

### Results vs. Baselines

| Method | FID ↓ | Realism (clinical) ↑ | Indirect Effects ↑ |
|---|---|---|---|
| SDEdit | 24.1 | 2.80/5 | 3.00/5 |
| Palette | 9.08 | 2.40/5 | 2.00/5 |
| Naïve RePaint | 8.31 | 2.55/5 | 1.85/5 |
| **MedEdit** | **3.07** | **3.20/5** | **3.15/5** |
| Real samples | — | 3.20/5 | — |

MedEdit's realism score **equals real samples** (3.20 = 3.20)!

### What We Take from MedEdit
1. **Mask dilation for indirect effects** — when we simulate tumor changes, surrounding tissue also changes
2. **RePaint-based editing** — keep anatomy intact, only regenerate the target region
3. **Clinical validation protocol** — neuroradiologist evaluation (we need this for Phase 6)
4. **Concatenation conditioning** — brain mask + pathology mask as extra input channels
5. **Our adaptation**: Instead of adding strokes, we simulate tumor growth/shrinkage under different treatments

---

## 🔬 Part 6: Counterfactual Diffusion Autoencoder (Atad et al., 2024)

> 📄 Paper: "Counterfactual Explanations for Medical Image Classification and Regression using Diffusion Autoencoder"
> Journal: JMLBI 2024

### What They Do

Uses a **Diffusion Autoencoder (DAE)** to learn a rich latent space where you can:
1. **Classify/grade pathologies** (healthy vs. fractured, severity levels)
2. **Generate counterfactuals** by moving in latent space: "what would this image look like if the pathology was more/less severe?"
3. **Visualize smooth progressions** between healthy and severe

### How It Works

```
Step 1: Train DAE (unsupervised — no labels needed!)
   Image → Semantic Encoder → z_sem (512-dim latent code)
   z_sem + noise → Conditional DDIM → Reconstructed image

Step 2: Train classifier on z_sem (supervised — uses labels)
   z_sem → Linear SVM → healthy/pathological
   The SVM finds a HYPERPLANE separating healthy from diseased

Step 3: Generate counterfactuals (the cool part)
   Take a patient's z_sem
   Move it ACROSS the hyperplane → image changes from healthy to diseased
   Distance from hyperplane = severity grade
```

### Key Insight for Our Project

The latent space is **linear** — pathology severity is a straight line in latent space:

```
← Healthy ─────────|hyperplane|───────── Severe →
     G0        G1       |        G2        G3
```

You can smoothly interpolate between grades to visualize progression!

### Results
- VCF detection: AUC 0.96 (better than Autoencoders, close to supervised DenseNet)
- BraTS peritumoral edema: AUC 0.63 (even with brain tumors!)
- Works on: vertebral fractures, diabetic retinopathy, intervertebral disc degeneration, edema

### What We Take
1. **Latent space manipulation for counterfactuals** — move in embedding space to simulate progression
2. **Linear pathology direction** — our Swin UNETR 768-dim embeddings may have a similar linear "tumor severity" direction
3. **Unsupervised feature extraction** — DAE learns useful representations without labels
4. **Our adaptation**: Instead of DAE, we use Swin UNETR embeddings (768-dim, already trained). Manipulate these embeddings to generate different progression scenarios.

---

## 🎥 Part 7: EchoNet-Synthetic — Our Starting Code (Reynaud et al., MICCAI 2024)

> 📄 Paper: "EchoNet-Synthetic: Privacy-preserving Video Generation for Safe Medical Data Sharing"
> arXiv: [2406.00808](https://arxiv.org/abs/2406.00808)
> Code: [github.com/HReynaud/EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic) ✅

### What They Do

Generates **synthetic echocardiogram videos** (heart ultrasound) that are:
- Temporally consistent (heart beats realistically)
- Privacy-preserving (can't be traced back to real patients)
- Good enough to **train AI models on** (downstream task performance matches real data)

### Architecture: 3-Model Pipeline

```
Model 1: VAE (Variational AutoEncoder)
   Frames → Encoder → 4×14×14 latent → Decoder → Frames
   Compresses 3×112×112 images to 4×14×14 latents (48× compression!)
   Trained: 5 days on 8×A100 GPUs

Model 2: LIDM (Latent Image Diffusion Model)
   Generates random heart anatomy images in latent space
   UNet backbone, unconditional
   Trained: 24h on 1×A100

Model 3: LVDM (Latent Video Diffusion Model)
   Given a heart image + ejection fraction → generates a video of it beating
   Spatio-Temporal UNet (space-time separated attention + convolutions)
   Conditioned on: heart image (spatial) + LVEF value (clinical parameter)
   Trained: 2 days on 8×A100
```

### Video Stitching (Same as Video LDM)

Generates long videos by processing overlapping 64-frame chunks:
- Train on 64 frames, generate 128+ frames
- Overlap = 32 frames (half the training length)
- Generated a **10-minute video (19,200 frames)** in 14 minutes on A100!

### Key Results

| Metric | Real Data | Synthetic Data | Gap |
|---|---|---|---|
| R² (LVEF regression) | 0.81 | 0.75 | Small |
| MAE | 3.98 | 4.55 | Acceptable |
| FID | — | 28.8 | Good |
| Sampling speed | — | 2.4s/64 frames | 116× faster than EchoDiffusion |

### Why This Is Our Starting Code

✅ Full pipeline (VAE + image diffusion + video diffusion)
✅ Open source (code + weights + dataset)
✅ Proven to work for medical video
✅ Clinical conditioning (LVEF → we replace with treatment info)
✅ Video stitching for long sequences
✅ Privacy filtering protocol

### What We Need to Change

| EchoNet-Synthetic | Our Adaptation |
|---|---|
| 2D echocardiograms (112×112) | 3D brain MRI (128³ or 256³) |
| LVEF conditioning (1 number) | Multi-modal: 768-dim Swin UNETR + RadFM text + treatment vector |
| Heart beating motion | Tumor growth/shrinkage over months |
| Unconditional LIDM | Conditioned on patient history |
| UNet 2D backbone | UNet 3D backbone (or 2.5D slice-by-slice) |

---

## 🏗️ Part 8: How We Put It All Together — Our Pipeline

Now that you understand all the pieces, here's how they combine for our project:

### The Complete Video Generation Pipeline

```
WHAT WE ALREADY HAVE (from Objectives 1-3):
├── 1,430 patients × multiple time-points (Yale dataset)
├── Clean, registered MRI scans (HD-BET → nnU-Net → FLIRE)
├── 768-dim embeddings per scan (Swin UNETR)
├── Temporal patterns (TaViT attention)
├── Clinical narratives per patient (RadFM → MedLLaMA-13B)
└── Treatment metadata (surgery, radiation, chemo info)

WHAT WE BUILD IN OBJECTIVE 4:

Step 1: Train 3D VAE (from LDM paper)
┌──────────────────────────────────────┐
│  3D Brain MRI (128³) → Encoder →     │
│  Latent (16³×4) → Decoder →          │
│  Reconstructed 3D Brain MRI          │
│                                       │
│  Use: EchoNet-Synthetic VAE code      │
│  Adapt: 2D conv → 3D conv            │
│  Compression: 8× per dimension = 512×│
└──────────────────────────────────────┘

Step 2: Train Conditional Video Diffusion (from all papers combined)
┌──────────────────────────────────────┐
│  UNet with:                           │
│  ├── Spatial layers (from LDM)        │  ← process each frame
│  ├── Temporal layers (from Video LDM) │  ← ensure consistency
│  ├── Treatment conditioning (TaDiff)  │  ← treatment embeddings
│  ├── Vision conditioning (our Swin UNETR)  ← cross-attention
│  └── Text conditioning (our RadFM)    │  ← cross-attention
│                                       │
│  Input: past scans [T0,T1,T2] + noise │
│  Output: future scan T3               │
│                                       │
│  Train on Yale longitudinal sequences │
│  Loss: MSE(noise) + λ·Dice(tumor)     │ ← from TaDiff
└──────────────────────────────────────┘

Step 3: Generate Counterfactuals (from MedEdit + Counterfactual DAE)
┌──────────────────────────────────────┐
│  Same patient, DIFFERENT treatments:  │
│  ├── Treatment = surgery → generate   │
│  ├── Treatment = radiation → generate │
│  ├── Treatment = chemo → generate     │
│  └── Treatment = none → generate      │
│                                       │
│  Keep patient embeddings FIXED         │
│  Change ONLY the treatment vector      │
│  → See different progression paths     │
│                                       │
│  Use MedEdit's mask dilation for       │
│  indirect effects around tumor         │
└──────────────────────────────────────┘
```

### Week-by-Week Plan

| Week | What | Papers Used |
|---|---|---|
| **Week 15** | Train 3D VAE on Yale MRIs, test encode/decode quality | LDM, EchoNet-Synthetic |
| **Week 16** | Build Video Diffusion UNet: spatial + temporal layers + conditioning | Video LDM, TaDiff |
| **Week 17** | Train on Yale sequences, implement counterfactual generation | TaDiff, MedEdit, CF-DAE |
| **Week 18** | Evaluate: FID, SSIM, temporal coherence, clinical plausibility | MedEdit validation protocol |

### Key Numbers to Remember

| What | Value | From |
|---|---|---|
| Diffusion steps $T$ | 1000 (can use DDIM for fewer) | DDPM |
| VAE compression | 4-8× per dimension | LDM |
| Temporal layers | ~20% of total parameters | Video LDM |
| Treatment conditioning boost | +3.7% SSIM, +16.3% DSC | TaDiff |
| Target SSIM | >0.85 (TaDiff external achieves 0.848) | TaDiff |
| Counterfactual FID target | <10 (MedEdit achieves 3.07) | MedEdit |

---

## 📚 Paper Dependency Chain

Read them in this order:

```
1. DDPM (2020)           ← Learn the math
   ↓
2. LDM (2022)            ← Learn latent space + conditioning
   ↓
3. Video LDM (2023)      ← Learn temporal layers
   ↓
4. EchoNet-Synthetic     ← See a complete medical video pipeline
   ↓
5. TaDiff (2025)         ← See it applied to brain tumors + treatment
   ↓
6. MedEdit (2024)        ← Learn counterfactual editing
   ↓
7. CF Diffusion AE       ← Learn latent space counterfactuals
```

---

## 🔑 Glossary

| Term | Meaning |
|---|---|
| **Diffusion model** | AI that learns to generate images by learning to remove noise |
| **Forward process** | Adding noise to an image step by step until it's pure noise |
| **Reverse process** | Removing noise step by step to generate a new image |
| **UNet** | Neural network architecture (encoder-decoder with skip connections) |
| **Latent space** | Compressed representation of images (smaller, faster to work with) |
| **VAE** | AutoEncoder that compresses images to latent space and back |
| **Conditioning** | Telling the model WHAT to generate (text, embeddings, treatment type) |
| **Cross-attention** | Mechanism to inject conditioning into the UNet |
| **Classifier-free guidance** | Technique to control how strongly conditioning affects generation |
| **Temporal layers** | Extra layers that ensure frame-to-frame consistency in video |
| **Counterfactual** | "What-if" scenario — same patient, different treatment |
| **FID** | Fréchet Inception Distance — measures how realistic generated images are (lower = better) |
| **FVD** | Fréchet Video Distance — like FID but for videos |
| **SSIM** | Structural Similarity — measures if generated image has same structure as real (higher = better) |
| **DDIM** | Faster sampling variant of DDPM (can skip steps, e.g., 50 instead of 1000) |
| **Treatment embedding** | Learned vector representation of treatment type (surgery, radiation, etc.) |

---

*Now you know enough about diffusion models to understand every paper in the `objectivefour/` folder and how they fit our pipeline. The key insight: we're not inventing anything new — we're combining proven techniques (LDM + Video LDM + TaDiff + MedEdit) in a new way for longitudinal brain tumor video generation.*
