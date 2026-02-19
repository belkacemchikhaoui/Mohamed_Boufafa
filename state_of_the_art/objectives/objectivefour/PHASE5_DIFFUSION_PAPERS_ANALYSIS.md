# Phase 5 — Video Generation & Counterfactual Diffusion: Paper Analysis

> **Objective 4**: Model and visualize cancer progression through AI-generated video sequences conditioned on multimodal inputs, including counterfactual treatment scenarios.
>
> **Phase 5 (Weeks 15–18)**: Selection of diffusion/transformer video models, conditioning on embeddings, temporal consistency, counterfactual generation.

---

## 📋 Paper Categories & Relevance Ranking

| Relevance | Paper | Category | Code? |
|---|---|---|---|
| ⭐⭐⭐ | Treatment-aware Diffusion for Glioma | Disease Progression | ❌ |
| ⭐⭐⭐ | MedEdit | Counterfactual | ❌ |
| ⭐⭐⭐ | EchoNet-Synthetic | Medical Video Diffusion | ✅ Code+Weights+Data |
| ⭐⭐ | Video LDM (Blattmann 2023) | Foundation Architecture | ❌ |
| ⭐⭐ | Counterfactual Diffusion Autoencoder | Counterfactual | ✅ Code |
| ⭐⭐ | CLIMATv2 | Multimodal Trajectory | ✅ Code |
| ⭐⭐ | Feature-Conditioned Cascaded Video Diffusion | Medical Video Diffusion | ✅ Code+Weights |
| ⭐ | MVG (Cao, CVPR 2024) | Medical Video Generation | ❌ |
| ⭐ | HeartBeat | Medical Video Diffusion | ❌ |
| ⭐ | Counterfactual MRI Data Augmentation | Counterfactual | ✅ Code |
| ⭐ | SurGen | Surgical Video | ❌ |
| ⭐ | Counterfactual Alzheimer's Disease Effect | Counterfactual | ❌ |
| ⚙️ | LDM (Rombach 2022) | Foundation (Backbone) | ✅ Code+Weights |
| ⚙️ | DDPM (Ho 2020) | Foundation (Theory) | ✅ Code |
| 🔍 | Interactive Laparoscopic Video | Surgical Video | ❌ |
| 🔍 | AI-Driven Cancer Symptom Trajectory | Symptom Prediction | ❌ |
| 🔍 | Temporal Learning Pediatric Brain Cancer | Temporal Prediction | ❌ (not found) |

**Legend**: ⭐⭐⭐ = Must-read, directly relevant | ⭐⭐ = Important architecture/method reference | ⭐ = Useful context | ⚙️ = Foundation building block | 🔍 = Informational only

---

## ⭐⭐⭐ TIER 1 — Must-Read Papers (Directly Match Our Pipeline)

---

### Paper 1: Treatment-aware Diffusion for Multi-parametric MRI Generation of Glioma Patients

- **Authors**: Liu et al.
- **Venue**: IEEE Transactions on Medical Imaging (TMI), 2025
- **arXiv**: [2309.05406](https://arxiv.org/abs/2309.05406)
- **Code**: ❌ Not publicly available
- **Dataset**: 23 patients local (18 train/5 test), 37 patients external (LUMIERE dataset, 132 exams)

#### What It Does
Generates **future tumor MRI scans** conditioned on **current scans + treatment information**. Uses a modified DDPM with treatment-aware conditioning to predict how gliomas will look after different treatment strategies (CRT: chemoradiation, TMZ: temozolomide). Produces both future **tumor segmentation masks** AND **realistic MRI volumes** simultaneously via joint learning.

#### Why It's Our #1 Paper ⭐⭐⭐
This is literally our project — same domain (brain tumors/gliomas), same imaging modalities (T1/T1c/T2/FLAIR), same task (longitudinal MRI generation), same counterfactual framing (treatment vs. no treatment). It validates that our Objective 4 is achievable.

#### Deep Technical Details (from full paper reading)

**Architecture — Modified DDPM UNet**:
- UNet channel dimensions: 64 → 128 → 256 → 512
- Input: Source MRI sequence [s₁, s₂, s₃] concatenated along channel dimension + noisy target
- Dual output heads: noise prediction ε̃ AND tumor segmentation masks m̃

**Treatment Conditioning Mechanism**:
- 4 treatment-day pairs per sample: 3 for source scans + 1 for target future scan
- Each pair ⟨τ, d⟩ encoded via:
  - Treatment type τ → embedding table → learned vector
  - Day number d → sinusoidal positional embedding → learned vector
  - Two separate MLPs process treatment and day embeddings independently
  - Final: treatment_MLP(τ) + day_MLP(d) = conditioning vector
- 4 conditioning vectors (one per time-point) injected into UNet via addition

**Joint Loss Function**:
```
L = ‖ω(ε − ε̃)‖² + λ · ℓ_seg

where:
  ℓ_seg = ℓ_dice(m̃_S, m_S) + √ᾱ_t · ℓ_dice(m̃_f, m_f)
  
  ω = m̂ · exp(−m̂ ∗ f_{k×k}) + 1    (tumor-focused weighting)
  
  ω values: [1.886, 5.451] for tumor region, 1.0 elsewhere
```
- √ᾱ_t scaling: segmentation loss weighted by noise level (more weight when less noisy → cleaner supervision)

**Training Details**:
- 2D slices, 192×192 resolution
- T=600 diffusion steps, linear β schedule
- 5M training iterations
- Batch size: 32, gradient accumulation: ×2
- Adam optimizer, lr=2.5×10⁻⁴, cosine decay
- **350 GPU-hours on V100 32GB**
- 5 stochastic samplings at inference for uncertainty

**Inference (Algorithm 2)**:
- T=600 reverse denoising steps
- Mask fusion over last T_m=10 steps (average segmentation predictions from final denoising steps)
- 5 stochastic samples → mean prediction + variance for uncertainty

**Results**:
| Metric | Local (5 patients) | External/LUMIERE (37 patients) |
|---|---|---|
| SSIM | 0.919 ± 0.03 | 0.848 (−7.1%) |
| PSNR | 27.97 ± 1.2 dB | ~24 dB |
| DSC (future tumor) | 0.719 ± 0.13 | ~0.55 |
| DSC (source tumor) | 0.849 ± 0.09 | — |

**Treatment Ablation** (critical finding):
| Condition | SSIM | DSC |
|---|---|---|
| With treatment conditioning | 0.919 | 0.719 |
| Without treatment conditioning | 0.882 (−3.7%) | 0.556 (−16.3%) |

**Limitations (acknowledged in paper)**:
- Only 23 patients — generalizability concerns
- Can't handle second surgery (non-linear treatment changes)
- Secondary glioblastomas poorly predicted
- 2D slices only, not full 3D volumes

#### What We Take
1. **Treatment conditioning mechanism** — encoding treatment type + time as sinusoidal+MLP conditioning
2. **Joint generation + segmentation** — predict tumor masks alongside images
3. **Tumor-focused loss weighting ω** — extra attention to tumor boundaries
4. **Proof it works** — validates that our Objective 4 is achievable
5. **What to improve**: more patients (we have 1,430 vs. their 23), 3D (vs. their 2D), actual video (vs. their single frame), multimodal conditioning (our Swin UNETR + RadFM vs. their treatment-only)

#### How We Adapt
- **They use 23 patients → We have 1,430** (Yale dataset) — massive scale advantage
- **They condition on treatment type only → We condition on multimodal embeddings** (Swin UNETR 768-dim + RadFM clinical narratives) — richer conditioning
- **They don't have temporal modeling → We have TaViT** — learned progression patterns inform generation
- **Architecture upgrade**: Their DDPM backbone → our LDM (latent space, much more efficient) + Video LDM temporal layers

---

### Paper 2: MedEdit — Counterfactual Diffusion-based Image Editing on Brain MRI

- **Authors**: Ben Alaya et al.
- **Venue**: MICCAI 2024, SASHIMI Workshop
- **arXiv**: [2407.15270](https://arxiv.org/abs/2407.15270)
- **Code**: ❌ Not publicly available
- **Dataset**: Atlas v2.0 stroke dataset — 655 T1-w brain MRI, 443 pathological, train=389, test=54

#### What It Does
Edits brain MRI scans to show **counterfactual disease effects** — "what would this brain look like if the patient had a stroke?" or "what if this stroke was more/less severe?". Uses diffusion-based inpainting with anatomical constraints to produce realistic medical counterfactuals.

#### Why It Matters ⭐⭐⭐
- **Outperforms Palette by 83%** and **SDEdit by 65.6%** on combined metric (1-Dice)×FID
- **Board-certified neuroradiologist** confirmed generated stroke scans were **indistinguishable from real ones** (3.20/5 clinical realism = same as real samples)
- Same modality as ours (brain MRI)
- Directly addresses the counterfactual component of Objective 4

#### Deep Technical Details (from full paper reading)

**Method — Conditional RePaint with Mask Dilation**:
- Extends RePaint algorithm with two key innovations:
  1. **Conditional denoiser**: concatenate brain mask b + pathology mask p as extra input channels → ε_θ(x_t, b, p, t)
  2. **Mask dilation**: instead of inpainting only pathology mask p, use dilated version m = dilate(p, kernel_size=25)
     - This allows the model to capture **indirect pathological effects** (brain atrophy, ventricle enlargement, mass effect)
     - 70% improvement over naïve RePaint for indirect effects

**Inpainting Process**:
```
For each reverse step t → t-1:
  1. Sample x_{t-1}^known from forward process (known region outside mask m)
  2. Denoise x_{t-1}^unknown using ε_θ (unknown region inside mask m)
  3. Combine: x_{t-1} = m ⊙ x_{t-1}^unknown + (1-m) ⊙ x_{t-1}^known
  4. RESAMPLE: diffuse x_{t-1} back to x_t and repeat (4 resampling steps)
     → ensures harmonization between generated and preserved regions
```

**UNet Architecture**:
- Standard DDPM UNet from Ho et al.
- T=1000 diffusion steps, linear β schedule (10⁻⁴ to 0.02)
- 128×128 resolution
- 1500 training epochs on 389 pathological scans
- Input channels: image (1) + brain mask (1) + pathology mask (1) = 3 channels

**Results**:
| Method | FID ↓ | (1-Dice)×FID ↓ | Realism ↑ | Indirect Effects ↑ |
|---|---|---|---|---|
| SDEdit | 24.1 | 7.95 | 2.80/5 | 3.00/5 |
| Palette | 9.08 | 5.63 | 2.40/5 | 2.00/5 |
| Naïve RePaint | 8.31 | 4.24 | 2.55/5 | 1.85/5 |
| **MedEdit** | **3.07** | **3.07** | **3.20/5** | **3.15/5** |
| Real samples | — | — | 3.20/5 | — |

Key: MedEdit realism score **equals real samples** (3.20 = 3.20)!

#### What We Take
1. **Mask dilation for indirect effects** — when we simulate tumor changes, surrounding tissue also changes (edema, brain shift, ventricle compression)
2. **RePaint-based editing** — keep anatomy intact, only regenerate the target region
3. **Clinical validation protocol** — neuroradiologist blind evaluation (we need this for Phase 6)
4. **Concatenation conditioning** — brain mask + pathology mask as extra input channels
5. **4 resampling steps** — harmonization between edited and preserved regions
6. **Our adaptation**: Instead of adding strokes, we simulate tumor growth/shrinkage under different treatments

#### How We Adapt
- **They edit single images → We generate temporal sequences** (progression videos, not single edits)
- **They do stroke → We do glioma** progression — different pathology but same brain MRI domain
- **They use manual disease specification → We use TaViT temporal predictions + RadFM narratives** as conditioning signals
- **Combine with TaDiff**: MedEdit's editing quality + TaDiff's temporal generation = our full pipeline

---

### Paper 3: EchoNet-Synthetic — Latent Video Diffusion Model for Echocardiography Generation

- **Authors**: Reynaud et al.
- **Venue**: MICCAI 2024
- **arXiv**: [2406.00808](https://arxiv.org/abs/2406.00808)
- **Code**: ✅ [github.com/HReynaud/EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic)
- **Weights**: ✅ Available
- **Dataset**: ✅ EchoNet-Dynamic (public echocardiography)

#### What It Does
Generates **temporally coherent medical video sequences** (echocardiograms) using a Latent Video Diffusion Model (LVDM). Produces realistic cardiac ultrasound videos with correct temporal dynamics (heart beating, valve movement). Can condition generation on clinical parameters (LVEF — left ventricular ejection fraction).

#### Why It Matters ⭐⭐⭐
- **Only medical video diffusion paper with full code + weights + dataset** — we can actually run and adapt this
- Demonstrates that LVDM works for medical video (not just natural images/video)
- Shows how to handle temporal coherence in medical context
- Clear architecture we can modify for brain MRI

#### Deep Technical Details (from full paper reading)

**Architecture — 3-Model Pipeline**:
```
Model 1: VAE (Variational AutoEncoder)
  Input: 3×112×112 frames → Encoder → 4×14×14 latent → Decoder → reconstructed frames
  Compression: 48× (8× per spatial dimension, ×3 channels → ×4 latent channels)
  Training: 5 days on 8×A100 GPUs, batch=256

Model 2: LIDM (Latent Image Diffusion Model)
  UNet backbone, UNCONDITIONAL
  Generates random heart anatomy images in latent space
  Training: 24 hours on 1×A100, batch=256

Model 3: LVDM (Latent Video Diffusion Model)
  Spatio-Temporal UNet (space-time separated attention + convolutions)
  Conditioned on: heart image (spatial concat) + LVEF value (clinical parameter)
  Uses v-prediction: v = α_t·ε − σ_t·z, update: z_{t-1} = α_t·z_t − σ_t·v_θ(z_t)
  Training: 2 days on 8×A100 GPUs, batch=128
```

**Video Stitching for Long Sequences**:
- Split long noisy video into overlapping chunks (overlap = l_m/2 = half training length)
- Denoise each chunk independently
- Discard overlapping regions, concatenate clean regions
- Generated **10-minute video (19,200 frames)** in just 14 minutes on single A100!
- Train on 64 frames, generate arbitrarily long sequences

**Privacy Filtering Protocol**:
- Train re-identification model with contrastive loss
- Pearson correlation distance metric
- Threshold at 95th percentile of training set distances
- Rejected 11-37% of synthetic samples as too similar to real patients
- Important: we may need similar filtering for Yale patient privacy

**Results**:
| Metric | Real Data | Synthetic Data | Gap |
|---|---|---|---|
| R² (LVEF regression) | 0.81 | 0.75 | Small |
| MAE | 3.98 | 4.55 | Acceptable |
| FID | — | 28.8 | Good |
| FVD₁₆ | — | 103.5 | Good |
| SSIM | 1.0 (ref) | 0.78 (VAE recon) | VAE quality |
| PSNR | — | 24.9 dB (VAE recon) | VAE quality |
| Sampling speed | — | 2.4s/64 frames | 116× faster than EchoDiffusion |

#### What We Take
1. **3-model pipeline architecture** — VAE → Image DM → Video DM (proven decomposition)
2. **Video stitching** — generate long sequences from short training (overlap+concatenate)
3. **v-prediction variant** — alternative to ε-prediction, may be better for video
4. **Privacy filtering protocol** — important for Yale patient data
5. **Complete codebase** — saves weeks of implementation
6. **Training compute reference** — 5 days VAE + 2 days LVDM on 8×A100s

#### How We Adapt
| EchoNet-Synthetic | Our Adaptation |
|---|---|
| 2D echocardiograms (3×112×112) | 3D brain MRI (4×128³ — 4 modalities) |
| LVEF conditioning (1 scalar) | Swin UNETR 768-dim + RadFM text + treatment vector |
| Heart beating motion (cyclic) | Tumor growth/shrinkage (non-cyclic, progressive) |
| Unconditional LIDM | Skip — we condition on patient history directly |
| 2D UNet backbone | 3D UNet backbone (or 2.5D slice-by-slice for efficiency) |
| v-prediction | Compare v-prediction vs. ε-prediction for our domain |

---

## ⭐⭐ TIER 2 — Important Architecture/Method References

---

### Paper 4: Video Latent Diffusion Models (Video LDM)

- **Authors**: Blattmann et al. (NVIDIA)
- **Venue**: CVPR 2023
- **arXiv**: [2304.08818](https://arxiv.org/abs/2304.08818)
- **Code**: ❌ (Official code not released; community implementations exist)
- **Params**: ~3.1B

#### What It Does
Extends image-based Latent Diffusion Models to video generation by inserting **temporal layers** into a pre-trained image LDM. Key insight: fine-tune only the temporal layers while keeping spatial layers frozen from the image model. This allows leveraging powerful pre-trained image generators for video.

#### Deep Technical Details (from full paper reading)

**Core Innovation — Temporal Layer Insertion**:
```
Pre-trained Image LDM UNet (FROZEN):
  Spatial Block 1 → Spatial Block 2 → ... → Output

Video LDM UNet (temporal layers ADDED, only these trained):
  Spatial Block 1 → TEMPORAL LAYER 1 →
  Spatial Block 2 → TEMPORAL LAYER 2 →
  ...
  Only ~20% of total parameters need training!
```

**Temporal Layer Types**:
1. **Temporal attention**: each frame attends to all other frames across time dimension
2. **3D convolution residual blocks**: capture local temporal patterns between adjacent frames

**Reshape Trick** (key implementation detail):
```python
# Spatial layers see: batch of independent images
z.shape = (Batch × Time, Channels, Height, Width)  # (b t) c h w

# Temporal layers see: video sequences
z = rearrange(z, '(b t) c h w -> b c t h w')  # temporal ops across t
z = rearrange(z, 'b c t h w -> (b t) c h w')  # back to spatial format
```

**Learnable Merge Parameter α** (Eq. 3):
- Output = αᵢ_φ · z + (1 − αᵢ_φ) · z'
- z = spatial output, z' = temporal output
- α initialized to 1 → at start, temporal layers are identity (recovers original image model)
- α learned during training → gradually incorporates temporal information

**Context-Guided Prediction Model**:
- Condition on 0, 1, or 2 context frames using masking
- Context frames = known frames from patient history
- Enables autoregressive long video: generate frames 1-16, use last few as context for frames 17-32, etc.

**Temporal Interpolation**:
- Generate keyframes first (e.g., frames 1, 5, 9, ...)
- Then interpolate intermediate frames conditioned on keyframes
- Enables high effective frame rates without training on long sequences

**Video Upsampler**:
- Pixel-space DM with temporal fine-tuning for super-resolution
- Low-res 64×128 → high-res 256×512 or beyond

**Training** (only temporal layers optimized):
```
argmin_φ E[‖y − f_{θ,φ}(z_τ; c, τ)‖²]
```
where θ = frozen spatial params, φ = trainable temporal params

**Results**:
| Dataset | FVD ↓ | FID ↓ | IS ↑ |
|---|---|---|---|
| RDS (driving) | 389 | 31.6 | — |
| UCF-101 (zero-shot) | — | — | 33.45 |
| Text-to-video | Up to 1280×2048 | — | — |

**DreamBooth Transfer**: Temporal layers trained on one image backbone TRANSFER to different fine-tuned backbones → personalized video generation without retraining temporal layers.

#### What We Take
1. **Frozen spatial + trainable temporal** strategy — efficient, only ~20% params
2. **Temporal attention** mechanism — ensures frame-to-frame consistency
3. **Reshape trick** `(b t) c h w ↔ b c t h w` — key implementation detail
4. **Learnable merge parameter α** — smooth transition from image to video model
5. **Context-guided prediction** — use patient's past scans as context frames
6. **Video stitching** — overlap+concatenate for long sequences
7. **Temporal layer transferability** — train once, reuse across different patient-specific models

#### How We Adapt
- Use this architecture strategy on top of a medical image LDM (fine-tuned on brain MRI)
- Insert temporal layers that attend to TaViT's learned progression patterns
- Condition temporal generation on treatment embeddings (treatment-aware temporal dynamics)
- Context frames = patient's T0, T1, T2 scans → predict T3, T4, T5

---

### Paper 5: Counterfactual Explanations with Diffusion Autoencoder

- **Authors**: Atad et al.
- **Venue**: JMLBI (Journal of Machine Learning for Biomedical Imaging), 2024
- **Code**: ✅ Available
- **Domain**: General medical imaging (vertebral fractures, brain tumors, retina, chest X-rays)

#### What It Does
Uses diffusion autoencoders to find **latent representations** of medical images, then manipulates these latents to generate counterfactual explanations: "what minimal change to this image would change the diagnosis?"

#### Deep Technical Details (from full paper reading)

**3-Step Method**:
1. **Train DAE unsupervised** → semantic latent z_sem (1×512 dimensions)
   - Encodes images into: z_sem (high-level semantics, 512-dim) + x_T (stochastic details)
   - Uses conditional DDIM for reconstruction
   - z_sem captures disease-relevant features WITHOUT labels
   - Training: 12M steps, batch=64, single A40 GPU

2. **Train linear classifier on z_sem** (supervised, uses labels)
   - SVM on z_sem → decision hyperplane separating healthy from diseased
   - Simple linear classifier suffices because DAE produces linearly separable features

3. **Generate counterfactuals** by reflecting latents across hyperplane:
   ```
   w_ce = w − 2 · dist(w, P)
   where dist = (n · w + b) / ‖n‖
   n = hyperplane normal, b = bias
   ```
   - Reflect patient's z_sem across the healthy/diseased boundary
   - Keep x_T unchanged → preserves patient anatomy/identity
   - Decode reflected z_sem + original x_T → counterfactual image

**Continuous Regression for Severity**:
- Distance from hyperplane calibrated to pathology grade scale
- Smooth interpolation between grades: G0 → G1 → G2 → G3
- Linear direction in latent space = pathology progression

**Results**:
| Dataset | Task | AUC (Linear) | AUC (SVM) |
|---|---|---|---|
| VerSe (VCF) | Fracture detection | 0.96 | 0.93 |
| SPIDER (IVD) | Disc degeneration | 0.90 | 0.93 |
| RetinaMNIST | Diabetic retinopathy | 0.73 | 0.73 |
| BraTS | Peritumoral edema | 0.57-0.63 | 0.57-0.63 |
| MIMIC-CXR | Cardiomegaly | 0.82 | 0.82 |

BraTS results lower — brain tumor features are complex, but still above chance. The approach works best when pathology has clear visual signatures.

#### What We Take
1. **Latent space manipulation for counterfactuals** — change specific latent dimensions to simulate progression
2. **Linear pathology direction** — our Swin UNETR 768-dim embeddings may have similar linear "tumor severity" direction
3. **Hyperplane reflection** — simple, elegant counterfactual generation technique
4. **Semantic vs. stochastic decomposition** — separate disease-relevant features from patient anatomy

#### How We Adapt
- Instead of DAE, we use Swin UNETR embeddings (768-dim, already trained for Obj 2)
- Manipulate these embeddings to generate different progression scenarios
- Less directly relevant than TaDiff/MedEdit but useful for understanding counterfactual latent space theory
- Could be complementary: use as a planning module to determine target latent code, then decode with our video diffusion

---

### Paper 6: CLIMATv2 — Multi-Agent Transformer for Disease Trajectory Forecasting

- **Authors**: (2024)
- **Code**: ✅ Available

#### What It Does
Uses a **multi-agent transformer** architecture to forecast disease trajectories from multimodal data. Multiple "agents" model different aspects of disease progression (imaging changes, clinical markers, treatment effects) and communicate through cross-attention.

#### What We Take
1. **Multi-agent design** — Separate agents for imaging vs. clinical vs. treatment trajectories
2. **Cross-modal trajectory forecasting** — How to combine vision and clinical data for prediction
3. **Temporal consistency across modalities** — Ensure imaging and clinical predictions align temporally

#### How We Adapt
- Use as a planning/prediction module BEFORE video generation: CLIMATv2 predicts trajectory → diffusion model generates the corresponding video
- The "agents" concept maps to our pipeline: Agent 1 = Swin UNETR (imaging), Agent 2 = RadFM (clinical reasoning), Agent 3 = diffusion model (generation)

---

### Paper 7: Feature-Conditioned Cascaded Video Diffusion (EchoDiffusion)

- **Authors**: (2023)
- **Venue**: Medical imaging
- **Code**: ✅ Code + Weights available

#### What It Does
Uses a **cascaded diffusion approach** for echocardiogram video generation: first generates low-resolution videos, then upsamples to high resolution. Features from clinical data condition each stage of the cascade.

#### What We Take
1. **Cascaded generation** — Low-res first → high-res refinement (practical for 3D MRI where full resolution is expensive)
2. **Feature conditioning at each stage** — Clinical features guide both coarse structure and fine details
3. **Quality-compute tradeoff** — Cascading allows generating at feasible resolutions first

#### How We Adapt
- Cascade: Stage 1 generates 64×64×64 MRI volumes → Stage 2 upsamples to 128×128×128 or 256×256×256
- Condition each stage: Stage 1 on TaViT embeddings (coarse progression), Stage 2 on RadFM-refined features (clinical detail)
- This makes 3D video generation computationally tractable

---

## ⭐ TIER 3 — Useful Context Papers

---

### Paper 8: MVG — Medical Video Generation (Cao et al., CVPR 2024)

- **Authors**: Cao et al.
- **Venue**: CVPR 2024
- **Code**: ❌

#### What It Does
Combines GPT-4 + Stable Diffusion + SEINE for medical video generation. Uses LLM reasoning to plan temporal progression, then generates video frames.

#### What We Take
- **LLM-guided generation planning** — Using LLM output (our RadFM narratives) to plan what each frame should show
- **Text-to-video conditioning** — How to translate clinical text into frame-by-frame generation targets
- Validates our approach of using RadFM narratives to condition video generation

---

### Paper 9: HeartBeat — Multimodal Controllable Echo Synthesis

- **Authors**: (2024)
- **Code**: ❌

#### What It Does
Generates echocardiograms with fine-grained control over cardiac parameters (ejection fraction, wall motion, etc.). Multiple conditioning modalities (text, numerical, segmentation masks).

#### What We Take
- **Multi-modal conditioning** — How to inject multiple types of conditioning (we have: embeddings, text narratives, treatment type, scan metadata)
- **Fine-grained control** — Controlling specific aspects of generation (tumor size, location, growth rate)

---

### Paper 10: Counterfactual MRI Data Augmentation (cDDGMs)

- **Authors**: (2024)
- **Code**: ✅ Available

#### What It Does
Uses conditional denoising diffusion generative models to create counterfactual MRI data showing what scans would look like under different acquisition parameters (scanner, protocol, field strength).

#### What We Take
- **Scanner-aware generation** — Understanding how acquisition parameters affect image appearance
- **Augmentation strategy** — Generate counterfactual data for training robustness
- Complements our ComBat harmonization: ComBat removes scanner effects, cDDGMs can generate scanner-specific versions

---

### Paper 11: SurGen — Text-Guided Surgical Video Generation

- **Authors**: Cho et al.
- **arXiv**: [2408.14028](https://arxiv.org/abs/2408.14028)
- **Code**: ❌
- **Params**: 2B (from CogVideoX)

#### What We Take (Limited)
- **Text-to-video in medical domain** — Demonstrates text-guided video generation works in medicine
- **CogVideoX backbone** — Alternative to Video LDM for our video generation base
- Less relevant: surgical domain ≠ brain MRI

---

### Paper 12: Counterfactual Alzheimer's Disease Effect

- **Authors**: (2024)
- **Code**: ❌

#### What We Take
- **Classifier-free guidance for disease maps** — Generate disease-specific changes without explicit classifier
- **Neurodegenerative counterfactuals** — Brain atrophy simulation (different from our tumor growth, but same organ)

---

## ⚙️ FOUNDATION MODELS (Building Blocks)

---

### LDM — Latent Diffusion Models (Rombach et al., CVPR 2022)

- **arXiv**: [2112.10752](https://arxiv.org/abs/2112.10752)
- **Code**: ✅ [github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) / [github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- **Role**: Foundation for ALL latent diffusion papers above

**Why it's essential**: Every medical diffusion paper above builds on LDM. Understanding LDM (VAE encoder → latent space → UNet denoiser → VAE decoder) is prerequisite for all of Objective 4.

**Deep Technical Details (from full paper reading)**:

**Two-Stage Architecture**:
1. **Perceptual Compression (Autoencoder)**:
   - Encoder E: x → z = E(x), Decoder D: z → x̃ = D(z)
   - Downsampling factor f = H/h (f=1,2,4,8,16,32 tested)
   - **f=4 to f=8 gives best quality-compute tradeoff** (Table 8, Fig. 6)
   - Two regularization variants:
     - KL-reg: slight KL penalty toward N(0,1) — more flexible, our choice
     - VQ-reg: vector quantization in decoder — discrete latents
   - Trained with: perceptual loss + adversarial loss + regularization

2. **Latent Diffusion Model**:
   - DDPM but on latent z instead of pixel x
   - **4-48× compute reduction** at same quality!
   - Loss: L_LDM = E[‖ε − ε_θ(z_t, t, τ_θ(y))‖²]

**Cross-Attention Conditioning** (the key mechanism for us):
```
Attention(Q, K, V) = softmax(QK^T / √d) · V

where:
  Q = W_Q · φ_i(z_t)        ← from UNet intermediate features
  K = W_K · τ_θ(y)          ← from conditioning encoder
  V = W_V · τ_θ(y)          ← from conditioning encoder
  
  τ_θ(y) can be:
  - CLIP/BERT text encoder for text prompts
  - Learned projection for class labels
  - Our Swin UNETR 768-dim embeddings (projected to token sequence)
  - Our RadFM narrative tokens
```

**Key Experimental Results**:
| Model | FID ↓ | Params | Compute |
|---|---|---|---|
| ADM (pixel-space) | 10.94 | 554M | 1000 V100-days |
| LDM-4 | 10.56 | 274M | ~250 V100-days |
| LDM-8 | 7.76 | 395M | ~200 V100-days |
| LDM-4 (cfg) | 3.60 | 274M | ~250 V100-days |

LDM-4/LDM-8: **Same or better quality at 4-5× less compute and half the parameters!**

**For our pipeline**: VAE (f=8) compresses 3D MRI volumes to latent space. UNet denoises in latent space. Cross-attention injects our Swin UNETR embeddings + RadFM text as conditioning. Foundation for everything.

---

### DDPM — Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)

- **arXiv**: [2006.11239](https://arxiv.org/abs/2006.11239)
- **Code**: ✅ [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- **Role**: Theoretical foundation for all diffusion models

**Why it's essential**: The math behind all diffusion models.

**Deep Technical Details (from full paper reading)**:

**Forward Process** (adding noise):
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$
- Closed-form: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε, where ε ~ N(0,I)
- One-step jump from x_0 to any x_t (no need to iterate through all steps)

**Reverse Process** (removing noise):
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**Three Parameterizations** (predict different things):
1. **x₀-prediction**: predict the clean image directly
2. **ε-prediction** (their choice, and ours): predict the noise → resembles Langevin dynamics
3. **v-prediction** (used by EchoNet-Synthetic): predict v = αε − σx

**Simplified Loss** (L_simple):
$$L_{simple} = E_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$
- Just MSE between true noise and predicted noise
- Unweighted version works better than theory-derived weighted version
- "Better sample quality" — simplicity wins

**Training Algorithm 1**: Sample x₀, sample t~Uniform{1,...,T}, sample ε~N(0,I), take gradient step on L_simple
**Sampling Algorithm 2**: Start from x_T~N(0,I), iterate x_{t-1} = (1/√α_t)(x_t − (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t,t)) + σ_t·z

**Key Numbers**:
- T=1000 steps, linear schedule β₁=10⁻⁴ to β_T=0.02
- UNet architecture with self-attention at 16×16 resolution
- CIFAR-10: FID=3.17, IS=9.46 (SOTA at time)
- LSUN 256×256: comparable to ProgressiveGAN

**For our pipeline**: The fundamental math. Every paper builds on these equations. We use ε-prediction, L_simple, T=600 (from TaDiff). The theory is complete — we just need to apply it in latent space (LDM) with temporal extension (Video LDM).

---

## 🔍 INFORMATIONAL ONLY (Low Relevance)

---

### Interactive Laparoscopic Video Generation
- Surgical domain, user-controlled — not relevant to brain MRI progression

### AI-Driven Cancer Symptom Trajectory
- LSTM/CNN for symptom prediction — not visual, not generative. Could inform clinical evaluation metrics only.

### Temporal Learning for Pediatric Brain Cancer
- Multi-timepoint recurrence prediction — prediction only, not generative. Paper not found on arXiv.

---

## 🏗️ How These Papers Build Our Objective 4 Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                 OUR VIDEO GENERATION PIPELINE            │
│                                                          │
│   INPUTS (from Objectives 1-3):                         │
│   ├── Swin UNETR embeddings (768-dim) ← Obj 2          │
│   ├── TaViT temporal patterns ← Obj 2                   │
│   ├── RadFM clinical narratives ← Obj 3                 │
│   └── Treatment metadata ← Yale dataset                 │
│                                                          │
│   ARCHITECTURE (what we build):                          │
│   ┌──────────────────────────────────────────────┐      │
│   │  3D VAE Encoder (from LDM)                    │      │
│   │  ├── Encode brain MRI → latent space          │      │
│   │  └── Train on Yale T1/T2/FLAIR volumes        │      │
│   └──────────────┬───────────────────────────────┘      │
│                  ▼                                       │
│   ┌──────────────────────────────────────────────┐      │
│   │  Temporal Video Diffusion UNet                 │      │
│   │  ├── Spatial layers (from LDM, frozen)         │      │
│   │  ├── Temporal layers (from Video LDM, train)   │      │
│   │  ├── Treatment conditioning (from Glioma paper)│      │
│   │  ├── Embedding conditioning (our Swin UNETR)   │      │
│   │  └── Text conditioning (RadFM narratives)      │      │
│   └──────────────┬───────────────────────────────┘      │
│                  ▼                                       │
│   ┌──────────────────────────────────────────────┐      │
│   │  Cascaded Refinement (from EchoDiffusion)      │      │
│   │  ├── Stage 1: 64³ low-res progression          │      │
│   │  └── Stage 2: 128³+ high-res refinement        │      │
│   └──────────────┬───────────────────────────────┘      │
│                  ▼                                       │
│   ┌──────────────────────────────────────────────┐      │
│   │  3D VAE Decoder                                │      │
│   │  └── Latent → realistic MRI volumes            │      │
│   └──────────────┬───────────────────────────────┘      │
│                  ▼                                       │
│   OUTPUTS:                                              │
│   ├── Natural progression video (T3, T4, T5 scans)      │
│   ├── Counterfactual: "what if surgery?" (from MedEdit) │
│   ├── Counterfactual: "what if radiation?"               │
│   └── Counterfactual: "what if no treatment?"            │
│                                                          │
│   EVALUATION:                                            │
│   ├── FID, SSIM, LPIPS (quantitative)                   │
│   ├── Temporal coherence metrics                         │
│   └── Neuroradiologist review (from MedEdit protocol)   │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Summary: What Each Paper Contributes to Our Pipeline

| Paper | What We Take | Component |
|---|---|---|
| **Treatment-aware Glioma** | Treatment conditioning, multi-param MRI gen, temporal conditioning | Core architecture |
| **MedEdit** | Counterfactual formulation, anatomical constraints, clinical validation | Counterfactual module |
| **EchoNet-Synthetic** | LVDM architecture, temporal attention, **full codebase** | Starting codebase |
| **Video LDM** | Temporal layer insertion, frozen spatial + trainable temporal | Architecture strategy |
| **Counterfactual Diff. AE** | Latent manipulation for counterfactuals, minimal edit principle | Counterfactual latents |
| **CLIMATv2** | Multi-agent trajectory forecasting, cross-modal alignment | Trajectory planning |
| **EchoDiffusion** | Cascaded low→high resolution generation | 3D feasibility |
| **MVG** | LLM-guided generation planning, text-to-video conditioning | RadFM→Video bridge |
| **LDM** | VAE + UNet + cross-attention foundation | Base architecture |
| **DDPM** | Diffusion theory, noise schedules, loss functions | Mathematical foundation |

---

## 🔑 Key Technical Decisions for Objective 4

Based on paper analysis:

1. **Start from EchoNet-Synthetic codebase** — Only paper with full code+weights. Adapt LVDM architecture from echo→brain MRI.

2. **Use Video LDM temporal layer strategy** — Insert temporal attention between frozen spatial layers. Efficient training (only temporal params trained).

3. **Treatment conditioning from Glioma paper** — Encode treatment type as conditioning vector, concatenate with Swin UNETR embeddings and RadFM tokens.

4. **Counterfactuals from MedEdit approach** — Use diffusion-based editing to generate "what-if" scenarios. Manipulate treatment-related latent dimensions while preserving anatomy.

5. **Cascaded generation for 3D feasibility** — 3D brain MRI is expensive. Use EchoDiffusion's cascade: low-res first, then refine. Makes 3D video generation tractable on available hardware.

6. **Clinical validation protocol from MedEdit** — Neuroradiologist blind evaluation of generated vs. real scans. Already proven to produce indistinguishable results.

---

## 📎 Code & Resources Quick Reference

| Resource | Link | Status |
|---|---|---|
| EchoNet-Synthetic (our starting code) | [github.com/HReynaud/EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic) | ✅ Code+Weights+Data |
| Latent Diffusion (CompVis) | [github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | ✅ Code+Weights |
| Stable Diffusion | [github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) | ✅ Code+Weights |
| DDPM (Ho et al.) | [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | ✅ Code |
| Counterfactual Diff. AE | Available (needs specific link) | ✅ Code |
| CLIMATv2 | Available (needs specific link) | ✅ Code |
| EchoDiffusion (Cascaded) | Available (needs specific link) | ✅ Code+Weights |
| Treatment-aware Glioma | [arXiv:2309.05406](https://arxiv.org/abs/2309.05406) | ❌ Paper only |
| MedEdit | [arXiv:2407.15270](https://arxiv.org/abs/2407.15270) | ❌ Paper only |
| SurGen | [arXiv:2408.14028](https://arxiv.org/abs/2408.14028) | ❌ Paper only |
| Video LDM | [arXiv:2304.08818](https://arxiv.org/abs/2304.08818) | ❌ Paper only |

---

*Analysis complete. 17 papers reviewed, 3 identified as must-reads, 4 as important references, 4 as useful context, 2 as foundation building blocks, 3 as informational only. 7 core papers deep-analyzed from full PDFs (DDPM, LDM, Video LDM, TaDiff, MedEdit, EchoNet-Synthetic, CF-DAE). See also: `objectivefour/DIFFUSION_MODELS_EXPLAINED.md` for an ELI10-style explanation of diffusion models and how they connect to our pipeline.*
