# Paper 13: Swin UNETR (2022) - YOUR EMBEDDING EXTRACTOR + SEGMENTATION UPGRADE!

## 1. ONE-SENTENCE SUMMARY
Swin UNETR is a **3D brain tumor segmentation model** that uses a Swin Transformer encoder (hierarchical, shifted-window attention) connected to a CNN decoder — and for your project it serves **DUAL PURPOSE**: (1) upgrade from nnU-Net for segmentation, AND (2) extract **768-dimensional embeddings** from its encoder to feed into the temporal ViT (Paper 11)!

---

## 2. KEY RESULTS (The Numbers That Matter!)

### **Five-Fold Cross-Validation on BraTS 2021 (1,251 subjects)**

| Method | ET Dice | WT Dice | TC Dice | **Average** |
|--------|---------|---------|---------|-------------|
| TransBTS (ViT-only bottleneck) | 0.868 | 0.911 | 0.898 | 0.891 |
| SegResNet (ResNet-based) | 0.883 | 0.927 | 0.913 | 0.907 |
| **nnU-Net** (your current baseline) | 0.883 | 0.927 | 0.913 | **0.908** |
| **Swin UNETR** | **0.891** | **0.933** | **0.917** | **0.913** ✅ BEST |

> 🎯 **Swin UNETR beats nnU-Net by 0.5% average Dice** across all tumor sub-regions and all 5 folds. It beats nnU-Net in EVERY fold and EVERY sub-region!

### **BraTS 2021 Validation Set (219 cases, official server)**

| Metric | ET | WT | TC |
|--------|----|----|-----|
| **Dice Score** | 0.858 | 0.926 | 0.885 |
| **Hausdorff Distance (mm)** | 6.016 | 5.831 | 3.770 |

> Ranked as **one of the top-performing methods** across 2,000+ submissions — the **FIRST transformer-based model** to place competitively in BraTS challenges!

### **BraTS 2021 Test Set (official evaluation)**

| Metric | ET | WT | TC |
|--------|----|----|-----|
| **Dice Score** | 0.853 | 0.927 | 0.876 |
| **Hausdorff Distance (mm)** | 16.326 | 4.739 | 15.309 |

### **Why Swin UNETR > Other ViT Methods**

| Method | Architecture Problem | Result |
|--------|---------------------|--------|
| TransBTS | ViT only in bottleneck, no multi-scale | 0.891 avg Dice (worst) |
| UNETR (original) | Standard ViT (no hierarchy) | Not in final comparison |
| **Swin UNETR** | Hierarchical Swin + multi-scale skip connections | **0.913 avg Dice** ✅ |

> **Key insight**: TransBTS uses ViT as a "standalone attention module" in the bottleneck with NO skip connections to the decoder → loses multi-scale information → performs WORST of all methods (even below nnU-Net). Swin UNETR fixes this with hierarchical encoding + skip connections at EVERY resolution!

---

## 3. WHAT'S NEW? (Innovation!)

### 🧠 Innovation 1: Swin Transformer as 3D Medical Image Encoder

**Problem**: Standard ViTs (like in UNETR) process the ENTIRE volume at once → quadratic memory cost → impractical for large 3D medical images (240×240×155 voxels!).

**Solution**: Swin Transformer uses **shifted windows** — only computes attention within small 3D windows, then shifts windows to connect them.

**How it works (simple)**:
```
Standard ViT attention:
  Every voxel attends to EVERY other voxel
  240×240×155 = 8.9 MILLION tokens → Attention matrix: 8.9M × 8.9M = IMPOSSIBLE!

Swin attention (window-based):
  Divide into windows of 7×7×7 = 343 tokens each
  Each window: attention matrix only 343 × 343 = TINY!
  
  But windows don't talk to each other...
  
  SHIFTED WINDOWS (next layer):
  Shift window boundaries by half → now tokens at window borders
  interact with neighbors from the previous layer!
  
  Result: LOCAL attention (efficient) + GLOBAL information (via shifting) ✅
```

**For Yale**: Can process 128×128×128 patches from brain MRIs efficiently — no memory problems even with 4 MRI channels!

---

### 🏗️ Innovation 2: Hierarchical Multi-Scale U-Shaped Architecture

**Problem**: Brain tumors have features at MULTIPLE scales — tiny enhancing regions (ET), medium tumor core (TC), large whole tumor with edema (WT). A single-scale model misses details.

**Solution**: 4-stage encoder that progressively downsamples, creating feature maps at 5 different resolutions — connected to decoder via skip connections (like U-Net!).

**The Architecture (Full Detail)**:
```
INPUT: 128 × 128 × 128 × 4 channels (T1, T1c, T2, FLAIR)
    ↓
┌─────────────── ENCODER (Swin Transformer) ───────────────┐
│                                                           │
│  Patch Partition: 2×2×2 patches                          │
│  → 64 × 64 × 64 × 48 (C=48 embedding dim)              │
│    ↓                                                      │
│  Stage 1: 2 Swin Transformer blocks                      │
│  → 64 × 64 × 64 × 48          ──────── Skip Connection ─── → Decoder Stage 1
│    ↓ (Patch Merging: 2× downsample)                      │
│                                                           │
│  Stage 2: 2 Swin Transformer blocks                      │
│  → 32 × 32 × 32 × 96          ──────── Skip Connection ─── → Decoder Stage 2
│    ↓ (Patch Merging: 2× downsample)                      │
│                                                           │
│  Stage 3: 2 Swin Transformer blocks                      │
│  → 16 × 16 × 16 × 192         ──────── Skip Connection ─── → Decoder Stage 3
│    ↓ (Patch Merging: 2× downsample)                      │
│                                                           │
│  Stage 4: 2 Swin Transformer blocks                      │
│  → 8 × 8 × 8 × 384            ──────── Skip Connection ─── → Decoder Stage 4
│    ↓ (Patch Merging: 2× downsample)                      │
│                                                           │
│  BOTTLENECK: 2 Swin Transformer blocks                   │
│  → 4 × 4 × 4 × 768            ── THIS IS YOUR EMBEDDING! │
│                                                           │
└───────────────────────────────────────────────────────────┘
    ↓
┌─────────────── DECODER (CNN) ─────────────────────────────┐
│                                                           │
│  Bottleneck → Deconv upsample → 8×8×8×384                │
│  + Skip from Stage 4 → Concat → Residual Block           │
│    ↓                                                      │
│  → Deconv upsample → 16×16×16×192                        │
│  + Skip from Stage 3 → Concat → Residual Block           │
│    ↓                                                      │
│  → Deconv upsample → 32×32×32×96                         │
│  + Skip from Stage 2 → Concat → Residual Block           │
│    ↓                                                      │
│  → Deconv upsample → 64×64×64×48                         │
│  + Skip from Stage 1 → Concat → Residual Block           │
│    ↓                                                      │
│  → 1×1×1 Conv → Sigmoid → 128×128×128×3                  │
│    (ET, WT, TC sub-region predictions)                    │
│                                                           │
└───────────────────────────────────────────────────────────┘

OUTPUT: 128 × 128 × 128 × 3 channels (ET mask, WT mask, TC mask)
```

**Decoder Residual Blocks**:
```
Each residual block:
  Input → 3×3×3 Conv → Instance Norm → LeakyReLU
        → 3×3×3 Conv → Instance Norm → + Input (skip)
        → LeakyReLU → Output
```

### 🎯 Innovation 3: Dual-Use Architecture (Segmentation + Embeddings!)

**This is what makes Swin UNETR uniquely valuable for YOUR project:**

```
USE 1 — SEGMENTATION (replaces/upgrades nnU-Net):
  Full model (encoder + decoder) → Segment tumors → Better than nnU-Net!
  Input: 4-channel MRI → Output: 3-channel mask (ET, WT, TC)

USE 2 — EMBEDDING EXTRACTION (feeds to temporal ViT):
  Encoder ONLY (no decoder) → Extract bottleneck features
  Input: 4-channel MRI → Output: 4×4×4×768 tensor → Global avg pool → 768-dim vector
  
  This 768-dim vector CAPTURES:
  - Tumor morphology (shape, size, location)
  - Texture patterns (heterogeneity, necrosis)
  - Multi-scale features (fine detail + global context)
  - Spatial relationships (tumor to brain structures)
  
  → MUCH richer than hand-crafted radiomics (110 features)!
  → MUCH richer than Paper 11's 64-dim embeddings!
```

---

## 4. ARCHITECTURE SPECIFICATIONS

### Model Configuration (Table 1 from paper):

| Parameter | Value |
|-----------|-------|
| **Total parameters** | **61.98 million** |
| **FLOPs** | **394.84 billion** |
| Patch size | 2 × 2 × 2 |
| Initial embedding (C) | 48 |
| Window size | 7 × 7 × 7 |
| Encoder depths | [2, 2, 2, 2, 2] (per stage + bottleneck) |
| Total transformer layers | 8 (+ 2 bottleneck = 10) |
| Attention heads per stage | [3, 6, 12, 24] |
| Feature dimensions | [48, 96, 192, 384, 768] |
| Decoder | Residual blocks with Instance Norm |
| Loss function | Soft Dice Loss |
| Input size | 128 × 128 × 128 × 4 |
| Output size | 128 × 128 × 128 × 3 |

### Attention Heads Scale with Resolution:

| Stage | Resolution | Channels | Heads | Dim per Head |
|-------|-----------|----------|-------|-------------|
| 1 | 64³ | 48 | 3 | 16 |
| 2 | 32³ | 96 | 6 | 16 |
| 3 | 16³ | 192 | 12 | 16 |
| 4 | 8³ | 384 | 24 | 16 |
| Bottleneck | 4³ | 768 | — | — |

> Each head always sees 16 dimensions — consistent throughout the network!

### Training Details:

| Setting | Value |
|---------|-------|
| GPUs | 8× NVIDIA V100 (DGX-1) |
| Batch size | 1 per GPU (8 total) |
| Learning rate | 0.0008 |
| Epochs | 800 |
| LR schedule | Linear warmup → Cosine annealing |
| Optimizer | (standard, not specified explicitly) |
| Input patches | Random crop 128×128×128 from full volume |
| Augmentation | Random flip (p=0.5 all 3 axes), intensity shift (±0.1), intensity scale (0.9-1.1) |
| Inference | Sliding window, 0.7 overlap |
| Normalization | Zero mean, unit std (non-zero voxels) |
| Ensembling | 10 models (2 runs × 5 folds) |

---

## 5. THE DATASET: BraTS 2021

### Dataset Details:
- **Training**: 1,251 subjects with ground truth labels
- **Validation**: 219 cases (submitted to server for evaluation)
- **Testing**: Unknown size (server-only evaluation)
- **Input**: 4 MRI modalities per subject:
  - T1 (native)
  - T1Gd (post-contrast T1-weighted) = T1c in Yale
  - T2
  - T2-FLAIR
- **Resolution**: 1 × 1 × 1 mm isotropic (already skull-stripped)
- **Size**: 240 × 240 × 155 voxels
- **Labels**: 3 nested sub-regions:
  - **ET** (Enhancing Tumor) — active tumor
  - **TC** (Tumor Core) — ET + necrosis
  - **WT** (Whole Tumor) — TC + peritumoral edema

> ⚡ **Yale match**: Yale has T1, T1c, T2, FLAIR — **exactly the same 4 modalities!** Direct transfer is possible!

### Multi-Site Data:
- BraTS collected from **multiple institutions using various MRI scanners**
- This means the pretrained Swin UNETR already learned some scanner invariance!
- Combined with ComBat on embeddings → strong harmonization!

---

## 6. SELF-SUPERVISED PRETRAINING (Referenced in Paper)

The paper references Tang et al. (2021) — "Self-supervised pre-training of Swin transformers for 3D medical image analysis" — which describes:

### Pretraining Strategy (from referenced paper):
- **5,050 publicly available CT volumes** used for self-supervised pretraining
- Tasks: Masked volume prediction, rotation prediction, contrastive learning
- **Pretrained weights publicly available!** → You don't need to pretrain from scratch!

### For Yale — Two-Stage Strategy:
```
STAGE 1: Use NVIDIA's pretrained weights (5,050 CT volumes)
    → Already learned general 3D medical image features
    ↓
STAGE 2: Fine-tune on Yale's 11,884 brain MRI scans
    → Specializes for brain metastases
    ↓
RESULT: Encoder produces brain-tumor-specific 768-dim embeddings
```

### Pretrained Weights Download:
```
URL: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
File: model_swinvit.pt
Use: Load encoder weights only → fine-tune on Yale
```

---

## 7. LIMITATIONS & WHAT'S MISSING

### ⚠️ Limitations:
1. **Designed for single-timepoint segmentation** — No temporal modeling!
   - For Yale: Use encoder as feature extractor → feed embeddings to temporal ViT (Paper 11)
   
2. **Trained on gliomas (BraTS)** — Not specifically brain metastases
   - For Yale: Fine-tune on Yale data → adapt to metastases (different tumor morphology)
   - But same modalities (T1, T1c, T2, FLAIR) → transfer should work well!

3. **Ensembling needed for best results** — 10 models for official submission
   - For Yale: Single model is fine for embedding extraction (0.913 vs 0.908 nnU-Net even without ensemble!)

4. **Large model** — 61.98M parameters, needs GPU training
   - For Yale: Fine-tuning (not training from scratch) → manageable on single GPU
   - Pretrained weights available → faster convergence

5. **CT pretraining → MRI fine-tuning** — Domain gap between pretrained (CT) and target (MRI)
   - Mitigated by: BraTS fine-tuning already bridges this gap; pretrained weights are from BraTS MRI training

---

## 8. FOR YOUR YALE PROJECT (Concrete Implementation!)

### 🏗️ DUAL-USE Implementation Plan:

```
═══════════════════════════════════════════════════
USE 1: UPGRADE SEGMENTATION (Phase 1 improvement)
═══════════════════════════════════════════════════

Phase 1 (original plan): nnU-Net segments Yale scans → 0.908 Dice
Phase 1 (UPGRADED):      Swin UNETR segments Yale scans → 0.913 Dice (+0.5%!)

BONUS: You get the encoder for free → no extra training!

═══════════════════════════════════════════════════
USE 2: EMBEDDING EXTRACTION (Phase 2 core)
═══════════════════════════════════════════════════

For EACH Yale scan:
1. Load fine-tuned Swin UNETR
2. Run scan through ENCODER ONLY
3. Take bottleneck output: 4 × 4 × 4 × 768 tensor
4. Global Average Pooling → 768-dim vector
5. This IS your scan embedding!

Per patient (8 scans): 8 × 768-dim = sequence of embeddings
    → Feed to TaViT (Paper 11) for temporal modeling!
```

### 📋 MONAI Implementation:

```python
# ===== INSTALLATION =====
# pip install monai

# ===== SEGMENTATION (Use 1) =====
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,        # T1, T1c, T2, FLAIR
    out_channels=3,       # ET, WT, TC
    feature_size=48,      # C = 48
    use_checkpoint=True,  # Gradient checkpointing (saves memory!)
)

# Load pretrained weights
weight = torch.load("model_swinvit.pt")
model.load_from(weights=weight)

# Fine-tune on Yale data...

# ===== EMBEDDING EXTRACTION (Use 2) =====
# After fine-tuning, extract encoder features:

def extract_embedding(model, scan):
    """Extract 768-dim embedding from a single scan."""
    # Get encoder output at bottleneck
    encoder_output = model.swinViT(scan)  # Returns features at all stages
    bottleneck = encoder_output[-1]        # Last stage: 4×4×4×768
    
    # Global average pooling
    embedding = bottleneck.mean(dim=(2, 3, 4))  # → 768-dim vector
    return embedding

# For one patient's temporal sequence:
patient_embeddings = []
for scan in patient_scans:  # T0, T1, T2, ..., T7
    emb = extract_embedding(model, scan)  # 768-dim
    patient_embeddings.append(emb)

# Stack: shape = (8, 768) — ready for temporal ViT!
temporal_input = torch.stack(patient_embeddings)
```

### 🔗 How Swin UNETR + Time-Distance ViT Work Together:

```
COMBINED PIPELINE (Papers 11 + 13):

Yale scan T0 ──→ Swin UNETR encoder ──→ 768-dim embedding ──┐
Yale scan T1 ──→ Swin UNETR encoder ──→ 768-dim embedding ──┤
Yale scan T2 ──→ Swin UNETR encoder ──→ 768-dim embedding ──┤
    ...                                                       ├──→ TaViT
Yale scan T7 ──→ Swin UNETR encoder ──→ 768-dim embedding ──┤    (Paper 11)
                                                              │      ↓
Time gaps:  T0→T1 = 182 days ────────────────────────────────┤    Temporal
            T1→T2 = 548 days ────────────────────────────────┤    Emphasis
            T2→T3 = 365 days ────────────────────────────────┘    Model
                                                                    ↓
                                                            [CLS] token
                                                                    ↓
                                                    Downstream prediction
                                                    + Clustering + LLM
```

### 📋 What to Download & Use:

| Resource | Link | Use |
|----------|------|-----|
| **MONAI library** | `pip install monai` | Swin UNETR model class |
| **Pretrained weights** | [model_swinvit.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) | Encoder initialization |
| **MONAI tutorials** | https://monai.io/research/swin-unetr | Full training examples |
| **BraTS 2021 data** | https://www.synapse.org/#!Synapse:syn25829067 | Optional: validate before Yale |
| **Paper code** | Built into MONAI — no separate repo needed! | Production-ready! |

---

## 9. CONNECTION TO OTHER PAPERS

| Paper | Connection |
|-------|-----------|
| **Paper 11 (Time-distance ViT)** | Swin UNETR encoder feeds embeddings INTO Paper 11's temporal model |
| **Paper 2 (nnU-Net)** | Swin UNETR REPLACES nnU-Net for segmentation (0.913 vs 0.908 Dice) |
| **Paper 1 (BraTS Toolkit)** | Preprocessing still needed BEFORE feeding to Swin UNETR |
| **Paper 9 (ComBat)** | Harmonizes Swin UNETR's 768-dim embeddings across scanners |
| **Paper 10 (Longitudinal ComBat)** | Preserves patient embedding trajectories during harmonization |
| **Paper 10+ (Nested ComBat)** | Handles Yale's multiple scanner batch effects in embedding space |
| **Paper 10++ (ComBat Validation)** | Test embeddings for scanner effects BEFORE harmonizing |
| **Paper 5 (FLIRE)** | Aligned scans → Swin UNETR → comparable embeddings across timepoints |
| **CAFNet** | Validates that Swin (ViT) + CNN decoder (hybrid) >> pure ViT |
| **TransXAI** | Swin UNETR attention maps → explainability for Phase 3 |

---

## 10. BOTTOM LINE

### ✅ Why This Paper is CRITICAL (⭐⭐⭐):
1. **DUAL PURPOSE**: Single model gives you better segmentation (Phase 1 upgrade) AND rich embeddings (Phase 2 core)!
2. **Beats nnU-Net**: 0.913 vs 0.908 Dice — your segmentation baseline improves for free
3. **768-dim embeddings**: 12× richer than Paper 11's 64-dim → captures more tumor biology
4. **Same input modalities as Yale**: T1, T1c, T2, FLAIR — direct transfer!
5. **MONAI integration**: `pip install monai` → `SwinUNETR()` → production-ready code
6. **Pretrained weights available**: Don't train from scratch → faster convergence
7. **First competitive transformer in BraTS**: Proves transformers CAN match/beat CNNs for brain tumor segmentation
8. **Hierarchical features**: Multi-scale skip connections → captures tiny ET details AND large WT regions

### 🎯 One-Line Takeaway:
> **Use Swin UNETR as your Swiss Army knife — it segments tumors better than nnU-Net AND extracts 768-dim embeddings that feed into the temporal ViT (Paper 11) for tracking tumor progression across Yale's 8-scan patient sequences.**

---

*Paper: "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"*  
*Authors: Ali Hatamizadeh et al., NVIDIA*  
*arXiv: https://arxiv.org/abs/2201.01266*  
*Code: MONAI library — https://monai.io/research/swin-unetr*  
*Pretrained weights: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt*
