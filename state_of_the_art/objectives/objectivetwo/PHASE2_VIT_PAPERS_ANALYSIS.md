# Phase 2: ViT Papers Analysis (6 Available Papers)

## 🎯 Context: What Objective 2 Needs

**Goal**: ViT processes Yale scans → outputs 768-dim embeddings → evaluate with:
1. **Downstream prediction**: Feed embedding to simple classifier → predict tumor type/growth/response
2. **Clustering**: Plot all embeddings (t-SNE/UMAP) → do similar tumors group together?

**Pipeline reminder**:
```
Phase 1 output (clean + segmented + aligned scans)
    ↓
ViT extracts embeddings (768-dim per scan)
    ↓
ComBat harmonizes embeddings (remove scanner effects)
    ↓
Evaluate: downstream prediction + clustering
    ↓
Temporal modeling (track T0→T1→T2...T7 progression)
```

---

## 📊 The 6 Papers You Have (Ranked by Priority)

### ⭐⭐⭐ PRIORITY 1: Paper 11 — Time-distance ViT (2022)
**File**: `11.Time-distance vision transformers in lung cancer diagnosis from.pdf`

**What it does**: ViT that encodes TIME INTERVALS between scans into self-attention.

**Why #1 Priority**: This is the ONLY paper that handles **longitudinal temporal data** with ViT!
All other 5 papers work on single scans. This one works on sequences (T0→T1→T2) — exactly what Yale needs.

**Key details**:
- Two methods: (1) vector time embeddings, (2) temporal emphasis scaling
- AUC 0.785 (temporal ViT) vs 0.734 (single-scan) — **+5% from temporal info!**
- Handles irregular intervals (Yale: scans 3 months apart, then 12 months apart)
- Tested on NLST lung CT dataset (53K participants, 3 timepoints)
- **Code**: https://github.com/tom1193/time-distance-transformer ✅ PUBLIC

**For your project**:
- Adapt time embedding for Yale's ~8 scans per patient
- Replace lung nodule classification → brain tumor growth tracking
- This paper = your Objective 2 temporal blueprint

**Downstream tasks used in paper**: Benign vs malignant classification (AUC metric)
**What you'd do**: Tumor growing/stable/shrinking classification from temporal embeddings

**arXiv**: https://arxiv.org/abs/2209.01676

---

### ⭐⭐⭐ PRIORITY 2: Paper 13 — Swin UNETR (2022)
**File**: `13.Swin UNETR for Brain Tumor Segmentation (2022, arXiv).pdf`

**What it does**: Swin Transformer encoder + CNN decoder for 3D brain tumor segmentation.

**Why #2 Priority**: Directly applicable to Yale brain MRI! The encoder produces the **embeddings** you need. Even though it's designed for segmentation, the encoder's intermediate features ARE your ViT representations.

**Key details**:
- Hierarchical Swin Transformer (shifted windows for 3D volumes)
- 5 resolution levels connected via skip connections to CNN decoder
- BraTS 2021 top performer: Dice 0.9005 (whole), 0.87 (core), 0.85 (enhancing)
- Pretrained on 5,050 CT volumes (self-supervised)
- **Code**: MONAI library ✅ PUBLIC
  - https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
  - `pip install monai` → use `monai.networks.nets.SwinUNETR`
- **Pretrained weights**: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt

**For your project (DUAL USE!)**:
1. **Segmentation**: Replace/complement nnU-Net for tumor masks
2. **Feature extraction**: Use encoder output as 768-dim embeddings for Objective 2
   - Remove decoder → freeze encoder → extract embeddings from bottleneck layer
   - Feed embeddings to ComBat → temporal model

**Downstream tasks**: Segmentation metrics (Dice score)
**What you'd do**: Extract encoder embeddings → downstream classification + clustering

**arXiv**: https://arxiv.org/abs/2201.01266

---

### ⭐⭐ PRIORITY 3: TransXAI — Explainable Hybrid Transformer (2024)
**File**: `Explainable hybrid vision transformers and convolutional network for multimodal glioma segmentation in brain MRI.pdf`

**What it does**: CNN + Transformer hybrid for glioma segmentation WITH explainability (Grad-CAM heatmaps).

**Why #3 Priority**: Only paper in your set that addresses **explainability** — directly useful for Phase 3 (LLM explanations). Plus it's a brain tumor segmentation paper on BraTS data.

**Key details**:
- Combines CNN local features + Transformer global features
- Produces attention heatmaps showing WHERE model focuses
- Competitive BraTS segmentation accuracy
- Grad-CAM adapted for transformer attention weights
- **Code**: https://github.com/razeineldin/TransXAI (check availability)

**For your project**:
- Explainability methods transfer to your temporal ViT
- Heatmaps → feed to LLM: "Model focused on right frontal lobe enhancement"
- Bridges Phase 2 (ViT) and Phase 3 (LLM explanations)

**arXiv**: Search "TransXAI glioma segmentation Razeineldin"

---

### ⭐ PRIORITY 4: CAFNet — CNN+ViT Cross-Attention Fusion (2025)
**File**: `A hybrid CNN–ViT framework with cross-attention fusion and data augmentation for robust brain tumor classification.pdf`

**What it does**: Cross-attention mechanism to fuse CNN (local texture) + ViT (global shape) features for tumor classification.

**Why useful**: Shows that pure ViT alone (87.34%) is MUCH WORSE than ViT+CNN fusion (96.41%). Important lesson: don't throw away CNNs!

**Key details**:
- Standalone ViT: 87.34% → CAFNet (ViT+CNN): 96.41% accuracy (+9%)
- Cross-attention: CNN features attend to ViT features and vice versa
- Data augmentation strategies for small medical datasets
- Single scan, no temporal component

**For your project**:
- Architecture idea: Combine nnU-Net CNN features + Swin UNETR Transformer features
- Cross-attention fusion could improve your embedding quality
- **BUT**: Single scan only, no temporal — you'd need to add Paper 11's time encoding yourself

**Lesson learned**: Don't use pure ViT — always fuse with CNN!

---

### ⭐ PRIORITY 5: ResAttU-Net-Swin (2025)
**File**: `An attention based residual U-Net with swin transformer for brain.pdf`

**What it does**: Residual U-Net + Attention gates + Swin Transformer for brain tumor segmentation.

**Why lower priority**: Similar idea to Swin UNETR (Priority 2) but with LOWER performance.

**Key details**:
- Dice 89.2% (BraTS 2019) vs Swin UNETR's 90.05%
- Adds attention gates to skip connections (interesting idea)
- Residual connections everywhere (fights vanishing gradients)
- No public code available

**For your project**:
- Skip connection attention idea could improve Swin UNETR
- But Swin UNETR is already better AND has public code/weights
- **Read only if** Swin UNETR underperforms on Yale data

---

### ⭐ PRIORITY 6: BRAIN-META — CNN-ViT Ensemble (2025)
**File**: `BRAIN-META: A reproducible CNN–vision transformer.pdf`

**What it does**: Ensemble of 10 CNN+ViT models with XGBoost meta-learner for tumor TYPE classification.

**Why lowest priority**: Solves a **different task** (tumor type classification, not segmentation or temporal tracking). Yale doesn't even have tumor type labels.

**Key details**:
- 97.10% accuracy on glioma/meningioma/pituitary classification
- 10-model ensemble (ResNet, EfficientNet, ViT-B, Swin-T, etc.)
- Meta-learner (XGBoost) combines all model predictions
- Reproducible framework with public code

**For your project**:
- Ensemble strategy (combine multiple models) is a useful general idea
- Meta-learner concept could apply to your final system
- **BUT**: Classification task, not temporal, not segmentation
- **Read only if** you need an ensemble strategy later

---

## 🎯 The Verdict: Do These 6 Papers Cover Objective 2?

### What's COVERED ✅:
| Need | Paper | Status |
|------|-------|--------|
| Temporal ViT for longitudinal data | Paper 11 (Time-distance ViT) | ✅ Perfect match |
| 3D brain tumor segmentation + embeddings | Paper 13 (Swin UNETR) | ✅ Perfect match |
| Explainability for Phase 3 | TransXAI | ✅ Good for later |
| CNN+ViT fusion strategy | CAFNet | ✅ Architecture idea |
| Attention mechanisms for U-Net | ResAttU-Net-Swin | ✅ Backup option |
| Ensemble strategy | BRAIN-META | ✅ General idea |

### What's MISSING ❌:
| Gap | What you need | Where to find |
|-----|---------------|---------------|
| ViT embedding evaluation | How to do downstream prediction + clustering properly | Standard ML — no paper needed, use sklearn |
| ComBat on ViT embeddings | Proof that ComBat works on learned features | You already have Papers 9-10+ (Phase 1) |
| Tumor growth prediction | Predict T3 from T0,T1,T2 | SDC-Transformer (Paper 12) — NOT in your 6 available papers |

### Bottom Line:

**YES, these 6 papers are enough to start Objective 2!**

- Paper 11 + Paper 13 = your core architecture (temporal ViT + 3D segmentation encoder)
- You DON'T need Papers 12, 14-21 to begin implementation
- The SDC-Transformer (Paper 12, growth prediction) would be nice but isn't essential — you can build growth prediction using Paper 11's temporal framework

---

## 📅 Analysis & Reading Order

### This Week (3 days total):
1. **Day 1**: Paper 13 (Swin UNETR) — Read Sections 2-3 (architecture), test MONAI code
2. **Day 2**: Paper 11 (Time-distance ViT) — Read Sections 2.2-3.1 (time embedding), clone GitHub
3. **Day 3**: TransXAI — Quick skim for explainability methods (30 min), then CAFNet for fusion idea (30 min)

### Skip for now:
- ResAttU-Net-Swin → Swin UNETR is better
- BRAIN-META → Different task (classification)

### Implementation Order:
```
Week 5: Read Papers 11 + 13 → understand architecture
Week 6: Implement Swin UNETR on BraTS sample data (test segmentation + extract embeddings)
Week 7: Add Paper 11's time-distance encoding to Swin UNETR embeddings
Week 8: Run on Yale data → evaluate embeddings (downstream prediction + clustering)
Week 9: ComBat harmonize embeddings → re-evaluate
```

---

## 🔗 All Public Links (Only Available Papers)

| Paper | arXiv/DOI | GitHub Code | Pretrained Weights |
|-------|-----------|-------------|-------------------|
| Paper 11: Time-distance ViT | https://arxiv.org/abs/2209.01676 | https://github.com/tom1193/time-distance-transformer ✅ | — |
| Paper 13: Swin UNETR | https://arxiv.org/abs/2201.01266 | https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR ✅ | [model_swinvit.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) ✅ |
| TransXAI | Search: "TransXAI glioma Razeineldin 2024" | https://github.com/razeineldin/TransXAI 🔜 | — |
| CAFNet | Search: "CNN ViT cross-attention brain tumor 2025" | ❌ Not public | — |
| ResAttU-Net-Swin | Search: "attention residual U-Net swin brain 2025" | ❌ Not public | — |
| BRAIN-META | Search: "BRAIN-META CNN vision transformer 2025" | Check paper for link | — |

### Essential Libraries:
| Tool | Install | Use |
|------|---------|-----|
| MONAI | `pip install monai` | Swin UNETR, medical ViT, transforms |
| HuggingFace | `pip install transformers` | Baseline ViT models |
| scikit-learn | `pip install scikit-learn` | Downstream prediction + clustering evaluation |
| UMAP | `pip install umap-learn` | Embedding visualization |
