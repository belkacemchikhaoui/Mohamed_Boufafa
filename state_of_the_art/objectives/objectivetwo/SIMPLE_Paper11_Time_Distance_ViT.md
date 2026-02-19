# Paper 11: Time-Distance Vision Transformers (2022) - YOUR TEMPORAL BLUEPRINT!

## 1. ONE-SENTENCE SUMMARY
Time-distance vision transformers encode **HOW FAR APART scans are in time** directly into the ViT's attention mechanism, so the model knows that "6 months between T0→T1" matters differently than "2 years between T1→T2" — **exactly what Yale needs for irregular clinical follow-ups!**

---

## 2. KEY RESULTS (The Numbers That Matter!)

### **Synthetic Benchmark: Tumor-CIFAR**
A synthetic dataset where CIFAR images "grow" over time — perfect controlled test.

**Regular time intervals** (scans every 6 months):

| Method | AUC |
|--------|-----|
| CS-CNN (single scan, no temporal) | 0.8558 |
| DLSTM (LSTM on features) | 0.9907 |
| Positional ViT (position-only, no time) | 0.9314 |
| **TeViT (sinusoidal time encoding)** | **0.9948** |
| **TaViT (temporal emphasis model)** | **0.9960** ✅ BEST |

**Irregular time intervals** (scans at random gaps — like real clinics!):

| Method | AUC |
|--------|-----|
| CS-CNN (single scan) | 0.8370 |
| DLSTM | 0.9991 |
| **Positional ViT (no time info)** | **0.5052** ❌ COMPLETE FAILURE! |
| **TeViT** | **0.9996** ✅ |
| **TaViT** | **0.9988** ✅ |

> 🚨 **CRITICAL FINDING**: Without time-distance encoding, positional ViT drops to **RANDOM CHANCE (0.50)** on irregular data! This proves you CANNOT just use scan order — you MUST encode actual time gaps!

### **Real Clinical Data: NLST Lung Cancer Screening**
National Lung Screening Trial: 535 cancer cases + 1,397 controls, 2 consecutive CT scans per patient.

| Method | AUC | Description |
|--------|-----|-------------|
| CS-CNN (single scan) | 0.734 | Uses only the latest scan |
| DLSTM | 0.779 | LSTM on CNN features |
| Positional ViT | 0.768 | Scan order only, no time info |
| **TeViT (sinusoidal time)** | **0.785** | Adds time encoding to tokens |
| **TaViT (temporal emphasis)** | **0.786** ✅ BEST | Scales attention by time |

**Key statistics:**
- TaViT vs CS-CNN: **p < 0.05** (statistically significant improvement!)
- TaViT vs Positional ViT: **p > 0.05** (NOT significant) — Authors explain: NLST has >90% screening adherence → intervals are REGULAR → time encoding adds less value when intervals are already uniform

> 💡 **FOR YALE**: Yale patients have IRREGULAR intervals (months to years between scans, real clinical follow-ups). This is EXACTLY the scenario where time-distance encoding shines — the synthetic irregular experiments prove it goes from 0.50 → 0.99!

### **Pretraining Results (Masked Autoencoding)**

| Method | With Pretraining | Without Pretraining | Improvement |
|--------|-----------------|---------------------|-------------|
| TaViT | **0.786** | 0.770 | +1.6% |
| TeViT | **0.785** | 0.764 | +2.1% |
| Positional ViT | **0.768** | 0.751 | +1.7% |

> Pretraining consistently helps! Use masked autoencoding on ALL Yale scans (11,884!) before fine-tuning.

---

## 3. WHAT'S NEW? (Two Innovations!)

### 🕐 Innovation 1: TeViT — Time Encoding ViT (Sinusoidal Time Embedding)

**Problem**: Standard ViTs use positional encoding (scan 1, scan 2, scan 3...) but don't know the TIME GAP between scans.

**Solution**: Encode the **continuous time distance** (in days/months) using sinusoidal functions — same math that original Transformers use for word positions, but now applied to TIME!

**The Math (Simple)**:
```
Time distance: r_t = days between scan and reference point

TE(r_t)[2i]   = sin(r_t / 10000^(2i/D))    ← even dimensions
TE(r_t)[2i+1] = cos(r_t / 10000^(2i/D))    ← odd dimensions

Where: D = embedding dimension (64 in paper)
       i = dimension index (0, 1, 2, ... D/2)
```

**How it's used**: Time encoding vector is **ADDED** to each scan's feature token:
```
Input token = [feature_embedding] + [time_encoding]
                    ↓                      ↓
         What the scan shows     When the scan was taken
```

**Yale example**:
```
Patient with 8 scans:
T0 (Jan 2015): r_t = 0 days      → TE(0)   = [0, 1, 0, 1, ...]
T1 (Jul 2015): r_t = 182 days    → TE(182)  = [sin(182/10000), cos(182/10000), ...]
T2 (Jan 2017): r_t = 730 days    → TE(730)  = [sin(730/10000), cos(730/10000), ...]
T3 (Mar 2020): r_t = 1886 days   → TE(1886) = [sin(1886/10000), cos(1886/10000), ...]

The ViT now KNOWS that T2 is 4× farther from T0 than T1 is!
```

**Why sinusoidal?**:
- Continuous → handles ANY time gap (not limited to fixed intervals)
- Smooth → similar times get similar encodings
- Unique → different times get different encodings
- Proven → same trick that made GPT/BERT work for language!

---

### ⏱️ Innovation 2: TaViT — Time Aware ViT (Temporal Emphasis Model)

**Problem**: Not all scans matter equally. Recent scans should matter MORE for current diagnosis than old ones.

**Solution**: The **Temporal Emphasis Model (TEM)** — a learnable sigmoid function that scales self-attention weights based on time distance between scans.

**The Math (Simple)**:
```
Time distance between scan i and scan j:  R_i,j = |time_i - time_j|

Temporal emphasis score: f(R_i,j) = 1 / (1 + exp(a × R_i,j - c))

Where: a = learnable steepness parameter (how fast emphasis drops)
       c = learnable shift parameter (when emphasis starts dropping)
```

**What this looks like (flipped sigmoid)**:
```
Emphasis
  1.0 |████████████████▄▄▄▄
  0.8 |                    ▄▄▄▄
  0.6 |                        ▄▄▄
  0.4 |                           ▄▄▄
  0.2 |                              ▄▄▄▄▄
  0.0 |                                   ▄▄▄▄▄▄▄▄▄
      └──────────────────────────────────────────────→
        0 days              Time distance            5 years

  → Recent scans: HIGH emphasis (close to 1.0)
  → Old scans: LOW emphasis (close to 0.0)
  → Transition: LEARNABLE (model decides the cutoff!)
```

**How it's used**: TEM scores MULTIPLY the self-attention weights:
```
Standard attention:  Attention(Q,K) = softmax(Q × K^T / √d)
TaViT attention:     Attention(Q,K) = softmax(Q × K^T / √d) × TEM(R)
                                                                  ↑
                                              Recent pairs get full weight
                                              Old pairs get reduced weight
```

**Why it's brilliant for Yale**:
```
Patient scanned 8 times over 10 years:
- Latest scan (2023) vs previous (2022): TEM ≈ 0.95 (very relevant!)
- Latest scan (2023) vs 5 years ago (2018): TEM ≈ 0.5 (moderate)
- Latest scan (2023) vs 10 years ago (2013): TEM ≈ 0.1 (minimal weight)

→ Model automatically learns to focus on RECENT progression!
→ BUT old scans still contribute (not zeroed out) — captures long-term trends!
```

**TaViT vs TeViT — Which is better?**

| Feature | TeViT | TaViT |
|---------|-------|-------|
| How time enters | Added to input tokens | Multiplies attention weights |
| Time representation | Fixed sinusoidal formula | Learnable sigmoid |
| Handles irregular | ✅ Yes | ✅ Yes |
| Emphasizes recent | ❌ All times weighted equally | ✅ Recent > Old (learnable) |
| NLST AUC | 0.785 | **0.786** ✅ |
| Tumor-CIFAR (regular) | 0.9948 | **0.9960** ✅ |
| Tumor-CIFAR (irregular) | **0.9996** ✅ | 0.9988 |
| Parameters | Zero extra | +2 learnable (a, c) |

**Verdict**: Very close! TaViT slightly better on real data (clinical relevance > synthetic). **Use TaViT for Yale** — the learnable emphasis is perfect for patients with varying follow-up patterns.

---

## 4. ARCHITECTURE DETAILS (How to Build It!)

### Full Pipeline:
```
Raw 3D CT scans (per patient: T0, T1, T2, ... T7)
    ↓
Feature Extractor (pretrained CNN extracts ROI features)
    → 5 ROIs per scan, each with feature vector
    ↓
Linear Projection → 64-dim embedding space
    → 5 tokens per scan × 8 scans = 40 tokens per patient
    ↓
Add [CLS] token + Time Encoding (TeViT) or Time Emphasis (TaViT)
    ↓
Transformer Encoder (8 layers, 8 attention heads)
    ↓
[CLS] token output → Linear classifier → Cancer / No Cancer
```

### Hyperparameters:
| Parameter | Value |
|-----------|-------|
| Embedding dimension (D) | 64 |
| Attention heads | 8 |
| Encoder depth | 8 layers |
| Optimizer | AdamW |
| Learning rate schedule | Cosine warmup |
| Masking ratio (pretraining, 5 timepoints) | 0.75 |
| Masking ratio (pretraining, 2 timepoints) | 0.50 |
| ROIs per scan | 5 (from nodule detector) |

### Masked Autoencoder Pretraining:
```
Self-supervised pretraining (no labels needed!):
1. Take patient's scan sequence (T0, T1, ..., T7)
2. Randomly mask 75% of tokens
3. ViT tries to RECONSTRUCT the masked tokens
4. Model learns temporal patterns without any labels!
5. THEN fine-tune on actual task (cancer diagnosis)
```

**For Yale**: Pretrain on ALL 11,884 scans → Fine-tune on labeled subset. This is HUGE because Yale has no tumor labels — pretraining uses the data itself!

---

## 5. LIMITATIONS & WHAT'S MISSING

### ⚠️ Limitations Acknowledged by Authors:
1. **Only tested 2 timepoints on real data** — NLST has just 2 consecutive scans per patient
   - Yale has ~8 scans per patient → **Opportunity to extend to longer sequences!**
   
2. **Only tested on lung cancer screening** — Not validated on brain tumors
   - Yale = brain metastases → **Need to adapt feature extraction (brain ROIs, not lung nodules)**
   
3. **Feature extraction from pretrained CNN** — Uses a fixed nodule detector (not end-to-end)
   - For Yale: Replace with Swin UNETR encoder (Paper 13) → Learn brain-specific features!

4. **"Need to validate on irregularly repeating medical images from real clinical setting"** — Authors literally call for this!
   - Yale IS that real clinical setting → **Your project directly answers their call!**

### 🔄 What We Need to Adapt for Yale:
1. **Replace lung nodule detector** → Use Swin UNETR encoder (Paper 13) for brain embeddings
2. **Scale from 2→8 timepoints** → Adjust masking ratio (maybe 0.6 for 8 timepoints?)
3. **Brain-specific ROIs** → Use nnU-Net masks to guide attention to tumor regions
4. **768-dim embeddings** (from Swin UNETR) → Much richer than paper's 64-dim
5. **ComBat harmonization** → Apply to embeddings before temporal modeling

---

## 6. FOR YOUR YALE PROJECT (Concrete Implementation!)

### 🏗️ How This Fits Your Pipeline:

```
PHASE 1 OUTPUT: Clean, segmented, aligned Yale scans
    ↓
STEP 1: Swin UNETR encoder (Paper 13) processes each scan
    → Outputs: 768-dim embedding per scan per patient
    ↓
STEP 2: Test for scanner effects in embeddings (Paper 10++)
    ↓
STEP 3: ComBat harmonize embeddings if needed (Paper 10+/10)
    ↓
STEP 4: Apply TIME-DISTANCE encoding (THIS PAPER!)
    → Option A (TeViT): Add sinusoidal time encoding to each embedding
    → Option B (TaViT): Scale attention weights by time distance ← RECOMMENDED
    ↓
STEP 5: Temporal Transformer (8 layers, 8 heads)
    → Input: 8 harmonized time-encoded embeddings per patient
    → Output: [CLS] token = temporal progression representation
    ↓
STEP 6: Downstream tasks
    → Classification: Growing / Stable / Shrinking
    → Clustering: t-SNE/UMAP of [CLS] tokens
    → Prediction: Next-timepoint volume estimation
```

### 📋 What to Download & Use:

| Resource | Link | Use |
|----------|------|-----|
| **Code** | https://github.com/tom1193/time-distance-transformer | Full implementation! |
| **Architecture** | `model.py` in repo | TeViT and TaViT classes |
| **Pretraining** | `pretrain.py` in repo | Masked autoencoder |
| **Time encoding** | `time_encoding.py` in repo | Sinusoidal formula |

### 🔑 Key Code Components to Reuse:
1. **Temporal Emphasis Model** (TEM) → `class TemporalEmphasisModel(nn.Module)` — learnable sigmoid for attention scaling
2. **Sinusoidal time encoding** → Same as positional encoding but with real time values
3. **Masked autoencoder pretraining** → Self-supervised learning on Yale's 11,884 unlabeled scans
4. **Multi-timepoint ViT** → Handles variable-length temporal sequences

### ⚡ Quick Start Plan:
```python
# Step 1: Clone the repo
# git clone https://github.com/tom1193/time-distance-transformer

# Step 2: Replace feature extraction
# OLD: Pretrained lung nodule CNN → 5 ROI features per scan
# NEW: Swin UNETR encoder → 768-dim embedding per scan

# Step 3: Adapt for brain MRI
# - Change input channels: 1 (CT) → 4 (T1, T1c, T2, FLAIR)
# - Change ROIs: lung nodules → brain tumor regions (from nnU-Net masks)
# - Scale embedding: 64-dim → 768-dim (to match Swin UNETR output)

# Step 4: Pretrain on Yale
# - Masked autoencoding on all 11,884 scans
# - 75% masking ratio for patients with 5+ scans
# - 50% masking ratio for patients with 2-4 scans

# Step 5: Fine-tune on downstream tasks
# - Tumor growth classification
# - Treatment response prediction
```

---

## 7. CONNECTION TO OTHER PAPERS

| Paper | Connection |
|-------|-----------|
| **Paper 13 (Swin UNETR)** | Provides the embedding extractor — replaces paper's lung CNN |
| **Paper 9 (ComBat)** | Harmonizes embeddings before temporal encoding |
| **Paper 10 (Longitudinal ComBat)** | Preserves patient trajectories during harmonization |
| **Paper 10+ (Nested ComBat)** | Handles Yale's multiple scanner effects |
| **Paper 10++ (ComBat Validation)** | Test BEFORE harmonizing — don't blindly apply! |
| **Paper 2 (nnU-Net)** | Provides tumor masks → guides attention to tumor ROIs |
| **Paper 5 (FLIRE)** | Temporal alignment ensures scans are spatially comparable |
| **TransXAI** | Attention maps from temporal ViT → feed to LLM for explanations (Phase 3) |

---

## 8. BOTTOM LINE

### ✅ Why This Paper is CRITICAL (⭐⭐⭐):
1. **Solves THE core problem**: How to handle irregular time intervals between scans
2. **Proven catastrophic failure without it**: Positional ViT → 0.50 AUC on irregular data!
3. **Two methods, both work**: TeViT (simple, no extra params) + TaViT (learnable, slightly better)
4. **Code is public**: https://github.com/tom1193/time-distance-transformer
5. **Authors literally ask for your project**: "Need to validate on real clinical data with irregular intervals" → That's Yale!
6. **Pretraining strategy**: Masked autoencoding works on unlabeled data → Perfect for Yale's 11,884 scans!

### 🎯 One-Line Takeaway:
> **Use TaViT's temporal emphasis model on Swin UNETR embeddings to track brain tumor progression across Yale's irregular 8-scan sequences — the paper proves this is essential (without time info → random chance!).**

---

*Paper: "Time-distance vision transformers in lung cancer diagnosis from longitudinal computed tomography" (2022)*  
*Authors: Thomas Z. Li et al., Vanderbilt University*  
*arXiv: https://arxiv.org/abs/2209.01676*  
*Code: https://github.com/tom1193/time-distance-transformer*
