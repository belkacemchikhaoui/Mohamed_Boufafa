# SIMPLE Analysis: RadFM — Towards Generalist Foundation Model for Radiology

> **Paper**: "Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data"  
> **Authors**: Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Hui Hui, Yanfeng Wang, Weidi Xie  
> **Institution**: Shanghai Jiao Tong University + Shanghai AI Laboratory  
> **Published**: Nature Communications 16:7866 (2025) — **TOP-TIER JOURNAL!**  
> **DOI**: 10.1038/s41467-025-62385-7  
> **License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (paper), MIT (code)

---

## 🎯 One-Sentence Summary

RadFM is a **14-billion parameter multimodal foundation model** that takes 2D or 3D medical scans (X-ray, CT, MRI) + text prompts as input and generates clinical text (diagnoses, VQA answers, reports, rationales) — the ONLY open model that natively processes **full 3D MRI/CT volumes**, not slice-by-slice.

---

## 🧠 What Problem Does It Solve?

**Before RadFM**: Medical AI = separate models for every task:
- One model for chest X-ray diagnosis
- One model for CT report generation
- One model for MRI VQA
- None handles 3D volumes natively
- None can do interleaved image + text reasoning

**RadFM**: ONE model that handles ALL radiology tasks, ALL modalities (2D + 3D), and generates free-form text responses.

---

## 🏗️ Architecture (Complete Breakdown)

### The Three Components

```
RadFM = 3D ViT Encoder (Φ_vis) + Perceiver Module (Φ_per) + LLM Decoder (Φ_llm)
```

### Component 1: 3D Vision Transformer (Visual Encoder)

| Property | Value |
|---|---|
| Type | 3D ViT (standard Vision Transformer adapted for 3D) |
| Layers | 12 transformer layers |
| Feature dimension | 768 |
| Patch size | 32 × 32 × 4 (spatial × depth) |
| Attention heads | 8 |
| MLP dimension | 2048 |
| Dropout | 0.1 |
| Position embedding | Learnable 3D: PositionEmbeddingLearned3d |
| Input (2D images) | 512 × 512 × 4 (pad depth to 4 by repeating) |
| Input (3D volumes) | 256 × 256 × D (D = nearest multiple of 4, max 64) |
| Output | Variable-length tokens: (H/32 × W/32 × D/4) × 768 dims |

**Code location**: `src/Model/RadFM/vit_3d.py` → class `ViT`

**How it works**:
1. Input volume (B, C, H, W, D) gets split into 3D patches of size 32×32×4
2. Each patch is linearly projected to 768 dims
3. Learnable 3D position embeddings added (separate row/column/depth embeddings, concatenated)
4. 12-layer standard transformer (prenorm, GELU, residual connections)
5. Output: (num_patches × 768) — varies by input size

**2D vs 3D handling**:
- 2D images: Padded to pseudo-3D by repeating depth = 4 → treated like thin 3D volume
- Rationale: Consecutive 3D slices are similar → padding doesn't hurt. Allows SAME model for 2D and 3D.
- 3D volumes: Depth rounded to nearest multiple of 4, max 64 slices

### Component 2: Perceiver Resampler (Compression Module)

| Property | Value |
|---|---|
| Type | 6-layer transformer decoder with cross-attention |
| Learnable latent array | 32 queries × 768 dims |
| Input | Variable-length ViT tokens (P_i × 768) |
| Output | Fixed 32 tokens × 768 dims (ALWAYS 32, regardless of input size!) |
| Attention heads | 8 |
| dim_head | 64 |
| Feed-forward multiplier | 4× |
| Frame embeddings | Optional (for multi-frame inputs) |
| Media time embeddings | Optional (for multi-image inputs) |

**Code location**: `src/Model/RadFM/helpers.py` → class `PerceiverResampler`

**How it works**:
1. Takes variable-length image tokens from ViT (could be 64 tokens for 2D or 256 tokens for 3D)
2. Uses 32 learnable "query" tokens (latents)
3. Cross-attention: queries attend to all image tokens
4. After 6 layers: outputs exactly 32 tokens × 768 dims per image
5. Then projected: `nn.Linear(768, 5120)` to match LLM embedding dimension

**WHY THIS MATTERS FOR US**: 
- Our Swin UNETR outputs 768-dim embeddings — SAME dimension as RadFM's ViT!
- We can replace RadFM's 3D ViT with our Swin UNETR and feed directly to Perceiver
- The Perceiver compresses any-size input to fixed 32 tokens → LLM always sees consistent input

### Component 3: LLM Decoder (Language Model)

| Property | Value |
|---|---|
| Base model | MedLLaMA-13B (PMC-LLaMA fine-tuned from LLaMA-13B) |
| Parameters | ~13B (LLM) + ~1B (ViT + Perceiver) = ~14B total |
| Vocabulary | 32,000 tokens + 2 special image tokens + 32×S image tokens |
| Max sequence length | 2048 tokens |
| Hidden dimension | 5120 |

**Code location**: `src/Model/RadFM/multimodality_model.py` → class `MultiLLaMAForCausalLM`

**How it works**:
1. Text is tokenized normally (LlamaTokenizer)
2. `<image>` placeholder tokens in text are replaced with 32 visual tokens from Perceiver
3. Special `<image>` and `</image>` tokens wrap visual tokens
4. Combined visual+text token sequence fed to LLaMA self-attention
5. Standard autoregressive next-token prediction

### Complete Forward Pass

```
Input: 3D MRI volume (B, 3, 256, 256, D) + text prompt "Describe the findings"
    ↓
Step 1: 3D ViT encoder
    Volume → patches (32×32×4) → 12 transformer layers
    Output: (B, num_patches, 768) — variable length
    ↓
Step 2: Perceiver resampler  
    32 learnable queries × cross-attention with ViT tokens
    Output: (B, 32, 768) — FIXED length, always 32 tokens
    ↓
Step 3: Linear projection
    nn.Linear(768, 5120) — maps to LLM embedding space
    Output: (B, 32, 5120) — ready for LLM
    ↓
Step 4: Token fusion
    [text tokens] + [32 visual tokens] + [text tokens] → interleaved sequence
    ↓
Step 5: MedLLaMA-13B decoder
    Autoregressive text generation
    Output: "The MRI shows a 15mm enhancing lesion in..."
```

---

## 📊 Datasets (Massive Scale)

### MedMD — Training Dataset
| Component | Size | Format | Source |
|---|---|---|---|
| PMC-Inline | ~11M 2D images | Interleaved image-text from papers | PubMed Central |
| PMC-CaseReport | 103K case reports → 1.1M QA pairs | Visual instruction tuning | PubMed Central + ChatGPT |
| RP3D (RadioPaedia 3D) | 3D scans + captions | Caption, VQA, rationale, modality | Radiopaedia.org |
| MPx (MedPix) | Multi-scan cases | Case-level + scan-level annotations | MedPix (NIH) |
| PMC-OA | 1.6M image-caption pairs | Figure-caption | PubMed Central |
| MIMIC-CXR | 224K chest X-rays | Reports | PhysioNet |
| VQA datasets | VQA-RAD, SLAKE, PMC-VQA | Question-answer | Various |
| **Total** | **~16M images (13M 2D + 615K 3D)** | | |

### RadMD — Instruction Tuning Dataset
- **~3M radiologic images** filtered from MedMD
- Removed non-radiologic, removed PMC-Inline/PMC-OA (academic style, not clinical)
- Emphasizes 3D data and clinical-style QA
- Covers **5,000+ diseases**

### RadBench — Evaluation Benchmark
| Task | Samples | Description |
|---|---|---|
| Medical VQA | 4,229 | Comprehensive VQ with 3D scan input |
| Report Generation | 1,468 | Non-X-ray imaging modalities |
| Rationale Diagnosis | 1,000 | Predict disease + explain radiological features |
| **Total** | **6,697** | Manually verified by 8 annotators |

---

## 🏋️ Training Details

### Two-Stage Training

**Stage 1: Pretraining (4 epochs on MedMD)**
- Epoch 1: LLM is **FROZEN** — only align visual embeddings with text space
- Epochs 2–4: All parameters updated
- Data: Full MedMD (~16M images)

**Stage 2: Domain-Specific Instruction Tuning (4 epochs on RadMD)**
- All parameters updated
- Data: RadMD (~3M radiologic images)
- Focus: Clinical-style QA, not academic text

### Loss Function
- **Weighted negative log-likelihood** (autoregressive next-token):
  - Medical terms (UMLS vocabulary): weight = **3×** (prioritize medical accuracy)
  - Non-medical response text: weight = **1×**
  - Instruction/prompt tokens: weight = **0** (don't learn to repeat questions)
  - Image placeholder tokens: weight = **0**

### Compute Requirements
| Resource | Value |
|---|---|
| GPUs | 32× NVIDIA A100 (80 GB each) |
| Parallelism | FSDP (Fully Sharded Data Parallel) |
| Precision | Automatic Mixed Precision (AMP) |
| Batch size (3D) | 1 per device |
| Batch size (2D) | 4 per device |
| Gradient accumulation | 4 steps |
| Gradient checkpointing | Yes |
| **Inference minimum** | 1× A100 80GB |

### Key Training Design Choices
- **Never mix 2D and 3D in same batch** — custom sampler ensures batch is all-2D or all-3D (avoids expensive padding)
- **UMLS-weighted loss** — forces model to be accurate on medical terms
- **Freeze LLM in epoch 1** — stabilizes visual-text alignment before joint training

---

## 📈 Results

### RadBench — RadFM vs All Baselines

| Task | Metric | OpenFlamingo | MedVInT | LLaVA-Med | MedFlamingo | **RadFM** |
|---|---|---|---|---|---|---|
| VQA | BLEU | 6.42 | 3.95 | 25.19 | 20.56 | **30.88** |
| VQA | UMLS_P | 17.40 | 8.62 | 19.57 | 21.93 | **22.89** |
| Report | UMLS_P | 1.13 | 0.77 | 6.86 | 2.57 | **18.97** |
| Report | UMLS_R | 1.35 | 0.71 | 6.00 | 2.03 | **9.32** |
| Rationale | BLEU | 4.12 | 0.96 | 14.32 | 7.38 | **41.89** |
| Rationale | UMLS_P | 11.58 | 7.73 | 7.73 | 6.01 | **42.95** |

### Human Evaluation (3 radiologists, 0–5 scale, 1200 cases)

| Model | VQA | Report | Rationale | **Average** |
|---|---|---|---|---|
| OpenFlamingo | 2.21 | 0.89 | 1.09 | 1.40 |
| MedVInT | 1.44 | 1.16 | 0.83 | 1.14 |
| MedFlamingo | 1.74 | 0.86 | 1.03 | 1.21 |
| **GPT-4V** | 2.13 | 1.64 | **2.22** | 1.99 |
| **RadFM** | **2.87** | **1.88** | 1.76 | **2.17** |

**Key**: RadFM beats GPT-4V overall (2.17 vs 1.99)! GPT-4V only wins on rationale diagnosis.

### 3D-Specific Results (Task-Specific Fine-tuning)

| Dataset | Modality | Metric | Previous SOTA | **RadFM** |
|---|---|---|---|---|
| **BraTS 2019** | **MRI 3D** | **AUC / F1** | **88.06 / 90.36** | **90.61 / 92.21** |
| LDCT | CT 3D | AUC | 82.10 | 83.23 |
| MosMedData | CT 3D | AUC / F1 | 77.47 / 50.70 | 78.33 / 52.35 |
| ADNI | MRI 3D | AUC | 79.34 | 80.39 |
| BTM-17 | MRI 2D | AUC / F1 | 92.80 / 70.35 | 94.47 / 74.19 |

**CRITICAL**: RadFM achieves **90.61 AUC / 92.21 F1 on BraTS 2019 brain tumors!** This is directly relevant to our Yale brain metastases project.

---

## ⚠️ Limitations (Authors' Own Assessment)

1. **Report generation still weak**: No model scores above 3/5 in human rating. Long-form text generation needs improvement.
2. **3D data underrepresented**: Despite efforts, 2D images dominate training data.
3. **Missing spacing metadata**: 3D images from web lack voxel spacing → can't make size statements ("tumor is 3cm large").
4. **Automatic evaluation inadequate**: UMLS metrics help but don't fully capture clinical correctness.
5. **14B params = heavy**: Requires A100 80GB minimum for inference.

---

## 🔗 Code & Weights — Everything You Need

### Repository Structure
```
RadFM/
├── Quick_demo/                     ← START HERE for testing
│   ├── test.py                     ← Inference script (load model, run on one image)
│   ├── Language_files/             ← Tokenizer files
│   └── Model/RadFM/               ← Model code (same as src/)
│       ├── multimodality_model.py  ← Main model class: MultiLLaMAForCausalLM
│       ├── my_embedding_layer.py   ← MyEmbedding: ViT + Perceiver + token fusion
│       ├── vit_3d.py               ← 3D Vision Transformer
│       ├── helpers.py              ← PerceiverResampler, PerceiverAttention
│       ├── position_encoding.py    ← PositionEmbeddingLearned3d
│       └── transformer_decoder.py  ← TransformerDecoder (for keyword matching, unused)
├── src/                            ← Full training code
│   ├── train.py                    ← Training script
│   ├── test.py                     ← Evaluation script
│   ├── datasampler.py              ← Custom 2D/3D batch sampler
│   ├── My_Trainer/                 ← Custom HuggingFace Trainer
│   ├── Dataset/                    ← All dataset loading code
│   │   ├── multi_dataset.py        ← Main training dataset class
│   │   └── dataset/                ← Per-dataset processing
│   └── Model/RadFM/               ← Same model code
└── requirements.txt                ← Dependencies
```

### Download Links

| Resource | URL | Size |
|---|---|---|
| **Model checkpoint** | https://huggingface.co/chaoyi-wu/RadFM | ~28 GB (pytorch_model.bin) |
| **Model (Baidu mirror)** | https://pan.baidu.com/s/1A-K5nXCbvWAVqvb6dLjYJg?pwd=q1eo | Same |
| **GitHub code** | https://github.com/chaoyi-wu/RadFM | MIT license |
| **Dataset CSVs** | https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv | Train/test splits |
| **PMC-Inline data** | https://huggingface.co/datasets/chaoyi-wu/PMC-Inline | 11M images |
| **Med-KEBERT** | https://huggingface.co/xmcmic/Med-KEBERT | Required dependency |

### Quick Demo (Test in 5 Minutes)

1. Clone: `git clone https://github.com/chaoyi-wu/RadFM.git`
2. Download checkpoint from HuggingFace → decompress → get `pytorch_model.bin`
3. Put `pytorch_model.bin` in `Quick_demo/`
4. Run: `python Quick_demo/test.py`
5. Input: chest X-ray + question → Output: diagnosis text

### Minimum Requirements
- **Inference**: 1× A100 80GB GPU (or equivalent VRAM)
- **Training**: 32× A100 80GB with FSDP
- **Disk**: ~30GB for model weights, ~500GB+ for datasets

---

## 🔧 How RadFM Integrates Into OUR Pipeline

### The Key Insight: We DON'T Use RadFM As-Is

RadFM uses its OWN 3D ViT encoder. We already have **Swin UNETR** (better for brain tumors — trained on BraTS, Dice 0.913). The insight:

**We take RadFM's Perceiver + LLM pattern, but replace its 3D ViT with our Swin UNETR.**

### What We Take From RadFM

| Component | From RadFM | Our Adaptation |
|---|---|---|
| **Architecture pattern** | 3D ViT → Perceiver → LLM | Swin UNETR → Perceiver → LLM |
| **Perceiver module** | Compresses variable tokens to fixed 32 | Compress our 768-dim embeddings to 32 tokens |
| **Token fusion** | Interleave visual + text tokens in LLM | Same approach — inject our visual tokens into text sequence |
| **LLM base** | MedLLaMA-13B | Use MedLLaMA-13B (or smaller 7B variant to save compute) |
| **Training strategy** | Stage 1: freeze LLM, align vision → Stage 2: full fine-tune | Same — first align Swin UNETR embeddings with text, then instruct-tune |
| **UMLS loss weighting** | 3× weight on medical terms | Critical — forces accurate medical terminology |
| **Batch separation** | Never mix 2D/3D in batch | We only have 3D, but useful for mixed experiments |

### What We DON'T Take

| Component | Why Skip |
|---|---|
| RadFM's 3D ViT encoder | Our Swin UNETR is specialized for brain tumors (Dice 0.913 vs generic RadFM) |
| RadFM's training data (16M images) | We fine-tune on Yale brain metastases specifically |
| Multi-modality support (X-ray, CT, MRI) | We only need brain MRI |
| Med-KEBERT keyword matching | Unused even in RadFM (commented "not used in final version") |

### Our Modified Architecture

```
OUR PIPELINE (RadFM-inspired, brain tumor specialized):

Yale 3D MRI (128×128×128×4)
    ↓
Swin UNETR encoder (already trained, Phase 3)
    → 768-dim embedding per scan
    → Grad-CAM heatmaps (from TransXAI, Phase 4 Step 4.1)
    ↓
TaViT temporal model (already trained, Phase 3)
    → 768-dim temporal representation per patient
    → Temporal attention weights (WHICH timepoints matter)
    ↓
RadFM-style Perceiver (NEW — from RadFM code)
    → Takes: [scan_embedding(768) + temporal_repr(768) + grad_cam_features(flattened)]
    → Outputs: Fixed 32 tokens × 5120 dims
    ↓
Clinical metadata injection (NEW — from RaDialog concept)
    → Concatenate structured findings: tumor size, location, treatment history
    → Appended as text tokens to prompt
    ↓
MedLLaMA decoder (from RadFM weights)
    → Autoregressive text generation
    → Output: "Patient 67yo female, right frontal 15mm met, 20% growth T0→T1..."
```

### Dimension Compatibility Check ✅

| Component | Dimension | Compatible? |
|---|---|---|
| Swin UNETR output | 768 dims | ✅ Same as RadFM's ViT output (768) |
| RadFM Perceiver input | 768 dims | ✅ Direct connection |
| Perceiver output | 32 tokens × 768 dims | ✅ |
| Linear projection | 768 → 5120 | ✅ `nn.Linear(768, 5120)` already in code |
| MedLLaMA-13B input | 5120 dims | ✅ |

**Perfect match!** Our Swin UNETR outputs 768-dim embeddings → plugs directly into RadFM's Perceiver → projects to LLM. No dimension changes needed.

### Training Strategy for Our Project

**Option A — Full RadFM Fine-tuning** (if we have A100 80GB):
1. Load RadFM checkpoint (all 14B params)
2. Replace 3D ViT with frozen Swin UNETR (our pretrained encoder)
3. Retrain Perceiver + LLM on Yale brain tumor data
4. Freeze LLM first (epoch 1), then unfreeze (epochs 2+)

**Option B — Adapter-only** (if limited compute):
1. Load RadFM checkpoint
2. Freeze EVERYTHING except Perceiver + projection layer
3. Train only Perceiver to map Swin UNETR → LLM (6-layer transformer = ~50M params)
4. Much cheaper: single A100 sufficient

**Option C — Hybrid** (recommended):
1. Load RadFM checkpoint
2. Replace 3D ViT with frozen Swin UNETR
3. Train Perceiver from scratch on our embeddings
4. LoRA fine-tune LLM (low-rank adapters, only ~4M extra params)
5. Total trainable: ~54M params (vs 14B full model)

### Adding Temporal Reasoning (Our Novel Contribution)

RadFM processes SINGLE scans. Our project needs LONGITUDINAL sequences. Novel adaptation:

**Multi-scan input format**:
```
Prompt: "<image_T0> <image_T1> <image_T2> Based on this temporal sequence of 
brain MRIs taken 6 months apart, describe the disease progression and 
treatment response."

Where:
  <image_T0> = 32 Perceiver tokens from Swin UNETR embedding at time T0
  <image_T1> = 32 Perceiver tokens from Swin UNETR embedding at time T1
  <image_T2> = 32 Perceiver tokens from Swin UNETR embedding at time T2

Total visual tokens: 3 × 32 = 96 tokens (well within 2048 max)
```

RadFM ALREADY supports multi-image input (interleaved images + text). Our temporal sequence is just a special case of multi-image input where images are from the same patient at different times.

**With TaViT integration**:
```
Alternative: Instead of 3 separate scans, feed TaViT's single 768-dim 
temporal summary → Perceiver → 32 tokens → LLM

Advantage: Captures TEMPORAL RELATIONSHIPS (TaViT attention weights)
Disadvantage: Loses per-timepoint detail

Recommendation: Feed BOTH:
  - TaViT temporal summary (32 tokens) — captures progression trajectory
  - Individual scan embeddings (32 tokens each) — captures per-timepoint details
  - Grad-CAM features — captures spatial attention
  Total: ~128 tokens (still within 2048 limit)
```

---

## 📝 Key Takeaways for Our MASTER_PLAN

1. **RadFM proves 3D MRI → LLM text generation WORKS** — Nature Communications paper with 90.61 AUC on BraTS brain tumors
2. **Perceiver is the bridge** — compresses any-size 3D features to fixed 32 tokens for LLM
3. **768-dim perfect match** — Swin UNETR output dimension = RadFM ViT dimension = Perceiver input dimension
4. **Two-stage training is critical** — freeze LLM first to align, then fine-tune
5. **UMLS-weighted loss** — 3× weight on medical terms forces clinical accuracy
6. **Multi-image input supported** — temporal sequences can be fed as interleaved images
7. **Need A100 80GB minimum** — plan compute accordingly (discuss with supervisor)

---

## 🔗 Links

| Resource | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2308.02463 |
| Nature Communications | https://doi.org/10.1038/s41467-025-62385-7 |
| GitHub | https://github.com/chaoyi-wu/RadFM |
| Model weights | https://huggingface.co/chaoyi-wu/RadFM |
| Dataset CSVs | https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv |
| Project website | https://chaoyi-wu.github.io/RadFM/ |
