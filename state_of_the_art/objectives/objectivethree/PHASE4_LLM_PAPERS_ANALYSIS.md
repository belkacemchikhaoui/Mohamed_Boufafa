# Phase 4: Medical Vision-Language Models Analysis (Objective 3 — LLM Integration)

## 🎯 Context: What Objective 3 Needs

**Goal**: Take Swin UNETR embeddings + TaViT temporal representations + Grad-CAM explainability maps (from Objective 2) → Feed to an LLM → Generate clinically coherent explanations of disease progression.

**What we need from LLM papers**:
1. **How to map vision features into LLM input space** (bridge ViT embeddings → text generation)
2. **How to keep LLM frozen** while only training the adapter (we don't have compute to retrain a 7B+ model)
3. **How to generate structured medical reports** (not just generic text)
4. **Ideally**: Support for 3D volumes (MRI, not just 2D X-rays)
5. **Ideally**: Interactive/conversational abilities (doctor can ask follow-up questions)

**Pipeline reminder**:
```
Objective 2 output:
  → 768-dim Swin UNETR embeddings (per scan)
  → TaViT temporal representation (per patient sequence)
  → Grad-CAM heatmaps (WHERE tumors detected)
  → Modality contribution maps (WHICH MRI sequence matters)
  → TaViT attention weights (WHICH timepoints model focused on)
      ↓
[ADAPTER / PROJECTION LAYER] ← This is what we need to learn from papers
      ↓
LLM generates clinical narratives
```

---

## 📊 The 8 Papers to Investigate (Ranked by Priority for Our Project)

---

### ⭐⭐⭐ PRIORITY 1: RadFM — Generalist Foundation Model for Radiology (2023)
**Authors**: Wu et al. (Shanghai Jiao Tong University)

**What it does**: First multimodal radiology foundation model that supports **both 2D AND 3D** medical scans. Trained on 16M images (the MedMD dataset), handles interleaved image-text input, and generates reports/diagnoses/rationales.

**Why #1 Priority**: This is the ONLY model that natively handles **3D MRI volumes** — exactly what Yale brain scans are! All other models below are 2D only (chest X-rays).

**Key details**:
- **Architecture**: 3D Vision Transformer encoder + Perceiver module (compresses 3D features to 32 fixed-length tokens) + LLaMA-based LLM decoder
- **Training data**: 16M images across 2D and 3D, covering 5,000+ diseases
- **3D support**: Processes full 3D volumes (CT/MRI) natively, not slice-by-slice
- **Tasks**: Diagnosis, VQA, report generation, rationale generation
- **Performance**: Outperforms OpenFlamingo, MedFlamingo, MedVInT, GPT-4V on RadBench
- **Compute**: Requires at least 1× NVIDIA A100 80GB for inference

**For your project**:
- **What we take**: 3D ViT → Perceiver → LLM architecture pattern
- **How to adapt**: Replace RadFM's 3D ViT with our Swin UNETR encoder (already trained on Yale). Keep Perceiver concept to compress 768-dim embeddings into LLM-compatible tokens. Fine-tune adapter on Yale metadata.
- **Critical advantage**: Perceiver module maps variable-length 3D features to fixed 32 tokens → LLM can handle it
- **Limitation**: RadFM is not specifically trained on brain tumors or longitudinal data — we'd need to adapt for temporal sequences

**Links**:
- **arXiv**: https://arxiv.org/abs/2308.02463
- **GitHub**: https://github.com/chaoyi-wu/RadFM ✅ (MIT license, 524 stars)
- **Model weights**: https://huggingface.co/chaoyi-wu/RadFM ✅
- **Dataset CSV**: https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv
- **Website**: https://chaoyi-wu.github.io/RadFM/

**Download priority**: ⭐⭐⭐ Download and analyze FIRST — best fit for 3D brain MRI

---

### ⭐⭐⭐ PRIORITY 2: LLaVA-Med — Large Language-and-Vision Assistant for Biomedicine (2023)
**Authors**: Li et al. (Microsoft Research) — **NeurIPS 2023 Spotlight**

**What it does**: Adapts the general-purpose LLaVA model to biomedicine using curriculum learning. First aligns biomedical vocabulary (image-caption pairs), then masters instruction-following (GPT-4 generated Q&A). Near GPT-4 performance on medical VQA.

**Why #2 Priority**: Largest biomedical training data (15M figure-caption pairs from PubMed Central), best general-purpose medical VLM, and demonstrates the **curriculum learning** approach we should follow.

**Key details**:
- **Architecture**: CLIP ViT-L/14 vision encoder + linear projection → Mistral-7B (v1.5) / Vicuna (v1.0)
- **Training**: Curriculum learning — Stage 1: biomedical concept alignment (500K pairs) → Stage 2: instruction tuning (60K GPT-4 generated Q&A)
- **Training time**: <15 hours on 8× A100 GPUs
- **Data**: 15M biomedical figure-caption pairs from PMC (PubMed Central)
- **Performance**: Outperforms supervised SOTA on PathVQA, VQA-RAD, SLAKE
- **v1.5 update (May 2024)**: Significantly better, easier to use, direct HuggingFace loading

**For your project**:
- **What we take**: Curriculum learning strategy (align first, then instruct-tune), linear projection approach (simple but effective mapping from vision to LLM)
- **How to adapt**: Replace CLIP ViT with our Swin UNETR embeddings → linear projection → LLM. Follow same two-stage training: first align embeddings with medical text, then instruction-tune for brain tumor reports
- **Limitation**: 2D only (figure-caption pairs) — needs adaptation for 3D MRI volumes. Not specialized for radiology reports (more VQA-oriented)

**Links**:
- **arXiv**: https://arxiv.org/abs/2306.00890
- **GitHub**: https://github.com/microsoft/LLaVA-Med ✅ (2.1K stars, 6 contributors)
- **HuggingFace model**: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b ✅
- **Data download**: `sh download_data.sh` in repo

**Download priority**: ⭐⭐⭐ Download and analyze — curriculum learning is directly applicable

---

### ⭐⭐ PRIORITY 3: RaDialog — Radiology Report Generation + Conversational Assistance (2023)
**Authors**: Pellegrini et al. (TU Munich) — **Accepted MIDL 2025**

**What it does**: First publicly available VLM for radiology report generation AND interactive dialog. Integrates visual features + structured pathology findings (CheXpert labels) with LLM. Doctor can ask follow-up questions, request corrections, and verify findings.

**Why #3 Priority**: The **conversational interface** is exactly what our Objective 3 describes — "a doctor can ask follow-up questions about the report." Also demonstrates how to combine image features WITH structured clinical findings (like our treatment metadata).

**Key details**:
- **Architecture**: BioViL-T vision encoder → alignment module → findings classifier (14 pathology labels) → Vicuna-7B with LoRA fine-tuning
- **Key innovation**: Injects BOTH visual features AND structured findings into the LLM prompt — not just raw embeddings
- **Tasks**: Report generation, report correction, question answering, findings-based dialog
- **Data**: MIMIC-CXR (377K chest X-rays) + custom instruct dataset (PhysioNet)
- **Performance**: SOTA clinical correctness on MIMIC-CXR
- **Efficient**: Uses LoRA parameter-efficient fine-tuning (not full LLM retraining)

**For your project**:
- **What we take**: Dual-input approach (visual features + structured findings → LLM), LoRA fine-tuning strategy, conversational instruct dataset design, report correction ability
- **How to adapt**: Replace BioViL-T with Swin UNETR embeddings, replace CheXpert findings with our Grad-CAM explainability outputs + tumor size/location metadata. Create brain tumor instruct dataset following their template.
- **Critical insight**: Combining image features with structured clinical labels >> image features alone
- **Limitation**: 2D chest X-ray only. Would need significant adaptation for 3D brain MRI.

**Links**:
- **arXiv**: https://arxiv.org/abs/2311.18681
- **GitHub (v1)**: https://github.com/ChantalMP/RaDialog ✅ (113 stars, pretrained weights available)
- **GitHub (v2)**: https://github.com/ChantalMP/RaDialog_LLaVA ✅ (improved version, HuggingFace model)
- **HuggingFace model**: https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation ✅
- **Instruct dataset**: https://physionet.org/content/radialog-instruct-dataset/1.1.0/ (PhysioNet, credentialed)

**Download priority**: ⭐⭐ Download — conversational approach and structured findings injection are key

---

### ⭐⭐ PRIORITY 4: R2GenGPT — Radiology Report Generation with Frozen LLMs (2024)
**Authors**: Wang et al.

**What it does**: Maps image features into frozen LLM's word-embedding space using a tiny adapter (~5M params, 0.07% of model). LLM stays completely frozen — only the visual alignment module is trained.

**Why #4 Priority**: Demonstrates the **most efficient** approach — freeze the LLM completely, train only a tiny adapter. This is critical for us because we likely won't have compute to fine-tune a 7B model.

**Key details**:
- **Architecture**: Visual encoder (Swin Transformer or ViT) → alignment layer (three variants: shallow/delta/deep) → frozen LLaMA-2/GPT-style LLM
- **Key innovation**: Three alignment strategies:
  - **Shallow**: Single linear layer (simplest, fastest training)
  - **Delta**: Adds learnable residual to frozen visual features
  - **Deep**: Multi-layer adapter (best quality, most compute)
- **Training**: Only adapter trained (5M params vs 7B model)
- **Data**: IU-Xray, MIMIC-CXR (chest X-rays)
- **License**: BSD 3-Clause (permissive!)

**For your project**:
- **What we take**: Frozen LLM + visual adapter pattern, three alignment strategy comparison
- **How to adapt**: Our Swin UNETR 768-dim embeddings → R2GenGPT-style adapter → frozen LLaMA. Test all three alignment strategies to find best for brain tumor reports.
- **Critical advantage**: Minimal compute needed — train only 5M params, not 7B
- **Limitation**: 2D X-rays only. Simple report generation (no dialog, no temporal).

**Links**:
- **GitHub**: https://github.com/wang-zhanyu/R2GenGPT ✅ (113 stars, BSD-3 license)
- **Pretrained checkpoints**: https://drive.google.com/drive/folders/1ywEITWfYIAAYy0VY1IZ24Ec_GoNmkqIY (Delta alignment)
- **Search on arXiv**: "R2GenGPT radiology report generation frozen LLM"

**Download priority**: ⭐⭐ Download — adapter-only training is our most realistic compute option

---

### ⭐ PRIORITY 5: Med-Flamingo — Multimodal Medical Few-shot Learner (2023)
**Authors**: Moor et al. (Stanford/SNAP Lab)

**What it does**: Adapts Flamingo architecture to medicine. Can learn from just a few examples at inference time (few-shot) — no fine-tuning needed. Shows image, gives a few example Q&A pairs, then asks about a new image.

**Why #5 Priority**: Few-shot learning is interesting for our project — if we have limited brain tumor report examples, Med-Flamingo can generate reasonable reports from just 2-3 examples.

**Key details**:
- **Architecture**: CLIP ViT-L/14 + LLaMA-7B with Perceiver Resampler + cross-attention (OpenFlamingo base)
- **Key innovation**: Few-shot medical VQA — give 2-3 image+answer examples → model answers new question
- **Training**: Continued pretraining on interleaved medical image-text (publications + textbooks)
- **Performance**: +20% accuracy in clinician ratings vs baseline
- **Generates rationales**: Can explain WHY it made a diagnosis (not just answer)

**For your project**:
- **What we take**: Few-shot learning strategy (useful early on when we have few brain tumor reports), rationale generation capability
- **How to adapt**: Use as early prototype — give Med-Flamingo a few brain tumor report examples + our Grad-CAM heatmaps → see if it generates reasonable reports. Then replace with fine-tuned model later.
- **Limitation**: 9B parameters (large), few-shot only (not fine-tuned for specific task), 2D only

**Links**:
- **arXiv**: https://arxiv.org/abs/2307.15189
- **GitHub**: https://github.com/snap-stanford/med-flamingo ✅ (443 stars)
- **Base model**: Requires LLaMA-7B v1 weights (download from HuggingFace)

**Download priority**: ⭐ Lower priority — useful as baseline comparison, not main model

---

### ⭐ PRIORITY 6: 3D-CT-GPT — Radiology Reports from 3D CT (2024)
**Authors**: (Preprint, 2024)

**What it does**: Vision-Language model specifically designed for **3D CT scans**. Uses a CT-specific ViT encoder processing full 3D volumes + 3D pooling + projection → LLM generates reports.

**Why #6 Priority**: One of the few models that handles **3D volumes natively** (like RadFM). Specifically designed for chest CT, which is closer to our brain MRI than 2D X-ray models.

**Key details**:
- **Architecture**: 3D ViT encoder (processes full volume) → 3D pooling → projection layer → LLM decoder (VQA framework)
- **Focus**: Chest CT report generation (lung nodules, cancer staging)
- **3D native**: Processes volumetric data, not slice-by-slice
- **Status**: Preprint, no code release noted

**For your project**:
- **What we take**: 3D volume → LLM architecture pattern, alternative to RadFM's Perceiver approach
- **How to adapt**: Compare their 3D pooling approach vs RadFM's Perceiver for compressing Swin UNETR 3D features into LLM input tokens
- **Limitation**: No public code, limited evaluation, untested clinically, chest CT (not brain MRI)

**Links**:
- **Search on arXiv**: "3D-CT-GPT radiology report generation 3D CT"
- **GitHub**: ❌ Not public
- **Status**: Preprint only

**Download priority**: ⭐ Read only — for architecture comparison with RadFM

---

### ⭐ PRIORITY 7: MRG-LLM — Multimodal LLM for Medical Report Generation (2025)
**Authors**: (Preprint, 2025)

**What it does**: Combines frozen LLM with trainable visual encoder + dynamic prompt tuning. Learns to generate a custom "soft prompt" for each image.

**Why #7 Priority**: Introduces **dynamic prompting** — each image gets its own optimized prompt, not a fixed template. Interesting for temporal data where each timepoint needs different context.

**Key details**:
- **Architecture**: Frozen LLM + trainable visual encoder + affine transform → dynamic soft prompt per image
- **Two versions**: "prompt-wise" (one prompt per image) and "promptbook-wise" (template + variations)
- **Data**: IU-Xray, MIMIC-CXR
- **Status**: Preprint, no code released

**For your project**:
- **What we take**: Dynamic prompt tuning concept — each patient's temporal sequence could get a unique prompt encoding their scan history
- **How to adapt**: Generate temporal-aware prompts: T0 prompt = "baseline scan", T3 prompt = "follow-up after 6 months of treatment"
- **Limitation**: No code, 2D only, may overfit to training patterns

**Links**:
- **Search on arXiv**: "MRG-LLM multimodal medical report generation dynamic prompt"
- **GitHub**: ❌ Not public
- **Status**: Preprint only

**Download priority**: ⭐ Read only — for prompting strategy ideas

---

### 📌 REFERENCE: Med-PaLM 2 (Google, 2023)
**Authors**: Singhal et al. (Google Research)

**What it does**: Google's medical LLM. Achieves 86.5% on USMLE-style questions (expert-level). Not open-source.

**Why listed but not ranked**: **NOT open-source, NOT reproducible**. We can't use it directly. But it sets the performance bar and demonstrates what's possible with massive compute.

**Key details**:
- **Architecture**: PaLM-2 base + medical fine-tuning
- **Performance**: 86.5% on MedQA (USMLE), expert-level on multiple benchmarks
- **Training data**: ~600K medical examples
- **Access**: Google internal only, API access limited

**For your project**:
- **Use only as benchmark reference** — "Our system achieves X% vs Med-PaLM 2's Y%"
- **Cannot download, fine-tune, or deploy**

**Links**:
- **Paper**: https://arxiv.org/abs/2305.09617 (published in Nature, 2023)
- **Code/Weights**: ❌ Not public (Google proprietary)

---

## 🎯 The Verdict: Do These Papers Cover Objective 3?

### What's COVERED ✅:

| Need | Paper | Status |
|------|-------|--------|
| 3D MRI → LLM pipeline | RadFM (3D native!) | ✅ Perfect match |
| Curriculum learning for medical VLM | LLaVA-Med | ✅ Training strategy |
| Conversational report + dialog | RaDialog | ✅ Dialog interface |
| Frozen LLM + tiny adapter (low compute) | R2GenGPT | ✅ Efficient training |
| Few-shot learning (limited examples) | Med-Flamingo | ✅ Backup approach |
| Image features + structured findings → LLM | RaDialog | ✅ Critical insight |
| Prompt engineering for medical reports | MRG-LLM | ✅ Dynamic prompting idea |

### What's MISSING ❌:

| Gap | What we need | Solution |
|-----|---------------|----------|
| Temporal/longitudinal report generation | No paper generates reports from SEQUENCES of scans | Combine TaViT temporal attention (Paper 11) + RadFM's 3D adapter — novel contribution! |
| Brain tumor-specific reports | All papers focus on chest X-ray/CT | Fine-tune on Yale brain metastases metadata — requires custom training data |
| Integration with Grad-CAM heatmaps | No paper explicitly feeds explainability maps to LLM | Use RaDialog's structured findings approach — replace CheXpert labels with Grad-CAM outputs |
| Counterfactual reasoning | No paper does "what if different treatment" | Novel contribution — combine LLM reasoning + temporal embeddings |

### Bottom Line:

**YES, these papers are enough to build Objective 3!**

The core pattern is clear across ALL papers:
1. **Vision encoder** extracts features (we have: Swin UNETR)
2. **Adapter/projection** maps features to LLM space (learn from: R2GenGPT, RadFM)
3. **LLM** generates text (use: open-source 7B model like Mistral/LLaMA)
4. **Fine-tuning strategy**: Curriculum learning (LLaVA-Med) + LoRA (RaDialog) + frozen LLM (R2GenGPT)

**Our novel contribution** (not in any paper): Temporal sequence of 3D brain MRI embeddings + Grad-CAM explainability maps → LLM generates longitudinal progression reports. This is NEW.

---

## 📅 Recommended Analysis & Reading Order

### Priority 1 — Download and Deep-Analyze (3 papers):
1. **RadFM** — 3D support, Perceiver architecture, closest to our 3D brain MRI use case
2. **LLaVA-Med** — Curriculum learning strategy, largest training data, best documented
3. **RaDialog** — Conversational interface, structured findings + visual features → LLM

### Priority 2 — Download and Quick-Analyze (1 paper):
4. **R2GenGPT** — Frozen LLM + adapter pattern, minimal compute requirement

### Priority 3 — Read Only, Don't Deep-Analyze (2 papers):
5. **Med-Flamingo** — Few-shot strategy reference
6. **3D-CT-GPT** — 3D architecture comparison with RadFM

### Skip:
7. **MRG-LLM** — No code, dynamic prompting idea is minor
8. **Med-PaLM 2** — Not open-source, benchmark reference only

---

## 🔗 All Public Links

| Paper | arXiv | GitHub | Weights/Models | Priority |
|-------|-------|--------|----------------|----------|
| **RadFM** | [2308.02463](https://arxiv.org/abs/2308.02463) | [chaoyi-wu/RadFM](https://github.com/chaoyi-wu/RadFM) ✅ | [HuggingFace](https://huggingface.co/chaoyi-wu/RadFM) ✅ | ⭐⭐⭐ |
| **LLaVA-Med** | [2306.00890](https://arxiv.org/abs/2306.00890) | [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med) ✅ | [HuggingFace](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) ✅ | ⭐⭐⭐ |
| **RaDialog** | [2311.18681](https://arxiv.org/abs/2311.18681) | [ChantalMP/RaDialog](https://github.com/ChantalMP/RaDialog) ✅ | [HuggingFace](https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation) ✅ | ⭐⭐ |
| **R2GenGPT** | Search arXiv | [wang-zhanyu/R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) ✅ | [Google Drive](https://drive.google.com/drive/folders/1ywEITWfYIAAYy0VY1IZ24Ec_GoNmkqIY) ✅ | ⭐⭐ |
| **Med-Flamingo** | [2307.15189](https://arxiv.org/abs/2307.15189) | [snap-stanford/med-flamingo](https://github.com/snap-stanford/med-flamingo) ✅ | Requires LLaMA-7B | ⭐ |
| **3D-CT-GPT** | Search arXiv | ❌ Not public | ❌ | ⭐ |
| **MRG-LLM** | Search arXiv | ❌ Not public | ❌ | ⭐ |
| **Med-PaLM 2** | [2305.09617](https://arxiv.org/abs/2305.09617) | ❌ Google proprietary | ❌ | Reference only |

---

## 🧩 How These Papers Combine for Our Pipeline

**Step 1 — Vision-to-LLM adapter** (from RadFM + R2GenGPT):
- Swin UNETR 768-dim embeddings → Perceiver-style compression (RadFM) or linear adapter (R2GenGPT) → LLM token space

**Step 2 — Curriculum training** (from LLaVA-Med):
- Stage 1: Align brain MRI embeddings with medical text (unsupervised, using radiology vocabulary)
- Stage 2: Instruction-tune for brain tumor report generation (supervised, using Yale metadata)

**Step 3 — Structured input** (from RaDialog):
- Combine: visual embeddings + Grad-CAM heatmap features + tumor size/location + treatment history → LLM prompt
- Not just raw embeddings — structured clinical context dramatically improves report quality

**Step 4 — Conversational interface** (from RaDialog):
- Doctor asks: "What changed between T0 and T3?"
- System answers with temporal evidence from TaViT + visual evidence from Grad-CAM

**Step 5 — Efficient training** (from R2GenGPT):
- Freeze LLM (7B params) → Train only adapter (5-50M params)
- LoRA fine-tuning (RaDialog) if adapter alone isn't enough

---

## 🔑 Key Architecture Decision for Our Project

**Two approaches to compare (decide after reading papers):**

| Approach | Based on | How it works | Compute needed | Best for |
|----------|----------|-------------|---------------|----------|
| **Perceiver adapter** | RadFM | Compress 3D features → 32 fixed tokens → LLM | Medium | 3D volume handling |
| **Linear projection + LoRA** | LLaVA-Med + RaDialog | Project 768-dim → LLM space + LoRA fine-tune | Low | Quick prototyping |

**Recommendation**: Start with approach B (simpler), upgrade to A if quality is insufficient.
