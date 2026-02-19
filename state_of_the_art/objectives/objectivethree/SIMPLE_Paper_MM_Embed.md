# SIMPLE Analysis: MM-Embed — Universal Multimodal Retrieval with Multimodal LLMs

> **Paper**: "MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs"
> **Authors**: Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping (NVIDIA) + Jimmy Lin (University of Waterloo)
> **Venue**: ICLR 2025 (top-tier ML conference)
> **Source**: Provided by supervisor as reference for multimodal LLM work
> **Weights**: https://huggingface.co/nvidia/MM-Embed
> **PDF**: `by_prof/Attach_Lin_Jimmy_SRC2_RGPIN-2026-04544.pdf`

---

## S — Summary (What is this paper about?)

MM-Embed is the **first universal multimodal retriever** built on top of a Multimodal Large Language Model (MLLM). It handles **all combinations** of query and document modalities (text, image, interleaved text+image) across 16 diverse retrieval tasks simultaneously, while also maintaining state-of-the-art text-only retrieval.

**The problem they solve**: Existing retrievers are narrow — text retrievers only handle text, CLIP only handles image-text pairs, and no single model handles ALL modality combinations (text→image, image→text, text→text, image+text→image, etc.).

**Their solution**: Fine-tune LLaVa-Next (a multimodal LLM) as a bi-encoder retriever using contrastive learning, with two key innovations:
1. **Modality-aware hard negative mining** — teaches the model to respect the user's desired output modality
2. **Continuous text-to-text fine-tuning** — preserves strong text retrieval while adding multimodal capability

**Result**: State-of-the-art on both M-BEIR (multimodal retrieval, 52.7 avg R@5) AND MTEB text retrieval (60.3 nDCG@10, top-5 on leaderboard).

---

## I — Innovation (What's new?)

### 1. Modality Bias Discovery & Fix
- **Discovery**: When you fine-tune an MLLM for retrieval, it develops **text modality bias** — given a text query asking for images, it retrieves relevant TEXT instead of images
- **Cause**: MLLMs are pre-trained predominantly on text, so they prefer text representations
- **Fix**: Modality-aware hard negative mining with two types of negatives:
  - **C₁**: Wrong modality but semantically correct (teaches modality awareness)
  - **C₂**: Right modality but wrong content (teaches content discrimination)
- **Impact**: +5 points on M-BEIR after modality-aware mining

### 2. Universal Multimodal Retriever
- Single model handles **8 task types** with **16 datasets**:
  - text→image, text→text, text→(image+text), image→text, image→image
  - (image+text)→text, (image+text)→image, (image+text)→(image+text)
- **Task-specific instructions** guide the retriever to understand different search intents
- Same image query + different instruction = different retrieval behavior

### 3. Continuous Fine-Tuning Strategy
- Sequential training: multimodal retrieval FIRST → text-to-text retrieval SECOND
- Better than joint training (curriculum learning insight)
- Preserves multimodal capability while boosting text retrieval

### 4. Zero-Shot MLLM Reranking
- Prompt LLaVa-Next as True/False reranker on top-10 retrieved candidates
- Works especially well on challenging multi-modal queries (composed image retrieval)
- +7 mAP@5 improvement on CIRCO dataset over state-of-the-art

---

## M — Method (How does it work?)

### Architecture

**Backbone**: LLaVa-Next (Mistral 7B) as bi-encoder retriever
- **Image encoder**: CLIP ViT-L/14 → 576 image tokens (24×24 global patches)
- **Vision-language projector**: MLP aligning CLIP features to LLM embedding space
- **LLM**: Mistral 7B (or NV-Embed-v1, which is fine-tuned Mistral 7B for text retrieval)
- **Embedding extraction**: Use `<eos>` token or prompted "one word" summary as the query/document representation
- **Output**: Normalized d-dimensional vector per query/document

### Bi-Encoder Retrieval

Both query and document encoded independently → cosine similarity for ranking:

$$\text{score}(q, c) = \eta_\theta(\text{inst}, q) \cdot \eta_\theta(c)$$

Contrastive loss with in-batch negatives + mined hard negatives:

$$L = -\frac{1}{|B|} \sum_{i=1}^{|B|} \log \frac{\exp(\eta_\theta(\text{inst}_i, q_i) \cdot \eta_\theta(c_i^+) / \tau)}{\sum_{c' \in D} \exp(\eta_\theta(\text{inst}_i, q_i) \cdot \eta_\theta(c') / \tau)}$$

### Training Pipeline (3 Stages)

**Stage 1 — Random negatives (M^rand)**:
- Fine-tune on M-BEIR 1.1M training queries
- In-batch samples as random negatives
- 2 epochs, lr = 1e-4
- Only train: vision-language projector + LoRA (r=8, α=64) on LLM

**Stage 2 — Hard negatives (M^hard)**:
- Mine 2 types of hard negatives from M^rand's top-50 results:
  - C₁: Wrong modality, high rank (modality-confused)
  - C₂: Right modality, rank > 45 (content-confused)
- Re-initialize from pre-trained model (NOT from M^rand)
- Same training procedure but with hard negative triplets

**Stage 3 — Continuous text fine-tuning (MM-Embed)**:
- Continuously fine-tune M^hard on mixture of M-BEIR + 11 text retrieval datasets
- 4.5K additional steps, lr = 2e-5
- Hard negatives mined by NV-Embed-v1 for text tasks

### Reranking (Optional)

- Prompt LLaVa-Next as True/False classifier
- Example: "Does the above daily-life image match the caption? True or False"
- Relevance score = softmax probability of "True" token
- Rerank top-10 candidates from retriever

### Compute

- Training: 8× A100 80GB GPUs
- Batch size: 128×8 (random neg) or 64×8 (hard neg)
- DeepSpeed ZeRO-2 + gradient checkpointing
- Max query/doc length: 128 tokens (training), 512 for text docs

---

## P — Performance (Key numbers)

### M-BEIR Universal Multimodal Retrieval

| Model | All Tasks (R@5) | Single-modal Qry | Multi-modal Qry |
|---|---|---|---|
| CLIP_SF | 48.3 | 53.5 | 39.5 |
| M^hard (NVEmb) | 52.3 | 56.4 | 45.5 |
| **MM-Embed** | **52.7** | **56.1** | **47.0** |
| E5-V | 11.5 | 14.6 | 6.3 |
| MagicLens | 5.8 | 8.1 | 2.0 |

### MTEB Text Retrieval (nDCG@10)

| Model | Score |
|---|---|
| NV-Embed-v1 (original) | 59.36 |
| **MM-Embed** | **60.3** |
| M^hard (NVEmb) | 49.7 |
| M^rand (NVEmb) | 51.6 |

→ MM-Embed **surpasses** the text-only NV-Embed-v1 while adding multimodal capability!

### Key Task-Level Results

| Task | MM-Embed R@5 | Best CLIP R@5 | Improvement |
|---|---|---|---|
| WebQA (text→text) | 95.9 | 88.2 | +7.7 |
| EDIS (text→image+text) | 68.8 | 54.2 | +14.6 |
| InfoSeek (image+text→text) | 42.3 | 27.6 | +14.7 |
| InfoSeek (image+text→image+text) | 57.7 | 47.1 | +10.6 |

### Reranking Boost (CIRCO composed image retrieval)

| Model | Retrieval (mAP@5) | After Reranking | Δ |
|---|---|---|---|
| MagicLens | 24.9 | 32.4 | +7.5 |
| MM-Embed | 32.3 | 39.9 | +7.6 |
| M^hard (NVEmb) | 32.4 | 40.9 | +8.5 |

### Efficiency Comparison

| Model | Index Storage (5.6M docs) | Encoding Latency (50th perc.) | Vector Search |
|---|---|---|---|
| CLIP_SF | 16 GB | 27 ms | 6 ms |
| MM-Embed | 86 GB | 194 ms | 33 ms |

→ MM-Embed is **~5× larger** and **~7× slower** than CLIP but far more capable.

---

## L — Limitations

1. **Efficiency**: 5× more storage, 7× slower encoding than CLIP — prohibitive for real-time applications
2. **Single image resolution**: Only uses global 576 image tokens (24×24) — no high-resolution detail
3. **2D images only**: No 3D volume support (important for our MRI use case)
4. **Modality**: Only handles text and images — no audio, video, or 3D volumes
5. **Training data**: M-BEIR is mostly natural images + Wikipedia — not medical domain
6. **No generative output**: Pure retrieval model — doesn't generate text explanations (unlike RadFM)
7. **Compute**: 8× A100 80GB for training
8. **Report generation absent**: Cannot generate clinical narratives — only retrieves relevant documents

---

## E — Extraction for Our Project (What do we take?)

### Direct Relevance to Our Pipeline: ⭐⭐ MEDIUM (Conceptual, not Architectural)

**MM-Embed is NOT a replacement for RadFM in our pipeline.** It's a retrieval model, not a generative model. However, several ideas are highly relevant:

### What We Take

#### 1. Modality-Aware Training Strategy → For Our RadFM Fine-Tuning
- **The insight**: MLLMs have inherent text bias — they prefer text representations over visual ones
- **For us**: When fine-tuning RadFM (MedLLaMA-13B) on brain MRI, the LLM may also exhibit text bias — generating generic text patterns instead of attending to visual features
- **Action**: During RadFM instruction tuning, include hard negatives where the model must distinguish between:
  - Reports that match the visual evidence vs. reports that are medically plausible but don't match the specific scan
  - This is analogous to MM-Embed's modality-aware hard negative mining but adapted for generation

#### 2. Contrastive Learning for Medical Embeddings → For Embedding Quality Validation
- **The insight**: Bi-encoder contrastive learning creates semantically meaningful embedding spaces
- **For us**: We can validate our Swin UNETR + ComBat embeddings using contrastive retrieval:
  - Query: "Show me scans with growing enhancing tumors"
  - If embeddings are good → correct scans retrieved
  - This serves as an embedding quality metric beyond clustering
- **Action**: After ComBat harmonization (Phase 3, Step 3.1), test embedding quality using contrastive retrieval

#### 3. Instruction-Guided Retrieval → For Clinical RAG (Retrieval-Augmented Generation)
- **The insight**: Task-specific instructions enable same model to handle different search intents
- **For us**: After RadFM generates a preliminary report, use retrieval to find:
  - Similar historical cases from Yale database
  - Relevant treatment outcomes for patients with similar progression patterns
  - Supporting literature for the generated explanation
- **Action**: Potential Phase 4 enhancement — RAG pipeline: Swin UNETR embedding → retrieve similar cases → augment RadFM prompt with retrieved context

#### 4. Curriculum Learning Strategy → For Our Two-Stage Training
- **The insight**: Sequential training (multimodal first → text fine-tuning second) outperforms joint training
- **For us**: Validates our RadFM training plan:
  - Stage 1: Vision-language alignment (Perceiver + projection, freeze LLM)
  - Stage 2: Instruction tuning (LoRA on LLM)
  - **Don't mix both stages** — curriculum learning is better
- **Key quote from paper**: "continuously fine-tuning a multimodal retriever to enhance its text-to-text retrieval is more effective than fine-tuning a retriever across all retrieval tasks simultaneously"

#### 5. Zero-Shot Reranking → For Report Validation
- **The insight**: Prompting an MLLM as a True/False verifier improves results over retrievers alone
- **For us**: After RadFM generates a report, use a separate MLLM (or RadFM itself) to verify:
  - "Does this report accurately describe the imaging findings? True/False"
  - "Is the described progression consistent with the temporal sequence? True/False"
  - Self-verification loop for report quality control
- **Action**: Phase 4 quality assurance step after report generation

#### 6. NV-Embed Initialization → Practical Tip
- **The insight**: Initializing from a strong text retriever (NV-Embed-v1) and adding multimodal capability works better than training from scratch
- **For us**: When fine-tuning MedLLaMA-13B, start from the medical-domain pre-trained checkpoint rather than generic LLaMA — domain knowledge transfers well

---

### How This Paper Fits in Our Pipeline

```
Our Current Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Swin UNETR → 768-dim → ComBat → TaViT            │
│     ↓                                                       │
│ Phase 4: RadFM Perceiver → MedLLaMA-13B → Report           │
│     ↓                                                       │
│ MM-Embed ideas enhance this pipeline:                       │
│                                                             │
│ ① Modality-aware negatives → better RadFM training          │
│ ② Contrastive retrieval → embedding quality validation      │
│ ③ Instruction-guided RAG → enrich RadFM context             │
│ ④ Curriculum learning → validates 2-stage training           │
│ ⑤ Reranking as verification → report quality check          │
└─────────────────────────────────────────────────────────────┘
```

**MM-Embed is NOT in our pipeline directly** — it's a retrieval model and we need generation. But its training insights (modality bias, curriculum learning, hard negative mining) directly improve how we train RadFM.

---

## Key Takeaways for Discussion with Supervisor

1. **Professor shared this for multimodal LLM insights** — the paper shows how MLLMs handle mixed modalities (text+image), which is exactly what our RadFM integration does
2. **Modality bias warning**: MLLMs inherently prefer text over images — we must watch for this in RadFM training (monitor if LLM ignores visual features and generates generic text)
3. **Curriculum learning validated**: Sequential training > joint training — confirms our 2-stage RadFM plan
4. **Retrieval-Augmented Generation potential**: MM-Embed could complement RadFM by retrieving similar historical cases to enrich report generation
5. **ICLR 2025 + NVIDIA** — high credibility, citable for methodology justification

---

## Comparison with RadFM (Our Primary LLM Paper)

| Aspect | RadFM | MM-Embed |
|---|---|---|
| **Task** | Generation (reports, VQA answers) | Retrieval (find relevant documents) |
| **Modality** | 2D + 3D medical images | 2D natural + Wikipedia images |
| **Domain** | Medical (radiology) | General (news, fashion, Wiki, etc.) |
| **Architecture** | 3D ViT + Perceiver + MedLLaMA-13B | CLIP ViT-L + MLP + Mistral 7B |
| **Output** | Clinical narrative text | Embedding vector for similarity search |
| **Our use** | **Primary LLM component** | **Training methodology reference** |
| **3D support** | ✅ Yes (native) | ❌ No |
| **Medical domain** | ✅ Trained on 16M medical images | ❌ Natural images only |
| **Temporal support** | ❌ Single scan | ❌ No temporal |
| **Code** | ✅ MIT license | ✅ Weights on HuggingFace |
| **Venue** | Nature Communications 2025 | ICLR 2025 |

**Bottom line**: RadFM = our pipeline component. MM-Embed = our training methodology guide.

---

## Paper Statistics

- **Published**: ICLR 2025 (International Conference on Learning Representations)
- **Authors**: NVIDIA Research + University of Waterloo
- **Pages**: 20 (main paper + appendix)
- **Datasets**: M-BEIR (10 datasets, 16 tasks, 5.6M documents) + MTEB (15 text retrieval tasks)
- **Model size**: ~7B parameters (Mistral 7B backbone)
- **Training**: 8× A100 80GB, LoRA fine-tuning
- **Key result**: First model to be SOTA on both multimodal (M-BEIR) AND text (MTEB top-5) retrieval simultaneously
