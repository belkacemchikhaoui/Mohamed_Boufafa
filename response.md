## Yes — This Is the End Goal. Here's Exactly What It Looks Like.

```
New unseen patient scans
         ↓
    STAGE 1: Embedding          (~5 sec/scan, GPU)
    STAGE 2: Temporal Profile   (~1 sec, CPU)
    STAGE 3: LLM Narrative      (~30 sec, API or local)
    STAGE 4: Video Generation   (~3-5 min, GPU)
         ↓
Clinical report + progression video
```

---

## The Full Inference Pipeline

### What You Need as Input (Per Patient)

```
patient_A/
  ├── tp100/
  │     ├── T1n.nii.gz
  │     ├── T1c.nii.gz
  │     ├── T2w.nii.gz
  │     └── T2-FLAIR.nii.gz  ← minimum: T1c only
  ├── tp101/
  │     └── (same 4 modalities)
  └── tp102/
        └── (same 4 modalities)

Optional: treatment_type, days_between_visits
```

### Stage 1 — Embedding Extraction (Already Built ✅)

```python
# Load once at startup
model = load_swinunetr(checkpoint="swinunetr_best_v4.pth")  # Option B weights
# Hook registered on layers3[0]

# Per scan (~5 sec)
emb = extract_embedding(scan_path)  # (3233,) vector
```

Produces per scan:
- `embedding (3233,)` — morphological snapshot
- `spatial_tokens (N, 384)` — for LLM Perceiver / TaDiff cross-attention
- `vol_features (9,)` — WT/TC/ET volumes

### Stage 2 — Temporal Profiling (Already Built ✅ — Phase 3C)

```python
# Sort visits, compute deltas
sequence = build_temporal_sequence(patient_visits)
# Produces:
#   drift per visit pair
#   ΔWT, ΔET volumes
#   most_changed_octant (growth direction)
#   is_accelerating flag
```

### Stage 3 — LLM Narrative (Phase 4 — to build)

Two options based on what's available:

```python
# Option A: Structured text prompt → any LLM (GPT-4, Llama, BioMedLM)
prompt = build_temporal_prompt(sequence)
narrative = llm_api(prompt)

# Option B: RadFM (if GPU available)
# spatial_tokens from all visits → Perceiver → MedLLaMA → narrative
narrative = radfm_inference(spatial_tokens_all_visits)

# Output: "Patient shows progressive ET recurrence at +90 days,
#          growing toward posterior-superior region. RANO: PD."
```

### Stage 4 — Video Generation (Phase 5 — to build)

```python
# Takes: past embeddings + narrative text embedding + treatment info
video_frames = tadiff_inference(
    past_embeddings = sequence['sequences_normed'][:3],  # last 3 visits
    text_embedding  = llm_embed(narrative),
    treatment_type  = "TMZ",
    time_gap_days   = 90
)
# Output: predicted future MRI + segmentation mask + video MP4
```

---

## What's Ready Right Now vs What Needs Phase 4/5

```
                                           Status
Stage 1: Embedding extraction            ✅ Built & tested
Stage 2: Temporal profiling              ✅ Built (Phase3_C1)
Stage 3: LLM — structured prompt         📋 Phase 4 (just prompt engineering, no training)
Stage 3: LLM — RadFM multimodal          📋 Phase 4 (model adaptation)
Stage 4: Video — TaDiff conditioning     📋 Phase 5 (model adaptation, not from scratch)
Stage 4: Video — counterfactual          📋 Phase 5 (swap treatment_type in input)
```

The key insight: **Stages 1 and 2 are done.** Stages 3 and 4 are *adaptations* of existing models, not new architectures. GPT-4 or Llama-3 already exists — you just need the right prompt. TaDiff/Make-A-Video already exists — you need to add your embedding as a conditioning input.

---

## For Unseen Scans — Two Scenarios

### Scenario A: New patient, single scan (no history)
```
Single scan → embedding → morphological report only
  - "WT volume: 18,200 voxels | ET: 450 | Core fraction: 42%"
  - "Necrosis detected | Enhancement moderate"
  - "Predicted trajectory: aggressive (H3 responder=0.816)"
  No video (need ≥2 visits for temporal conditioning)
```

### Scenario B: New patient, multiple visits (the real clinical use case)
```
3 visits → embeddings → temporal sequences → LLM narrative → video
  Full pipeline — everything works
  ~4-6 minutes total on a T4 GPU
  Output: PDF report + MP4 video + segmentation overlays
```

---

## Why This Matters for Your Research

The pipeline **is the contribution**. The paper's claim is:

> *"We present a framework that takes longitudinal glioma MRI scans as input and produces:  
> (1) semantically rich temporal embeddings,  
> (2) LLM-generated clinical evolution narratives,  
> (3) temporally consistent video predictions of future tumor state."*

The 18-test evaluation battery proves that embeddings are good enough to support all three. Once Option B is done and M6/T8 pass → the entire pipeline works on any new BraTS-format patient without retraining anything.

**Want me to start building Phase 4 (the LLM prompt template and narrative generation notebook) now?**