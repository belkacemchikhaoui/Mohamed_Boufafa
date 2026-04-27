# Our Goal — Explainable Cancer Progression Pipeline

## The Vision

Build an **end-to-end pipeline** that takes raw MRI scans provided by a clinician and produces **explainable, temporally grounded clinical narratives and progression videos** for brain metastasis patients.

---

## The Pipeline (5 Stages)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  STAGE 1 — DATA PREPARATION                                                │
│  Raw MRI scans (from the doctor)                                            │
│  → Intensity normalization (Z-score, BraTS protocol)                        │
│  → Spatial resampling (96³ for SwinUNETR)                                   │
│  → Longitudinal alignment (timeline + days-since-baseline)                  │
│  → Quality checks (resolution, orientation, artifacts)                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 2 — SEGMENTATION (if needed)                                         │
│  Preprocessed scans                                                         │
│  → Automatic tumor segmentation (if doctor hasn't provided masks)           │
│  → Output: binary tumour masks per scan                                     │
│  → Volume computation + morphological measurements                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 3 — REPRESENTATION LEARNING (Best Approach Found)                    │
│                                                                             │
│  Per-scan static representation:                                            │
│  → SwinUNETR (BrainSegFounder, pretrained on 42k brain MRIs)                │
│  → ROI crop (tumour bbox + padding) + Octant pooling                        │
│  → BSF v2 embedding (8448-dim per scan)                                     │
│  → + 18-dim PyRadiomics shape features                                      │
│  = Hybrid embedding (8466-dim) that captures tumour morphology              │
│                                                                             │
│  Per-patient temporal representation:                                       │
│  → TaViT (Temporal Attention Vision Transformer)                            │
│  → Sinusoidal time encoding on actual days since baseline                   │
│  → TEM: learnable temporal emphasis decay                                   │
│  → 128-dim trajectory embedding that captures tumour EVOLUTION              │
│                                                                             │
│  Key finding: CNN representations fail at temporal modeling.                 │
│  ViT + temporal attention (TaViT) is the way to handle                      │
│  longitudinal data — it captures progression, not just snapshots.           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 4 — LLM CLINICAL NARRATIVE GENERATION                                │
│                                                                             │
│  Inputs to LLM:                                                             │
│  → Trajectory embedding (128-dim) — encodes progression pattern             │
│  → Scan-level embeddings (8466-dim × T) — per-visit morphology              │
│  → Volume sequence [v₀, v₁, ..., vₜ] — tumour growth/shrinkage             │
│  → Clinical metadata (histology, dose, Karnofsky, age, # mets)             │
│  → Time points (days since baseline per visit)                              │
│                                                                             │
│  LLM tasks:                                                                 │
│  → Integrate imaging-derived features with clinical metadata                │
│  → Provide coherent, medically grounded reasoning about disease             │
│    progression (e.g., "initial response followed by regrowth")              │
│  → Explain WHY the model predicts response/non-response                     │
│  → Generate clinician-readable temporal narratives per patient              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 5 — PROGRESSION VIDEO GENERATION                                     │
│                                                                             │
│  Generate temporally consistent and clinically plausible videos             │
│  that visualize cancer progression over time:                               │
│                                                                             │
│  → Use the learned trajectory embeddings as conditioning signal             │
│  → Interpolate between real scan time points                                │
│  → Synthesize intermediate tumour states (what the tumour                   │
│    looked like between visits)                                              │
│  → Overlay clinical annotations (volume, response status)                   │
│  → Output: video showing tumour evolution from baseline to                  │
│    last follow-up, with narrated clinical interpretation                    │
│                                                                             │
│  The video must be:                                                         │
│  (a) Temporally consistent — smooth transitions, no artifacts               │
│  (b) Clinically plausible — volume/shape changes match real data            │
│  (c) Aligned with the LLM narrative — visual matches explanation            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What We Proved So Far (Phases 1–3)

| Phase | What we built | What we showed |
|---|---|---|
| **Phase 1** | Data preparation pipeline | Robust preprocessing for Cyprus-PROTEAS (39 patients, 170 scans) |
| **Phase 2** | CNN (DynUNet) representation | 8/16 downstream tests — CNNs fail at temporal modeling (T1=0.21, T3=-0.05) |
| **Phase 3** | ViT (SwinUNETR) + TaViT | **12/16 downstream tests** — temporal attention captures progression (T4 AUC 0.509→0.739) |

**Key scientific finding:** Static per-scan CNN embeddings cannot model longitudinal tumour evolution. Temporal attention (TaViT) over ViT embeddings is required to capture treatment response and volume dynamics.

---

## What Remains

| Phase | Task | Status |
|---|---|---|
| **Validation** | Run frozen pipeline on OpenBTAI (75 patients, never seen) → test generalisation | ⏳ Next |
| **Phase 4** | LLM clinical narrative generation from trajectory embeddings + clinical data | ⏳ After validation |
| **Phase 5** | Temporally consistent progression video generation | ⏳ Research phase |
| **Deployment** | Package everything into automated Python scripts | ⏳ Final |

---

## The End Product — Production Pipeline

> **The final deliverable is NOT Jupyter notebooks.**
> It is a set of **Python scripts** that run automatically, end-to-end,
> from raw MRI scans to clinical output — ready for a doctor to use.

### How It Works (Doctor's Perspective)

```
$ python run_pipeline.py \
    --scans /path/to/patient_scans/ \
    --clinical patient_info.json \
    --output /results/patient_P001/

# That's it. The doctor gets:
#   results/patient_P001/
#   ├── clinical_narrative.pdf      ← LLM-generated explanation
#   ├── progression_video.mp4       ← temporal evolution visualization
#   ├── summary_report.html         ← interactive dashboard
#   └── raw_outputs/
#       ├── embeddings.npz          ← for research use
#       ├── volumes.csv             ← volume at each time point
#       └── trajectory.json         ← response prediction + confidence
```

### Folder Structure of the Final Tool

```
explainable_cancer_progression/
│
├── run_pipeline.py                 ← MAIN ENTRY POINT (runs everything)
│
├── config/
│   ├── default_config.yaml         ← model paths, thresholds, LLM settings
│   └── model_weights/
│       ├── brainsegfounder.pt      ← frozen SwinUNETR checkpoint
│       ├── tavit_model.pt          ← frozen TaViT checkpoint
│       └── segmentation_model.pt   ← auto-segmentation model (if needed)
│
├── pipeline/
│   ├── __init__.py
│   ├── step1_preprocess.py         ← Z-score norm, resample 96³, quality check
│   ├── step2_segment.py            ← auto-segment if no masks provided
│   ├── step3_extract_static.py     ← BSF v2 → 8448-dim + 18-dim shape = 8466
│   ├── step4_extract_temporal.py   ← TaViT → 128-dim trajectory per patient
│   ├── step5_generate_narrative.py ← LLM integration: embeddings + clinical → text
│   └── step6_generate_video.py     ← progression video from trajectory embeddings
│
├── models/
│   ├── swin_unetr.py               ← BrainSegFounder architecture
│   ├── tavit.py                     ← TaViT architecture (TEM + sinusoidal PE)
│   └── utils.py                     ← octant pooling, shape features, etc.
│
├── outputs/
│   └── templates/
│       ├── report_template.html     ← HTML summary template
│       └── narrative_prompt.txt     ← LLM prompt template
│
└── tests/
    ├── test_preprocessing.py        ← unit tests for each step
    ├── test_extraction.py
    └── test_full_pipeline.py        ← integration test end-to-end
```

### What Each Script Does

| Script | Input | Output | GPU? |
|---|---|---|---|
| `step1_preprocess.py` | Raw `.nii` scans folder | Normalised 96³ volumes | No |
| `step2_segment.py` | Preprocessed scans (no masks) | Binary tumour masks | Yes |
| `step3_extract_static.py` | Preprocessed scans + masks | `embeddings.npz` (8466-dim per scan) | Yes |
| `step4_extract_temporal.py` | embeddings + timeline | `trajectory.npz` (128-dim per patient) | Yes |
| `step5_generate_narrative.py` | trajectory + volumes + clinical | `clinical_narrative.pdf` | No (API) |
| `step6_generate_video.py` | trajectory + scans | `progression_video.mp4` | Yes |
| `run_pipeline.py` | Raw scans + clinical info | **Everything above** | Yes |

### The Pipeline Runs Automatically

```python
# run_pipeline.py (simplified)
from pipeline import (
    preprocess, segment, extract_static,
    extract_temporal, generate_narrative, generate_video
)

def run(scan_dir, clinical_info, output_dir):
    # Stage 1: Preprocessing
    preprocessed = preprocess.run(scan_dir)
    
    # Stage 2: Segmentation (skip if masks already provided)
    masks = segment.run(preprocessed) if not masks_exist(scan_dir) else load_masks(scan_dir)
    
    # Stage 3: Representation extraction
    static_embs = extract_static.run(preprocessed, masks)      # 8466-dim per scan
    trajectory   = extract_temporal.run(static_embs)             # 128-dim per patient
    
    # Stage 4: Clinical narrative
    narrative = generate_narrative.run(trajectory, static_embs, clinical_info)
    
    # Stage 5: Progression video
    video = generate_video.run(trajectory, preprocessed, masks)
    
    # Save everything
    save_outputs(output_dir, narrative, video, static_embs, trajectory)
```

### How the Notebooks → Scripts Mapping Works

| Research Phase (notebooks) | Production Script |
|---|---|
| Phase1 preprocessing notebooks | `step1_preprocess.py` |
| MetSeg/BrainSegFounder training | `models/swin_unetr.py` (frozen weights only) |
| `Phase3_B1_BSF_ReExtraction_v2` | `step3_extract_static.py` |
| `Phase3_A4B_HybridBSF_Eval` (shape features) | `step3_extract_static.py` (shape part) |
| `Phase3_B3_TaViT_Training` | `step4_extract_temporal.py` (inference only) |
| Phase 4 LLM notebook (TBD) | `step5_generate_narrative.py` |
| Phase 5 video notebook (TBD) | `step6_generate_video.py` |

> [!IMPORTANT]
> **Every notebook we write during research becomes a script in the final tool.**
> This is why each notebook must have clean, modular code — it will be refactored into
> production Python at the end. The notebooks are for development and validation.
> The scripts are for the doctor.

---

## The Full Journey

```
Research notebooks (development)
    ↓
Validation on OpenBTAI (prove it works on unseen data)
    ↓
Refactor into Python scripts (production-ready)
    ↓
Doctor runs: python run_pipeline.py --scans /mri/ --output /results/
    ↓
Doctor reads: clinical_narrative.pdf + watches progression_video.mp4
```

**This is Explainable Cancer Progression.**
