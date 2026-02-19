# Explanation of `project.txt` + Study Plan

## 1) What `project.txt` is
`project.txt` is a **research project description** written in the style of a **Mitacs Globalink proposed research** document.

It contains:
- A **Background / literature review** (“state of the art”) describing what has been done in the field.
- A **research problem + gaps + research questions** explaining what is missing in current methods.
- **Objectives** describing what the intern/researcher will build and evaluate.
- A **timeline** (phases over ~20 weeks) describing how the work will be executed.
- A **references list** supporting the background.

The core idea of the project:
- Use **longitudinal medical imaging** (multiple scans over time for the same patient) to learn how tumors evolve.
- Combine:
  - **Large Vision Models / Vision Transformers** to extract meaningful visual representations.
  - **Large Language Models** to produce clinically grounded explanations.
  - **Generative video models (diffusion / transformers)** to visualize plausible future tumor progression as videos.
- Extend to **counterfactual scenarios**, e.g., “what if treatment A vs no treatment?”.

## 2) Section-by-section explanation

### 2.1 Background (1–2 pages)
This part is meant to convince reviewers that:
- You understand the existing literature.
- Your proposed project addresses a real, unsolved need.

#### (a) State of the Art and Relevant Literature
This subsection summarizes the evolution of methods:

1) **CNNs in medical imaging**
- CNNs are described as the dominant approach for classic tasks:
  - **Detection** (where is the tumor?)
  - **Segmentation** (pixel/voxel-level tumor mask)
  - **Classification** (tumor type, malignancy, etc.)
- The text mentions that progress is strongly supported by **public datasets**:
  - **TCGA**: includes multiple cancer types, often with pathology + genomic data; imaging availability varies.
  - **BraTS**: benchmark for brain tumor MRI segmentation.
  - **LIDC-IDRI**: lung CT scans with nodules annotated.

2) **Vision Transformers (ViT, Swin) as alternatives to CNNs**
- ViTs model images as sequences of patches and use **self-attention** to capture long-range relationships.
- Swin Transformer is a hierarchical transformer that scales better with image size using windowed attention.
- The motivation in medical imaging:
  - Better modeling of global context.
  - Potential gains in segmentation/classification.

3) **Gap: most models are “static”**
- Many models analyze a **single time point**.
- But cancer is a **temporal process** (progression/regression, response to treatment).
- Some longitudinal work exists (RNNs, temporal CNNs), but outputs are often **numbers** (risk scores) rather than **visual explanations**.

4) **LLMs in medicine**
- LLMs are positioned as a way to produce:
  - reasoning
  - summarization
  - explanations aligned with clinical language
- The limitation highlighted:
  - many systems generate text (reports), but do not tightly integrate **imaging-derived evidence** into reasoning.

5) **Generative models (diffusion) and the missing piece**
- Diffusion models are state-of-the-art in image/video generation.
- In medical imaging, generative models are often used for:
  - augmentation
  - reconstruction
  - privacy-preserving synthetic data
- The gap emphasized:
  - limited work on **disease progression visualization** (especially **temporally coherent** video conditioned on clinical context).

#### (b) Research Problem, Knowledge Gaps, and Research Questions
This subsection states the “why now / why this project” part.

The project claims 3 main limitations in current cancer AI:
- **Static analysis** (no explicit temporal modeling)
- **Low interpretability** (numbers are hard to communicate clinically)
- **No counterfactual support** (cannot simulate alternative treatments)

Research questions listed in the text (interpreted):
- **RQ1 (Representation learning)**: Can ViT-based models learn representations of tumor morphology over time?
- **RQ2 (Multimodal reasoning)**: Can LLMs combine imaging features + clinical metadata to produce medically grounded narratives?
- **RQ3 (Generative modeling)**: Can we generate temporally consistent, clinically plausible progression videos?
- **RQ4 (Counterfactuals)**: Can the system simulate different trajectories under different treatments?

### 2.2 Objectives (1–3 pages)
This part tells reviewers what you will actually do.

#### Overall Objective
Build and evaluate a **multimodal framework** for:
- analyzing longitudinal scans
- explaining progression
- simulating progression videos

It also states it fits into a larger TELUQ research initiative.

#### Specific Objectives (1–5)
Each objective is a “work package”.

**Objective 1 — Longitudinal data pipeline**
- Acquire datasets.
- Preprocess (normalization, alignment, temporal consistency).
- Organize into sequences per patient.

Why this matters:
- Longitudinal projects fail if the data is not aligned and consistently preprocessed.

**Objective 2 — Tumor representations with ViTs**
- Train/fine-tune ViT/Swin on cancer imaging.
- Compare against CNN baselines.
- Study embeddings over time.

Typical outputs:
- embeddings per scan/timepoint
- plots showing embedding trajectories
- downstream task performance (e.g., classification/segmentation)

**Objective 3 — LLM integration for reasoning/explanations**
- Fuse imaging embeddings with structured clinical variables (stage, treatment, etc.).
- Prompt LLMs to generate clinically coherent explanations.
- Connect text explanations to visual changes.

Important nuance:
- In practice, LLMs can’t “see” embeddings directly unless you design a representation the LLM can consume (e.g., summarized features, retrieved descriptors, structured tables, or a multimodal model).

**Objective 4 — Generative cancer progression videos**
- Adapt video diffusion / transformer video generation methods.
- Condition generation on multimodal embeddings.
- Explore counterfactual trajectories.

Key technical difficulty:
- Ensuring **temporal coherence** and **clinical plausibility** (not just pretty videos).

**Objective 5 — Evaluation + writing**
- Quantitative evaluation (accuracy, temporal consistency).
- Qualitative evaluation (clinical plausibility, interpretability).
- Final report + manuscript draft.

### 2.4 Timeline (6 phases over 20 weeks)
This is a proposed execution plan.

- **Phase 1 (Weeks 1–4)**: dataset selection + preprocessing + EDA
- **Phase 2 (Weeks 5–7)**: CNN baselines
- **Phase 3 (Weeks 8–11)**: ViT/Swin longitudinal embeddings
- **Phase 4 (Weeks 12–14)**: multimodal reasoning with LLM prompts
- **Phase 5 (Weeks 15–18)**: generative video modeling + counterfactuals
- **Phase 6 (Weeks 19–20)**: evaluation + final writing

### 2.5 Literature cited
The references are a starting point for your reading list.

Notes:
- There is a numbering inconsistency in the text (it references [13,14] earlier, but the list jumps from 12 to 14). That likely means one citation entry is missing or misnumbered.

## 3) What the “State of the Art” means here (and how to read it)
In this document, “state of the art” means:
- the best-known methods (as of 2023–2024) for:
  - vision models in medical imaging
  - transformers in imaging
  - multimodal clinical reasoning with LLMs
  - diffusion-based generative models

To *understand* the state-of-the-art, you want to understand:
- **What problem each paper solved**
- **What data + evaluation protocol** it used
- **What the bottlenecks are** (compute, data scarcity, bias, temporal alignment)
- **What remains unsolved** (in this project: interpretable temporal visualization + counterfactual progression)

## 4) Explanation of the key technical building blocks

### A) Longitudinal medical imaging (the “time” dimension)
Typical structure:
- Patient `p` has visits `t1, t2, …, tn`
- Each visit has 3D scan(s): CT or MRI (sometimes multiple sequences for MRI)

Core challenges:
- Different scanners / protocols
- Different slice thickness / voxel spacing
- Patient positioning differences
- Tumor annotations not always available at all timepoints

What “temporal consistency checks” usually mean:
- ensure timepoints are in correct order
- ensure imaging volumes are comparable (spacing/orientation)
- ensure alignment/registration is reasonable

### B) CNNs vs Vision Transformers
- **CNNs**: strong inductive bias for locality, efficient, common baselines.
- **ViTs**: global attention, may need more data or strong pretraining.
- **Swin**: practical transformer for high-res images.

In medical imaging, you must decide:
- **2D vs 3D** (slices vs volumes)
- segmentation vs classification vs self-supervised pretraining

### C) LLMs for clinical reasoning
In realistic workflows, LLM usage is often:
- summarizing structured patient info
- generating explanations from *interpretable* signals

A practical way to connect vision to LLM:
- Vision model produces:
  - tumor volume change
  - texture/heterogeneity descriptors
  - embedding similarity between timepoints
  - segmentation-derived measurements
- You convert those into a structured “patient timeline table”, then prompt the LLM to explain changes.

### D) Diffusion/video generation for progression
Video generation is harder than image generation because:
- coherence across frames
- preventing drift / hallucinated anatomy

In medical contexts, evaluation must include:
- plausibility of anatomy
- stability of non-tumor regions
- clinically meaningful tumor changes

### E) Counterfactual reasoning
Counterfactuals here means:
- generate a plausible tumor trajectory under a different treatment condition.

Key risk:
- you need strong assumptions and careful disclaimers; counterfactual outputs are hypothetical.

## 5) How to understand “the projects referred to” (datasets + papers)

### Datasets mentioned
- **BraTS**: best to start with for MRI tumor segmentation.
- **LIDC-IDRI**: good for CT nodules.
- **TCGA**: broader oncology ecosystem; data access/structure can be more complex.

### Papers/works mentioned (how to read them)
Suggested reading order:
1) **Vision Transformers basics**
- Dosovitskiy et al. (ViT)
- Liu et al. (Swin)

2) **Medical imaging transformers**
- UNETR (transformers for 3D segmentation)
- Self-supervised ViT pretraining for medical imaging

3) **Clinical language models**
- BioBERT (domain adaptation)
- “Large Language Models Encode Clinical Knowledge”

4) **Multimodal medical V-L**
- Li et al. (benchmark + empirical study)

5) **Diffusion models**
- DDPM (Ho et al.)
- Video latent diffusion (Align your Latents)

## 6) A concrete learning + execution plan (practical)
This is a plan you can follow to “understand the state of the art” *and* be able to implement the project.

### Phase 0 (prep, 3–5 days): build your foundations
- **Outcome**: you can load 3D medical images, visualize them, and understand the dataset structure.
- Learn:
  - NIfTI format (`.nii.gz`), DICOM basics
  - coordinate systems (RAS/LPS), spacing, resampling
  - train/val/test splitting at patient level

### Phase 1 (Week 1–2): baseline pipeline + EDA
- Build a small pipeline:
  - load volumes
  - normalize
  - resample to common spacing
  - (optional) register timepoints
- Create EDA:
  - tumor size distributions
  - number of visits per patient
  - missing metadata

### Phase 2 (Week 3–4): CNN baseline
- Pick one task:
  - segmentation (BraTS is ideal)
  - or classification
- Train a baseline CNN (e.g., U-Net for segmentation).
- Track metrics and save outputs.

### Phase 3 (Week 5–7): transformer baseline (ViT/Swin/UNETR)
- Fine-tune transformer model on the same task.
- Compare against CNN.
- Extract embeddings for each timepoint.

### Phase 4 (Week 8–10): “temporal representation” experiments
- Convert patient data into sequences.
- Analyze embedding trajectories:
  - distance between `t_i` and `t_{i+1}`
  - clustering progression patterns
- Define measurable temporal tasks:
  - predict increase/decrease in tumor volume
  - predict response category

### Phase 5 (Week 11–13): LLM explanation layer
- Define structured input to LLM:
  - clinical variables
  - quantitative changes from images
- Prompting experiments:
  - produce a “progression narrative”
  - produce “what changed and why” explanations

### Phase 6 (Week 14–18): generative progression (ambitious part)
Start simple:
- generate short sequences (few frames/timepoints)
- condition on:
  - previous scan
  - treatment label
  - time delta

Add constraints:
- preserve anatomy outside tumor
- enforce monotonicity where clinically reasonable (if applicable)

### Phase 7 (Week 19–20): evaluation + writing
- Quantitative metrics:
  - task performance (segmentation/classification)
  - temporal coherence metrics
- Qualitative review:
  - sanity-check generated videos
  - clinician feedback if possible

## 7) Templates you can use to learn faster

### Paper reading template
For each paper, write:
- **Problem**:
- **Data**:
- **Method**:
- **Loss / training**:
- **Evaluation**:
- **Key result**:
- **Limitations**:
- **How it connects to this project**:

### Experiment log template
For each run:
- **Dataset split** (patient IDs):
- **Preprocessing**:
- **Model**:
- **Hyperparameters**:
- **Metrics**:
- **Observations**:
- **Next change**:

## 8) Practical cautions (important in medical AI)
- **Data leakage**: never mix timepoints of the same patient across train/val/test.
- **Evaluation realism**: if you generate “future” frames, ensure you evaluate on held-out patients/timepoints.
- **Interpretability**: link explanations to measurable changes (volume, shape, enhancement, etc.).
- **Clinical claims**: generated counterfactuals must be presented as *hypotheses*, not predictions for treatment decisions.

## 9) If you want, I can tailor this to your situation
If you tell me:
- which dataset you plan to start with (BraTS / LIDC / other)
- whether you want segmentation or classification first
- what compute you have (GPU model, VRAM)

…I can rewrite the plan into a more specific week-by-week checklist.
