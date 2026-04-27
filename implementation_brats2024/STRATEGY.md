# BraTS 2024 Post-Treatment Glioma — Definitive Implementation Plan

## Data Reality Check

After inspecting the downloaded zip files:

```
ZIP 1: Main Training Data          (still downloading, ~23 GB)
  → training_data/ → ~1350 cases → 5 files each (t1n, t1c, t2w, t2f, seg) ✅ HAS LABELS

ZIP 2: Additional Training Data    (5 GB, complete)
  → training_data_additional/ → 271 cases → 5 files each ✅ HAS LABELS
  
ZIP 3: Validation Data             (5 GB, complete)
  → validation_data/ → 188 cases → 4 files each ❌ NO LABELS (no seg masks!)
```

### Implications
- **Validation set has NO ground truth** → we cannot compute Dice on it directly
  (it's a challenge setting — you submit predictions to Synapse for scoring)
- **For our purposes:** We use Training + Additional Training data with our own train/val split
- **Or:** We use the pretrained models' known Dice scores as our benchmark,
  fine-tune on the training data, and evaluate with our own held-out validation

### Corrected Data Strategy
```
Option A (Recommended):
  Train:  Main Training Data (~1350 scans) — fine-tune here
  Val:    Additional Training Data (271 scans) — evaluate Dice here (has labels!)
  
Option B:
  Train:  Main + Additional combined (~1621 scans)
  Val:    Hold out 20% for validation with labels
  Test:   Submit to Synapse for official scoring (optional)
```

---

## What Metadata Says About Longitudinal Structure

From the Excel metadata (1,809 entries):
- **818 unique patients**
- **608 patients with ≥2 timepoints** (longitudinal)
- **179 patients with ≥3 timepoints**
- Naming: `BraTS-GLI-XXXXX-YYY` where XXXXX=patient, YYY=timepoint index

### Glioma Types
- Glioblastoma: 1,018 scans (56%)
- Astrocytoma: 315 (17%)
- Oligodendroglioma: 261 (14%)
- Glioma NOS: 177 (10%)
- Other: 38 (2%)

### Sites
- UCSF: 596, Duke: 542, Missouri: 373, UCSD: 254, Indiana: 44

---

## Phase 1 — Data Preparation & EDA

**Goal:** Reproduce preprocessing pipeline, organize longitudinal sequences, generate EDA report

**Notebook:** `Phase1_A1_BraTS2024_EDA.ipynb`

### Steps:
1. **Unzip** all 3 zip files into organized directories
2. **Verify** file structure: 4 modalities (t1n, t1c, t2w, t2f) + seg per training scan
3. **Label verification:** confirm labels are {0, 1, 2, 4} (BraTS glioma convention)
4. **Spatial alignment:** verify all modalities are co-registered (they should be)
5. **Tumor volume statistics:** WT, TC, ET volumes per scan
6. **Longitudinal organization:**
   - Parse patient IDs → group scans by patient
   - Order timepoints chronologically (100, 101, 102...)
   - Count longitudinal pairs available
7. **Temporal variability:** for patients with ≥2 timepoints, compute:
   - Volume change between timepoints
   - Spatial displacement of tumor center
8. **Create scan index** (`scan_index.json`) mapping patient → [ordered timepoints]

### Deliverables:
- Preprocessing pipeline (notebook is the pipeline)
- Cleaned, temporally organized scan index
- EDA report with figures (tumor volumes, longitudinal coverage, glioma type distribution)

---

## Phase 2 — CNN Baselines

**Goal:** Fine-tune TWO CNN models, evaluate segmentation, extract embeddings, compare

### Model Selection

| # | Model | Architecture | Pretrained | Source | Why |
|---|-------|-------------|------------|--------|-----|
| 1 | **nnU-Net** | nnU-Net v2 | BraTS 2021 | [Zenodo](https://zenodo.org/records/11582627) | Gold standard CNN, highest reported Dice |
| 2 | **SegResNet** | SegResNet (MONAI) | BraTS 2018 | [NVIDIA NGC](https://catalog.ngc.nvidia.com) | MONAI native, lightweight, easy to fine-tune |

### Notebook A: `Phase2_A1_nnUNet_Finetune.ipynb`

**What it does:**
1. Install nnU-Net v2 from pip
2. Download pretrained checkpoint (256 MB from Zenodo)
3. Set up nnU-Net-compatible folder structure from our NIfTI data
4. Fine-tune (transfer learning) on BraTS 2024 Post-Treatment training set
   - Pretrained on pre-treatment BraTS 2021
   - Post-treatment MRIs look DIFFERENT (resection cavities, radiation effects)
   - Fine-tuning IS necessary and justified
5. Evaluate Dice on held-out validation (Additional Training Data = 271 scans)
6. Extract embeddings from encoder bottleneck
7. Save: checkpoint + embeddings + Dice scores

### Notebook B: `Phase2_A2_SegResNet_Finetune.ipynb`

**What it does:**
1. Load SegResNet from MONAI (`monai.bundle.download('brats_mri_segmentation')`)
2. Fine-tune on BraTS 2024 Post-Treatment training set
3. Evaluate Dice on validation
4. Extract embeddings with v2 pipeline (ROI crop + octant + mask-weighted)
5. Save: checkpoint + embeddings + Dice scores

### Notebook C: `Phase2_B1_CNN_Embedding_Comparison.ipynb`

**What it does:**
1. Load embeddings from both models
2. t-SNE visualization (color by glioma type, by institution)
3. Linear probe: can embeddings predict glioma type?
4. Volume regression: can embeddings predict tumor volume?
5. **Key analysis:** Show limitations of STATIC (single-timepoint) modeling
   - "Same patient at different timepoints" should show embedding drift
   - But a single-timepoint CNN can't model this drift → motivates Phase 3

### Deliverables:
- nnU-Net trained weights + Dice scores
- SegResNet trained weights + Dice scores
- CNN embedding comparison report
- "Limitations of static modeling" analysis (key for Phase 3 motivation)

---

## Phase 3 — Vision Transformer + Longitudinal

**Goal:** Fine-tune TWO ViT models, extract temporal embeddings, model longitudinal evolution

### Model Selection

| # | Model | Architecture | Pretrained | Source | Why |
|---|-------|-------------|------------|--------|-----|
| 1 | **Swin UNETR** | Swin UNETR | BraTS 2021 (SSL + supervised) | [GitHub](https://github.com/INSTIG8R/swin-unetr) | Top BraTS performer, transformer architecture |
| 2 | **3D-TransUNet** | TransUNet 3D | BraTS 2023 | Published paper | Transformer + U-Net hybrid |

### Notebook A: `Phase3_A1_SwinUNETR_Finetune.ipynb`

**What it does:**
1. Download Swin UNETR fold 0 + fold 1 pretrained weights
2. Fine-tune on BraTS 2024 Post-Treatment
3. Evaluate Dice on validation
4. Extract ViT embeddings (encoder10 bottleneck → 768-dim features)
   - ROI crop to WT bbox + padding → resize to 96³
   - Octant pooling (8 × 768 = 6,144-dim)
   - Mask-weighted pooling (3 × 768 = 2,304-dim)
   - Total: 8,448-dim per scan

### Notebook B: `Phase3_A2_TransUNet_Finetune.ipynb`

1. Load 3D-TransUNet
2. Fine-tune on BraTS 2024 Post-Treatment
3. Extract embeddings from transformer encoder

### Notebook C: `Phase3_B1_Longitudinal_Embeddings.ipynb`

**THE key novel analysis:**
1. For each patient with ≥2 timepoints:
   - Extract embedding at each timepoint
   - Create temporal embedding sequence: [emb_t0, emb_t1, emb_t2, ...]
2. **M7 test:** Δembedding magnitude vs Δtumor_volume → Pearson correlation
   - "Do embedding changes track real biological changes?"
3. **Temporal trajectories:** t-SNE/UMAP with arrows showing patient trajectories
4. **CNN vs ViT comparison:**
   - Which architecture's embeddings better capture temporal evolution?
   - Expected: ViT captures more global context → better temporal tracking

### Notebook D: `Phase3_B2_TaViT_Training.ipynb` (if time allows)

1. Temporal Attention ViT: lightweight transformer that takes embedding sequences as input
2. Predicts future embedding or treatment response
3. This is the "longitudinal representation framework" deliverable

### Deliverables:
- Swin UNETR trained weights + Dice scores
- TransUNet trained weights + Dice scores
- Longitudinal embedding trajectories
- M7 temporal correlation results
- CNN vs ViT comparison for temporal tracking

---

## Kaggle Resource Budget

| Task | Sessions | Hours |
|------|----------|-------|
| Phase 2: nnU-Net fine-tune (1 run) | 1-2 | 6-12h |
| Phase 2: SegResNet fine-tune (1 run) | 1-2 | 4-8h |
| Phase 2: Embedding extraction | 1 | 2-4h |
| Phase 3: Swin UNETR fine-tune (1 run) | 2-3 | 8-12h |
| Phase 3: TransUNet fine-tune (1 run) | 1-2 | 6-10h |
| Phase 3: Embedding extraction + longitudinal | 1 | 3-5h |
| **Total** | **~8-12** | **~30-50h** |

This fits within 2-3 weeks of Kaggle quota (30h/week × 2 = 60h available).

---

## File Naming Convention

```
implementation_brats2024/
├── Phase1/
│   └── notebooks/
│       └── Phase1_A1_BraTS2024_EDA.ipynb
├── Phase2/
│   └── notebooks/
│       ├── Phase2_A1_nnUNet_Finetune.ipynb
│       ├── Phase2_A2_SegResNet_Finetune.ipynb
│       └── Phase2_B1_CNN_Embedding_Comparison.ipynb
├── Phase3/
│   └── notebooks/
│       ├── Phase3_A1_SwinUNETR_Finetune.ipynb
│       ├── Phase3_A2_TransUNet_Finetune.ipynb
│       ├── Phase3_B1_Longitudinal_Embeddings.ipynb
│       └── Phase3_B2_TaViT_Training.ipynb
├── Validation/
│   └── notebooks/
│       └── Validation_MU_Glioma_External.ipynb
├── STRATEGY.md        ← this file
└── README.md
```

---

## Label Convention Reminder

```
BraTS Glioma Labels:
  0 = Background
  1 = NCR (Necrotic / Non-Enhancing Core)
  2 = ED  (Peritumoral Edema / FLAIR hyperintensity)
  4 = ET  (Enhancing Tumor)    ← NOT label 3!

Sub-regions:
  WT (Whole Tumor)  = labels {1, 2, 4}
  TC (Tumor Core)   = labels {1, 4}
  ET (Enhancing)    = label {4}
```
