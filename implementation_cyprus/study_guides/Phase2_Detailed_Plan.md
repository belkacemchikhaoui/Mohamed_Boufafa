# Phase 2 -- Baseline Vision Models for Tumor Representation

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink -- TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Duration:** 3 weeks (Weeks 5-7)  
**Predecessor:** Phase 1 (COMPLETE)

---

## What Is Phase 2?

Phase 2 is the **control experiment** of the entire project. We train a CNN (3D U-Net) to do tumor segmentation and extract features. Then in Phase 3, we train a ViT (Swin UNETR) on the SAME task with the SAME data. By comparing the two, we prove whether ViT features are actually better.

**In plain words**: We're going to teach two "students" (CNN and ViT) the same lesson (tumor segmentation), give them the same exam (test data), and then see which one has a deeper understanding (embedding quality). The one with the deeper understanding will be better at the harder tasks later (LLM narratives in Phase 4, video generation in Phase 5).

```
Phase 2 (CNN, single timepoint)    vs    Phase 3 (ViT, temporal)
─────────────────────────────          ───────────────────────────
Met-Seg segmentation                    Swin UNETR segmentation
CNN encoder → 1024-dim embedding        ViT encoder → 768-dim embedding
Embedding quality metrics               Same exact metrics → fair comparison
Cannot model temporal evolution         Models temporal evolution via TaViT
```

Without the CNN baseline, we cannot claim the ViT adds value. This is standard scientific methodology.

---

## What Exactly Are We Doing? (Task Definition)

### Primary Task: Tumor Segmentation

We take a brain MRI volume (4 channels: T1, T1c, T2, FLAIR) and predict which voxels belong to which tumor subregion.

| Parameter | Value | Why This Value |
| --------- | ----- | -------------- |
| Input | 4-channel MRI, 96x96x96 | 4 modalities capture different tissue contrasts |
| Output | 4-class voxel mask (BG, NCR, ET, ED) | Standard BraTS tumor subregions |
| Ground truth | 171 expert masks from Cyprus | Professional annotations = gold standard |
| Primary metric | **Dice score per class** | Measures overlap: 0=nothing, 1=perfect |
| Secondary | Hausdorff distance 95% (mm) | Worst-case boundary error |

**Why segmentation?** Because it's the foundational task that both CNN and ViT must do. A model that can segment well has learned to UNDERSTAND tumor anatomy. But more importantly, the INTERNAL features the model builds while learning segmentation are what we really want — those become the embeddings for Phases 4-5.

### Secondary Task: Embedding Quality

After training segmentation, we DON'T throw away the model. We open it up and extract the internal features (embeddings) — these are the model's "understanding" of the tumor.

**How it works:**
1. The model has an **encoder** (analyzes the image) and a **decoder** (produces the segmentation mask)
2. Between encoder and decoder, there's a **bottleneck** — a compressed representation of the image
3. We extract this bottleneck → Global Average Pool → we get a single vector per scan (1024-dim for Met-Seg CNN, 768-dim for ViT)
4. This vector IS the model's understanding of that tumor at that moment

**Then we test these embeddings on tasks the model was NEVER trained for:**

| Test | What We Do | What It Proves |
| ---- | ---------- | -------------- |
| **Clustering** | Plot all 171 embeddings colored by histology → do same tumor types cluster? | Model learned tumor type without being told |
| **Linear probe** | Train tiny logistic regression on embeddings → predict treatment response | Embedding carries clinically useful info |
| **t-SNE** | Visualize embedding space | Qualitative structure check |

**Why these tests matter**: Two models can get the same Dice score (segmentation accuracy) but have VERY different "understanding." Think of two students who both score 80% on an exam — one memorized answers (CNN local patterns), the other truly understands the material (ViT global context). The second student will do better on new, harder questions. The embedding quality tests are those "harder questions."

This is exactly what the supervisor asks: *"Learn embeddings that capture tumor morphology, spatial heterogeneity, and temporal evolution."*

---

## Cross-Validation Strategy: Progressive Approach

### The Three Options

| Metric | 5-fold | 3-fold | Single split (1-fold) |
| ------ | ------ | ------ | --------------------- |
| Training runs | 5 | 3 | 1 |
| Kaggle GPU time | ~27 hrs | ~16 hrs | ~5.3 hrs |
| Kaggle sessions | 4 | 2 | 1 (same day) |
| Statistical reliability | Best (gold standard) | Good enough for paired comparison | Weak — no variance estimate |
| Can compute mean ± std | Yes (n=5) | Yes (n=3) | No — single number |
| Time to first result | ~2 weeks | ~1 week | ~Same day |
| For final publication | Required | Acceptable | Not publishable alone |

### Our Strategy: Start Quick, Scale Up

We use a **progressive approach** — start with the fastest option to validate everything works, then scale up:

```
Day 1:  Single split  → "Does the pipeline work? Is the model learning?" (2 hrs)
            │
            ▼
Week 1: 3-fold CV     → "What's the real performance? (mean ± std)" (16 hrs)
            │
            ▼
Later:  5-fold CV     → "Publication-quality numbers with tight confidence" (27 hrs)
```

**Why this order makes sense:**

**Step 1 — Single split (same day, ~2 hrs)**: This is NOT for final numbers. This is to catch bugs. If the data loading crashes, if the loss doesn't decrease, if there's a label mismatch — you find out in 2 hours instead of wasting 16 hours. Think of it as a dress rehearsal before the actual performance.

**Step 2 — 3-fold CV (1 week, ~16 hrs)**: This is the working result. With 3 folds, we get mean Dice ± standard deviation. When we compare CNN vs ViT in Phase 3, the comparison is statistically fair: both models see exact same train/test patients. 3-fold is enough for a paired Wilcoxon test (p-value).

**Step 3 — 5-fold CV (later, ~27 hrs)**: For the final report and publication, we upgrade to 5-fold. This gives tighter confidence intervals and more statistical power. We do this AFTER Phase 3 is complete, so we run 5-fold for BOTH CNN and ViT at once — making the final comparison table as strong as possible.

### Data Split Strategy

```
45 patients → 3 folds (15 patients each)    OR    5 folds (9 patients each)
                        ┌──────────────┐
Fold 1: Train on 30 pts │ Test on 15   │  (each run evaluates on unseen patients)
Fold 2: Train on 30 pts │ Test on 15   │
Fold 3: Train on 30 pts │ Test on 15   │
                        └──────────────┘
Stratified by: histology (NSCLC/SCLC/Breast) + treatment (RS/FSRT)
Constraint: a/b split patients (P04, P07, P17, P20, P23) always in the same fold
```

**Why patient-level splits (not scan-level)**: If patient P01's baseline scan is in training and P01's follow-up 1 is in test, the model has already "seen" P01's brain. This is data leakage — the test score would be artificially high. By splitting at patient level, the test patients are truly unseen.

**Why stratified**: We have 36 RS patients and 11 FSRT patients. Random splitting could put all 11 FSRT patients in one fold → that fold sees no FSRT during training. Stratified splitting ensures every fold has a proportional mix.

---

## Quick Test Plan — Session 0 (2 Hours Max)

**Purpose**: Validate the entire pipeline in one Kaggle session before burning GPU quota.

### What Happens During the Quick Test

```
00:00 - 00:05  Upload/load data, check all 171 scans load correctly
00:05 - 00:10  Build model, verify forward pass (no crash, no OOM)
00:10 - 01:30  Train Fold 0 for 15 epochs (~5 min/epoch)
01:30 - 01:40  Evaluate: compute Dice on test fold
01:40 - 01:50  Extract embeddings from all 171 scans
01:50 - 02:00  Save everything (model checkpoint, embeddings, metrics)
```

### What We Check

| Check | What We Look At | Expected | If It Fails |
| ----- | --------------- | -------- | ----------- |
| Data loading | All 171 scans load as (4, 96, 96, 96) tensors | No errors, shapes correct | Fix paths, check NIfTI integrity |
| Forward pass | Model produces (batch, 4, 96, 96, 96) output | No OOM, no NaN | Reduce batch_size to 1 or resolution to 64³ |
| Loss decreasing | DiceCE loss drops by >50% over 15 epochs | Smooth decrease | Check: labels correct? All zeros? Wrong loss function? |
| Dice > 0 | At least one class has Dice > 0.1 after 15 epochs | Dice ED > 0.2 (edema is easiest) | Model isn't learning — check everything |
| Embedding extraction | Can hook into bottleneck and extract 256-dim vectors | 171 vectors saved | Fix hook layer index |
| Checkpoint saves | Model + metrics save to Kaggle output | Files exist | Check disk space |

### Decision After Quick Test

| Result | Action |
| ------ | ------ |
| ✅ Loss drops, Dice > 0.3 on at least one class | **Proceed to 3-fold training** |
| ⚠️ Loss drops slowly, Dice 0.1-0.3 | Try: higher LR (3e-4), more augmentation, deeper model |
| ❌ Loss flat or increasing | Debug: visualize predictions, check if masks are all zeros, check class balance |
| 💥 OOM crash | Reduce resolution to 64³ or batch_size to 1 |
| 💥 Data loading crash | Fix NIfTI paths, check spacing/orientation |

```python
# === QUICK TEST CONFIG ===
config = {
    'resolution': (96, 96, 96),
    'batch_size': 2,
    'epochs': 15,
    'fold': 0,
    'n_folds': 3,
    'patience': 10,
    'lr': 1e-4,
    'save_embeddings': True,
}
```

---

## Available Data (from Phase 1)

| Resource | Details | Size |
| -------- | ------- | ---- |
| BraTS MRI | 45 patients, 187 timepoints, 4 modalities | 1.67 GB |
| Tumor masks | 171 expert segmentations (labels 0-3) | 3.7 MB |
| Brain masks | 45 brain region masks | 11 MB |
| CT + RTP | 45 each (for later phases) | 960 MB |
| Clinical metadata | `PROTEAS_Clinical_Cleaned.xlsx` (28 cols) | 12 KB |
| Radiomic features | 7,980 pre-extracted | 6 MB |
| **Total on Kaggle** | **All NIfTI + Excel** | **2.65 GB** |

**Important data facts from Phase 1 EDA**:
- 84/171 masks have all 3 tumor subregions (NCR+ET+ED), 72 have only 2 (ET+ED), 14 have only ET, 1 has only ED
- Mean tumor volume: WT ~22.8 cm³, but high variance (CV=1.61)
- 16 timepoints have no masks (likely post-treatment resolution) → excluded from training

---

## Compute Requirements

### GPU Memory (VRAM)

| Component | 96³ input |
| --------- | --------- |
| 3D U-Net model (~5.3M params) | 21 MB |
| Feature maps (forward + backward) | 300 MB |
| Skip connections | 150 MB |
| Optimizer states (AdamW) | 42 MB |
| **Batch=2, total** | **~1.0 GB** |
| **Batch=4, total** | **~2.0 GB** |

Fits easily in T4 (15 GB VRAM). Room for Swin UNETR (~62M params, ~4-6 GB) in Phase 3.

### Training Time Estimates

| Config | Per epoch | Full (100 ep) | 3-fold total | 5-fold total |
| ------ | --------- | ------------- | ------------ | ------------ |
| 96³, batch=2, 1xT4 | ~5.5 min | ~9 hrs | ~27 hrs | ~45 hrs |
| 96³, batch=4, 2xT4 | ~3.2 min | ~5.3 hrs | **~16 hrs** | ~27 hrs |

### Kaggle Constraints

| Constraint | Limit | Our usage (3-fold) | Our usage (5-fold) |
| ---------- | ----- | ------------------ | ------------------ |
| Session duration | 12 hrs max | ~8 hrs/session | ~8 hrs/session |
| Weekly GPU quota | 30 hrs | ~16 hrs (1 week) | ~27 hrs (1 week) |
| T4 VRAM | 15 GB x 2 | ~2 GB used | ~2 GB used |
| System RAM | 13 GB | ~3 GB (lazy loading) | ~3 GB |
| Disk (input) | 73 GB | 2.65 GB | 2.65 GB |

---

## Session Plan

### Session 0: Quick Test (~2 hrs)

```
├── Load data from Kaggle Dataset
├── Train Fold 0 for 15 epochs
├── Check: loss decreasing? Dice > 0.1?
├── Extract embeddings for all 171 scans
├── Save checkpoint + metrics + embeddings
└── DECISION: proceed to 3-fold or debug
```

### Session 1: 3-Fold Training Part 1 (~8 hrs)

```
├── Train Fold 0: 100 epochs with early stopping (~5.3 hrs)
├── Train Fold 1: 100 epochs (~5.3 hrs) 
│   [if time tight: start Fold 1, save checkpoint at 8hr mark]
├── Save best checkpoints to Kaggle output
└── Save metrics JSON for each fold
```

### Session 2: 3-Fold Training Part 2 + Baselines (~8 hrs)

```
├── Resume Fold 1 (if not done) or Train Fold 2 (~5.3 hrs)
├── Train remaining fold (~5.3 hrs)
├── RF/XGBoost on radiomic features (~10 min, no GPU)
├── Extract CNN embeddings for ALL 171 scans (all 3 fold models)
├── Run embedding quality tests (clustering, linear probe, t-SNE)
├── Generate comparison table + figures
└── Save everything for Phase 3 comparison
```

### Session 3 (LATER): 5-Fold Final (~11 hrs)

```
This runs AFTER Phase 3 is complete, for both CNN and ViT simultaneously:
├── Train Fold 3 + Fold 4 for CNN (~10.6 hrs)
├── (Phase 3 also trains Fold 3 + Fold 4 for ViT)
├── Combine all 5 folds → publication-quality numbers
└── Final CNN vs ViT comparison table with 5-fold mean ± std
```

**Total for 3-fold: 3 sessions, ~18 hrs GPU (within 1 week quota)**  
**Total for 5-fold: +1 session later, ~11 hrs extra**

---

## Activities — Explained Step by Step

### Activity 1: Data Preparation (Week 5, Days 1-3) — Local, No GPU

**1.1 Create data splits**

Why we save splits as JSON: Phase 3 MUST use identical splits. If we randomly split again, we can't fairly compare. The JSON file is the contract.

```python
import json
from sklearn.model_selection import StratifiedKFold

# Create both 3-fold and 5-fold splits (for later)
# Saved as JSON so Phase 3 uses IDENTICAL splits
splits = {
    '3fold': {
        'fold_0': {'train': ['P01', 'P02', ...], 'test': ['P03', ...]},
        'fold_1': {'train': [...], 'test': [...]},
        'fold_2': {'train': [...], 'test': [...]},
    },
    '5fold': {
        'fold_0': {...}, 'fold_1': {...}, 'fold_2': {...},
        'fold_3': {...}, 'fold_4': {...},
    }
}
json.dump(splits, open('data_splits.json', 'w'))
```

**1.2 MONAI data pipeline**

```python
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Spacingd, Orientationd, RandFlipd, RandRotate90d, 
    RandScaleIntensityd, CropForegroundd, Resized
)

# Why these specific transforms:
# - LoadImaged: reads NIfTI files
# - Spacingd: resamples to uniform 1mm³ (already done for Cyprus, but defensive)
# - RandFlipd/Rotate90d: augmentation (2x effective dataset size)
# - RandScaleIntensityd: ±10% intensity jitter (robustness to scanner variation)

model = UNet(
    spatial_dims=3,
    in_channels=4,       # T1, T1c, T2, FLAIR stacked
    out_channels=4,      # BG=0, NCR=1, ET=2, ED=3
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,     # residual connections per level
)
# ~5.3M parameters
```

### Activity 2: Quick Test (Week 5, Day 4) — Kaggle Session 0

Run 15-epoch single-fold test. See Quick Test section above.

### Activity 3: Full 3-Fold Training (Week 5-6) — ✅ COMPLETE

**Actual implementation used TWO architectures (not one as originally planned):**

| Model | Architecture | Params | Epochs | Best Mean Dice | Status |
| ----- | ----------- | ------ | ------ | -------------- | ------ |
| SegResNet | MONAI SegResNet (BraTS 2023 pretrained) | 4.7M | 50 | 0.368 ± 0.044 | ✅ 3/3 folds |
| Met-Seg v1 | DynUNet + DenseNet121 (BraTS-METS pretrained) | 28M | 30 | 0.413 ± 0.016 | ✅ 3/3 folds |
| Met-Seg v2 | Same arch, cosine LR + heavy aug | 28M | 50 | 0.360 ± 0.002 | ❌ FAILED |
| Met-Seg v3/v4 | Same arch, AdamW + step LR + det unfreeze | 28M | 60-80 | **0.505 ± 0.024** | ✅ 3/3 folds |

**3.1 Training configuration**

| Parameter | Value | Why |
| --------- | ----- | --- |
| Architecture | MONAI `UNet` (3D) | Standard, well-tested, MONAI built-in |
| Input | 4ch, 96x96x96 | 4 modalities, fits T4 VRAM at batch=4 |
| Output | 4 classes | BG, NCR, ET, ED (matches expert masks) |
| Loss | `DiceCELoss` | Dice handles class imbalance, CE stabilizes training |
| Optimizer | AdamW, lr=1e-4, wd=1e-5 | AdamW prevents over-fitting; 1e-4 is safe default for 3D medical |
| Scheduler | Cosine annealing, T_max=100 | Slowly reduces LR → finer convergence |
| Epochs | 100, early stopping patience=20 | 100 is enough; stop early if validation plateaus |
| Mixed precision | Yes (AMP) | ~2x speed, no quality loss |
| Augmentation | Flip, Rotate90, Intensity ±10% | Prevents overfitting on 30 train patients |

**3.2 Embedding extraction**

After training, we extract the **bottleneck features** from the trained model. This is what we compare to ViT in Phase 3.

```python
# Extract bottleneck features as "CNN embeddings"
def extract_embeddings(model, dataloader):
    """
    What this does:
    1. Pass each MRI through the trained 3D U-Net
    2. Hook into the bottleneck layer (between encoder and decoder)
    3. The bottleneck contains a 256-dim feature map (shape: 256 x 12 x 12 x 12)
    4. Global Average Pool → single 256-dim vector per scan
    5. This vector IS the CNN's "understanding" of that tumor
    
    In Phase 3, we do the same with Swin UNETR → 768-dim vector.
    We then compare: which vector carries more useful information?
    """
    embeddings = {}
    bottleneck_output = []
    
    # Hook into bottleneck layer
    def hook_fn(module, input, output):
        bottleneck_output.append(output)
    
    handle = model.model[8].register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch['image'].cuda())
            # Global average pool: (batch, 256, 12, 12, 12) → (batch, 256)
            emb = bottleneck_output[-1].mean(dim=(-3, -2, -1))
            for i, pid in enumerate(batch['patient_id']):
                embeddings[pid] = emb[i].cpu().numpy()
            bottleneck_output.clear()
    
    handle.remove()
    return embeddings  # dict: patient_timepoint → 256-dim vector
```

**3.3 Feature-based baselines** (no GPU needed)

These use the 7,980 pre-extracted radiomic features from Phase 1, NOT our CNN embeddings. They serve as a second reference point:

| Model | Input | Task | Purpose |
| ----- | ----- | ---- | ------- |
| Random Forest | 107 radiomic features | Response prediction | How good are hand-crafted features? |
| XGBoost | 107 radiomic features | Histology classification | Same |
| Logistic Regression | 107 radiomic features | Both | Simplest possible baseline |

The comparison chain becomes:
```
Hand-crafted radiomic features  <  CNN embeddings  <  ViT embeddings  (expected)
```

---

### Activity 4: Evaluation — Complete Embedding Test Battery (Week 7) — ✅ COMPLETE

**Status: ✅ DONE** — Met-Seg: 12/16 passed, SegResNet: 7/16 passed

**Notebooks:** `Phase2_A4_MetSeg_Embedding_Eval.ipynb` + `Phase2_A4_SegResNet_Embedding_Eval.ipynb` — ran locally on CPU (~2 hours total)

**Prerequisites (all met ✅):**
- SegResNet embeddings: 170 × 128-dim, 3 folds ✅
- Met-Seg v1 embeddings: 170 × 1024-dim, 3 folds ✅
- Met-Seg v3 embeddings: will be extracted after each fold
- Radiomic features from Phase 1: 7,980 features ✅
- Clinical metadata: `PROTEAS_Clinical_Cleaned.xlsx` ✅
- Patient timelines: `cyprus_patient_timelines.csv` ✅

**Concrete execution plan:**

```
Step 1: Load all data (~5 min)
  ├── Load SegResNet embeddings (3 folds × 170 × 128-dim)
  ├── Load Met-Seg v1 embeddings (3 folds × 170 × 1024-dim)
  ├── Load Met-Seg v3 embeddings (3 folds × 170 × 1024-dim)
  ├── Load radiomic features (7,980 features)
  ├── Load clinical metadata (histology, treatment, dates)
  └── Load patient timelines (timestamps for each visit)

Step 2: Morphology tests M1-M6 (~15 min)
  ├── M1: Ridge(embedding → tumor volume), report R²
  ├── M2: Ridge(embedding → sphericity), report R²
  ├── M3: Ridge(embedding → surface-volume ratio), report R²
  ├── M4: LogReg(embedding → necrosis present), report F1
  ├── M5: Ridge(embedding → elongation), report R²
  └── M6: 5-NN consistency check for morphology

Step 3: Heterogeneity tests H1-H4 (~15 min)
  ├── H1: Pearson(embedding PCA vs GLCM features)
  ├── H2: Ridge(embedding → GLCM entropy), report R²
  ├── H3: LogReg(embedding → #subregions), report F1
  └── H4: Multi-output Ridge(embedding → texture bundle)

Step 4: Temporal tests T1-T7 (~30 min)
  ├── T1: Pearson(embedding L2 distance, volume change)
  ├── T2: t-SNE trajectories per patient (visualization)
  ├── T3: Ridge(Δembedding → Δvolume), report R²
  ├── T4: LogReg(baseline embedding → response), report AUC
  ├── T5: Cosine similarity between consecutive timepoints
  ├── T6: Embedding velocity vs progression speed
  └── T7: RS vs FSRT group separation over time

Step 5: Generate figures and tables (~15 min)
  ├── t-SNE colored by histology, treatment, timepoint
  ├── Temporal trajectory plots (5 example patients)
  ├── Bar charts: R² scores across all tests
  └── Final 4-column comparison table
```

The supervisor asks us to: *"Learn embeddings that capture tumor morphology, spatial heterogeneity, and temporal evolution."*

Below is the **complete test battery** to prove each of these three properties, evaluated against what our Cyprus data actually has. Each test is marked ✅ (feasible now), ⚠️ (feasible with limitations), or ❌ (not possible with our data).

---

#### 4.1 Segmentation — "Which model draws boundaries closer to the expert?"

| Model | Dice NCR | Dice ET | Dice ED | Mean Dice | HD95 (mm) |
| ----- | -------- | ------- | ------- | --------- | --------- |
| 3D U-Net (Phase 2, 3-fold) | \_\_ ± \_\_ | \_\_ ± \_\_ | \_\_ ± \_\_ | \_\_ ± \_\_ | \_\_ ± \_\_ |
| Swin UNETR (Phase 3) | TBD | TBD | TBD | TBD | TBD |
| 3D U-Net (5-fold, later) | TBD | TBD | TBD | TBD | TBD |
| Swin UNETR (5-fold, later) | TBD | TBD | TBD | TBD | TBD |

---

#### 4.2 🔬 Tumor Morphology Tests — "Do embeddings encode shape, size, and structure?"

"Morphology" = shape, size, boundary regularity, necrotic core, enhancement pattern. We have **14 shape radiomic features** from Phase 1 (Sphericity, Elongation, MeshVolume, SurfaceVolumeRatio, etc.) and **171 tumor masks with known label distributions** (84 with NCR, 72 without). These are our ground truth.

| # | Test | What We Do | What It Proves | Feasibility | Time |
| - | ---- | ---------- | -------------- | ----------- | ---- |
| M1 | **Volume prediction** | Train linear regression: embedding → tumor volume (mm³). Gold standard: actual volume from mask. | Embedding encodes tumor SIZE | ✅ 171 samples, ground truth from masks | 2 min |
| M2 | **Sphericity prediction** | Train linear regression: embedding → sphericity score. Gold standard: radiomic `original_shape_Sphericity` | Embedding encodes tumor SHAPE | ✅ We have this radiomic feature per mask | 2 min |
| M3 | **Surface-to-volume ratio** | Train linear regression: embedding → SVR. Gold standard: radiomic `original_shape_SurfaceVolumeRatio` | Embedding encodes boundary complexity | ✅ We have this | 2 min |
| M4 | **Necrosis detection** | Binary classification: embedding → necrosis present/absent. Gold standard: 84 masks have label 1 (NCR), 87 don't | Embedding encodes internal architecture | ✅ Clean binary split ~50/50 | 2 min |
| M5 | **Elongation prediction** | Train linear regression: embedding → elongation. Gold standard: radiomic `original_shape_Elongation` | Embedding captures asymmetric shapes | ✅ We have this | 2 min |
| M6 | **Nearest-neighbor consistency** | For each embedding, find 5 nearest neighbors → are they morphologically similar? Measure: how often are NN's volumes within ±30% | Embedding space geometry reflects morphology | ✅ No extra data needed | 5 min |

**How we physically do M1 (example)**:
```python
# 1. Extract CNN embeddings for all 171 timepoints (already done in Activity 3)
# 2. Compute actual tumor volume from each mask (sum of non-zero voxels × voxel volume)
# 3. Train: LogisticRegression(embedding → volume)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X = np.stack([embeddings[pid] for pid in patient_ids])  # (171, 256)
y = np.array([tumor_volumes[pid] for pid in patient_ids])  # (171,)

r2 = cross_val_score(Ridge(), X, y, cv=3, scoring='r2').mean()
print(f"Volume prediction R² = {r2:.3f}")
# If R² > 0.6 → embedding captures tumor size
# If R² < 0.3 → embedding doesn't encode size well
```

**Pass criteria**: R² > 0.6 for volume and shape predictions. We expect CNN to get R² ~0.5-0.7 and ViT to get R² ~0.7-0.9 because ViT's global attention captures the full tumor extent, while CNN may miss parts of large tumors due to limited receptive field.

---

#### 4.3 🗺️ Spatial Heterogeneity Tests — "Do embeddings capture non-uniform texture?"

"Spatial heterogeneity" = non-uniform internal texture. A homogeneous tumor looks the same everywhere inside; a heterogeneous one has necrotic regions, different enhancement patterns, and variable texture. We have **1,824 GLCM features** and **1,064 GLDM features** from Phase 1 as ground truth.

| # | Test | What We Do | What It Proves | Feasibility | Time |
| - | ---- | ---------- | -------------- | ----------- | ---- |
| H1 | **GLCM entropy correlation** | Compute Pearson correlation: embedding PCA components vs `glcm_ClusterTendency`, `glcm_Contrast`. | Embedding dimensions align with established heterogeneity metrics | ✅ We have GLCM per mask+modality | 5 min |
| H2 | **Heterogeneity score probe** | Train linear regression: embedding → GLCM entropy. Gold standard: radiomic `original_glcm_DifferenceEntropy` | Embedding directly encodes heterogeneity | ✅ Clear scalar target | 2 min |
| H3 | **Multi-label detection** | Classify embedding → how many subregions present (1, 2, or 3 out of NCR/ET/ED). Gold standard: known from masks (84 have 3, 72 have 2, 15 have 1) | Embedding distinguishes complex vs simple tumors | ✅ Clear 3-class target | 2 min |
| H4 | **Texture feature bundle** | Train multi-output regression: embedding → [GLCM contrast, GLRLM run-length variance, GLSZM zone entropy]. Measure average R² across texture features | Embedding captures texture ensemble | ✅ We have all these features | 5 min |
| H5 | **Attention map analysis** | Extract ViT attention maps → do they highlight heterogeneous intra-tumoral regions? | Model FOCUSES on spatially varied regions | ⚠️ Phase 3 only (ViT has attention maps, CNN doesn't) | 10 min |

**How we physically do H2 (example)**:
```python
# For each of 171 timepoints, we have radiomics AND embeddings
# GLCM DifferenceEntropy = one number measuring how "textured" the tumor is
# High entropy = heterogeneous, Low entropy = uniform

X = np.stack(all_embeddings)      # (171, 256)
y = glcm_entropy_values            # (171,) from radiomics

r2 = cross_val_score(Ridge(), X, y, cv=3, scoring='r2').mean()
# If R² > 0.5 → embedding carries heterogeneity info
```

**Pass criteria**: Pearson r > 0.5 between embedding-derived predictions and actual radiomic heterogeneity features. CNN should get r ~0.4-0.5 (captures local texture but misses global patterns), ViT should get r ~0.6-0.8 (self-attention sees texture at ALL scales).

**Why ViT should win here**: CNN convolutions look at local 3x3x3 patches — they can see "this patch has high contrast" but cannot see "the left side of the tumor has high contrast while the right side doesn't." ViT's self-attention compares ALL patches simultaneously, so it naturally captures SPATIAL differences in texture.

---

#### 4.4 ⏱️ Temporal Evolution Tests — "Do embeddings track how tumors change?"

"Temporal evolution" = how the tumor changes between visits. We have 45 patients with 3-7 timepoints each (mean 5.2), and timestamps in `cyprus_patient_timelines.csv`. We can compute tumor volume at each timepoint from the masks, giving us ground truth for biological change.

| # | Test | What We Do | What It Proves | Feasibility | Time |
| - | ---- | ---------- | -------------- | ----------- | ---- |
| T1 | **Embedding distance vs volume change** | Compute L2(emb_fu1 - emb_baseline) and Δvolume(fu1 - baseline). Correlation? | Embedding shift = biological change | ✅ 45 patients × avg 4 pairs | 5 min |
| T2 | **Temporal t-SNE trajectories** | Plot each patient's embeddings over time as connected trajectory in t-SNE space | Do trajectories look smooth and directional? | ✅ Visual, no labels needed | 5 min |
| T3 | **Volume change prediction** | Train regression: Δembedding → Δvolume. Gold standard: actual volume change from masks | Embedding CHANGE predicts tumor CHANGE | ✅ ~140 pairs (consecutive timepoints) | 5 min |
| T4 | **Response classification** | Train classifier on baseline embedding → predict response at 6 months (volume decreased ≥20% = response, increased ≥20% = progression, else stable) | Pre-treatment embedding predicts future | ⚠️ n=29 with 6-month data. Small but testable | 5 min |
| T5 | **Temporal coherence** | For each patient's scan sequence, compute cosine similarity between consecutive embeddings → average across patients | Nearby-in-time embeddings should be similar (high cosine) | ✅ All 45 patients | 2 min |
| T6 | **Embedding velocity vs time-to-progression** | Rate of embedding change (‖emb_t2 - emb_t1‖ / days) → correlate with clinical progression speed | Fast-changing embeddings = aggressive tumors | ✅ Timestamps available | 5 min |
| T7 | **Treatment group separation** | At each timepoint, compute mean embedding for RS group vs FSRT group → does gap change over time? | Treatment effect visible in embedding space | ✅ 36 RS vs 11 FSRT | 5 min |

**How we physically do T1 (example)**:
```python
# For each patient, compute embedding distance and volume change
# between baseline and each follow-up

results = []
for patient in patients:
    emb_baseline = embeddings[f"{patient}_baseline"]
    vol_baseline = volumes[f"{patient}_baseline"]
    
    for fu in ['fu1', 'fu2', 'fu3']:
        if f"{patient}_{fu}" in embeddings:
            emb_fu = embeddings[f"{patient}_{fu}"]
            vol_fu = volumes[f"{patient}_{fu}"]
            
            emb_distance = np.linalg.norm(emb_fu - emb_baseline)
            vol_change = vol_fu - vol_baseline
            results.append({'emb_dist': emb_distance, 'vol_change': abs(vol_change)})

# If Pearson correlation > 0.5 → embedding distance tracks biological change
r, p = pearsonr([r['emb_dist'] for r in results], 
                [r['vol_change'] for r in results])
print(f"Correlation: r={r:.3f}, p={p:.4f}")
```

**Pass criteria**: 
- T1: Pearson r > 0.4 between embedding distance and volume change
- T3: R² > 0.3 for volume change prediction
- T4: AUC > 0.6 for response classification (with n=29, even 0.65 is meaningful)
- T5: CNN cosine similarity should be ~0.7-0.8 (random, no temporal info), ViT+TaViT should be ~0.9+ (temporally coherent)

**CNN vs ViT expectation for temporal tests**:
- **CNN**: Can compute T1, T2, T5, T6, T7 (it produces per-scan embeddings). But embeddings from different timepoints are INDEPENDENT — there's no temporal information built in. So correlations will be moderate.
- **ViT + TaViT (Phase 3)**: Same tests but with time-aware embeddings. TaViT explicitly models temporal relationships, so T5 (coherence) should be much higher and T3 (volume change prediction) should be much better.

---

#### 4.5 Summary — Full Test Battery with Time Estimates

| Category | Tests | Total Samples | GPU Time | CPU Time | Notes |
| -------- | ----- | ------------- | -------- | -------- | ----- |
| Segmentation | Dice, HD95 | 171 × 3 folds | 0 (already trained) | ~10 min | Standard metrics |
| **Morphology (M1-M6)** | 6 tests | 171 embeddings | 0 | **~15 min** | All use radiomics as ground truth |
| **Heterogeneity (H1-H4)** | 4 tests (H5 Phase 3 only) | 171 embeddings | 0 | **~15 min** | All use GLCM/texture radiomics |
| **Temporal (T1-T7)** | 7 tests | 45 patients, ~140 pairs | 0 | **~30 min** | Uses timeline + volume changes |
| **Total embedding tests** | **17 tests** | | **0 GPU** | **~60 min** | All run on CPU after embedding extraction |

**Key insight: the embedding tests cost ZERO additional GPU time.** We extract embeddings once (during Activity 3, ~1 hr GPU), then all 17 tests run on CPU in ~1 hour. The only cost is the analysis code, which we can prepare locally before the Kaggle session.

---

#### 4.6 What We Produce — Comparison Tables

**Segmentation table**:

| Model | Dice NCR | Dice ET | Dice ED | Mean Dice | HD95 (mm) |
| ----- | -------- | ------- | ------- | --------- | --------- |
| 3D U-Net (Phase 2) | \_\_ | \_\_ | \_\_ | \_\_ | \_\_ |
| Swin UNETR (Phase 3) | TBD | TBD | TBD | TBD | TBD |

**Morphology table** (R² for regressions, F1 for classification):

| Test | Radiomic Features | CNN Embedding (256-dim) | ViT Embedding (768-dim) |
| ---- | ----------------- | ----------------------- | ----------------------- |
| M1: Volume prediction (R²) | \_\_ | \_\_ | TBD |
| M2: Sphericity prediction (R²) | \_\_ | \_\_ | TBD |
| M3: Surface-volume ratio (R²) | \_\_ | \_\_ | TBD |
| M4: Necrosis detection (F1) | \_\_ | \_\_ | TBD |
| M5: Elongation prediction (R²) | \_\_ | \_\_ | TBD |
| M6: NN morphology consistency (%) | \_\_ | \_\_ | TBD |

**Heterogeneity table**:

| Test | Radiomic Features | CNN Embedding | ViT Embedding |
| ---- | ----------------- | ------------- | ------------- |
| H1: GLCM correlation (r) | 1.0 (trivially) | \_\_ | TBD |
| H2: Entropy probe (R²) | \_\_ | \_\_ | TBD |
| H3: Multi-label detection (F1) | \_\_ | \_\_ | TBD |
| H4: Texture bundle (avg R²) | \_\_ | \_\_ | TBD |

**Temporal table** (CNN baseline vs ViT in Phase 3):

| Test | CNN Embedding | ViT Embedding | ViT + TaViT |
| ---- | ------------- | ------------- | ----------- |
| T1: Emb distance vs ΔVolume (r) | \_\_ | TBD | TBD |
| T3: ΔEmb → ΔVolume (R²) | \_\_ | TBD | TBD |
| T4: Response prediction (AUC) | \_\_ | TBD | TBD |
| T5: Temporal coherence (cosine) | \_\_ | TBD | TBD |
| T6: Emb velocity vs progression (r) | \_\_ | TBD | TBD |
| T7: Treatment group separation (d) | \_\_ | TBD | TBD |

**The three-column comparison (Radiomic vs CNN vs ViT) is key**: it shows the hypothesis `Radiomics < CNN < ViT` at every level.

---

#### 4.7 Limitation Analysis — "What Can CNN NOT Do?"

| Limitation | Example | Why CNN Fails | Which Test Quantifies It |
| ---------- | ------- | ------------- | ------------------------ |
| Growth vs stability | P01 baseline=10cm³, fu3=12cm³ | CNN has no concept of "before" | T1: low emb_dist↔ΔVol correlation |
| Recurrence vs necrosis | Post-RT bright lesion | Same appearance on single scan | T4: low response AUC |
| Future prediction | Will P06 respond to FSRT? | No temporal modeling | T4: cannot predict from baseline |
| Treatment history | Did P03 already receive WBR? | Sees only images | T7: cannot separate RS vs FSRT |
| Temporal dynamics | Is tumor slowing down? | No concept of rate | T6: no velocity information |

These limitations directly justify Phase 3 (ViT + TaViT).

---

## Phase 2 → Phase 3 Comparison Plan

### What Stays the SAME (Controlled Variables)

| Item | Value | Why Shared |
| ---- | ----- | ---------- |
| Data splits | `data_splits.json` (3-fold AND 5-fold) | Fair comparison — same patients in test |
| Input resolution | 96x96x96 | Both models see same input |
| Input channels | 4 (T1, T1c, T2, FLAIR) | Same modalities |
| Output classes | 4 (BG, NCR, ET, ED) | Same task |
| Loss function | DiceCELoss | Same training objective |
| Optimizer | AdamW, lr=1e-4 | Same optimization |
| Evaluation metrics | Dice, HD95, Silhouette, linear probe F1 | Same ruler |
| Augmentation | Flip, Rotate90, intensity ±10% | Same data augmentation |

### What CHANGES (Experimental Variables)

| Aspect | Phase 2 (CNN) | Phase 3 (ViT) | What We're Testing |
| ------ | ------------- | -------------- | ------------------ |
| Architecture | 3D U-Net | Swin UNETR | Does global attention beat local convolution? |
| Parameters | ~5M | ~62M | Does scale help? |
| Receptive field | 3x3x3 local kernel | Shifted windows → whole volume | Does seeing more context help? |
| Embedding dim | 256 | 768 | Does more capacity = more information? |
| Temporal modeling | ❌ None | ✅ TaViT with time encoding | Does time-awareness help? |
| Input per prediction | Single scan | Can use sequence | Does more history help? |

### Phase 3 Additional Tests (Not Possible with CNN)

1. **Temporal coherence**: Cosine similarity between consecutive timepoint embeddings → are nearby timepoints similar?
2. **Delta-embedding prediction**: Does Δembedding predict Δtumor_volume? (R²)
3. **Outcome prediction**: Can baseline embedding predict 12-month status? (AUC)
4. **Temporal t-SNE**: Plot same patient's timepoints as trajectory → does it show progression?

These tests are the "value-add" of ViT over CNN. CNN temporal results provide the baseline:
- T1 (r=0.049), T3 (R²=-0.210), T4 (AUC=0.458) — all failing → clear improvement targets for ViT.
- T5 (cos=0.995) — too high coherence means CNN sees all timepoints as identical → ViT should differentiate.

---

## Deliverables

| # | Deliverable | Format | Used In |
| - | ----------- | ------ | ------- |
| 1 | `data_splits.json` (3-fold + 5-fold) | JSON | Phase 2 AND Phase 3 |
| 2 | 3D U-Net trained weights (3 folds) | `.pth` | Phase 2 evaluation |
| 3 | CNN embeddings for all 171 timepoints | `.npz` | Phase 3 comparison, Phase 4 baseline |
| 4 | Segmentation Dice table (mean ± std) | Markdown/notebook | Phase 2+3 comparison table |
| 5 | Embedding quality metrics | Markdown/notebook | Phase 2+3 comparison table |
| 6 | Phase 2 Kaggle notebook (reproducible) | `.ipynb` | Documentation + reproducibility |
| 7 | Limitation analysis document | Markdown | Motivates Phase 3 proposal |
| 8 | t-SNE and clustering figures | `.png` | Report, Phase 3 comparison |

---

## Timeline (Actual)

| Week | Planned | Actual | Status |
| ---- | ------- | ------ | ------ |
| 5 | Data splits, MONAI pipeline | Data splits, SegResNet + Met-Seg setup | ✅ Done |
| 5 | Quick test 15 epochs | Quick test both models | ✅ Done |
| 6 | Full 3-fold training (1 model) | SegResNet 3-fold + Met-Seg v1 3-fold + v2 3-fold | ✅ Done |
| 6-7 | Embedding extraction | Embeddings for SegResNet + Met-Seg v1 (all folds) | ✅ Done |
| 7 | Met-Seg v3/v4 training | 3-fold v3/v4, 60-80 epochs, AdamW + step LR | ✅ Done |
| 7 | v3 embedding extraction | 170 × 1024-dim, 3 folds | ✅ Done |
| 7 | Evaluation battery (16 tests) | Met-Seg 12/16, SegResNet 7/16 | ✅ Done |
| 7 | Report finalization | Phase2_Complete_Report.md | ✅ Done |

**Total GPU time used:** ~30 hrs across 3 Kaggle accounts  
**Evaluation:** ~2 hrs CPU (no GPU needed)  
**Phase 2: COMPLETE**
