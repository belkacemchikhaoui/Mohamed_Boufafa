# Phase 3 — Vision Transformer-Based Longitudinal Representation Learning

**Project:** Explainable Disease Progression and Counterfactual Video Generation  
**Program:** Mitacs Globalink — TELUQ University  
**Supervisor:** Dr. Belkacem Chikhaoui  
**Duration:** 4 weeks (Weeks 8–11)  
**Predecessor:** Phase 2 (COMPLETE — Met-Seg CNN baseline: Dice=0.505, 12/16 embedding tests)

---

## What Is Phase 3?

Phase 3 replaces the CNN encoder with a **Vision Transformer (Swin UNETR)** and adds **temporal modeling (TaViT)** to learn representations that capture how tumors evolve over time — the core capability that CNN lacked.

Phase 2 proved that CNN embeddings encode tumor anatomy (M1-M5 pass) and heterogeneity (H1-H4 pass) but **cannot model temporal change** (T1 r=0.049, T3 R²=-0.210, T4 AUC=0.458 — all failures). Phase 3 directly addresses these failures.

**The scientific comparison:**

| Aspect | Phase 2 (CNN) | Phase 3 (ViT) | What We're Testing |
|---|---|---|---|
| Architecture | Met-Seg DynUNet | Swin UNETR | Does global attention beat local convolution? |
| Parameters | 28M | 62M | Does scale help? |
| Receptive field | 3×3×3 local | Shifted windows → full volume | Does seeing more context help? |
| Embedding dim | 1024 | 768 | Does architecture matter more than dimensionality? |
| Temporal modeling | ❌ None | ✅ TaViT with time encoding | Does time-awareness help? |
| Input per prediction | Single scan | Sequence of scans | Does history help? |
| Pretraining | BraTS-METS 2023 | BraTS 2021 (5,050 CT + 1,251 MRI) | Domain-specific vs general pretraining? |

---

## Theoretical Foundation

### Swin UNETR (Hatamizadeh et al., 2022)

Swin UNETR uses a **hierarchical Swin Transformer encoder** with shifted-window self-attention connected to a CNN decoder via skip connections. Unlike standard ViTs that compute attention over the entire volume (quadratic cost), Swin partitions the 3D volume into non-overlapping windows and computes attention locally, then shifts windows in alternating layers to enable cross-window communication.

**Why this matters for us:** The encoder produces features at 5 resolution levels (48, 96, 192, 384, 768 channels), capturing both fine-grained details (enhancing tumor boundaries) and global context (whole-brain spatial relationships). The bottleneck features (4×4×4×768) are our ViT embeddings — each scan reduced to a 768-dim vector via global average pooling.

**Key advantage over CNN:** Swin UNETR's self-attention mechanism compares ALL spatial patches simultaneously, so it can detect "the left hemisphere has high enhancement while the right doesn't" — a capability fundamentally impossible with CNN's local 3×3×3 kernels. This should improve heterogeneity tests (H1-H4) and, importantly, capture more discriminative features for temporal comparison.

### TaViT — Temporal Emphasis ViT (Li et al., 2022)

TaViT solves the critical problem of **irregular time intervals** between clinical scans. Standard sequence models treat scan 1 → scan 2 → scan 3 as equally spaced, but in reality, Cyprus patients have follow-ups at 6 weeks, 3 months, 6 months, 9 months, and 12 months — highly irregular.

TaViT introduces a **Temporal Emphasis Model (TEM)** — a learnable sigmoid function that scales self-attention weights based on the temporal distance between two scans:

| Component | Description |
|---|---|
| Input | Time gap (in days) between scan i and scan j |
| Function | f(R) = 1 / (1 + exp(a × R - c)) |
| Parameters | a = steepness (how fast emphasis drops), c = shift (when it drops) |
| Effect | Recent scan pairs get high attention, distant pairs get reduced weight |
| Key result | Without time encoding, positional ViT drops to **random chance (AUC=0.50)** on irregular data |

**Why TaViT over TeViT:** Both handle irregular intervals, but TaViT's learnable emphasis is better for clinical data where recent scans are more predictive of current status. On real clinical data (NLST): TaViT AUC=0.786 vs TeViT AUC=0.785. The difference is small but TaViT's 2 learnable parameters (a, c) give interpretable temporal weighting.

---

## Controlled Variables (Same as Phase 2)

These MUST stay identical to ensure fair CNN vs ViT comparison:

| Item | Value | Why Shared |
|---|---|---|
| Data splits | `Phase2/outputs/data_splits.json` (3-fold) | Same patients in train/test |
| Input resolution | 96×96×96 | Both models see same input |
| Input channels | 4 (T1, T1c, T2, FLAIR) | Same modalities |
| Output classes | 4 (BG, NCR, ET, ED) | Same segmentation task |
| Loss function | DiceCELoss | Same training objective |
| Evaluation battery | Same 16 tests (M1-M6, H1-H4, T1-T7) | Same ruler |
| Ground truth features | Same radiomic features from Phase 1 | Same reference |
| Clinical metadata | Same `PROTEAS_Clinical_Cleaned.xlsx` | Same patient info |

---

## Activity 1: Swin UNETR Segmentation (Week 8)

### 1.1 Objective

Fine-tune Swin UNETR on Cyprus PROTEAS for 3D brain metastasis segmentation using the same 3-fold cross-validation splits as Phase 2. The primary goal is not to beat Met-Seg on segmentation (though we expect competitive performance), but to train the encoder to understand tumor anatomy so it produces high-quality 768-dim embeddings.

### 1.2 Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Model | MONAI `SwinUNETR` | Official implementation, pretrained weights available |
| Input | 4ch, 96×96×96 | Same as Phase 2 for fair comparison |
| Output | 4 classes (BG, NCR, ET, ED) | Same task as Phase 2 |
| Encoder | Swin Transformer (5 stages) | Hierarchical, shifted-window attention |
| Decoder | CNN with residual blocks | Skip connections from all encoder stages |
| Parameters | ~62M total | 2.2× larger than Met-Seg (28M) |
| Embedding dim | 768 (bottleneck) | From encoder Stage 5 |
| Pretrained | BraTS 2021 weights (`model_swinvit.pt`) | 5,050 CT + 1,251 MRI pretrained |

### 1.3 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Loss | DiceCELoss | Same as Phase 2 |
| Optimizer | AdamW, lr=1e-4, wd=1e-5 | Same as Met-Seg v3/v4 |
| LR schedule | Warmup (3 epochs) → Flat (27) → Step (15) → Refine (15) | Proven schedule from Phase 2 |
| Epochs | 60 | Same range as Met-Seg v3 |
| Early stopping | Patience=20 on validation Dice | Prevent overfitting |
| Batch size | 1 (96³ with Swin UNETR requires more VRAM) | 62M params need more memory |
| Mixed precision | AMP (fp16) | Essential for fitting Swin UNETR on T4 |
| Gradient checkpointing | Enabled (`use_checkpoint=True`) | Reduces VRAM at cost of ~20% speed |
| Augmentation | RandFlip, RandRotate90, RandScaleIntensity (±10%) | Same as Phase 2 |
| Inference | Sliding window (96³ patches, overlap=0.5) | Standard for Swin UNETR |

### 1.4 Kaggle Session Plan

| Session | Content | Est. Time | Account |
|---|---|---|---|
| S1 | Quick test: 15 epochs fold 0 → validate pipeline works | ~3 hrs | Account 1 |
| S2 | Train fold 0: 60 epochs with early stopping | ~8 hrs | Account 1 |
| S3 | Train fold 1: 60 epochs | ~8 hrs | Account 2 |
| S4 | Train fold 2: 60 epochs | ~8 hrs | Account 3 |

**GPU memory estimate:** Swin UNETR with 96³ input, batch=1, AMP+gradient checkpointing → ~8-10 GB VRAM (fits T4 15GB).

### 1.5 Expected Results

| Metric | Met-Seg (Phase 2) | Swin UNETR (Expected) | Reasoning |
|---|---|---|---|
| Mean Dice | 0.505 ± 0.024 | 0.45–0.55 | Swin UNETR has more parameters but our data is small (171 scans); pretraining helps but domain shift from glioma → metastasis may hurt |
| WT Dice | 0.519 | 0.48–0.56 | Global attention should help with large edema regions |
| ET Dice | 0.497 | 0.46–0.52 | Small enhancing regions may be harder with window-based attention |

**Important note:** Segmentation Dice is NOT the primary objective. Even if Swin UNETR gets slightly lower Dice than Met-Seg, the ViT embeddings may be superior for downstream evaluation tests — because the encoder architecture captures different (potentially richer) features.

### 1.6 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Swin UNETR training notebook (per fold) | `.ipynb` | `Phase3/notebooks/Phase3_A1_SwinUNETR_Training_Fold{0,1,2}.ipynb` |
| Trained weights (3 folds) | `.pth` | `Phase3/weights/swinunetr_fold{0,1,2}_best.pth` |
| Training logs + metrics | JSON | `Phase3/training_logs/` |
| Segmentation Dice table (mean ± std) | In report | `Phase3/reports/` |

---

## Activity 2: ViT Embedding Extraction + TaViT Temporal Modeling (Week 9)

### 2.1 Objective

Extract 768-dim ViT embeddings from the trained Swin UNETR encoder for all 171 scans across 3 folds, then implement TaViT temporal modeling to produce time-aware embeddings that capture tumor evolution.

### 2.2 ViT Embedding Extraction

**Method:** Pass each scan through the trained Swin UNETR. Hook into the encoder bottleneck (Stage 5 output: 4×4×4×768). Apply global average pooling → 768-dim vector per scan.

| Property | Value |
|---|---|
| Source layer | Swin UNETR encoder bottleneck (last stage) |
| Raw shape | (batch, 768, 4, 4, 4) |
| After GAP | (batch, 768) |
| Scans per fold | 171 (all scans, including 16 without masks) |
| Total embeddings | 171 × 3 folds |
| Storage | ~1 MB per fold (~3 MB total) |

### 2.3 TaViT Temporal Modeling

**Architecture:** A lightweight Transformer that takes a patient's sequence of ViT embeddings + time encodings and produces temporally-informed representations.

**Input Construction (per patient):**

| Component | Description |
|---|---|
| Scan embeddings | 768-dim vectors from Swin UNETR (2-6 per patient) |
| Time gaps | Days since baseline for each scan (from `cyprus_patient_timelines.csv`) |
| Time encoding | TEM sigmoid function applied to attention weights |
| CLS token | Learnable token prepended to sequence → captures full temporal summary |

**TaViT Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 768 (from Swin UNETR) | Match encoder output |
| Transformer layers | 4 | Smaller than paper's 8 (we have 45 patients, not 53K) |
| Attention heads | 8 | Standard for 768-dim |
| Dim per head | 96 | 768 / 8 = 96 |
| TEM parameters | a (steepness), c (shift) — learnable | 2 extra parameters |
| Sequence length | 2–6 tokens + 1 CLS = 3–7 total | Variable per patient |
| Output | CLS token → 768-dim temporal embedding | One vector per patient summarizing full trajectory |
| Training objective | Self-supervised masked reconstruction | Reconstruct masked timepoint embeddings |

**Why 4 layers (not 8):**  The original TaViT paper used 53,000 patients with 2–5 timepoints. We have 45 patients with 2–6 timepoints — dramatically less data. A 4-layer model with 768-dim input already has ~19M parameters. Overfitting is the primary risk, so we keep the model small and rely heavily on the pretrained Swin UNETR features.

### 2.4 Training Strategy

**Phase A — Self-supervised pretraining (no labels needed):**

| Setting | Value |
|---|---|
| Task | Masked timepoint reconstruction |
| Masking ratio | 0.5 (mask half of each patient's timepoints) |
| Objective | Reconstruct masked scan embeddings from unmasked ones + time encoding |
| Data | All 45 patients × all timepoints (no CV split needed — unsupervised) |
| Epochs | 200 |
| Optimizer | AdamW, lr=5e-4, cosine schedule |
| Purpose | Learn temporal relationships without labels |

**Phase B — Fine-tune for downstream tasks:**

| Setting | Value |
|---|---|
| Task 1 | Response prediction (volume decreased ≥20% at 6 months) |
| Task 2 | Volume change regression (predict fu1 volume from baseline) |
| Data | 3-fold CV using same splits as Phase 2 |
| Epochs | 50 |
| Optimizer | AdamW, lr=1e-4 |
| Head | Linear layer on CLS token → task prediction |

### 2.5 Embedding Types Produced

| Type | Dim | Description | Tests It Feeds |
|---|---|---|---|
| **ViT scan embedding** | 768 | Per-scan, from Swin UNETR encoder (no temporal info) | M1-M6, H1-H4, T1 (static comparison to CNN) |
| **ViT temporal embedding** | 768 | Per-patient CLS token from TaViT (temporal info included) | T1-T7 (temporal tests) |
| **CNN scan embedding** | 1024 | From Phase 2 Met-Seg (baseline comparison) | Same tests — direct comparison |

### 2.6 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Embedding extraction notebook | `.ipynb` | `Phase3/notebooks/Phase3_A2_Embedding_Extraction.ipynb` |
| TaViT training notebook | `.ipynb` | `Phase3/notebooks/Phase3_A2_TaViT_Training.ipynb` |
| ViT scan embeddings (3 folds) | `.npz` | `Phase3/embeddings/swinunetr_embeddings_fold{0,1,2}.npz` |
| TaViT temporal embeddings | `.npz` | `Phase3/embeddings/tavit_temporal_embeddings_fold{0,1,2}.npz` |

---

## Activity 3: Embedding Evaluation Battery (Week 10)

### 3.1 Objective

Run the **same 16-test evaluation battery** from Phase 2 on the new ViT embeddings, then produce a direct 4-column comparison table: Radiomic Features vs CNN (Met-Seg) vs ViT (Swin UNETR) vs ViT+TaViT.

### 3.2 Test Battery (Identical to Phase 2)

**Morphology tests M1-M6 (ViT scan embeddings):**

| Test | Description | Metric | CNN Baseline | Expected ViT |
|---|---|---|---|---|
| M1 | Volume prediction | R² | 0.379 | 0.5–0.7 (global attention captures full tumor extent) |
| M2 | Log-volume shape | R² | 0.388 | 0.5–0.7 |
| M3 | Surface-volume ratio | R² | 0.006 ❌ | 0.2–0.4 (ViT multi-scale features should capture boundaries) |
| M4 | Necrosis detection | F1 | 0.707 | 0.7–0.8 |
| M5 | Elongation proxy | R² | 0.387 | 0.4–0.6 |
| M6 | NN consistency | % | 26.8% | 30–40% |

**Heterogeneity tests H1-H4 (ViT scan embeddings):**

| Test | Description | Metric | CNN Baseline | Expected ViT |
|---|---|---|---|---|
| H1 | PCA structure | \|r\| | 0.796 | 0.8–0.9 (self-attention correlates all spatial regions) |
| H2 | Heterogeneity probe | R² | 0.386 | 0.5–0.7 |
| H3 | Subregion detection | F1 | 0.540 | 0.6–0.7 |
| H4 | Texture bundle | R² | 0.258 | 0.3–0.5 |

**Temporal tests T1-T7 (ViT scan embeddings AND TaViT temporal embeddings):**

| Test | Description | Metric | CNN Baseline | Expected ViT (scan) | Expected ViT+TaViT |
|---|---|---|---|---|---|
| T1 | Emb dist vs ΔVol | r | 0.049 ❌ | 0.2–0.4 | **0.4–0.6** |
| T3 | ΔEmb→ΔVol | R² | -0.210 ❌ | 0.0–0.2 | **0.2–0.4** |
| T4 | Response pred | AUC | 0.458 ❌ | 0.5–0.6 | **0.6–0.7** |
| T5 | Temporal coherence | cos | 0.995 | 0.90–0.95 | **0.85–0.92** |
| T6 | Velocity corr | r | 0.209 | 0.3–0.5 | **0.4–0.6** |
| T7 | Treatment sep | d | 15.201 | 10–20 | **15–25** |

**Key hypothesis:** ViT scan embeddings should improve morphology and heterogeneity tests (global attention). ViT+TaViT temporal embeddings should dramatically improve temporal tests T1, T3, T4 — the failures that justify Phase 3.

### 3.3 Evaluation Protocol

| Step | Input | Output |
|---|---|---|
| 1. PCA(30) on embeddings | 768-dim ViT or 1024-dim CNN | 30-dim reduced features |
| 2. Ridge/LogReg probe | PCA features + ground truth labels | R²/F1/AUC scores |
| 3. 5-fold inner CV | Probe models | Avoid overfitting the probe |
| 4. Repeat for each outer fold | 3 outer folds | Mean ± std across folds |

**Software:** Same `activity4_evaluation.py` framework from Phase 2, parametrized for different embedding inputs.

### 3.4 Comparison Tables to Produce

**Table 1 — Segmentation (3-Fold CV):**

| Model | Dice NCR | Dice ET | Dice ED | Mean Dice | HD95 (mm) |
|---|---|---|---|---|---|
| SegResNet (Phase 2) | — | — | — | 0.368 ± 0.044 | — |
| Met-Seg v3/v4 (Phase 2) | — | — | — | 0.505 ± 0.024 | — |
| **Swin UNETR (Phase 3)** | — | — | — | **—** | — |

**Table 2 — Embedding Quality (4-Column Comparison):**

| Test | Radiomic Features | CNN (Met-Seg 1024-dim) | ViT (Swin UNETR 768-dim) | ViT+TaViT (768-dim temporal) |
|---|---|---|---|---|
| M1: Volume (R²) | — | 0.379 | — | N/A |
| ... | ... | ... | ... | ... |
| T4: Response (AUC) | — | 0.458 | — | — |
| **Total passed** | — | **12/16** | **—/16** | **—/16** |

### 3.5 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| ViT evaluation notebook | `.ipynb` | `Phase3/notebooks/Phase3_A3_SwinUNETR_Embedding_Eval.ipynb` |
| TaViT evaluation notebook | `.ipynb` | `Phase3/notebooks/Phase3_A3_TaViT_Temporal_Eval.ipynb` |
| Evaluation results | JSON | `Phase3/outputs/activity3_results.json` |
| Comparison figures | PNG | `Phase3/outputs/activity3_figures/` |

---

## Activity 4: Visualization and Analysis (Week 11)

### 4.1 Objective

Produce the visualizations and analysis that demonstrate ViT superiority (or reveal limitations), generate attention maps for explainability (feeds Phase 4), and write the comprehensive comparison report.

### 4.2 Visualizations to Produce

| Figure | Description | Purpose |
|---|---|---|
| **t-SNE: CNN vs ViT** | Side-by-side t-SNE colored by histology, treatment, timepoint | Visual comparison of embedding space structure |
| **Temporal trajectories** | Per-patient paths in t-SNE space (CNN vs ViT vs TaViT) | Show temporal coherence improvement |
| **Attention maps** | Swin UNETR attention overlaid on MRI slices | Where does the model focus? Tumor regions? |
| **TEM curve** | Learned sigmoid: temporal emphasis vs time gap | What temporal decay did TaViT learn? |
| **Bar chart comparison** | R²/F1/AUC across all 16 tests, CNN vs ViT vs TaViT | Direct numerical comparison |
| **Training curves** | Dice vs epoch for all 3 folds (Swin UNETR) | Show convergence behavior |

### 4.3 Attention Map Extraction

Swin UNETR's self-attention weights can be extracted from any encoder stage. The attention maps show which spatial regions the model considers important for its prediction.

| Property | Value |
|---|---|
| Source | Swin UNETR encoder self-attention weights |
| Resolution | Stage 1 (64³, finest), Stage 4 (8³, coarsest) |
| Visualization | Overlay attention heatmap on original MRI slice |
| Expected finding | High attention on tumor boundaries and enhancing regions |
| Phase 4 link | Attention maps feed LLM: "The model focused on right frontal lobe enhancement" |

### 4.4 Limitation Analysis

| Expected Limitation | Why | Which Test | Phase 4/5 Solution |
|---|---|---|---|
| Small dataset (45 patients) | 62M params with 171 scans → overfitting risk | All tests may show high variance | Data augmentation, pretraining |
| Response prediction still weak | Only 29 patients with 6-month data | T4 AUC may stay < 0.7 | More clinical endpoints |
| Domain shift (glioma → metastasis) | Pretrained on glioma, fine-tuned on metastasis | Segmentation Dice may not beat Met-Seg | Transfer learning analysis |
| TaViT overfitting | 19M temporal params with 45 patients | Temporal tests may overfit | Self-supervised pretraining helps |

### 4.5 Report Structure

The Phase 3 Complete Report will follow the same format as Phase 2:

1. Executive Summary
2. Architecture Description (Swin UNETR + TaViT)
3. Segmentation Results (3-fold CV, comparison to Phase 2)
4. Embedding Evaluation (16-test battery, 4-column comparison table)
5. Temporal Modeling Results (TaViT analysis)
6. Attention Map Analysis (explainability preview)
7. CNN vs ViT: Comprehensive Comparison
8. Limitations and Discussion
9. Phase 4 Justification
10. References

### 4.6 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Visualization notebook | `.ipynb` | `Phase3/notebooks/Phase3_A4_Visualization_Analysis.ipynb` |
| Complete Phase 3 report | `.md` | `Phase3/reports/Phase3_Complete_Report.md` |
| Phase 3 status checklist | `.md` | `Phase3/reports/PHASE3_STATUS.md` |
| All figures | PNG | `Phase3/outputs/activity4_figures/` |
| Attention maps | PNG | `Phase3/outputs/attention_maps/` |

---

## Timeline

| Week | Activity | Sessions | Key Milestone |
|---|---|---|---|
| **Week 8** | A1: Swin UNETR segmentation | S1 (quick test) + S2-S4 (3 folds) | Trained ViT, segmentation Dice |
| **Week 9** | A2: Embedding extraction + TaViT | S5 (extraction) + S6 (TaViT training) | ViT + temporal embeddings |
| **Week 10** | A3: Evaluation battery | Local CPU (~2 hrs) | 4-column comparison table |
| **Week 11** | A4: Visualization + report | Local | Complete report, attention maps |

**Total Kaggle GPU time:** ~30 hrs (6 sessions across 3 accounts)  
**Total CPU time:** ~4 hrs (evaluation + visualization)

---

## Compute Requirements

### Swin UNETR GPU Memory

| Component | 96³ input, batch=1 |
|---|---|
| Model (62M params) | 248 MB |
| Feature maps (forward + backward) | ~4 GB |
| Gradient checkpointing savings | -2 GB |
| AMP (fp16) savings | -1 GB |
| **Total** | **~3-4 GB** |

Fits T4 (15 GB VRAM) with comfortable margin. Batch size > 1 may be possible but not necessary.

### TaViT GPU Memory

| Component | Value |
|---|---|
| Model (4 layers, 768-dim) | ~76 MB |
| Sequence (max 7 tokens × 768) | < 1 MB |
| **Total** | **< 1 GB** |

TaViT can train on CPU if needed (tiny model). Alternatively, train on Kaggle GPU in <1 hour.

---

## Resources Required

### Pretrained Weights

| Resource | URL | Size |
|---|---|---|
| Swin UNETR encoder weights | [model_swinvit.pt](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) | ~400 MB |
| TaViT reference code | [github.com/tom1193/time-distance-transformer](https://github.com/tom1193/time-distance-transformer) | Clone repo |

### Software Dependencies

| Library | Version | Use |
|---|---|---|
| MONAI | ≥1.3 | `SwinUNETR`, transforms, metrics |
| PyTorch | ≥2.0 | Backend |
| scikit-learn | ≥1.3 | Evaluation probes (Ridge, LogReg, PCA) |
| matplotlib/seaborn | Latest | Visualizations |

### From Phase 2 (Reused)

| Asset | Location | Use in Phase 3 |
|---|---|---|
| Data splits (3-fold + 5-fold) | `Phase2/outputs/data_splits.json` | **Same splits** for fair comparison |
| Met-Seg embeddings (1024-dim) | `Phase2/embeddings/metseg/` | CNN baseline for comparison |
| SegResNet embeddings (128-dim) | `Phase2/embeddings/segresnet/` | Second CNN baseline |
| Evaluation results | `Phase2/outputs/activity4_results.json` | Phase 2 numbers for comparison table |
| Clinical metadata | `Phase1/outputs/PROTEAS_Clinical_Cleaned.xlsx` | Ground truth for evaluation tests |
| Patient timelines | `Phase1/outputs/cyprus_patient_timelines.csv` | Time gaps for TaViT |
| Radiomic features | `Phase1/outputs/radiomic_features.csv` | Ground truth for M1-M5, H1-H4 |
| Tumor volumes | `Phase1/outputs/tumor_volumes.csv` | Ground truth for T1, T3 |

---

## Notebook Naming Convention

Following the Phase 2 pattern (`Phase2_A[X]_Name.ipynb`):

| Notebook | Description | GPU? |
|---|---|---|
| `Phase3_A1_SwinUNETR_Training_Fold0.ipynb` | Swin UNETR training fold 0 | ✅ Kaggle T4 |
| `Phase3_A1_SwinUNETR_Training_Fold1.ipynb` | Swin UNETR training fold 1 | ✅ Kaggle T4 |
| `Phase3_A1_SwinUNETR_Training_Fold2.ipynb` | Swin UNETR training fold 2 | ✅ Kaggle T4 |
| `Phase3_A2_Embedding_Extraction.ipynb` | Extract 768-dim embeddings from all scans | ✅ Kaggle T4 |
| `Phase3_A2_TaViT_Training.ipynb` | TaViT temporal model training | ✅ or CPU |
| `Phase3_A3_SwinUNETR_Embedding_Eval.ipynb` | 16-test battery on ViT embeddings | ❌ CPU only |
| `Phase3_A3_TaViT_Temporal_Eval.ipynb` | Temporal tests on TaViT embeddings | ❌ CPU only |
| `Phase3_A4_Visualization_Analysis.ipynb` | t-SNE, attention maps, comparison figures | ❌ CPU only |

---

## Success Criteria

### Minimum Success (Phase 3 delivers value)

| Criterion | Threshold |
|---|---|
| Swin UNETR trains successfully on Cyprus data | Dice > 0.35 (at least matches SegResNet) |
| ViT embeddings extracted for all 171 scans | 3 folds × 171 × 768-dim |
| ViT passes more embedding tests than CNN | >12/16 (beat Met-Seg's 12/16) |
| At least ONE temporal test improves | T1, T3, or T4 passes where CNN failed |
| 4-column comparison table complete | Radiomics vs CNN vs ViT vs ViT+TaViT |

### Target Success (Publication-quality)

| Criterion | Threshold |
|---|---|
| Swin UNETR segmentation competitive | Dice > 0.45 |
| ViT passes ≥14/16 tests | Including M3 (surface-volume ratio) which CNN failed |
| TaViT temporal tests improve significantly | T1 r > 0.3, T3 R² > 0.1, T4 AUC > 0.6 |
| Temporal coherence differentiates | T5 cos < 0.95 (CNN had 0.995 = no differentiation) |
| Attention maps highlight tumor regions | Qualitative validation by visual inspection |

### Stretch Goals

| Goal | Description |
|---|---|
| 5-fold CV | Run both CNN and ViT with 5-fold for publication (requires ~11 hrs extra GPU per model) |
| Masked autoencoder pretraining | Self-supervised pretraining on all 171 scans before fine-tuning |
| Cross-attention fusion | Combine CNN + ViT features using CAFNet-inspired cross-attention |

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Swin UNETR OOM on Kaggle T4 | Medium | High | Use gradient checkpointing + AMP; reduce to 64³ if needed |
| ViT overfits (62M params, 171 scans) | High | Medium | Strong pretraining, early stopping, light augmentation |
| TaViT overfits (45 patients) | High | High | Self-supervised pretraining, 4-layer model, heavy regularization |
| Segmentation worse than Met-Seg | Medium | Low | Embedding quality matters more than Dice |
| Temporal tests still fail | Low | High | Investigate: is the data too small for temporal modeling? |
| Kaggle GPU quota exceeded | Medium | Medium | Spread across 3 accounts, efficient session planning |

---

## References

| # | Reference | Use in Phase 3 |
|---|---|---|
| [1] | Hatamizadeh et al. (2022). *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumours.* BrainLes@MICCAI | **Primary architecture** — Swin Transformer encoder for segmentation and embeddings |
| [2] | Li et al. (2022). *Time-distance vision transformers in lung cancer diagnosis from longitudinal CT.* arXiv:2209.01676 | **Temporal modeling** — TaViT temporal emphasis model for irregular time intervals |
| [3] | Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021 | **Foundation** — shifted-window attention mechanism |
| [4] | Tang et al. (2022). *Self-supervised pre-training of swin transformers for 3D medical image analysis.* CVPR 2022 | **Pretraining strategy** — self-supervised on 5,050 CT volumes |
| [5] | Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021 | **ViT foundation** — patch embedding and self-attention for images |
| [6] | Sadegheih & Merhof (2024). *Met-Seg: Two-Stage Brain Metastasis Pipeline.* MICCAI PRIME | **CNN baseline** — Phase 2 architecture for comparison |
| [7] | Alain & Bengio (2017). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644 | **Evaluation methodology** — linear probing protocol |
| [8] | Chen et al. (2020). *SimCLR: A Simple Framework for Contrastive Learning.* ICML 2020 | **Evaluation protocol** — freeze encoder → linear head |
| [9] | Flouri et al. (2025). *Cyprus PROTEAS Dataset.* Zenodo DOI: 10.5281/zenodo.17253793 | **Primary dataset** |
| [10] | Vaswani et al. (2017). *Attention is All You Need.* NeurIPS 2017 | **Transformer foundation** — self-attention mechanism |
