# TransXAI: Explainable Hybrid Vision Transformers (2024) - YOUR EXPLAINABILITY BLUEPRINT!

## 1. ONE-SENTENCE SUMMARY
TransXAI is a **hybrid CNN-Transformer** that segments brain gliomas from multimodal MRI AND generates **surgeon-understandable heatmaps** using Grad-CAM — showing WHICH MRI modality matters for WHICH tumor region and HOW the network processes information layer by layer — **exactly the explainability you need for Phase 3!**

---

## 2. KEY RESULTS (The Numbers That Matter!)

### **BraTS 2019 Validation Set (125 cases, 5-fold ensemble)**

| Metric | ET | TC | WT | **Average** |
|--------|----|----|-----|-------------|
| **Dice (DSC)** | 0.745 | 0.782 | 0.882 | **0.803** |
| **Hausdorff 95% (mm)** | 4.31 | 7.90 | 6.36 | **6.19** |

### **Comparison to State-of-the-Art (BraTS 2019)**

| Method | ET Dice | TC Dice | WT Dice | Avg Dice | Avg HD95 |
|--------|---------|---------|---------|----------|----------|
| KiU-Net | 0.664 | 0.706 | 0.861 | 0.744 | 11.75 |
| Res U-Net | 0.667 | 0.706 | 0.853 | 0.742 | 8.46 |
| 3D U-Net | 0.709 | 0.725 | 0.874 | 0.769 | 7.74 |
| V-Net | 0.739 | 0.766 | 0.887 | 0.797 | 7.03 |
| Attention U-Net | 0.760 | 0.772 | 0.888 | **0.807** | 7.07 |
| **TransXAI (Ours)** | **0.745** | **0.782** | 0.882 | **0.803** | **6.19** ✅ Best HD95 |

> **Key finding**: TransXAI achieves **best-in-class boundary precision** (lowest HD95 across all sub-regions!) while maintaining competitive Dice scores. The HD95 advantage matters clinically — precise tumor boundaries = safer surgery!

### **External Multi-Site Validation (FeTS 2022 / BraTS 2021)**
Tested on 974 cases from 6 different institutions (multi-site!):

| Institute | # Cases | BraTS 2019 Overlap | ET Dice | TC Dice | WT Dice | Avg Dice |
|-----------|---------|--------------------|---------|---------|---------|----------|
| 1 | 511 | 129 | 0.710 | 0.891 | 0.739 | 0.780 |
| 2 | 6 | None | 0.638 | 0.913 | 0.646 | 0.732 |
| 18 | 382 | None | 0.687 | 0.854 | 0.718 | 0.753 |
| 20 | 33 | None | 0.759 | 0.946 | 0.784 | 0.830 |
| 21 | 35 | None | 0.804 | 0.931 | 0.800 | **0.845** |
| 22 | 7 | None | 0.778 | 0.897 | 0.823 | 0.833 |

> **Multi-site generalization proven!** Works across 6 different institutions with different scanners and protocols. Relevant for Yale's multi-scanner, multi-year data!

### **Cross-Validation Consistency (5-fold)**

| Fold | ET Dice | TC Dice | WT Dice |
|------|---------|---------|---------|
| 0 | 0.725 | 0.767 | 0.883 |
| 1 | 0.730 | 0.770 | 0.869 |
| 2 | 0.720 | 0.777 | 0.876 |
| 3 | 0.746 | 0.758 | 0.868 |
| 4 | 0.734 | 0.756 | 0.874 |

> Consistent across folds → robust model, not dependent on data splits.

---

## 3. WHAT'S NEW? (Three Innovations!)

### 🔍 Innovation 1: Hybrid CNN-Transformer for Glioma Segmentation

**Architecture (Fig. 6 from paper)**:
```
INPUT: 192 × 192 × 4 (2D axial slice: T1, T1Gd, T2, FLAIR)
    ↓
┌─── CNN ENCODER (local feature extraction) ───┐
│  8 convolutional blocks:                      │
│  Each: 2 × (3×3 Conv → Batch Norm → ReLU)    │
│  + 2×2 Max Pooling between blocks             │
│  K=32 filters, downsamples to H/8 × W/8 × C  │
│  → Captures LOCAL textures, edges, boundaries │
└──────────────────────────────────────────────┘
    ↓
┌─── TRANSFORMER BOTTLENECK (global context) ──┐
│  ViT blocks on CNN features (not raw input!)  │
│  Patch embedding from CNN output              │
│  Multi-Head Self-Attention (MSA)              │
│  + MLP + Layer Norm + Residual connections    │
│  → Captures LONG-RANGE spatial dependencies   │
└──────────────────────────────────────────────┘
    ↓
┌─── FEATURE RESTORATION ──────────────────────┐
│  1×1 Conv to reduce feature maps              │
│  Reshape back to spatial dimensions           │
└──────────────────────────────────────────────┘
    ↓
┌─── CNN DECODER (upsampling path) ────────────┐
│  2×2 Up-convolutions                          │
│  Skip connections from encoder (like U-Net!)  │
│  Progressive upsampling to full resolution    │
│  Final: multi-label softmax                   │
└──────────────────────────────────────────────┘
    ↓
OUTPUT: 192 × 192 × 3 (ET, TC, WT predictions)
```

**Why hybrid?**
- CNN alone: Good at local features (edges, textures) but misses global context
- Transformer alone: Good at global relationships but needs HUGE datasets (14M-300M images!)
- **Hybrid**: CNN extracts local features → Transformer connects them globally → Best of both worlds on SMALL medical datasets!

### 🗺️ Innovation 2: Post-Hoc Grad-CAM Explainability (Zero Accuracy Cost!)

**The big advantage**: Explainability is added AFTER training — no architecture modifications, no accuracy tradeoff!

**How Grad-CAM works (simple)**:
```
1. Run the segmentation model (forward pass) → get prediction
2. Pick a TARGET class (e.g., "Enhancing Tumor")
3. Compute gradients flowing BACK through the network
4. Weight each feature map by its average gradient
5. Apply ReLU → get HEATMAP showing "which regions matter most"

Math:
  α_k = (1/N) Σ ∂y_c / ∂A_k     ← average gradient for feature map k
  L_GradCAM = ReLU( Σ α_k × A_k )  ← weighted sum → heatmap
```

**Three types of explanations generated**:

#### A) MRI Modality Contribution Maps
Feed EACH MRI modality separately → see which modality detects which tumor region:
```
T1    → Very little contribution (can potentially be REMOVED to save compute!)
T1Gd  → Detects Enhancing Tumor (ET) + Tumor Core (TC)     ← contrast agent highlights active tumor
T2    → Detects Tumor Core (TC) + some Whole Tumor          ← shows tissue structure
FLAIR → Detects Peritumoral Edema + Whole Tumor (WT)        ← best for edema boundaries
```

> 💡 **FOR YALE**: This tells you WHICH MRI sequences matter most. If a patient is missing T1, the model still works! If T1Gd is available, ET detection is strongest.

#### B) Internal Layer Saliency Maps
Visualize what EACH layer of the encoder/decoder learns:

**Implicit concepts** (NOT in training labels!):
- Encoder Block 1: Differentiates white matter vs gray matter
- Deeper layers: Brain tissue boundaries

**Explicit concepts** (trained labels):
- Decoder Block 4: ET (enhancing tumor) boundaries
- Decoder Block 5: NC (necrotic core) + WT (whole tumor)

> 🧠 **Key finding**: The model follows a **top-down approach** — first learns global brain structure, then narrows to tumor boundaries, then fine details. This matches how surgeons analyze MRI scans!

#### C) Clinical Validation of Explainability
- **Two neurosurgeons** (7 and 10 years experience) evaluated the heatmaps
- Found them **"consistent with clinical knowledge"**
- Heatmaps matched how surgeons actually identify tumor boundaries
- **Increased their trust** in the AI system

### 📐 Innovation 3: 2D Slice-Based Processing (Practical Efficiency)

**Choice**: Processes 2D axial slices (not full 3D volumes)
- **Pro**: Runs on single GPU (RTX 2080 Ti / RTX 3060 — consumer hardware!)
- **Pro**: 5 days training time for 5-fold cross-validation
- **Con**: Loses some 3D spatial context (acknowledged by authors as future work)
- **For Yale**: Can start with 2D for explainability, use Swin UNETR (Paper 13) for 3D segmentation

---

## 4. IMPLEMENTATION DETAILS

| Setting | Value |
|---------|-------|
| **Input** | 192 × 192 × 4 (2D axial, T1 + T1Gd + T2 + FLAIR) |
| **Output** | 192 × 192 × 3 (ET, TC, WT) |
| **Framework** | TensorFlow |
| **Optimizer** | SGD, momentum 0.9 |
| **Learning rate** | 8e-3 |
| **Epochs** | 250 per fold |
| **Batch size** | 16 |
| **GPU** | Single NVIDIA RTX 2080 Ti (11GB) or RTX 3060 (12GB) |
| **Training time** | 5 days total (5-fold) |
| **Loss** | Generalized Dice + Categorical Cross-Entropy |
| **Ensemble** | STAPLE (5 folds combined) |
| **XAI method** | Grad-CAM via NeuroXAI framework |
| **Code** | https://github.com/razeineldin/TransXAI |

**Data Augmentation**:
- Horizontal/Vertical shift and flip
- Random rotation (0-20°)
- Adaptive zoom (up to 20%)
- Random brightness (±20%)
- Gaussian noise (σ=0.01)

**Loss Function**:
```
L_total = L_GeneralizedDice + L_CategoricalCrossEntropy

L_GD handles class imbalance (adaptive weights per class)
L_CE provides standard classification loss
Combined → robust to unbalanced tumor regions (WT >> ET in volume)
```

---

## 5. LIMITATIONS & WHAT'S MISSING

### ⚠️ Acknowledged Limitations:
1. **2D only** — Processes axial slices, misses coronal/sagittal 3D context
   - For Yale: Use Swin UNETR (Paper 13) for 3D segmentation; use TransXAI's explainability approach on the 3D model
   
2. **Lower Dice than Swin UNETR** — 0.803 avg vs Swin UNETR's 0.913 avg
   - But: TransXAI is on BraTS 2019 (335 training), Swin UNETR on BraTS 2021 (1,251 training) — different datasets!
   - The VALUE of TransXAI is EXPLAINABILITY, not raw segmentation numbers

3. **BraTS 2019 dataset** — Smaller and older than BraTS 2021
   - FeTS 2022 external validation partially addresses this

4. **No temporal component** — Single-timepoint segmentation only
   - For Yale: Combine with Paper 11's temporal attention for longitudinal tracking

### 🔑 What We TAKE from This Paper (Not the Model Itself!):
The primary value is the **EXPLAINABILITY METHODOLOGY**, not the segmentation architecture:
1. **Grad-CAM on decoder layers** → Apply to Swin UNETR's decoder
2. **MRI modality contribution analysis** → Understand which sequences Yale patients need
3. **Internal layer visualization** → Verify Swin UNETR learns meaningful features
4. **Clinical validation approach** → How to get surgeons to evaluate your explanations
5. **NeuroXAI framework** → Ready-to-use explainability tools

---

## 6. FOR YOUR YALE PROJECT (Concrete Implementation!)

### 🏗️ How TransXAI Fits Your Pipeline (Phase 3 — Explainability!):

```
PHASE 2 OUTPUT: Swin UNETR encoder embeddings + TaViT temporal model
    ↓
PHASE 3 STEP 1: Apply Grad-CAM to Swin UNETR
    → Which REGIONS of the brain scan activate for each tumor sub-region?
    → Which MRI MODALITY matters most for each prediction?
    → Generate heatmaps overlaid on patient MRI
    ↓
PHASE 3 STEP 2: Apply attention visualization to TaViT
    → Which TIMEPOINTS does the model focus on?
    → TaViT's temporal emphasis weights = natural explainability!
    → Show: "Model focused 95% on last 2 scans, 5% on historical"
    ↓
PHASE 3 STEP 3: Feed explanations to LLM
    → Heatmap: "Enhancing tumor detected in right frontal lobe"
    → Temporal: "Tumor grew 20% over 6 months, then stabilized"
    → MRI info: "T1Gd contrast best shows active enhancement"
    → LLM combines → generates radiology-style report
    ↓
OUTPUT: "Patient 67yo female with lung cancer metastasis. 
         Swin UNETR detected 15mm enhancing lesion in right 
         frontal lobe (T1Gd primary contributor, confidence 92%).
         Temporal analysis shows 20% growth over 6 months 
         (T0→T1), stabilization after radiosurgery (T1→T2).
         Attention focused on post-treatment scans (TEM weight 0.94).
         Recommend: continued surveillance, next scan in 3 months."
```

### 📋 Explainability Tools to Reuse:

| Tool | Source | Use for Yale |
|------|--------|-------------|
| **Grad-CAM** | TransXAI / NeuroXAI framework | Spatial attention heatmaps on Swin UNETR |
| **Modality analysis** | TransXAI method (infer per-modality) | Determine T1/T1c/T2/FLAIR contributions |
| **Layer visualization** | TransXAI method (saliency per layer) | Verify encoder learns meaningful features |
| **Clinical evaluation** | TransXAI protocol (surgeon interviews) | Validate your LLM-generated reports |
| **NeuroXAI framework** | https://github.com/razeineldin/NeuroXAI | Ready-to-use XAI for medical imaging |

### ⚡ Explainability Implementation Sketch:
```python
# Apply Grad-CAM to Swin UNETR (adapted from TransXAI approach)
import torch
from monai.networks.nets import SwinUNETR

# 1. Load fine-tuned Swin UNETR
model = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3)
model.load_state_dict(torch.load("swin_unetr_yale.pt"))

# 2. Forward pass
scan = load_yale_scan(patient_id, timepoint)  # (1, 4, 128, 128, 128)
prediction = model(scan)

# 3. Grad-CAM on decoder output layer
target_class = 2  # e.g., Enhancing Tumor
prediction[:, target_class].backward()  # backpropagate gradients

# 4. Get last decoder layer activations + gradients
activations = model.decoder_hooks[-1]  # hooked feature maps
gradients = model.gradient_hooks[-1]    # corresponding gradients
weights = gradients.mean(dim=(2,3,4))   # global average pooling
heatmap = torch.relu((weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                      * activations).sum(dim=1))

# 5. Overlay heatmap on MRI for visualization
# → Shows WHERE the model focuses for each tumor sub-region

# 6. Per-modality analysis (TransXAI approach)
for i, modality in enumerate(["T1", "T1c", "T2", "FLAIR"]):
    single_modality = scan.clone()
    single_modality[:, [j for j in range(4) if j != i]] = 0  # zero others
    pred = model(single_modality)
    # Compare → which modality contributes most to each sub-region
```

---

## 7. CONNECTION TO OTHER PAPERS

| Paper | Connection |
|-------|-----------|
| **Paper 13 (Swin UNETR)** | Apply TransXAI's Grad-CAM approach TO Swin UNETR's decoder layers |
| **Paper 11 (Time-distance ViT)** | TaViT's temporal emphasis weights = built-in temporal explainability |
| **Paper 2 (nnU-Net)** | TransXAI's explainability can also be applied to nnU-Net as baseline comparison |
| **Phase 3 (LLM)** | Heatmaps + attention weights → structured input for LLM report generation |
| **CAFNet** | Confirms hybrid CNN+ViT > pure approaches; cross-attention adds interpretability |

---

## 8. BOTTOM LINE

### ✅ Why This Paper Matters (⭐⭐ — Explainability Blueprint):
1. **Explainability WITHOUT accuracy loss** — post-hoc Grad-CAM, no architecture changes needed
2. **Three levels of explanation**: MRI modality contribution → internal layer features → spatial heatmaps
3. **Clinically validated** — neurosurgeons confirmed heatmaps match clinical knowledge
4. **Key MRI finding**: T1Gd → ET detection, FLAIR → edema/WT, T1 → minimal (can skip!)
5. **Code available**: https://github.com/razeineldin/TransXAI + NeuroXAI framework
6. **Multi-site validated** — Works across 6 institutions (FeTS 2022)
7. **Runs on consumer GPU** — RTX 2080 Ti sufficient (no DGX needed for explainability!)

### ⚠️ NOT Used For:
- **NOT your segmentation model** — Swin UNETR (Paper 13) is better for that (3D, higher Dice)
- **NOT temporal** — Single timepoint only, no progression tracking
- **Value is the METHODOLOGY** — how to make ANY model explainable

### 🎯 One-Line Takeaway:
> **Apply TransXAI's Grad-CAM explainability methodology to your Swin UNETR + TaViT pipeline to generate surgeon-understandable heatmaps showing WHERE tumors are detected, WHICH MRI modalities contribute, and HOW the model processes temporal progression — feeding all of this into the LLM for Phase 3 report generation.**

---

*Paper: "Explainable hybrid vision transformers and convolutional network for multimodal glioma segmentation in brain MRI"*  
*Authors: Ramy A. Zeineldin et al., FAU Erlangen-Nürnberg / Ulm University*  
*Published: Scientific Reports (2024), Nature*  
*DOI: https://doi.org/10.1038/s41598-024-54186-7*  
*Code: https://github.com/razeineldin/TransXAI*
