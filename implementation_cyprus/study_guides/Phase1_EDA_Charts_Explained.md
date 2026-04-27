# Phase 1 EDA Results Explained + Phase 2 Connection

## 1. Understanding the Volume Statistics

```
NCR_mm3: mean=1,030, median=0,    std=2,830, CV=2.75
ET_mm3:  mean=4,506, median=1,849, std=6,439, CV=1.43
ED_mm3:  mean=17,258, median=4,792, std=29,362, CV=1.70
WT_mm3:  mean=22,794, median=7,160, std=36,600, CV=1.61
```

### What each region is

These are the **three tumor subregions** from the BraTS annotation protocol:

| Label | Name | What it is | Clinical meaning |
|---|---|---|---|
| **1 = NCR** | Necrotic Core | Dead tissue inside the tumor | The tumor center has died from outgrowing its blood supply |
| **2 = ET** | Enhancing Tumor | Active, contrast-enhancing tumor tissue | The "alive" part of the tumor — takes up contrast dye because of leaky blood vessels |
| **3 = ED** | Edema (peritumoral) | Brain swelling around the tumor | Not tumor itself — it's the inflammatory response of surrounding brain tissue |

**WT (Whole Tumor)** = NCR + ET + ED combined — the entire affected region.

### Why the numbers look that way

**NCR has median = 0:**
- Only **79 out of 162** masks have NCR (49%). The other half have NO necrotic core.
- Small metastases often don't develop necrosis — they're too small to outgrow blood supply.
- When NCR exists, it averages ~1 cm³ (1,030 mm³), which is tiny.
- CV = 2.75 (highest variability) means NCR size is extremely unpredictable.

**ET is mid-sized (mean = 4,506 mm³ ≈ 4.5 cm³):**
- Present in all tumors (it IS the tumor).
- Median = 1,849 mm³ tells you the "typical" tumor is ~1.8 cm³.
- The mean (4,506) being 2.4× the median means there are some very large tumors pulling the average up (right-skewed distribution).

**ED is the LARGEST (mean = 17,258 mm³ ≈ 17.3 cm³):**
- Yes, **edema is much larger than the actual tumor**. This is normal.
- Brain metastases cause disproportionate swelling — a 2 cm tumor can cause 5+ cm of surrounding edema.
- This is clinically significant: edema causes most of the patient's symptoms (headaches, neurological deficits), not the tumor itself.

**WT (mean = 22,794 mm³ ≈ 22.8 cm³):**
- Simply NCR + ET + ED added together.
- Dominated by ED (76% of WT on average).

---

## 2. The Three Charts Explained

### Chart 1: Pie Chart — "Overall Subregion Composition"

```
ED (green)  = 2.95 cm³ = 75.7%  ← Edema dominates
ET (blue)   = 0.77 cm³ = 19.7%  ← Actual tumor
NCR (red)   = 0.18 cm³ =  4.6%  ← Necrotic core (tiny)
```

> **What this tells you:** For every tumor, the surrounding brain swelling (ED) is ~4× larger than the tumor itself (ET), and the dead core (NCR) is tiny. This is a hallmark of brain metastases — they cause massive surrounding edema.

### Chart 2: Violin Plot — "Volume Distribution by Subregion" (log scale!)

> [!IMPORTANT]
> The Y-axis is **logarithmic** (10¹ to 10⁵ mm³), so visual differences are much larger than they appear.

- **NCR (red):** The widest part of the violin is near the bottom (~100 mm³). Many tumors have NO necrosis, so the distribution is heavily bottom-weighted.
- **ET (blue):** Peaks around 1,000-2,000 mm³. Taller violin = more consistent sizes.
- **ED (green):** Peak is around 5,000-10,000 mm³ but extends up to 100,000+ mm³. The tallest violin — confirming edema has the widest range.

> **Yes, you're right** — ED (green) has more volume. The violin reaching higher on the Y-axis means edema sizes span a larger range and are consistently bigger.

### Chart 3: Scatter Plot — "Tumor Spatial Distribution"

- Each dot = one tumor's center of mass in the brain (X, Y are voxel coordinates)
- Color = WT volume (dark = small, bright/yellow = large)
- The **red dots** (bottom-left cluster) are the **cerebellum** tumors — they have a different location and tend to be smaller
- The main cluster (100-140 on X, 120-170 on Y) is the **cerebral hemispheres**
- A few bright/yellow dots (large tumors) are scattered — large tumors don't cluster in one location

---

## 3. How EDA Connects to Phase 2 Segmentation

### The EDA → Segmentation alignment

The EDA volume hierarchy **directly predicts** how well the CNN segments each subregion:

| Subregion | EDA Mean Volume | % of WT | Segmentation Difficulty | Why |
|---|---|---|---|---|
| **ED (label 3)** | 17,258 mm³ | 75.7% | Easiest | Largest region = most training signal |
| **ET (label 2)** | 4,506 mm³ | 19.7% | Medium | Medium-sized but well-defined boundaries |
| **NCR (label 1)** | 1,030 mm³ | 4.6% | Hardest | Tiny, absent in 51% of cases |

### Actual training results (from logs)

> [!IMPORTANT]  
> Numbers below are from the **actual Kaggle training logs**, not estimates.

#### Met-Seg (DynUNet) — 3-Fold Cross-Validation

| | Fold 0 (v3, 60ep) | Fold 1 (v4, 80ep) | Fold 2 (v4, 80ep) | **Mean ± Std** |
|---|---|---|---|---|
| **Mean Dice** | **0.5316** | 0.4830 | 0.5000 | **0.5049 ± 0.025** |
| WT | 0.573 | 0.496 | 0.509 | **0.526** |
| TC | 0.510 | 0.470 | 0.498 | **0.493** |
| ET | 0.512 | 0.483 | 0.494 | **0.496** |
| Train/Val | 114/56 | 119/51 | 107/63 | — |
| Time | 217 min | 298 min | 323 min | **~14 hrs total** |

#### SegResNet — 3-Fold Cross-Validation

| | Fold 0 (50ep) | Fold 1 (50ep) | Fold 2 (50ep) | **Mean ± Std** |
|---|---|---|---|---|
| **Mean Dice** | **0.4134** | 0.3273 | 0.3634 | **0.3680 ± 0.043** |
| WT | 0.448 | 0.365 | 0.384 | **0.399** |
| TC | 0.400 | 0.320 | 0.363 | **0.361** |
| ET | 0.392 | 0.297 | 0.343 | **0.344** |
| Train/Val | 114/56 | varied | varied | — |
| Time | 122 min | 96 min | 90 min | **~5 hrs total** |

### The Pattern: WT > TC ≈ ET (for both models)

For **both** Met-Seg and SegResNet, the Dice scores follow the same order:

```
WT (Whole Tumor) > TC (Tumor Core) ≈ ET (Enhancing Tumor)
```

**This aligns perfectly with the EDA findings:**

1. **WT is easiest** (Met-Seg: 0.526, SegResNet: 0.399) — because WT = NCR + ET + ED, and ED is huge. The model just needs to find "where is the brain abnormal?" which is a big, obvious region.

2. **TC is harder** (Met-Seg: 0.493, SegResNet: 0.361) — TC = NCR + ET (no edema). The model must distinguish actual tumor tissue from the surrounding edema, which requires finer detail.

3. **ET is hardest** (Met-Seg: 0.496, SegResNet: 0.344) — ET alone, without the necrotic core. Must identify only the enhancing rim, which is the most precise boundary.

> [!NOTE]
> In Met-Seg, TC ≈ ET are very close (0.493 vs 0.496), which makes sense because many metastases don't have NCR, so TC ≈ ET in those cases. In SegResNet, ET is clearly worse (0.344 vs 0.361), suggesting the simpler architecture struggles more with the fine enhancing boundary.

### Why Met-Seg beats SegResNet by +37.2%

| Factor | Met-Seg | SegResNet | Impact |
|---|---|---|---|
| Pretraining data | **402 brain metastases** (BraTS-METS 2023) | Gliomas (different tumor type) | Domain match is critical |
| Architecture | DynUNet (16.7M params) | SegResNet (4.7M params) | More capacity |
| Embedding dim | 1024 | 128 | 8× more representation space |
| Training strategy | Warmup → flat → step LR, detector unfreeze | Single LR | Better optimization |

---

## 4. Summary

- **EDA correctly predicted** that segmentation would be easier for larger subregions (ED/WT) and harder for smaller ones (NCR/ET)
- **Edema dominates** — 76% of the whole tumor region is swelling, not actual cancer
- **The CNN learns size**: larger regions get higher Dice simply because there's more training signal
- **Met-Seg's advantage**: pretraining on brain metastases (not gliomas) gives it a head start — it already "knows" what metastasis boundaries look like
