# Papers Tracker - Updated

## ✅ Papers Analyzed (10 total - PHASE 1 COMPLETE! 🎉)

### Paper 1: BraTS Toolkit (2020)
- **What**: Preprocessing tool for brain tumor scans
- **Key numbers**: 84.9-87.1% accuracy, tested on 219 patients
- **For Yale**: Use their preprocessing methods to clean Yale scans
- **Full analysis**: `SIMPLE_Paper1_BraTS_Toolkit.md`

### Paper 2: nnU-Net (2021)
- **What**: Self-configuring tumor segmentation (your baseline)
- **Key numbers**: 67.7-95.2% accuracy across 10 tasks, 1st place winner
- **For Yale**: Segment tumors in Yale scans (no labels provided)
- **Full analysis**: `SIMPLE_Paper2_nnUNet.md`

### Paper 3: Yale Brain Metastases Dataset (2025) ⭐ YOUR MAIN DATASET
- **What**: 11,884 longitudinal brain MRI scans, 1,430 patients
- **Key numbers**: Avg 8 scans per patient, 2004-2023, FREE download
- **For Yale**: THIS IS YALE! Your entire project dataset
- **Full analysis**: `SIMPLE_Paper3_Yale_Dataset.md`

### Paper 4: Treatment-Aware Longitudinal Registration (2024) 🔑 CRITICAL!
- **What**: Aligns scans over time while preserving tumor size changes
- **Key numbers**: 94.7% alignment accuracy, 5.35mm error, 11% volume preservation
- **For Yale**: Align 8 scans per patient, preserve tumor measurements
- **Full analysis**: `SIMPLE_Paper4_Registration.md`

### Paper 5: FLIRE - Fast Longitudinal Registration (2024) ❌ SKIP: Code Unavailable
- **What**: Fast breast MRI registration algorithm (MATLAB, not released)
- **Why skip**: Code NOT public, only breast MRI (no brain), MATLAB-only (no Python integration)
- **Alternative**: Use **itk-elastix** instead (pip install, brain-proven, Python native)
- **Full analysis**: `SIMPLE_Paper5_FLIRE.md`

### Paper 6: **itk-elastix** - Medical Image Registration in Python (2023) ✅ PRIMARY METHOD!
- **What**: Mature, pip-installable Python package for brain MRI registration
- **Key numbers**: 0.95 correlation (Yale brain), 14.5 min/scan, 100+ brain papers, 15 years mature
- **For Yale**: THE registration method! Multi-parametric (T1/T1c/T2/FLAIR), MONAI integration, ready brain parameter files
- **Full analysis**: `SIMPLE_Paper_ITK_Elastix.md`

### Paper 7: Cyprus Brain Metastases Dataset (2025) 🤝 VALIDATION PARTNER!
- **What**: 744 MRI scans from 40 patients with expert-verified tumor labels
- **Key numbers**: 18.6 scans/patient, 65 tumors with 3 subregions labeled, FREE download
- **For Yale**: VALIDATION dataset! Test Yale-trained models on Cyprus for generalization
- **Full analysis**: `SIMPLE_Paper7_Cyprus_Dataset.md`

### Paper 9: ComBat Harmonization (2022) 🔧 BASELINE HARMONIZATION!
- **What**: Math-based method to fix scanner differences (works on already-extracted features)
- **Key numbers**: 41% studies improved, 0% made worse, needs 20-30 patients/scanner minimum
- **For Yale**: CRITICAL! 20 years of scanners (2004-2023) need harmonization to be comparable
- **Full analysis**: `SIMPLE_Paper9_ComBat_Harmonization.md`

### Paper 10: Longitudinal ComBat (2020) ⏱️ TEMPORAL HARMONIZATION!
- **What**: ComBat adapted for repeated measures (same person scanned multiple times)
- **Key numbers**: 663 ADNI patients, 126 scanners, more powerful than cross-sectional ComBat
- **For Yale**: Track tumor changes across scanner switches (patient might see 3 scanners over 8 visits)
- **Full analysis**: `SIMPLE_Paper10_Longitudinal_ComBat.md`

### Paper 10++: ComBat Validation Study (2022) ✅ PROOF IT WORKS!
- **What**: First validation using traveling subjects (same 73 people on 4 scanners)
- **Key numbers**: Diffusion data ✅ harmonized, Structural data ⚠️ can be harmed if no scanner effect!
- **For Yale**: WARNING → Test for scanner effects BEFORE harmonizing! Not all data needs it!
- **Full analysis**: `SIMPLE_Paper10++_ComBat_Validation.md`

### Paper 10+: Generalized ComBat (2022) 🎯 ADVANCED HARMONIZATION!
- **What**: Nested ComBat (multiple batch effects) + GMM ComBat (unknown effects + bimodal data)
- **Key numbers**: Handles 3+ batch effects, discovers hidden confounds, 10-11% better than standard
- **For Yale**: Yale has 5+ batch effects (year/site/field strength/manufacturer/protocol) → use Nested!
- **Full analysis**: `SIMPLE_Paper10+_Generalized_ComBat.md`
- **Full analysis**: `SIMPLE_Paper9_ComBat_Harmonization.md`

---

## 📚 Phase 2 Papers: Vision Transformers (6 papers available - IN PROGRESS)

### ⭐⭐⭐ PRIORITY 1: Paper 11 — Time-distance ViT (2022) — TEMPORAL BLUEPRINT! ✅ ANALYZED
- **What**: ViT that encodes TIME INTERVALS between scans into self-attention
- **Key numbers**: AUC 0.786 (TaViT best) vs 0.734 (single-scan CNN) → +5% from temporal! Without time encoding → 0.50 AUC on irregular data (RANDOM CHANCE!)
- **For Yale**: Shows HOW to encode T0→T1 (6 months) vs T1→T2 (12 months) into ViT. Yale's irregular clinical intervals = EXACTLY the scenario where this shines!
- **Two methods**: TeViT (sinusoidal time encoding added to tokens) + TaViT (learnable sigmoid scales attention weights) → Use TaViT for Yale
- **Architecture**: D=64, 8 heads, 8 layers, masked autoencoder pretraining (75% masking)
- **Code**: https://github.com/tom1193/time-distance-transformer ✅ PUBLIC
- **arXiv**: https://arxiv.org/abs/2209.01676
- **Full analysis**: `objectivetwo/SIMPLE_Paper11_Time_Distance_ViT.md` ✅

### ⭐⭐⭐ PRIORITY 2: Paper 13 — Swin UNETR (2022) — SEGMENTATION + EMBEDDINGS! ✅ ANALYZED
- **What**: Swin Transformer for 3D brain tumor segmentation (DUAL USE: segmentation + feature extraction)
- **Key numbers**: Dice 0.913 avg (beats nnU-Net 0.908!), 61.98M params, bottleneck = 768-dim embeddings
- **For Yale**: (1) Upgrade segmentation from nnU-Net, (2) Extract 768-dim embeddings from encoder → feed to TaViT
- **Architecture**: 4-stage encoder [48,96,192,384,768], window 7×7×7, heads [3,6,12,24], CNN decoder with residual blocks + skip connections
- **Input**: 4-channel MRI (T1, T1c, T2, FLAIR) — SAME AS YALE!
- **Code**: MONAI library ✅ PUBLIC — `pip install monai` → `SwinUNETR()`
- **Weights**: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
- **arXiv**: https://arxiv.org/abs/2201.01266
- **Full analysis**: `objectivetwo/SIMPLE_Paper13_Swin_UNETR.md` ✅

### ⭐⭐ PRIORITY 3: TransXAI — Explainable Hybrid Transformer (2024) ✅ ANALYZED
- **What**: Hybrid CNN-Transformer for glioma segmentation WITH post-hoc Grad-CAM explainability
- **Key numbers**: Dice 0.803 avg (BraTS 2019), HD95 6.19mm (BEST boundary precision!), validated on 6 institutions (FeTS 2022)
- **For Yale**: EXPLAINABILITY BLUEPRINT for Phase 3! Apply Grad-CAM to Swin UNETR decoder, per-modality analysis (T1Gd→ET, FLAIR→WT, T1→skip), internal layer visualization, clinical validation protocol
- **Key finding**: T1 MRI contributes minimally — could be skipped for compute savings!
- **Key finding**: Model follows top-down approach (global brain → tumor boundaries → fine detail) — matches clinical reasoning!
- **Code**: https://github.com/razeineldin/TransXAI + NeuroXAI framework
- **Full analysis**: `objectivetwo/SIMPLE_Paper_TransXAI.md` ✅

### ⭐ PRIORITY 4: CAFNet — CNN+ViT Cross-Attention Fusion (2025) ✅ ANALYZED
- **What**: Hybrid CNN (MobileNetV2) + ViT with Cross-Attention Fusion for brain tumor classification
- **Key numbers**: Pure ViT 87.34% → CNN+ViT concat 92.20% → CAFNet (with CAF) **96.41%** (+9% over ViT!)
- **Ablation gold**: CNN alone 86.96%, ViT alone 87.34%, concat 92.20%, cross-attention 96.41% — fusion METHOD matters!
- **For Yale**: VALIDATES Swin UNETR hybrid design (Transformer encoder + CNN decoder). Cross-attention could enhance multi-scale skip connections or temporal model.
- **Thesis argument**: Direct citation for "why hybrid architecture?" in methodology section
- **Code**: ❌ Not public (but cross-attention is standard, easy to implement)
- **Full analysis**: `objectivetwo/SIMPLE_Paper_CAFNet.md` ✅

### ⭐ PRIORITY 5: ResAttU-Net-Swin (2025)
- **What**: Residual U-Net + Attention + Swin Transformer hybrid
- **Key numbers**: Dice 89.2% — lower than Swin UNETR (90.05%)
- **For Yale**: Backup if Swin UNETR underperforms
- **Code**: ❌ Not public
- **Status**: ⏸️ SKIP (Swin UNETR is better + has code)

### ⭐ PRIORITY 6: BRAIN-META — CNN-ViT Ensemble (2025)
- **What**: 10-model ensemble with XGBoost meta-learner for tumor classification
- **Key numbers**: 97.10% accuracy on tumor type classification
- **For Yale**: Ensemble strategy reference (but Yale has no tumor type labels)
- **Code**: Check paper
- **Status**: ⏸️ SKIP (different task — classification not temporal tracking)

---

### ❌ Papers Previously Listed but NOT Available (Removed):
- ~~Paper 12: SDC-Transformer~~ — Code not released, paper not in your collection
- ~~Paper 18: BraTS 2023 Ensemble~~ — Not in your collection
- ~~Papers 19-21: Pathology ViTs~~ — Different domain (histology, not MRI)

---

## 📊 Progress Summary

### Phase 1 (Preprocessing): ✅ 10/10 papers (100% COMPLETE!)
### Phase 2 (ViT): 🔄 4/6 analyzed (4 papers DONE, 2 remaining — low priority)
- **✅ ANALYZED**: Paper 11 (TaViT temporal blueprint) + Paper 13 (Swin UNETR embeddings) — CORE ARCHITECTURE!
- **✅ ANALYZED**: TransXAI (explainability blueprint for Phase 3) + CAFNet (hybrid >> pure ViT proof)
- **⏸️ SKIP**: ResAttU-Net-Swin, BRAIN-META (lower value, not needed for core pipeline)

**Detailed analysis**: See `PHASE2_VIT_PAPERS_ANALYSIS.md`

---

## 🔗 How All 10 Papers Connect (Complete Phase 1 Pipeline!)

```
TWO DATASETS STRATEGY WITH FULL HARMONIZATION:

YALE DATASET (Paper 3) - TRAINING          CYPRUS DATASET (Paper 7) - VALIDATION
11,884 scans, 1,430 patients       +       744 scans, 40 patients
USA, 20 years (2004-2023)                   Mediterranean, expert labels
    ↓                                            ↓
Apply BraTS Toolkit (Paper 1)               Already BraTS format!
    ↓                                            ↓
Apply FLIRE Registration (Paper 5)          Apply FLIRE Registration
80 days processing                          5 days processing
    ↓                                            ↓
Apply nnU-Net (Paper 2)                     Already has expert labels!
Generate tumor segmentations                3 subregions verified
    ↓                                            ↓
Extract Features                            Extract Features
(volumes, intensities, radiomics)           (110+ radiomic features)
    ↓                                            ↓
TEST for Scanner Effects (Paper 10++)       Use as ground truth
    ↓                                            ↓
Apply Nested ComBat (Paper 10+)             Validate harmonization
Multiple batch effects:                     Check if labels preserved
 • Year (2004→2023)                              ↓
 • Field strength                           Compare Yale vs Cyprus
 • Site/manufacturer                        Same tumor measurements?
    ↓                                            ↓
Apply Longitudinal ComBat (Paper 10)        ─────┘
Track temporal changes                      
Remove scanner-switch artifacts             
    ↓                                       
HARMONIZED TRAINING DATA                    
    ↓                                       
TRAIN MODELS HERE                          VALIDATE HERE
(Temporal ViT + LLM)                       (Ground truth)
    ↓                                           ↓
    └───────────────────┬───────────────────────┘
                        ↓
             ROBUST AI SYSTEM
         (Works across populations!)
```

**Complete Phase 1 Pipeline (All Papers Integrated!)**:
1. **BraTS Toolkit** (Paper 1): Clean + normalize + skull strip
2. **FLIRE** (Paper 5): Fast registration (10 mins/scan, 80 days total for Yale)
3. **nnU-Net** (Paper 2): Segment tumors (Yale needs labels, Cyprus validates)
4. **ComBat** (Paper 9): Baseline harmonization (fix scanner differences)
5. **Test scanner effects** (Paper 10++): Don't blindly harmonize! Verify need first
6. **Nested ComBat** (Paper 10+): Harmonize multiple batch effects (year/site/scanner)
7. **Longitudinal ComBat** (Paper 10): Remove temporal scanner artifacts
8. **Cyprus validation** (Paper 7): Verify harmonization preserves tumor biology

**Why This 4-Paper Harmonization Strategy?**
- **Paper 9 (ComBat)**: Baseline cross-sectional harmonization
- **Paper 10++ (Validation)**: Proves harmonization works, warns about over-harmonization
- **Paper 10+ (Nested)**: Handles Yale's 5+ batch effects (year/site/field/manufacturer/protocol)
- **Paper 10 (Longitudinal)**: Removes scanner-switch artifacts in temporal tracking

**Critical Insight**: Yale's 20 years (2004-2023) = patient scanned on scanner A (2010) → scanner B (2015) → scanner C (2020). Longitudinal ComBat ensures tumor growth measurements aren't confounded by scanner upgrades!

---

## 📚 Papers Still Needed (By Category)

### Category 1: Preprocessing & Data (100% COMPLETE! ✅✅✅)
1. ✅ Yale dataset paper (main training data - 1,430 patients)
2. ✅ Cyprus dataset paper (validation data - 40 patients with expert labels)
3. ✅ BraTS Toolkit (basic preprocessing)
4. ✅ nnU-Net (tumor segmentation)
5. ✅ Registration paper (align temporal sequences - tumor preservation)
6. ✅ FLIRE (fast registration - speed optimization)
7. ✅ ComBat harmonization (baseline scanner harmonization)
8. ✅ Longitudinal ComBat (temporal scanner harmonization)
9. ✅ ComBat validation (proof it works + warnings!)
10. ✅ Generalized ComBat (multiple batch effects + unknown confounds)

**Phase 1 Status: 10/10 papers analyzed = 100% COMPLETE! 🎉**

### Category 2: Vision Transformers (6 papers available ✅)
- ✅ Paper 11: Time-distance ViT — temporal longitudinal ViT (PUBLIC CODE!)
- ✅ Paper 13: Swin UNETR — 3D brain segmentation + embeddings (MONAI + weights!)
- ✅ TransXAI — explainability for Phase 3
- ✅ CAFNet — proves ViT+CNN fusion >> pure ViT
- ✅ ResAttU-Net-Swin — backup hybrid architecture
- ✅ BRAIN-META — ensemble strategy reference

**Phase 2 Status: 6 papers found, 2 critical (Papers 11 + 13), analysis in progress**

### Category 3: LLMs / Vision-Language Models (8 papers identified, 1 deep-analyzed ✅)
- ⭐⭐⭐ **RadFM** — 3D+2D generalist radiology foundation model (3D native!) ✅ **DEEP-ANALYZED**
  - **What**: 14B param model (3D ViT + Perceiver + MedLLaMA-13B), trained on 16M images
  - **Key numbers**: AUC 90.61 on BraTS brain tumors, beats GPT-4V overall (2.17 vs 1.99 human rating)
  - **For Yale**: Take Perceiver + LLM pattern, replace 3D ViT with our Swin UNETR (768-dim = exact match!)
  - **Published**: Nature Communications 2025 (top-tier!)
  - **Full analysis**: `objectivethree/SIMPLE_Paper_RadFM.md`
  - [GitHub](https://github.com/chaoyi-wu/RadFM) ✅ MIT | [Weights](https://huggingface.co/chaoyi-wu/RadFM) ✅
- ⭐⭐⭐ **MM-Embed** — Universal multimodal retrieval with MLLMs (ICLR 2025, NVIDIA) ✅ **ANALYZED**
  - **What**: First MLLM-based universal multimodal retriever handling all modality combinations
  - **Key numbers**: 52.7 R@5 M-BEIR (SOTA), 60.3 nDCG@10 MTEB (top-5), beats CLIP by +4.4 points
  - **For Yale**: Training methodology reference — modality bias discovery, curriculum learning validation, hard negative mining strategy. NOT a pipeline component (retrieval, not generation)
  - **Published**: ICLR 2025 (top-tier!) + NVIDIA Research
  - **Provided by**: Supervisor (recommended for multimodal LLM insights)
  - **Full analysis**: `objectivethree/SIMPLE_Paper_MM_Embed.md`
  - [Weights](https://huggingface.co/nvidia/MM-Embed) ✅
- ⭐⭐⭐ **LLaVA-Med** — Biomedical VLM with curriculum learning (NeurIPS 2023) — [GitHub](https://github.com/microsoft/LLaVA-Med) ✅
- ⭐⭐ **RaDialog** — Report generation + conversational dialog (MIDL 2025) — [GitHub](https://github.com/ChantalMP/RaDialog) ✅
- ⭐⭐ **R2GenGPT** — Frozen LLM + tiny adapter (5M params only!) — [GitHub](https://github.com/wang-zhanyu/R2GenGPT) ✅
- ⭐ **Med-Flamingo** — Few-shot medical VQA (Stanford) — [GitHub](https://github.com/snap-stanford/med-flamingo) ✅
- ⭐ **3D-CT-GPT** — 3D CT report generation (preprint, no code)
- ⭐ **MRG-LLM** — Dynamic prompt tuning (preprint, no code)
- 📌 **Med-PaLM 2** — Google proprietary (reference benchmark only, NOT usable)

**Phase 4 Status: RadFM deep-analyzed ✅, MM-Embed analyzed ✅ (provided by supervisor), 7 others identified for reference**
**Detailed analysis**: See `objectivethree/PHASE4_LLM_PAPERS_ANALYSIS.md`

### Category 4: Video/Diffusion (Need 3-4 papers)
- ⏳ Diffusion models for medical imaging
- ⏳ Video generation models
- ⏳ Temporal prediction in medical images

### Category 5: Clinical Validation (Need 2-3 papers)
- ⏳ RANO criteria for brain tumors
- ⏳ Clinical evaluation methods

---

## 🎯 Your Yale-Centered Workflow

**Now that you have the Yale dataset, here's your plan**:

### Phase 1: Get Yale Data Ready (Weeks 1-4)
Papers needed: ✅ BraTS, ✅ nnU-Net, ✅ Registration, ⏳ Harmonization

**What you'll do**:
1. Download Yale (11,884 scans)
2. Clean with BraTS methods
3. Segment with nnU-Net
4. **Register temporal sequences** (Paper 4's method - align T0→T1→T2...)
5. Harmonize scanner differences (need 1 more paper)

### Phase 2: Train Temporal ViT (Weeks 5-11)
Papers needed: ⏳ ViT papers (need to find these!)

**What you'll do**:
1. Design temporal ViT architecture
2. Train on Yale sequences (1,430 patients × 8 scans)
3. Learn tumor evolution patterns
4. Beat nnU-Net baseline

### Phase 3: Add LLM (Weeks 12-14)
Papers needed: ⏳ LLM papers (need to find these!)

**What you'll do**:
1. Connect ViT features to LLM
2. Use Yale clinical metadata
3. Generate explanations: "Tumor grew 15% post-radiation..."

### Phase 4: Generate Videos (Weeks 15-18)
Papers analyzed: ✅ 17 papers reviewed (6 key + 11 surveyed)

**Key papers**:
| Paper | Relevance | Code? | arXiv |
|---|---|---|---|
| ⭐⭐⭐ Treatment-aware Diffusion for Glioma (IEEE-TMI 2025) | Same domain, same task! | ❌ | [2309.05406](https://arxiv.org/abs/2309.05406) |
| ⭐⭐⭐ MedEdit (MICCAI24 SASHIMI) | Counterfactual brain MRI, beats Palette 45% | ❌ | [2407.15270](https://arxiv.org/abs/2407.15270) |
| ⭐⭐⭐ EchoNet-Synthetic (MICCAI 2024) | LVDM for medical video — **our starting code** | ✅ Code+Weights | [2406.00808](https://arxiv.org/abs/2406.00808) |
| ⭐⭐ Video LDM (CVPR 2023) | Temporal layer insertion strategy | ❌ | [2304.08818](https://arxiv.org/abs/2304.08818) |
| ⭐⭐ Counterfactual Diff. AE | Latent counterfactual manipulation | ✅ Code | — |
| ⭐⭐ CLIMATv2 | Multi-agent trajectory forecasting | ✅ Code | — |
| ⚙️ LDM (Rombach 2022) | Foundation backbone | ✅ Code+Weights | [2112.10752](https://arxiv.org/abs/2112.10752) |
| ⚙️ DDPM (Ho 2020) | Diffusion theory foundation | ✅ Code | [2006.11239](https://arxiv.org/abs/2006.11239) |

Full analysis → `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md`

**What you'll do**:
1. Adapt EchoNet-Synthetic LVDM codebase to 3D brain MRI
2. Insert temporal layers (Video LDM strategy) + treatment conditioning (Glioma paper)
3. Condition on Swin UNETR embeddings + RadFM narratives + treatment metadata
4. Generate natural progression + counterfactual scenarios (MedEdit approach)
5. Validate on held-out Yale sequences

### Phase 5: Validate (Weeks 19-20)
Papers needed: ⏳ Validation papers

**What you'll do**:
1. Test on Yale hold-out set
2. Compare to radiologist assessments
3. Measure clinical utility

---

## 📊 Dataset Strategy (Simple Version)

**Primary dataset: Yale (11,884 scans)**
- Use for: Everything (training, validation, testing)
- Provides: Temporal sequences, clinical data, real-world variability

**Optional additions** (if time permits):
- BraTS: For ViT pre-training (more data = better features)
- UCSF: For external validation (proves it works elsewhere)
- LUMIERE: For clinical assessment validation

**Recommendation**: **Start with Yale ONLY**
- It's big enough (1,430 patients)
- Has everything you need (temporal, clinical, diverse)
- Adding more datasets = more complexity
- Focus > quantity

---

## 🎯 Remaining Work

**Only Phase 6 (Evaluation) papers remain**:
1. ⏳ RANO criteria for brain tumor assessment
2. ⏳ Clinical evaluation methodology papers
3. ⏳ Video quality assessment metrics for medical imaging

---

## 💾 Download Checklist

**Immediate action items**:
- [ ] Download Yale dataset from TCIA (~200GB, may take 1-2 days)
- [ ] Download BraTS 2023 for preprocessing learning (~100GB)
- [ ] Install: HD-BET, nnU-Net, BraTS Toolkit
- [ ] Prepare compute: Need GPU (8-16GB VRAM minimum)

**Expected storage**:
- Yale: ~200 GB
- BraTS: ~100 GB
- Processed data: ~300 GB
- **Total: ~600 GB** (get 1TB hard drive!)

---

## 📊 Current Progress

**Papers Analyzed**: 22 key papers + 18 surveyed across all 5 phases

**Phase 1 (Preprocessing)**: ✅ **COMPLETE** — 10/10 papers
- ✅ Data identified (Yale)
- ✅ Skull stripping (HD-BET via BraTS Toolkit)
- ✅ Tumor segmentation (nnU-Net)
- ✅ Registration (FLIRE + Elastix)
- ✅ Scanner harmonization (ComBat + LongComBat + Generalized ComBat)

**Phase 2-3 (ViT)**: ✅ **COMPLETE** — 4/6 papers (2 lower priority skipped)
- ✅ Swin UNETR (768-dim embeddings)
- ✅ TaViT (temporal attention)
- ✅ TransXAI (explainability), CAFNet (hybrid validation)

**Phase 4 (LLM)**: ✅ **COMPLETE** — 2 deep-analyzed + 7 surveyed
- ✅ RadFM (Nature Comms 2025) — main pipeline component
- ✅ MM-Embed (ICLR 2025) — training methodology guide

**Phase 5 (Video/Diffusion)**: ✅ **COMPLETE** — 7 papers deep-analyzed from PDFs + 10 surveyed (17 total)
- ✅ DDPM (Ho et al., NeurIPS 2020) — foundation math, ε-prediction, L_simple, T=1000
- ✅ LDM (Rombach et al., CVPR 2022) — VAE + latent diffusion + cross-attention conditioning, f=4-8 optimal
- ✅ Video LDM (Blattmann et al., CVPR 2023) — temporal layers, reshape trick, merge α, frozen spatial
- ✅ TaDiff / Treatment-aware Diffusion (Liu et al., IEEE-TMI 2025) — same domain! sinusoidal+MLP treatment conditioning, joint seg+gen, ω weighting, T=600, 350 GPU-hours
- ✅ MedEdit (Ben Alaya et al., MICCAI24 SASHIMI) — counterfactual brain MRI, mask dilation k=25, realism=3.20/5 (=real!), RePaint+resampling
- ✅ EchoNet-Synthetic (Reynaud et al., MICCAI 2024) — LVDM, 3-model pipeline (VAE→LIDM→LVDM), video stitching, **full code+weights** (starting codebase)
- ✅ Counterfactual Diffusion AE (Atad et al., JMLBI 2024) — DAE latent space, hyperplane reflection, z_sem 512-dim
- 📄 `objectivefour/DIFFUSION_MODELS_EXPLAINED.md` — ELI10-style explainer document for diffusion concepts
- 📄 `objectivefour/PHASE5_DIFFUSION_PAPERS_ANALYSIS.md` — deep analysis of all papers

**Phase 6 (Evaluation)**: ⏳ Not started

---

**Last updated**: All 7 Phase 5 PDFs deep-analyzed, explainer doc created, MASTER_PLAN updated with technical details
**Main dataset**: ✅ Yale Brain Mets (11,884 scans, 1,430 patients)
**Full pipeline designed**: ✅ Raw MRI → HD-BET → nnU-Net → FLIRE → Swin UNETR → ComBat → TaViT → RadFM → Video Diffusion
**Ready to implement**: ✅ All phases have paper backing and architecture decisions made
