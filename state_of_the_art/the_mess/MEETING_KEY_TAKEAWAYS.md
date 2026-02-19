# Key Takeaways for Supervisor Meeting
## State of the Art Understanding & Project Progression

**Project:** Explainable Disease Progression and Counterfactual Video Generation with Vision–Language Models  
**Intern:** Mohamed Boufafa  
**Meeting Preparation Date:** January 2026

---

## 1. Project Understanding Summary

### What We're Building
A **multimodal AI framework** that can:
1. Analyze longitudinal cancer images (MRI/CT over time)
2. Extract meaningful tumor representations using Vision Transformers
3. Generate clinical explanations using Large Language Models
4. Create videos showing predicted cancer progression
5. Enable "what-if" scenarios (counterfactual analysis)

### Why It Matters
- Current AI systems give **static predictions** (single image → risk score)
- Clinicians need **temporal understanding** (how the tumor evolves)
- Patients and doctors need **explainability** (why the AI thinks this)
- Treatment planning needs **counterfactual reasoning** (what if we try treatment X vs Y?)

---

## 2. State of the Art - Key Points to Discuss

### Vision Transformers (Core of Objective 2)
| Technology | Key Insight | Relevance |
|------------|-------------|-----------|
| **ViT** | Images as 16×16 patches → transformer | Foundation architecture |
| **Swin Transformer** | Hierarchical + shifted windows → efficient | Handles high-res medical images |
| **UNETR** | ViT encoder + CNN decoder | Best for 3D segmentation |
| **Swin UNETR (self-supervised)** | Pretraining with inpainting, rotation, contrastive | **Critical for limited labeled data** |

**Key Insight:** Self-supervised pretraining on unlabeled medical data achieves **84.72% Dice** with only 10-50% of labels needed.

### Large Language Models (Core of Objective 3)
| Technology | Key Insight | Relevance |
|------------|-------------|-----------|
| **BioBERT** | Pretrained on PubMed (4.5B words) | Medical text encoding |
| **Med-PaLM** | 67.6% on USMLE-level questions | Clinical reasoning capability |
| **Med-VLP** | Vision-language pretraining | Aligns images with clinical text |

**Key Insight:** Domain-specific pretraining (BioBERT > BERT) is **essential** for medical applications.

### Generative Models (Core of Objective 4)
| Technology | Key Insight | Relevance |
|------------|-------------|-----------|
| **DDPM** | Progressive denoising → high quality | Foundation for video generation |
| **Video LDM** | Temporal layers + pretrained image LDM | **Primary architecture for our videos** |

**Key Insight:** Video LDM achieves temporal coherence by inserting temporal attention layers into pretrained Stable Diffusion.

---

## 3. Dataset Understanding

### Primary Datasets for Phase 1

| Dataset | Type | Size | Key Features |
|---------|------|------|--------------|
| **BraTS** | Brain MRI | 2,000+ scans | Multimodal (T1, T2, FLAIR), expert annotations |
| **TCGA** | Multi-cancer | 11,000+ patients | Clinical metadata, genomics integration |
| **LIDC-IDRI** | Lung CT | 1,018 scans | 4 radiologist annotations, malignancy scores |

### Questions for Supervisor
1. **Dataset Priority:** Should we focus primarily on BraTS (brain) or also include LIDC-IDRI (lung)?
2. **Longitudinal Data:** BraTS has limited longitudinal cases - should we consider synthetic augmentation?
3. **Annotation Strategy:** For temporal modeling, do we need to re-annotate tumor boundaries at each time point?

---

## 4. Technical Approach Discussion Points

### Phase 1 Approach (Current Focus)
```
Week 1-2: Dataset acquisition and initial exploration
Week 3: Preprocessing pipeline implementation
Week 4: EDA and longitudinal organization
```

### Key Technical Decisions Needed
1. **Preprocessing Framework:** MONAI vs. custom SimpleITK pipeline?
2. **Normalization Strategy:** Z-score per volume vs. histogram matching?
3. **Registration Method:** Rigid vs. deformable for longitudinal alignment?
4. **Storage Format:** NIfTI vs. preprocessed numpy arrays?

---

## 5. Potential Challenges to Discuss

| Challenge | Current Understanding | Need Supervisor Input |
|-----------|----------------------|----------------------|
| Limited longitudinal data | Plan: self-supervised pretraining | Is synthetic data augmentation acceptable? |
| Computational resources | Need A100 GPUs for training | What cluster access is available? |
| Clinical validation | Need expert evaluation | Can we get clinical collaborators? |
| Temporal coherence in videos | Video LDM approach | Any medical video benchmarks to use? |

---

## 6. Questions for Supervisor

### Technical Questions
1. For 3D medical image segmentation, should we start with **UNETR or Swin UNETR**?
2. What's the preferred approach for handling **variable scan intervals** in longitudinal data?
3. Should we implement **both** brain (BraTS) and lung (LIDC-IDRI) pipelines, or focus on one?

### Project Scope Questions
4. What level of **clinical validation** is expected for the final deliverables?
5. Are there existing **preprocessing pipelines** from the lab we should build upon?
6. What's the priority: **video quality** vs. **clinical plausibility**?

### Resource Questions
7. What **computational resources** are available (GPUs, storage)?
8. Are there **clinical collaborators** who can provide feedback on generated videos?
9. What's the expected **publication venue** (conference deadline to target)?

---

## 7. Proposed Next Steps

### Immediate (Week 1)
- [ ] Confirm dataset selection with supervisor
- [ ] Set up development environment (PyTorch, MONAI)
- [ ] Begin BraTS data download and exploration

### Short-term (Weeks 2-4)
- [ ] Implement preprocessing pipeline
- [ ] Complete EDA report
- [ ] Organize longitudinal sequences

### Medium-term (Weeks 5-7)
- [ ] Implement CNN baselines
- [ ] Begin ViT adaptation for medical imaging

---

## 8. Key References to Mention

If supervisor asks about specific papers:

1. **For ViT approach:** "We're following the Swin UNETR self-supervised pretraining from Tang et al. (CVPR 2022)"
2. **For LLM integration:** "Med-PaLM shows instruction prompt tuning achieves 67.6% on MedQA without fine-tuning"
3. **For video generation:** "Video LDM from Blattmann et al. (CVPR 2023) achieves FVD 550 on UCF-101"
4. **For medical VLP:** "The Med-VLP benchmark shows cross-modal alignment is key for clinical reasoning"

---

## Meeting Agenda Suggestion

1. **5 min:** Quick project overview confirmation
2. **10 min:** State of the art discussion (ViT, LLM, diffusion models)
3. **10 min:** Phase 1 planning and dataset decisions
4. **10 min:** Technical decisions (preprocessing, tools, resources)
5. **5 min:** Questions and next steps

---

*Prepared for supervisor meeting - Mitacs Globalink Research Award at TÉLUQ University*
