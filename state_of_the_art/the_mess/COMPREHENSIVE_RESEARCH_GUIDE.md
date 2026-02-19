# Comprehensive Research Guide: Explainable Disease Progression and Counterfactual Video Generation with Vision–Language Models

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Current State of the Art](#current-state-of-the-art)
4. [Research Objectives and Realization Plan](#research-objectives-and-realization-plan)
5. [Reference Analysis with Helpfulness Scores](#reference-analysis-with-helpfulness-scores)
6. [Reference-to-Objective Mapping](#reference-to-objective-mapping)
7. [Technical Implementation Roadmap](#technical-implementation-roadmap)
8. [Key Takeaways and Recommendations](#key-takeaways-and-recommendations)

---

## Executive Summary

This document serves as a comprehensive reference for the **Mitacs Globalink Research Award** project at TÉLUQ University, Montreal, Canada. The project aims to develop a **multimodal AI framework** that integrates:
- **Large Vision Models (LVMs)** and **Vision Transformers (ViTs)** for tumor representation
- **Large Language Models (LLMs)** for clinical reasoning and explainability
- **Diffusion-based generative models** for cancer progression video synthesis

**Duration:** 20 weeks (April 1 - August 19, 2026)

**Supervisors:**
- **Host (Canada):** Dr. Belkacem Chikhaoui, TÉLUQ University
- **Home (Algeria):** Dr. Belkacem Khaldi, ESI-SBA

**Research Intern:** Mohamed Boufafa

---

## Project Overview

### Research Problem
The primary research problem is the **lack of interpretable and dynamic AI models for longitudinal cancer progression analysis**. Current systems suffer from three critical limitations:

1. **Static Analysis:** Most approaches rely on single-image analysis, failing to capture the temporal nature of cancer development
2. **Limited Interpretability:** Predictions are abstract numerical outputs (probabilities, risk scores) with minimal clinical utility
3. **No Counterfactual Reasoning:** Existing models cannot simulate "what-if" scenarios essential for personalized medicine

### Research Questions
1. Can Vision Transformer-based models capture meaningful representations of tumor morphology evolution from longitudinal medical imaging?
2. Can Large Language Models integrate imaging-derived features with clinical metadata to provide medically grounded reasoning?
3. Is it possible to generate temporally consistent, clinically plausible videos visualizing cancer progression?
4. Can such a system support counterfactual analysis by simulating alternative tumor trajectories under different treatment scenarios?

---

## Current State of the Art

### 1. Medical Imaging Datasets and Benchmarks

#### Cancer Genomics and Multi-Omics
The **Cancer Genome Atlas (TCGA)** represents the largest and most comprehensive cancer genomics resource, covering **33 cancer types** with over **11,000 patients**. TCGA provides multi-platform genomic data including:
- Exome sequencing
- DNA copy number arrays
- DNA methylation
- RNA sequencing
- microRNA sequencing
- Reverse phase protein arrays

The Pan-Cancer analysis project has enabled discovery of **127 cancer driver genes** and **299 driver mutations**, establishing molecular classification that often transcends traditional tissue-of-origin boundaries.

#### Brain Tumor Imaging
The **BraTS (Brain Tumor Segmentation) Benchmark** has become the standard for evaluating brain tumor segmentation algorithms. It uses multimodal MRI data (T1, T1-Gd, T2, FLAIR) with expert annotations for:
- Enhancing tumor core
- Non-enhancing tumor core
- Peritumoral edema

State-of-the-art methods achieve **Dice scores of 0.85-0.90** for whole tumor segmentation, with significant improvements coming from transformer-based architectures.

#### Lung Nodule Detection
The **LIDC-IDRI database** contains **1,018 CT scans** with annotations from 4 experienced radiologists. It includes:
- 7,371 lesions marked as "nodule ≥3mm"
- 2,669 lesions marked as "nodule <3mm"
- 2,331 "non-nodules ≥3mm"
- XML-formatted annotations with spatial coordinates and radiologist characteristics

### 2. Vision Transformers for Medical Imaging

#### Foundational Architecture: ViT
The **Vision Transformer (ViT)** introduced the paradigm of treating images as sequences of patches, applying the transformer architecture originally designed for NLP. Key innovations:
- Images split into **16×16 patches**
- Linear projection to embedding space
- Learnable [CLS] token for classification
- Positional embeddings for spatial information
- **State-of-the-art results:** 88.55% ImageNet (ViT-H/14)

#### Hierarchical Vision: Swin Transformer
The **Swin Transformer** addresses ViT's limitations for dense prediction tasks through:
- **Hierarchical feature maps** at multiple scales (like CNNs)
- **Shifted window attention** for cross-window connectivity
- **Linear computational complexity** with respect to image size
- **Results:** 87.3% ImageNet top-1, 58.7 box AP on COCO, 53.5 mIoU on ADE20K

#### Medical Image Segmentation: UNETR
**UNETR** (UNEt TRansformers) combines transformer encoders with CNN decoders for 3D medical image segmentation:
- **Architecture:** ViT encoder + CNN decoder via skip connections
- **Performance:** 0.891 average Dice on BTCV (multi-organ CT)
- **Key advantage:** Captures global context while maintaining fine-grained localization

#### Self-Supervised Learning: Swin UNETR
**Swin UNETR with self-supervised pretraining** represents the current state-of-the-art for medical image segmentation:
- Pretrained on **5,050 CT scans** (10,480 labeled, 3,045 unlabeled)
- **Three pretraining tasks:**
  1. Masked volume inpainting
  2. 3D rotation prediction (4 classes)
  3. Contrastive learning
- **Results:** 84.72% Dice on BTCV (4.05% improvement over random initialization)
- **Few-shot capability:** Comparable performance with only 10-50% of labeled data

### 3. Deep Learning for Medical Classification

#### CNN Ensemble Methods
For medical image classification, **ensemble methods** combining multiple CNN architectures have proven effective:
- **Components:** GoogLeNet, AlexNet, ResNet, VGGNet
- **Fusion strategies:** Weighted averaging, voting
- **Skin lesion classification:** Average AUC of 0.891 across 7 disease categories
- **Key insight:** Ensemble diversity improves robustness and generalization

### 4. Large Language Models for Clinical Knowledge

#### Biomedical NLP: BioBERT
**BioBERT** is a domain-specific language model pretrained on biomedical corpora:
- **Pretraining data:** 
  - PubMed Abstracts: 4.5B words
  - PMC Full-text: 13.5B words
- **Tasks:** Named Entity Recognition, Relation Extraction, Question Answering
- **Results:** Outperforms BERT on all biomedical NLP benchmarks
- **Key finding:** Domain-specific pretraining is crucial for biomedical text understanding

#### Clinical Question Answering: Med-PaLM
**Med-PaLM** (Medical Pathways Language Model) represents the frontier of LLMs for clinical knowledge:
- Built on **PaLM 540B** with medical instruction prompt tuning
- **MultiMedQA benchmark:** Combines 6 existing medical QA datasets + new HealthSearchQA
- **Results:** 67.6% accuracy on MedQA (US Medical Licensing Exam questions)
- **Human evaluation framework:** Clinical alignment, safety, comprehension across 8 dimensions
- **Key innovation:** Instruction prompt tuning without parameter updates

#### Clinical Database: MIMIC-III
**MIMIC-III** provides comprehensive clinical data for developing and validating AI systems:
- **Scope:** 53,423 ICU admissions, 38,597 patients, Beth Israel Deaconess (2001-2012)
- **Data types:**
  - Charted observations (vital signs, medications)
  - Laboratory test results
  - Procedures and diagnosis codes (ICD-9)
  - Free-text clinical notes
  - Imaging reports
- **Significance:** Enables research on clinical reasoning, prediction, and multimodal fusion

### 5. Medical Vision-Language Pretraining

**Medical Vision-Language Pretraining (Med-VLP)** combines visual and textual modalities for healthcare AI:
- **Tasks:** Medical VQA, Report Generation, Image-Text Retrieval
- **Architecture:** Dual encoders (vision + language) with cross-modal attention
- **RGC Dataset:** 50,000 Radiology images with Generated Captions
- **Challenges:**
  - Domain-specific visual features
  - Medical terminology understanding
  - Alignment between images and clinical text
- **Applications:** Automated report generation, clinical decision support

### 6. Generative Models for Medical Imaging

#### Denoising Diffusion Probabilistic Models (DDPM)
**DDPM** has revolutionized generative modeling with superior sample quality:
- **Forward process:** Gradually adds Gaussian noise over T steps
- **Reverse process:** Learns to denoise, generating samples from pure noise
- **Results:** FID 3.17 on CIFAR-10 (unconditional), competitive with GANs
- **Theory:** Connections to score matching, Langevin dynamics, variational inference
- **Key insight:** Progressive denoising enables stable training and high-quality generation

#### Video Latent Diffusion Models (Video LDM)
**Video LDM** extends diffusion models to video generation:
- **Architecture:** 
  - Image LDM backbone (Stable Diffusion)
  - Temporal layers for video coherence
  - Latent space operation for efficiency
- **Key innovations:**
  - Temporal attention layers inserted into U-Net
  - Learned temporal interpolation for high frame rates
  - Video-finetuned autoencoders for reduced flickering
- **Results:** 
  - UCF-101 FVD: 550.61 (SD 2.1)
  - MSR-VTT CLIPSIM: 0.2929
  - Inception Score: 33.45
- **Applications:** Text-to-video, driving scene generation, personalized generation (DreamBooth)

---

## Research Objectives and Realization Plan

### Objective 1: Develop a Robust Pipeline for Longitudinal Cancer Imaging Analysis
**Duration:** Weeks 1-4 (Phase 1)

**Goal:** Build a reproducible, clinically meaningful data pipeline for processing longitudinal cancer imaging data.

**Tasks:**
1. **Dataset Acquisition:**
   - Download TCGA imaging data (multi-cancer)
   - Acquire BraTS challenge data (brain MRI)
   - Obtain LIDC-IDRI scans (lung CT)

2. **Preprocessing Pipeline:**
   - Intensity normalization (z-score, histogram matching)
   - Spatial resampling to isotropic resolution
   - Longitudinal image alignment using registration
   - Tumor annotation processing (segmentation masks)

3. **Data Organization:**
   - Structure patient data chronologically
   - Create train/validation/test splits
   - Implement data loading utilities

**Tools:** Python, MONAI, SimpleITK, nibabel, PyTorch DataLoaders

**Deliverables:**
- Reproducible preprocessing pipeline
- Cleaned, standardized datasets
- Exploratory data analysis report

---

### Objective 2: Learn High-Level Tumor Representations Using Vision Transformers
**Duration:** Weeks 5-11 (Phases 2-3)

**Goal:** Extract semantically rich, temporally informative tumor representations.

**Phase 2 (Weeks 5-7): CNN Baselines**
- Implement ResNet, EfficientNet baselines
- Train on single time-point classification/segmentation
- Establish benchmark performance metrics

**Phase 3 (Weeks 8-11): Vision Transformers**
1. **Model Implementation:**
   - Adapt ViT for medical imaging
   - Implement Swin Transformer for hierarchical features
   - Configure UNETR for segmentation tasks

2. **Self-Supervised Pretraining:**
   - Implement Swin UNETR pretraining strategy
   - Tasks: Masked volume inpainting, rotation prediction, contrastive learning
   - Pretrain on available unlabeled data

3. **Temporal Modeling:**
   - Extract embeddings for each time point
   - Model temporal evolution as embedding sequences
   - Analyze representation changes over time

**Deliverables:**
- Baseline model implementations
- Transformer-based embeddings
- Longitudinal representation framework
- Visualization of tumor evolution features

---

### Objective 3: Integrate Imaging and Clinical Context Using Large Language Models
**Duration:** Weeks 12-14 (Phase 4)

**Goal:** Enable multimodal clinical reasoning by combining visual representations with clinical metadata.

**Tasks:**
1. **Multimodal Representation Design:**
   - Fuse imaging embeddings with clinical variables
   - Design projection layers for modality alignment
   - Implement cross-attention mechanisms

2. **LLM Integration:**
   - Experiment with BioBERT for medical text encoding
   - Design prompt templates for disease progression narratives
   - Implement few-shot prompting strategies

3. **Clinical Reasoning:**
   - Generate textual explanations linked to imaging changes
   - Align visual evidence with medical reasoning
   - Validate clinical plausibility with medical knowledge

**Deliverables:**
- Multimodal embedding representation
- LLM prompt templates
- Generated clinical narratives
- Qualitative validation results

---

### Objective 4: Generate and Analyze Cancer Progression Videos
**Duration:** Weeks 15-18 (Phase 5)

**Goal:** Visualize cancer progression through AI-generated video sequences.

**Tasks:**
1. **Model Adaptation:**
   - Adapt DDPM for medical imaging domain
   - Implement Video LDM architecture
   - Configure temporal attention layers

2. **Conditional Generation:**
   - Condition on vision embeddings (tumor features)
   - Condition on language embeddings (clinical context)
   - Implement classifier-free guidance

3. **Video Generation:**
   - Generate temporally consistent progression videos
   - Implement temporal interpolation for smooth transitions
   - Quality enhancement through video-finetuned decoders

4. **Counterfactual Scenarios:**
   - Treatment vs. no treatment trajectories
   - Different treatment response simulations
   - Alternative progression pathways

**Deliverables:**
- Cancer progression video sequences
- Counterfactual trajectory visualizations
- Analysis of realism and coherence

---

### Objective 5: Evaluate Explainability, Clinical Plausibility, and Scientific Impact
**Duration:** Weeks 19-20 (Phase 6)

**Goal:** Rigorously assess the framework's performance and medical relevance.

**Tasks:**
1. **Quantitative Evaluation:**
   - FID/FVD for video quality
   - Dice scores for segmentation
   - Temporal consistency metrics
   - Classification accuracy (if applicable)

2. **Qualitative Evaluation:**
   - Visual plausibility assessment
   - Text-to-video alignment verification
   - Clinician-aligned interpretation analysis

3. **Documentation:**
   - Comprehensive technical report
   - Publication-ready manuscript draft
   - Open-source code release

**Deliverables:**
- Evaluation results report
- Final technical documentation
- Publication-ready manuscript

---

## Reference Analysis with Helpfulness Scores

### Reference 1: TCGA Pan-Cancer Analysis Project
**Paper:** Weinstein et al. "The Cancer Genome Atlas Pan-Cancer analysis project" Nature Genetics, 2013

**Summary:**
The Cancer Genome Atlas provides the most comprehensive cancer genomics resource, covering 33 cancer types with multi-platform molecular profiling. The Pan-Cancer analysis enables cross-tumor comparisons and discovery of shared molecular features.

**Key Contributions:**
- 11,000+ patient samples across 33 cancer types
- Integration of exome sequencing, copy number, methylation, RNA-seq, miRNA, protein arrays
- Discovery of 127 driver genes and 299 driver mutations
- Molecular subtyping that transcends tissue-of-origin

**Relevance to Project:**
- Foundation dataset for multi-cancer analysis
- Enables understanding of cancer progression mechanisms
- Provides ground truth for molecular-level disease evolution
- Supports clinical metadata integration (staging, treatment)

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **9/10** |
| Data Foundation | 10/10 |
| Methodology Guidance | 7/10 |
| Direct Implementation | 8/10 |
| Clinical Relevance | 10/10 |

**Primary Objectives:** 1 (Data Pipeline), 3 (Clinical Context)

---

### Reference 2: BraTS Benchmark (Brain Tumor Segmentation)
**Paper:** Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark" IEEE TMI, 2015

**Summary:**
BraTS establishes the standard benchmark for brain tumor segmentation using multimodal MRI (T1, T1-Gd, T2, FLAIR). Provides expert annotations and evaluation framework for glioma segmentation.

**Key Contributions:**
- Multimodal MRI dataset with expert annotations
- Standardized evaluation metrics (Dice, Hausdorff distance)
- Three tumor regions: enhancing core, non-enhancing core, edema
- Annual challenge driving methodological advances

**Relevance to Project:**
- Primary dataset for brain tumor progression analysis
- Longitudinal cases available for temporal modeling
- Well-defined segmentation tasks for ViT evaluation
- Established baseline for comparison

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 10/10 |
| Methodology Guidance | 9/10 |
| Direct Implementation | 10/10 |
| Clinical Relevance | 10/10 |

**Primary Objectives:** 1 (Data Pipeline), 2 (ViT Representations), 5 (Evaluation)

---

### Reference 3: LIDC-IDRI (Lung Image Database Consortium)
**Paper:** Armato et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)" Medical Physics, 2011

**Summary:**
LIDC-IDRI provides 1,018 CT scans with annotations from four radiologists, enabling research on lung nodule detection and characterization.

**Key Contributions:**
- 1,018 diagnostic CT scans
- Two-phase annotation process (blind + unblind review)
- 7,371 lesions ≥3mm, 2,669 lesions <3mm
- Radiologist characteristic ratings for malignancy assessment

**Relevance to Project:**
- Secondary dataset for lung cancer analysis
- Enables multi-site cancer progression modeling
- Provides radiologist variability information
- Supports counterfactual analysis with malignancy scores

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **8/10** |
| Data Foundation | 9/10 |
| Methodology Guidance | 7/10 |
| Direct Implementation | 8/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 1 (Data Pipeline), 4 (Video Generation)

---

### Reference 4: Vision Transformer (ViT)
**Paper:** Dosovitskiy et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" ICLR, 2021

**Summary:**
ViT demonstrates that pure transformer architectures can achieve state-of-the-art image classification when pretrained on large datasets, establishing the foundation for vision transformers.

**Key Contributions:**
- Patch-based image tokenization (16×16 patches)
- Direct application of transformer encoder
- [CLS] token for classification
- 88.55% ImageNet accuracy (ViT-H/14)
- Demonstrates importance of large-scale pretraining

**Relevance to Project:**
- Core architecture for tumor representation learning
- Foundation for understanding transformer-based medical imaging
- Basis for implementing and fine-tuning on cancer data
- Establishes patch-based processing paradigm

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 6/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 10/10 |
| Clinical Relevance | 7/10 |

**Primary Objectives:** 2 (ViT Representations)

---

### Reference 5: UNETR (Transformers for 3D Medical Image Segmentation)
**Paper:** Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation" WACV, 2022

**Summary:**
UNETR combines transformer encoders with CNN decoders for volumetric medical image segmentation, achieving state-of-the-art results on multi-organ segmentation.

**Key Contributions:**
- First transformer-based 3D medical image segmentation
- Skip connections from transformer to CNN decoder
- 0.891 average Dice on BTCV (multi-organ CT)
- Superior performance on MSD brain tumor segmentation
- Captures global context with fine-grained localization

**Relevance to Project:**
- Direct architecture for tumor segmentation task
- Enables volumetric tumor representation extraction
- Proven effectiveness on brain tumor data
- Foundation for temporal embedding extraction

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 7/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 10/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 2 (ViT Representations), 1 (Data Pipeline)

---

### Reference 6: Self-Supervised Swin UNETR
**Paper:** Tang et al. "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis" CVPR, 2022

**Summary:**
Proposes self-supervised pretraining framework for 3D medical image analysis using Swin UNETR, achieving state-of-the-art with limited labeled data.

**Key Contributions:**
- Pretraining on 5,050 CT volumes (10,480 labeled + 3,045 unlabeled)
- Three pretext tasks: inpainting, rotation prediction, contrastive learning
- 84.72% Dice on BTCV (4.05% improvement)
- Effective with only 10-50% labeled data
- State-of-the-art on MSD segmentation challenges

**Relevance to Project:**
- Critical for leveraging unlabeled medical data
- Provides pretraining strategy for limited annotations
- State-of-the-art architecture for implementation
- Enables robust tumor representations

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 8/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 10/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 2 (ViT Representations), 1 (Data Pipeline)

---

### Reference 7: Swin Transformer
**Paper:** Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" ICCV, 2021

**Summary:**
Swin Transformer introduces hierarchical vision transformers with shifted window attention, achieving linear computational complexity and state-of-the-art across vision tasks.

**Key Contributions:**
- Hierarchical feature maps (like CNNs)
- Shifted window attention for cross-window connections
- Linear complexity O(n) vs ViT's O(n²)
- 87.3% ImageNet top-1 accuracy
- SOTA on COCO (58.7 box AP) and ADE20K (53.5 mIoU)

**Relevance to Project:**
- Core architecture for efficient tumor processing
- Enables multi-scale tumor feature extraction
- Foundation for Swin UNETR implementation
- Scalable to high-resolution medical images

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **9/10** |
| Data Foundation | 5/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 9/10 |
| Clinical Relevance | 7/10 |

**Primary Objectives:** 2 (ViT Representations)

---

### Reference 8: CNN Ensembles for Medical Classification
**Paper:** Harangi, B. "Skin lesion classification with ensembles of deep convolutional neural networks" Journal of Biomedical Informatics, 2018

**Summary:**
Demonstrates effectiveness of CNN ensemble methods for medical image classification, achieving competitive results on dermoscopy images.

**Key Contributions:**
- Ensemble of GoogLeNet, AlexNet, ResNet, VGGNet
- Weighted averaging fusion strategy
- Average AUC of 0.891 on skin lesion classification
- Shows ensemble diversity improves generalization
- Practical approach for limited medical data

**Relevance to Project:**
- Baseline methodology for comparison
- Ensemble strategies applicable to ViT models
- Demonstrates importance of model diversity
- Practical implementation guidance

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **6/10** |
| Data Foundation | 4/10 |
| Methodology Guidance | 7/10 |
| Direct Implementation | 6/10 |
| Clinical Relevance | 7/10 |

**Primary Objectives:** 2 (ViT Representations - Baseline), 5 (Evaluation)

---

### Reference 9: BioBERT
**Paper:** Lee et al. "BioBERT: a pre-trained biomedical language representation model" Bioinformatics, 2020

**Summary:**
BioBERT is a domain-specific BERT pretrained on biomedical literature, significantly outperforming general BERT on biomedical NLP tasks.

**Key Contributions:**
- Pretrained on PubMed (4.5B words) + PMC (13.5B words)
- SOTA on biomedical NER, relation extraction, QA
- Demonstrates importance of domain-specific pretraining
- Publicly available pretrained weights
- Foundation for biomedical NLU tasks

**Relevance to Project:**
- Text encoder for clinical metadata
- Enables medical report understanding
- Foundation for LLM-based reasoning
- Supports clinical narrative generation

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **8/10** |
| Data Foundation | 6/10 |
| Methodology Guidance | 9/10 |
| Direct Implementation | 8/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 3 (LLM Integration)

---

### Reference 10: Med-PaLM (Large Language Models Encode Clinical Knowledge)
**Paper:** Singhal et al. "Large Language Models Encode Clinical Knowledge" Nature, 2023

**Summary:**
Med-PaLM demonstrates that large language models can encode clinical knowledge and achieve physician-level performance on medical question answering with instruction prompt tuning.

**Key Contributions:**
- Built on PaLM 540B with medical instruction tuning
- MultiMedQA benchmark (6 medical QA datasets + HealthSearchQA)
- 67.6% accuracy on MedQA (USMLE-style questions)
- Human evaluation framework for clinical quality
- First LLM to approach physician-level medical QA

**Relevance to Project:**
- Guidance for LLM integration strategies
- Prompt engineering techniques for medical domain
- Evaluation framework for clinical reasoning
- Understanding LLM capabilities and limitations

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **9/10** |
| Data Foundation | 6/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 7/10 |
| Clinical Relevance | 10/10 |

**Primary Objectives:** 3 (LLM Integration), 5 (Evaluation)

---

### Reference 11: MIMIC-III Clinical Database
**Paper:** Johnson et al. "MIMIC-III, a freely accessible critical care database" Scientific Data, 2016

**Summary:**
MIMIC-III provides comprehensive clinical data from ICU admissions, enabling research on clinical decision support and multimodal medical AI.

**Key Contributions:**
- 53,423 ICU admissions, 38,597 patients
- 11 years of data (2001-2012)
- Vital signs, laboratory results, medications, procedures
- Free-text clinical notes and imaging reports
- ICD-9 diagnosis and procedure codes

**Relevance to Project:**
- Source of clinical metadata for multimodal fusion
- Enables training clinical reasoning models
- Provides structured clinical variables
- Supports longitudinal patient analysis

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **7/10** |
| Data Foundation | 9/10 |
| Methodology Guidance | 6/10 |
| Direct Implementation | 7/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 1 (Data Pipeline), 3 (LLM Integration)

---

### Reference 12: Medical Vision-Language Pretraining
**Paper:** Li et al. "Multi-modal Pre-training for Medical Vision-language Understanding and Generation" arXiv, 2023

**Summary:**
Comprehensive study of vision-language pretraining for medical applications, covering medical VQA, report generation, and introducing the RGC benchmark.

**Key Contributions:**
- Empirical study of medical VLP methods
- RGC dataset: 50,000 radiology images with captions
- Analysis of pretraining strategies for medical domain
- Benchmark for medical vision-language tasks
- Insights on cross-modal alignment in healthcare

**Relevance to Project:**
- Direct methodology for vision-language fusion
- Guidance on aligning imaging with clinical text
- Benchmark for evaluating multimodal models
- Foundation for explainability through language

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **9/10** |
| Data Foundation | 7/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 9/10 |
| Clinical Relevance | 9/10 |

**Primary Objectives:** 3 (LLM Integration), 4 (Video Generation), 5 (Evaluation)

---

### Reference 14: Denoising Diffusion Probabilistic Models (DDPM)
**Paper:** Ho et al. "Denoising Diffusion Probabilistic Models" NeurIPS, 2020

**Summary:**
DDPM establishes the modern framework for diffusion-based generative models, achieving state-of-the-art image quality through progressive denoising.

**Key Contributions:**
- Forward diffusion: gradual noise addition over T steps
- Reverse process: learned denoising for generation
- FID 3.17 on CIFAR-10 (unconditional)
- Connection to score matching and Langevin dynamics
- Progressive lossy compression interpretation
- Foundation for all modern diffusion models

**Relevance to Project:**
- Core generative framework for progression videos
- Foundation for understanding Video LDM
- Enables high-quality medical image synthesis
- Basis for conditional generation strategies

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 5/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 9/10 |
| Clinical Relevance | 7/10 |

**Primary Objectives:** 4 (Video Generation)

---

### Reference 15: Video Latent Diffusion Models (Video LDM)
**Paper:** Blattmann et al. "Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models" CVPR, 2023

**Summary:**
Video LDM extends latent diffusion models to video generation by adding temporal layers, achieving state-of-the-art video synthesis quality.

**Key Contributions:**
- Temporal layers inserted into pretrained image LDMs
- Video-finetuned autoencoders for reduced flickering
- Temporal interpolation for high frame rates (30 fps)
- UCF-101 FVD: 550.61, IS: 33.45
- MSR-VTT CLIPSIM: 0.2929
- Conditioning on text, bounding boxes, previous frames
- DreamBooth integration for personalized video

**Relevance to Project:**
- Primary architecture for cancer progression videos
- Direct implementation guide for video generation
- Temporal coherence strategies for medical sequences
- Conditioning mechanisms for clinical context

| Aspect | Score |
|--------|-------|
| **Helpfulness Score** | **10/10** |
| Data Foundation | 5/10 |
| Methodology Guidance | 10/10 |
| Direct Implementation | 10/10 |
| Clinical Relevance | 7/10 |

**Primary Objectives:** 4 (Video Generation), 5 (Evaluation)

---

## Reference-to-Objective Mapping

### Summary Table

| Reference | Obj 1: Data | Obj 2: ViT | Obj 3: LLM | Obj 4: Video | Obj 5: Eval | Overall |
|-----------|-------------|------------|------------|--------------|-------------|---------|
| **1. TCGA** | ★★★★★ | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | 9/10 |
| **2. BraTS** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | 10/10 |
| **3. LIDC-IDRI** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | 8/10 |
| **4. ViT** | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | 10/10 |
| **5. UNETR** | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | 10/10 |
| **6. Swin UNETR SS** | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | 10/10 |
| **7. Swin Transformer** | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | 9/10 |
| **8. CNN Ensembles** | ★★☆☆☆ | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★☆☆ | 6/10 |
| **9. BioBERT** | ★★☆☆☆ | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | 8/10 |
| **10. Med-PaLM** | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★★ | 9/10 |
| **11. MIMIC-III** | ★★★★★ | ★☆☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | 7/10 |
| **12. Med-VLP** | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | 9/10 |
| **14. DDPM** | ★☆☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★★★★ | ★★★★☆ | 10/10 |
| **15. Video LDM** | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★★ | 10/10 |

### Detailed Objective Mapping

#### Objective 1: Longitudinal Data Pipeline
**Primary References:**
- **TCGA (9/10):** Multi-cancer genomics foundation with clinical metadata
- **BraTS (10/10):** Standard brain tumor imaging dataset with longitudinal cases
- **LIDC-IDRI (8/10):** Lung CT dataset with radiologist annotations
- **MIMIC-III (7/10):** Clinical database for metadata integration

**Secondary References:**
- Swin UNETR SS: Pretraining data organization strategies
- UNETR: Medical imaging preprocessing guidance

#### Objective 2: Vision Transformer Representations
**Primary References:**
- **ViT (10/10):** Foundational transformer architecture
- **Swin Transformer (9/10):** Hierarchical, efficient transformer design
- **UNETR (10/10):** 3D medical image segmentation transformer
- **Swin UNETR SS (10/10):** Self-supervised pretraining for medical imaging

**Secondary References:**
- CNN Ensembles: Baseline comparison methodology
- BraTS: Evaluation dataset for segmentation

#### Objective 3: LLM Integration for Clinical Reasoning
**Primary References:**
- **BioBERT (8/10):** Biomedical language model for text encoding
- **Med-PaLM (9/10):** Clinical knowledge encoding and prompt engineering
- **Med-VLP (9/10):** Vision-language pretraining for medical domain

**Secondary References:**
- TCGA: Clinical metadata for multimodal fusion
- MIMIC-III: Clinical notes and structured variables

#### Objective 4: Cancer Progression Video Generation
**Primary References:**
- **DDPM (10/10):** Core diffusion framework
- **Video LDM (10/10):** Video generation architecture and implementation

**Secondary References:**
- Med-VLP: Conditional generation with clinical context
- LIDC-IDRI: Progression data for evaluation

#### Objective 5: Evaluation and Scientific Documentation
**Primary References:**
- **BraTS (10/10):** Established evaluation metrics and baselines
- **Med-PaLM (9/10):** Human evaluation framework for clinical AI
- **Video LDM (10/10):** FVD, FID, IS metrics for video evaluation

**Secondary References:**
- All datasets: Quantitative benchmark results
- CNN Ensembles: Baseline performance comparison

---

## Technical Implementation Roadmap

### Phase 1: Environment Setup (Week 0-1)
```
1. Compute Resources:
   - GPU cluster access (A100 preferred)
   - Storage for large medical datasets
   
2. Software Environment:
   - PyTorch 2.0+
   - MONAI (medical imaging)
   - HuggingFace Transformers
   - Diffusers library
   
3. Dataset Access:
   - TCGA data portal registration
   - BraTS challenge registration
   - LIDC-IDRI access request
```

### Phase 2: Data Pipeline (Weeks 1-4)
```
Implementation Order:
1. Download and organize raw data
2. Implement preprocessing (MONAI)
3. Create PyTorch DataLoaders
4. Validate longitudinal consistency
5. Generate EDA report
```

### Phase 3: Vision Transformers (Weeks 5-11)
```
Implementation Order:
1. Implement CNN baselines (ResNet, EfficientNet)
2. Adapt ViT for medical imaging
3. Implement Swin Transformer
4. Configure UNETR architecture
5. Self-supervised pretraining (Swin UNETR)
6. Extract and analyze embeddings
```

### Phase 4: Multimodal LLM Integration (Weeks 12-14)
```
Implementation Order:
1. Load pretrained BioBERT
2. Design multimodal fusion architecture
3. Implement cross-attention mechanisms
4. Design prompt templates
5. Generate and validate clinical narratives
```

### Phase 5: Video Generation (Weeks 15-18)
```
Implementation Order:
1. Implement DDPM for medical images
2. Add temporal attention layers (Video LDM)
3. Implement conditioning mechanisms
4. Video-finetune autoencoder
5. Generate progression videos
6. Implement counterfactual scenarios
```

### Phase 6: Evaluation (Weeks 19-20)
```
Evaluation Metrics:
- Segmentation: Dice, Hausdorff distance
- Classification: AUC, accuracy, F1
- Video Quality: FID, FVD
- Temporal Consistency: custom metrics
- Clinical Plausibility: expert evaluation
```

---

## Key Takeaways and Recommendations

### Critical Success Factors

1. **Start with Strong Baselines**
   - Implement CNN baselines in Phase 2 to establish benchmarks
   - Use pretrained weights when available (ImageNet, medical pretraining)

2. **Leverage Self-Supervised Pretraining**
   - Swin UNETR's pretraining strategy is crucial for limited labeled data
   - Implement all three tasks: inpainting, rotation, contrastive learning

3. **Domain Adaptation is Essential**
   - BioBERT significantly outperforms general BERT on medical tasks
   - Medical vision-language pretraining improves downstream performance

4. **Video LDM Architecture is Proven**
   - Temporal layers + pretrained image LDM is the most practical approach
   - Video-finetuned decoders reduce flickering artifacts

5. **Evaluation Must Be Multi-faceted**
   - Quantitative: FID, FVD, Dice, temporal consistency
   - Qualitative: Clinical plausibility, expert evaluation
   - Both are necessary for medical AI validation

### Potential Challenges and Mitigations

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| Limited longitudinal data | Self-supervised pretraining, data augmentation |
| Computational resources | Efficient architectures (Swin), mixed precision |
| Domain shift | Medical-specific pretraining, fine-tuning |
| Temporal coherence | Video LDM temporal layers, interpolation |
| Clinical validation | Collaborate with medical experts |

### Publication Potential

The project addresses a novel intersection of:
- Vision Transformers for medical imaging
- LLM-based clinical reasoning
- Diffusion models for disease progression

**Target venues:**
- Medical Imaging: MICCAI, IEEE TMI, Medical Image Analysis
- AI/ML: NeurIPS, ICML, CVPR
- Interdisciplinary: Nature Medicine, npj Digital Medicine

---

## Appendix: Quick Reference Cards

### Reference Priority by Objective

**For Objective 1 (Data):** Start with BraTS → TCGA → LIDC-IDRI

**For Objective 2 (ViT):** Start with ViT → Swin → UNETR → Swin UNETR SS

**For Objective 3 (LLM):** Start with BioBERT → Med-PaLM → Med-VLP

**For Objective 4 (Video):** Start with DDPM → Video LDM

**For Objective 5 (Eval):** Use metrics from all papers, especially BraTS, Video LDM, Med-PaLM

### Helpfulness Score Summary

| Score | References |
|-------|------------|
| 10/10 | BraTS, ViT, UNETR, Swin UNETR SS, DDPM, Video LDM |
| 9/10 | TCGA, Swin Transformer, Med-PaLM, Med-VLP |
| 8/10 | LIDC-IDRI, BioBERT |
| 7/10 | MIMIC-III |
| 6/10 | CNN Ensembles |

---

*Document prepared for Mitacs Globalink Research Award project at TÉLUQ University*
*Last updated: Research preparation phase*
