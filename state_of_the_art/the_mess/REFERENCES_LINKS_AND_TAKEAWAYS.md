# References: Direct Links + Takeaways for Our Project

This document expands `refrences.txt` into:
- a **direct link** to each reference (publisher / DOI / arXiv)
- a brief **what it contributes** summary
- **what we take from it** for our research
- **how it helps** our specific topic: longitudinal cancer imaging + ViTs/LVMs + LLM reasoning + diffusion/video generation + counterfactual trajectories

---

## 1) Weinstein, J. N., et al. *The Cancer Genome Atlas Pan-Cancer analysis project.* Nature Genetics (2013)
- **Paper link**: https://www.nature.com/articles/ng.2764
- **Data portal (useful)**: https://portal.gdc.cancer.gov/
- **What it contributes**:
  - Establishes TCGA Pan-Cancer as a large-scale, standardized resource for multi-cancer molecular characterization.
  - Popularizes cross-cancer comparison and reproducible community-scale benchmarks.
- **What we take from it**:
  - The idea of using **public, standardized datasets** and reporting results in a **reproducible** way.
  - A source (depending on cohort availability) of **clinical variables** that can condition progression modeling or support LLM explanations.
- **How it helps our project**:
  - Supports Objective 1 (dataset acquisition/curation) and Objective 3 (link imaging-derived features with clinical context).

---

## 2) Menze, B. H., et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI (2015)
- **Paper link**: https://ieeexplore.ieee.org/document/6975210/
- **Benchmark info (useful)**: https://www.med.upenn.edu/cbica/brats/
- **What it contributes**:
  - Defines a benchmark for brain tumor MRI segmentation with clear evaluation protocols.
  - Encourages comparable reporting across methods.
- **What we take from it**:
  - A strong starting point for **tumor segmentation baselines** (Phase 2/3 of your timeline).
  - Standard metrics and evaluation habits.
- **How it helps our project**:
  - Segmentation masks (or derived measurements like tumor volume) can become structured signals for:
    - longitudinal progression modeling
    - explainable trend summaries for the LLM
    - conditioning generative models (e.g., “tumor region changes over time”)

---

## 3) Armato, S. G., et al. *The Lung Image Database Consortium (LIDC) …* Medical Physics (2011)
- **Paper link (DOI page)**: https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.3528204
- **Dataset access (useful)**: https://www.cancerimagingarchive.net/collection/lidc-idri/
- **What it contributes**:
  - A widely used lung CT dataset with nodule annotations from multiple readers.
  - Demonstrates realistic label uncertainty / inter-reader variability.
- **What we take from it**:
  - Practical lessons about:
    - preprocessing CT volumes (spacing, intensity normalization)
    - handling noisy labels / variability
    - building robust baselines before advanced modeling
- **How it helps our project**:
  - Good candidate dataset for Objective 1 and Objective 2 (representation learning).
  - The annotation variability is relevant for Objective 5 (evaluation and uncertainty-aware interpretation).

---

## 4) Dosovitskiy, A., et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR (2021)
- **Paper link**: https://arxiv.org/abs/2010.11929
- **What it contributes**:
  - Introduces the Vision Transformer (ViT) paradigm: image patches as tokens + transformer encoder.
  - Shows strong performance with sufficient pretraining and transfer.
- **What we take from it**:
  - The core **ViT design choices**:
    - patch embedding
    - self-attention for global context
    - transfer learning strategy
- **How it helps our project**:
  - Direct foundation for Objective 2: learning tumor representations that can capture global context and morphology.

---

## 5) Hatamizadeh, A., et al. *UNETR: Transformers for 3D Medical Image Segmentation.* WACV (2022)
- **Paper link**: https://arxiv.org/abs/2103.10504
- **What it contributes**:
  - A transformer-based encoder for 3D medical segmentation paired with a U-Net-style decoder.
  - Strong demonstration of transformers on volumetric medical tasks.
- **What we take from it**:
  - A practical architecture blueprint for 3D tumor segmentation and 3D feature extraction.
  - Implementation patterns often used with MONAI-based pipelines.
- **How it helps our project**:
  - Helps produce high-quality tumor masks/features, enabling:
    - better longitudinal signals
    - interpretable measurements (volume, shape)
    - more grounded conditioning for generative progression

---

## 6) Tang, Y., et al. *Self-Supervised Pretraining of Vision Transformers for Medical Imaging.* (listed as CVPR 2022)
- **Likely match (medical 3D Swin self-supervised)**: https://arxiv.org/abs/2111.14791
- **What it contributes**:
  - Shows how self-supervised learning can pretrain transformer backbones on medical data to improve downstream performance.
- **What we take from it**:
  - A strategy to reduce dependence on large labeled datasets by learning general-purpose medical features first.
  - Ideas for proxy tasks / pretraining objectives that preserve anatomical structure.
- **How it helps our project**:
  - Supports Objective 2 by improving embeddings (especially when longitudinal labels are scarce).
  - Better embeddings make Objective 3 (LLM explanations) more stable and Objective 4 (generation conditioning) more meaningful.

*Note*: if you want, paste the DOI/arXiv you intended for #6 and I’ll update the link to the exact one.

---

## 7) Liu, Z., et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV (2021)
- **Paper link**: https://arxiv.org/abs/2103.14030
- **What it contributes**:
  - Efficient, scalable transformer backbone with hierarchical multi-scale features.
- **What we take from it**:
  - Swin as a practical backbone to compare against CNNs and ViTs.
  - Better scalability for high-resolution medical images.
- **How it helps our project**:
  - Strong candidate backbone for Objective 2 (representations) and Objective 1/2 pipelines where resolution is large.

---

## 8) Harangi, B. *Skin lesion classification with ensembles of deep convolutional neural networks.* Journal of Biomedical Informatics (2018)
- **Paper link**: https://www.sciencedirect.com/science/article/pii/S1532046418301618
- **What it contributes**:
  - Demonstrates that strong CNN baselines + ensembles can be highly competitive in medical image classification.
- **What we take from it**:
  - A reminder to build **solid baselines** (single model, ensemble, proper evaluation) before claiming transformer advantages.
- **How it helps our project**:
  - Supports Phase 2 (baseline vision models) so your comparisons in Phase 3 are credible.

---

## 9) Lee, J., et al. *BioBERT: a pre-trained biomedical language representation model …* Bioinformatics (2020)
- **Paper link**: https://academic.oup.com/bioinformatics/article/36/4/1234/5566506
- **What it contributes**:
  - Shows that domain-adapted language models (trained on biomedical corpora) outperform general models on biomedical NLP tasks.
- **What we take from it**:
  - The principle: **domain adaptation matters** for medical terminology and reasoning.
- **How it helps our project**:
  - Motivates using medical-domain language models and careful prompting for Objective 3 (clinical narratives).

---

## 10) Singhal, K., et al. *Large language models encode clinical knowledge.* Nature (2023)
- **Paper link**: https://www.nature.com/articles/s41586-023-06291-2
- **What it contributes**:
  - Benchmarks clinical knowledge (MultiMedQA) and analyzes strengths/limitations of medical LLMs.
  - Highlights safety, evaluation, and alignment issues.
- **What we take from it**:
  - A realistic approach to evaluating medical reasoning: not just accuracy, but also **harmfulness**, **calibration**, and **expert review**.
- **How it helps our project**:
  - Informs Objective 3/5: the LLM explanation layer must be evaluated for clinical plausibility and safety.

---

## 11) Johnson, A. E. W., et al. *MIMIC-III, a freely accessible critical care database.* Scientific Data (2016)
- **Paper link**: https://www.nature.com/articles/sdata201635
- **Dataset access (useful)**: https://physionet.org/content/mimiciii/
- **What it contributes**:
  - A large, open clinical database (structured + notes) with strong documentation, enabling reproducible research.
- **What we take from it**:
  - Patterns for working with structured clinical variables + text, and handling data governance/access.
- **How it helps our project**:
  - Supports Objective 3: many multimodal medical systems need robust handling of structured metadata and clinical narratives.

---

## 12) Li, X., et al. *Multi-modal Pre-training for Medical Vision-language Understanding and Generation …* arXiv (2023)
- **Paper link**: https://arxiv.org/abs/2306.06494
- **What it contributes**:
  - Systematic study of medical vision-language pretraining and evaluation.
  - Focus on aligning images and medical text (reports/captions) for understanding and generation tasks.
- **What we take from it**:
  - Methods and evaluation ideas for **image-text alignment**.
  - Common pitfalls: dataset bias, noisy reports, weak grounding.
- **How it helps our project**:
  - Directly informs Objective 3: connecting imaging evidence to LLM outputs in a more principled way.

---

## 14) Ho, J., et al. *Denoising Diffusion Probabilistic Models.* NeurIPS (2020)
- **Paper link**: https://arxiv.org/abs/2006.11239
- **What it contributes**:
  - The core diffusion framework used by many modern generative image/video models.
- **What we take from it**:
  - The basic training objective and sampling process.
  - The idea that generation can be framed as iterative denoising.
- **How it helps our project**:
  - Foundation for Objective 4 (generative modeling of progression), especially if you adapt diffusion to medical video.

---

## 15) Blattmann, A., et al. *Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models.* CVPR (2023)
- **Paper link**: https://arxiv.org/abs/2304.08818
- **What it contributes**:
  - Shows how latent diffusion can be extended to video with better temporal coherence and manageable compute.
- **What we take from it**:
  - Practical design patterns for video diffusion:
    - operating in latent space
    - strategies to stabilize temporal consistency
- **How it helps our project**:
  - Closely aligned with Objective 4: generating temporally coherent sequences (progression videos).

---

## 16) Frid-Adar, M., et al. *GAN-based synthetic medical image augmentation …* Neurocomputing (2018)
- **Paper link**: https://www.sciencedirect.com/science/article/abs/pii/S0925231218310749
- **What it contributes**:
  - Demonstrates that synthetic images can augment limited medical datasets and improve CNN performance.
- **What we take from it**:
  - A simple and practical justification for using generative methods when labeled data is limited.
- **How it helps our project**:
  - Supports the motivation for Objective 4 (generation) and helps frame why synthetic sequences can be useful (with careful evaluation).

---

## 17) Chen, R. J., et al. *Synthetic data in machine learning for medicine and healthcare.* Nature Biomedical Engineering (2021)
- **Paper link**: https://www.nature.com/articles/s41551-021-00751-8
- **What it contributes**:
  - A high-level review of synthetic data benefits and risks in healthcare (privacy, bias, evaluation, governance).
- **What we take from it**:
  - A checklist of concerns to address when presenting generated medical images/videos:
    - bias amplification
    - misuse risk
    - overclaiming clinical validity
- **How it helps our project**:
  - Directly supports Objective 5: framing evaluation, limitations, and ethical considerations of generated progression/counterfactual videos.

---

# Missing reference #13 (your list skips it)
Your `refrences.txt` goes from **12 to 14**, but the background in `project.txt` mentions diffusion models with citations like `[13,14]`.

- **Action to fix**:
  - Find where `[13]` is cited in `project.txt` and decide which paper it should refer to.
- **Common candidates for #13** (depending on what you want to cite):
  - The original diffusion/score lineage paper(s) (historical foundation)
  - A major modern diffusion improvement paper
  - A latent diffusion paper (often used in practice)

If you paste the exact sentence(s) where `[13]` appears, I can recommend the best match and provide the direct link.
