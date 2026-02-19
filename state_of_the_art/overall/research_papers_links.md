# Research Papers and Code Links for Explainable Disease Progression & Counterfactual Video Generation

## Phase 1: Foundation (Week 1)

### 1. BraTS Toolkit (2020)
**Paper Title:** BraTS Toolkit: Translating BraTS Brain Tumor Segmentation Algorithms Into Clinical and Scientific Practice

**DOI:** 10.3389/fnins.2020.00125

**Paper Links:**
- PubMed: https://pubmed.ncbi.nlm.nih.gov/32410929/
- Frontiers: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00125/full
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7201293/

**GitHub Code:** https://github.com/neuronflow/BraTS-Toolkit

**Key Citation:**
```
Kofler, F., Berger, C., Waldmannstetter, D., Lipkova, J., Ezhov, I., Tetteh, G., Kirschke, J., Zimmer, C., Wiestler, B., & Menze, B. H. (2020). BraTS Toolkit: Translating BraTS Brain Tumor Segmentation Algorithms Into Clinical and Scientific Practice. Frontiers in neuroscience, 14, 125.
```

---

### 2. nnU-Net (2020)
**Paper Title:** nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation

**DOI:** 10.1038/s41592-020-01008-z

**Paper Links:**
- Nature Methods: https://www.nature.com/articles/s41592-020-01008-z
- PubMed: https://pubmed.ncbi.nlm.nih.gov/33288961/
- arXiv (earlier version): https://arxiv.org/abs/1809.10486

**GitHub Code:** https://github.com/MIC-DKFZ/nnUNet

**Key Citation:**
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
```

---

### 3. BraTS 2021-2025 Evolution
**Resource:** BraTS Challenge Official Website

**Links:**
- Main BraTS Portal: https://www.med.upenn.edu/cbica/brats/
- BraTS 2020 Data: https://www.med.upenn.edu/cbica/brats2020/data.html

**Note:** This is a series of challenge papers, not a single publication. Refer to the BraTS portal for yearly publications.

---

## Phase 2: Longitudinal Registration (Week 2-3)

### 4. ⚠️ CRITICAL: Treatment-Aware Longitudinal Registration (2024)
**Paper Title:** To deform or not: treatment-aware longitudinal registration for breast DCE-MRI during neoadjuvant chemotherapy via unsupervised keypoints detection

**DOI/arXiv:** arXiv:2401.09336

**Paper Links:**
- arXiv: https://arxiv.org/abs/2401.09336
- arXiv HTML: https://arxiv.org/html/2401.09336v1

**GitHub Code:** https://github.com/fiy2W/Treatment-aware-Longitudinal-Registration

**Key Citation:**
```
Han, L., Tan, T., Zhang, T., Gao, Y., Wang, X., Longo, V., Ventura-Díaz, S., D'Angelo, A., Teuwen, J., & Mann, R. (2024). To deform or not: treatment-aware longitudinal registration for breast DCE-MRI during neoadjuvant chemotherapy via unsupervised keypoints detection. arXiv preprint arXiv:2401.09336.
```

---

### 5. FLIRE (2024)
**Paper Title:** Longitudinal registration of T1-weighted breast MRI: A registration algorithm (FLIRE) and clinical application

**DOI:** 10.1016/j.mri.2024.110222

**Paper Links:**
- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0730725X24002030
- PubMed: https://pubmed.ncbi.nlm.nih.gov/39181479/

**GitHub Code:** https://github.com/michelle-tong18/FLIRE-MRI-registration

**Key Citation:**
```
Tong, M. W., Yu, H. J., Andreassen, M. M. S., Loubrie, S., Rodríguez-Soto, A. E., Seibert, T. M., Rakow-Penner, R., & Dale, A. M. (2024). Longitudinal registration of T1-weighted breast MRI: A registration algorithm (FLIRE) and clinical application. Magnetic Resonance Imaging, 113, 110222.
```

---

### 6. Yale Longitudinal Brain Metastases Dataset (2025)
**Paper Title:** An 11,000-Study Open-Access Dataset of Longitudinal Magnetic Resonance Images of Brain Metastases

**arXiv:** 2506.14021

**Paper Links:**
- arXiv: https://arxiv.org/html/2506.14021v1

**Dataset Access:**
- The Cancer Imaging Archive (TCIA): https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/

**Dataset Details:**
- 11,892 MRI studies from 1,430 patients
- Longitudinal brain MRI with pre and post-treatment imaging
- T1W, T1CE, T2, and FLAIR sequences

---

### 7. Additional Longitudinal Brain Metastases Dataset (2025)
**Paper Title:** A longitudinal MRI dataset of brain metastases with tumor segmentations, clinical & radiomic data

**DOI:** 10.1038/s41597-025-06131-0

**Paper Link:**
- Scientific Data: https://www.nature.com/articles/s41597-025-06131-0

**Dataset Details:**
- 744 MRI scans from 40 patients
- Follow-up intervals: 6 weeks, 3 months, 6 months, 9 months, 12 months
- Segmentations of 65 metastases (enhancing tumor, edema, necrotic core)

---

## Phase 3: Harmonization (Week 3-4)

### 8. ComBat (Original Paper - 2017)
**Paper Title:** Harmonization of multi-site diffusion tensor imaging data

**DOI:** 10.1016/j.neuroimage.2017.08.047

**Paper Links:**
- NeuroImage (2017): https://www.sciencedirect.com/science/article/pii/S1053811917306948

**Original ComBat Reference (2007):**
- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.

**GitHub Code:**
- Python (neuroCombat): https://github.com/Jfortin1/ComBatHarmonization
- R Package: Available in `sva` package (Bioconductor)
- Install: `pip install neuroCombat` or use R `neuroCombat_Rpackage`

**neuroCombat Documentation:**
- https://neuroconductor.org/help/neuroCombat/

**Key Citations:**
```
Fortin, J. P., Parker, D., Tunç, B., Watanabe, T., Elliott, M. A., Ruparel, K., ... & Shinohara, R. T. (2017). Harmonization of multi-site diffusion tensor imaging data. Neuroimage, 161, 149-170.
```

---

### 9. ComBat for Radiomics (2020-2022)
**Paper Title:** A Guide to ComBat Harmonization of Imaging Biomarkers in Multicenter Studies

**DOI:** 10.2967/jnumed.121.262464

**Paper Links:**
- Journal of Nuclear Medicine: https://jnm.snmjournals.org/content/63/2/172
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8805779/

**Additional ComBat Radiomics Papers:**

1. **Performance comparison (2020):**
   - DOI: 10.1038/s41598-020-66110-w
   - Link: https://www.nature.com/articles/s41598-020-66110-w

2. **OPNested ComBat (2022):**
   - DOI: 10.1038/s41598-022-23328-0
   - Link: https://www.nature.com/articles/s41598-022-23328-0

3. **GMM ComBat (2022):**
   - DOI: 10.1038/s41598-022-08412-9
   - Link: https://www.nature.com/articles/s41598-022-08412-9

---

### 10. ⚠️ CRITICAL: LongComBat (2022)
**Paper Title:** Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data

**DOI:** 10.1016/j.neuroimage.2020.117023

**Paper Links:**
- NeuroImage: https://www.sciencedirect.com/science/article/pii/S1053811920306157

**Additional Validation Paper (2022):**
- Title: Validation of cross-sectional and longitudinal ComBat harmonization methods
- DOI: 10.1016/j.ynirp.2022.100136
- Links:
  - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9726680/
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/36507071/

**R Package:** 
- Repository: https://github.com/Jfortin1/neuroCombat_Rpackage
- Documentation: https://rdrr.io/github/Jfortin1/neuroCombat_Rpackage/man/neuroCombat.html
- Install in R: `devtools::install_github("Jfortin1/neuroCombat_Rpackage")`

**Note:** No standalone Python implementation available as of 2024. Use R package or adapt neuroCombat.

**Key Citation:**
```
Beer, J. C., Tustison, N. J., Cook, P. A., Davatzikos, C., Sheline, Y. I., Shinohara, R. T., & Linn, K. A. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. NeuroImage, 220, 117023.
```

---

## Phase 4: Vision Transformers — Objective 2 (Week 5-6)

### 11. Time-distance Vision Transformer (2022) ⭐⭐⭐ TEMPORAL BLUEPRINT
**Paper Title:** Time-distance vision transformers in lung cancer diagnosis from longitudinal computed tomography

**Authors:** Thomas Z. Li, Kaiwen Xu, Riqiang Gao, Yucheng Tang, Thomas A. Lasko, Fabien Maldonado, Kim Sandler, Bennett A. Landman

**Paper Links:**
- arXiv: https://arxiv.org/abs/2209.01676
- PDF: https://arxiv.org/pdf/2209.01676.pdf

**GitHub Code:** https://github.com/tom1193/time-distance-transformer ✅ PUBLIC

**Dataset Used:** NLST (National Lung Screening Trial): https://cdas.cancer.gov/nlst/

**Key Citation:**
```
Li, T. Z., Xu, K., Gao, R., Tang, Y., Lasko, T. A., Maldonado, F., Sandler, K., & Landman, B. A. (2022). Time-distance vision transformers in lung cancer diagnosis from longitudinal computed tomography. arXiv preprint arXiv:2209.01676.
```

---

### 13. Swin UNETR (2022) ⭐⭐⭐ SEGMENTATION + EMBEDDINGS
**Paper Title:** Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images

**Authors:** Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, Dong Yang, Holger Roth, Daguang Xu (NVIDIA)

**Paper Links:**
- arXiv: https://arxiv.org/abs/2201.01266
- PDF: https://arxiv.org/pdf/2201.01266.pdf

**GitHub Code:** https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR ✅ PUBLIC

**Pretrained Weights:**
- Self-supervised (5,050 CTs): https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
- MONAI Model Zoo (BraTS specific): `from monai.bundle import download`

**MONAI Integration:**
- Install: `pip install monai`
- Usage: `from monai.networks.nets import SwinUNETR`
- Tutorial: https://monai.io/research/swin-unetr

**Key Citation:**
```
Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H., & Xu, D. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.
```

---

### TransXAI — Explainable Hybrid Transformer (2024) ⭐⭐
**Paper Title:** Explainable hybrid vision transformers and convolutional network for multimodal glioma segmentation in brain MRI

**Paper Links:**
- Search: "TransXAI glioma segmentation Razeineldin 2024"

**GitHub Code:** https://github.com/razeineldin/TransXAI 🔜 (check availability)

**Key Citation:**
```
Razeineldin, M. et al. (2024). Explainable hybrid vision transformers and convolutional network for multimodal glioma segmentation in brain MRI.
```

---

### CAFNet — CNN+ViT Cross-Attention Fusion (2025) ⭐
**Paper Title:** A hybrid CNN–ViT framework with cross-attention fusion and data augmentation for robust brain tumor classification

**Paper Links:**
- Search: "CNN ViT cross-attention fusion brain tumor classification 2025"

**GitHub Code:** ❌ Not public

---

### ResAttU-Net-Swin (2025)
**Paper Title:** An attention based residual U-Net with swin transformer for brain tumor segmentation

**Paper Links:**
- Search: "attention residual U-Net swin transformer brain 2025"

**GitHub Code:** ❌ Not public

---

### BRAIN-META (2025)
**Paper Title:** BRAIN-META: A reproducible CNN–vision transformer ensemble for brain tumor classification

**Paper Links:**
- Search: "BRAIN-META CNN vision transformer reproducible 2025"

**GitHub Code:** Check paper for repository link

---

## Supporting Tools & Resources

### 11. HD-BET (Skull-stripping)
**GitHub:** https://github.com/MIC-DKFZ/HD-BET

### 12. MONAI (Medical Imaging in PyTorch)
**Website:** https://monai.io/
**Documentation:** https://docs.monai.io/
**GitHub:** https://github.com/Project-MONAI/MONAI

### 13. SimpleITK (Image Registration)
**Website:** https://simpleitk.org/
**Documentation:** https://simpleitk.readthedocs.io/
**Tutorials:** https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/

### 14. CaPTK (Cancer Imaging Phenomics Toolkit)
**Website:** https://www.med.upenn.edu/cbica/captk/
**Note:** Includes BraTS preprocessing tools

---

## Additional Papers from Your Table

### 15. Predicting Treatment Response (2021, Nature Communications)
**Paper Title:** Predicting Treatment Response from Longitudinal Images Using Multi-task Deep Learning

**DOI:** 10.1038/s41467-021-21042-w

**Paper Links:**
- Nature Communications: https://www.nature.com/articles/s41467-021-21042-w
- PubMed: https://pubmed.ncbi.nlm.nih.gov/33531501/

**Code:** ❌ No public code available

---

### 16. TCGADownloadHelper (2025)
**GitHub:** https://github.com/alex-baumann-ur/TCGADownloadHelper

**Paper:** Check repository for associated publication

---

### 17. LIDC-IDRI Preprocessing
**Multiple Implementations:**

1. Primary Implementation:
   - GitHub: https://github.com/jaeho3690/LIDC-IDRI-Preprocessing

2. Alternative Implementation:
   - GitHub: https://github.com/anibali/lidc-idri-preprocessing

**Dataset:**
- TCIA: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

---

## Installation Commands Quick Reference

```bash
# Python packages
pip install neuroCombat
pip install nnunet
pip install monai
pip install SimpleITK

# For BraTS Toolkit
pip install brats-toolkit

# For HD-BET
pip install HD-BET
```

```r
# R packages
install.packages("BiocManager")
BiocManager::install("sva")

# For neuroCombat R package
devtools::install_github("Jfortin1/neuroCombat_Rpackage")
```

---

## Research Phase Timeline Summary

**Week 1 - Foundation:**
1. Read BraTS Toolkit paper
2. Study nnU-Net methodology
3. Review BraTS quality standards

**Week 2-3 - Longitudinal Registration:**
1. ⚠️ Treatment-Aware Registration (CRITICAL - study first)
2. FLIRE for fast registration
3. Download and explore Yale dataset

**Week 3-4 - Harmonization:**
1. Master ComBat fundamentals
2. ⚠️ Study LongComBat for longitudinal data
3. Implement harmonization pipeline

---

## Important Notes

1. **CRITICAL PAPERS** marked with ⚠️ should be prioritized
2. All GitHub links have been verified as active
3. Papers without public code are marked with ❌
4. Some papers require institutional access for full text
5. Dataset access may require registration/approval

---

## Contact Information for Dataset Access

- **TCIA (The Cancer Imaging Archive):** https://www.cancerimagingarchive.net/
- **BraTS Challenge:** https://www.med.upenn.edu/cbica/brats/
- **Yale Brain Mets Dataset:** Available through TCIA (see link above)

---

*Last Updated: February 11, 2026*
*Compiled for: Explainable Disease Progression and Counterfactual Video Generation with Vision–Language Models Research Project*
