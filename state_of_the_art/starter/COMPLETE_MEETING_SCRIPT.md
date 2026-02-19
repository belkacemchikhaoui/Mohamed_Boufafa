# Explainable Disease Progression and Counterfactual Video Generation
## Complete Presentation Script & Speaker Notes

**Duration:** 25-30 minutes  
**Audience:** Supervisor meeting  
**Slides:** 32 (including title and references)  
**Tone:** "I found that..." — showing findings, not lecturing

---

## 📚 Quick Reference: Key Terms

| Term | Meaning | Quick Explanation |
|------|---------|-------------------|
| **MONAI** | Medical Open Network for AI | PyTorch framework for medical imaging (NVIDIA) |
| **Dice** | Dice Score | Segmentation accuracy (0-1, higher = better) |
| **SSIM** | Structural Similarity Index | Image quality metric (0-1, higher = better) |
| **FID** | Fréchet Inception Distance | Image realism (lower = better) |
| **ComBat** | Combating Batch Effects | Statistical harmonization method |
| **VAE** | Variational Autoencoder | Compresses images to latent space |
| **ViT** | Vision Transformer | Treats image patches like text tokens |

### 🧠 MRI Modalities (Yale Dataset)

| Modality | What It Shows |
|----------|---------------|
| **T1** | Basic anatomy, gray/white matter |
| **T1c** | **Active tumor lights up!** (contrast-enhanced) |
| **T2** | Edema (swelling), fluid = bright |
| **FLAIR** | Clearest tumor edges (fluid-attenuated) |

**Our input:** All 4 modalities + 3 tumor region masks = 7 channels

---

## Slide-by-Slide Script

---

### **SLIDE 1: Title Slide**
*Duration: 30 seconds*

**Script:**
> "So this is my overview of the state of the art for our project on Explainable Disease Progression and Counterfactual Video Generation. I've gone through the literature systematically — 15 core papers covering every phase of the pipeline. I'd like to walk you through what I found and get your feedback on the approach I'm proposing."

**Key Points:**
- You've done systematic research (not random)
- Inviting feedback from the start
- 15 papers = comprehensive review

---

### **SLIDE 2: Presentation Roadmap**
*Duration: 45 seconds*

**Script:**
> "I've organized everything into three parts. First, I'll recap the problem and objectives — just to make sure we're aligned on what we're trying to solve. Then I'll go through the datasets, because I found two that work really well together. And finally, the state of the art — 15 papers organized into 5 phases that map directly to the pipeline I'm proposing."

**Key Points:**
- Clear structure = you're organized
- "Make sure we're aligned" = checking understanding
- Datasets + papers = thorough coverage

**🎯 Question to Ask Supervisor:**
> "Is there any particular phase you'd like me to go into more detail on, or should I keep it balanced across all five?"

---

### **TRANSITION: Part I**
*Duration: 5 seconds*

*(No script needed — just transition slide)*

---

### **SLIDE 3: The Problem**
*Duration: 2 minutes*

**Script:**
> "So as I understand it, the core problem is this: current medical AI can do impressive things in isolation. Segmentation models like nnU-Net hit Dice scores of 0.85 to 0.90 on BraTS. LLMs can generate radiology reports. Diffusion models can create synthetic medical images.

> But what I noticed from the literature is there are three critical gaps. First, everything is static — models analyze one time-point and stop there. Second, it's all black-box — we get a segmentation mask but no explanation of *why* the model thinks this is a tumor. And third, there's no 'what-if' capability — doctors can't ask 'what happens if we use radiation instead of chemo?'

> From what I read about clinical workflows, what doctors actually need is to track changes over time, understand the AI's reasoning in plain language, and visualize potential futures to plan treatment.

> So our goal, as I understand it, is to build a system that can do all three: segment accurately using Vision Transformers, explain the findings using an LLM, and generate progression videos using diffusion. Does that match your understanding?"

**Key Points:**
- You acknowledge current AI is good (not dismissive)
- Three gaps identified from literature review
- Connected to clinical need (not just technical)
- Ending with confirmation question

**🎯 Questions to Ask Supervisor:**
> "From your experience, which of these three gaps — temporal modeling, explainability, or counterfactual generation — would you say is the highest priority for clinicians?"

> "Are there specific clinical scenarios where this would be most valuable? Like comparing treatment plans for a specific patient?"

---

### **SLIDE 4: Five Research Objectives**
*Duration: 1.5 minutes*

**Script:**
> "Based on our proposal and my reading, I broke the project down into five objectives. Let me know if I'm missing anything or if the priorities should be different.

> Objective 1 is getting the data pipeline set up — preprocessing and organizing the longitudinal data. Objective 2 is extracting temporal representations with Vision Transformers. Objective 3 is integrating an LLM to generate explanations. Objective 4 is the video generation with diffusion models. And Objective 5 is evaluation — both quantitative metrics and, ideally, clinical validation.

> For each objective, I found 2-4 papers that directly address it. So the 15 papers I'm covering today map one-to-one to specific steps in the pipeline — each paper is mentioned once, where it's used."

**Key Points:**
- Frame as YOUR breakdown (invites feedback)
- Five objectives = manageable chunks
- 15 papers = comprehensive but not overwhelming
- One paper per step = clean organization

**🎯 Question to Ask Supervisor:**
> "For the evaluation, what would you consider a success? Should we aim primarily for quantitative metrics like Dice and SSIM, or is clinical validation with radiologists more important?"

---

### **SLIDE 5: Complete Pipeline at a Glance**
*Duration: 1.5 minutes*

**Script:**
> "Here's the full pipeline at a glance before I go into the details. The idea is: raw MRI comes in from Yale or Cyprus, goes through preprocessing in Phase 1, feature extraction with Vision Transformers in Phase 2-3, gets harmonized to remove scanner effects, feeds into RadFM for explanation in Phase 4, and finally generates counterfactual videos in Phase 5.

> What I like about this structure is it's modular — we can train and evaluate each phase independently, and there are pretrained models available for most steps. The three outputs align with our objectives: 768-dimensional embeddings capture what the model sees, clinical text explains it in plain language, and the video shows potential futures.

> Does this overall flow make sense, or should I be thinking about it differently?"

**Key Points:**
- High-level overview before details
- Modularity = practical advantage
- Three outputs = three objectives
- Inviting feedback on architecture

**🎯 Question to Ask Supervisor:**
> "Would you recommend training end-to-end after individual phases work, or keeping it modular throughout? I'm thinking modular is more debuggable."

---

### **TRANSITION: Part II — Datasets**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 6: Why Longitudinal Data**
*Duration: 1.5 minutes*

**Script:**
> "Before I go into the papers, I want to talk about datasets, because this was actually one of the trickier parts. Our project specifically needs longitudinal data — the same patient scanned multiple times over months or years — so we can model progression and treatment response.

> What I found is that all the common datasets are static. BraTS is single time-point per patient. TCGA is mostly snapshots. LIDC-IDRI is lung CT with no follow-up. None of them support temporal modeling.

> So I looked for longitudinal brain MRI datasets and found two that were both released in 2025 — very recent. Yale is massive, 11,884 scans from 1,430 patients. Cyprus PROTEAS is smaller, 40 patients, but has expert-verified tumor labels and much denser temporal coverage.

> My strategy is to use Yale for training because of the scale, and Cyprus for validation because of the ground truth labels. Both are free and public, which is great. Does that approach make sense?"

**Key Points:**
- You identified a real constraint (longitudinal requirement)
- Did thorough research on existing datasets
- Found complementary solution (not just one)
- Strategic split: train vs validate

**🎯 Question to Ask Supervisor:**
> "Have you worked with either of these datasets before, or should I expect any particular challenges with Yale given it's multi-scanner over 20 years?"

---

### **SLIDE 7: Yale Dataset**
*Duration: 2 minutes*

**Script:**
> "Let me tell you what I found about Yale. This was published in *Scientific Data* in 2025 by Ramakrishnan et al. It's the largest public longitudinal brain MRI dataset ever released — specifically for brain metastases, which are secondary brain cancers from lung, breast, or other primary sites.

> The numbers are impressive: 1,430 patients, 11,884 total scans, average 8.3 scans per patient over time. The data spans 20 years — 2004 to 2023 — and includes all four standard MRI modalities. Most scans are from Siemens scanners, about 86%, with GE making up 13%. Mix of 3T and 1.5T field strengths.

> The key thing for us is those 8.3 visits per patient on average — that gives us the temporal sequences we need. They also have clinical metadata including pre- and post-treatment scans.

> But there are two challenges we'll need to solve. First, Yale has no tumor segmentation labels — we'll need to generate those ourselves using nnU-Net. Second, with 20 years of data and multiple scanners, there are at least 5 batch effects we'll need to harmonize — that's where the Nested ComBat paper comes in.

> The dataset is free from TCIA — The Cancer Imaging Archive."

**Key Points:**
- Cite the paper properly (journal + authors)
- Understand the clinical context (brain metastases)
- Aware of both advantages (scale) and challenges (labels, harmonization)
- Know where to get it

**🎯 Question to Ask Supervisor:**
> "Do you think 8.3 visits per patient is enough temporal resolution to train a progression model, or should I look for additional data with denser follow-up?"

---

### **SLIDE 8: Cyprus PROTEAS Dataset**
*Duration: 1.5 minutes*

**Script:**
> "The second dataset I found is Cyprus PROTEAS by Trimithiotis et al., also 2025, from Zenodo. This one is much smaller — 40 patients, 744 scans — but what makes it valuable is the annotations.

> First, the temporal coverage is actually denser — 18.6 scans per patient on average, more than double Yale. Second, they have expert neuroradiologist-verified segmentations for 65 tumors, including the 3 subregions we care about: enhancing tumor, necrotic core, and edema. Third, they include radiotherapy plans and survival data.

> My thinking is this serves as our external validation set. We train on Yale's 1,430 patients, then test on Cyprus's different scanners — they use Philips and Siemens, different from Yale's mix — and evaluate against those expert labels. Target would be Dice above 0.85 to match nnU-Net's performance on BraTS.

> The other advantage is we can validate the video generation on Cyprus's denser temporal coverage — with 18.6 scans per patient, we can withhold later time-points and see if our model predicts them accurately."

**Key Points:**
- Understand the complementary role (validation, not training)
- Know the specific advantages (expert labels, radiotherapy plans)
- Strategic thinking (external validation proves generalization)
- Concrete target (Dice > 0.85)

**🎯 Question to Ask Supervisor:**
> "Should I prioritize getting Cyprus working first since it's already preprocessed and has labels, then scale to Yale? Or go straight to Yale for the bigger dataset?"

---

### **SLIDE 9: Dataset Comparison**
*Duration: 1 minute*

**Script:**
> "Here's a direct comparison table I put together. Yale wins on scale — 1,430 patients versus 40. Cyprus wins on annotations — it has expert tumor labels, 110 pre-extracted radiomic features, radiotherapy plans, and survival data, all of which Yale doesn't have.

> The strategic split is clear: train on Yale for the large-scale patterns, validate on Cyprus for ground truth accuracy and cross-scanner generalization. The fact that Cyprus uses Philips scanners while Yale is mostly Siemens/GE is actually a feature, not a bug — it forces our harmonization to work across vendors.

> One thing I considered was merging them, but Cyprus is too small to meaningfully add to training, and merging would invalidate the external validation. So I'm keeping them separate."

**Key Points:**
- Created your own analysis (comparison table)
- Strategic decision-making (train vs validate)
- Understand tradeoffs (scale vs annotations)
- Thought through alternatives (merging)

---

### **TRANSITION: Part III — State of the Art**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 10: Paper Map**
*Duration: 1 minute*

**Script:**
> "Now for the 15 papers. I organized them by which phase of the pipeline they serve. Phase 1 has the two datasets plus 3 preprocessing papers. Phase 2-3 is Vision Transformers plus harmonization — 5 papers. Phase 4 is LLM integration — 2 papers, though one is optional. Phase 5 is video generation — 3 papers.

> Each paper is mentioned exactly once, where it's used. So when I talk about Swin UNETR, that's our feature extractor. TaDiff is our video generation blueprint. And so on. I tried to keep it clean so there's no confusion about which paper does what."

**Key Points:**
- Well-organized (not random literature review)
- One paper, one purpose (clear roles)
- 15 papers = comprehensive coverage

---

### **TRANSITION: Phase 1**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 11: BraTS Toolkit**
*Duration: 1.5 minutes*

**Script:**
> "Phase 1 is preprocessing. The first paper is BraTS Toolkit by Kofler et al., *Frontiers in Neuroscience* 2020. This is a standardized preprocessing pipeline specifically for brain tumor MRI.

> What it does: co-registers the 4 MRI modalities within a single visit, does skull stripping using HD-BET — which is AI-based, much better than old threshold methods — and resamples everything to 1mm isotropic resolution.

> The great thing is it's one command. We point it at Yale's raw scans, and it gives us clean, aligned, skull-stripped volumes. Since Yale is already in NIfTI format, we can skip the DICOM conversion step they mention.

> One important thing I learned: BraTS Toolkit aligns modalities *within* one visit. To align *across* visits for the same patient — like registering their January scan to their March scan — we need itk-elastix, which is Paper 5. So these two tools solve different parts of the registration problem."

**Key Points:**
- Cite paper properly (authors, journal, year)
- Understand what it does AND doesn't do
- Practical advantage (one command)
- Clear about limitations (within-visit only)

**🎯 Question to Ask Supervisor:**
> "Should I run BraTS Toolkit on all 11,884 Yale scans right away, or start with a smaller subset to make sure the preprocessing works as expected?"

---

### **SLIDE 12: nnU-Net**
*Duration: 2 minutes*

**Script:**
> "The second paper is nnU-Net by Isensee et al., *Nature Methods* 2021. This is probably the most famous medical segmentation paper — it's a self-configuring framework that automatically adapts its architecture and training to any dataset.

> Their results on BraTS 2021 are impressive: Dice 0.85 for enhancing tumor, 0.88 for tumor core, 0.91 for whole tumor. They won BraTS 2020 and 2021. And they provide BraTS-pretrained weights, which is huge for us.

> Here's why we need nnU-Net: Yale has no tumor labels. So we're going to use nnU-Net's BraTS-pretrained model to generate 3-region segmentation masks — enhancing, necrotic, edema — for all 11,884 scans. Then we'll validate those pseudo-labels on Cyprus, where we have expert ground truth. Target is Dice above 0.85, matching nnU-Net's BraTS performance.

> The other reason I'm including nnU-Net is it serves as our CNN baseline. Later when we switch to Swin UNETR in Phase 3, we can compare: does the Vision Transformer actually beat the CNN? That's an empirical question we can answer."

**Key Points:**
- Understand the method (self-configuring)
- Know the results (Dice scores)
- Clear on our use case (generate labels for Yale)
- Forward-thinking (baseline for comparison)

**🎯 Question to Ask Supervisor:**
> "Do you think it's valid to treat nnU-Net's outputs on Yale as pseudo-labels for training, or should I only use Cyprus's expert labels and keep Yale unlabeled for self-supervised learning?"

---

### **SLIDE 13: itk-elastix**
*Duration: 1.5 minutes*

**Script:**
> "The third Phase 1 paper is itk-elastix by Niessen et al., *Journal of Open Source Software* 2023. This is a Python wrapper for Elastix, which is the gold standard for medical image registration.

> What it does: aligns brain scans across different visits for the same patient. First rigid registration to correct head position, then B-spline deformable registration to correct brain shift and subtle anatomical changes. It outputs a deformation field per voxel.

> Why we need this: each Yale patient has about 8 visits over time. If we don't align them to a common space — say, their first visit — we can't tell whether a voxel changed because the tumor grew or because their head was tilted differently in the scanner. Registration removes that geometric variability so we only see the biological changes.

> I chose itk-elastix over FLIRE because FLIRE is MATLAB-based with no Python API, and itk-elastix is pip-installable and actively maintained. It integrates cleanly with our PyTorch pipeline."

**Key Points:**
- Understand the need (cross-visit alignment)
- Know the method (rigid + deformable)
- Justified the choice (vs alternatives)
- Practical consideration (Python integration)

**🎯 Question to Ask Supervisor:**
> "Should I register every visit to the first time-point, or to the previous visit? I'm thinking first time-point gives us absolute changes, but previous visit might be more stable."

---

### **TRANSITION: Phase 2-3**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 14: Swin UNETR**
*Duration: 2 minutes*

**Script:**
> "Now Phase 2-3, Vision Transformers. The core paper here is Swin UNETR by Tang et al., CVPR 2022. This is a hierarchical Swin Transformer encoder with a U-Net decoder, designed for 3D medical segmentation.

> Their results on BraTS: Dice 0.9005, which is state-of-the-art. It has 62 million parameters — smaller than the original UNETR which has 92 million. The embedding dimension is 768, which is going to be important later because that matches RadFM perfectly.

> What really sold me on Swin UNETR is the pre-training. Tang et al. trained it on 5,050 CT scans using self-supervised learning, and they released the weights in MONAI. So our approach is: load MONAI's pretrained weights, fine-tune on Yale brain MRI, then use the encoder only — just the feature extractor — to get 768-dimensional embeddings per scan.

> The hierarchical structure is also good for us because brain tumors appear at multiple scales. Small metastases might be 5mm, large ones could be 50mm. Swin UNETR's multi-scale features should capture both."

**Key Points:**
- Cite paper and venue (CVPR = top tier)
- Understand the architecture (encoder-decoder, hierarchical)
- Know the numbers (Dice, parameters, embedding dim)
- Clear transfer learning plan (MONAI → Yale)
- Connect to problem (multi-scale tumors)

**🎯 Question to Ask Supervisor:**
> "The MONAI weights are from CT scans. Should I fine-tune on the Yale brain MRI first, or try the pretrained weights directly to see if they transfer?"

---

### **SLIDE 15: TaViT**
*Duration: 2 minutes*

**Script:**
> "The second Phase 2-3 paper is TaViT by Hager et al., 2022. This paper was a real eye-opener for me. What they showed is: if you take a standard Vision Transformer and give it a sequence of medical images over time, it performs at chance level — AUC 0.50, basically random.

> But if you add time-distance positional embeddings — encoding the actual number of days between scans using sinusoidal functions — performance jumps to AUC 0.786. That's a 57% relative improvement. So time awareness isn't optional, it's essential.

> For our project, here's how I'm thinking we integrate this: after Swin UNETR gives us 768-dimensional embeddings, we compute the time gaps between visits — say, 0 days for the baseline scan, 90 days for the 3-month follow-up, 180 days for 6 months, and so on. We encode those gaps with TaViT's sinusoidal functions and add them to the embeddings. That gives us time-aware representations.

> One adaptation we'll need: TaViT was designed for lung CT with annual screening intervals. We're dealing with brain MRI with scans every 2-3 months. So the time scales are different, but the core idea should transfer."

**Key Points:**
- Understand the key finding (time encoding essential)
- Know the numbers (0.50 → 0.786)
- Clear integration plan (add to Swin UNETR embeddings)
- Aware of adaptations needed (different time scales)

**🎯 Question to Ask Supervisor:**
> "Do you think the time encoding should be learned from data, or should I use TaViT's fixed sinusoidal functions? I'm leaning toward learned because our time intervals might be irregular."

---

### **SLIDE 16: Harmonization Overview**
*Duration: 1.5 minutes*

**Script:**
> "Before I go into the harmonization papers, let me explain why Yale needs special treatment. The same tumor imaged on different scanners can look completely different in the feature space. Yale has at least 5 batch effects: year — 2004 to 2023, protocols changed; field strength — 1.5T versus 3T; manufacturer — Siemens versus GE; hospital site — different locations; and hidden patterns that we can discover with Gaussian Mixture Models.

> Where does harmonization happen in the pipeline? After TaViT, not before. Here's why: time encoding needs to be added to the raw Swin UNETR embeddings first. Then we clean the scanner noise. If we harmonized before TaViT, we'd be smoothing out information that the time encoding needs.

> My solution uses two papers in sequence: Nested ComBat from Paper 6 removes the 5 batch effects one by one, largest first. Then Longitudinal ComBat from Paper 7 fixes remaining scanner jumps within each patient's timeline.

> What's novel here is no existing work combines Nested + Longitudinal ComBat applied to Vision Transformer embeddings. Previous work used radiomics. So this is a methodological contribution."

**Key Points:**
- Identified the problem (5 batch effects)
- Clear on pipeline order (after TaViT)
- Two-step solution (Nested + Longitudinal)
- Aware of novelty (ViT embeddings, not radiomics)

---

### **SLIDE 17: Generalized ComBat (Nested + GMM)**
*Duration: 2 minutes*

**Script:**
> "Let me go into the first harmonization paper. Generalized ComBat by Horng et al., *Scientific Reports* 2022. They extend standard ComBat in two ways: Nested and GMM.

> Nested means you remove batch effects sequentially instead of all at once. For Yale, we'd do: Year first — that's 20 years of protocol changes, probably the biggest effect. Then field strength. Then manufacturer. Then site. We clean one, move to the next, clean that, and so on.

> GMM — Gaussian Mixture Models — automatically discovers hidden confounds. Like maybe one scanner had a software upgrade mid-study that we don't know about. GMM will find it.

> Their results show this is 10-11% better than standard ComBat. They tested it on survival prediction — c-statistic of 0.63 versus 0.59 for non-harmonized data.

> Here's how I'm planning to use it: after TaViT gives us time-aware embeddings, I'll run a Kruskal-Wallis test for each batch effect. If p < 0.05, run ComBat on that effect. If not significant, skip it. This is important because you don't want to over-harmonize — if there's no real scanner effect, don't remove it."

**Key Points:**
- Understand both extensions (Nested + GMM)
- Know the improvement (10-11% better)
- Statistical approach (Kruskal-Wallis test first)
- Aware of over-harmonization risk

**🎯 Question to Ask Supervisor:**
> "Do you think I should run the harmonization on the full 768-dimensional embeddings, or project them to a lower dimension first? I'm worried about overfitting with 768 dimensions and limited data per batch."

---

### **SLIDE 18: Longitudinal ComBat**
*Duration: 2 minutes*

**Script:**
> "The second harmonization paper is Longitudinal ComBat by Beer et al., *NeuroImage* 2020. This is specifically for repeated measures — when the same patient is scanned multiple times.

> Here's the problem it solves. Imagine a patient scanned on 3 different scanners over time: Scanner A → 0.5, Scanner B → 0.7, Scanner C → 0.9. Did the tumor grow, or did the scanner change? Nested ComBat treats each scan independently, so it might accidentally erase the real tumor growth while removing scanner noise.

> Longitudinal ComBat knows these scans are from the same patient. It uses random effects: each patient has their own baseline and growth rate. The model only removes scanner noise while preserving the patient-specific trajectory.

> Their test dataset was 663 patients scanned on 126 different scanners from the ADNI Alzheimer's study. They showed it gives more statistical power than cross-sectional ComBat.

> For us, this is Step 2 of 2. After Nested ComBat removes the global effects like Year and Field Strength, Longitudinal ComBat fixes per-patient scanner jumps. The combination should give us clean embeddings that reflect real tumor changes, not scanner variability."

**Key Points:**
- Understand the specific problem (repeated measures)
- Clear on what it preserves (per-patient trajectory)
- Know the test case (ADNI, 126 scanners)
- Sequential application (after Nested)

**🎯 Question to Ask Supervisor:**
> "The Longitudinal ComBat paper uses R. Should I call it via rpy2, or is there a Python port you'd recommend? I want to keep everything in one language if possible."

---

### **TRANSITION: Phase 4**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 20: RadFM**
*Duration: 2.5 minutes*

**Script:**
> "Now Phase 4, LLM integration. The main paper here is RadFM by Wu et al., 2025. This is the first radiology foundation model that handles both 2D and 3D images plus text. Trained on 16 million radiology images — that's the largest I found.

> Let me explain their original architecture versus what we're doing. Originally, RadFM takes a 2D image, expands it to pseudo-3D, runs it through a ViT-3D encoder, pools it with a Perceiver to get 32 tokens, then feeds those into MedLLaMA-13B to generate the report.

> Our modification: instead of their 2D-to-pseudo-3D ViT, we plug in Swin UNETR. We already have 768-dimensional embeddings from Phase 3, so we feed those directly into RadFM's Perceiver. The Perceiver compresses them to 32 query tokens, which go to MedLLaMA. Everything downstream — the Perceiver and the language model — we keep unchanged.

> Why is Swin UNETR better here? First, it's native 3D, not pseudo-3D. Second, it's pre-trained on brain tumors from BraTS, not general radiology. Third, it takes 7 channels — 4 MRI modalities plus 3 tumor masks — as input, so it sees both the tumor and its subregions. Fourth, it's more efficient because of the shifted window attention.

> RadFM is fully open-source — MIT license. The code is on GitHub, weights on HuggingFace, pip-installable. That's a huge advantage because we can fine-tune it on our domain.

> Here's an example output I saw in their paper: 'Enhancing tumor decreased 17%, necrotic core increased 400% — treatment-induced necrosis — edema increased 67%. Consistent with low-grade glioma, good prognosis.' That's the level of detail we're aiming for."

**Key Points:**
- Understand original architecture
- Clear on our modification (Swin UNETR replacement)
- Justified why it's better (4 reasons)
- Know it's open-source (practical advantage)
- Concrete example output (shows understanding)

**🎯 Questions to Ask Supervisor:**
> "For fine-tuning RadFM, do we have access to any radiology reports from Yale that we could use as supervision, or should I look into self-supervised objectives?"

> "The Perceiver compresses from 768-dim to 32 tokens. That's aggressive. Should I try different numbers of query tokens, or stick with their 32?"

---

### **TRANSITION: Phase 5**
*Duration: 5 seconds*

*(No script needed)*

---

### **SLIDE 22: Paper 13 — LDM (Latent Diffusion Models)**
*Duration: 1.5 minutes*

**Script:**
> "Now Phase 5, video generation. The first thing I needed to figure out was — how do you even run diffusion on brain MRI? Our volumes are 4 modalities × 128 cubed = 8 million values. That's way too big to work with directly.

> What I found is the Latent Diffusion paper by Rombach et al., CVPR 2022 — this is the foundation behind Stable Diffusion, over 10,000 citations. The key idea is simple: train a VAE to compress images to a tiny latent space — 8× smaller per dimension — then do all the diffusion work in that compressed space.

> For us, that means we compress each scan from (4, 128³) down to (4, 16³), which is 512× smaller. We train the VAE once on all 11,884 Yale scans, freeze it, and never touch it again. All the denoising happens in that tiny 16-cubed latent space.

> The other thing we take from LDM is cross-attention — that's how you inject conditioning into the UNet. In their case it's text prompts like 'a dog wearing a hat.' In our case it's the RadFM text narrative from Phase 4 — like 'tumor growing rapidly, high-grade features.'

> The code is fully open-source from CompVis on GitHub."

**Key Points:**
- Identified the problem (8M values too big)
- Understand the solution (latent space diffusion)
- Clear integration plan (VAE on Yale, cross-attention for text)
- Open-source code

**🎯 Question to Ask Supervisor:**
> "For the VAE, should I train it on all 11,884 scans including the masks as separate channels, or just the 4 MRI modalities? I'm thinking all 7 channels so the VAE learns tumor structure."

---

### **SLIDE 23: Paper 14 — TaDiff (Treatment-Aware Diffusion)**
*Duration: 2.5 minutes*

**Script:**
> "This is the most important paper I found. TaDiff by Liu et al., published in IEEE TMI 2025. It's the closest thing to what we want to build.

> What they do: take 3 past brain scans, concatenate them as channels alongside a noisy future scan, and train a UNet to predict what the future scan should look like. But here's the key part — they also feed in treatment information. They embed the treatment type — like chemo or radiation — and the number of days between scans, create difference vectors, and inject those into the UNet.

> The results show it really matters: with treatment info, they get SSIM of 0.919 and a tumor Dice of 0.719. Without treatment info, Dice drops to 0.556. That's a 29% improvement just from telling the model what treatment the patient received.

> They also have two output heads: one predicts the noise — that's standard diffusion — and one predicts the 3-region tumor masks directly. So the model generates both the image and the segmentation jointly. That joint training helps: the segmentation task forces the UNet to understand tumor anatomy, which makes the image generation better.

> Two other clever things: first, they weight the loss 5× higher on tumor voxels, so the model focuses on getting the tumor right, not just the healthy brain. Second, for uncertainty, they generate 5 stochastic samples and compute the standard deviation — that gives doctors a confidence map.

> For counterfactual generation, they just swap the treatment label. 'What if we used radiation instead of chemo?' Same past scans, different treatment embedding, different video output.

> Now, here's where we improve on TaDiff. They only had 23 patients with gliomas — primary brain cancers. We have 1,430 patients with brain metastases. They worked on 2D slices. We're doing full 3D volumes. They had 3 MRI types. We have 4. And we're adding RadFM text conditioning on top, which they didn't have.

> One downside: they didn't release the code. So we'll need to implement this from the paper. But the methods section is detailed enough."

**Key Points:**
- Understand the method (3 past + treatment + future)
- Know the results (0.919 SSIM, 0.719 Dice)
- Understand joint prediction (image + masks)
- Aware of tricks (5× tumor weight, uncertainty)
- Clear on counterfactual (swap treatment)
- Know our improvements (scale, 3D, +text)
- Acknowledge limitation (no code)

**🎯 Questions to Ask Supervisor:**
> "Do you think we can get away with 3 past scans like TaDiff, or should we try feeding in more given Yale has 8.3 scans per patient on average?"

> "For the treatment embedding, should I include dose and schedule information, or just treatment type? Yale's metadata might have that."

---

### **SLIDE 24: Paper 15 — EchoNet-Synthetic**
*Duration: 2 minutes*

**Script:**
> "The third Phase 5 paper is EchoNet-Synthetic by Reynaud et al., MICCAI 2024. This is the first medical video diffusion model with full open-source code and weights. They trained it on echocardiograms — ultrasound videos of the heart.

> Now, we're not interested in the heart part, and we're definitely not interested in the 'synthetic' part where they generate fake datasets. What we're taking from EchoNet is three technical pieces.

> First, the VAE architecture. They use a 3D VAE with a triple loss: MSE for pixel reconstruction, LPIPS for perceptual similarity, and KL divergence for regularization. That triple loss is better than just MSE.

> Second, temporal attention. They take a pre-trained image diffusion model, freeze the spatial layers, and add new temporal layers. The temporal layers let frames look at their neighbors in time, which prevents flickering and gives you smooth video. They only train 20% of the parameters — just the temporal layers — which is really efficient.

> Third, video stitching. They generate 64-frame chunks with 50% overlap, then stitch them together to make videos of any length. They showed they could generate 19,200 frames in 14 minutes on a single GPU.

> Their metrics: FID of 28.8, which means realistic-looking images. 64 frames in 2.4 seconds, versus 279 seconds for prior work. That's a huge speedup.

> For us, the plan is: clone their repo, replace their 2D echo VAE with our 3D brain MRI VAE from the LDM step, replace their heartbeat UNet with the TaDiff treatment-aware UNet, keep the temporal attention and video stitching code exactly as-is, and add a 3-region mask output head.

> The code is on GitHub, MIT license, includes pretrained weights. So this is our starting point for Phase 5 implementation."

**Key Points:**
- Clear what you're taking (3 components) vs not taking (synthetic)
- Understand each component (VAE, temporal attention, stitching)
- Know the results (FID, speed)
- Clear integration plan (what to replace vs keep)
- Open-source advantage

**🎯 Question to Ask Supervisor:**
> "Should I start by getting their echo video generation working first to understand the codebase, then adapt it to brain MRI? Or jump straight to brain MRI?"

---

### **SLIDE 25: Complete Pipeline (Detailed)**
*Duration: 2 minutes*

**Script:**
> "Now let me show you the complete detailed pipeline, all 5 phases connected. This diagram shows what goes in, what comes out, and how the papers fit together.

> Phase 1: Yale raw MRI comes in. BraTS Toolkit does skull stripping. nnU-Net generates the 3-region masks. itk-elastix aligns across visits. Output: 4 MRI modalities + 3 masks = 7 channels, spatially registered over time.

> Phase 2-3: Swin UNETR takes all 7 channels and produces 768-dimensional embeddings. TaViT adds time encoding. ComBat — Nested then Longitudinal — harmonizes away scanner effects. Output: clean, time-aware 768-dim features.

> Phase 4: RadFM's Perceiver compresses to 32 tokens. MedLLaMA generates clinical text — 512-dimensional embedding. Output: narrative like 'enhancing tumor decreased, necrosis increased, treatment response likely.'

> Phase 5: Two parallel streams. The registered scans from Phase 1 go into a 3D VAE, compressed to latent space. The text from Phase 4 goes into TaDiff UNet as cross-attention conditioning. TaDiff takes 3 past latent scans, treatment info, and text, generates future latent scan plus 3-region masks. Temporal attention smooths across frames for video. Output: progression video with per-frame masks and uncertainty maps.

> The three boxes at the bottom summarize: Input is 4 MRI + 3 masks per scan. Conditioning is past scans + treatment + RadFM text. Output is video + 3-region masks + uncertainty.

> Does this pipeline make sense? Any steps you'd change or add?"

**Key Points:**
- Walk through each phase systematically
- Show what flows between phases
- Three outputs match three objectives
- Inviting feedback

**🎯 Question to Ask Supervisor:**
> "The VAE and the Swin UNETR both take the MRI as input, separately. Should I share weights between them, or keep them independent? I'm thinking independent for modularity."

---

### **SLIDE 26: References**
*Duration: 30 seconds*

**Script:**
> "Here are all 15 papers with full citations and DOI links. Everything is either published in peer-reviewed journals — Nature Methods, IEEE TMI, Scientific Data — or top-tier conferences like CVPR and MICCAI. Most have open-source code, which is critical for reproducibility.

> I can send you this slide deck and the list of papers if you'd like to dig into any of them more."

---

## 📊 Summary of Key Questions to Ask Supervisor

### **Dataset & Preprocessing:**
1. Should I start with Cyprus (smaller, labeled) or Yale (larger, unlabeled)?
2. Is 8.3 visits per patient enough temporal resolution?
3. BraTS Toolkit: run on all 11,884 scans or test on subset first?
4. itk-elastix: register to first visit or previous visit?
5. nnU-Net pseudo-labels: valid for training or only use Cyprus expert labels?

### **Vision Transformers:**
6. Swin UNETR MONAI weights (from CT): fine-tune first or use directly?
7. TaViT time encoding: learned or fixed sinusoidal?
8. Harmonization: full 768-dim or project to lower dimension first?
9. Longitudinal ComBat: use R via rpy2 or find Python port?

### **LLM Integration:**
10. RadFM fine-tuning: do we have Yale radiology reports for supervision?
11. Perceiver query tokens: keep 32 or experiment with different numbers?

### **Video Generation:**
12. VAE training: 4 MRI modalities only or include 3 masks (7 channels)?
13. TaDiff: 3 past scans like the paper, or more given Yale's 8.3 average?
14. Treatment embedding: just type or include dose/schedule?
15. EchoNet: validate on echo first or jump to brain MRI?
16. VAE vs Swin UNETR: share weights or independent?

### **Evaluation:**
17. Success criteria: quantitative metrics or clinical validation more important?
18. Training: end-to-end after phases work, or stay modular?

---

## ⏱️ Time Management

| Section | Duration |
|---------|----------|
| Slides 1-5 (Overview) | 7 min |
| Slides 6-9 (Datasets) | 6 min |
| Slides 11-13 (Phase 1) | 5 min |
| Slides 14-18 (Phase 2-3) | 9 min |
| Slide 20 (Phase 4) | 3 min |
| Slides 22-24 (Phase 5) | 6 min |
| Slide 25 (Pipeline) | 2 min |
| Slide 26 (References) | 1 min |
| **Total** | **~30 min** |
| Questions & Discussion | +10-15 min |

---

## 🎯 Key Takeaways to Emphasize

1. **Systematic Coverage:** 15 papers, each serves one specific purpose
2. **Modularity:** Can train and evaluate each phase independently
3. **Pretrained Models:** Leverage MONAI, RadFM, EchoNet weights
4. **Scale:** 1,430 patients (60× more than TaDiff's 23)
5. **3D Native:** Full volumetric processing, not 2D slices
6. **Open Source:** Most tools available (nnU-Net, Swin UNETR, RadFM, EchoNet, LDM)
7. **Validation Strategy:** Train on Yale, validate on Cyprus (external)
8. **Novel Contribution:** Nested + Longitudinal ComBat on ViT embeddings
9. **Clinical Output:** Video + masks + uncertainty + text explanation
10. **Counterfactual:** Swap treatment → "what-if" scenarios

---

## 💭 Anticipated Questions & Answers

**Q: "Why not use GPT-4 or Med-PaLM instead of RadFM?"**
> A: Both are closed-source APIs. Can't fine-tune, can't inspect, can't guarantee they'll exist in 3 years. RadFM is open-source (MIT license), we can fine-tune on our domain, and it's specifically trained on radiology with 3D support.

**Q: "Why Swin UNETR and not just ViT?"**
> A: Three reasons. First, hierarchical features handle multi-scale tumors better. Second, pretrained weights from MONAI on 5,050 CT scans. Third, 62M parameters vs 92M for UNETR — smaller and better.

**Q: "How do you handle missing visits? Not every patient has exactly 8 scans."**
> A: Good question. TaViT's time encoding handles irregular intervals naturally — we just encode the actual day gaps. For TaDiff, we'll need to zero-pad or mask missing past scans. That's a detail we'll work out in implementation.

**Q: "What if the harmonization removes real biological signal?"**
> A: That's why we do Kruskal-Wallis tests first. If a batch effect isn't statistically significant (p ≥ 0.05), we don't harmonize it. We only remove confirmed scanner noise. Longitudinal ComBat specifically preserves per-patient trajectories.

**Q: "Have you considered temporal consistency losses for the video?"**
> A: Yes! EchoNet's temporal attention handles that architecturally — frames attend to neighbors. But we could also add explicit losses like optical flow consistency or temporal smoothness. That's a good extension.

**Q: "What's the plan for evaluation without clinical labels on Yale?"**
> A: Three-pronged. First, validate segmentation on Cyprus expert labels (Dice > 0.85). Second, video quality metrics — SSIM, FID, perceptual similarity. Third, leave-out validation: withhold later Yale visits, generate them, compare to ground truth. Fourth, ideally radiologist evaluation of a sample.

**Q: "Timeline for implementation?"**
> A: April-August = 5 months. Month 1: Phase 1 preprocessing on Yale + Cyprus. Month 2: Phase 2-3 Swin UNETR + TaViT + harmonization. Month 3: Phase 4 RadFM integration. Month 4: Phase 5 video generation (biggest chunk). Month 5: evaluation + paper writing. Realistic?

---

**End of Script**
