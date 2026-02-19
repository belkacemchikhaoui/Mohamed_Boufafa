# Explainable Disease Progression and Counterfactual Video Generation
## Presentation Script & Speaker Notes

**Duration:** 15-20 minutes  
**Audience:** Supervisor meeting / Academic presentation  
**Slides:** 17 (including title and references)

---

## 📚 Quick Reference: Key Terms & Concepts

### Acronyms You Should Know

| Term | Full Name | Simple Explanation |
|------|-----------|-------------------|
| **MONAI** | Medical Open Network for AI | PyTorch framework for medical imaging (by NVIDIA) |
| **VQA** | Visual Question Answering | AI answers questions about images |
| **CLIP** | Contrastive Language-Image Pre-training | AI that understands both text AND images together |
| **FVD** | Fréchet Video Distance | Video realism score (lower = better) |
| **IS** | Inception Score | Quality + diversity score (higher = better) |
| **LDM** | Latent Diffusion Model | Diffusion in compressed space (8× smaller = faster) |

### 🧠 MRI Modalities (BraTS Dataset)

| Modality | Full Name | What It Shows |
|----------|-----------|---------------|
| **T1** | T1-weighted | Basic anatomy, gray/white matter |
| **T1c** | T1 + Contrast (Gadolinium) | **Active tumor lights up!** ⭐ |
| **T2** | T2-weighted | Edema (swelling), fluid = bright |
| **FLAIR** | Fluid-Attenuated Inversion Recovery | Edema without CSF noise (clearest tumor edges) |

**Input to Swin UNETR:** `[Batch, 4, 240, 240, 155]` = all 4 modalities as channels

### 🎯 Why 3D Support is Critical

1. **Tumors are 3D structures** - 2D sees slices, 3D sees whole shape
2. **Spatial relationships** - "tumor near motor cortex" requires 3D reasoning
3. **Growth in all directions** - tumor can grow "upward" through slices
4. **Volume measurement** - needs full 3D understanding

**RadFM** = 3D support ✅ | **LLaVA-Med** = 2D only ❌

---

## 💡 Core Concept Explanations

### Diffusion Models - The Core Idea
```
TRAINING:  Clean Image → Add noise → Add noise → Pure Static
GENERATION: Pure Static → Remove noise → Remove noise → NEW Image!
```
The AI learns to denoise, then we use it BACKWARDS to create images from random noise.

### Latent Diffusion - Why It's Faster
```
512×512 image = 262,144 pixels = SLOW
    ↓ Compress (Encoder)
64×64 latent = 4,096 values = FAST (8× smaller)
    ↓ Do diffusion HERE
    ↓ Expand (Decoder)
512×512 result = Same quality!
```

### Text Conditioning via CLIP
```
You type: "brain tumor growing"
    ↓
CLIP converts text to numbers
    ↓
Diffusion follows these "instructions"
    ↓
Image matches your description!
```
Text becomes the remote control for image generation! 🎮

### Video LDM Architecture
```
Stable Diffusion (makes images)
    +
Temporal layers (connects frames)
    +
3D convolutions (smooth transitions)
    =
Video generator!
```
Training: Freeze spatial → Train temporal → Fine-tune all

### Handling Time Gaps Between Scans
```
Real Scan (Jan) -------- 1 month gap -------- Real Scan (Feb)
```
Solution: Model INTERPOLATES!
```
Real Scan → [AI: Day 7] → [Day 14] → [Day 21] → Real Scan
```
Model learns growth patterns and generates realistic intermediate frames!

### MVG 3-Step Pipeline
```
1. GPT-4:  "Describe disease stages" → "mild → moderate → severe"
2. SD:     Generate image for each stage 🖼️ 🖼️ 🖼️
3. SEINE:  Animate smoothly between images 🎬
```
They did: X-rays ✅, Retinal ✅, Skin ✅  
They didn't: 3D MRI ❌, Brain tumors ❌, Explanations ❌ = OUR contribution!

---

## Open-Source Models vs. Architecture-Only Papers

### ✅ **Open-Source / Freely Available Models**

| Model | Purpose | Repository | Pretrained Weights |
|-------|---------|------------|-------------------|
| **Swin UNETR** | 3D Medical Segmentation | MONAI (`monai.networks.nets.SwinUNETR`) | ✅ Yes (5,050 CT scans) |
| **UNETR** | 3D Medical Segmentation | MONAI (`monai.networks.nets.UNETR`) | ✅ Yes |
| **nnU-Net** | Auto-configuring Segmentation | `MIC-DKFZ/nnUNet` | ✅ Self-configures |
| **MedSAM** | Universal Medical Segmentation | `bowang-lab/MedSAM` | ✅ Yes (1.57M images) |
| **LLaVA-Med** | Medical Vision-Language | `microsoft/LLaVA-Med` | ✅ Yes |
| **RadFM** | Radiology Foundation Model | `chaoyi-wu/RadFM` | ✅ Yes (16M images, 3D support) |
| **Stable Diffusion** | Image Generation | `CompVis/stable-diffusion` | ✅ Yes |
| **DDPM** | Diffusion Models | `hojonathanho/diffusion` | ✅ Yes |
| **ViT** | Vision Transformer | `google-research/vision_transformer`, `timm` | ✅ Yes |
| **Swin Transformer** | Hierarchical ViT | `microsoft/Swin-Transformer` | ✅ Yes |

### ⚠️ **Needs Adaptation / Fine-tuning**

| Model | Issue | Solution |
|-------|-------|----------|
| **Video LDM** | No medical pretrained weights | Fine-tune on medical video data |
| **MVG** | Not for brain tumors | Adapt pipeline for MRI modality |
| **MedEdit** | Trained on stroke, code not released | Adapt methodology for tumors |

### ❌ **Architecture Only / Closed Source**

| Model | Issue | Alternative |
|-------|-------|-------------|
| **Med-PaLM** (Google) | Closed source, API only | Use RadFM or LLaVA-Med |
| **GPT-4V** (OpenAI) | API only, not downloadable | Use open VLMs |

---

## Slide-by-Slide Script

---

### **SLIDE 1: Title Slide**
*Duration: 30 seconds*

**Script:**
> "So, this is my overview of the state of the art for our project on Explainable Disease Progression and Counterfactual Video Generation. I've gone through the literature and I'd like to walk you through what I've found and get your feedback on the direction I'm taking."

**Key Points:**
- Set the tone: you're sharing your research findings
- Invite feedback from the start
- Show you've done the work

---

### **SLIDE 2: Presentation Roadmap**
*Duration: 1 minute*

**Script:**
> "I've organized my findings into three parts. First, I'll recap the problem we're tackling and what I understand our objectives to be—please correct me if I'm missing anything. Then I'll go through what I found in the literature for Vision Transformers, Vision-Language Models, and Video Generation. Finally, I'll show you the architecture I'm proposing based on this research."

**Key Points:**
- Show organization of your research
- Invite corrections early
- Emphasize this is YOUR understanding

**🎯 Question to Ask Supervisor:**
> "Before I continue, is there any particular area you'd like me to focus on more, or anything you think I might have missed?"

---

### **SLIDE 3: The Problem - Medical AI Black Boxes**
*Duration: 2 minutes*

**Script:**
> "So as I understand it, the core problem is this: current medical AI can achieve impressive results—like Dice scores around 0.85 to 0.90 on BraTS for tumor segmentation. But what's missing is the *why* and the *what next*.

> From what I've read, doctors need more than just a segmentation mask. They need to understand the reasoning, and they need to visualize potential futures for treatment planning.

> So our goal, as I understand it, is to build a system that does all three: segment, explain, and generate progression videos. Is that correct?"

**Key Points:**
- Show you understand the problem
- Reference specific metrics you found
- End with a confirmation question

**🎯 Questions to Ask Supervisor:**
> "From your experience working in this area, what do clinicians actually prioritize more—the text explanation or the visual progression?"

> "Are there specific clinical scenarios you had in mind where this would be most useful?"

---

### **SLIDE 4: Research Objectives**
*Duration: 1.5 minutes*

**Script:**
> "Based on our discussions and my reading, I've broken down the project into five objectives. Let me know if I'm missing anything or if the priorities should be different.

> First, setting up the data pipeline with BraTS and related datasets. Second, getting the segmentation working with Vision Transformers. Third, connecting that to a language model for explanations. Fourth, the video generation component. And fifth, evaluation.

> The key questions I'm trying to answer through this research are: Can these three components actually work together? Will the outputs be medically meaningful?"

**Key Points:**
- Frame objectives as YOUR understanding
- Invite corrections
- Show awareness of challenges

**🎯 Question to Ask Supervisor:**
> "For the evaluation, what would you consider a success? Should we aim for quantitative metrics, or is clinical validation more important?"

---

### **SLIDE 5: Vision Transformers - The Foundation**
*Duration: 2 minutes*

**Script:**
> "Let me walk you through what I learned about Vision Transformers. The key insight from the ViT paper is treating image patches as tokens—this lets the model capture both local and global patterns.

> What I found interesting is how Swin Transformer improved on this with shifted windows, which gives you linear complexity instead of quadratic. That's important for 3D medical volumes.

> Looking at the benchmarks, Swin-T gets 81.3% on ImageNet with only 28 million parameters, compared to ViT-B needing 86 million for 77.9%. So it's both smaller and better.

> Both are open source, which is good for our purposes."

**Key Points:**
- Share what YOU learned
- Highlight insights that stood out to you
- Connect to our project needs (3D volumes)

---

### **SLIDE 6: Medical Segmentation - UNETR & Swin UNETR**
*Duration: 2 minutes*

**Script:**
> "For medical segmentation specifically, I compared UNETR and Swin UNETR. UNETR uses a standard ViT encoder with a CNN decoder—it gets 0.899 Dice on BTCV.

> Swin UNETR seemed like the better choice for us. It uses hierarchical features, achieves 0.9005 Dice on BraTS, and has fewer parameters—62 million versus 92 million.

> What convinced me was that Tang et al. pretrained it on 5,050 CT scans with self-supervised learning, and the weights are available in MONAI.

> So my recommendation is Swin UNETR with MONAI pretrained weights. The hierarchical features should also work well for feeding into the language model. Does that reasoning make sense?"

**Key Points:**
- Explain YOUR comparison process
- Give specific reasons for your choice
- Ask for validation

**📝 Side Note - What is MONAI?**
> MONAI = **Medical Open Network for AI**. It's a PyTorch-based framework by NVIDIA specifically for medical imaging. Provides pretrained models, medical-specific transforms, and optimized data loading for 3D volumes. Website: monai.io

**🎯 Question to Ask Supervisor:**
> "The MONAI weights are from CT scans. Should we fine-tune on BraTS MRI first, or use them directly?"

---

### **SLIDE 7: Segmentation Model Comparison**
*Duration: 1.5 minutes*

**Script:**
> "Here's a comparison table I put together. Swin UNETR leads on Dice, but nnU-Net is interesting as a baseline—it's much smaller at 19 million parameters and still gets 0.89.

> MedSAM is more for interactive use cases, so probably not our main choice, but could be useful for annotation or refinement.

> One thing I came across: GMIM from 2024 showed you can get 2-3% Dice improvement with self-supervised grid masking. That might be worth exploring later."

**Key Points:**
- Present as YOUR analysis
- Show you've done thorough comparison
- Mention potential future directions

---

### **SLIDE 8: Medical Vision-Language Models**
*Duration: 2 minutes*

**Script:**
> "For the language model component, I evaluated several options. LLaVA-Med looked promising—it's near GPT-4 level on medical VQA—but it only supports 2D images, which is a problem for brain MRI.

> RadFM stood out because it supports 3D input, it was trained on 16 million radiology images—the largest I found—and the code and weights are available.

> Med-PaLM 2 from Google has impressive results, but it's closed source, so we can't use it.

> So my recommendation is RadFM. The 3D support is critical for our use case, and it has the largest training data. What do you think?"

**Key Points:**
- Explain YOUR evaluation criteria
- Justify the choice with specific reasons
- Ask for feedback

**📝 Side Note - What is VQA?**
> VQA = **Visual Question Answering**. The task of answering natural language questions about images. Example: Show brain MRI → Ask "Is there a tumor?" → Model answers "Yes, in the right temporal lobe."

**📝 Side Note - Why 3D Support is Critical:**
> Brain MRI is inherently 3D (e.g., 240×240×155 voxels). A tumor spans ~20-50 slices. 2D models analyze slice-by-slice and LOSE spatial context. 3D models see the tumor as ONE connected structure—essential for accurate volume measurement, anatomical relationships ("near motor cortex"), and progression tracking (tumor can grow in Z direction).

**🎯 Questions to Ask Supervisor:**
> "RadFM was trained on radiology reports. Do you think we'll need to fine-tune it specifically for brain tumors?"

> "For the output, should we aim for structured reports or more conversational explanations?"

---

### **SLIDE 9: Diffusion Models Fundamentals**
*Duration: 2 minutes*

**Script:**
> "Moving to the generative side—I spent time understanding diffusion models since they're central to the video generation.

> The basic idea from Ho et al. is: add noise gradually, then learn to reverse it. What made Latent Diffusion practical was doing this in a compressed space—8x smaller—which makes training feasible.

> Stable Diffusion seems like the right base to build on. It's well-documented, has good community support, and the code is available from CompVis."

**Key Points:**
- Show you understand the fundamentals
- Explain why Stable Diffusion is the base
- Keep it concise—this is background

**📝 Side Note - Diffusion Explained Simply:**
```
TRAINING:  Clean Image → Add noise → Add noise → Pure Static
GENERATION: Pure Static → Remove noise → Remove noise → NEW Image!
```
> The AI learns to DENOISE. Then we use it BACKWARDS—start from random noise, denoise step by step, get a brand new image!

**📝 Side Note - Latent Diffusion (Why 8× smaller):**
```
512×512 image = 262,144 pixels = SLOW
    ↓ Compress
64×64 latent = 4,096 values = FAST (8× smaller each dimension)
    ↓ Do diffusion here
    ↓ Expand back
512×512 result
```
> Same quality, 8× faster! That's why Stable Diffusion works on normal GPUs.

**📝 Side Note - Text-Conditioned via CLIP:**
> CLIP = **Contrastive Language-Image Pre-training** (by OpenAI). It understands BOTH text and images. When you type "brain tumor growing," CLIP converts it to numbers that guide the diffusion process. Text becomes the remote control for generation! 🎮

---

### **SLIDE 10: Video LDM**
*Duration: 1.5 minutes*

**Script:**
> "For video specifically, Video LDM extends image diffusion with temporal layers—3D convolutions and temporal attention to keep frames consistent.

> The training approach is interesting: freeze the spatial layers, add temporal layers, then fine-tune. That's efficient because you leverage pretrained image models.

> The challenge I see: there are no medical pretrained weights. Video LDM was trained on driving videos and web content. So we'll need to figure out how to adapt it for medical imaging."

**Key Points:**
- Show understanding of the architecture
- Identify the main challenge
- Frame as something to solve together

**📝 Side Note - Video LDM Architecture:**
```
Stable Diffusion (makes images)
    +
Temporal layers (connects frames in time)
    +
3D convolutions (smooth frame-to-frame transitions)
    =
Video generator!
```
> Training: Freeze spatial → Train temporal → Fine-tune all
> Key metrics: FVD 550.61 (realism, lower=better), IS 33.45 (quality+diversity, higher=better)
> FVD = Fréchet Video Distance, IS = Inception Score

**📝 Side Note - Handling Time Gaps Between Scans:**
> Problem: Real scans might be 1+ month apart!
> Solution: Model INTERPOLATES (fills the gap):
```
Real Scan (Jan) → [AI: Day 7] → [Day 14] → [Day 21] → Real Scan (Feb)
```
> Model learns growth patterns and generates realistic intermediate frames!

**🎯 Question to Ask Supervisor:**
> "Given the lack of medical video data, should we start with single-image counterfactuals first, then extend to video? Or tackle video directly?"

---

### **SLIDE 11: MVG - Medical Video Generation**
*Duration: 1.5 minutes*

**Script:**
> "MVG is the closest work to what we're doing. They generate disease progression videos using GPT-4 for prompts, Stable Diffusion for images, and SEINE for animation.

> Their clinical validation was encouraging—doctors preferred their videos 2:1 over generic models.

> But there are gaps we can fill: they don't do brain tumors, they don't work with 3D MRI, and they don't have the explicit segmentation and explanation components. That's where our contribution would be."

**Key Points:**
- Acknowledge prior work
- Identify gaps = our contribution
- Show you understand the novelty

**📝 Side Note - MVG 3-Step Pipeline:**
```
1. GPT-4:  "Describe disease stages" → "mild → moderate → severe"
2. SD:     Generate image for each stage 🖼️ 🖼️ 🖼️
3. SEINE:  Animate smoothly between images 🎬
```
> They tested: X-rays ✅, Retinal ✅, Skin lesions ✅
> They DIDN'T do: 3D MRI ❌, Brain tumors ❌, Explanations ❌ = OUR contribution!

---

### **SLIDE 12: MedEdit - Counterfactual Generation**
*Duration: 2 minutes*

**Script:**
> "MedEdit really helped me understand counterfactual generation. The idea is: given a healthy brain and a mask, generate what it would look like with pathology.

> What's clever is they model not just the direct change—the lesion—but also indirect effects like brain atrophy. Real pathology causes secondary changes, and they capture that.

> They got 47% better FID than prior methods, and a neuroradiologist couldn't distinguish their outputs from real scans.

> One limitation: the code isn't publicly released yet. So we'd either need to implement their approach ourselves or adapt similar methods. Also, they trained on stroke, not tumors, so there might be differences in morphology."

**Key Points:**
- Show you understand the methodology deeply
- Acknowledge limitations honestly
- Connect to our project

**📝 Side Note - MedEdit Counterfactual Concept:**
```
Healthy Brain + Mask ("put pathology HERE")
    ↓ Diffusion
Brain WITH Pathology (fake but realistic!)
```
> Key innovation: Models BOTH direct effects (the lesion itself) AND indirect effects (brain atrophy, ventricle enlargement)
> 
> For OUR project: Instead of "healthy → diseased," we do "diseased now → diseased FUTURE" = tumor progression over time!

**🎯 Questions to Ask Supervisor:**
> "Do you think the stroke-trained methodology will transfer to tumors, or will we need tumor-specific training?"

> "Should we start with single-timepoint counterfactuals before moving to temporal progression?"

---

### **SLIDE 13: The Complete Pipeline Overview**
*Duration: 1.5 minutes*

**Script:**
> "So now let me show you the pipeline I'm proposing based on everything I've reviewed in the literature. The idea is: MRI goes into Swin UNETR for segmentation, those features feed into RadFM for generating explanations, and then everything conditions a diffusion model for video generation.

> I chose RadFM over other options because it supports 3D input, which seemed essential for brain MRI. Each component I've covered feeds into this overall design.

> Does this overall architecture make sense to you, or should I be thinking about it differently?"

**Key Points:**
- This is YOUR synthesis of the research
- Show how all the pieces connect
- Ask for feedback on the overall flow

**🎯 Questions to Ask Supervisor:**
> "For connecting Swin UNETR to RadFM, would you recommend a simple projection layer or something more complex like cross-attention?"

> "Should the video generation be conditioned on the raw features, the text, or both?"

---

### **SLIDE 14: Proposed Architecture**
*Duration: 2 minutes*

**Script:**
> "Based on everything I've reviewed, here's the architecture I'm proposing.

> Stage 1: Swin UNETR for segmentation—multi-modal MRI in, tumor mask and features out.

> Stage 2: RadFM for explanation—we'll need an adapter to connect the features, and it generates the text report.

> Stage 3: Video LDM conditioned on the scan, segmentation, and explanation to generate progression frames.

> The key design decisions are: keep it modular so we can train each part separately, use pretrained weights wherever possible, and have the text as an explicit intermediate for interpretability.

> This is my proposal—I'd really value your thoughts on whether this makes sense or if I should reconsider any part of it."

**Key Points:**
- Present as YOUR synthesis of the research
- Justify each choice
- Explicitly ask for feedback

**🎯 Questions to Ask Supervisor:**
> "For modular training, should we freeze earlier stages when training later ones?"

> "What temporal resolution makes sense—monthly frames, or longer intervals like 3-6 months?"

---

### **SLIDE 15: Available Open-Source Resources**
*Duration: 1.5 minutes*

**Script:**
> "I mapped out what's available versus what we'll need to build or adapt.

> In green—ready to use: Swin UNETR and nnU-Net for segmentation, RadFM and LLaVA-Med for language, Stable Diffusion for generation.

> In orange—needs work: Video LDM has no medical weights, and MedEdit methodology is described but code isn't released.

> In red—can't use: Med-PaLM and GPT-4V are closed source.

> So we have a good foundation, but the video component will need the most original work."

**Key Points:**
- Show thorough inventory
- Be honest about gaps
- Identify where effort is needed

---

### **SLIDE 16: Available Datasets Overview**
*Duration: 1.5 minutes*

**Script:**
> "For data, BraTS is our main resource—over 2,000 volumes with expert labels. MIMIC-III can help with text training.

> The challenge I see is longitudinal data. Most public datasets are single timepoint. We might need to use pre/post-operative scans as a proxy for progression, or generate synthetic sequences.

> Do you have any suggestions for accessing longitudinal tumor data, or should we plan for synthetic generation from the start?"

**Key Points:**
- Show awareness of data landscape
- Identify the key challenge
- Ask for guidance

**📝 Side Note - BraTS Input Format & MRI Modalities:**
```
Input shape: [Batch, 4, 240, 240, 155]
                   └─ T1, T1c, T2, FLAIR (4 modalities)
```
> Each modality highlights different things:
> - **T1**: Basic anatomy, gray/white matter distinction
> - **T1c** (with contrast): **Active tumor lights up!** ⭐ (Most important)
> - **T2**: Edema/swelling (fluid appears bright)
> - **FLAIR**: Clear tumor boundaries (suppresses CSF noise)

**🎯 Questions to Ask Supervisor:**
> "Do you know of any longitudinal brain tumor datasets available through clinical partnerships?"

> "For synthetic progression, would growth models like Gompertz be appropriate?"

---

### **SLIDE 17: Conclusion & Research Gap**
*Duration: 1.5 minutes*

**Script:**
> "So to summarize what I've found: there's a clear gap in the literature. Systems can segment tumors, or generate reports, or create synthetic images—but nothing does all three together with temporal progression.

> Our contribution would be integrating ViT segmentation, LLM explanation, and diffusion-based video generation into one pipeline.

> I think it's feasible because we can build on these open-source components, but the video generation will be the hardest part given the lack of medical pretrained models.

> That's my understanding of the state of the art. I'd really appreciate your feedback—am I on the right track? Is there anything I've missed or misunderstood?"

**Key Points:**
- Summarize YOUR findings
- State the gap clearly
- End by asking for feedback

---

### **SLIDE 18: References**
*Duration: As needed for questions*

**Notes:**
- 13 key references listed with arXiv links
- Keep this slide up during Q&A
- Be prepared to discuss any cited paper in detail

---

## 🎯 Strategic Questions to Ask Your Supervisor

### Technical Direction Questions

1. **On Model Selection:**
   > "We've chosen Swin UNETR for segmentation and RadFM for language. Do you see any potential issues with this combination, or would you suggest alternatives?"

2. **On Architecture:**
   > "For connecting the segmentation features to the language model, would you recommend a simple linear adapter, cross-attention, or Q-Former style architecture like BLIP-2?"

3. **On Training Strategy:**
   > "Given the 20-week timeline, should we prioritize getting each component working separately first, or attempt end-to-end training early?"

4. **On Video Generation:**
   > "Video LDM has no medical weights. Would you recommend starting with image-to-image translation (like MedEdit) first, then extending to video, or tackling video directly?"

### Data & Evaluation Questions

5. **On Longitudinal Data:**
   > "The biggest data gap is longitudinal tumor sequences. Do you have any suggestions for accessing such data, or should we plan for synthetic generation from the start?"

6. **On Evaluation:**
   > "For evaluating the text explanations, should we prioritize automated metrics (BLEU/ROUGE), factual consistency checking, or try to arrange clinician evaluation?"

7. **On Clinical Validation:**
   > "Is there potential to involve a radiologist for qualitative evaluation of the outputs, even informally?"

### Scope & Feasibility Questions

8. **On Project Scope:**
   > "If we need to reduce scope, which component would you prioritize: better segmentation, better explanations, or better video generation?"

9. **On Fallback Plans:**
   > "If video generation proves too challenging, would a sequence of static images at different timepoints still be a valuable contribution?"

10. **On Publication:**
    > "What would be the minimum viable result for a publication—a working demo, quantitative improvements, or clinical validation?"

### Research Direction Questions

11. **On Counterfactuals:**
    > "For counterfactual scenarios, should we focus on 'what if the tumor grows' or also include 'what if treatment is applied'? The latter requires modeling treatment effects."

12. **On Explainability:**
    > "Should the text explanations be structured like radiology reports, or more conversational for patient communication?"

13. **On Generalization:**
    > "Should we focus exclusively on brain tumors, or also test on other pathologies like lung nodules to show generalization?"

---

## Anticipated Questions & Answers from Supervisor

### Q1: "How will you handle the lack of longitudinal tumor data?"
**A:** Three approaches: (1) Use pre/post-operative BraTS scans as pseudo-longitudinal pairs, (2) Use MedEdit-style counterfactual generation to synthesize progression sequences, (3) Apply tumor growth models (like Gompertz growth) to generate realistic intermediate states.

### Q2: "Why Swin UNETR over nnU-Net?"
**A:** Swin UNETR slightly outperforms nnU-Net (0.9005 vs 0.89 Dice) and—crucially—provides rich feature embeddings from its transformer encoder that we can feed to RadFM. nnU-Net's CNN features are less suitable for direct language model integration via attention mechanisms.

### Q3: "Why RadFM over LLaVA-Med?"
**A:** RadFM supports 3D volumetric input, which is essential for brain MRI. LLaVA-Med is 2D only. RadFM also has the largest training data (16M images vs 600K for LLaVA-Med) and is specifically designed for radiology.

### Q4: "How will you evaluate the text explanations?"
**A:** Multi-faceted: (1) Automated metrics like BLEU/ROUGE against reference reports, (2) Factual accuracy checking—does the text correctly describe tumor size, location, and characteristics from the segmentation? (3) Ideally, informal clinician evaluation for coherence and usefulness.

### Q5: "What if video generation doesn't work well?"
**A:** Fallback to image sequences rather than smooth video. Even a series of static images at different timepoints (month 0, 3, 6) would be valuable for clinical planning. MedEdit-style single counterfactual images are also a valid contribution.

### Q6: "Is 20 weeks enough?"
**A:** Ambitious but realistic because we're building on pretrained models, not training from scratch. The modular design means we can have working intermediate results even if the full pipeline isn't perfect. Each component (segmentation, explanation, generation) is independently valuable.

### Q7: "What makes this novel compared to MVG?"
**A:** MVG generates progression videos but doesn't do explicit segmentation or natural language explanations. Our contribution is the three-way integration: ViT segmentation → LLM explanation → Diffusion video, creating a fully explainable system. Also, MVG hasn't been applied to brain tumors or 3D MRI.

### Q8: "How do you know the generated videos will be clinically plausible?"
**A:** We'll use multiple validation approaches: (1) FID/FVD scores against real tumor sequences, (2) Check that tumor growth follows known biological patterns (e.g., Gompertz curves), (3) Verify segmentation consistency across frames, (4) Ideally, radiologist review.

---

## Timing Guide

| Section | Slides | Duration |
|---------|--------|----------|
| Title + Roadmap | 1-2 | 1.5 min |
| Part I: Vision | 3-4 | 3.5 min |
| Part II: State of Art | 5-12 | 10 min |
| Part III: Approach | 13-16 | 6 min |
| Conclusion | 17-18 | 1.5 min |
| **Total** | | **~22 min** |

Leave 5-10 minutes for questions and discussion with supervisor.

---

## Key Takeaways to Emphasize

1. **The Gap is Real:** No system does segmentation + explanation + video together
2. **Building on Giants:** We use pretrained SOTA models (Swin UNETR, RadFM, Stable Diffusion)
3. **Clinical Value:** Doctors need explainability, not just predictions
4. **Our Choices are Justified:** Swin UNETR (best Dice + features), RadFM (3D support + largest data)
5. **Novel Contribution:** First explainable tumor progression video system

---

## Notes on Model Choices

### Why Swin UNETR?
- ✅ Best automatic Dice (0.9005 on BraTS)
- ✅ Hierarchical features suitable for multi-scale understanding
- ✅ Linear O(n) complexity for 3D volumes
- ✅ MONAI pretrained weights available
- ✅ Transformer features integrate well with LLMs

### Why RadFM?
- ✅ Supports 3D volumetric input (critical for brain MRI)
- ✅ Largest training data (16M images)
- ✅ 83% diagnosis accuracy
- ✅ Open-source with pretrained weights
- ❌ LLaVA-Med: 2D only
- ❌ Med-PaLM: Closed source

### Why Video LDM + Stable Diffusion?
- ✅ State-of-the-art video generation architecture
- ✅ Temporal consistency via 3D convolutions
- ✅ Can leverage Stable Diffusion pretrained weights
- ⚠️ Needs medical fine-tuning (no pretrained medical weights)

---

## Understanding MedEdit (For Your Reference)

MedEdit generates **counterfactual** brain MRI scans—answering "what would this healthy brain look like if it had a stroke?"

### The Pipeline:
1. **Healthy Input** → A real brain MRI of a healthy patient
2. **Diffusion Model** → Denoising process to transform images
3. **Conditioning on Semantic Maps** → A mask tells WHERE to put pathology
4. **Counterfactual Output** → Realistic brain scan with induced pathology

### Key Innovation:
- **Direct effects**: The stroke lesion itself (shown in purple)
- **Indirect effects**: Brain atrophy, ventricle enlargement (shown in cyan)

Real strokes cause secondary effects—MedEdit models these too!

### Relevance to Your Project:
| MedEdit (Stroke) | Your Project (Tumor) |
|------------------|----------------------|
| Healthy → Stroke | Tumor at time T → Tumor at T+6 months |
| Adds stroke lesion | Shows tumor growth/shrinkage |
| Models brain atrophy | Models tissue changes around tumor |

**Your adaptation**: Instead of "healthy → diseased", you'd do "diseased now → diseased future" to show **tumor progression over time** as a counterfactual video.
