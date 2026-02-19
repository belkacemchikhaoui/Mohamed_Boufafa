# Deep Research Prompt for State of the Art

## How to Use This Prompt
Copy and paste this into: **Perplexity Pro**, **ChatGPT-4**, **Gemini Advanced**, or **Claude**

---

## THE PROMPT

```
I am a student starting a 20-week research internship on "Explainable Disease Progression and Counterfactual Video Generation with Vision–Language Models." I need you to act as my research mentor and give me:

1. A COMPLETE UNDERSTANDING of where the field currently stands (not just paper lists)
2. SIMPLE EXPLANATIONS (explain like I'm 10 years old) for each technical concept
3. Specific focus on GENERATIVE MODELS FOR EXPLAINABLE DISEASE PROGRESSION

---

## PART 1: EXPLAIN MY PROJECT LIKE I'M 10 YEARS OLD

First, explain each phase of my research in very simple terms with analogies:

### Phase 1: Vision Transformers (ViT) for Medical Images
- What is a Vision Transformer? (simple analogy)
- What does "segmentation" mean? Why do we need it?
- What is Swin UNETR and why is it special for medical images?
- How does the computer "see" a brain scan?

### Phase 2: Large Language Models (LLMs) for Explanations
- What is an LLM? (like ChatGPT but for doctors)
- How can a computer "explain" what it sees in an image?
- What does it mean to connect images to text?
- Why do doctors need explanations, not just predictions?

### Phase 3: Diffusion Models for Video Generation
- What is a diffusion model? (simple analogy with noise)
- How can AI "imagine" what a tumor will look like in 6 months?
- What is a "counterfactual"? (like asking "what if?")
- How do we make a video from still images?

### Phase 4: Putting It All Together
- How do these 3 pieces connect?
- What is the final goal? What will doctors see?

---

## PART 2: CURRENT STATE OF THE FIELD (February 2026)

For each area, tell me:
- WHERE ARE WE NOW? (what can AI currently do?)
- WHAT'S MISSING? (what problems are not solved yet?)
- WHO IS LEADING? (top research groups/companies)
- WHAT JUST HAPPENED? (breakthroughs in 2024-2025)

### A) Vision Transformers for Medical Imaging

Current state:
- What is the best model RIGHT NOW for brain tumor segmentation?
- How accurate are these models? (give me numbers)
- Can they work in real hospitals yet?
- What are the latest models (2024-2025)?

Key questions to answer:
- Swin UNETR vs nnFormer vs SAM-Med3D: which is best and why?
- What is "self-supervised pretraining" and why does it matter?
- Are there pretrained models I can download and use?

### B) Medical Vision-Language Models

Current state:
- Can AI currently look at a brain scan AND write a report about it?
- How good are these reports compared to human doctors?
- What models exist? (LLaVA-Med, Med-Flamingo, RadFM, etc.)

Key questions to answer:
- Is there an open-source medical vision-language model I can use?
- How do you connect a Vision Transformer to an LLM?
- What is the state of "explainable medical AI"?

### C) Generative Models for Medical Imaging (MOST IMPORTANT FOR US)

Current state:
- Can AI generate realistic medical images today?
- Can AI generate medical VIDEOS (not just single images)?
- Has anyone done "counterfactual medical image generation"?
- Has anyone generated "disease progression videos"?

Key questions to answer:
- What is Video LDM and how does it work?
- Has anyone applied video diffusion to medical imaging?
- What is the gap between general video generation (Sora, etc.) and medical applications?
- Are there ANY papers on counterfactual disease progression videos?

### D) Explainable Disease Progression (OUR SPECIFIC FOCUS)

Current state:
- Has anyone combined: ViT + LLM + Diffusion for disease explanation?
- What is the closest existing work to our project?
- Is this a NEW research direction or are others already doing it?

Key questions:
- What are the research gaps we can fill?
- Is our project novel? What would be our contribution?

---

## PART 3: SPECIFIC RESOURCES I NEED

### For each resource, provide:
1. **Paper title and authors**
2. **Link** (arXiv, DOI)
3. **Code** (GitHub link if available)
4. **Simple summary** (1 sentence, explain like I'm 10)
5. **Why it matters for MY project**

### Categories:

#### A) Must-Read Foundation Papers (to understand the basics)
- The original ViT paper
- The original diffusion model paper (DDPM)
- The original Stable Diffusion / LDM paper
- Swin Transformer paper
- UNETR and Swin UNETR papers

#### B) Medical Vision Transformers (2023-2026)
- Best papers on 3D medical segmentation
- Self-supervised pretraining for medical imaging
- Foundation models (SAM-Med3D, MedSAM)

#### C) Medical LLMs and Vision-Language Models (2023-2026)
- Med-PaLM papers
- LLaVA-Med, Med-Flamingo, RadFM
- Any paper on "explaining medical images with text"

#### D) Medical Image/Video Generation (2023-2026)
- Medical image synthesis with diffusion
- Video generation models (Video LDM, etc.)
- ANY paper on temporal/longitudinal medical image generation
- ANY paper on counterfactual medical imaging

#### E) Disease Progression Modeling
- Tumor growth prediction
- Longitudinal analysis
- Treatment response prediction

#### F) Datasets I Should Use
- BraTS (which version? 2023? 2024?)
- MIMIC for text
- Any dataset with LONGITUDINAL data (multiple scans over time)

---

## PART 4: PRACTICAL GUIDANCE

### A) Where should I start coding?
- What pretrained models can I download today?
- What frameworks should I use? (MONAI? HuggingFace? PyTorch?)
- Step-by-step: what should I implement first, second, third?

### B) What are the easy wins vs hard challenges?
- What parts of my project are "solved problems" I can build on?
- What parts are truly novel/difficult?

### C) Tutorials and Learning Resources
- Best tutorials for medical image segmentation
- Best tutorials for diffusion models
- Best tutorials for vision-language models
- Any courses on medical AI?

---

## PART 5: SUMMARY I NEED

At the end, give me:

1. **THE BIG PICTURE** (2-3 paragraphs):
   - Where is the field of "explainable disease progression" right now?
   - What is possible today vs. what is still science fiction?
   - How novel/ambitious is our project?

2. **TOP 10 PAPERS** I absolutely must read (in order of importance)

3. **TOP 5 CODE REPOSITORIES** I should start with

4. **RESEARCH GAP STATEMENT**:
   - In 2-3 sentences, what is the gap in current research that our project addresses?
   - This will help me write my research motivation.

5. **SIMPLE PROJECT SUMMARY**:
   - Explain our entire project in 5 sentences that a 10-year-old would understand.

---

## CONTEXT ABOUT ME
- I am a student, not an expert
- I learn best with analogies and simple explanations
- I have basic Python/PyTorch knowledge
- I have access to GPUs for training
- My internship is 20 weeks (April-August 2026)
- I need to produce: working code + a paper/report

Please be comprehensive but also explain things simply. I want to UNDERSTAND, not just have a list of papers.
```

---

## QUICK VERSION (for fast searches)

```
Act as a medical AI research mentor. I'm building a system that:
1. Looks at brain tumor scans (using Vision Transformers like Swin UNETR)
2. Explains what it sees in plain English (using LLMs)
3. Generates "what-if" videos showing how the tumor might change (using Diffusion Models)

Questions:
1. Explain each step like I'm 10 years old with simple analogies
2. What is the current state of the art (2024-2026) for each component?
3. Has anyone combined all 3 for "explainable disease progression"?
4. What are the research gaps? Is this novel?
5. Give me the top 10 papers and top 5 code repos to start with
6. What pretrained models can I use today?

Focus especially on: generative models for medical imaging, counterfactual generation, and disease progression prediction.
```

---

## FOLLOW-UP PROMPTS

After getting the initial response, use these follow-ups:

### For deeper understanding:
```
You mentioned [CONCEPT]. Can you explain this more simply? Use an analogy a child would understand. Then show me a concrete example of how it works.
```

### For practical implementation:
```
For [SPECIFIC MODEL/PAPER], show me:
1. The exact GitHub repo or HuggingFace model
2. How to load it in Python (5 lines of code)
3. What input it expects and what output it gives
```

### For finding gaps:
```
Looking at all the papers you mentioned, what HASN'T been done yet? Specifically:
- Has anyone generated counterfactual disease progression VIDEOS (not just images)?
- Has anyone combined ViT + LLM + Video Diffusion for medical explainability?
- What would be a novel contribution we could make?
```

### For timeline planning:
```
Given these resources and a 20-week timeline, create a week-by-week plan:
- Weeks 1-4: What to read and implement?
- Weeks 5-8: What to build?
- Weeks 9-12: What to train?
- Weeks 13-16: What experiments to run?
- Weeks 17-20: What to write up?
```
