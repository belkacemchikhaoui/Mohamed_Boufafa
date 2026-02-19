# Phase 5: Video Generation — Meeting Script
## Slides 22–25 (Papers 13–15 + Complete Pipeline)

**Duration:** ~8 minutes for these 4 slides  
**Tone:** Showing findings to supervisor, not lecturing  

---

### **SLIDE: Phase 5 Transition**
*Duration: 15 seconds*

**Script:**
> "Now for the last phase — video generation. This is where the system takes everything from Phases 1–4 and actually generates future brain MRI videos showing how the tumor might progress under different treatments."

---

### **SLIDE 22: Paper 13 — LDM (Latent Diffusion Models)**
*Duration: 1.5 minutes*

**Script:**
> "So the first thing I needed to figure out was — how do you even run diffusion on brain MRI? Our volumes are 4 modalities × 128³ = 8 million values. That's way too big to work with directly.

> What I found is the Latent Diffusion paper by Rombach et al. — this is the foundation behind Stable Diffusion, over 10,000 citations. The key idea is simple: train a VAE to compress images to a tiny latent space — 8× smaller per dimension — then do all the diffusion work in that compressed space.

> For us, that means we compress each scan from (4, 128³) down to (4, 16³), which is 512× smaller. We train the VAE once on all 11,884 Yale scans, freeze it, and then never touch it again.

> The other thing we take from LDM is cross-attention — that's how you inject conditioning into the UNet. In their case it's text prompts, in our case it'll be the RadFM text narrative from Phase 4.

> The code is fully open-source from CompVis."

**Key Points:**
- You understand WHY latent space is needed (computational)
- VAE is trained once and frozen — clean separation
- Cross-attention = how text gets injected

**🎯 Question to Ask Supervisor:**
> "For the VAE, should we train it on all 11,884 scans including the masks as separate channels, or just the 4 MRI modalities?"

---

### **SLIDE 23: Paper 14 — TaDiff (Treatment-Aware Diffusion)**
*Duration: 2.5 minutes*

**Script:**
> "This is the most important paper I found. TaDiff by Liu et al., published in IEEE TMI 2025. It's the closest thing to what we want to build.

> What they do: take 3 past brain scans, concatenate them as channels alongside a noisy future scan, and train a UNet to predict what the future scan should look like. But here's the key part — they also feed in treatment information. They embed the treatment type — like chemo or radiation — and the number of days between scans, create a difference vector, and inject that into the UNet.

> The results show it really matters: with treatment info, they get SSIM of 0.919 and a tumor Dice of 0.719. Without treatment info, Dice drops to 0.556. That's a 29% improvement just from telling the model what treatment the patient received.

> What I thought was really clever is they have two output heads — one for the image, one for 3-region tumor masks: enhancing tumor, necrotic core, and edema. This means every generated frame comes with a segmentation. And they weight tumor voxels 5× more in the loss, so the model pays extra attention to the tumor region.

> They also do counterfactual generation — you keep the same 3 past scans but swap the treatment label. Run it with 'radiation,' then with 'chemo,' then with 'no treatment.' Compare the three outputs and you see what-if scenarios.

> For uncertainty, they run each scenario 5 times with different random noise and compute mean and standard deviation maps.

> Now, their limitations — and this is where we come in. They only had 23 glioma patients. We have 1,430 brain metastases patients. They work on 2D slices only — we'll do full 3D. They use 3 MRI types, we have 4. And they don't have any text conditioning — we add RadFM narratives via cross-attention.

> There's no public code, so we'll implement it from the paper."

**Key Points:**
- This is YOUR blueprint — everything else builds around it
- Treatment info makes a massive difference (+29% Dice)
- Joint prediction (image + masks) is elegant
- You clearly know the limitations and your improvements

**📝 Side Note — Channel Concatenation:**
> TaDiff does NOT use cross-attention for past scans. The 3 past scans are literally stacked as extra channels alongside the noisy future. Simple and effective.

**📝 Side Note — Treatment Diff-Vectors:**
> Each scan gets: embed(treatment_type) + embed(day_number). Then they compute the DIFFERENCE between source and target embeddings. This tells the model "patient went from treatment A at day 30 to treatment B at day 90."

**📝 Side Note — Counterfactual:**
```
Same 3 past scans + Same text →
    Run 1: treatment = "radiation"  → Video A + masks
    Run 2: treatment = "chemo"      → Video B + masks
    Run 3: treatment = "none"       → Video C + masks
Compare A vs B vs C → which treatment works best?
```

**🎯 Questions to Ask Supervisor:**
> "TaDiff was designed for gliomas — primary brain cancer. Our dataset is brain metastases — secondary cancer that spread from elsewhere. The MRI types are the same, so the approach should transfer. Do you agree?"

> "They used 600 diffusion steps. Do you think we should try fewer steps with DDIM sampling to speed things up?"

> "For the counterfactual, should we define a fixed set of treatment scenarios, or let the clinician choose?"

---

### **SLIDE 24: Paper 15 — EchoNet-Synthetic (Video Diffusion Pipeline)**
*Duration: 1.5 minutes*

**Script:**
> "The last paper is EchoNet-Synthetic by Reynaud et al. from MICCAI 2024. Now, this paper is actually about generating synthetic echocardiogram videos — so not brain tumors. But what I found valuable is their video diffusion pipeline code.

> They solved three practical problems we need: First, the VAE architecture with a triple loss — MSE plus LPIPS plus KL — which gives clean compression. Second, temporal attention — they freeze all the spatial layers from the image model, add temporal layers on top, and only train those. That means only 20% of parameters are trained, and it makes the video smooth, no flickering between frames. Third, video stitching — they generate 64-frame chunks with 50% overlap and blend them together, so you can make any video length. They did 19,200 frames in 14 minutes on one GPU.

> What we DON'T need from this paper: the LIDM conditioning — because we have TaDiff for that — the privacy filter, and the synthetic dataset generation.

> So the plan is: clone their repo, replace the 2D echo VAE with our 3D brain MRI VAE, replace the heartbeat UNet with our TaDiff treatment-aware UNet, keep the temporal attention and stitching code, and add the 3-region mask output head.

> The code and weights are fully open-source."

**Key Points:**
- You're taking the VIDEO ENGINE, not the whole paper
- Temporal attention = freeze spatial, train temporal = efficient
- Video stitching solves the "arbitrary length" problem
- Clear about what you skip

**🎯 Question to Ask Supervisor:**
> "For temporal attention, their 64-frame chunks are at a fixed frame rate. In our case, scans might be weeks or months apart. Should we encode the time gap into the temporal attention somehow, or just let the model learn it?"

---

### **SLIDE 25: Complete Pipeline — From Raw MRI to Counterfactual Video**
*Duration: 2 minutes*

**Script:**
> "So let me put it all together. This slide shows the complete pipeline from raw MRI to the final output.

> Phase 1: Yale MRI data goes through BraTS Toolkit for skull stripping, nnU-Net for segmentation — which gives us the 3-region masks — and itk-elastix for aligning scans across visits.

> Phase 2–3: Each scan — 4 MRI modalities plus 3 masks = 7 channels — goes into Swin UNETR, which produces a 768-dimensional embedding. TaViT adds time encoding so the model knows when each scan was taken. Then ComBat harmonizes everything to remove scanner differences.

> Phase 4: The harmonized features go into RadFM's Perceiver, which compresses them to 32 tokens, and MedLLaMA generates a 512-dimensional text narrative.

> Phase 5: The aligned MRI volumes go into the 3D VAE for compression. The TaDiff UNet takes the compressed past scans, the treatment info, and the RadFM text via cross-attention, and generates the future scan plus masks. Then temporal attention smooths everything into a video.

> The final output: a progression video with per-frame 3-region masks and uncertainty maps. And by swapping the treatment label, we get counterfactual comparisons.

> That's the full pipeline. What are your thoughts?"

**Key Points:**
- Walk through the diagram left-to-right, top-to-bottom
- Mention specific numbers (7 channels, 768-dim, 32 tokens, 512-dim)
- End with an open question

**🎯 Questions to Ask Supervisor:**
> "Looking at the full pipeline, is there any component you think is missing? Or anything that's redundant?"

> "What should I prioritize implementing first? Should I start from Phase 1 and go sequentially, or jump to the most novel part — the video generation?"

> "For the Mitacs timeline, do you think 5 months is realistic for this full pipeline, or should we scope it down?"

---

## 🔄 Anticipated Q&A

### Q: "Why not just use Video LDM directly?"
> "Video LDM doesn't have any medical pretrained weights — it was trained on driving videos and web content. We'd be starting from scratch. By combining LDM's latent space idea, TaDiff's treatment-aware UNet, and EchoNet's temporal attention code, we get a medical video pipeline where each component has proven results. TaDiff already showed 0.919 SSIM on brain MRI, and EchoNet already generates smooth medical videos."

### Q: "How does the text from RadFM actually get into the diffusion model?"
> "Through cross-attention in the UNet — same mechanism that Stable Diffusion uses for text prompts. The RadFM narrative becomes a 512-dimensional embedding. At every level of the UNet, there's a cross-attention layer where the image features attend to this text embedding. So the text guides the generation: if RadFM says 'enhancing tumor is shrinking,' the diffusion model is encouraged to generate frames where that happens."

### Q: "TaDiff only had 23 patients. How confident are you this scales?"
> "That's actually one of our main contributions — scaling from 23 to 1,430 patients. The architecture itself is sound — the SSIM of 0.919 on just 23 patients is already strong. With 60× more data and full 3D volumes instead of 2D slices, we should see significant improvements. And the Yale dataset has an average of 8.3 visits per patient, which gives us much denser temporal sampling."

### Q: "What about the 3 masks — can you trust the generated masks?"
> "TaDiff's joint prediction means the masks and images are generated together from a shared UNet encoder. The Dice score of 0.719 for tumor masks is already reasonable, and that's with only 23 patients. The 5× tumor weight in the loss forces the model to prioritize getting the tumor region right. And with uncertainty maps from 5 runs, clinicians can see where the model is confident versus uncertain about the mask predictions."

### Q: "What's the difference between TaViT time and TaDiff treatment time?"
> "Different information, no redundancy. TaViT time is 'when was this scan taken' — it's baked into the 768-dim embedding from Phase 2-3, capturing the scan's position in the patient's timeline. TaDiff treatment time is 'what treatment was given and how many days before the target scan' — this is clinical metadata about the intervention. One is about the scan's temporal context, the other is about what happened between scans."

### Q: "Is 5 months enough?"
> "The modular design helps. Phase 1 tools are all open-source with existing pipelines. Phases 2-3 use pretrained models that just need fine-tuning. Phase 4 RadFM has weights available. Phase 5 is the most work — implementing TaDiff from paper and adapting EchoNet code — but having a clear blueprint helps. If time is tight, we can demonstrate single-frame counterfactuals first, then extend to video."
