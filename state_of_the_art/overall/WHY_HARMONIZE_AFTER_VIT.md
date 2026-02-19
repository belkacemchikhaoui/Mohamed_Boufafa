# Why Do We Harmonize AFTER ViT, Not Before?

## 🎯 Simple Answer (ELI10)

Imagine you have photos from 10 different cameras. Some cameras make colors brighter, some darker. You want to compare faces in the photos.

**Option 1 — Fix photos first, THEN use AI**:
- Make all photos look identical (same brightness, same colors)
- **Problem**: You might accidentally erase REAL differences! Maybe some people actually have darker skin — but you "fixed" it thinking it was camera bias.
- You can't tell what's a camera problem vs. what's real.

**Option 2 — Use AI first, THEN fix AI's measurements**:
- Let AI look at all photos as-is
- AI creates a "face fingerprint" (768 numbers describing each face)
- NOW fix just the fingerprints — remove the "camera brand pattern" but keep the "face pattern"
- **Advantage**: The AI learned to see faces despite camera differences. You only remove camera bias from the measurements, not from the original photos.

**We do Option 2** because:
1. Swin UNETR (our AI) is TRAINED to handle different-looking scans — that's its job!
2. Harmonizing raw scans might delete real tumor differences (aggressive tumors might naturally look different from slow tumors)
3. Harmonizing embeddings (the 768 numbers) is safer — we KNOW what's biology (tumor size, spread) vs. scanner (brightness, noise)

---

## 📊 Does It Impact Results? YES — Huge Impact!

### Without Harmonization (❌ Bad):
```
Patient scanned on Scanner A (2015) → embedding [0.5, 0.3, ...]
Same patient scanned on Scanner B (2018) → embedding [0.8, 0.9, ...]

Model sees: HUGE change in embeddings!
Model thinks: "Tumor exploded in size!"
Reality: Tumor grew 10%, scanner difference caused other 90% of change
```

**Results**:
- LLM generates: "Aggressive 90% tumor growth detected!"
- Video shows: Giant tumor jump between timepoints
- Doctor says: "This AI is broken, the tumor barely changed!"

### With Harmonization (✅ Good):
```
Patient scanned on Scanner A (2015) → embedding [0.5, 0.3, ...] → HARMONIZE → [0.5, 0.3, ...]
Same patient scanned on Scanner B (2018) → embedding [0.8, 0.9, ...] → HARMONIZE → [0.55, 0.35, ...]

Model sees: Small change in embeddings
Model thinks: "Tumor grew 10%"
Reality: Correct! Tumor grew 10%
```

**Results**:
- LLM generates: "Moderate 10% tumor growth, stable progression"
- Video shows: Smooth, realistic tumor growth
- Doctor says: "This matches what I see!"

---

## 🧠 Technical Reason (For Your Thesis)

ComBat removes **batch effects** (systematic differences between groups like scanners, years, protocols) while preserving **biological effects** (real tumor characteristics).

**Why on embeddings**:
- Embeddings are **high-level features** (tumor size, shape, texture) — easier to separate biological signal from scanner noise
- Raw pixels are **low-level** (brightness, contrast) — biological and technical effects are mixed together inseparably
- Swin UNETR's training already makes it somewhat robust to scanner differences — harmonization just cleans up remaining artifacts

**Yale's specific challenge**:
- 2004–2023 = 20 years of scanner upgrades
- Same patient might see 3 different scanners across 8 visits
- Longitudinal ComBat ensures tumor growth measurements aren't confounded by scanner switches

**Impact on objectives**:
- **Obj 2**: Clean temporal patterns → TaViT learns real progression, not scanner artifacts
- **Obj 3**: LLM explains real biology, not scanner differences
- **Obj 4**: Videos show smooth progression, not scanner-switch jumps
- **Obj 5**: External validation on Cyprus works (different scanners than Yale)

---

## 🔬 MM-Embed Paper — ELI10 Explanation

**MM-Embed is like teaching a robot to understand photos AND text together, but the robot has a bad habit — it's lazy and only reads the text, ignoring photos!**

**The Problem**: When you show a robot a photo + caption, it cheats by just reading the caption because text is easier than looking at images. So if you ask "What's in this medical scan?", the robot might say the right answer BUT it never actually looked at the scan — it just read the caption you gave it!

**What MM-Embed Discovered**:
1. **Modality bias** = The robot prefers text over images (because text is simpler)
2. **Sequential training > Joint training** = Teach the robot to look at images FIRST, text SECOND (like learning to walk before running)
3. **Hard negatives** = Show the robot WRONG answers that SOUND right (so it learns to actually look at the image to catch the lie)

**Why You Care**: When you train RadFM (your LLM), you must:
- Check if it's actually using the Swin UNETR embeddings or just repeating generic medical text
- Train in stages: image features first, then add text
- Use hard negatives: show similar-looking but wrong cases so it learns to look carefully

**In one paragraph**: MM-Embed (NVIDIA, ICLR 2025) discovered that when AI models combine images and text, they often cheat by only reading the text and ignoring images because text is easier — this is called "modality bias." They proved that training in stages (images first, then text) works better than training everything at once, and you need to use "hard negatives" (tricky wrong answers that sound right) to force the model to actually look at the images. For your project, this warns you to watch out when training RadFM — make sure the LLM is actually using the Swin UNETR scan features, not just generating generic medical text based on the clinical metadata alone.
