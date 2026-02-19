# Simple Paper Analysis Prompt (Full PDF Version)

**Copy and paste this prompt to analyze ANY paper:**

---

## ANALYSIS REQUEST

Read the COMPLETE PDF file from first page to last page. Extract ALL text before analyzing. Do not skip any sections, tables, figures, or appendices.

After reading the full PDF, provide this analysis:

---

### 1️⃣ ONE-SENTENCE SUMMARY
What does this paper do? (Max 20 words)

---

### 2️⃣ KEY RESULTS (List specific numbers)
- Dataset used: [name, size]
- Main accuracy/performance: [exact numbers]
- Comparison to baselines: [who they beat, by how much]
- Any other important metrics

---

### 3️⃣ WHAT'S NEW (Their contribution)
- What did nobody else do before?
- Why is this important?
- What problem does it solve?

---

### 4️⃣ LIMITATIONS (What's missing/broken)
- What doesn't work well?
- What did they NOT test?
- What do they say needs future work?
- Any problems you noticed?

---

### 5️⃣ METHODS (Explain simply)
How did they do it? Break down into steps:
- Step 1: ...
- Step 2: ...
- Step 3: ...
(No complex math, just the process)

---

### 6️⃣ CODE & RESOURCES
- Code available: [Yes/No, link if yes]
- Dataset available: [Yes/No, how to access]
- Can reproduce: [Yes/No, why]

---

### 7️⃣ CONNECTION TO YALE DATASET

**Our main dataset**: Yale Longitudinal Brain Metastases (11,892 scans, 1,430 patients, avg 8 timepoints per patient)

**How this paper helps with Yale**:
- Does it show how to preprocess Yale data?
- Does it provide methods we can use on Yale?
- Does it have similar temporal analysis?
- What specific techniques can we apply to Yale?

---

### 8️⃣ CONNECTION TO OUR OBJECTIVES

**Our 5 objectives**:
1. Preprocessing pipeline (clean and align Yale scans)
2. Vision Transformer (learn temporal patterns from Yale)
3. LLM integration (explain changes in plain language)
4. Video generation (predict future progression)
5. Clinical validation (prove it works)

**Which objectives does this paper help?** (Check all that apply)
- [ ] Objective 1: Preprocessing
- [ ] Objective 2: Vision Transformer
- [ ] Objective 3: LLM
- [ ] Objective 4: Video generation
- [ ] Objective 5: Validation

**Specifically how**: [Explain in 2-3 sentences]

---

### 9️⃣ HOW IT COMBINES WITH OTHER PAPERS

**Papers we already analyzed**:
1. BraTS Toolkit - Preprocessing and tumor finding
2. nnU-Net - Baseline segmentation

**How this paper fits**:
- Does it overlap with papers above? If yes, which is better?
- Does it fill a gap the other papers don't cover?
- What ORDER should we use these papers? (First BraTS, then this, then...)

---

### 🔟 CITATION EXAMPLE

Write one sentence showing how we'd cite this in our paper:

"We adopted [specific method] from [Authors, Year], which [what it does]..."

---

## IMPORTANT REMINDERS

✅ Read ENTIRE PDF before answering
✅ Include exact numbers from results
✅ Keep explanations simple (10-year-old level)
✅ Focus on connection to Yale dataset
✅ Show how it fits with other papers
✅ No overwhelming code examples

---

## QUICK CHECKLIST AFTER ANALYSIS

Did you:
- [ ] Read beyond the abstract?
- [ ] Extract specific result numbers?
- [ ] Explain how it helps Yale dataset?
- [ ] Show which objectives it helps?
- [ ] Compare with papers we already have?
- [ ] Keep it simple and readable?

If all checked ✅, the analysis is complete!
