# 🎯 COPY-PASTE PROMPT FOR ANALYZING RESEARCH PAPERS

**Copy the text below and paste it with any research paper PDF:**

---

## PROMPT START

I need you to analyze this research paper completely. **Read the ENTIRE PDF** from first page to last page without cutting or skipping sections. Extract text from the full document before analyzing.

### Analysis Format (Explain Like I'm 10 Years Old):

**1. What Problem Did They Try to Solve?**
- In 2-3 simple sentences, what was missing or broken before this research?

**2. What Did They Actually Do?**
- Describe their solution simply (avoid jargon)
- What tools, methods, or techniques did they use?
- How did they test it?

**3. What Results Did They Get?**
- List specific numbers, scores, or measurements
- Did it work well? How well compared to what existed before?
- Include any tables or figures showing results

**4. What's New Here?**
- What did this paper add that nobody else did before?
- Why is this important or special?

**5. What Are the Limitations?**
- What doesn't work well?
- What problems remain unsolved?
- What did the authors say needs more work?

**6. Code & Resources**
- Is code available? Provide exact links
- What datasets did they use? (with names and sizes)
- Can someone reproduce this work?

**7. How Does This Connect to Our Project?**

**Our project context:**
> We're building an AI system to help doctors understand how cancer tumors change over time. It will:
> - Track tumor changes across multiple scans (months/years)
> - Use Vision Transformers to spot patterns over time
> - Use Large Language Models to explain what's happening in plain language
> - Generate videos showing what might happen with different treatments
> - Give doctors visual and text explanations they can understand

Answer:
- Which parts of this paper can we use directly?
- What specific methods, tools, or ideas help our goals?
- What would we need to modify or add on top of their work?
- Does this overlap with any preprocessing, segmentation, temporal modeling, explanation, or prediction tasks?

**8. Simple Summary**
- Write 3-4 sentences that a 10-year-old could understand
- Explain: What they did, why it matters, what we learn from it

**9. How to Cite Without Stealing**
- Write one example sentence showing how we'd reference this paper in our work
- Example format: "We adopted [specific technique] from [Authors, Year], which [what it does]..."

**10. Where Does This Fit?**
Does this paper help with:
- [ ] Image preprocessing/cleaning
- [ ] Finding tumors (segmentation)
- [ ] Tracking changes over time (temporal modeling)
- [ ] Explaining results (interpretability)
- [ ] Predicting future (generative modeling)
- [ ] Something else? (describe)

---

**IMPORTANT INSTRUCTIONS:**
- Read the COMPLETE PDF (don't just read abstract)
- Extract ALL pages before analyzing
- Include exact numbers from results tables
- Note any code repositories or GitHub links
- Keep explanations simple (10-year-old level)
- If technical terms are necessary, explain them in parentheses

## PROMPT END

---

## How to Use This Prompt:

### Step 1: Find Your Paper
Locate the PDF file you want to analyze

### Step 2: Copy the Prompt
Copy everything between "PROMPT START" and "PROMPT END"

### Step 3: Give to AI
Paste the prompt and attach the PDF (or provide the path to the PDF)

### Step 4: Review the Output
The AI will give you a simple analysis following the 10-point format

### Step 5: Add to Your Collection
Save the analysis with a simple filename like:
- `SIMPLE_Paper3_[PaperName].md`
- `SIMPLE_Paper4_[PaperName].md`

### Step 6: Update Connection Guide
Add the new paper to `SIMPLE_How_Papers_Connect.md` using the template provided there

---

## Example Command (if using AI with file access):

```
"Use the prompt in SIMPLE_PAPER_ANALYSIS_PROMPT.md to analyze 
the paper located at: ./papers/new_paper.pdf

Read the entire PDF before analyzing."
```

---

## Tips for Best Results:

✅ **DO**:
- Make sure the AI reads the FULL PDF (check it extracted all pages)
- Ask follow-up questions if something is unclear
- Compare with papers you already analyzed to spot overlaps
- Keep the analysis simple and readable

❌ **DON'T**:
- Don't let the AI just read the abstract
- Don't accept analysis with "..." or "sections omitted"
- Don't include code examples in the simple analysis (just describe what the code does)
- Don't use complex mathematical notation (explain concepts in words)

---

## Quality Check:

After getting the analysis, verify:
- [ ] Did it read beyond just the abstract?
- [ ] Are there specific result numbers included?
- [ ] Is it written simply (could a 10-year-old understand)?
- [ ] Does it connect clearly to your project?
- [ ] Are there proper citations/references?

If any checkbox is NO, ask the AI to redo that section.
