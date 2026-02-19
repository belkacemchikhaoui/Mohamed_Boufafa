# 📋 QUICK REFERENCE - Your Research at a Glance

## 🎯 YOUR GOAL
Build AI that tracks brain tumor changes over time, explains what's happening, and predicts future progression.

---

## 📊 YOUR MAIN DATASET
**Yale Longitudinal Brain Metastases (2025)**
- 11,884 scans
- 1,430 patients  
- Avg 8 scans per patient
- FREE download: TCIA

---

## ✅ PAPERS ANALYZED (4/~20)

| # | Paper | What It Gives You | For Yale |
|---|-------|-------------------|----------|
| 1 | BraTS Toolkit | Preprocessing methods | Clean Yale scans |
| 2 | nnU-Net | Tumor segmentation | Find tumors in Yale |
| 3 | Yale Dataset | YOUR DATA | Everything! |
| 4 | Registration | Align over time | T0→T1→T2... lined up |

---

## ⏳ PAPERS NEEDED

| Category | Count | Purpose |
|----------|-------|---------|
| Preprocessing | 2-3 | Align scans, harmonize scanners |
| Vision Transformers | 4-5 | Temporal tracking |
| LLMs | 3-4 | Generate explanations |
| Diffusion/Video | 3-4 | Predict future |
| Validation | 2-3 | Prove it works |
| **TOTAL** | **~20** | **Complete literature** |

---

## 🔄 YOUR WORKFLOW

```
Yale Data → BraTS Clean → nnU-Net Segment → Register T0→T1→T2 → YOUR MODELS
                                                                      ↓
                                                               ViT + LLM + Video
```

---

## 📝 HOW TO ANALYZE PAPERS

1. Open: `SIMPLE_ANALYSIS_PROMPT.md`
2. Copy prompt
3. Give to AI + PDF
4. Save result
5. Update tracker

**10 minutes per paper!**

---

## 🎯 NEXT 3 ACTIONS

1. ⏳ Analyze Paper 9 (ComBat Harmonization)
2. ⏳ Analyze Paper 13 ("Explainable Progression")
3. ⏳ Find ViT papers for temporal analysis

---

## 💾 STORAGE NEEDED
- Yale: 200 GB
- BraTS: 100 GB
- Processed: 300 GB
- **Total: 600 GB** (get 1TB drive)

---

## ⏰ TIMELINE
- **Weeks 1-4**: Preprocess Yale
- **Weeks 5-11**: Train ViT
- **Weeks 12-14**: Add LLM
- **Weeks 15-18**: Generate videos
- **Weeks 19-20**: Validate

---

## 🔑 KEY FILES

| File | Purpose |
|------|---------|
| `CURRENT_STATUS.md` | Where you are now |
| `SIMPLE_ANALYSIS_PROMPT.md` | Copy-paste for papers |
| `PAPERS_TRACKER.md` | Progress tracker |
| `MASTER_PLAN.md` | Full research plan |

---

## 💡 REMEMBER

✅ Yale = Your ONLY necessary dataset
✅ Foundation nearly complete (4/5 papers done!)
✅ Simple system = No overwhelming code
✅ Registration solved = Can align temporal sequences!
✅ One paper at a time = Steady progress

---

**NEXT**: Tell me which paper to analyze!

**Suggested**: Paper 13 - "Explainable Disease Progression"
