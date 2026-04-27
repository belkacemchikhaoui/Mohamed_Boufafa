# Phase 3 — Status Tracker

**Status:** 🔜 NOT STARTED  
**Last Updated:** April 5, 2026

---

## Activity Checklist

### Activity 1: Swin UNETR Segmentation (Week 8)

- [ ] Download Swin UNETR pretrained weights (`model_swinvit.pt`)
- [ ] Upload weights + data to Kaggle dataset
- [ ] Quick test: 15 epochs fold 0 → validate pipeline
- [ ] Train fold 0 (60 epochs)
- [ ] Train fold 1 (60 epochs)
- [ ] Train fold 2 (60 epochs)
- [ ] Record segmentation Dice (mean ± std across folds)
- [ ] Compare to Met-Seg (0.505) and SegResNet (0.368)

### Activity 2: Embedding Extraction + TaViT (Week 9)

- [ ] Extract 768-dim ViT embeddings for all 171 scans (3 folds)
- [ ] Validate: check norms, dimensionality, no NaN/Inf
- [ ] Compute time gaps from `cyprus_patient_timelines.csv`
- [ ] Implement TaViT temporal emphasis model
- [ ] Self-supervised pretraining (masked reconstruction)
- [ ] Fine-tune on downstream tasks (response prediction, volume change)
- [ ] Extract TaViT temporal embeddings (CLS token per patient)
- [ ] Save all embeddings to `Phase3/embeddings/`

### Activity 3: Evaluation Battery (Week 10)

- [ ] Run 16-test battery on ViT scan embeddings
- [ ] Run temporal tests on TaViT embeddings
- [ ] Produce 4-column comparison table (Radiomics vs CNN vs ViT vs TaViT)
- [ ] Generate evaluation summary figure
- [ ] Verify: ViT passes >12/16 tests?
- [ ] Verify: at least ONE temporal test improved?

### Activity 4: Visualization + Report (Week 11)

- [ ] t-SNE comparison: CNN vs ViT vs TaViT
- [ ] Temporal trajectory visualization
- [ ] Attention map extraction and visualization
- [ ] TEM learned curve plot (temporal emphasis vs time gap)
- [ ] Bar chart: all 16 tests across architectures
- [ ] Write Phase 3 Complete Report
- [ ] Phase 3 → Phase 4 justification

---

## Key Metrics to Track

| Metric | Target | Actual | Status |
|---|---|---|---|
| Swin UNETR Dice (mean) | >0.45 | — | ⬜ |
| ViT embedding tests passed | >12/16 | — | ⬜ |
| T1 (emb dist vs ΔVol) | r > 0.3 | — | ⬜ |
| T3 (ΔEmb→ΔVol) | R² > 0.1 | — | ⬜ |
| T4 (response prediction) | AUC > 0.6 | — | ⬜ |
| T5 (temporal coherence) | cos < 0.95 | — | ⬜ |
| TaViT total tests passed | >14/16 | — | ⬜ |

---

## Kaggle Session Log

| Session | Date | GPU | Account | Fold | Epochs | Status | Notes |
|---|---|---|---|---|---|---|---|
| S1 | — | T4 | Account 1 | 0 | 15 (quick test) | ⬜ | — |
| S2 | — | T4 | Account 1 | 0 | 60 | ⬜ | — |
| S3 | — | T4 | Account 2 | 1 | 60 | ⬜ | — |
| S4 | — | T4 | Account 3 | 2 | 60 | ⬜ | — |
| S5 | — | T4 | Any | — | — | ⬜ | Embedding extraction |
| S6 | — | T4/CPU | Any | — | — | ⬜ | TaViT training |
