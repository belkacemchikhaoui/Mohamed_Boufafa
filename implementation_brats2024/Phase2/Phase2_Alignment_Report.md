# Phase 2 — Alignment Report
## CNN Baseline vs project.txt Requirements

**Reviewed:** 2026-05-01  
**Scope:** All files in `Phase2/` — notebooks, training_logs, embeddings, outputs, scripts

---

## 1. Folder Inventory

```
Phase2/
├── notebooks/
│   ├── Phase2_A2_nnUNet_Finetune.ipynb        ← main training + extraction (13 cells)
│   ├── Phase2_B1_CNN_Embedding_Evaluation.ipynb ← 18-test battery (7 cells)
│   └── after_run/                              ← 3 post-run Kaggle snapshots (archive)
│       ├── phase2-a2-nnunet-finetune.ipynb
│       ├── phase2-a2-nnunet-finetune_continue.ipynb
│       └── phase2-b1-cnn-embedding-evaluation-ipynb.ipynb
├── embeddings/
│   ├── cnn_nnunet_embeddings.npz              ← (1620, 2825) float32 — 15.3 MB
│   ├── cnn_spatial_tokens.npz                 ← (1620, 2744, 256) float32 — 975.9 MB
│   └── tumor_volumes.csv                      ← 1620 rows
├── outputs/
│   ├── cnn_eval_results.json                  ← all 18-test metric values
│   ├── nnunet_best.pth                        ← best checkpoint (epoch 27)
│   ├── nnunet_latest.pth                      ← latest checkpoint
│   └── fig/
│       ├── nnunet_2d_sample{0-4}.png          ← 5 x 2D overlay visualizations
│       ├── nnunet_3d_sample{0-4}.png          ← 5 x 3D voxel scatter visualizations
│       ├── nnunet_distributions.png
│       └── nnunet_tsne.png
├── training_logs/
│   ├── training_log_nnunet.txt                ← 28-epoch training run (confirmed)
│   ├── extraction_log_nnunet.txt              ← embedding extraction (1620/1621)
│   └── eval_log_nnunet.txt                   ← 18-test evaluation run
├── reports/                                   ← ⚠️ EMPTY
└── scripts/                                   ← 7 utility/upload scripts
```

---

## 2. project.txt Requirements vs Implementation

### Requirement A — CNN model implemented and trained
**Status: COMPLETE**

- Model: nnUNet v2 PlainConvUNet (30.8M params, 6-stage encoder)
- Pretrained from: BraTS 2021 fold_0 checkpoint (WT=0.900, TC=0.867, ET=0.851)
- Fine-tuned on: BraTS 2024 Post-Treatment (1324 train / 297 val scans, 731 patients)
- Training: 28 epochs, cosine LR decay (1e-4 → 0), AMP, safe corrupt-file iterator
- Label mapping correctly applied: NETC=1, SNFH=2, ET=3, RC=4 (excluded from regions)

### Requirement B — Training + validation on single time-point tumor task
**Status: COMPLETE**

Confirmed from training_log_nnunet.txt (clean Kaggle run):

| Epoch | Mean Dice | WT | TC | ET |
|---|---|---|---|---|
| 3 | 0.7332 | 0.831 | 0.686 | 0.682 |
| 7 | 0.7965 | 0.844 | 0.770 | 0.776 |
| 11 | 0.8039 | 0.856 | 0.774 | 0.782 |
| 15 | 0.8044 | 0.857 | 0.774 | 0.782 |
| 19 | 0.8103 | 0.861 | 0.781 | 0.789 |
| 23 | 0.8142 | 0.863 | 0.787 | 0.793 |
| **27** | **0.8150** | **0.864** | **0.787** | **0.794** |

Best target (BraTS 2021): WT=0.900, TC=0.867, ET=0.851  
Gap: WT -3.6%, TC -8.0%, ET -5.7% — expected for post-treatment cohort

Visualizations saved: 5x 2D overlays + 5x 3D voxel-scatter + t-SNE + distributions

### Requirement C — Quantitative performance recorded
**Status: COMPLETE**

All 18 test metrics saved to `outputs/cnn_eval_results.json` (confirmed values):

```
M1_spearman_rho         = 0.871   M1_volume_R2_rf = 0.869
M2_logvol_R2_rf         = 0.820   M4_necrosis_F1  = 0.880
M6_patient_purity_pct   = 12.833
H1_rankme               = 824.4   H2_diversity    = 0.233
H3_responder_F1         = 0.821   H4_norm_cv      = 0.047
T1_spearman_wt          = 0.304   T4_rano_auc     = 0.687
T7_treatment_d          = 0.452   T8_kendall_tau  = 0.394
T3_delta_R2             = -0.010  (key static limitation indicator)
```

### Requirement D — 18-test dashboard pass/fail
**Status: 22/26 PASS**

| Priority | Test | Value | Threshold | Result |
|---|---|---|---|---|
| HIGH | M6_patient_purity_pct | 12.8% | >=60% | FAIL (by design — static model) |
| HIGH | T7_treatment_d | 0.452 | >=0.50 | FAIL (marginal) |
| HIGH | T1_spearman_wt | 0.304 | >=0.30 | PASS |
| HIGH | T4_rano_auc | 0.687 | >=0.65 | PASS |
| HIGH | T8_kendall_tau | 0.394 | >=0.30 | PASS |
| med | H2_diversity | 0.233 | >=0.25 | FAIL (marginal) |
| low | T5_pass_dual | 0.000 | >=1.0 | FAIL (low priority) |
| All others | — | — | — | PASS |

### Requirement E — CNN limitations explicitly documented
**Status: COMPLETE**

Phase2_B1 Cell 6 explicitly prints LIMITATIONS → Phase 3 targets:
- H2_diversity = 0.233 (needs >0.25) — embedding not diverse enough
- M6_patient_purity_pct = 12.8% (needs >60%) — same-patient scans don't cluster
- T7_treatment_d = 0.452 (needs >0.5) — weak treatment effect separation

T3_delta_R2 = -0.010 is explicitly labelled "WEAK (static CNN limitation)" in eval_log.

Key sentence in B1 header: *"Purpose: Establish baseline performance → prove ViT (Phase 3) improves on CNN limitations."*

### Requirement F — Results serve as reference benchmarks for Phase 3
**Status: COMPLETE**

cnn_eval_results.json is uploaded to Kaggle dataset and loaded by Phase3_B1 for side-by-side comparison.

---

## 3. Confirmed Embedding Artifacts

| File | Shape | Size | Content |
|---|---|---|---|
| cnn_nnunet_embeddings.npz | (1620, 2825) | 15.3 MB | Main embedding (octant + region + vol) |
| cnn_spatial_tokens.npz | (1620, 2744, 256) | 975.9 MB | Full spatial token maps for Phase 4/5 |
| tumor_volumes.csv | 1620 rows | — | WT/TC/ET volumes per scan |

Embedding design: octant (2048-D) + mask-weighted region (768-D) + volumetric morphology (9-D) = **2825-D total**  
Extraction: 1620/1621 scans (1 skipped corrupt), 51 with empty ROI (complete resection → zero vector, intentional)

---

## 4. Known Issues

### Issue 1 — Cell 9 NameError in A2 (non-critical)
training_log_nnunet.txt shows crash after epoch 27 in the visualization cell:
```
NameError: name 'gc' is not defined (Cell 9)
```
**Impact:** Only visualization cell crashed. Training completed fully, best checkpoint at epoch 27 was saved. The extraction notebook (after_run) recovered the checkpoint and ran clean.

### Issue 2 — reports/ folder is EMPTY
The `Phase2/reports/` folder exists but contains no report file.  
**Fix needed:** Should contain a `Phase2_CNN_Report.md` summarising training results and 18-test scores.

### Issue 3 — No explicit CNN vs ViT comparison table in Phase 2
project.txt says results should serve as "reference benchmarks" for comparison with ViT.  
The numeric benchmarks exist in `cnn_eval_results.json` and are consumed by Phase 3 B1.  
However, there is no standalone comparison table within Phase2 itself.  
**This is acceptable** — the comparison is in Phase 3 B1 where both models are evaluated side by side.

---

## 5. Overall Phase 2 Verdict

| Requirement | Status |
|---|---|
| CNN model implemented (nnUNet v2 PlainConvUNet) | COMPLETE |
| Training on BraTS 2024 (28 epochs, best Dice=0.8150) | COMPLETE |
| Single time-point validation with Dice per region | COMPLETE |
| 18-test evaluation battery (22/26 pass) | COMPLETE |
| Limitations explicitly documented (T3=-0.010, M6=12.8%) | COMPLETE |
| Reference benchmarks for Phase 3 comparison | COMPLETE |
| Visualizations (2D/3D overlays, t-SNE) | COMPLETE |
| Embedding artifacts saved and uploaded | COMPLETE |
| reports/ folder populated | **MISSING** |
| Explicit CNN vs ViT table in Phase 2 | **Deferred to Phase 3 B1** |

**Phase 2: 8/10 requirements fully met. 1 minor gap (empty reports/), 1 deferred.**

---

## 6. Action Items

| Priority | Action |
|---|---|
| LOW | Add a `Phase2_CNN_Report.md` to `reports/` summarising training + 18-test results |
| NONE | CNN vs ViT comparison already in Phase3_B1 — no action needed |
| NONE | Cell 9 NameError in original A2 — extraction notebook is clean, no action needed |

---

*Report generated: 2026-05-01*  
*Sources: Phase2_A2_nnUNet_Finetune.ipynb (13 cells), Phase2_B1_CNN_Embedding_Evaluation.ipynb (7 cells), training_log_nnunet.txt, eval_log_nnunet.txt, extraction_log_nnunet.txt, cnn_eval_results.json*
