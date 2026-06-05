# Volume Anchor Features, Information Leakage, and Clinical Pipeline Interpretation

# Introduction

A critical methodological consideration in the TaViT framework concerns the use of segmentation-derived volumetric anchor features (`vol_feats`) within the downstream longitudinal transformer architecture.

These features contain:
- log1p(WT volume)
- log1p(TC volume)
- log1p(ET volume)
- tumour burden ratios
- additional volumetric descriptors extracted from nnU-Net representations.

Since TaViT also predicts WT, TC, and ET tumour volumes, an important scientific question arises regarding whether these features constitute target leakage or whether they represent a legitimate component of a clinically motivated hierarchical pipeline.

This section discusses the methodological implications of these design choices and provides a scientifically rigorous interpretation of the framework.

---

# Is the Use of `vol_feats` Fundamentally Wrong?

No. The use of volumetric anchor features is not inherently invalid.

In the current framework, TaViT is designed as a multi-stage longitudinal clinical pipeline:

```text
MRI
 → nnU-Net segmentation
 → segmentation-derived embeddings + volumetric descriptors
 → TaViT longitudinal reasoning
 → trajectory prediction / treatment-aware modeling
```

This architecture is conceptually similar to many real-world medical AI systems where:
- segmentation is first performed,
- structural measurements are extracted,
- and downstream models perform prognostic or temporal reasoning.

Examples of similar hierarchical medical pipelines include:
- lesion detection → prognosis prediction,
- segmentation → radiomics → survival modelling,
- tumour delineation → recurrence forecasting,
- or volumetric analysis → treatment response estimation.

Therefore, the use of segmentation-derived priors is clinically realistic and scientifically defensible when appropriately framed.

---

# Why Reviewers May Raise Leakage Concerns

Although the pipeline itself is valid, the current implementation introduces a methodological concern because the decoder receives information that is highly correlated with the prediction targets.

Specifically:
- the model predicts WT, TC, and ET tumour volumes,
- while simultaneously receiving explicit volume-derived features as input.

Mathematically, this means the decoder may partially rely on:
- already estimated tumour burden,
- direct scale information,
- or segmentation-derived target-adjacent descriptors.

Consequently, reviewers may argue that:

> “The model may be refining or smoothing already-known tumour volumes rather than independently learning longitudinal tumour dynamics from imaging and temporal context.”

This criticism becomes especially relevant if the model is presented as:
- a pure imaging-based predictor,
- or a fully latent representation learner.

Therefore, the issue is not that the architecture is invalid, but rather that the scientific interpretation must be carefully defined.

---

# Two Distinct Scientific Interpretations

The methodological interpretation of TaViT depends on the scientific claim being made.

## Interpretation A — Pure Predictive Imaging Intelligence

Under this interpretation, TaViT would claim to:
- infer tumour trajectories directly from latent imaging representations,
- learn temporal disease dynamics,
- and predict tumour evolution independently from explicit volumetric priors.

If this interpretation is adopted, then:
- strict no-anchor evaluation becomes essential,
- and strong performance without `vol_feats` must be demonstrated.

Otherwise, the model may be viewed as partially shortcutting the learning process.

---

## Interpretation B — Clinically Integrated Longitudinal Pipeline

Under the second interpretation, TaViT is viewed as:

> “A clinically informed longitudinal reasoning system operating on segmentation-derived structural priors, treatment history, molecular information, and temporal context.”

Under this framing:
- the volumetric features become clinically motivated priors,
- rather than unintended leakage.

This interpretation is significantly more defensible in real clinical settings because:
- clinicians already have tumour measurements available during follow-up,
- segmentation systems are commonly integrated into downstream workflows,
- and longitudinal decision-making frequently depends on prior volumetric estimates.

Thus, the architecture becomes:
- hierarchical,
- modular,
- and clinically realistic.

---

# Recommended Thesis Strategy

The strongest scientific strategy is not to remove the volumetric features entirely, but rather to evaluate both settings.

---

# Recommended Dual-Model Evaluation

## 1. Full Clinical Pipeline (Current TaViT)

```text
MRI
→ nnU-Net
→ embeddings + volume descriptors
→ TaViT
→ longitudinal trajectory prediction
```

This version represents:
- the highest-performing clinical system,
- a segmentation-informed longitudinal predictor,
- and a clinically assisted trajectory modelling pipeline.

Possible naming conventions include:
- “Volume-Anchored TaViT”
- “Segmentation-Informed TaViT”
- “Clinical TaViT Pipeline”

This model should remain the primary system for practical clinical evaluation.

---

## 2. Strict No-Anchor TaViT

A second version should remove:
- WT/TC/ET volume descriptors,
- volume ratios,
- and other target-adjacent volumetric priors.

The model would then rely only on:
- latent scan embeddings,
- treatment tokens,
- molecular markers,
- and temporal encoding.

This configuration evaluates whether the transformer backbone truly learns:
- imaging-temporal dynamics,
- treatment-conditioned progression patterns,
- and longitudinal disease representations.

---

# Why This Strengthens the Thesis

Evaluating both settings substantially improves scientific rigor.

Instead of presenting only one architecture, the thesis can demonstrate:

| Model Version | Scientific Purpose |
|---|---|
| Full volume-anchored TaViT | Best clinical performance |
| Strict no-anchor TaViT | Validation of genuine temporal representation learning |

This creates a much stronger narrative because:
1. the full pipeline demonstrates practical clinical utility,
2. while the no-anchor version validates the transformer’s true temporal reasoning capability.

---

# What Would Be an Ideal Outcome?

An ideal result would look similar to the following:

| Configuration | Example WT R² |
|---|---|
| Full model | 0.91 |
| No-anchor model | 0.78–0.84 |

Such a result would demonstrate:
- the transformer genuinely learns meaningful longitudinal structure,
- while volume anchors provide clinically useful refinement.

This would strongly support the architectural design.

---

# What Would Be Concerning?

A potentially concerning outcome would be:

| Configuration | Example WT R² |
|---|---|
| Full model | 0.91 |
| No-anchor model | 0.35 |

In this scenario, the model would appear heavily dependent on explicit volumetric shortcuts.

However, even in this case:
- the architecture would not become invalid,
- but the interpretation would shift toward:
  - “clinical refinement pipeline”
  - rather than
  - “pure temporal imaging intelligence.”

This distinction is important for accurate scientific communication.

---

# Recommended Thesis Wording

The following wording provides a scientifically balanced interpretation:

> “The volumetric anchor features were intentionally incorporated as segmentation-derived structural priors within a clinically motivated hierarchical pipeline. While these features are correlated with the prediction targets, they provide explicit tumour burden information analogous to measurements available during real clinical follow-up workflows. To assess dependence on these priors, strict no-anchor ablation experiments were additionally conducted.”

This formulation:
- acknowledges the correlation,
- avoids overstating the model’s independence,
- and demonstrates methodological transparency.

---

# Final Assessment

The current TaViT design is not methodologically invalid.

On the contrary:
- the multimodal longitudinal transformer design is strong,
- the treatment-aware conditioning is clinically meaningful,
- and the hierarchical segmentation-to-trajectory pipeline is realistic.

The key requirement is transparency.

The thesis should therefore:
1. clearly distinguish between:
   - segmentation-informed prediction,
   - and purely latent prediction;
2. include strict no-anchor ablation studies;
3. avoid overstating the model as fully independent from volumetric priors when such priors are explicitly included.

If these points are addressed rigorously, the resulting framework becomes substantially more scientifically robust and reviewer-resistant.
