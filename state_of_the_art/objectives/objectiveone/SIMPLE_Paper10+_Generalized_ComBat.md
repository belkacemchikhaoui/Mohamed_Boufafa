# Paper 10+: Generalized ComBat for Radiomic Features (2022)

## 1. One-Sentence Summary
Two new ComBat methods (Nested ComBat for multiple batch effects simultaneously, GMM ComBat for unknown batch effects and bimodal distributions) validated on lung CT radiomics with improved harmonization over standard ComBat.

## 2. Key Results (Numbers!)
- **2 public lung CT datasets**: Lung3 and Radiogenomics with different reconstruction kernels, manufacturers, contrast protocols
- **Nested ComBat**: Similar performance to standard ComBat for known batch effects
- **NestedD ComBat** (dropping bad features): -2% to -32% fewer significant differences than standard ComBat
- **GMM ComBat**: -11% and -10% improvement over standard ComBat for kernel harmonization (Lung3 dataset)
- **Multiple batch effects**: Nested method handles 3 simultaneously (contrast, kernel, manufacturer)
- **Bimodal distributions**: 14-28% of radiomic features have bimodal distributions (GMM ComBat fixes this)
- **Survival analysis**: NestedD features → c-statistic 0.63-0.64 (vs 0.59-0.62 original features)
- **False positive rate**: Not reported, but survival analysis suggests methods preserve clinical utility

## 3. What's New
- **FIRST multi-batch ComBat**: Standard ComBat handles ONE batch effect at a time, Nested handles MULTIPLE sequentially
- **FIRST for unknown batch effects**: GMM automatically discovers hidden batch effects (e.g., unknown scanner differences)
- **FIRST for bimodal features**: GMM splits bimodal distributions into Gaussian components before harmonization
- **Radiomic features**: Validated on 100+ texture features (GLCM, GLRLM, etc.) not just simple measurements
- **Sequential harmonization**: Nested method chains ComBat calls (harmonize by contrast → then kernel → then manufacturer)
- **Automatic grouping**: GMM finds scan subgroups without knowing imaging parameters

## 4. Limitations
- **Lung CT only**: Validated on lung cancer, not brain metastases (different tissue, contrast patterns)
- **14-28% features dropped**: NestedD drops features with persistent batch effects (loses information)
- **No ground truth**: Unlike Paper 10++, no traveling subjects → can't prove biology preserved
- **Bimodal assumption**: GMM assumes 2 groups (what if 3+ unknown batch effects?)
- **Computational cost**: Nested method requires multiple ComBat runs (slower than standard)
- **Order matters**: Nested harmonization order affects results (which batch effect to harmonize first?)
- **Small improvement**: GMM ComBat only 10-11% better than standard (not revolutionary)

## 5. Methods (Simple Steps)

### Nested ComBat (for multiple known batch effects):
1. **Harmonize by batch effect 1** (e.g., contrast enhancement): Remove contrast-related variability
2. **Harmonize by batch effect 2** (e.g., kernel): Remove kernel-related variability from step 1 output
3. **Harmonize by batch effect 3** (e.g., manufacturer): Remove manufacturer variability from step 2 output
4. **NestedD variant**: After each step, DROP features that still show significant batch effect (more aggressive)

### GMM ComBat (for unknown batch effects + bimodal data):
1. **Detect bimodal features**: Find radiomic features with 2 peaks in distribution
2. **Fit Gaussian Mixture Model**: Automatically split scans into 2 groups based on feature patterns
3. **Harmonize by GMM groups first**: Treat discovered groups as "batch effect" → remove with ComBat
4. **Then harmonize by known batches**: Remove contrast, kernel, manufacturer effects afterward
5. **Result**: Both unknown + known batch effects removed

## 6. Code/Resources Available
- **Not released yet**: Methods described in paper but no public GitHub repository found
- **Uses standard ComBat**: Builds on neuroCombat R package (available)
- **GMM**: Uses standard Gaussian Mixture Models (scikit-learn Python, mclust R)
- **Radiomic extraction**: CapTK and PyRadiomics (both open source)
- **Lung datasets**: Lung3 (211 patients) and Radiogenomics (89 patients) publicly available
- **Survival analysis**: Uses Cox proportional hazards models (survival R package)

## 7. Connection to Yale Dataset
- **Multiple batch effects in Yale**: Site + scanner + year + field strength + protocol = 5+ batch effects!
- **Unknown effects in 20 years**: 2004-2023 = software upgrades, reconstruction changes not documented → GMM can find them
- **Radiomic features**: If extracting tumor texture (GLCM, GLRLM) → this paper directly applies
- **Order for Yale**: Nested ComBat → harmonize by year → field strength → site → manufacturer → protocol
- **Bimodal tumor features**: Metastases from different primary cancers (lung vs breast) might have bimodal intensity/texture → GMM handles this
- **Cyprus validation**: Could test if Nested/GMM ComBat preserves expert-labeled tumor subregions (enhancing vs necrosis vs edema)

## 8. Connection to 5 Research Objectives
1. **Automatic tumor tracking** ✅✅: Nested ComBat removes multiple confounds (scanner + site + time) → cleaner tracking
2. **Predict future changes** ✅: GMM discovers hidden patterns → might reveal tumor subtypes with different trajectories
3. **LLM reports** ✅✅: "Tumor texture changed" = real biology, not scanner artifact (after Nested ComBat)
4. **Video generation** ✅: Nested harmonization removes scanner changes across video frames → smoother evolution
5. **Clinical decision support** ✅✅: GMM might discover prognostic subgroups (fast-growing vs slow-growing tumors)

## 9. How It Combines With Other Papers
- **Extends Papers 9 + 10**: Standard ComBat = 1 batch effect, Nested = multiple batch effects, longCombat + Nested = multiple batch effects over time
- **After radiomic extraction**: 
  1. nnU-Net segments tumor
  2. Extract 110+ radiomic features (like Cyprus dataset has)
  3. Apply Nested ComBat to harmonize by all Yale's batch effects
- **Before Phase 2 (ViT)**: Nested ComBat → feed harmonized radiomic features to ViT as additional input channels
- **GMM for subtyping**: Use GMM to discover brain metastases subtypes (might correlate with primary tumor type: lung/breast/melanoma)
- **With Paper 10++ validation**: Nested ComBat + traveling subject validation = prove multi-batch harmonization works
- **Processing order**:
  1. Preprocess → FLIRE → nnU-Net
  2. Extract radiomic features (PyRadiomics)
  3. **Nested ComBat** (harmonize by year → site → scanner → protocol)
  4. Validate on Cyprus (do expert labels still match after harmonization?)
  5. Feed to ViT (Phase 2)

## 10. Citation
Horng, H., Singh, A., Yousefi, B., Cohen, E.A., Haghighi, B., Katz, S., Noël, P.B., Shinohara, R.T., Kontos, D. (2022). Generalized ComBat harmonization methods for radiomic features with multi-modal distributions and multiple batch effects. *Scientific Reports*, 12, 4493. https://doi.org/10.1038/s41598-022-08412-9

---

## Comparison: Standard vs Nested vs GMM ComBat

| Method | Batch Effects | Unknown Effects? | Bimodal Data? | When to Use |
|--------|---------------|------------------|---------------|-------------|
| **Standard ComBat** (Paper 9) | 1 at a time | ❌ No | ❌ No | Simple datasets, 1 known batch effect |
| **Longitudinal ComBat** (Paper 10) | 1 batch + time | ❌ No | ❌ No | Repeated measures, 1 known batch effect |
| **Nested ComBat** (this paper) | Multiple sequential | ❌ No | ⚠️ Partial | Complex datasets, multiple known batch effects |
| **GMM ComBat** (this paper) | 1 known + 1 unknown | ✅ Yes | ✅ Yes | Unknown confounds, bimodal features |
| **NestedD ComBat** (this paper) | Multiple, drops bad features | ❌ No | ❌ No | When losing 14-28% features acceptable |

## Yale-Specific Harmonization Strategy

### Step 1: Identify All Batch Effects
```
Yale batch effects:
1. Year (2004-2023) → 20-level categorical
2. Site (if multi-site TCIA) → N-level categorical  
3. Field strength (1.5T vs 3.0T) → 2-level
4. Manufacturer (Siemens/GE/Philips) → 3-level
5. Protocol (sequence parameters) → Unknown levels
```

### Step 2: Choose Harmonization Order
**Recommendation: Harmonize in order of importance**
```
Nested ComBat order for Yale:
1. Year (2004→2023, biggest technology change)
2. Field strength (1.5T vs 3.0T, big signal difference)
3. Manufacturer (Siemens/GE/Philips)
4. Site (if applicable)
5. GMM for unknown effects (software updates, etc.)
```

### Step 3: Validate Each Step
After each Nested ComBat step, check on Cyprus:
- Do expert tumor labels still match automated segmentations?
- Do tumor subregions (enhancing/necrosis/edema) preserve their characteristics?
- Do known treatment responses (radiotherapy) still visible in data?

## Advanced Application: Tumor Subtype Discovery

**GMM ComBat side benefit**: Might discover prognostically relevant tumor subgroups!

```
GMM analysis on Yale:
1. Extract 110+ radiomic features (like Cyprus)
2. Apply GMM → discovers 2 groups
3. Check if groups correlate with:
   - Primary tumor type (lung/breast/melanoma)
   - Survival outcomes
   - Treatment response patterns
4. If clinically meaningful → use for stratification in Phase 2 training
```

## Critical Decision Point

**Should you use NestedD (drops 14-28% features)?**

**YES if**: 
- Features with persistent batch effects might confuse ViT
- Prefer fewer, cleaner features
- 110+ radiomic features (can afford to lose 28%)

**NO if**:
- Every feature carries biological information
- Using simple features (volumes, intensities only)
- ViT architecture can learn to ignore batch effects

**For Yale**: Probably YES for radiomic features, NO for simple volume/intensity measurements
