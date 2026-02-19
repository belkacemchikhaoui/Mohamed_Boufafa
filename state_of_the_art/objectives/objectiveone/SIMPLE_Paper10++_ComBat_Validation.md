# Paper 10++: Validation of Cross-Sectional and Longitudinal ComBat (2022)

## 1. One-Sentence Summary
First validation of neuroCombat, longCombat, and gamCombat using "traveling subjects" (73 people scanned 161 times across 4 scanners) proving harmonization removes scanner effects for diffusion MRI but can HARM structural data if no scanner effect exists.

## 2. Key Results (Numbers!)
- **73 healthy volunteers**, 161 total scans across 2 sites, 4 machines
- **Ground truth**: Same person scanned twice within 180 days = real test of scanner effects
- **Structural data (volumes, cortical thickness)**: NO scanner effect detected → harmonization HARMFUL in some regions!
- **Diffusion data (FA, MD)**: LARGE scanner effects → all 3 methods successfully harmonized
- **False positive rate**: < 5% for all methods (safe to use)
- **LongCombat**: Slightly better at avoiding false positives than neuroCombat/gamCombat
- **7 brain regions tested**: Ventricles, cortical gray matter, white matter, deep gray, cerebellar regions, brainstem
- **5 diffusion protocols**: Single-shell (32-64 directions) and multi-shell (12-98 directions) acquisitions

## 3. What's New
- **FIRST traveling subject validation**: Previous studies assumed harmonization worked, this PROVES it with ground truth
- **WARNING discovered**: Harmonization can make things WORSE if no scanner effect exists!
- **All 3 methods validated**: neuroCombat, longCombat, gamCombat tested on structural AND diffusion data
- **Cross-sectional + longitudinal**: Tested both data types (previous studies only one or the other)
- **Common pipeline**: All scans processed identically (FSL + MALP-EM + ANTs) for fair comparison
- **Clinical recommendation**: Don't harmonize blindly → TEST for scanner effect first!

## 4. Limitations
- **Healthy volunteers only**: No tumor patients → doesn't prove harmonization works for brain metastases
- **Small sample**: 73 people across 4 scanners (Yale has 1,430 people, 20+ years of scanners)
- **Short timespan**: Scans < 180 days apart (Yale tracks patients for years)
- **Different disease**: Healthy brains stable over 180 days, tumors grow/shrink/respond to treatment
- **Structural data caveat**: Their T1 protocols were very consistent → Yale's 20 years of protocols may have larger scanner effects
- **No tumor segmentation**: Didn't test if nnU-Net segmentation volumes need harmonization

## 5. Methods (Simple Steps)
1. **Traveling subject design**: Scan same 73 people on 4 different scanners within 6 months
2. **Extract features**: Get volumes, cortical thickness (T1), fractional anisotropy, mean diffusivity (diffusion)
3. **Within-scanner baseline**: Measure scan-rescan variability when person stays on SAME scanner (2% noise)
4. **Across-scanner test**: Measure scan-rescan variability when person switches scanners (3% noise)
5. **Calculate scanner effect**: Extra noise when switching scanners = scanner effect (3% - 2% = 1% scanner effect)
6. **Harmonize**: Apply neuroCombat, longCombat, gamCombat separately
7. **Compare**: Does harmonization reduce across-scanner noise to match within-scanner noise?
8. **False positive check**: Randomly label people "Group A" vs "Group B" → should find NO difference (tests overfitting)

## 6. Code/Resources Available
- **R packages**: neuroCombat_1.0.13, longCombat_0.0.0.90000, neuroHarmonize (gamCombat)
- **Processing pipeline**: FSL for diffusion, MALP-EM for parcellation, ANTs DiReCT for cortical thickness
- **Public**: Code/methods described in detail, though data access restricted (healthy volunteers)
- **Coefficient of variation**: Main metric = standard deviation / mean * 100 (measures scan-rescan reliability)

## 7. Connection to Yale Dataset
- **CRITICAL WARNING**: Test for scanner effect BEFORE harmonizing Yale data!
- **How to test**: 
  1. Pick patients scanned on same scanner twice (no scanner change)
  2. Pick patients scanned on different scanners (scanner change)
  3. Compare variability → if significantly different, scanner effect exists
- **Expect scanner effects in Yale**: 20 years (2004-2023) vs this study's consistent protocols
- **Diffusion analogy**: Yale's tumor volumes/intensities like diffusion data (large scanner effects expected)
- **Validation strategy**: Use Cyprus 40-patient dataset as Yale's "traveling cohort" → same patients, different scanners (3T vs 1.5T)

## 8. Connection to 5 Research Objectives
1. **Automatic tumor tracking** ✅✅: Validation proves harmonization works for longitudinal data (if scanner effect exists)
2. **Predict future changes** ✅: Warning about false biology → ensures predictions based on real tumor patterns
3. **LLM reports** ✅✅: Avoids "hallucinated" tumor changes from over-harmonization
4. **Video generation** ✅: Validated longitudinal harmonization = smooth temporal videos
5. **Clinical decision support** ⚠️: WARNING = don't blindly harmonize → test first, or risk false alarms!

## 9. How It Combines With Other Papers
- **Validates Paper 9 + 10**: Proves ComBat methods actually work when scanner effects exist
- **Before applying Paper 9/10 to Yale**: Use this paper's method to TEST if Yale has scanner effects (spoiler: 20 years = definitely yes)
- **Cyprus validation role**: Use Cyprus as Yale's "traveling subjects" → validate harmonization preserves tumor biology
- **After Phase 1 preprocessing**: Run this validation check before Phase 2 (ViT training)
- **Quality control**: Add to processing pipeline → detect if harmonization making things worse
- **Processing order**: 
  1. Preprocess (BraTS → FLIRE → nnU-Net) 
  2. Extract features 
  3. **TEST for scanner effects (this paper's method)** ← NEW STEP!
  4. If effects exist → Apply ComBat (Paper 9/10)
  5. If no effects → Don't harmonize (this paper's warning)

## 10. Citation
Richter, S., Winzeck, S., Correia, M.M., Kornaropoulos, E.N., Manktelow, A., Outtrim, J., Chatfield, D., Posti, J.P., Tenovuo, O., Williams, G.B., Menon, D.K., Newcombe, V.F.J. (2022). Validation of cross-sectional and longitudinal ComBat harmonization methods for magnetic resonance imaging data on a travelling subject cohort. *NeuroImage: Reports*, 2, 100136. https://doi.org/10.1016/j.ynirp.2022.100136

---

## Key Validation Results Summary

| Data Type | Scanner Effect? | Harmonization Helpful? | Best Method |
|-----------|----------------|----------------------|-------------|
| **Structural (volumes)** | NO (p > 0.05) | ❌ NO - Can be harmful! | Don't harmonize |
| **Structural (cortical thickness)** | NO (p > 0.05) | ❌ NO - Can be harmful! | Don't harmonize |
| **Diffusion (FA)** | YES (large, p < 0.001) | ✅ YES - All methods work | neuroCombat or gamCombat |
| **Diffusion (MD)** | YES (large, p < 0.001) | ✅ YES - All methods work | neuroCombat or gamCombat |
| **Longitudinal diffusion** | YES | ✅ YES - All methods work | longCombat (slightly safer) |

## Application to Yale Brain Metastases Project

### MUST DO: Test for Scanner Effects First!
```
Yale Test Strategy:
1. Identify patients with stable disease (no treatment change between scans)
2. Group A: Same scanner for consecutive scans
3. Group B: Different scanner for consecutive scans
4. Compare tumor volume variability: Group A vs Group B
5. If Group B >> Group A → Scanner effect exists → Harmonize!
6. If Group B ≈ Group A → No scanner effect → Don't harmonize!
```

### Why Yale LIKELY Needs Harmonization:
- **20 years of technology** (2004-2023) vs study's consistent protocols
- **Tumor measurements** more like diffusion (large scanner effects) than healthy cortical thickness
- **Multiple sites** in TCIA collection
- **Treatment effects** mixed with scanner effects (need to separate)

### Cyprus as Validation Dataset:
- **40 patients with expert labels** = ground truth for tumor volumes
- **Multiple scanners** (3T + 1.5T) = test harmonization preserves true biology
- **Before deploying system**: Validate on Cyprus that harmonization doesn't create fake tumor changes!

## Critical Lesson for PhD Thesis
**"Harmonization is powerful but not magic - test first, harmonize second, validate third!"**

Don't assume Yale needs harmonization just because it's multi-scanner. This paper shows blindly harmonizing can HURT your results. The validation workflow:
1. Preprocess all data
2. Test for scanner effects using traveling subject approach
3. If effects exist → harmonize + re-test
4. Validate on Cyprus dataset (ground truth labels)
5. Then proceed to Phase 2 (ViT training)
