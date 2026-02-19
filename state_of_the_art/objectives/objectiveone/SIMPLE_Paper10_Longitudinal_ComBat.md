# Paper 10: Longitudinal ComBat (2020)

## 1. One-Sentence Summary
ComBat adapted for longitudinal multi-scanner brain imaging that accounts for within-subject repeated measures, more powerful than cross-sectional ComBat for detecting cortical thickness changes over time.

## 2. Key Results (Numbers!)
- **663 ADNI patients** scanned on 126 different scanners over 3 years
- **126 scanners**: 35 at 3.0T, 91 at 1.5T (2004-2023 timespan)
- **62 brain regions** analyzed for cortical thickness
- **All 62 regions** had significant scanner effects (p < 0.05)
- **More power**: Longitudinal ComBat detects temporal changes better than cross-sectional ComBat
- **Type I error control**: Better than unharmonized data with scanner as covariate
- **3.0T scanners** produce larger cortical thickness measurements than 1.5T
- **Additive effects**: Largest in medial occipital and medial temporal regions
- **Multiplicative effects**: Largest in superior frontal and superior parietal regions

## 3. What's New
- **FIRST longitudinal-specific ComBat**: Explicitly accounts for within-subject correlation across timepoints (not just treating each scan independently)
- **Random intercept modeling**: Captures subject-specific baseline differences across time
- **Better than cross-sectional**: More powerful for detecting longitudinal change (e.g., cortical thickness loss in Alzheimer's)
- **REML + empirical Bayes**: Uses restricted maximum likelihood for linear mixed effects, then applies empirical Bayes shrinkage
- **No overlap cohort needed**: Unlike some harmonization methods, doesn't require subjects scanned on multiple scanners

## 4. Limitations
- **Requires timepoints**: Only works with longitudinal data (2+ scans per person), not for single timepoint studies
- **Scanner minimum**: Needs at least 2 scans per scanner to estimate effects (16 scanners excluded)
- **Assumes linearity**: Time effects assumed linear (may not capture complex trajectories)
- **Cortical thickness only**: Validated on cortical thickness from ADNI, not tested on Yale tumor data yet
- **No validation on tumors**: Brain atrophy ≠ tumor growth patterns (different biology)
- **Slow processing**: More complex than cross-sectional ComBat (random effects modeling)

## 5. Methods (Simple Steps)
1. **Get longitudinal data**: Each person has 2+ scans over time on potentially different scanners
2. **Fit mixed effects model**: Model = biological effects (age, sex, diagnosis) + additive scanner effects + multiplicative scanner effects + random subject intercept
3. **Standardize features**: Make all 62 brain regions have similar mean/variance for unbiased estimation
4. **Empirical Bayes**: Estimate scanner effect prior distributions (assume scanner effects across regions come from common distribution)
5. **Shrink estimates**: Borrow information across all 62 regions to get robust scanner effect estimates
6. **Adjust data**: Remove additive (mean shift) and multiplicative (variance scaling) scanner effects
7. **Preserve biology**: Add back age, sex, diagnosis, time effects and subject-specific trajectories

## 6. Code/Resources Available
- **R package**: `longCombat` (https://github.com/jcbeer/longCombat)
- **Free and open source**: GitHub repository with full implementation
- **ANTs processing**: Uses Advanced Normalization Tools (ANTs) for cortical thickness extraction
- **ADNI data**: Public dataset from Alzheimer's Disease Neuroimaging Initiative (adni.loni.usc.edu)
- **62 regions**: Desikan-Killiany-Tourville atlas used for brain parcellation
- **R version 3.5.3**: lme4 package for mixed effects modeling, pbkrtest for significance testing

## 7. Connection to Yale Dataset
- **Perfect match**: Yale has 11,884 scans over 20 years (2004-2023) = MASSIVE scanner changes!
- **Same problem**: Multiple scanners, sites, field strengths (like ADNI's 126 scanners)
- **Longitudinal data**: Yale averages 8.3 timepoints per patient (1,430 patients) = exactly what longCombat designed for
- **Two-step strategy**: 
  1. Use Paper 9 ComBat for baseline scanner harmonization (cross-sectional)
  2. Use Paper 10 longCombat for temporal tracking across scanners (longitudinal)
- **Tumor vs atrophy**: Method developed for brain atrophy, needs validation on tumor growth (opposite direction!)
- **Cyprus validation**: Could test if longCombat preserves tumor growth patterns on Cyprus 40-patient dataset with expert labels

## 8. Connection to 5 Research Objectives
1. **Automatic tumor tracking** ✅✅✅: Harmonizes scanners so temporal tumor changes = biology, not scanner differences
2. **Predict future changes** ✅✅: Removes scanner noise from longitudinal data → cleaner temporal patterns for prediction
3. **LLM reports** ✅: Harmonized data = trustworthy reports ("tumor grew 2mm" not confounded by scanner change)
4. **Video generation** ✅✅: Videos show TRUE tumor evolution, not scanner artifacts (smoother videos)
5. **Clinical decision support** ✅✅✅: Doctors trust system when scanner changes don't create false "tumor growth" alarms

## 9. How It Combines With Other Papers
- **After Paper 9 (ComBat)**: Paper 9 harmonizes baseline scans, Paper 10 harmonizes temporal trajectories
- **After Paper 4 (Registration)**: Registration aligns anatomy across timepoints, longCombat removes scanner effects across timepoints
- **After Paper 5 (FLIRE)**: Fast registration first (80 days), then longCombat harmonization (adds ~1 week)
- **Before Phase 2 (ViT)**: Clean longitudinal data → Vision Transformer learns tumor biology, not scanner noise
- **With Paper 2 (nnU-Net)**: nnU-Net segments tumors, longCombat ensures segmentation volumes comparable across scanners
- **Processing order**: Download → BraTS Toolkit → FLIRE → nnU-Net → Extract features → ComBat (baseline) → longCombat (trajectories)

## 10. Citation
Beer, J.C., Tustison, N.J., Cook, P.A., Davatzikos, C., Sheline, Y.I., Shinohara, R.T., Linn, K.A. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage*, 220, 117129. https://doi.org/10.1016/j.neuroimage.2020.117129

---

## Key Differences: Cross-Sectional vs Longitudinal ComBat

| Feature | Cross-Sectional ComBat (Paper 9) | Longitudinal ComBat (Paper 10) |
|---------|----------------------------------|--------------------------------|
| **Data type** | Each scan treated independently | Accounts for repeated measures per person |
| **Model** | Fixed effects only | Fixed effects + random intercept per subject |
| **Power** | Lower for detecting temporal change | Higher for detecting temporal change |
| **Use case** | Baseline harmonization, different people | Tracking change over time, same people |
| **Yale application** | Harmonize all 11,884 scans to "standard scanner" | Track tumor growth trajectories across scanner changes |
| **Speed** | Fast (no random effects) | Slower (mixed effects modeling) |

## Critical for Yale Project
- **20 years of scanners** (2004-2023) = scanner changes DURING patient follow-up
- **Patient switches scanners**: Person scanned 8 times might see 3 different scanners
- **False tumor growth**: Without longCombat, scanner upgrade might look like tumor explosion!
- **Treatment decisions**: Doctor sees "tumor doubled" → actually just scanner change → wrong treatment!
- **LongCombat fixes this**: Removes scanner effects from temporal trajectories → real tumor biology visible
