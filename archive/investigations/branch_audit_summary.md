# Branch Count Prediction Audit Summary

**Date:** 2026-04-11  
**Source data:** `outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv`  
**2,542 test molecules analysed**

---

## Headline

**Branch count prediction is a more severe failure mode than ring count prediction.**

- Ring count match (exact): **39.7%** of all molecules
- Branch count match (exact): **23.0%** of all molecules
- Both correct simultaneously: **13.7%** of all molecules

Branches fail at lower complexity thresholds, and they fail in the **opposite direction** from rings.

---

## Key Asymmetry: Opposite Prediction Bias

| Token type | Mean delta (gen - gt) | Direction |
|-----------|----------------------|-----------|
| Ring tokens  | -0.060 | Slight under-prediction |
| Branch tokens | +0.989 | Strong over-prediction |

The model generates nearly **1 extra branch per molecule** on average.  
Ring count is close to correct on average but fails at extreme complexity.  
Branch count is systematically inflated across all complexity levels.

---

## Branch Count Accuracy by Complexity

| GT branches | n | Match % | Mean gen branches | Mean Morgan |
|------------|---|---------|------------------|------------|
| 0 | 53 | 35.8% | 1.26 | 0.276 |
| 1 | 135 | 45.2% | 2.19 | 0.428 |
| 2 | 237 | 30.8% | 3.49 | 0.378 |
| 3 | 257 | 23.7% | 4.90 | 0.325 |
| 4 | 313 | 21.1% | 5.70 | 0.298 |
| 5+ | 1547 | 19.7% | > gt | 0.273 |
| 16 | 10 | 0.0% | 13.30 | 0.290 |
| 17+ | 9 | 0.0% | < gt | varies |

Accuracy starts low (35.8% at 0 branches) and degrades steadily.  
Even at 0 branches, 64.2% of molecules receive spurious branch tokens.

---

## Branch Count Accuracy by Bin

| Bin | n | Match % | Mean Morgan |
|-----|---|---------|------------|
| 0 branches | 53 | 35.8% | 0.276 |
| 1 branch | 135 | 45.2% | 0.428 |
| 2 branches | 237 | 30.8% | 0.378 |
| 3 branches | 257 | 23.7% | 0.325 |
| 4 branches | 313 | 21.1% | 0.298 |
| 5+ branches | 1547 | 19.7% | 0.273 |

---

## Ring Count Accuracy (For Comparison)

| Bin | n | Match % | Mean Morgan |
|-----|---|---------|------------|
| 0 rings | 609 | 77.7% | 0.417 |
| 1 ring | 484 | 53.9% | 0.361 |
| 2 rings | 336 | 26.5% | 0.267 |
| 3 rings | 200 | 24.5% | 0.242 |
| 4 rings | 172 | 19.8% | 0.202 |
| 5+ rings | 741 | varies | < 0.21 |

---

## Combined Complexity Stratification (Check 4)

| Count positions | n | Branch match % | Ring match % | Morgan |
|----------------|---|---------------|-------------|--------|
| 0 | 48 | 37.5% | 77.1% | 0.293 |
| 1 | 98 | 40.8% | 84.7% | 0.460 |
| 2 | 167 | 33.5% | 78.4% | 0.401 |
| 3-4 | 392 | 27.3% | 65.1% | 0.364 |
| 5-7 | 544 | 23.0% | 43.2% | 0.326 |
| 8+ | 1293 | 18.4% | 20.8% | 0.244 |

Ring accuracy is much better at low complexity (77-85% at 0-1 count positions).  
Branch accuracy starts low and compounds with ring complexity.  
Morgan degrades monotonically with total count-position complexity.

---

## Branch Length Accuracy (Check 5)

| Branch length | n | Match % |
|--------------|---|---------|
| 1 symbol | 10,085 | 68.1% |
| 2-3 symbols | 1,529 | 20.6% |
| 4-7 symbols | 824 | 8.1% |
| 8+ symbols | 2,482 | 10.2% |

Single-atom branches ([Branch1][C]) are predicted correctly 68% of the time.  
Longer branches drop to 8-20% accuracy.  
This confirms: the count token Q encodes branch length, and predicting larger Q values is harder.

---

## Position Effect (Check 6)

| Position | n | Match % |
|----------|---|---------|
| 0-25% (start) | 3,649 | 50.2% |
| 25-50% | 4,259 | 51.5% |
| 50-75% | 4,231 | 53.7% |
| 75-100% (end) | 2,781 | 43.4% |

Position effect is minimal and non-monotonic. Slight accuracy drop at the end of sequences  
(75-100%), but no strong gradient. The parallel decoding failure is approximately uniform  
across the sequence -- it is not a tail-end masking artifact.

---

## Interpretation

### Why branches fail differently from rings

**Rings:** Under-predict ring count (delta = -0.06). The model fails to generate enough ring  
closure tokens for complex multi-ring systems. This is the expected parallel decoding failure.

**Branches:** Over-predict branch count (delta = +0.99). The model generates approximately  
1 extra branch per molecule. This is a different failure: the model has learned that ChEBI-20  
molecules tend to have many branches (median ~5-6), and produces that distribution rather than  
correctly conditioning on the specific prompt's complexity.

These are two distinct failure modes:
1. **Ring under-prediction:** parallel decoding can't coordinate ring closure tokens
2. **Branch over-generation:** distribution mismatch / weak conditioning signal for branch count

---

## Recommendation

Treat branches and rings as **separate failure modes** in the report.

They share the same underlying mechanism (parallel prediction of correlated count tokens)  
but manifest differently:
- Rings: coordination failure -> under-prediction in complex molecules
- Branches: conditioning failure -> systematic over-generation (+0.99 mean excess)

The branch over-generation is potentially more actionable:
- It may respond to stronger text conditioning (text prompts mention atom count, not branch count)
- Best-of-N reranking (already implemented) partially compensates by selecting candidates  
  with better structural alignment, which correlates with correct branch count

The ring failure is more fundamental and would require architectural changes  
(explicit ring-count prediction head, or sequential decoding for ring closure tokens).
