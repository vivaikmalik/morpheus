# Truncation Analysis

**Date:** 2026-04-11  
**max_selfies_len:** 73 (from train_chebi20.py L105)  
**Sequence stored in CSV:** `response` column (already SELFIES, not SMILES)

## Key Finding: Data is Pre-converted and Pre-filtered

The ChEBI-20 CSVs store SELFIES directly in the `response` column (not SMILES).  
All molecules were converted to SELFIES and filtered to <= 73 tokens **before training**.

## Length Distribution (SELFIES token count)

| Split | n | min | mean | median | p75 | p90 | p95 | max |
|-------|---|-----|------|--------|-----|-----|-----|-----|
| train | 20158 | 1 | 36.5 | 35 | 49 | 62 | 67 | 73 |
| val   | 2504  | 3 | 36.1 | 34 | 49 | 61 | 67 | 73 |
| test  | 2542  | 1 | 36.3 | 34 | 49 | 62 | 67 | 73 |

**Max token length across all splits: 73 -- exactly at the limit.**

## Truncation at Training Time

- **Molecules exceeding 73 tokens in any split: 0 (0.0%)**
- No ring or branch tokens appear in dropped tails (no tails exist)
- **No OOV tokens found in any split**

## Interpretation

The training data was pre-filtered: molecules that would exceed 73 tokens were excluded  
from the dataset (not truncated mid-sequence). This explains the hard cutoff at max=73.

**This is NOT a bug.** However, it means:
- ~17.6% of the original HF ChEBI-20 molecules were excluded from training/eval (too long)
- The excluded molecules tend to be larger (mean 60 heavy atoms vs 23 for included)
- The test set is biased toward simpler molecules

## Verdict

**No truncation bugs found.** Truncation is not causing ring prediction failures.
