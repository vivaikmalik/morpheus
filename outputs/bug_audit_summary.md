# Bug Audit Summary: SELFIES Ring/Branch Token Investigation

**Date:** 2026-04-11  
**Hypothesis tested:** Implementation bugs (tokenization, truncation, indexing) could explain  
the ring topology failure mode, rather than architectural limitations.

## Check Results

| Check | Finding | Verdict |
|-------|---------|---------|
| 1. Tokenizer integrity | 110 tokens, no duplicates, no mapping mismatches | PASS -- no bugs |
| 2. Length truncation | All 25,204 training/val/test molecules <= 73 tokens; 0 truncated | PASS -- no bugs |
| 3. Index table round-trip | 20/20 ring molecules: lossless SELFIES -> IDs -> SELFIES -> SMILES | PASS -- no bugs |
| 4. Generated count tokens | 0 special tokens in count positions; ring count accuracy degrades with complexity | PASS (no token bugs) |
| 5. Version consistency | selfies 2.1.1, vocab_size=110 matches checkpoint, EOS fix applied | PASS -- no bugs |

## Issues Found

### Minor Finding: OOV Tokens in Generated Molecules (not a model bug)

127 instances where `[CH0]`, `[CH1]`, `[SH1]`, `[O+1]` appear after ring/branch tokens  
in re-encoded generated SMILES. These tokens are valid SELFIES but absent from training vocab.  
The model cannot generate these tokens -- they appear only in post-hoc re-encoding.  
**Not a training or inference bug.**

### Minor Finding: [Ring3]/[Branch3] Not in Vocab

Molecules requiring Ring3 or Branch3 tokens cannot be represented.  
These are extremely rare in ChEBI-20 (not found in any training/val/test molecule).  
**Not causing current failures.**

## Root Cause of Ring Prediction Degradation

The ring count accuracy analysis (Check 4) confirms the failure is **architectural**:

- 96% accuracy for 0-ring molecules
- 57% accuracy for 1-ring molecules
- 0% accuracy for 6+ ring molecules
- Mean ring delta: -0.171 (systematic under-prediction)

This pattern is consistent with the **parallel decoding hypothesis**:  
MaskGIT fills multiple token positions simultaneously without explicit coordination.  
Predicting correlated ring closure tokens (e.g., five `[Ring1][Count]` pairs that must  
form consistent closure with each other) is hard when each position is sampled independently.

No implementation bug is responsible. The limitation is capacity + parallel decoding architecture.

## Recommendation

**Proceed with full token-level diagnostic analysis.** No bugs need to be fixed first.

The implementation is correct. The ring prediction degradation is a genuine architectural  
limitation worth characterizing more precisely:
1. Token-level analysis: which specific ring/branch patterns are hardest to predict
2. Compare per-step confidence of ring tokens vs non-ring tokens during generation
3. Evaluate whether iterative refinement (already implemented) partially mitigates this

The contrastive reranking already helps (rerank@10 improves Morgan by +1.1pp) -- this works  
because it selects candidates with better overall structure, which correlates with ring accuracy.
