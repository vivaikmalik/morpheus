# Branch Token Inventory

**Date:** 2026-04-11

## Branch Tokens in Vocabulary

| ID | Token | Count in Train | Count in Test |
|----|-------|---------------|--------------|
| 6  | [Branch1]   | 66,915 | 8,217 |
| 7  | [=Branch1]  | 30,783 | 3,971 |
| 11 | [Branch2]   | 10,765 | 1,323 |
| 15 | [#Branch1]  |  6,114 |   716 |
| 18 | [=Branch2]  |  4,498 |   554 |
| 20 | [#Branch2]  |  4,644 |   607 |

**Branch tokens missing from vocab:** None. All 6 branch token types seen in data are in vocab.

**Branch tokens in vocab never seen in data:** None.

Compare to ring tokens:
- 7 ring token types in vocab
- `[-/Ring2]` (7 occurrences in train) and `[-\Ring1]` (1 occurrence in train) are extremely rare
- No `[Ring3]` or `[Branch3]` in vocab (absent from ChEBI-20 data)

## Branch Token Frequency Distribution

[Branch1] dominates (66,915 train occurrences = 55% of all branch tokens).  
[=Branch1] is second (30,783 = 25%).  
Together they account for 80% of branch tokens.

## Verdict

Full branch vocabulary coverage. No dead tokens.
