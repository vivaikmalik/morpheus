# Branch Round-Trip Verification

**Date:** 2026-04-11  
**Sample:** 20 random branch-containing molecules from ChEBI-20 test set

## Results

**20 / 20 round-trips passed (lossless)**

SELFIES -> token IDs -> token strings -> SELFIES -> SMILES recovers the identical canonical SMILES in all cases.

## Branch Token Context Observations

Sample contexts from round-trip molecules:

| Branch token | Next token (count) | Q value | Branch len |
|-------------|-------------------|---------|-----------|
| [Branch2] | [Ring1] (id=5) | Q=2 | 3 symbols |
| [=Branch1] | [/C] (id=27) | Q=24 | 25 symbols |
| [Branch1] | [C] (id=3) | Q=0 | 1 symbol |
| [=Branch2] | [Ring2] (id=12) | Q=9 | 10 symbols |

Most common: [Branch1][C] -> Q=0 -> 1-symbol branch (single atom substituent).

## Mechanics

In SELFIES, [BranchN] reads N count tokens:
- [Branch1] reads 1 count token: branch length = Q1 + 1 symbols
- [Branch2] reads 2 count tokens: branch length encodes higher-order count
- [#Branch1] and [=Branch1] are typed variants with same count semantics

The count token's identity is its vocab position minus CHEM_START (= 3).
[C] (id=3) -> Q=0 -> 1-symbol branch.
[=C] (id=4) -> Q=1 -> 2-symbol branch. Etc.

## Special Token Check

**Special tokens (PAD/MASK/EOS) in branch count positions: 0**

## Verdict

No branch tokenization bugs. Round-trips are lossless.
