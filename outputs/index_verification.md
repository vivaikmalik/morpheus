# Index Table Verification (Ring Token Round-Trip)

**Date:** 2026-04-11  
**Sample:** 20 randomly selected ring-containing molecules from test set  
**selfies version:** 2.1.1

## Round-Trip Results

**20 / 20 molecules passed lossless round-trip:**  
SELFIES tokens -> token IDs -> token strings -> SELFIES -> SMILES -> canonical SMILES  
Recovered SMILES matches ground-truth SMILES in all 20 cases.

## Ring Token Context Analysis

67 ring token occurrences found across 20 molecules.

- **Special tokens (PAD/MASK/EOS) in count positions: 0**
- All count positions contain valid chemical tokens

### Top next-tokens after ring tokens

| Token | Count | Valid? |
|-------|-------|--------|
| [C] | 12 | Yes |
| [Ring1] | 12 | Yes |
| [#Branch1] | 11 | Yes |
| [=Branch1] | 5 | Yes |
| [Branch1] | 5 | Yes |
| [N] | 4 | Yes |
| [Ring2] | 3 | Yes |
| [O] | 3 | Yes |

## Interpretation of SELFIES Ring Semantics

In SELFIES, `[RingN]` does NOT use a separate integer count token.  
The "count" Q is encoded by the NEXT token's position in the vocabulary  
(Q = token_id - chemical_start_id, where chemical_start_id = 3).  
The ring closure target is the atom Q+1 positions back in the derivation sequence.

This means: whatever chemical token follows `[Ring1]` determines the ring size.  
`[C]` (id=3) -> Q=0 -> closes to 1 position back (3-membered ring in derivation).  
`[=C]` (id=4) -> Q=1 -> closes to 2 positions back, etc.

**This is handled correctly by the selfies library.** No custom index logic is needed  
and no bug exists here -- it's transparent to the model which just predicts tokens.

## Verdict

**No index table bugs found.** Round-trips are lossless.  
Ring token count positions always contain valid chemical tokens, never special tokens.
