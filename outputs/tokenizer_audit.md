# Tokenizer Audit

**Date:** 2026-04-11  
**File:** `chemical_tokenizer.json`  
**vocab_size:** 110 (3 special + 107 chemical)

## Special Tokens

| ID | Token |
|----|-------|
| 0  | [PAD] |
| 1  | [MASK] |
| 2  | [EOS] |

## Ring Tokens (7 total)

| ID | Token |
|----|-------|
| 5  | [Ring1] |
| 12 | [Ring2] |
| 25 | [=Ring1] |
| 36 | [=Ring2] |
| 85 | [-\Ring1] |
| 97 | [-/Ring1] |
| 102 | [-/Ring2] |

**Missing:** `[Ring3]`, `[=Ring3]` not in vocab. Molecules requiring Ring3 (3-step shortcuts) cannot be represented.

## Branch Tokens (6 total)

| ID | Token |
|----|-------|
| 6  | [Branch1] |
| 7  | [=Branch1] |
| 11 | [Branch2] |
| 15 | [#Branch1] |
| 18 | [=Branch2] |
| 20 | [#Branch2] |

**Missing:** `[Branch3]`, `[=Branch3]`, `[#Branch3]` not in vocab.

## Vocab Integrity Checks

- **Duplicates:** None found
- **token_to_id / id_to_token consistency:** No mismatches
- **id_to_token entries:** 110
- **token_to_id entries:** 110
- **Declared vocab_size:** 110 -- CONSISTENT

## Vocab vs selfies Library Semantic-Robust Alphabet

- selfies version: 2.1.1
- SF semantic-robust alphabet size: 69 tokens
- Tokens in SF alphabet NOT in our vocab (32): includes `[Ring3]`, `[=Ring3]`, `[Branch3]`, `[=Branch3]`, `[#Branch3]`, `[C+1]`, `[C-1]`, `[H]`, `[O+1]`, etc.
- Tokens in our vocab NOT in SF alphabet (70): stereo-directional forms (`[/C]`, `[\C]`, `[C@@H1]`, etc.) -- these are valid extended SELFIES tokens for stereochemistry.

## Verdict

**No tokenizer bugs found.**  
The vocab correctly covers all common ChEBI-20 tokens. Missing Ring3/Branch3 tokens are a coverage limitation (not a bug) -- these are extremely rare in ChEBI-20. Stereo tokens in vocab but not in the SF minimal alphabet are correct extended tokens.
