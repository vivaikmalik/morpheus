# Dep-aware Implementation Verification

Checks performed using `_compute_not_ready_mask` and `RING_OPENER_IDS` imported
directly from `finetune/train_chebi20.py` via `importlib.util`. No source files
were modified.

---

## Check 1: Does the rule fire on real data?

**Simulation setup:** 8 molecules selected from `data/chebi20_test.csv` with ≥3 ring
opener tokens in their ground-truth SELFIES. The `response` column contains SELFIES
directly (not SMILES). Ground-truth tokens were committed into an all-MASK sequence
progressively over 50 steps (~2% of remaining masked positions committed per step,
random seed 42). `_compute_not_ready_mask` was called at each step before committing.

**Note on the first investigation attempt:** Using `test_df.head(8)` (the first 8 rows
by CSV order) produced 0 ring openers because `sf.encoder()` was mistakenly called on
the SELFIES-already response column, which raised `failed to parse input` silently and
returned 0 openers for all 8 rows. The re-run below tokenizes directly via
`ChemicalTokenizer.encode(selfies_str)`.

**Results (8 molecules with 3–8 ring opener tokens):**

| Mol | GT ring openers | not_ready_total | steps fired / 50 | fire fraction |
|-----|----------------|-----------------|-------------------|---------------|
| 0 | 6 | 84 | 47 | 0.940 |
| 1 | 8 | 165 | 42 | 0.840 |
| 2 | 5 | 113 | 43 | 0.860 |
| 3 | 7 | 146 | 42 | 0.840 |
| 4 | 7 | 92 | 42 | 0.840 |
| 5 | 6 | 85 | 36 | 0.720 |
| 6 | 6 | 68 | 28 | 0.560 |
| 7 | 6 | 54 | 26 | 0.520 |

- **Total not_ready=True across all 8 molecules and 50 steps: 807**
- **Maximum not_ready count in any single step: 30**
- **Mean fraction of steps with at least one firing: 0.765**

The rule fires extensively on high-ring-count molecules during realistic intermediate
states. It is not a dead code path.

**Context from full test set:** 609/2542 molecules (24.0%) have 0 ground-truth ring
opener tokens — the rule provably cannot fire on those. 1113/2542 (43.8%) have 3+
ring openers. The remaining 820 (32.2%) have 1–2 ring openers.

---

## Check 2: Unit Tests on `_compute_not_ready_mask`

Tokens used: MASK=1, OPENER=[Ring1]=5, ATOM=8.
`opener_ids_tensor = torch.tensor([5])` (single opener for test clarity).

**Mathematical derivation of expected outputs:**

For position i, `not_ready[i]` is True iff:
1. `prev_input_ids[i-1]` is a committed ring opener (not MASK, not sentinel -1, value in opener set), AND
2. `cumulative_mask[i-2] > 0` — i.e., any of positions 0..i-2 contains MASK.

`prefix_has_mask_strict[i]` = `cumulative_mask[i-2]` (implemented via `cumulative_mask[:, :-2]` shifted to `[:, 2:]`; positions 0 and 1 are hardcoded False).

| Case | Input `prev_input_ids` | Expected `not_ready` | Actual | Result |
|------|------------------------|---------------------|--------|--------|
| A: All MASK | `[1,1,1,1,1,1,1,1]` | `[F,F,F,F,F,F,F,F]` | `[F,F,F,F,F,F,F,F]` | **PASS** |
| B: Opener pos 4, prefix committed | `[8,8,8,8,5,1,1,1]` | `[F,F,F,F,F,F,F,F]` | `[F,F,F,F,F,F,F,F]` | **PASS** |
| C: Opener pos 4, pos 2 masked | `[8,8,1,8,5,1,1,1]` | `[F,F,F,F,F,T,F,F]` | `[F,F,F,F,F,T,F,F]` | **PASS** |
| D: Opener at pos 0 (empty prefix) | `[5,1,1,1,1,1,1,1]` | `[F,F,F,F,F,F,F,F]` | `[F,F,F,F,F,F,F,F]` | **PASS** |
| E: Two openers, mixed readiness | `[8,5,1,8,8,5,1,1]` | `[F,F,F,F,F,F,T,F]` | `[F,F,F,F,F,F,T,F]` | **PASS** |
| F: No openers committed | `[8,8,8,8,8,1,1,1]` | `[F,F,F,F,F,F,F,F]` | `[F,F,F,F,F,F,F,F]` | **PASS** |

**6/6 tests passed.** The implementation is correct for all hand-constructed cases.

**Case derivations (for audit):**

- **Case B (False at pos 5):** opener at pos 4 → count token at pos 5.
  `cumulative_mask[3]` = False (positions 0–3 all committed → no MASK) → `prefix_has_mask_strict[5]` = False.
  Result: `not_ready[5]` = True AND False = **False** ✓

- **Case C (True at pos 5):** opener at pos 4 → count token at pos 5.
  `cumulative_mask[3]` = True (position 2 is MASK → cumsum > 0 at pos 3) → `prefix_has_mask_strict[5]` = True.
  Result: `not_ready[5]` = True AND True = **True** ✓

- **Case D (False at pos 1):** opener at pos 0 → count token at pos 1.
  `prefix_has_mask_strict[1]` = 0 (hardcoded zero for indices < 2).
  Result: `not_ready[1]` = True AND False = **False** ✓
  The implementation correctly handles the empty-prefix edge case.

- **Case E (True at pos 6, False at pos 2):**
  - Pos 2: opener at pos 1 → `cumulative_mask[0]` = False (pos 0 is committed) → `prefix_has_mask_strict[2]` = False → `not_ready[2]` = **False** ✓
  - Pos 6: opener at pos 5 → `cumulative_mask[4]` = True (pos 2 is MASK) → `prefix_has_mask_strict[6]` = True → `not_ready[6]` = **True** ✓

---

## Check 3: Does -inf Propagate Correctly Through `topk` on MPS?

**Setup:** Confidence tensor `[0.5, 0.3, 0.1, 0.7, 0.2, 0.4, 0.6, 0.8]` on target device.
Positions 2 and 4 set to -inf. `torch.topk(conf, k=4, largest=False)` called.
Expected: positions 2 and 4 always in bottom-4 (smallest values).
Next smallest after -inf: position 1 (0.3) and position 5 (0.4).

| Device | Indices returned | Pos 2 present | Pos 4 present | Result |
|--------|-----------------|---------------|---------------|--------|
| CPU | [4, 2, 1, 5] | ✓ | ✓ | **PASS** |
| MPS | [2, 4, 1, 5] | ✓ | ✓ | **PASS** |

Both CPU and MPS return the same 4 indices {1, 2, 4, 5}, in different order (ordering
among equal-valued or -inf ties differs between devices, which is expected and harmless
since only the set of indices matters for the re-masking scatter operation).

**The -inf → always-re-masked path is correct on this machine's MPS device.**

---

## Verdict

IMPLEMENTATION CORRECT: All checks passed. The hypothesis is the weak link, not the
implementation.

Specifically:
- The `_compute_not_ready_mask` logic is mathematically correct (6/6 unit tests pass).
- The rule fires 807 times across 8 high-ring molecules over 50 simulated steps
  (mean 76.5% of steps have at least one firing), so it is not a dead code path.
- `-inf` confidence correctly forces positions into the re-mask set on both CPU and MPS.

The failure to achieve a meaningful full-set improvement (+0.0009 Morgan) is therefore
attributable to one or more of the hypotheses being wrong, not to a bug:

1. **24% of the test set has 0 ring opener tokens** — the rule cannot fire there.
2. **The investigation found the rule also changes outputs on 41.5% of 0–1 ring molecules**
   (where the not_ready mask is all-False). This points to an incidental PRNG state
   perturbation: calling `torch.isin` + `masked_fill` even with an all-False mask
   introduces additional GPU ops that shift the MPS random number stream for subsequent
   `torch.distributions.Categorical.sample()` calls, producing different outputs regardless
   of whether the rule was semantically active. The "dep_aware" improvement signal is
   therefore confounded with random perturbation noise.
3. **In the 3+ ring stratum, outputs differ on 92.1% of molecules** (vs 41.5% for low-ring),
   but the mean Morgan delta is only +0.0046, and the exact ring-opener-count match rate
   actually decreased (0.0993 → 0.0794). Deferring count tokens does not reliably improve
   ring topology — the model may be generating entirely different ring structures rather than
   correcting count values.
