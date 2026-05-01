# Dep-aware Investigation: Full Eval Comparison

---

## Summary

Standard Morgan 0.2435, dep_aware Morgan 0.2444, delta +0.0009 (+0.37%) on 2542 molecules.
Of 2542 rows, 876 have identical gen_smiles (dep_aware changed nothing); 1666 differ.
Of those 1666 changed rows, 810 improved Morgan and 776 worsened it — a near-50/50 split.
Ring count token exact-match accuracy under standard is 0.3505 (overall); under dep_aware it
is 0.3434 — dep_aware does not improve ring count token accuracy overall, and in the 3+ ring
stratum it is lower (dep 0.0794 vs std 0.0993).
However, dep_aware fires on 92.1% of 3+ ring molecules but also on 41.5% of 0–1 ring
molecules, where the not-ready mask should evaluate to all-False and leave outputs unchanged.

---

## Section 1: Difference Counting

| Metric | Count |
|--------|-------|
| Total rows | 2542 |
| Rows with identical gen_smiles (std == dep) | 876 (34.5%) |
| Rows with different gen_smiles | 1666 (65.5%) |

Of the 1666 rows where gen_smiles differs:

| Outcome | Count |
|---------|-------|
| morgan_sim[dep] > morgan_sim[std] | 810 (48.6%) |
| morgan_sim[dep] < morgan_sim[std] | 776 (46.6%) |
| morgan_sim[dep] == morgan_sim[std] (different SMILES, same Morgan) | 80 (4.8%) |

---

## Section 2: Aggregate Deltas

| Metric | Standard | Dep-aware | Delta | Delta % | Rows dep>std | Rows dep<std |
|--------|----------|-----------|-------|---------|--------------|--------------|
| morgan_sim | 0.243493 | 0.244395 | +0.000902 | +0.37% | 810 | 776 |
| maccs_sim | 0.543935 | 0.544567 | +0.000632 | +0.12% | 777 | 791 |
| rdk_sim | 0.324677 | 0.325240 | +0.000563 | +0.17% | 813 | 787 |
| atom_bleu2 | 0.486010 | 0.488068 | +0.002058 | +0.42% | 827 | 778 |
| atom_bleu4 | 0.360050 | 0.362702 | +0.002651 | +0.74% | 779 | 787 |
| bleu2 | 0.499459 | 0.501823 | +0.002364 | +0.47% | 815 | 796 |
| bleu4 | 0.390020 | 0.391756 | +0.001736 | +0.45% | 804 | 780 |
| lev_sim | 0.441586 | 0.443796 | +0.002210 | +0.50% | 799 | 781 |

All metrics improved slightly. The BLEU/lev improvements are relatively larger than the
Morgan improvement; maccs_sim is the smallest relative gain.

---

## Section 3: Stratified Morgan by Ring Count

ring_count from stratified_analysis_best_model.csv; all 2542 prompts matched (0 missing).
No fresh RDKit computation was needed.

| ring_count | n | morgan_std | morgan_dep | delta_morgan | maccs_std | maccs_dep |
|-----------|---|-----------|-----------|-------------|---------|---------|
| 0 | 762 | 0.3796 | 0.3775 | −0.0022 | 0.6044 | 0.6025 |
| 1 | 521 | 0.2348 | 0.2368 | +0.0020 | 0.5232 | 0.5319 |
| 2 | 403 | 0.1769 | 0.1744 | −0.0025 | 0.4888 | 0.4774 |
| 3 | 421 | 0.1653 | 0.1730 | +0.0077 | 0.5124 | 0.5250 |
| 4 | 263 | 0.1644 | 0.1638 | −0.0007 | 0.5476 | 0.5361 |
| 5 | 137 | 0.1396 | 0.1464 | +0.0068 | 0.5446 | 0.5497 |
| 6+ | 35 | 0.1171 | 0.1145 | −0.0026 | 0.5205 | 0.5225 |

3-ring molecules show the largest Morgan gain (+0.0077), followed by 5-ring (+0.0068).
4-ring and 6+ ring molecules show small Morgan regression. The improvement is not monotonic
with ring count.

---

## Section 4: Stratified Morgan by Branch Count

Branch count = number of "(" characters in gt_smiles.

| branch_count | n | morgan_std | morgan_dep | delta_morgan |
|-------------|---|-----------|-----------|-------------|
| 0 | 91 | 0.2192 | 0.2154 | −0.0038 |
| 1–2 | 580 | 0.2900 | 0.2859 | −0.0041 |
| 3–4 | 744 | 0.2366 | 0.2427 | +0.0061 |
| 5–6 | 571 | 0.2440 | 0.2506 | +0.0066 |
| 7+ | 556 | 0.2076 | 0.2017 | −0.0059 |

Mid-range branch counts (3–4, 5–6) improve; low and high branch counts regress.
The 7+ branch stratum (n=556) shows the largest regression (−0.0059).

---

## Section 5: Ring Count Token Accuracy

**Position-alignment approach:**  
Only 4.8% of molecules (std: 122/2542, dep: 119/2542) have identical SELFIES length between
generated and ground-truth. Position-aligned comparison was restricted to those molecules only.

| | Standard | Dep-aware |
|---|---------|---------|
| Molecules with same SELFIES length as gt | 122 (4.8%) | 119 (4.7%) |
| Total ring opener positions (same-len only) | 210 | 193 |
| Ring count token matches | 60 | 61 |
| **Position-aligned match rate** | **0.2857** (60/210) | **0.3161** (61/193) |

The position-aligned sample is too small (n≈120 molecules, 210 opener positions) to be
reliable. Fallback metrics below use all 2542 molecules.

**Fallback: ring opener count per molecule**

| | Standard | Dep-aware |
|---|---------|---------|
| Total gt ring opener tokens | 7548 | 7548 |
| Total gen ring opener tokens | 5242 | 5391 |
| Mean delta gen−gt (opener count) | −0.9072 | −0.8485 |
| Fraction of molecules with exact opener count match | 0.3505 | 0.3434 |

Dep_aware generates slightly MORE ring openers per molecule (+149 total, mean delta −0.85 vs
−0.91), but the exact opener count match rate is slightly LOWER (0.3434 vs 0.3505).

**Stratified by ring count bin:**

| ring_bin | n | std match_rate (pos) | dep match_rate (pos) | std exact_count | dep exact_count | morgan_delta |
|---------|---|---------------------|---------------------|----------------|----------------|-------------|
| 0 | 762 | 0.9000 (9/10) | 0.6667 (6/9) | 0.7415 | 0.7375 | −0.0022 |
| 1–2 | 924 | 0.3269 (34/104) | 0.4130 (38/92) | 0.2608 | 0.2630 | +0.0000 |
| 3+ | 856 | 0.1771 (17/96) | 0.1848 (17/92) | 0.0993 | 0.0794 | +0.0046 |

For the 3+ ring stratum: Morgan improved +0.0046 but exact ring opener count match rate
DECREASED (0.0993 → 0.0794). Dep_aware generates more ring openers in the 3+ ring stratum
(mean gt=6.22, std=4.00, dep=4.17), which is closer in absolute terms to gt but not
sufficient to raise exact-count accuracy.

Detailed ring opener count by ring_count value:

| ring_count | n | gt mean openers | std mean openers | std delta | dep mean openers | dep delta |
|-----------|---|----------------|-----------------|-----------|-----------------|-----------|
| 0 | 762 | 0.23 | 0.29 | +0.06 | 0.28 | +0.05 |
| 1 | 521 | 1.47 | 1.24 | −0.23 | 1.26 | −0.21 |
| 2 | 403 | 3.18 | 2.35 | −0.83 | 2.34 | −0.83 |
| 3 | 421 | 5.26 | 3.19 | −2.07 | 3.61 | −1.65 |
| 4 | 263 | 6.37 | 4.05 | −2.32 | 4.14 | −2.23 |
| 5 | 137 | 7.99 | 5.77 | −2.22 | 5.30 | −2.69 |

For rc=3 molecules, dep generates more openers (3.61 vs 3.19, delta improvement of +0.42).
For rc=5 molecules, dep generates FEWER openers (5.30 vs 5.77, further from gt 7.99).
The relationship is not monotonic.

---

## Section 6: Branch Count Token Analysis

Branch openers: [Branch1], [Branch2], [=Branch1], [=Branch2], [#Branch1], [#Branch2].

**Position-aligned branch count token match (same-length molecules only):**

| | Standard | Dep-aware |
|---|---------|---------|
| Total branch opener positions (same-len only) | 537 | 507 |
| Branch count token matches | 266 | 287 |
| **Position-aligned match rate** | **0.4953** | **0.5661** |
| Mean delta gen−gt branch opener count | −0.8399 | −0.7923 |
| Fraction exact branch opener count match | 0.1699 | 0.1727 |

Dep_aware improved branch count token match rate from 0.4953 to 0.5661 (+0.0708) in
position-aligned molecules.  
Mean delta for branch opener count improved from −0.8399 to −0.7923.  
The exact branch opener count match fraction changed only marginally (0.1699 → 0.1727).

**Dep_aware is changing branch generation even though the not-ready rule was designed only
for ring openers.** The position-aligned branch match rate shows a clear improvement (+7.1
percentage points), which is larger than the corresponding ring improvement.

---

## Section 7: Output Sequence Length Differences

| | Mean length | Std dev |
|---|------------|---------|
| Ground truth SELFIES | 36.25 | 17.03 |
| Standard gen SELFIES | 33.81 | 19.00 |
| Dep-aware gen SELFIES | 34.28 | 18.85 |

Dep_aware generates sequences that are on average 0.47 tokens longer than standard.

Distribution of length difference between std and dep:

| |  Fraction |
|---|---------|
| std length == dep length | 0.4226 (42.3%) |
| differ by 1 token | 0.0543 (5.4%) |
| differ by 2–5 tokens | 0.1706 (17.1%) |
| differ by 5+ tokens | 0.3525 (35.2%) |

57.7% of generated molecules have different SELFIES lengths between the two runs.
35.2% differ by more than 5 tokens.

---

## Section 8: Low-Ring vs High-Ring Firing Pattern

Partition: low = 0–1 rings (n=1283), mid = 2 rings (n=403), high = 3+ rings (n=856).

| Partition | n | Fraction gen_smiles differs | Mean Morgan delta (dep−std) |
|-----------|---|----------------------------|----------------------------|
| low (0–1 rings) | 1283 | 0.4154 (41.5%) | −0.0005 |
| mid (2 rings) | 403 | 0.8561 (85.6%) | −0.0025 |
| high (3+ rings) | 856 | 0.9206 (92.1%) | +0.0046 |

The high-ring partition has a much higher fraction of changed outputs (92.1%) than low-ring
(41.5%), consistent with dep_aware firing more on high-ring molecules.

However, 41.5% of low-ring molecules also have different gen_smiles between the two runs,
even though the not-ready mask should be all-False for molecules with 0 or 1 ring. This means
dep_aware is producing different outputs on molecules where its logic should be a no-op.

Mean Morgan delta is negative for low-ring (−0.0005) and mid (−0.0025), and positive only for
high-ring (+0.0046). The aggregate +0.0009 is the net of a positive high-ring effect and a
negative low/mid-ring effect.

---

## Anomalies

1. **41.5% of 0–1 ring molecules have different gen_smiles between std and dep.**  
   The not-ready mask logic in `_compute_not_ready_mask` should return all-False for molecules
   with no committed ring openers. If outputs differ, either (a) the mask is not all-False on
   these molecules despite no ring openers, or (b) calling `torch.isin` and `masked_fill` even
   with an all-False mask is changing subsequent PRNG state on MPS, causing divergent sampling
   from that step onward.

2. **Dep_aware changed branch token generation (Section 6).**  
   Branch count token match rate improved from 0.4953 to 0.5661 in position-aligned molecules.
   Since the not-ready rule only covers ring opener tokens, branch changes are unexpected. This
   is consistent with anomaly 1 — PRNG state perturbation could affect all subsequent token
   sampling regardless of whether the not-ready mask was non-trivial.

3. **NaN gen_smiles: std=51, dep=59.**  
   Dep_aware produced 8 more molecules where gen_smiles is NaN. Of these, 36 prompts overlap
   (both runs failed), 15 are std-only failures, 23 are dep-only failures. All NaN gen_smiles
   rows are tagged valid=True with morgan_sim=0.0 in both CSVs (inconsistency in the eval
   pipeline — NaN SMILES cannot be valid).

4. **Valid column is 1.0 for all 2542 rows in both CSVs.**  
   `valid` is True for all rows including those with NaN gen_smiles. The validity check does
   not account for generation failures that produce NaN instead of a SMILES string.

5. **Position-alignment rate is 4.8%.**  
   Only 122/2542 (std) and 119/2542 (dep) generated molecules have the same SELFIES token
   count as their ground truth. The position-aligned ring count token accuracy (Section 5)
   is therefore based on ~120 molecules and 200 opener positions, making those numbers
   statistically weak.

6. **Ring opener count accuracy is NON-MONOTONIC with ring_count.**  
   For rc=5 molecules, dep generates FEWER ring openers than std (5.30 vs 5.77), moving
   further from gt (7.99). The dep_aware intervention improves opener count for rc=3 but not
   for rc=4 or rc=5.

7. **6+ ring molecules regress under dep_aware (Morgan −0.0026, n=35).**  
   This is a small stratum but the direction is against the hypothesis.

---

## End of investigation
