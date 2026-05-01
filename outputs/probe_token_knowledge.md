# Probe: What the Model Knows About Structural Tokens

## Methodology

For each of 2542 ChEBI-20 test molecules, probe positions were identified in three categories:
**atom** (non-structural tokens, vocab id ≥3 and not ring/branch opener; subsampled to 3 per molecule),
**ring_count** (position immediately following a ring opener token), and
**branch_opener** (positions carrying a branch opener token id).
Two masking modes were tested: **mode 1** masks one target position at a time (all other
ground-truth tokens visible); **mode 2** masks all positions of that category simultaneously.
For each masked position, one conditional forward pass was run (CFG=1.0, timestep proportional
to fraction masked) and the predicted probability distribution over the 110-token vocab was recorded.
Total forward passes: 1906 batches × ≤32 = 60982 probe instances.
Wall-clock time: 333.6s on mps.

---

## Headline Results

| Category | Mode | N positions | Top-1 acc | Mean p(gt) | Mean entropy |
|---|---|---|---|---|---|
| atom | one-at-a-time | 7621 | 0.9033 | 0.8776 | 0.1914 |
| atom | all-at-once | 7621 | 0.8983 | 0.8706 | 0.2078 |
| ring_count | one-at-a-time | 7482 | 0.7760 | 0.7387 | 0.4177 |
| ring_count | all-at-once | 7482 | 0.7288 | 0.6901 | 0.4780 |
| branch_opener | one-at-a-time | 15388 | 0.9452 | 0.9325 | 0.1194 |
| branch_opener | all-at-once | 15388 | 0.9322 | 0.9166 | 0.1508 |

---

## Stratified by Ring Complexity

### Ring Count Token Accuracy

| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |
|---|---|---|---|---|
| 0 | one-at-a-time | 178 | 0.9213 | 0.8858 |
| 0 | all-at-once | 178 | 0.9157 | 0.8853 |
| 1-2 | one-at-a-time | 2030 | 0.8034 | 0.7763 |
| 1-2 | all-at-once | 2030 | 0.7823 | 0.7533 |
| 3+ | one-at-a-time | 5274 | 0.7605 | 0.7193 |
| 3+ | all-at-once | 5274 | 0.7019 | 0.6592 |

### Branch Opener Token Accuracy

| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |
|---|---|---|---|---|
| 0 | one-at-a-time | 2786 | 0.9720 | 0.9645 |
| 0 | all-at-once | 2786 | 0.9659 | 0.9588 |
| 1-2 | one-at-a-time | 5099 | 0.9555 | 0.9454 |
| 1-2 | all-at-once | 5099 | 0.9455 | 0.9317 |
| 3+ | one-at-a-time | 7503 | 0.9283 | 0.9118 |
| 3+ | all-at-once | 7503 | 0.9107 | 0.8906 |

---

## Distribution Analysis

Maximum possible entropy (uniform over 110 tokens): 4.7005 nats.

| Category | Mean entropy | % of max entropy | Mean p(gt) mode1 |
|---|---|---|---|
| atom | 0.1914 | 4.1% | 0.8776 |
| ring_count | 0.4177 | 8.9% | 0.7387 |
| branch_opener | 0.1194 | 2.5% | 0.9325 |

A low entropy (small % of max) means the model is confident. A high entropy means
the predicted distribution is diffuse. Mean p(gt) shows how much probability mass
the model assigns to the correct token regardless of its rank.

---

## Key Numbers to Remember

- **Atom (control) top-1 accuracy (mode1):** 0.9033  (7621 positions)
- **Ring count top-1 accuracy (mode1):** 0.7760  (7482 positions)
- **Branch opener top-1 accuracy (mode1):** 0.9452  (15388 positions)

---

## Verdict

MODEL KNOWS COUNT TOKENS: high accuracy on count tokens with full context. The dep_aware null result is not explained by capability gap; the failure must be elsewhere (decoding ordering interactions, encoder routing, etc).

---
