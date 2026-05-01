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
Wall-clock time: 371.6s on mps.

---

## Headline Results

| Category | Mode | N positions | Top-1 acc | Mean p(gt) | Mean entropy |
|---|---|---|---|---|---|
| atom | one-at-a-time | 7621 | 0.9093 | 0.8903 | 0.1607 |
| atom | all-at-once | 7621 | 0.9051 | 0.8841 | 0.1756 |
| ring_count | one-at-a-time | 7482 | 0.7914 | 0.7569 | 0.3904 |
| ring_count | all-at-once | 7482 | 0.7513 | 0.7153 | 0.4365 |
| branch_opener | one-at-a-time | 15388 | 0.9510 | 0.9390 | 0.1100 |
| branch_opener | all-at-once | 15388 | 0.9390 | 0.9261 | 0.1353 |

---

## Stratified by Ring Complexity

### Ring Count Token Accuracy

| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |
|---|---|---|---|---|
| 0 | one-at-a-time | 178 | 0.9326 | 0.9019 |
| 0 | all-at-once | 178 | 0.9326 | 0.9014 |
| 1-2 | one-at-a-time | 2030 | 0.8310 | 0.8009 |
| 1-2 | all-at-once | 2030 | 0.8148 | 0.7840 |
| 3+ | one-at-a-time | 5274 | 0.7713 | 0.7350 |
| 3+ | all-at-once | 5274 | 0.7207 | 0.6826 |

### Branch Opener Token Accuracy

| Ring bin | Mode | N | Top-1 acc | Mean p(gt) |
|---|---|---|---|---|
| 0 | one-at-a-time | 2786 | 0.9788 | 0.9723 |
| 0 | all-at-once | 2786 | 0.9763 | 0.9693 |
| 1-2 | one-at-a-time | 5099 | 0.9649 | 0.9539 |
| 1-2 | all-at-once | 5099 | 0.9551 | 0.9420 |
| 3+ | one-at-a-time | 7503 | 0.9312 | 0.9166 |
| 3+ | all-at-once | 7503 | 0.9143 | 0.8992 |

---

## Distribution Analysis

Maximum possible entropy (uniform over 110 tokens): 4.7005 nats.

| Category | Mean entropy | % of max entropy | Mean p(gt) mode1 |
|---|---|---|---|
| atom | 0.1607 | 3.4% | 0.8903 |
| ring_count | 0.3904 | 8.3% | 0.7569 |
| branch_opener | 0.1100 | 2.3% | 0.9390 |

A low entropy (small % of max) means the model is confident. A high entropy means
the predicted distribution is diffuse. Mean p(gt) shows how much probability mass
the model assigns to the correct token regardless of its rank.

---

## Key Numbers to Remember

- **Atom (control) top-1 accuracy (mode1):** 0.9093  (7621 positions)
- **Ring count top-1 accuracy (mode1):** 0.7914  (7482 positions)
- **Branch opener top-1 accuracy (mode1):** 0.9510  (15388 positions)

---

## Verdict

MODEL KNOWS COUNT TOKENS: high accuracy on count tokens with full context. The dep_aware null result is not explained by capability gap; the failure must be elsewhere (decoding ordering interactions, encoder routing, etc).

---
