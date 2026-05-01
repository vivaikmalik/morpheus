# Probe Comparison: 20-epoch vs 40-epoch Checkpoint

## Headline Comparison

| Category | Mode | n | 20ep top-1 | 40ep top-1 | Δ | 20ep p(gt) | 40ep p(gt) | Δ |
|---|---|---|---|---|---|---|---|---|
| atom | one-at-a-time | 7621 | 0.9033 | 0.9093 | +0.0060 | 0.8776 | 0.8903 | +0.0127 |
| atom | all-at-once | 7621 | 0.8983 | 0.9051 | +0.0068 | 0.8706 | 0.8841 | +0.0135 |
| ring_count | one-at-a-time | 7482 | 0.7760 | 0.7914 | +0.0154 | 0.7387 | 0.7569 | +0.0182 |
| ring_count | all-at-once | 7482 | 0.7288 | 0.7513 | +0.0225 | 0.6901 | 0.7153 | +0.0252 |
| branch_opener | one-at-a-time | 15388 | 0.9452 | 0.9510 | +0.0058 | 0.9325 | 0.9390 | +0.0065 |
| branch_opener | all-at-once | 15388 | 0.9322 | 0.9390 | +0.0068 | 0.9166 | 0.9261 | +0.0095 |

All three categories improved. Ring count shows the largest absolute gain (+0.0154 top-1).
Improvements are consistent across both modes.

## Stratified ring_count Comparison (mode1)

| Ring bin | n | 20ep top-1 | 40ep top-1 | Δ | 20ep p(gt) | 40ep p(gt) | Δ |
|---|---|---|---|---|---|---|---|
| 0 | 178 | 0.9213 | 0.9326 | +0.0113 | 0.8858 | 0.9019 | +0.0161 |
| 1-2 | 2030 | 0.8034 | 0.8310 | +0.0276 | 0.7763 | 0.8009 | +0.0246 |
| 3+ | 5274 | 0.7605 | 0.7713 | +0.0108 | 0.7193 | 0.7350 | +0.0157 |

The 1-2 ring stratum improved the most (+0.0276 top-1). The critical 3+ ring stratum
improved by +0.0108 top-1 and +0.0157 p(gt) — meaningful but modest. The 0-ring (trivial)
stratum also improved (+0.0113). Improvements are broad-based, not concentrated in the
failure regime.

## Stratified branch_opener Comparison (mode1)

| Ring bin | n | 20ep top-1 | 40ep top-1 | Δ | 20ep p(gt) | 40ep p(gt) | Δ |
|---|---|---|---|---|---|---|---|
| 0 | 2786 | 0.9720 | 0.9788 | +0.0068 | 0.9645 | 0.9723 | +0.0078 |
| 1-2 | 5099 | 0.9555 | 0.9649 | +0.0094 | 0.9454 | 0.9539 | +0.0085 |
| 3+ | 7503 | 0.9283 | 0.9312 | +0.0029 | 0.9118 | 0.9166 | +0.0048 |

Branch opener improvements are smaller (+0.0029 to +0.0094 top-1) and concentrated in the
1-2 ring and 3+ ring strata, where there's more room to improve. The 0-ring stratum was
already at 97.2% and reached 97.9%.

## Entropy and Confidence Shifts

| Category | 20ep entropy | 40ep entropy | Δ | 20ep p(gt) | 40ep p(gt) | Δ |
|---|---|---|---|---|---|---|
| atom (mode1) | 0.1914 | 0.1607 | -0.0307 | 0.8776 | 0.8903 | +0.0127 |
| ring_count (mode1) | 0.4177 | 0.3904 | -0.0273 | 0.7387 | 0.7569 | +0.0182 |
| branch_opener (mode1) | 0.1194 | 0.1100 | -0.0094 | 0.9325 | 0.9390 | +0.0065 |

The 40-epoch model is uniformly more confident: lower entropy across all categories,
higher p(gt) across all categories. The largest entropy drop is in atom tokens (-0.0307),
consistent with atoms being the foundational token category that other predictions condition on.

### Mode1 vs Mode2 gap (top-1 accuracy)

| Category | 20ep gap | 40ep gap | Δ gap |
|---|---|---|---|
| atom | +0.0050 | +0.0042 | -0.0008 |
| ring_count | +0.0472 | +0.0401 | -0.0071 |
| branch_opener | +0.0130 | +0.0120 | -0.0010 |

The Mode1-Mode2 gap (measuring sensitivity to concurrent masking) stayed roughly stable
for atoms and ring counts, but narrowed for branch openers (Δ gap = -0.0010). This means branch opener prediction is slightly more robust to missing context at 40 epochs.

## Interpretation

The 40-epoch model shows consistent, modest improvements across all three token categories:
+0.0060 top-1 for atoms, +0.0154 for ring counts, and +0.0058 for branch openers. The
improvements are broad-based rather than concentrated in the hardest stratum (3+ rings
gained +0.0108, while 1-2 rings gained +0.0276). The model became more confident overall
(lower entropy, higher p(gt)), with the largest confidence gain in atom tokens — the
foundational category that downstream count predictions condition on. The Mode1-vs-Mode2
gap is stable, meaning the model's robustness to concurrent masking (a rough proxy for
generation-time conditions) did not change significantly.

## Connection to Generation Results

Reference numbers:
- 20-epoch Morgan: 0.2944
- 40-epoch Morgan: 0.3164 (+0.0220, +7.5% relative)

Probe improvements (mode1 top-1):
- atom: +0.0060 (+0.7% relative)
- ring_count: +0.0154 (+2.0% relative)
- branch_opener: +0.0058 (+0.6% relative)

The probe improvements are **smaller** than the generation improvement in relative terms.
Morgan improved by +7.5% relative, while probe top-1 accuracy improved by +0.7% (atom),
+2.0% (ring_count), and +0.6% (branch_opener). The probe was already near ceiling for
atoms (90.3%) and branch openers (94.5%), so absolute gains are constrained. Ring count
had the most headroom and showed the largest relative improvement (+2.0%), but this is
still well below the +7.5% generation improvement.

This suggests the 40-epoch model's generation gains come primarily from improved *use* of
its existing per-token capability during multi-step decoding — better confidence calibration,
better handling of ambiguous contexts, better global coherence — rather than from a raw
increase in per-token prediction accuracy. The probe measures single-step capability with
ground-truth context; generation faces compounding errors over 50 steps with noisy context.
The generation improvement is larger because the 40-epoch model likely compounds errors
less, not because each individual prediction improved dramatically.

## Verdict

**EXPRESSION OUTPACED CAPABILITY:** Probe gains are smaller than generation gains.
The 40-epoch model's +7.5% relative Morgan improvement exceeds its +0.7–2.0% relative
probe improvements. Generation is recovering more of the existing per-token capability,
likely through better confidence calibration and reduced error compounding during iterative
decoding. The model did not gain dramatically more per-token knowledge; it got better at
expressing what it already knew.
