# Replication Experiment Results

**Setup:** 3 new molecule draws (seeds 43/44/45) × 2 strategies × 40 molecules each.
**Goal:** Verify whether the original smoke test signal (seed=42, +0.023 Morgan on 3+r/3+b) was real.

## Per-seed Summary

| Seed | n | Morgan std | Morgan dep | Morgan Δ | MACCS std | MACCS dep | MACCS Δ | atom_bleu2 std | atom_bleu2 dep | atom_bleu2 Δ |
|------|---|-----------|-----------|---------|----------|----------|--------|---------------|---------------|-------------|
| 43 | 40 | 0.1982 | 0.1960 | -0.0023 | 0.5727 | 0.5889 | +0.0161 | 0.5029 | 0.5253 | +0.0224 |
| 44 | 40 | 0.2084 | 0.2113 | +0.0030 | 0.6036 | 0.6116 | +0.0080 | 0.5493 | 0.5652 | +0.0159 |
| 45 | 40 | 0.2286 | 0.2184 | -0.0102 | 0.6197 | 0.6090 | -0.0108 | 0.6089 | 0.5772 | -0.0317 |
| **mean** | - | 0.2117 | 0.2086 | -0.0032 | 0.5987 | 0.6031 | +0.0044 | 0.5537 | 0.5559 | +0.0022 |

## Per-stratum Results (averaged across seeds)

*Std dev of Morgan Δ across seeds = noise estimate for each stratum.*

| Stratum | n (total) | Morgan std (mean) | Morgan dep (mean) | Morgan Δ (mean) | Δ std-dev |
|---------|-----------|------------------|------------------|-----------------|-----------|
| 3+r/3+b (hardest) | 60 | 0.1890 | 0.1870 | -0.0020 | 0.0083 |
| 1-2r/3+b | 30 | 0.2184 | 0.2103 | -0.0081 | 0.0068 |
| 3+r/1-2b | 15 | 0.1897 | 0.1771 | -0.0126 | 0.0218 |
| 0r/3+b (CONTROL) | 15 | 0.3115 | 0.3230 | +0.0115 | 0.0289 |

## Comparison to Original Smoke Test (seed=42)

| Stratum | Original Δ (n) | Replication Δ (mean) | Replication Δ std |
|---------|---------------|---------------------|------------------|
| 3+r/3+b (hardest) | +0.023 (n=20) | -0.0020 | 0.0083 |
| 1-2r/3+b | +0.011 (n=10) | -0.0081 | 0.0068 |
| 3+r/1-2b | +0.001 (n=5) | -0.0126 | 0.0218 |
| 0r/3+b (CONTROL) | +0.000 (n=5) | +0.0115 | 0.0289 |

## Verdict

**REPLICATION FAILED:** The 3 replication seeds show inconsistent or near-zero deltas on the hard 3+r/3+b cell (mean Δ = -0.0020, std = 0.0083). The original smoke test result was within sample variance and not a real signal.
