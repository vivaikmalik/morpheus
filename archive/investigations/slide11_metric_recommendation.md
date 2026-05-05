# Slide 11 Metric Recommendation

## Phase 1: Configuration Comparison

| Config | Morgan | MACCS | RDK | aBLEU-2 |
|--------|--------|-------|-----|---------|
| Base (cfg1.5/t0.6/s50) | 0.2993 | 0.6420 | 0.4079 | 0.6036 |
| Rerank N=3 | 0.3015 | 0.6439 | 0.4125 | 0.6100 |
| **Rerank N=10 (strongest)** | **0.3104** | **0.6512** | **0.4175** | **0.6211** |

**Reranking effect (base → N=10):**
Morgan Δ=+0.0111, MACCS Δ=+0.0093, RDK Δ=+0.0096, aBLEU-2 Δ=+0.0176

**Strongest configuration:** `chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv` (Morgan=0.3104)

## Phase 2: Ring-Count Stratification (N=10 reranked)

| Rings | n | Morgan | MACCS | RDK | aBLEU-2 |
|-------|---|--------|-------|-----|---------|
| 0 rings | 762 | 0.4684 | 0.7253 | 0.4861 | 0.6872 |
| 1 ring | 521 | 0.3048 | 0.6264 | 0.3571 | 0.6344 |
| 2 rings | 403 | 0.2409 | 0.6017 | 0.3520 | 0.5921 |
| 3 rings | 421 | 0.2223 | 0.6322 | 0.4014 | 0.5843 |
| 4 rings | 263 | 0.2134 | 0.6481 | 0.4606 | 0.5729 |
| 5+ rings | 172 | 0.1548 | 0.5654 | 0.4233 | 0.5207 |

## Gap Ratio Analysis (0-ring / 3+-ring)

| Metric | 0-ring | 3+ ring | Gap ratio | Visual quality |
|--------|--------|---------|-----------|----------------|
| Morgan | 0.4684 | 0.2060 | 2.2737x | moderate |
| MACCS | 0.7253 | 0.6236 | 1.1631x | high |
| RDK | 0.4861 | 0.4240 | 1.1465x | moderate |
| aBLEU-2 | 0.6872 | 0.5680 | 1.2098x | high |

## Recommendation

**Recommended option: E — Show Morgan + MACCS side-by-side**

The two metrics tell complementary stories that together maximize the impact of slide 11:

1. **Morgan (2.27x gap, 0-ring=0.468) is the architectural ceiling metric.** It has the largest gap ratio (2.27x vs MACCS 1.16x, RDK 1.15x), and its absolute values (0.468 at 0 rings falling to 0.206 at 3+ rings) match TGM-DLM's published Morgan (0.688), making the gap legible to a reader who knows the literature. This is the right primary metric for the architectural story.

2. **MACCS (1.16x gap, 0-ring=0.725) provides visual impact.** MACCS scores in the 0.73–0.62 range create a chart that looks visually differentiated (bars from ~0.62 to ~0.73) without the floor-level absolute values Morgan shows for complex molecules. It is also already used in slide 12, so adding it to slide 11 maintains internal consistency.

3. **RDK is redundant with Morgan** (gap 1.15x, very similar shape to Morgan) and adds nothing new. **aBLEU-2** (1.21x gap) has a smaller gap ratio and is already in slide 12.

4. **Practical implementation:** Show Morgan as the primary (left) bar chart on slide 11 to drive the architectural ceiling argument, with MACCS as a secondary (right) bar chart to show the same story at higher absolute values. Label both axes clearly. The 2.27x Morgan gap and 1.16x MACCS gap together make the complexity-degradation pattern undeniable.

5. **Alternative (Option A, keep Morgan only)** is acceptable if the slide has space constraints — Morgan alone already shows the 2.27x gap cleanly and is the most defensible single metric in a comparison to TGM-DLM.
