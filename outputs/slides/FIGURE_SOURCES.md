# Figure Sources — Morpheus Slide Deck

Generated: 2026-04-10

---

## fig_progression_waterfall (Slide 10)

**Type:** Horizontal waterfall bar chart
**Slide:** 10 — Ablation / progression
**Source data:** `outputs/summary_all_runs.csv` (Morgan column, best cfg per stage)

| Stage | Morgan | Delta | Source row |
|-------|--------|-------|-----------|
| BGE frozen (pre-EOS-fix) | 0.159 | — | chebi20_eval.csv (pre_eos_fix=True) |
| EOS fix + tuning | 0.199 | +0.040 | chebi20_27M_frozen (cfg=2.0, t=0.6) |
| Contrastive BGE | 0.239 | +0.040 | chebi20_27M_contrastive (cfg=1.0, t=0.6) |
| SciBERT contrastive 10ep | 0.251 | +0.012 | chebi20_27M_scibert_contrastive (cfg=1.5) |
| SciBERT 20ep | 0.299 | +0.048 | chebi20_27M_scibert_20ep_contrastive (cfg=1.5) |
| N=10 reranking | 0.310 | +0.011 | same checkpoint, --rerank_n 10 |

TGM-DLM reference: 0.688 (AAAI 2024, 180M params, A100)

---

## fig_cfg_sweep (Slide 10)

**Type:** Line chart
**Slide:** 10 — Hyperparameter analysis
**Source data:** `outputs/summary_all_runs.csv`, model_size=27M_scibert_20ep, temp=0.6, steps=50

| CFG | Morgan |
|-----|--------|
| 0.5 | 0.230 |
| 0.8 | 0.271 |
| 1.0 | 0.288 |
| 1.5 | 0.299 |
| 2.0 | 0.288 |
| 2.5 | 0.259 |
| 3.0 | 0.224 |

---

## fig_ring_stratification (Slide 11)

**Type:** Grouped bar chart
**Slide:** 11 — Error analysis / stratification
**Source data:** `outputs/stratified_summary_tables.csv`, stratification=ring_count

| Rings | n | Morgan | BLEU-2 |
|-------|---|--------|--------|
| 0 rings | 762 | 0.451 | 0.672 |
| 1-2 rings | 924 | 0.266 | 0.605 |
| 3+ rings | 856 | 0.200 | 0.597 |

---

## fig_quality_distribution (Slide 11)

**Type:** Horizontal bar chart
**Slide:** 11 — Quality overview
**Source data:** `outputs/stratified_analysis_best_model.csv`, morgan_sim column

| Bin | Morgan range | Count | % |
|-----|-------------|-------|---|
| Poor | <0.1 | 202 | 7.9% |
| Mediocre | 0.1-0.3 | 1430 | 56.3% |
| Decent | 0.3-0.5 | 510 | 20.1% |
| Good | 0.5-0.7 | 225 | 8.9% |
| Excellent | >0.7 | 175 | 6.9% |

---

## fig_rl_results (Slide 8)

**Type:** Dual grouped bar chart
**Slide:** 8 — RL fine-tuning results
**Source data:** `outputs/results_summary.csv`

| Config | QED | MW (Da) |
|--------|-----|---------|
| Base (no RL) | 0.742 | 295.0 |
| QED only | 0.799 | 293.0 |
| QED + MW | 0.818 | 330.4 |
| QED + MW + Div | 0.837 | 317.7 |

---

## fig_eos_fix (Slide 9)

**Type:** Grouped bar chart
**Slide:** 9 — EOS fix ablation
**Source data:** `outputs/summary_all_runs.csv`, pre_eos_fix=True vs post-fix BGE frozen best

| Metric | Pre-fix | Post-fix | Delta |
|--------|---------|----------|-------|
| Morgan | 0.159 | 0.199 | +25% |
| MACCS | 0.462 | 0.536 | +16% |
| RDK | 0.248 | 0.313 | +26% |

Pre-fix: chebi20_eval.csv (cfg=3.0, temp=1.0, steps=200)
Post-fix: chebi20_27M_frozen (cfg=2.0, temp=0.6, steps=50)

---

## fig_scaling (Slide 12)

**Type:** Log-scale scatter plot
**Slide:** 12 — Scaling analysis
**Source data:** Checkpoint param counts + best Morgan scores

| Model | Params (M) | Morgan | Notes |
|-------|-----------|--------|-------|
| Morpheus 27M | 27 | 0.310 | rerank@10, cfg=1.5 |
| Morpheus 150M | 150 | 0.463 | teammate's numbers — placeholder |
| TGM-DLM | 180 | 0.688 | AAAI 2024 reported |

---

## fig_encoder_comparison (Slide 10, supplementary)

**Type:** Grouped bar chart (3 metrics x 4 encoders)
**Slide:** 10 — Encoder ablation
**Source data:** `outputs/summary_all_runs.csv`, best cfg per encoder variant

| Encoder | Morgan | MACCS | Atom BLEU-2 |
|---------|--------|-------|-------------|
| BGE frozen | 0.199 | 0.536 | 0.424 |
| BGE contrastive | 0.239 | 0.589 | 0.515 |
| SciBERT 10ep | 0.251 | 0.599 | 0.554 |
| SciBERT 20ep | 0.299 | 0.642 | 0.604 |


---

## fig_rl_results — **DEPRECATED** (superseded by fig_rl_progression)

Original RL figure with dual QED/MW subplots. Retired because the framing
of QED-only as "reward hacking" was overclaimed (MW 295→293 is noise).
File kept for reference, not used in slides.

---

## fig_scaling — **DEPRECATED** (superseded by fig_comparison_bars)

Log-scale scatter plot. Replaced by direct grouped bar comparison which
communicates the performance gap more clearly.
File kept for reference, not used in slides.

---

## fig_rl_progression (Slide 8 — replaces fig_rl_results)

**Type:** Dual subplot — QED bars + grouped diversity bars
**Slide:** 8 — RL fine-tuning results
**Source data:** `outputs/results_summary.csv`

Story: RL steadily improves QED (+12.8%); adding MW and diversity constraints
maintains structural variety rather than collapsing to a narrow distribution.

| Config | QED | Uniqueness % | Scaffold Div % |
|--------|-----|-------------|----------------|
| Base (no RL) | 0.742 | 100.0 | 95.4 |
| QED only | 0.799 | 98.8 | 88.6 |
| QED + MW | 0.818 | 91.2 | 95.2 |
| QED + MW + Div | 0.837 | 93.4 | 94.0 |

---

## fig_comparison_bars — **DEPRECATED** (superseded by fig_efficiency_comparison)

**Formerly Slide 12.** Replaced by efficiency-focused framing.

~~## fig_comparison_bars (Slide 12 — replaces fig_scaling)~~

**Type:** Grouped bar chart (3 metrics × 3 systems)
**Slide:** 12 — Architecture comparison
**Source data:** `outputs/summary_all_runs.csv` + teammate 150M numbers (pending)

| System | Morgan | MACCS | Atom BLEU-2 |
|--------|--------|-------|-------------|
| Morpheus 27M | 0.310 | 0.651 | 0.621 |
| Morpheus 150M | 0.463 | 0.699 | — (pending) |
| TGM-DLM ~180M | 0.688 | 0.854 | 0.826 |

Note: 150M Atom BLEU-2 missing — shown as blank bar with "—" annotation.
TGM-DLM reference: AAAI 2024 reported numbers.


---

## fig_efficiency_comparison — **SUPERSEDED** by fig_comparison_three_panel

~~## fig_efficiency_comparison (Slide 12 — replaces fig_comparison_bars)~~

**Type:** Two-panel figure (grouped bar + dual horizontal bar)
**Slide:** 12 — Comparison / positioning
**Source data:** `outputs/summary_all_runs.csv` + TGM-DLM AAAI 2024 reported numbers

### Left panel: Per-Parameter Efficiency
| Metric | Morpheus 27M (score/M) | TGM-DLM 180M (score/M) | Multiplier |
|--------|----------------------|----------------------|-----------|
| Morgan | 0.01148 | 0.00382 | 3.0× |
| MACCS | 0.02411 | 0.00474 | 5.1× |
| Atom BLEU-2 | 0.02300 | 0.00459 | 5.0× |

### Right panel upper: % of TGM-DLM score achieved
| Metric | Morpheus score | TGM-DLM score | % achieved |
|--------|---------------|---------------|-----------|
| Morgan | 0.310 | 0.688 | 45.1% |
| MACCS | 0.651 | 0.854 | 76.2% |
| Atom BLEU-2 | 0.621 | 0.826 | 75.2% |

### Right panel lower: Validity comparison
| System | Validity | Notes |
|--------|---------|-------|
| Morpheus 27M | 100% | SELFIES guarantee — no correction needed |
| TGM-DLM (raw) | 87% | Before post-processing correction |
| TGM-DLM (corrected) | ~100% | Requires separate correction network |

**Visual story:** 3-5× more efficient per parameter; reach 75-76% on MACCS/BLEU at 15% param count;
100% validity by construction vs 87% raw for TGM-DLM.
Morgan gap (45%) shown honestly in lighter orange.
Footer: training hardware comparison (MacBook M2 vs research GPU cluster).


---

## fig_comparison_three_panel — **SUPERSEDED** (split into three separate figures)

Replaced by fig_compare_params, fig_compare_validity, fig_compare_performance.

~~## fig_comparison_three_panel — CURRENT VERSION (Slide 12)~~

**Type:** Three-panel figure (vertical bars | horizontal bars | grouped vertical bars)
**Slide:** 12 — Main comparison vs TGM-DLM
**Source data:** Morpheus evaluation CSVs + Gong et al. (AAAI 2024), Table 1

Numbers verified from TGM-DLM paper (Gong et al., AAAI 2024):

| Metric | Morpheus 27M | TGM-DLM ~180M | Source |
|--------|-------------|---------------|--------|
| Parameters | 27M | ~180M | checkpoints/README.md; paper abstract |
| Validity | 100% | 87.1% (with correction) | SELFIES property; paper Table 1 |
| MACCS FTS | 0.651 | 0.854 | eval CSV; paper Table 1 |
| atom-BLEU-2 | 0.621 | 0.826 | eval CSV; paper Table 1 |

Note: TGM-DLM validity 87.1% is their BEST reported number (WITH correction phase).
Raw validity without correction is 78.9% — not shown to be fair.
Morgan not included in this figure (preliminary presentation leads with strongest claims).

### Panel 1: Model size
- Morpheus 27M (blue) vs TGM-DLM ~180M (orange)
- Double-headed arrow annotation: "6.7× smaller"

### Panel 2: Validity
- Morpheus 100% (green, SELFIES guarantee, no correction)
- TGM-DLM 87.1% (orange, with correction phase)
- Note below: correction network required by TGM-DLM

### Panel 3: MACCS & atom-BLEU-2
- Morpheus reaches 76% of TGM-DLM on MACCS, 75% on atom-BLEU-2
- At 15% of parameter count


---

## fig_compare_params (Slide 12) — CURRENT

**Type:** Vertical bar chart
**Slide:** 12, panel 1
**Numbers verified:** Gong et al. AAAI 2024 (abstract / model description)

| System | Parameters |
|--------|-----------|
| Morpheus | 27M (trainable) |
| TGM-DLM | ~180M |

Annotation: "6.7x smaller" (lowercase x)

---

## fig_compare_validity (Slide 12) — CURRENT

**Type:** Horizontal bar chart
**Slide:** 12, panel 2
**Numbers verified:** Gong et al. AAAI 2024, Table 1 (with correction phase)

| System | Validity |
|--------|---------|
| Morpheus | 100% (SELFIES construction, no correction) |
| TGM-DLM | 87.1% (with their separate correction network) |

Note: TGM-DLM raw validity (without correction) is 78.9% per paper. Not shown here.

---

## fig_compare_performance (Slide 12) — CURRENT

**Type:** Grouped vertical bar chart
**Slide:** 12, panel 3
**Numbers verified:** Gong et al. AAAI 2024, Table 1; Morpheus eval CSV

| Metric | Morpheus 27M | TGM-DLM ~180M | % of TGM-DLM |
|--------|-------------|---------------|-------------|
| MACCS FTS | 0.651 | 0.854 | 76% |
| atom-BLEU-2 | 0.621 | 0.826 | 75% |

Morgan not included (preliminary presentation leads with strongest claims).
