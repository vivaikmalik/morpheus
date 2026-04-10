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
