# Slide 10 Data Inventory
Read-only audit of eval CSVs to support MACCS regeneration of slide 10 charts.

---

## STEP 1 — Eval CSV Inventory

### outputs/evaluations/ (RL evaluation only, 5 files)
| Filename | Size (KB) | Rows | Modified | Relevant? |
|----------|-----------|------|----------|-----------|
| eval_base_model.csv | 143.2 | 500 | 2026-03-21 | No — RL config, not encoder ablation |
| eval_base_model_recheck.csv | 142.7 | 500 | 2026-03-22 | No |
| eval_rl_qed_mw_50step.csv | 163.8 | 500 | 2026-03-21 | No |
| eval_rl_qed_mw_div_50step.csv | 158.8 | 500 | 2026-03-21 | No |
| eval_rl_qed_only.csv | 150.4 | 500 | 2026-03-21 | No |

### outputs/ (ChEBI-20 encoder ablation, 66 total eval CSVs)
Relevant subset for slide 10 (6 ablation stages + CFG sweep): listed in Steps 5–6 below.

---

## STEP 2/3 — Column Schema

All encoder-ablation eval CSVs share the same per-molecule columns:
```
prompt, gt_smiles, gen_smiles, valid, exact_match,
bleu2, bleu4, lev_sim, morgan_sim, maccs_sim, rdk_sim
```
Newer files (SciBERT 10ep, 20ep) also have: `atom_bleu2`, `atom_bleu4`

**Exception — `eval_27M_frozen.csv`:** Uses an *older* schema with `tanimoto` (not `morgan_sim`/`maccs_sim`/`rdk_sim`), `gt_selfies`/`gen_selfies` instead of SMILES, plus `bleu`, `levenshtein`, `qed`, `logp`, `mw`. This is a MolInst-dataset baseline, NOT the ChEBI-20 BGE frozen baseline used in the waterfall. Do not confuse.

**Exception — `chebi20_eval.csv` and `chebi20_eval_eost.csv`:** Have morgan_sim, maccs_sim, rdk_sim but **do NOT have** atom_bleu2 or atom_bleu4.

**Exception — `chebi20_eval_cfg*.csv` (BGE frozen post-EOS-fix):** Have morgan_sim, maccs_sim, rdk_sim but **do NOT have** atom_bleu2 or atom_bleu4.

---

## STEP 4 — Mean Metric Values (ablation stages at best CFG, valid only)

| Stage / Encoder | File | Morgan | MACCS | RDK | aBLEU-2 | Validity |
|-----------------|------|--------|-------|-----|---------|---------|
| BGE frozen (pre-EOS-fix) | chebi20_eval.csv | 0.1587 | 0.4617 | 0.2477 | N/A | 100% |
| BGE frozen (post-EOS-fix) best cfg=1.5 | chebi20_eval_cfg1.5_t0.6_s50.csv | 0.2022 | 0.5385 | 0.3198 | N/A | 100% |
| BGE frozen (post-EOS-fix) cfg=2.0 (used in current waterfall) | chebi20_eval_cfg2.0_t0.6_s50.csv | 0.1987 | 0.5358 | 0.3128 | N/A | 100% |
| BGE contrastive best (cfg=1.5, t=0.4) | chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv | 0.2385 | 0.5891 | 0.3486 | 0.5146 | 100% |
| SciBERT 10ep best (cfg=1.5, t=0.6) | chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.6_s50.csv | 0.2511 | 0.5993 | 0.3627 | 0.5537 | 100% |
| SciBERT 20ep best (cfg=1.5, t=0.6) | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv | 0.2993 | 0.6420 | 0.4079 | 0.6036 | 100% |
| SciBERT 20ep + N=3 rerank | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank3.csv | 0.3015 | 0.6439 | 0.4125 | 0.6100 | 100% |
| SciBERT 20ep + N=10 rerank | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv | 0.3104 | 0.6512 | 0.4175 | 0.6211 | 100% |

---

## STEP 5 — Ablation Waterfall Stages

The current waterfall (fig_progression_waterfall, Slide 10) uses 6 stages per FIGURE_SOURCES.md.
MACCS values are available for all 6 stages.

| Stage | Morgan (current) | MACCS (available) | Δ MACCS | File to use |
|-------|-----------------|-------------------|---------|-------------|
| 1. BGE frozen (pre-EOS-fix) | 0.159 | **0.462** | — | chebi20_eval.csv |
| 2. EOS fix + tuning | 0.199 | **0.536** | +0.074 | chebi20_eval_cfg2.0_t0.6_s50.csv |
| 3. BGE contrastive | 0.239 | **0.589** | +0.053 | chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv |
| 4. SciBERT contrastive 10ep | 0.251 | **0.599** | +0.010 | chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.6_s50.csv |
| 5. SciBERT contrastive 20ep | 0.299 | **0.642** | +0.043 | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv |
| 6. N=10 reranking | 0.310 | **0.651** | +0.009 | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv |

**UNCLEAR FLAG — BGE contrastive stage file mapping:**
FIGURE_SOURCES.md labels the BGE contrastive stage as "cfg=1.0, t=0.6" (Morgan=0.239),
but `chebi20_27M_contrastive_eval_cfg1.0_t0.6_s50.csv` gives Morgan=0.2298 (not 0.239).
The value 0.239 actually matches `cfg1.5_t0.4` (Morgan=0.23850, rounds to 0.239).
This suggests FIGURE_SOURCES has a stale label; the actual file used was likely cfg=1.5, t=0.4.
**Recommendation: use `chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv` (Morgan=0.239, MACCS=0.589).**

**UNCLEAR FLAG — EOS-fix stage CFG choice:**
FIGURE_SOURCES uses cfg=2.0 for EOS-fix stage (Morgan=0.199), but cfg=1.5 is actually the
peak (Morgan=0.202, MACCS=0.539). The current slide deliberately uses cfg=2.0 (lower but
consistent with "what was deployed"). Regenerating MACCS should use the same cfg=2.0 file
for waterfall continuity, UNLESS the intent is to show best possible per stage.

**aBLEU-2 gap:** Stages 1 and 2 (pre-EOS-fix and BGE frozen) do NOT have atom_bleu2 column.
Only 4 of 6 stages have aBLEU-2. This is not a problem for MACCS regeneration but rules out
showing aBLEU-2 in the waterfall without a data gap at the first two stages.

---

## STEP 6 — CFG Sweep Files (SciBERT 20ep)

Standard sweep: CFG ∈ {0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0} at t=0.6 (or closest available).
**All 7 points exist. All have maccs_sim. No gaps for MACCS regeneration.**

| CFG | Temp | Morgan (current) | MACCS (available) | File |
|-----|------|-----------------|-------------------|------|
| 0.5 | 0.6 | 0.230 | **0.539** | chebi20_27M_scibert_20ep_contrastive_eval_cfg0.5_t0.6_s50.csv |
| 0.8 | 0.6 | 0.271 | **0.612** | chebi20_27M_scibert_20ep_contrastive_eval_cfg0.8_t0.6_s50.csv |
| 1.0 | 0.6 | 0.288 | **0.633** | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.0_t0.6_s50.csv |
| 1.5 | 0.6 | 0.299 | **0.642** ← peak | chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv |
| 2.0 | 0.6 | 0.288 | **0.630** | chebi20_27M_scibert_20ep_contrastive_eval_cfg2.0_t0.6_s50.csv |
| 2.5 | 0.8* | 0.259 | **0.591** | chebi20_27M_scibert_20ep_contrastive_eval_cfg2.5_t0.8_s50.csv |
| 3.0 | 0.8* | 0.224 | **0.551** | chebi20_27M_scibert_20ep_contrastive_eval_cfg3.0_t0.8_s50.csv |

*No t=0.6 file exists for CFG 2.5 or 3.0 — t=0.8 is used in both Morgan and MACCS sweeps.

**Sweep shape:** MACCS peaks at CFG=1.5 (0.642), same as Morgan. The curve is consistent.
MACCS absolute range (0.539–0.642) gives a visually tighter sweep than Morgan (0.224–0.299),
but the relative shape is identical. Both tell the same "CFG=1.5 is optimal" story.

---

## STEP 7 — Current Slide Figures

Located in `outputs/slides/`:
| Figure | Size (KB) | Modified |
|--------|-----------|----------|
| fig_progression_waterfall.png | 177.2 | 2026-04-10 14:23 |
| fig_cfg_sweep.png | 174.3 | 2026-04-10 14:23 |

Source data: both charts draw from `outputs/summary_all_runs.csv` (41 rows).
summary_all_runs.csv contains pre-computed maccs_sim values for all relevant runs.
No separate waterfall_data.csv or cfg_sweep_data.csv exists — the generation script
reads summary_all_runs.csv directly and selects rows by encoder + cfg.

**Key note:** summary_all_runs.csv is INCOMPLETE for BGE contrastive (only 2 rows: cfg=1.0 at t=0.6 and t=0.8).
The full BGE contrastive sweep (cfg=1.5, 2.0, 2.5, 3.0) is NOT in summary_all_runs.csv but IS available
as individual per-molecule CSVs in outputs/. The regeneration script must read directly from the per-molecule CSVs,
not from summary_all_runs.csv, to get the correct best-cfg BGE contrastive value.

---

## STEP 8 — Summary and Recommendation

**Six-sentence summary:**

1. All 6 waterfall stages have complete MACCS data (maccs_sim column present in every file).
   The MACCS values for the 6-stage waterfall are: 0.462 → 0.536 → 0.589 → 0.599 → 0.642 → 0.651
   (vs. Morgan: 0.159 → 0.199 → 0.239 → 0.251 → 0.299 → 0.310).

2. All 7 CFG sweep points have complete MACCS data.
   MACCS peak is at CFG=1.5 (0.642), same as Morgan — the curve shapes are identical.
   No missing files and no column gaps anywhere in either chart.

3. One UNCLEAR issue requires human judgment: FIGURE_SOURCES.md mislabels the BGE contrastive
   stage as "cfg=1.0" (which gives Morgan=0.2298), but the value 0.239 on the current slide
   actually corresponds to cfg=1.5, t=0.4 (Morgan=0.2385 ≈ 0.239). The regeneration script
   should use `chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv` for that stage.

4. A second UNCLEAR issue: the EOS-fix stage currently shows cfg=2.0 (Morgan=0.199) not the
   peak cfg=1.5 (Morgan=0.202). This is deliberate in the current slide (showing the
   "deployed configuration"), but for MACCS regeneration it changes MACCS from 0.539 (cfg=1.5)
   to 0.536 (cfg=2.0) — a tiny difference. Match existing behavior: use cfg=2.0 for consistency.

5. aBLEU-2 CANNOT be added to the waterfall: stages 1–2 (pre-EOS-fix and BGE frozen post-fix)
   do not have atom_bleu2 columns. MACCS and Morgan are the only metrics available for all 6 stages.

6. Recommended scope for regeneration: **full waterfall + full CFG sweep, both on MACCS**.
   Specific files: use the 6-file list in Step 5 for the waterfall, and the 7-file list in Step 6
   for the CFG sweep. Read maccs_sim from per-molecule CSVs (not summary_all_runs.csv, which is
   incomplete for BGE contrastive).

---

## Exact File List for Regeneration

### Waterfall (6 stages)
```
chebi20_eval.csv                                                      → Stage 1 BGE frozen pre-fix
chebi20_eval_cfg2.0_t0.6_s50.csv                                      → Stage 2 EOS fix
chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv                      → Stage 3 BGE contrastive
chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.6_s50.csv              → Stage 4 SciBERT 10ep
chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv         → Stage 5 SciBERT 20ep
chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv → Stage 6 N=10 reranking
```

### CFG sweep (7 points, SciBERT 20ep)
```
chebi20_27M_scibert_20ep_contrastive_eval_cfg0.5_t0.6_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg0.8_t0.6_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg1.0_t0.6_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg2.0_t0.6_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg2.5_t0.8_s50.csv
chebi20_27M_scibert_20ep_contrastive_eval_cfg3.0_t0.8_s50.csv
```
