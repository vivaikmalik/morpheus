# Paper Claim Index

Map from paper text to the file in this repo that supports the claim. Paths reflect the state of the repo at the time of the cleanup; some paths under `outputs/` will be relocated to `archive/investigations/` in a later cleanup step and will be updated atomically when that move happens.

## Literal file references in `main.tex`

| main.tex line | Reference | Current path |
|---------------|-----------|--------------|
| L5 | `\input{math_commands.tex}` | not in repo (required for compile, expected on local LaTeX env) |
| L26 | `\url{https://github.com/vivaikmalik/morpheus}` | external URL |
| L66 | `\texttt{MolecularDiffusionModel}` | model/molecularDiffusionModel.py |
| L72 | `\texttt{encoder_proj}`, `\texttt{text_proj}` | model/molecularDiffusionModel.py:45-49,55,114-122 |
| L74 | `\texttt{null_token}` | model/molecularDiffusionModel.py:58 |
| L137 | `\texttt{selfies}` library | external Python package, version 2.1.1 |
| L321 | `\includegraphics{training_curves_comparison.png}` (Fig 1) | outputs/plots/comparison/training_curves_comparison.png |

## Section §3.1 architecture claims

| Claim | Supporting file |
|-------|-----------------|
| 8 blocks, hidden 512, 8 heads, FFN 1024, dropout 0.1, seq_len 74 | finetune/train_chebi20.py:82-88 (CONFIG) |
| Pre-norm; self-attn → cross-attn → FFN with SiLU | model/transformerBlock.py:130-148 |
| Cross-attention output projection zero-initialized | model/transformerBlock.py:66-67; model/molecularDiffusionModel.py:96-98 |
| Learned absolute position embedding | model/embeddings.py:65-79 |
| Output projection weight-tied with token embedding | model/molecularDiffusionModel.py:80 |
| `encoder_proj` 768→1024 near-identity init; `text_proj` 1024→1024→512 | model/molecularDiffusionModel.py:45-49,114-122 |
| Learned 512-d `null_token`; CFG dropout p=0.1 | model/molecularDiffusionModel.py:58,147-153 |
| Sinusoidal timestep + 2-layer MLP, added before block 0 | model/embeddings.py:9-55; model/molecularDiffusionModel.py:160-162 |
| Param counts 4.4M / 17M / 28M / 787K | computed from checkpoints/zinc_5M_pretrain.pt, zinc_17M_pretrain.pt, chebi20_27M_scibert_40ep_contrastive.pt, contrastive_scibert_chebi20_projonly.pt |

## Section §3.2 forward diffusion + training

| Claim | Supporting file |
|-------|-----------------|
| α_t = cos²(tπ/2); per-token Bernoulli masking; t per sequence | finetune/diffusionCollatorPrompt.py:30-36 |
| Loss: (1−t) timestep weight, w_EOS=5.0, w_PAD=0.05, ignore_index=-100 | finetune/train_chebi20.py:416-442 |
| All-MASK init, T=50, t linearly 1→0; least-confident re-mask | finetune/train_chebi20.py:539-572 |
| CFG: ℓ_cfg = ℓ_uncond + w·(ℓ_cond − ℓ_uncond), w=1.5, t=0.7 | finetune/train_chebi20.py:545-553 |
| AdamW, weight_decay=0.01, grad clip 1.0, cosine schedule with warmup | finetune/train_chebi20.py:1201-1212 |
| Pretrain LR 4e-4 batch 128 1000 warmup; FT LR 2e-4 batch 32 500 warmup | training/train_upscaled.py:60-65; finetune/train_chebi20.py:93-98 |
| Early stopping patience 5 | finetune/train_chebi20.py:107,1322-1326 |

## Section §3.3 contrastive aligner

| Claim | Supporting file |
|-------|-----------------|
| T_proj 768→768, M_proj 256→768, total 787,968 trainable | contrastive/model.py:34-40; checkpoints/contrastive_scibert_chebi20_projonly.pt config |
| Both encoders frozen for projonly run | contrastive/train_contrastive.py:201-208; checkpoints/contrastive_scibert_chebi20_projonly.pt config |
| Symmetric InfoNCE, τ=0.07 | contrastive/model.py:59-62 |
| ChEMBL pretrain → ChEBI-20 projection FT (batch 64, LR 2e-5, 25 epochs) | contrastive/train_contrastive.py; checkpoints/contrastive_scibert_chembl.history.json |
| ChEBI-20 retrieval R@1=0.554, R@5=0.872, R@10=0.934, MRR=0.689 | NOT VERIFIABLE — projonly history file not committed; chembl baseline in checkpoints/contrastive_scibert_chembl.history.json |

## Section §3.4 RL

| Claim | Supporting file |
|-------|-----------------|
| Per-step log-prob accumulation; advantage clamp [-2,2]; one-sided KL β=0.1; LR 5e-6; 15 denoising steps; T=0.9 | training/rl_reinforce_v2.py |
| Reward: r = QED + 0.3·s_MW − 0.3·s̄_Tanimoto | training/rl_reinforce_v2.py:188-232 |
| MW score piecewise-linear: 1.0 on [250,450], falls to 0 at 150/550 | training/rl_reinforce_v2.py:158-168 |

## Section §4 datasets

| Claim | Supporting file |
|-------|-----------------|
| ZINC250K = 249,455 molecules | data/combined_pretrain_v1/raw/zinc250k_raw_dl.csv (untracked) |
| ChEBI-20 split 20,158 / 2,504 / 2,542 | data/chebi20_train.csv, data/chebi20_val.csv, data/chebi20_test.csv |
| 23% excluded by 74-token cap; 60 vs 23 heavy atoms | archive/investigations/truncation_analysis.md |

## Section §5 results — per-claim evidence

| Claim | Supporting file |
|-------|-----------------|
| Table 1: 40ep Morgan 0.318, MACCS 0.661, RDK 0.431, atom-BLEU-2 0.614 | outputs/chebi20_27M_scibert_40ep_contrastive_eval_cfg1.5_t0.7_s50.csv |
| Table 1: TGM-DLM reference numbers | external — Gong et al. 2024 Table 1 |
| Table 2: cumulative ablation deltas | outputs/summary_all_runs.csv (rebuilt to 82 rows) |
| Table 2: BGE no-EOS Morgan 0.159 | outputs/chebi20_eval.csv |
| Table 2: BGE+EOS Morgan 0.202 | outputs/chebi20_eval_cfg1.5_t0.6_s50.csv |
| Table 2: BGE contrastive Morgan 0.238 | outputs/chebi20_27M_contrastive_eval_cfg1.5_t0.4_s50.csv |
| Table 2: SciBERT 10ep Morgan 0.252 | outputs/chebi20_27M_scibert_contrastive_eval_cfg1.5_t0.4_s50.csv |
| Table 2: SciBERT 20ep Morgan 0.299 | outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50.csv |
| Table 2: SciBERT 20ep + rerank10 Morgan 0.310 | outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_rerank10.csv |
| Table 3: ring-stratified analysis | outputs/stratified_analysis_best_model.csv |
| Table 4: probe top-1 accuracy 79.1% / 90.9% / 95.1% (40ep) | archive/investigations/probe_token_knowledge_40ep.md; outputs/probe_token_knowledge_raw_40ep.csv |
| Table 4: probe 20ep ring_count 77.6% | archive/investigations/probe_token_knowledge.md; outputs/probe_token_knowledge_raw.csv |
| §5.4 ring-count "around 40%" at generation | archive/investigations/branch_audit_summary.md (39.7% molecule-level) — see archived 40pct_source_audit.md for caveat |
| Table 5: dep-aware Δ Morgan +0.001 / −0.004 / −0.002 | outputs/chebi20_27M_scibert_20ep_contrastive_eval_cfg1.5_t0.6_s50_eost_FULL_DEPAWARE.csv (vs FULL_STANDARD); archive/investigations/replication_results.md |
| Table 5: iterative refinement deltas | outputs/chebi20_27M_scibert_40ep_contrastive_eval_cfg1.5_t0.7_s50_refine*.csv |
| Table 6: RL final QED 0.742/0.799/0.818/0.837 | outputs/evaluations/eval_base_model.csv, eval_rl_qed_only.csv, eval_rl_qed_mw_50step.csv, eval_rl_qed_mw_div_50step.csv |
| Figure 1 training trajectories | outputs/plots/comparison/training_curves_comparison.png |
| Dep-aware investigation | archive/investigations/dep_aware_investigation.md, archive/investigations/dep_aware_verification.md |
| Replication 4-seed verdict | archive/investigations/replication_results.md; outputs/replication_seed{43,44,45}_{standard,dep_aware}.csv |
| §5.5 hard-cell subset n=20/seed | data/chebi20_smoke_hard40_seed{42,43,44,45}*.csv |

## Section §6 discussion / §7 limitations / §8 conclusion

No new specific file references; all evidence already mapped above.

## Notes

- Evaluation CSVs in `outputs/` (top level, ~110 files) are run records; they remain at their current paths after cleanup.
- Audit/investigation MDs in `outputs/` (the ~19 `.md` files) are scheduled to move to `archive/investigations/` in a later cleanup step. When that happens, this index will be updated in the same commit.
- `outputs/plots/`, `outputs/slides/`, `outputs/rl/` subdirectories are not affected by cleanup.

## Known issues

Note: scripts/rl_distribution_analysis.py has unfilled f-string placeholders at script lines 362 and 364 (inner strings inside a conditional f-string expression that are not f-prefixed); the archived rl_distribution_summary.md reflects correct values from a pre-bug run. Re-running the script would produce output with literal '{var:.1f}' text rather than computed values.
