# Morpheus Checkpoint Directory

## Naming Convention

```
{dataset}_{params}_{encoder/stage}[_variant].pt
```

- **Pretrain** (`zinc_*`): uses stored parameter count (5M, 17M) — no cross-attention layers yet
- **Fine-tune** (`chebi20_*`, `molinst_*`): uses trainable parameter count (27M = 17M diffusion transformer + 10M cross-attention layers). Total including frozen BGE-large encoder: ~362M params.
- **RL** (`zinc_5M_rl_*`): starts from 5M pretrain, optimises molecular properties via REINFORCE

## Backward Compatibility

Symlinks from old names point to the canonical files. Teammate code using old names will continue to work — no changes needed.

| Old name (symlink) | Points to |
|---|---|
| `best_model.pt` | `zinc_5M_pretrain.pt` |
| `best_model_upscaled.pt` | `zinc_17M_pretrain.pt` |
| `best_finetuned_model.pt` | `molinst_27M_frozen.pt` |
| `best_finetuned_model_old.pt` | `molinst_27M_frozen_no_eos.pt` |
| `best_chebi20_model.pt` | `chebi20_27M_frozen.pt` |
| `best_chebi20_27M_frozen.pt` | `chebi20_27M_frozen.pt` |

## Checkpoint Table

| Filename | Stored Params | Trainable Params | Dataset | Encoder | Stage | EOS Fixed | Notes |
|---|---|---|---|---|---|---|---|
| `zinc_5M_pretrain.pt` | 4.4M | 4.4M | ZINC250K | — | pretrain | — | Original 5M unconditional diffusion model |
| `zinc_17M_pretrain.pt` | 17.5M | 17.5M | ZINC250K | — | pretrain | — | Upscaled 17M unconditional diffusion model (Stage 1) |
| `chebi20_27M_frozen.pt` | ~362M | 27M | ChEBI-20 | frozen BGE | fine-tune | ✅ | **Best ChEBI-20 model** — 10 epochs, EOS fix applied |
| `chebi20_27M_frozen_no_eos.pt` | ~362M | 27M | ChEBI-20 | frozen BGE | fine-tune | ❌ | Legacy backup before EOS fix |
| `chebi20_27M_frozen_epoch3.pt` | ~362M | 27M | ChEBI-20 | frozen BGE | fine-tune | ✅ | Per-epoch checkpoint at epoch 3 |
| `chebi20_27M_frozen_epoch6.pt` | ~362M | 27M | ChEBI-20 | frozen BGE | fine-tune | ✅ | Per-epoch checkpoint at epoch 6 |
| `chebi20_27M_frozen_epoch9.pt` | ~362M | 27M | ChEBI-20 | frozen BGE | fine-tune | ✅ | Per-epoch checkpoint at epoch 9 |
| `molinst_27M_frozen.pt` | ~362M | 27M | Mol-Instructions | frozen BGE | fine-tune | ✅ | **Best MolInst model** — EOS fix applied |
| `molinst_27M_frozen_no_eos.pt` | ~362M | 27M | Mol-Instructions | frozen BGE | fine-tune | ❌ | Legacy backup before EOS fix |
| `molinst_27M_frozen_epoch3.pt` | ~362M | 27M | Mol-Instructions | frozen BGE | fine-tune | ✅ | Kaggle T4 epoch 3 checkpoint |
| `contrastive_text_selfies.pt` | ~1.3GB | — | ChEBI-20 + custom | — | encoder | — | **Contrastive text-SELFIES aligner** (encoder only, not a diffusion checkpoint). Best val loss 1.648 at epoch 13. Use via `--encoder contrastive` flag. Requires `weights_only=False` when loading. |
| `contrastive_text_selfies.last.pt` | ~1.3GB | — | — | — | encoder | — | Last epoch (16) of contrastive training — slightly worse than best |
| `contrastive_text_selfies.history.json` | — | — | — | — | — | — | Training loss history for contrastive run |

### RL Checkpoints (all on 5M base model)

| Filename | Reward | Steps | Notes |
|---|---|---|---|
| `zinc_5M_rl_v1_step100.pt` | QED + diversity | 100 | REINFORCE v1 (deprecated) |
| `zinc_5M_rl_v1_step200.pt` | QED + diversity | 200 | REINFORCE v1 (deprecated) |
| `zinc_5M_rl_v2_step{100-500}.pt` | QED only | 100–500 | REINFORCE v2. Key finding: reward hacking via MW exploitation |
| `zinc_5M_rl_qed_mw_step{100-300}.pt` | QED + MW constraint | 100–300 | Corrected reward hacking, 32 denoising steps |
| `zinc_5M_rl_qed_mw_50s_step{100-300}.pt` | QED + MW constraint | 100–300 | Same reward, 50 denoising steps |
| `zinc_5M_rl_qed_mw_div_50s_step{100-300}.pt` | QED + MW + diversity | 100–300 | **Best RL run** — QED 0.84 vs base 0.74. 50 denoising steps + diversity penalty |

## Future Checkpoints

These will be created once contrastive fine-tuning runs complete:

| Filename | Command |
|---|---|
| `chebi20_27M_contrastive.pt` | `python finetune/train_chebi20.py --encoder contrastive` |
| `molinst_27M_contrastive.pt` | (via `train_text_condition.py` with contrastive encoder) |
