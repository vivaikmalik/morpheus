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

## Diffusion Model Checkpoints

| Filename | Trainable Params | Dataset | Encoder | EOS Fixed | Notes |
|---|---|---|---|---|---|
| `zinc_5M_pretrain.pt` | 4.4M | ZINC250K | — | — | Original 5M unconditional diffusion model |
| `zinc_17M_pretrain.pt` | 17.5M | ZINC250K | — | — | Upscaled 17M unconditional diffusion model (Stage 1) |
| `chebi20_27M_frozen.pt` | 27M | ChEBI-20 | frozen BGE | yes | BGE frozen, 10 epochs |
| `chebi20_27M_frozen_no_eos.pt` | 27M | ChEBI-20 | frozen BGE | no | Legacy backup before EOS fix |
| `chebi20_27M_frozen_epoch3.pt` | 27M | ChEBI-20 | frozen BGE | yes | Per-epoch checkpoint |
| `chebi20_27M_frozen_epoch6.pt` | 27M | ChEBI-20 | frozen BGE | yes | Per-epoch checkpoint |
| `chebi20_27M_frozen_epoch9.pt` | 27M | ChEBI-20 | frozen BGE | yes | Per-epoch checkpoint |
| `chebi20_27M_contrastive.pt` | 27M | ChEBI-20 | BGE contrastive | yes | BGE contrastive encoder, 10 epochs |
| `chebi20_27M_scibert_contrastive.pt` | 27M | ChEBI-20 | SciBERT contrastive | yes | SciBERT contrastive, 10 epochs. Use with `contrastive_scibert_chembl.pt` |
| `chebi20_27M_scibert_20ep_contrastive.pt` | 27M | ChEBI-20 | SciBERT contrastive | yes | **Best model** — SciBERT contrastive, 20 epochs, early-stop at epoch 18. Use with `contrastive_scibert_chembl.pt` |
| `molinst_27M_frozen.pt` | 27M | Mol-Instructions | frozen BGE | yes | Best MolInst model |
| `molinst_27M_frozen_no_eos.pt` | 27M | Mol-Instructions | frozen BGE | no | Legacy backup before EOS fix |
| `molinst_27M_frozen_epoch3.pt` | 27M | Mol-Instructions | frozen BGE | yes | Kaggle T4 epoch 3 checkpoint |

### Per-Epoch Checkpoints (SciBERT 20ep run)

| Filename | Notes |
|---|---|
| `chebi20_27M_scibert_20ep_contrastive_epoch_3.pt` | Epoch 3 |
| `chebi20_27M_scibert_20ep_contrastive_epoch_6.pt` | Epoch 6 |
| `chebi20_27M_scibert_20ep_contrastive_epoch_9.pt` | Epoch 9 |
| `chebi20_27M_scibert_20ep_contrastive_epoch_12.pt` | Epoch 12 |
| `chebi20_27M_scibert_20ep_contrastive_epoch_15.pt` | Epoch 15 |
| `chebi20_27M_scibert_20ep_contrastive_epoch_18.pt` | Epoch 18 (best val loss — copied to `chebi20_27M_scibert_20ep_contrastive.pt`) |

## Contrastive Encoder Checkpoints

These are text-molecule aligner checkpoints (not diffusion models). Load via `_load_contrastive_scorer()` in `finetune/train_chebi20.py` or `--encoder contrastive --contrastive_ckpt <path>`.

| Filename | Size | Trained on | Notes |
|---|---|---|---|
| `contrastive_scibert_chembl.pt` | ~439M | ChEMBL | SciBERT + molecule encoder, InfoNCE loss |
| `contrastive_scibert_chebi20_projonly.pt` | ~447M | ChEBI-20 | Projection layer fine-tuned on ChEBI-20; backbone frozen |

## RL Checkpoints (5M base model)

| Filename | Reward | Steps | Notes |
|---|---|---|---|
| `zinc_5M_rl_v1_step100.pt` | QED + diversity | 100 | REINFORCE v1 (deprecated) |
| `zinc_5M_rl_v1_step200.pt` | QED + diversity | 200 | REINFORCE v1 (deprecated) |
| `zinc_5M_rl_v2_step{100-500}.pt` | QED only | 100–500 | REINFORCE v2. Key finding: reward hacking via MW exploitation |
| `zinc_5M_rl_qed_mw_step{100-300}.pt` | QED + MW constraint | 100–300 | Corrected reward hacking, 32 denoising steps |
| `zinc_5M_rl_qed_mw_50s_step{100-300}.pt` | QED + MW constraint | 100–300 | Same reward, 50 denoising steps |
| `zinc_5M_rl_qed_mw_div_50s_step{100-300}.pt` | QED + MW + diversity | 100–300 | **Best RL run** — QED 0.84 vs base 0.74 |
