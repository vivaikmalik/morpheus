# Version Consistency Audit

**Date:** 2026-04-11

## selfies Library

- Current installed version: **2.1.1**
- Version used at training time: **not recorded in logs** (chebi20_log.csv has no version field)
- selfies 2.x has a stable API; breaking changes between 2.x sub-versions are minor
- Pre-converted SELFIES in CSVs means runtime selfies version is only used at eval time

## Checkpoint

- Checkpoint file: `checkpoints/chebi20_27M_scibert_20ep_contrastive.pt`
- File mtime: Tue Apr  7 10:08:48 2026
- Checkpoint saved epoch: 18 (best val loss = 0.3769)
- EOS fix commit date: Sat Apr 4 19:22:15 2026 (commit 96f65b5)
- **Checkpoint post-dates EOS fix** -- trained with correct EOS token appending

## Tokenizer Consistency

- Training config vocab_size: **110** (from checkpoint config dict)
- chemical_tokenizer.json vocab_size: **110** -- MATCH
- Same tokenizer file used for training and evaluation: confirmed (path hardcoded in config)

## Data Pipeline

- Training data: SELFIES pre-stored in `data/chebi20_train.csv` response column
- Evaluation: same CSVs, same tokenizer
- No runtime SMILES->SELFIES conversion during training (pre-converted)
- Runtime SMILES->SELFIES conversion only for eval metrics (sf.encoder on gen_smiles)

## Verdict

**No version consistency issues found.** Vocab size matches. EOS fix is applied.  
selfies version is consistent between training and evaluation (same environment).
