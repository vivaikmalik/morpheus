# MolGen — Masked Diffusion for Text-Conditional Molecule Generation

Masked diffusion language model for text-to-SELFIES generation, trained on ChEBI-20 and
Mol-Instructions. Supports frozen and contrastive text encoders, classifier-free guidance,
best-of-N contrastive reranking, and iterative refinement at inference.

## Model Pipeline

```
zinc_5M_pretrain.pt           (ZINC250K, unconditional)
    └─ zinc_17M_pretrain.pt   (ZINC250K, upscaled, unconditional)
           └─ chebi20_27M_*.pt    (ChEBI-20, text-conditional)
           └─ molinst_27M_*.pt    (Mol-Instructions, text-conditional)

contrastive_scibert_chembl.pt             (SciBERT text <-> molecule aligner, ChEMBL)
    └─ contrastive_scibert_chebi20_projonly.pt  (projection fine-tuned on ChEBI-20)
```

## Setup

Developed and tested with Python 3.10. Key dependencies:

```bash
pip install torch transformers huggingface_hub datasets \
            rdkit selfies pandas numpy scipy scikit-learn \
            matplotlib tqdm wandb
```

Reference versions used during development: rdkit 2022.09.5, selfies 2.1.1, datasets 4.8.4, pandas 2.3.3, torch (any recent 2.x with MPS or CUDA backend). Apple M2 with MPS was the primary development device; CUDA T4 (Kaggle) was used for upscaled pretraining.

## Quick Start

```bash
# Evaluate best ChEBI-20 model (frozen BGE encoder)
python3 finetune/train_chebi20.py --eval_only --encoder frozen \
    --cfg 1.5 --temp 0.6 --steps 50

# Evaluate with contrastive SciBERT encoder (20-epoch fine-tune)
python3 finetune/train_chebi20.py --eval_only \
    --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_20ep \
    --cfg 1.5 --temp 0.6 --steps 50

# Best-of-N reranking (N=10)
python3 finetune/train_chebi20.py --eval_only --encoder contrastive \
    --contrastive_ckpt checkpoints/contrastive_scibert_chembl.pt \
    --model_size 27M_scibert_20ep --cfg 1.5 --temp 0.6 --steps 50 \
    --rerank_n 10

# Fine-tune contrastive encoder on ChEBI-20
python3 contrastive/train_contrastive.py \
    --mol_ckpt checkpoints/zinc_5M_pretrain.pt \
    --input_csv data/chebi20_train.csv \
    --val_fraction 0.1
```

## Data

| File | Rows | Description |
|---|---|---|
| `data/chebi20_train.csv` | 20,158 | ChEBI-20 SMILES + description |
| `data/chebi20_val.csv` | 2,504 | Validation split |
| `data/chebi20_test.csv` | 2,542 | Test split (held out) |
| `chemical_tokenizer.json` | — | SELFIES vocabulary (110 tokens) |

## Repository layout

```
model/          Core diffusion model (MolecularDiffusionModel, embeddings, transformer)
tokenizer/      SELFIES chemical tokenizer
training/       Pretraining (ZINC) and RL fine-tuning scripts
finetune/       Text-conditional fine-tuning (ChEBI-20, Mol-Instructions)
contrastive/    Contrastive text-molecule aligner (training + inference)
inference/      Standalone evaluation utilities
scripts/        Analysis tools (summary table, atom BLEU, Kaggle notebooks)
data/           CSV datasets
checkpoints/    Model weights (see checkpoints/README.md)
outputs/        Evaluation CSVs and plots
archive/        Investigation reports, audits, and superseded code (see archive/README.md and archive/paper_claim_index.md)
dataExtractor/  Legacy dataset class kept for backward compatibility with earlier scripts
logs/           Training run console logs
wandb/          Weights & Biases run records
```

## Results (ChEBI-20 test set, 2,542 molecules)

Best configuration per encoder variant (cfg=1.5, temp=0.6, 50 steps unless noted).

| Model | Encoder | BLEU-2 | Atom BLEU-2 | Morgan | Lev Sim |
|---|---|---|---|---|---|
| Morpheus 27M | BGE-large frozen (cfg=2.0) | 0.443 | 0.424 | 0.199 | 0.385 |
| Morpheus 27M | BGE contrastive (cfg=1.0) | 0.521 | 0.499 | 0.230 | 0.432 |
| Morpheus 27M | SciBERT contrastive (10ep) | 0.576 | 0.554 | 0.251 | 0.483 |
| Morpheus 27M | SciBERT contrastive (20ep) | 0.623 | 0.604 | 0.299 | 0.522 |
| Morpheus 27M | SciBERT 20ep + rerank@3 | 0.629 | 0.610 | 0.302 | 0.528 |
| Morpheus 27M | SciBERT 20ep + rerank@10 | 0.642 | 0.621 | 0.310 | 0.538 |
| Morpheus 27M | SciBERT contrastive (40ep, t=0.7) | **0.635** | **0.614** | **0.318** | **0.531** |
| TGM-DLM (AAAI 2024)* | SciBERT | 0.826 | — | 0.688 | — |

*TGM-DLM: 180M params, ~100K+ training steps on A100. Morpheus: 27M trainable params, T4 GPU.
