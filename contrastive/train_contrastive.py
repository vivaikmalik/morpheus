"""
contrastive/train_contrastive.py
---------------------------------
Standalone contrastive fine-tuning of a (text, molecule) aligner on any
text-SELFIES CSV.  Designed to run locally or on Kaggle unchanged.

Imports from contrastive/model.py, contrastive/dataset.py, contrastive/utils.py
— no inline class copies.

Checkpoint schema (compatible with _load_contrastive_scorer and
_load_contrastive_text_encoder in finetune/train_chebi20.py):
  {
      'epoch':            int,
      'best_epoch':       int,
      'best_val_loss':    float,
      'model_state':      model.state_dict(),
      'molecular_config': dict,   # from zinc backbone checkpoint
      'config':           dict,   # all training hyper-params + text_model name
  }

Usage examples:
  # Train from scratch on ChEBI-20 with SciBERT:
  python3 contrastive/train_contrastive.py

  # Resume from existing contrastive checkpoint:
  python3 contrastive/train_contrastive.py \\
      --resume_ckpt checkpoints/contrastive_scibert_chembl.pt \\
      --output_ckpt checkpoints/contrastive_scibert_chebi20.pt

  # Unfreeze molecule encoder:
  python3 contrastive/train_contrastive.py --no_freeze_molecule
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Project imports ────────────────────────────────────────────────────────────
# Support running from repo root OR from inside contrastive/
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contrastive.model   import ContrastiveAligner, MoleculeEncoder, \
                                  contrastive_loss, retrieval_metrics
from contrastive.dataset import PairDataset, read_pairs_from_csv, train_val_split
from tokenizer.chemicalTokenizer import ChemicalTokenizer


# =============================================================================
# INLINE UTILITIES
# =============================================================================

def load_molecule_backbone(mol_ckpt_path: Path, tokenizer_path: Path, device):
    """Load zinc_17M_pretrain.pt and return (chem_tokenizer, MoleculeEncoder, mol_config).

    The contrastive backbone uses self-attention-only blocks (no cross-attention),
    matching the PretrainDiffusionModel architecture.  RoPE is used for positional
    encoding (buffer-only, no trainable params).
    """
    from model.embeddings       import TimestepEmbedding, RotaryEmbedding
    from model.transformerBlock import SelfAttention, FeedForward

    chem_tokenizer = ChemicalTokenizer(str(tokenizer_path))
    ckpt = torch.load(mol_ckpt_path, map_location="cpu", weights_only=False)
    mc   = ckpt["config"]

    class _SABlock(nn.Module):
        def __init__(self, h, n, f):
            super().__init__()
            self.norm1     = nn.LayerNorm(h)
            self.attention = SelfAttention(h, n, dropout=0.0)
            self.norm2     = nn.LayerNorm(h)
            self.ffn       = FeedForward(h, f, dropout=0.0)

        def forward(self, x, padding_mask=None, rope=None):
            x = x + self.attention(self.norm1(x), padding_mask, rope=rope)
            x = x + self.ffn(self.norm2(x))
            return x

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            h, n, f = mc["hidden_size"], mc["num_heads"], mc["ffn_dim"]
            v, ml   = mc["vocab_size"], mc["max_length"]
            self.token_embedding    = nn.Embedding(v, h)
            self.timestep_embedding = TimestepEmbedding(h)
            self.input_norm         = nn.LayerNorm(h)
            head_dim = h // n
            self.rope = RotaryEmbedding(
                head_dim=head_dim,
                max_length=max(ml, 4096),
            )
            self.blocks = nn.ModuleList(
                [_SABlock(h, n, f) for _ in range(mc["num_layers"])]
            )
            self.output_norm = nn.LayerNorm(h)
            self.lm_head     = nn.Linear(h, v, bias=False)

    backbone = _Backbone()
    missing, unexpected = backbone.load_state_dict(ckpt["model"], strict=False)
    non_trivial = [k for k in missing    if "lm_head" not in k] + \
                  [k for k in unexpected if "lm_head" not in k]
    if non_trivial:
        print(f"[backbone] WARNING — unexpected/missing keys: {non_trivial}")

    mol_encoder = MoleculeEncoder(backbone).to(device)
    return chem_tokenizer, mol_encoder, mc


def encode_selfies_batch(selfies_list, chem_tokenizer: ChemicalTokenizer,
                          max_length: int, device) -> torch.Tensor:
    """Encode a list of SELFIES strings to padded token-ID tensors [B, max_length]."""
    batch_ids = []
    for s in selfies_list:
        ids = chem_tokenizer.encode(s)
        ids = ids[:max_length - 1]              # truncate, leave room for EOS
        ids.append(chem_tokenizer.eos_token_id)
        ids += [chem_tokenizer.pad_token_id] * (max_length - len(ids))
        batch_ids.append(ids)
    return torch.tensor(batch_ids, dtype=torch.long, device=device)


# =============================================================================
# COLLATE
# =============================================================================

def make_collate_fn(text_tokenizer, max_text_length: int, device):
    def collate(batch):
        texts   = [x[0] for x in batch]
        selfies = [x[1] for x in batch]
        enc = text_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_text_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, selfies
    return collate


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── Molecular backbone ────────────────────────────────────────────────────
    chem_tokenizer, mol_encoder, mol_config = load_molecule_backbone(
        Path(args.mol_ckpt), Path(args.tokenizer_path), device
    )
    print(f"[mol]   backbone loaded  hidden={mol_config['hidden_size']}  "
          f"layers={mol_config['num_layers']}  vocab={mol_config['vocab_size']}  "
          f"max_len={mol_config['max_length']}")

    # ── Text encoder ──────────────────────────────────────────────────────────
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model     = AutoModel.from_pretrained(args.text_model).to(device)
    text_hidden    = text_model.config.hidden_size
    print(f"[text]  {args.text_model}  hidden={text_hidden}")

    # ── ContrastiveAligner ────────────────────────────────────────────────────
    model = ContrastiveAligner(
        text_model       = text_model,
        molecule_encoder = mol_encoder,
        text_hidden      = text_hidden,
        mol_hidden       = mol_config["hidden_size"],
        proj_dim         = args.proj_dim,
        project_molecule = True,   # mol_proj: mol_hidden → proj_dim
    ).to(device)

    # ── Resume from existing contrastive checkpoint ───────────────────────────
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_ckpt not found: {resume_path}")
        print(f"[resume] loading {resume_path.name} ...")
        prev = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state"], strict=True)
        print(f"         epoch={prev.get('best_epoch', prev.get('epoch', '?'))}  "
              f"best_val_loss={prev.get('best_val_loss', float('nan')):.4f}")

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────
    if args.freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad_(False)
        print("[freeze] text encoder frozen")
    if args.freeze_molecule:
        for p in model.molecule_encoder.parameters():
            p.requires_grad_(False)
        print("[freeze] molecule encoder frozen")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[params] {n_total:,} total  |  {n_trainable:,} trainable")

    # ── Data ──────────────────────────────────────────────────────────────────
    pairs = read_pairs_from_csv(args.input_csv, args.text_column, args.selfies_column)
    print(f"[data]  {len(pairs):,} valid pairs from {Path(args.input_csv).name}")
    train_pairs, val_pairs = train_val_split(
        pairs, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"[data]  train={len(train_pairs):,}  val={len(val_pairs):,}")

    collate = make_collate_fn(text_tokenizer, args.max_text_length, device)
    train_loader = DataLoader(
        PairDataset(train_pairs), batch_size=args.batch_size,
        shuffle=True, drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs), batch_size=args.batch_size * 2,
        shuffle=False, drop_last=False, collate_fn=collate,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Config saved in checkpoint (all Path → str for portability) ───────────
    save_config = {
        "text_model"      : args.text_model,
        "input_csv"       : str(args.input_csv),
        "text_column"     : args.text_column,
        "selfies_column"  : args.selfies_column,
        "mol_ckpt"        : str(args.mol_ckpt),
        "proj_dim"        : args.proj_dim,
        "project_molecule": True,
        "freeze_text"     : args.freeze_text,
        "freeze_molecule" : args.freeze_molecule,
        "epochs"          : args.epochs,
        "batch_size"      : args.batch_size,
        "lr"              : args.lr,
        "weight_decay"    : args.weight_decay,
        "temperature"     : args.temperature,
        "val_fraction"    : args.val_fraction,
        "max_text_length" : args.max_text_length,
        "seed"            : args.seed,
        "resume_ckpt"     : str(args.resume_ckpt) if args.resume_ckpt else None,
    }

    output_ckpt = Path(args.output_ckpt)
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_steps = 0.0, 0
        n_batches = len(train_loader)

        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=True,
                   unit="it", dynamic_ncols=True)
        for batch_idx, (text_inputs, selfies_list) in enumerate(bar):
            t_batch = time.time()
            mol_ids = encode_selfies_batch(
                selfies_list, chem_tokenizer, mol_config["max_length"], device
            )
            z_text, z_mol = model(text_inputs, mol_ids, chem_tokenizer.pad_token_id)
            loss = contrastive_loss(z_text, z_mol, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_steps += 1

            bar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx == 0:
                elapsed_b1 = time.time() - t_batch
                print(f"  [batch 1/{n_batches}] loss={loss.item():.4f} ({elapsed_b1:.1f}s)")

        train_loss = tr_loss / max(tr_steps, 1)

        # Validation
        model.eval()
        val_loss, val_steps = 0.0, 0
        z_texts, z_mols = [], []
        with torch.no_grad():
            for text_inputs, selfies_list in val_loader:
                mol_ids = encode_selfies_batch(
                    selfies_list, chem_tokenizer, mol_config["max_length"], device
                )
                z_text, z_mol = model(text_inputs, mol_ids, chem_tokenizer.pad_token_id)
                val_loss  += contrastive_loss(z_text, z_mol, args.temperature).item()
                val_steps += 1
                z_texts.append(z_text)
                z_mols.append(z_mol)

        val_loss /= max(val_steps, 1)
        metrics   = retrieval_metrics(
            torch.cat(z_texts), torch.cat(z_mols), ks=(1, 5, 10)
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"R@1={metrics['recall@1']:.4f}  R@5={metrics['recall@5']:.4f}  "
            f"R@10={metrics['recall@10']:.4f}  MRR={metrics['mrr']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch"           : epoch,
                "best_epoch"      : epoch,
                "best_val_loss"   : best_val_loss,
                "model_state"     : model.state_dict(),
                "molecular_config": mol_config,
                "config"          : save_config,
            }, output_ckpt)
            print(f"  ✓ saved best → {output_ckpt.name}  (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("  early stopping.")
                break

    print(f"\nDone. best_epoch={best_epoch}  best_val_loss={best_val_loss:.4f}")
    print(f"Checkpoint: {output_ckpt}")


# =============================================================================
# ARGPARSE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contrastive (text, molecule) aligner fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--input_csv",       default="data/chebi20_train.csv",
                        help="Training CSV path")
    parser.add_argument("--text_column",     default="prompt",
                        help="Column name for text descriptions")
    parser.add_argument("--selfies_column",  default="response",
                        help="Column name for SELFIES strings")

    # Checkpoints
    parser.add_argument("--mol_ckpt",        default="checkpoints/zinc_17M_pretrain.pt",
                        help="Molecular backbone checkpoint (zinc_17M_pretrain.pt)")
    parser.add_argument("--tokenizer_path",  default="chemical_tokenizer.json",
                        help="Path to chemical_tokenizer.json")
    parser.add_argument("--resume_ckpt",     default=None,
                        help="Existing contrastive checkpoint to resume from (optional)")
    parser.add_argument("--output_ckpt",     default="checkpoints/contrastive_scibert_chebi20.pt",
                        help="Output checkpoint path")

    # Model
    parser.add_argument("--text_model",      default="allenai/scibert_scivocab_uncased",
                        help="HuggingFace text encoder model name")
    parser.add_argument("--proj_dim",        type=int,   default=768,
                        help="Projection dimension for both text and mol embeddings")
    parser.add_argument("--max_text_length", type=int,   default=256,
                        help="Max token length for text tokenizer")

    # Freeze flags — molecule encoder frozen by default, text encoder unfrozen
    parser.add_argument("--freeze_text",       action="store_true",
                        help="Freeze text encoder weights")
    parser.add_argument("--freeze_molecule",   action="store_true", default=True,
                        help="Freeze molecule encoder weights (default: True)")
    parser.add_argument("--no_freeze_molecule", dest="freeze_molecule",
                        action="store_false",
                        help="Unfreeze molecule encoder weights")

    # Training hyper-parameters
    parser.add_argument("--epochs",          type=int,   default=25)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--temperature",     type=float, default=0.07,
                        help="InfoNCE temperature")
    parser.add_argument("--val_fraction",    type=float, default=0.1)
    parser.add_argument("--patience",        type=int,   default=3,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seed",            type=int,   default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
