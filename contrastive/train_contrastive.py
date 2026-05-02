"""
contrastive/train_contrastive.py
---------------------------------
Train a CLIP-style aligner on (text, SELFIES) pairs.

Two encoders:
  - Text encoder (e.g. BGE-large-en-v1.5)
  - Molecule encoder (the pretrained ZINC backbone, frozen)

Both project to a shared embedding space. InfoNCE contrastive loss pulls
matching (text, mol) pairs together and pushes non-matching apart.

Output: a checkpoint with text_encoder weights tuned to "speak chemistry."
This encoder gets loaded into the diffusion model via set_text_encoder().

Usage:
    python contrastive/train_contrastive.py \\
        --mol_ckpt checkpoints/best_model.pt \\
        --text_model BAAI/bge-large-en-v1.5 \\
        --input_csv data/chebi20_train.csv \\
        --output_ckpt checkpoints/contrastive_bge_chebi20.pt
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# DATASET
# =============================================================================

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def read_pairs(csv_path, text_col="description", selfies_col="selfies"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    pairs = []
    for _, row in df.iterrows():
        t = str(row[text_col]) if text_col in row else ""
        s = str(row[selfies_col]) if selfies_col in row else ""
        if t.strip() and s.strip():
            pairs.append((t, s))
    return pairs


# =============================================================================
# MEAN POOL
# =============================================================================

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


# =============================================================================
# MOLECULE ENCODER (uses pretrained backbone)
# =============================================================================

class MoleculeEncoder(nn.Module):
    """Wraps a pretrained MolecularDiffusionModel for molecule embedding."""

    def __init__(self, mol_model):
        super().__init__()
        self.mol_model = mol_model

    def forward(self, input_ids, pad_token_id):
        device = input_ids.device
        B, seq_len = input_ids.shape
        padding_mask = (input_ids == pad_token_id)
        attention_mask = (~padding_mask).long()

        x = self.mol_model.token_embedding(input_ids)
        t = torch.zeros((B, 1), device=device)
        x = x + self.mol_model.timestep_embedding(t).unsqueeze(1)
        x = self.mol_model.input_norm(x)

        for block in self.mol_model.blocks:
            # No text — pretraining-style forward
            x = block(x, None, padding_mask, None)

        x = self.mol_model.output_norm(x)
        return mean_pool(x, attention_mask)


# =============================================================================
# CONTRASTIVE ALIGNER
# =============================================================================

class ContrastiveAligner(nn.Module):
    def __init__(self, text_model, molecule_encoder,
                 text_hidden, mol_hidden, proj_dim):
        super().__init__()
        self.text_model = text_model
        self.molecule_encoder = molecule_encoder
        self.text_proj = nn.Linear(text_hidden, proj_dim)
        self.mol_proj = nn.Linear(mol_hidden, proj_dim)
        self.proj_dim = proj_dim

    def encode_text(self, text_inputs):
        out = self.text_model(**text_inputs)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = mean_pool(out.last_hidden_state, text_inputs['attention_mask'])
        return F.normalize(self.text_proj(pooled), dim=-1)

    def encode_molecule(self, mol_ids, pad_id):
        pooled = self.molecule_encoder(mol_ids, pad_id)
        return F.normalize(self.mol_proj(pooled), dim=-1)

    def forward(self, text_inputs, mol_ids, pad_id):
        return self.encode_text(text_inputs), self.encode_molecule(mol_ids, pad_id)


def contrastive_loss(z_text, z_mol, temperature=0.07):
    logits = (z_text @ z_mol.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieval_metrics(z_text, z_mol, ks=(1, 5, 10)):
    sims = z_text @ z_mol.T
    sorted_idx = torch.argsort(sims, dim=1, descending=True)
    targets = torch.arange(sims.shape[0], device=sims.device)
    metrics = {}
    for k in ks:
        topk = sorted_idx[:, :k]
        metrics[f'recall@{k}'] = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return metrics


# =============================================================================
# LOAD BACKBONE
# =============================================================================

def load_molecule_backbone(mol_ckpt_path, tokenizer_path, device):
    """Load pretrained backbone and wrap in MoleculeEncoder."""
    tokenizer = ChemicalTokenizer(str(tokenizer_path))
    ckpt = torch.load(mol_ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Build model WITHOUT text encoder
    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
        num_heads=config["num_heads"], ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"], max_length=config["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=None,
        dropout=0.0,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        # Filter cross-attn keys (expected to be missing in pretraining checkpoint)
        unexpected = [k for k in unexpected if "cross" not in k and "text_" not in k
                      and "null_token" not in k]
        if unexpected:
            print(f"  WARNING — unexpected: {unexpected[:3]}")

    encoder = MoleculeEncoder(model).to(device)
    return tokenizer, encoder, config


# =============================================================================
# TRAIN
# =============================================================================

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Backbone
    tokenizer, mol_encoder, mol_config = load_molecule_backbone(
        Path(args.mol_ckpt), Path(args.tokenizer_path), device)
    print(f"Backbone loaded: hidden={mol_config['hidden_size']}, "
          f"layers={mol_config['num_layers']}, max_len={mol_config['max_length']}")

    # Text encoder
    text_tok = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModel.from_pretrained(args.text_model).to(device)
    text_hidden = text_model.config.hidden_size
    print(f"Text: {args.text_model}, hidden={text_hidden}")

    # Aligner
    aligner = ContrastiveAligner(
        text_model=text_model, molecule_encoder=mol_encoder,
        text_hidden=text_hidden, mol_hidden=mol_config["hidden_size"],
        proj_dim=args.proj_dim).to(device)

    # Freeze molecule encoder by default
    if args.freeze_molecule:
        for p in aligner.molecule_encoder.parameters():
            p.requires_grad = False
        print("Molecule encoder frozen")

    n_train = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
    print(f"Trainable: {n_train:,}")

    # Data
    pairs = read_pairs(args.input_csv, args.text_column, args.selfies_column)
    print(f"Pairs: {len(pairs):,}")
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    split = int(len(pairs) * (1 - args.val_fraction))
    train_pairs = [pairs[i] for i in idx[:split]]
    val_pairs = [pairs[i] for i in idx[split:]]

    def collate(batch):
        texts = [x[0] for x in batch]
        selfies = [x[1] for x in batch]
        enc = text_tok(texts, padding=True, truncation=True,
                        max_length=args.max_text_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, selfies

    train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size,
                                shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in aligner.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(exist_ok=True)

    best_val = float("inf")
    save_config = {
        "text_model": args.text_model,
        "proj_dim": args.proj_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
    }

    for epoch in range(1, args.epochs + 1):
        aligner.train()
        tr_loss, n = 0.0, 0
        for text_inputs, selfies_list in tqdm(train_loader, desc=f"E{epoch}"):
            # Encode SELFIES → token IDs
            mol_ids = []
            for s in selfies_list:
                ids = tokenizer.encode(s)[:mol_config["max_length"] - 1]
                ids.append(tokenizer.eos_token_id)
                ids += [tokenizer.pad_token_id] * (mol_config["max_length"] - len(ids))
                mol_ids.append(ids)
            mol_ids = torch.tensor(mol_ids, dtype=torch.long, device=device)

            z_text, z_mol = aligner(text_inputs, mol_ids, tokenizer.pad_token_id)
            loss = contrastive_loss(z_text, z_mol, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            n += 1

        # Validation
        aligner.eval()
        val_loss, vn = 0.0, 0
        z_texts, z_mols = [], []
        with torch.no_grad():
            for text_inputs, selfies_list in val_loader:
                mol_ids = []
                for s in selfies_list:
                    ids = tokenizer.encode(s)[:mol_config["max_length"] - 1]
                    ids.append(tokenizer.eos_token_id)
                    ids += [tokenizer.pad_token_id] * (mol_config["max_length"] - len(ids))
                    mol_ids.append(ids)
                mol_ids = torch.tensor(mol_ids, dtype=torch.long, device=device)
                zt, zm = aligner(text_inputs, mol_ids, tokenizer.pad_token_id)
                val_loss += contrastive_loss(zt, zm, args.temperature).item()
                vn += 1
                z_texts.append(zt)
                z_mols.append(zm)

        val_loss /= max(vn, 1)
        m = retrieval_metrics(torch.cat(z_texts), torch.cat(z_mols))
        print(f"Epoch {epoch}: train={tr_loss / n:.4f}  val={val_loss:.4f}  "
              f"R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "best_epoch": epoch,
                "best_val_loss": best_val,
                "model_state": aligner.state_dict(),
                "molecular_config": mol_config,
                "config": save_config,
            }, output_path)
            print(f"  ✓ saved best → {output_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/chebi20_train.csv")
    parser.add_argument("--text_column", default="description")
    parser.add_argument("--selfies_column", default="selfies")
    parser.add_argument("--mol_ckpt", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer_path", default="chemical_tokenizer.json")
    parser.add_argument("--output_ckpt", default="checkpoints/contrastive_aligner.pt")
    parser.add_argument("--text_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--proj_dim", type=int, default=1024)  # Match new 1024 backbone dim
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--freeze_molecule", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()