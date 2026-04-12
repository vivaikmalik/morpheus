"""
training/train.py — SMILES-based pretraining on ZINC250k
"""

import sys
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.smilesTokenizer import SmilesTokenizer
from dataExtractor.smilesDataset import SmilesDataset
from training.diffusionCollator import ConditionalDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel

CONFIG = {
    # Model — matches TGM-DLM architecture
    "vocab_size"  : None,       # set from tokenizer
    "hidden_size" : 1024,
    "num_heads"   : 16,
    "ffn_dim"     : 4096,
    "num_layers"  : 12,
    "max_length"  : 256,        # CHANGED: 74 → 256 to match TGM-DLM
    "dropout"     : 0.1,

    "text_model_name": None,

    # Training
    "batch_size"           : 64,    # reduced from 128 — longer sequences
    "gradient_accumulation": 2,
    "learning_rate"        : 5e-5,
    "weight_decay"         : 0.01,
    "max_grad_norm"        : 1.0,
    "num_epochs"           : 30,
    "warmup_steps"         : 3000,

    # Data
    "val_fraction" : 0.05,
    "num_workers"  : 0,
    "seed"         : 42,

    # Logging
    "log_every"  : 50,
    "val_every"  : 1000,
    "save_every" : 5000,

    # Generation
    "gen_steps"       : 64,
    "gen_temperature" : 1.2,
    "gen_num_mols"    : 6,

    # Paths
    "project_root"  : str(Path(__file__).parent.parent),
    "checkpoint_dir": str(Path(__file__).parent.parent / "checkpoints"),
    "wandb_project" : "morpheus-diffusion",
}


def diffusion_loss(logits, labels, timesteps, pad_token_id=0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device
    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id] = 0.1
    raw_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), labels.view(-1),
        weight=vocab_weights, ignore_index=-100, reduction='none')
    weights = (1.0 - timesteps)
    weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)
    valid = (labels.view(-1) != -100).float()
    return (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)


def run_validation(model, val_loader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            logits = model(input_ids, timesteps)
            loss = diffusion_loss(logits, labels, timesteps)
            total_loss += loss.item()
            total_steps += 1
    model.train()
    return total_loss / total_steps


def generate_samples(model, tokenizer, device, step):
    model.eval()
    num_mols = CONFIG["gen_num_mols"]
    max_length = CONFIG["max_length"]
    num_steps = CONFIG["gen_steps"]
    temperature = CONFIG["gen_temperature"]

    input_ids = torch.full((num_mols, max_length), tokenizer.mask_token_id, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(num_mols).unsqueeze(-1)
            logits = model(input_ids, step_t)
            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float('-inf')
            probs = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)
            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mi = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mi, tokenizer.mask_token_id)
            input_ids = sampled

    print(f"\n{'='*60}\n  GENERATED MOLECULES — Step {step}\n{'='*60}")
    valid_count = 0
    results = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        smiles = tokenizer.decode(ids)
        token_count = len(ids)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            logp = round(Descriptors.MolLogP(mol), 3)
            heavy = mol.GetNumHeavyAtoms()
            valid_count += 1
            print(f"  [{i}] ✅ {token_count} tok | LogP: {logp:+.2f} | "
                  f"Heavy: {heavy:>2} | {canonical[:55]}")
            results.append({"valid": True, "smiles": canonical, "logp": logp,
                          "heavy": heavy, "tokens": token_count})
        else:
            print(f"  [{i}] ❌ ({token_count} tok): {smiles[:50]}")
            results.append({"valid": False, "smiles": smiles[:50], "tokens": token_count})

    print(f"\n  Valid: {valid_count}/{num_mols} ({100*valid_count/num_mols:.0f}%)")
    print(f"{'='*60}\n")

    wandb.log({"val/valid_pct": 100 * valid_count / num_mols}, step=step)
    model.train()


def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    project_root = Path(CONFIG["project_root"])
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    # --- SMILES tokenizer ---
    vocab_path = project_root / "smiles_vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: {vocab_path} not found. Run: python scripts/build_smiles_vocab.py")
        sys.exit(1)
    tokenizer = SmilesTokenizer(str(vocab_path))
    CONFIG["vocab_size"] = tokenizer.vocab_size
    print(f"SMILES tokenizer: {tokenizer.vocab_size} tokens")

    # --- Load ZINC250k ---
    print("Loading ZINC250k...")
    df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
    print(f"  Raw: {len(df):,} molecules")

    # Filter: must have valid SMILES and fit in max_length
    valid_mask = []
    for smi in df["smiles"]:
        try:
            toks = tokenizer.encode(str(smi))
            valid_mask.append(len(toks) + 1 <= CONFIG["max_length"])
        except:
            valid_mask.append(False)
    df = df[valid_mask].reset_index(drop=True)
    print(f"  After filter: {len(df):,} molecules")

    full_dataset = SmilesDataset(df, tokenizer, max_length=CONFIG["max_length"])

    val_size = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]))
    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    collator = ConditionalDiffusionCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, collate_fn=collator, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=collator, num_workers=CONFIG["num_workers"], pin_memory=True)

    # --- Model ---
    model = MolecularDiffusionModel(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["hidden_size"],
        num_heads=CONFIG["num_heads"], ffn_dim=CONFIG["ffn_dim"],
        num_layers=CONFIG["num_layers"], max_length=CONFIG["max_length"],
        pad_token_id=tokenizer.pad_token_id, text_model_name=None,
        dropout=CONFIG["dropout"]).to(device)

    total, trainable, frozen = model.count_parameters()
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                      weight_decay=CONFIG["weight_decay"])
    acc_steps = CONFIG["gradient_accumulation"]
    total_steps = (len(train_loader) // acc_steps) * CONFIG["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps)

    print(f"Total steps: {total_steps:,} | Warmup: {CONFIG['warmup_steps']:,}")
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    global_step, best_val_loss = 0, float('inf')
    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, timesteps)
                loss = diffusion_loss(logits, labels, timesteps) / acc_steps
            loss.backward()

            if (step_idx + 1) % acc_steps == 0 or (step_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["log_every"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step:>6} | loss: {loss.item()*acc_steps:.4f} | "
                          f"grad: {grad_norm:.4f} | lr: {lr:.2e}")
                    wandb.log({"train/loss": loss.item()*acc_steps,
                              "train/grad_norm": float(grad_norm),
                              "train/lr": lr}, step=global_step)

                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, val_loader, device)
                    print(f"  → Val loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss}, step=global_step)
                    generate_samples(model, tokenizer, device, global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({"step": global_step, "epoch": epoch+1,
                                    "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "val_loss": val_loss, "config": CONFIG},
                                   checkpoint_dir / "best_model.pt")
                        print(f"  ✅ New best (val_loss: {val_loss:.4f})")

                if global_step % CONFIG["save_every"] == 0:
                    torch.save({"step": global_step, "epoch": epoch+1,
                                "model": model.state_dict(), "config": CONFIG},
                               checkpoint_dir / f"checkpoint_step{global_step}.pt")
                    print(f"  💾 Saved step {global_step}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    wandb.finish()

if __name__ == "__main__":
    train()