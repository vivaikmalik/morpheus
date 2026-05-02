"""
training/train.py — pretraining on ZINC250k with RoPE
"""

import sys
import torch
import random
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.datasets import SELFIESDataset
from training.diffusionCollator import DiffusionCollator
from training.losses import diffusion_loss
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Model — match dep-aware 27M backbone
    "vocab_size":   110,
    "hidden_size":  1024,
    "num_heads":    16,
    "ffn_dim":      4096,
    "num_layers":   12,
    "max_length":   74,
    "dropout":      0.1,

    # Training
    "batch_size":            128,
    "gradient_accumulation": 1,
    "learning_rate":         3e-4,
    "weight_decay":          0.01,
    "max_grad_norm":         1.1,
    "num_epochs":            11,
    "warmup_steps":          2222,

    # Loss weights
    "eos_weight":  5.0,
    "pad_weight":  0.05,
    "use_timestep_weighting": False,  # uniform weighting

    # Data
    "val_fraction": 0.05,
    "num_workers":  2,
    "seed":         42,

    # Logging
    "log_every":  50,
    "val_every":  1000,
    "save_every": 5000,

    # Generation samples during training
    "gen_steps":       50,
    "gen_temperature": 1.0,
    "gen_num_mols":    16,

    # Paths
    "project_root":   str(ROOT),
    "checkpoint_dir": str(ROOT / "checkpoints"),
    "wandb_project":  "morpheus-pretrain",
}


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, tokenizer):
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            logits = model(input_ids, timesteps)
            loss = diffusion_loss(logits, labels, tokenizer,
                                   CONFIG["eos_weight"], CONFIG["pad_weight"],
                                   timesteps, CONFIG["use_timestep_weighting"])
            total_loss += loss.item()
            total_steps += 1
    model.train()
    return total_loss / max(total_steps, 1)


# =============================================================================
# UNCONDITIONAL GENERATION
# =============================================================================

def generate_samples(model, tokenizer, device, step):
    model.eval()
    num_mols = CONFIG["gen_num_mols"]
    max_length = CONFIG["max_length"]
    num_steps = CONFIG["gen_steps"]
    temperature = CONFIG["gen_temperature"]

    input_ids = torch.full((num_mols, max_length), tokenizer.mask_token_id,
                           dtype=torch.long, device=device)
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

    valid_count = 0
    print(f"\n{'=' * 60}\n  GENERATED — Step {step}\n{'=' * 60}")
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        selfies_str = tokenizer.decode(ids)
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                if i < 6:
                    canon = Chem.MolToSmiles(mol)
                    print(f"  [{i}] ✅ tokens={len(ids)}  "
                          f"LogP={Descriptors.MolLogP(mol):+.2f}  "
                          f"{canon[:55]}")
        except Exception:
            pass

    valid_pct = 100 * valid_count / num_mols
    print(f"\n  Valid: {valid_count}/{num_mols} ({valid_pct:.0f}%)\n{'=' * 60}\n")
    wandb.log({"val/valid_pct": valid_pct}, step=step)
    model.train()


# =============================================================================
# TRAINING
# =============================================================================

def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))

    # Data
    print("Loading ZINC250k...")
    df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
    full_dataset = SELFIESDataset(df, tokenizer, max_length=CONFIG["max_length"])

    val_size = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]))
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    collator = DiffusionCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                               shuffle=True, collate_fn=collator,
                               num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, collate_fn=collator,
                             num_workers=CONFIG["num_workers"], pin_memory=True)

    # Model — NO text encoder during pretraining
    model = MolecularDiffusionModel(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["hidden_size"],
        num_heads=CONFIG["num_heads"], ffn_dim=CONFIG["ffn_dim"],
        num_layers=CONFIG["num_layers"], max_length=CONFIG["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=None,  # pretraining: no text
        dropout=CONFIG["dropout"],
    ).to(device)

    total, trainable, _ = model.count_parameters()
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                       weight_decay=CONFIG["weight_decay"])
    acc = CONFIG["gradient_accumulation"]
    total_steps = (len(train_loader) // acc) * CONFIG["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps)
    print(f"Total steps: {total_steps:,} | Warmup: {CONFIG['warmup_steps']:,}")

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    global_step, best_val = 0, float("inf")
    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, timesteps)
                loss = diffusion_loss(logits, labels, tokenizer,
                                       CONFIG["eos_weight"], CONFIG["pad_weight"],
                                       timesteps, CONFIG["use_timestep_weighting"])
                loss = loss / acc
            loss.backward()

            if (step_idx + 1) % acc == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["log_every"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step:>6} | loss: {loss.item() * acc:.4f} | "
                          f"grad: {grad_norm:.4f} | lr: {lr:.2e}")
                    wandb.log({"train/loss": loss.item() * acc,
                               "train/grad_norm": float(grad_norm),
                               "train/lr": lr}, step=global_step)

                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, val_loader, device, tokenizer)
                    print(f"  → Val loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss}, step=global_step)
                    generate_samples(model, tokenizer, device, global_step)

                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({
                            "step": global_step, "epoch": epoch + 1,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "val_loss": val_loss, "config": CONFIG,
                        }, checkpoint_dir / "best_model.pt")
                        print(f"  ✅ Best (val: {val_loss:.4f})")

                if global_step % CONFIG["save_every"] == 0:
                    torch.save({
                        "step": global_step, "epoch": epoch + 1,
                        "model": model.state_dict(), "config": CONFIG,
                    }, checkpoint_dir / f"checkpoint_step{global_step}.pt")
                    print(f"  💾 Saved step {global_step}")

    print(f"\nDone. Best val: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    train()