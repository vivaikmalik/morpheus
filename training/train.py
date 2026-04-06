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

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.conditionalSELFIESDataset import ConditionalSELFIESDataset
from training.diffusionCollator import ConditionalDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # Model
    "vocab_size"  : 110,
    "hidden_size" : 1024,
    "num_heads"   : 16,
    "ffn_dim"     : 4096,
    "num_layers"  : 12,
    "max_length"  : 74,
    "dropout"     : 0.1,

    # No text encoder during pretraining
    "text_model_name": None,

    # Training
    "batch_size"           : 128,
    "gradient_accumulation": 2,
    "learning_rate"        : 5e-5,
    "weight_decay"         : 0.01,
    "max_grad_norm"        : 1.0,
    "num_epochs"           : 55,
    "warmup_steps"         : 7700,

    # Data
    "val_fraction" : 0.05,
    "num_workers"  : 0,
    "seed"         : 42,

    # Logging
    "log_every"  : 50,
    "val_every"  : 1000,
    "save_every" : 5000,

    # Generation
    "gen_steps"       : 32,
    "gen_temperature" : 1.2,
    "gen_num_mols"    : 6,

    # Paths
    "project_root"  : str(Path(__file__).parent.parent),
    "checkpoint_dir": str(Path(__file__).parent.parent / "checkpoints"),
    "wandb_project" : "morpheus-diffusion",
}

# =============================================================================
# LOSS FUNCTION
# =============================================================================

def diffusion_loss(logits, labels, timesteps, pad_token_id=0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id] = 0.1

    raw_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction='none'
    )

    weights = (1.0 - timesteps)
    weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)

    valid = (labels.view(-1) != -100).float()

    loss = (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)
    return loss

# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device):
    model.eval()
    total_loss  = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            logits = model(input_ids, timesteps)
            loss   = diffusion_loss(logits, labels, timesteps)

            total_loss  += loss.item()
            total_steps += 1

    model.train()
    return total_loss / total_steps

# =============================================================================
# GENERATION
# =============================================================================

def generate_samples(model, tokenizer, device, step):
    model.eval()

    num_mols    = CONFIG["gen_num_mols"]
    max_length  = CONFIG["max_length"]
    num_steps   = CONFIG["gen_steps"]
    temperature = CONFIG["gen_temperature"]

    input_ids = torch.full(
        (num_mols, max_length),
        tokenizer.mask_token_id,
        device=device
    )

    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(num_mols).unsqueeze(-1)
            logits = model(input_ids, step_t)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float('-inf')

            scaled_logits = logits / temperature
            probs         = torch.softmax(scaled_logits, dim=-1)
            sampled       = torch.distributions.Categorical(probs=probs).sample()
            confidence    = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(
                    confidence, num_to_mask, dim=-1, largest=False
                )
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    # --- Decode ---
    results = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]

        selfies_str = tokenizer.decode(ids)
        token_count = len(ids)

        row = {
            "idx": i, "selfies": selfies_str, "token_count": token_count,
            "valid": False, "smiles": "", "logp": None, "heavy": None, "rings": None,
        }

        try:
            smiles = sf.decoder(selfies_str)
            mol    = Chem.MolFromSmiles(smiles)
            if mol is not None:
                row.update({
                    "valid" : True,
                    "smiles": Chem.MolToSmiles(mol),
                    "logp"  : round(Descriptors.MolLogP(mol), 3),
                    "heavy" : mol.GetNumHeavyAtoms(),
                    "rings" : mol.GetRingInfo().NumRings(),
                })
            else:
                row["smiles"] = smiles
        except Exception as e:
            row["smiles"] = f"ERROR: {e}"

        results.append(row)

    # --- Print ---
    print(f"\n{'='*60}")
    print(f"  GENERATED MOLECULES — Step {step}")
    print(f"{'='*60}")

    valid_count = sum(r["valid"] for r in results)
    for r in results:
        if r["valid"]:
            print(f"  [{r['idx']}] ✅ {r['token_count']} tokens | "
                  f"LogP: {r['logp']:+.2f} | Heavy: {r['heavy']:>2} | "
                  f"Rings: {r['rings']} | {r['smiles'][:55]}")
        else:
            print(f"  [{r['idx']}] ❌ Invalid ({r['token_count']} tokens): "
                  f"{r['selfies'][:50]}")

    print(f"\n  Valid: {valid_count}/{num_mols} "
          f"({100*valid_count/num_mols:.0f}%)")
    print(f"{'='*60}\n")

    columns = ["step", "valid", "tokens", "logp", "heavy_atoms", "rings", "smiles", "selfies"]
    table_data = [[
        step, r["valid"], r["token_count"], r["logp"],
        r["heavy"], r["rings"], r["smiles"], r["selfies"],
    ] for r in results]

    wandb.log({
        "molecules": wandb.Table(columns=columns, data=table_data),
        "val/valid_pct": 100 * valid_count / num_mols,
    }, step=step)

    model.train()

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    project_root   = Path(CONFIG["project_root"])
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    tokenizer = ChemicalTokenizer(project_root / "chemical_tokenizer.json")

    print("Loading ZINC250k...")
    df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")

    full_dataset = ConditionalSELFIESDataset(df, tokenizer)

    val_size   = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    collator = ConditionalDiffusionCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, collate_fn=collator,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=collator,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )

    # --- Model (no SciBERT — cross-attention acts as 2nd self-attention) ---
    model = MolecularDiffusionModel(
        vocab_size      = CONFIG["vocab_size"],
        hidden_size     = CONFIG["hidden_size"],
        num_heads       = CONFIG["num_heads"],
        ffn_dim         = CONFIG["ffn_dim"],
        num_layers      = CONFIG["num_layers"],
        max_length      = CONFIG["max_length"],
        pad_token_id    = tokenizer.pad_token_id,
        text_model_name = None,    # NO SciBERT during pretraining
        dropout         = CONFIG["dropout"],
    ).to(device)

    total, trainable, frozen = model.count_parameters()
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    acc_steps = CONFIG["gradient_accumulation"]
    total_steps  = (len(train_loader) // acc_steps) * CONFIG["num_epochs"]
    warmup_steps = CONFIG["warmup_steps"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Total training steps : {total_steps:,}")
    print(f"Warmup steps         : {warmup_steps:,}")

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    global_step   = 0
    best_val_loss = float('inf')
    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # text_embeds=None → cross-attention uses x (2nd self-attention)
                logits = model(input_ids, timesteps)
                loss   = diffusion_loss(logits, labels, timesteps)
                loss   = loss / acc_steps

            loss.backward()

            if (step_idx + 1) % acc_steps == 0 or (step_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["log_every"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    actual_loss = loss.item() * acc_steps
                    print(f"Step {global_step:>6} | "
                          f"loss: {actual_loss:.4f} | "
                          f"grad_norm: {grad_norm:.4f} | "
                          f"lr: {lr:.2e}")
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": lr,
                        "train/epoch": epoch + (step_idx / len(train_loader)),
                    }, step=global_step)

                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, val_loader, device)
                    print(f"  → Val loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss}, step=global_step)

                    generate_samples(model, tokenizer, device, global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = checkpoint_dir / "best_model.pt"
                        torch.save({
                            "step": global_step, "epoch": epoch + 1,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "val_loss": val_loss, "config": CONFIG,
                        }, save_path)
                        print(f"  ✅ New best model (val_loss: {val_loss:.4f})")

                if global_step % CONFIG["save_every"] == 0:
                    save_path = checkpoint_dir / f"checkpoint_step{global_step}.pt"
                    torch.save({
                        "step": global_step, "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": CONFIG,
                    }, save_path)
                    print(f"  💾 Checkpoint saved at step {global_step}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    train()