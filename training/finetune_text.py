"""
training/finetune_text.py
--------------------------
Fine-tunes pretrained Morpheus on CheBI-20 for text-conditioned generation.

The pretrained model has cross-attention acting as a 2nd self-attention
(Q=molecule, K=molecule, V=molecule). During fine-tuning:

  - Cross-attention K and V projections adapt from molecule→text input
  - Cross-attention Q and out_proj are FROZEN (transfer directly)
  - All self-attention, FFN, embeddings, norms are FROZEN
  - Text projector MLP and null token train from scratch
  - SciBERT is frozen

This is an extremely targeted fine-tune: only the K/V projections in each
block need to learn "text looks like this instead of molecule." Everything
else already works.

Usage:
    python training/finetune_text.py
    python training/finetune_text.py --checkpoint checkpoints/best_model.pt
"""

import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import wandb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.textConditionedDataset import TextConditionedDataset
from training.textDiffusionCollator import TextDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Model (must match pretrained checkpoint)
    "vocab_size"  : 110,
    "hidden_size" : 768,
    "num_heads"   : 12,
    "ffn_dim"     : 3072,
    "num_layers"  : 16,
    "max_length"  : 74,
    "dropout"     : 0.1,

    # Text conditioning
    "text_model"    : "allenai/scibert_scivocab_uncased",
    "max_text_len"  : 256,
    "uncond_prob"   : 0.1,

    # Fine-tuning
    "batch_size"           : 32,
    "gradient_accumulation": 2,
    "new_lr": 3e-4, # cross-attn, text_proj, null_token, norm_cross
    "pretrained_lr": 1e-4, # everything else (pretrained) — very gentle to preserve knowledge
    "weight_decay"         : 0.01,
    "max_grad_norm"        : 1.33,
    "num_epochs"           : 66,
    "warmup_ratio"         : 0.1,

    # Data
    "num_workers" : 0,
    "seed"        : 42,

    # Logging & checkpoints
    "log_every"  : 100,
    "val_every"  : 1000,
    "save_every" : 2500,

    # Generation
    "gen_steps"       : 32,
    "gen_temperature" : 1.0,
    "gen_num_mols"    : 4,
    "cfg_scale"       : 3.0,

    # Paths
    "pretrained_checkpoint": str(ROOT / "checkpoints" / "best_model.pt"),
    "data_dir"             : str(ROOT / "data"),
    "checkpoint_dir"       : str(ROOT / "checkpoints"),
    "wandb_project"        : "morpheus-text-finetune",
}


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def diffusion_loss(logits, labels, timesteps, pad_token_id=0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id] = 0.1

    raw_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction='none'
    )

    #weights = (1.0 - timesteps)
    #weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)

    valid = (labels.view(-1) != -100).float()
    loss = (raw_loss * valid).sum() / (valid).sum().clamp(min=1e-8)
    return loss


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, max_batches=100):
    model.eval()
    total_loss  = 0.0
    total_steps = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            text_ids  = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            text_pad  = batch["text_padding_mask"].to(device)

            text_embeds = model.get_text_embeddings(text_ids, text_mask)
            logits = model(input_ids, timesteps, text_embeds, text_pad)
            loss   = diffusion_loss(logits, labels, timesteps)

            total_loss  += loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


# =============================================================================
# GENERATION WITH CFG
# =============================================================================

def generate_with_cfg(model, tokenizer, text_tokenizer, device,
                      prompts, cfg_scale=3.0, num_steps=32, temperature=1.0):
    """
    CFG for cross-attention-as-self-attention pretraining:
    - Conditioned: cross-attention attends to text embeddings
    - Unconditioned: cross-attention attends to molecule states (pretraining mode)
    """
    model.eval()
    max_length = CONFIG["max_length"]
    B = len(prompts)

    text_enc = text_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=CONFIG["max_text_len"], return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            text_enc["input_ids"], text_enc["attention_mask"]
        )
    text_pad = (text_enc["attention_mask"] == 0)

    input_ids = torch.full(
        (B, max_length), tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(B).unsqueeze(-1)

            # Conditioned: cross-attention attends to text
            logits_cond = model(input_ids, step_t, text_embeds, text_pad)

            # Unconditioned: cross-attention attends to molecule (text_embeds=None)
            logits_uncond = model(input_ids, step_t, None, None)

            # CFG interpolation
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float('-inf')

            probs   = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()

            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(
                    confidence, num_to_mask, dim=-1, largest=False
                )
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    model.train()
    return input_ids


def generate_and_log(model, tokenizer, text_tokenizer, device, step,
                     val_df=None):
    if val_df is not None and len(val_df) >= CONFIG["gen_num_mols"]:
        sample_df = val_df.sample(n=CONFIG["gen_num_mols"], random_state=step)
        prompts    = sample_df["description"].tolist()
        gt_selfies = sample_df["selfies"].tolist()
    else:
        prompts = [
            "The molecule is a member of the class of pyrimidines.",
            "The molecule is a primary alcohol with a hydroxyl group.",
            "The molecule contains a benzene ring with a carboxylic acid group.",
            "The molecule is a simple amino acid.",
        ][:CONFIG["gen_num_mols"]]
        gt_selfies = [None] * len(prompts)

    raw_ids = generate_with_cfg(
        model, tokenizer, text_tokenizer, device,
        prompts, cfg_scale=CONFIG["cfg_scale"],
        num_steps=CONFIG["gen_steps"], temperature=CONFIG["gen_temperature"],
    )

    print(f"\n{'='*70}")
    print(f"  GENERATED MOLECULES — Step {step}")
    print(f"{'='*70}")

    valid_count = 0
    for i in range(len(prompts)):
        ids = raw_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]

        selfies_str = tokenizer.decode(ids)
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                logp  = round(Descriptors.MolLogP(mol), 2)
                heavy = mol.GetNumHeavyAtoms()
                valid_count += 1
                print(f"  [{i}] ✅ LogP={logp:+.2f}  Heavy={heavy:>2}  {canonical[:50]}")
            else:
                print(f"  [{i}] ❌ RDKit rejected: {smiles[:50]}")
        except Exception as e:
            print(f"  [{i}] ❌ Error: {e}")

        prompt_short = prompts[i][:70] + ("..." if len(prompts[i]) > 70 else "")
        print(f"       Prompt: {prompt_short}")

        if gt_selfies[i] is not None:
            try:
                gt_smiles = sf.decoder(gt_selfies[i])
                gt_mol = Chem.MolFromSmiles(gt_smiles)
                if gt_mol:
                    print(f"       GT:     {Chem.MolToSmiles(gt_mol)[:50]}")
            except Exception:
                pass
        print()

    print(f"  Valid: {valid_count}/{len(prompts)}")
    print(f"{'='*70}\n")


# =============================================================================
# LOAD PRETRAINED CHECKPOINT
# =============================================================================

def load_pretrained(model, checkpoint_path, device):
    print(f"Loading pretrained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    pretrained_state = checkpoint["model"]

    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)

    print(f"  Trained for {checkpoint.get('step', '?'):,} steps "
          f"(epoch {checkpoint.get('epoch', '?')})")
    print(f"  Loaded keys:    {len(pretrained_state) - len(unexpected)}")
    print(f"  Missing keys:   {len(missing)} (new modules — expected)")
    print(f"  Unexpected keys: {len(unexpected)}")

    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected[:5]}...")

    # All cross-attention weights should LOAD (they were pretrained)
    cross_loaded = [k for k in pretrained_state if "cross_attention" in k and k not in unexpected]
    text_missing = [k for k in missing if "text_" in k or "null_token" in k]
    print(f"  Cross-attention keys loaded: {len(cross_loaded)} (pretrained as 2nd self-attn)")
    print(f"  Text encoder/proj keys (new): {len(text_missing)}")

    return checkpoint.get("config", {})


# =============================================================================
# FREEZE STRATEGY
# =============================================================================

def setup_freeze(model):
    """
    Freeze everything except:
      - cross_attention.k_proj  (adapt molecule→text input)
      - cross_attention.v_proj  (adapt molecule→text input)
      - text_proj               (learn SciBERT→model alignment)
      - null_token              (for CFG)
      - norm_cross              (adapt to new cross-attn distribution)

    Everything else transfers directly from pretraining:
      - self-attention (Q,K,V,out_proj) — unchanged
      - cross_attention.q_proj — already knows how to query from molecule states
      - cross_attention.out_proj — already knows how to project back
      - FFN — unchanged
      - embeddings, norms — unchanged
      - SciBERT — frozen always
    """
    trainable_keywords = [
        "cross_attention",
        "text_proj",
        "null_token",
        "norm_cross",
    ]

    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        should_train = any(kw in name for kw in trainable_keywords)

        if should_train:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Freeze strategy:")
    print(f"    Trainable param groups: {trainable_count}")
    print(f"    Frozen param groups:    {frozen_count}")
    print(f"    Trainable params:       {trainable_params:,}")
    print(f"    Total params:           {total_params:,}")
    print(f"    Training ratio:         {trainable_params/total_params:.1%}")

    print(f"\n  Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name:60s}  {param.numel():>10,}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args):
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    # --- Tokenizers ---
    mol_tokenizer  = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    # --- Load data ---
    data_dir = Path(CONFIG["data_dir"])
    train_path = data_dir / "chebi20_train.csv"
    val_path   = data_dir / "chebi20_val.csv"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        print("Run: python scripts/prepare_chebi20.py")
        sys.exit(1)

    print("Loading CheBI-20 data...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    train_dataset = TextConditionedDataset(train_df, mol_tokenizer)
    val_dataset   = TextConditionedDataset(val_df, mol_tokenizer)

    collator = TextDiffusionCollator(
        mol_tokenizer, text_tokenizer,
        max_text_len=CONFIG["max_text_len"]
    )

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

    # --- Build model WITH SciBERT ---
    print(f"\nBuilding model with SciBERT ({CONFIG['text_model']})...")
    model = MolecularDiffusionModel(
        vocab_size      = CONFIG["vocab_size"],
        hidden_size     = CONFIG["hidden_size"],
        num_heads       = CONFIG["num_heads"],
        ffn_dim         = CONFIG["ffn_dim"],
        num_layers      = CONFIG["num_layers"],
        max_length      = CONFIG["max_length"],
        pad_token_id    = mol_tokenizer.pad_token_id,
        text_model_name = CONFIG["text_model"],
        uncond_prob     = CONFIG["uncond_prob"],
        dropout         = CONFIG["dropout"],
    ).to(device)

    # --- Load pretrained weights (including cross-attention!) ---
    load_pretrained(model, args.checkpoint, device)

    # --- Freeze strategy: only K/V + text_proj + null_token train ---
    #setup_freeze(model)

    # --- Optimizer (only trainable params) ---
    acc_steps = CONFIG["gradient_accumulation"]
    steps_per_epoch = len(train_loader) // acc_steps
    total_steps = steps_per_epoch * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    new_module_keywords = ["cross_attention", "text_proj", "null_token", "norm_cross"]
    
    new_params = []
    pretrained_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen SciBERT
        if any(kw in name for kw in new_module_keywords):
            new_params.append(param)
        else:
            pretrained_params.append(param)
    
    print(f"New module params:      {sum(p.numel() for p in new_params):,}")
    print(f"Pretrained params:      {sum(p.numel() for p in pretrained_params):,}")

    optimizer = AdamW([
        {"params": new_params,       "lr": CONFIG["new_lr"]},
        {"params": pretrained_params, "lr": CONFIG["pretrained_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nSteps per epoch: {steps_per_epoch:,}")
    print(f"Total steps:     {total_steps:,}")
    print(f"Warmup steps:    {warmup_steps:,}")

    # --- W&B ---
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)

    # --- Training loop ---
    global_step   = 0
    best_val_loss = float('inf')
    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            text_ids  = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            text_pad  = batch["text_padding_mask"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                text_embeds = model.get_text_embeddings(text_ids, text_mask)
                logits = model(input_ids, timesteps, text_embeds, text_pad)
                loss = diffusion_loss(logits, labels, timesteps)
                loss = loss / acc_steps

            loss.backward()

            if (step_idx + 1) % acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    CONFIG["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- Logging ---
                if global_step % CONFIG["log_every"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    actual_loss = loss.item() * acc_steps
                    print(f"Step {global_step:>6} | "
                          f"loss: {actual_loss:.4f} | "
                          f"grad: {grad_norm:.4f} | "
                          f"lr: {lr:.2e}")
                    wandb.log({
                        "train/loss"     : actual_loss,
                        "train/grad_norm": float(grad_norm),
                        "train/lr"       : lr,
                        "train/epoch"    : epoch + (step_idx / len(train_loader)),
                    }, step=global_step)

                # --- Validation + generation ---
                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, val_loader, device)
                    print(f"  → Val loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss}, step=global_step)

                    generate_and_log(
                        model, mol_tokenizer, text_tokenizer,
                        device, global_step, val_df
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = checkpoint_dir / "best_finetuned_model.pt"
                        torch.save({
                            "step"      : global_step,
                            "epoch"     : epoch + 1,
                            "model"     : model.state_dict(),
                            "optimizer" : optimizer.state_dict(),
                            "scheduler" : scheduler.state_dict(),
                            "val_loss"  : val_loss,
                            "config"    : CONFIG,
                        }, save_path)
                        print(f"  ✅ New best model (val_loss: {val_loss:.4f})")

                # --- Periodic checkpoint ---
                if global_step % CONFIG["save_every"] == 0:
                    save_path = checkpoint_dir / f"finetune_step{global_step}.pt"
                    torch.save({
                        "step"     : global_step,
                        "epoch"    : epoch + 1,
                        "model"    : model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config"   : CONFIG,
                    }, save_path)
                    print(f"  💾 Checkpoint saved at step {global_step}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    wandb.finish()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Morpheus with text conditioning on CheBI-20"
    )
    parser.add_argument(
        "--checkpoint",
        default=CONFIG["pretrained_checkpoint"],
        help="Path to pretrained checkpoint (with cross-attention)"
    )
    parser.add_argument("--new_lr", type=float, default=None)
    parser.add_argument("--pretrained_lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    if args.new_lr is not None:
        CONFIG["new_lr"] = args.new_lr
    if args.pretrained_lr is not None:
        CONFIG["pretrained_lr"] = args.pretrained_lr
    if args.epochs is not None:
        CONFIG["num_epochs"] = args.epochs
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size

    train(args)