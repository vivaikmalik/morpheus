"""
finetune/resume_eos_fix.py
--------------------------
Resumes fine-tuning from checkpoints/best_finetuned_model.pt with two fixes:

  1. EOS weight fix: diffusion_loss now uses eos_weight=5.0, pad_weight=0.05
     (matching train_upscaled.py) so the model learns to terminate sequences
     at the correct length instead of padding to the 74-token window.

  2. Lower LR (1e-5) with short warmup (200 steps) — gentle continuation,
     not aggressive retraining.

Changes vs train_text_condition.py:
  - Load checkpoint: best_finetuned_model.pt  (not best_model_upscaled.pt)
  - diffusion_loss: eos_weight=5.0, pad_weight=0.05
  - learning_rate: 1e-5
  - num_epochs: 3
  - warmup_steps: 200 (fixed, not ratio-based)
  - Best checkpoint saved as: best_finetuned_eos_fix.pt
  - Logs/plots: outputs/finetune_eos_fix_log.csv,
                outputs/plots/finetune_eos_fix/
  - CFG samples: outputs/finetune_eos_fix_samples.txt

Usage:
    python finetune/resume_eos_fix.py
"""

import csv
import sys
import os
import time
import torch
import random
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModel
import wandb
import glob as _glob
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from finetune.conditionalSELFIESDataset import ConditionalSELFIESDataset
from finetune.diffusionCollatorPrompt import ConditionalDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # Model — must match best_finetuned_model.pt
    "vocab_size"  : 110,
    "hidden_size" : 512,
    "num_heads"   : 8,
    "ffn_dim"     : 1024,
    "num_layers"  : 8,
    "max_length"  : 74,
    "dropout"     : 0.1,
    "text_model"  : "BAAI/bge-large-en-v1.5",
    "uncond_prob" : 0.1,

    # Training — gentle continuation
    "batch_size"      : 32,
    "learning_rate"   : 1e-5,       # ← low LR for continuation
    "weight_decay"    : 0.01,
    "max_grad_norm"   : 1.0,
    "num_epochs"      : 3,          # ← short run
    "warmup_steps"    : 200,        # ← fixed steps, not ratio-based

    # EOS/PAD loss weights (the core fix)
    "eos_weight"  : 5.0,
    "pad_weight"  : 0.05,

    # Data
    "data_path"     : "data/train.csv",
    "val_data_path" : "data/val.csv",
    "val_fraction"  : 0.05,         # fallback if val CSV absent
    "num_workers"   : 0,
    "seed"          : 42,

    # Logging
    "log_every"       : 50,
    "val_every"       : 1500,
    "val_max_batches" : 200,
    "early_stop_patience": 5,

    # Generation (for mid-training samples)
    "gen_steps"       : 32,
    "gen_temperature" : 1.2,

    # Paths
    "project_root"  : str(Path(__file__).parent.parent),
    "checkpoint_dir": str(Path(__file__).parent.parent / "checkpoints"),
    "resume_ckpt"   : "best_finetuned_model.pt",   # checkpoint to resume from
    "save_ckpt"     : "best_finetuned_eos_fix.pt",  # checkpoint to save to
    "log_path"      : str(Path(__file__).parent.parent / "outputs" / "finetune_eos_fix_log.csv"),
    "plot_dir"      : str(Path(__file__).parent.parent / "outputs" / "plots" / "finetune_eos_fix"),
    "wandb_project" : "morpheus-diffusion-finetune",
    "freeze_text_encoder": True,
}

# =============================================================================
# LOSS — with EOS weight fix
# =============================================================================

def diffusion_loss(logits, labels, timesteps,
                   pad_token_id=0, eos_token_id=2,
                   pad_weight=0.05, eos_weight=5.0):
    """
    Weighted cross-entropy on masked positions only, with:
      - eos_weight=5.0  → emphasise learning sequence termination
      - pad_weight=0.05 → de-emphasise padding (was 0.1 in original)
      - (1-t) weighting → emphasise low-noise steps
    """
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights                = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id]  = pad_weight
    vocab_weights[eos_token_id]  = eos_weight

    raw_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction="none",
    )

    weights = (1.0 - timesteps)
    weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)
    valid   = (labels.view(-1) != -100).float()

    loss = (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)
    return loss

# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, tokenizer):
    model.eval()
    total_loss, total_steps = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False,
                          total=min(CONFIG["val_max_batches"], len(val_loader))):
            input_ids           = batch["input_ids"].to(device)
            labels              = batch["labels"].to(device)
            timesteps           = batch["timesteps"].to(device)
            text_input_ids      = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            text_padding_mask   = (text_attention_mask == 0)

            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)
            logits      = model(input_ids, timesteps, text_embeds, text_padding_mask)
            loss        = diffusion_loss(
                logits, labels, timesteps,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_weight=CONFIG["pad_weight"],
                eos_weight=CONFIG["eos_weight"],
            )

            total_loss  += loss.item()
            total_steps += 1
            if total_steps >= CONFIG["val_max_batches"]:
                break

    model.train()
    return total_loss / max(total_steps, 1)

# =============================================================================
# GENERATION (CFG, for mid-training sanity check)
# =============================================================================

def generate_samples(model, tokenizer, hf_tokenizer, device, step):
    model.eval()
    max_length  = CONFIG["max_length"]
    num_steps   = CONFIG["gen_steps"]
    temperature = CONFIG["gen_temperature"]
    cfg_scale   = 3.0

    prompts = [
        "A drug-like molecule with high QED score.",
        "A small fragment molecule with MW below 250.",
        "A molecule containing fluorine.",
        "A complex molecule with multiple rings.",
    ]
    num_mols = len(prompts)

    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)
    input_ids         = torch.full((num_mols, max_length), tokenizer.mask_token_id, device=device)
    t_vals            = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        null_embeds = model.null_token.expand(num_mols, text_embeds.size(1), -1)

        for step_idx, t_val in enumerate(t_vals):
            step_t        = t_val.repeat(num_mols).unsqueeze(-1)
            cond_logits   = model(input_ids, step_t, text_embeds,  text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds,  text_padding_mask)
            logits        = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")

            probs      = torch.softmax(logits / temperature, dim=-1)
            sampled    = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            import math
            alpha_t     = (torch.cos(t_val * math.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)
            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_idx = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mask_idx, tokenizer.mask_token_id)
            input_ids = sampled

    print(f"\n{'='*60}\n  CFG SAMPLES — Step {step}\n{'='*60}")
    token_lengths = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        tok_len = len(ids)
        token_lengths.append(tok_len)
        selfies_str = tokenizer.decode(ids)
        try:
            smiles = sf.decoder(selfies_str)
            mol    = Chem.MolFromSmiles(smiles)
            status = f"✓ {Chem.MolToSmiles(mol)[:55]}" if mol else "✗ invalid"
        except Exception:
            status = "✗ error"
        print(f"  [{i}] {tok_len} tok | {prompts[i][:40]}\n      {status}")
    print(f"  Avg token length: {sum(token_lengths)/len(token_lengths):.1f} "
          f"(target ~43)\n{'='*60}\n")
    model.train()

# =============================================================================
# CFG QUALITATIVE EVALUATION (end-of-training)
# =============================================================================

def save_cfg_samples(model, tokenizer, hf_tokenizer, device, out_path: str,
                     cfg_scale=3.0, num_steps=32, temperature=1.2):
    model.eval()
    import math
    max_length = CONFIG["max_length"]

    prompts = [
        "A drug-like molecule with high QED score and good oral bioavailability.",
        "A drug-like molecule with high QED score and good oral bioavailability.",
        "A small fragment molecule with molecular weight below 250 Da.",
        "A small fragment molecule with molecular weight below 250 Da.",
        "A molecule containing fluorine with aromatic rings.",
        "A molecule containing fluorine with aromatic rings.",
        "A complex molecule with multiple fused rings and nitrogen atoms.",
        "A complex molecule with multiple fused rings and nitrogen atoms.",
    ]
    num_mols = len(prompts)

    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)
    input_ids = torch.full((num_mols, max_length), tokenizer.mask_token_id, device=device)
    t_vals    = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        null_embeds = model.null_token.expand(num_mols, text_embeds.size(1), -1)

        for step_idx, t_val in enumerate(t_vals):
            step_t        = t_val.repeat(num_mols).unsqueeze(-1)
            cond_logits   = model(input_ids, step_t, text_embeds,  text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds,  text_padding_mask)
            logits        = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")

            probs      = torch.softmax(logits / temperature, dim=-1)
            sampled    = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            alpha_t     = (torch.cos(t_val * math.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)
            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_idx = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mask_idx, tokenizer.mask_token_id)
            input_ids = sampled

    results = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        selfies_str = tokenizer.decode(ids)
        entry = {"prompt": prompts[i], "selfies": selfies_str,
                 "token_count": len(ids), "valid": False,
                 "smiles": "", "mw": None, "qed": None}
        try:
            smiles = sf.decoder(selfies_str)
            mol    = Chem.MolFromSmiles(smiles)
            if mol is not None:
                entry["valid"]  = True
                entry["smiles"] = Chem.MolToSmiles(mol)
                entry["mw"]     = round(Descriptors.MolWt(mol), 1)
                entry["qed"]    = round(QED.qed(mol), 3)
        except Exception:
            pass
        results.append(entry)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    unique_prompts = list(dict.fromkeys(prompts))
    valid_count    = sum(r["valid"] for r in results)
    avg_tok        = sum(r["token_count"] for r in results) / len(results)

    with open(out_path, "w") as f:
        f.write("CFG GENERATION SAMPLES — EOS Fix Resume\n")
        f.write(f"cfg_scale={cfg_scale}  temperature={temperature}  steps={num_steps}\n")
        f.write(f"Valid: {valid_count}/{num_mols}  |  Avg token length: {avg_tok:.1f} (target ~43)\n")
        f.write("=" * 70 + "\n\n")
        for prompt in unique_prompts:
            f.write(f"PROMPT: {prompt}\n")
            f.write("-" * 70 + "\n")
            for j, r in enumerate([r for r in results if r["prompt"] == prompt]):
                status = "VALID" if r["valid"] else "INVALID"
                f.write(f"  [{j+1}] {status}  ({r['token_count']} tokens)\n")
                if r["valid"]:
                    f.write(f"       SMILES : {r['smiles']}\n")
                    f.write(f"       MW={r['mw']}  QED={r['qed']}\n")
                else:
                    f.write(f"       SELFIES: {r['selfies'][:80]}\n")
            f.write("\n")

    print(f"\n{'='*70}\nCFG GENERATION SAMPLES — EOS Fix")
    print(f"Valid: {valid_count}/{num_mols}  |  Avg token length: {avg_tok:.1f} (target ~43)")
    print(f"{'='*70}")
    for prompt in unique_prompts:
        print(f"\nPROMPT: {prompt}")
        for r in [r for r in results if r["prompt"] == prompt]:
            if r["valid"]:
                print(f"  ✓ {r['token_count']} tok | {r['smiles'][:60]}  MW={r['mw']} QED={r['qed']}")
            else:
                print(f"  ✗ {r['token_count']} tok | {r['selfies'][:60]}")
    print(f"\nSaved → {out_path}")
    model.train()

# =============================================================================
# PLOTS
# =============================================================================

def save_plots(log_path: str, plot_dir: str):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    if df.empty:
        print("[plot] Log empty, skipping.")
        return

    train_rows = df[df["train_loss"].notna()].copy()
    val_rows   = df[df["val_loss"].notna()].copy()
    style      = {"linewidth": 1.5, "alpha": 0.9}

    fig, ax = plt.subplots(figsize=(9, 5))
    if not train_rows.empty:
        ax.plot(train_rows["step"], train_rows["train_loss"],
                label="Train loss", color="#2196F3", **style)
    if not val_rows.empty:
        ax.plot(val_rows["step"], val_rows["val_loss"],
                label="Val loss", color="#F44336", linestyle="--", **style)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("EOS Fix Resume — Fine-tuning Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", dpi=300)
    plt.close(fig)

    if not train_rows.empty and "lr" in train_rows.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_rows["step"], train_rows["lr"], color="#4CAF50", **style)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (cosine + warmup)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "lr_schedule.png", dpi=300)
        plt.close(fig)

    print(f"[plot] Saved → {plot_dir}/")

# =============================================================================
# TRAINING
# =============================================================================

def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    project_root   = Path(CONFIG["project_root"])
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    # --- Tokenizers ---
    tokenizer    = ChemicalTokenizer(project_root / "chemical_tokenizer.json")
    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    # --- Data ---
    print(f"[data] Loading {CONFIG['data_path']} ...")
    df = pd.read_csv(CONFIG["data_path"])
    if "response" not in df.columns:
        raise ValueError("CSV must have a 'response' column.")
    if "prompt" not in df.columns:
        df["prompt"] = "A chemical molecule"

    train_dataset = ConditionalSELFIESDataset(df, tokenizer)

    if CONFIG["val_data_path"] and os.path.exists(CONFIG["val_data_path"]):
        val_df = pd.read_csv(CONFIG["val_data_path"])
        if "prompt" not in val_df.columns:
            val_df["prompt"] = "A chemical molecule"
        val_dataset = ConditionalSELFIESDataset(val_df, tokenizer)
        train_size  = len(train_dataset)
        print(f"[data] train={train_size:,}  val={len(val_dataset):,} (separate CSV)")
    else:
        val_size      = int(len(train_dataset) * CONFIG["val_fraction"])
        train_size    = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print(f"[data] train={train_size:,}  val={val_size:,} (random split)")

    collator     = ConditionalDiffusionCollator(tokenizer, hf_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True,  collate_fn=collator,
                              num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False, collate_fn=collator,
                              num_workers=CONFIG["num_workers"])

    # --- Model ---
    model = MolecularDiffusionModel(
        vocab_size      = CONFIG["vocab_size"],
        hidden_size     = CONFIG["hidden_size"],
        num_heads       = CONFIG["num_heads"],
        ffn_dim         = CONFIG["ffn_dim"],
        num_layers      = CONFIG["num_layers"],
        max_length      = CONFIG["max_length"],
        pad_token_id    = tokenizer.pad_token_id,
        text_model_name = CONFIG["text_model"],
        uncond_prob     = CONFIG["uncond_prob"],
        dropout         = CONFIG["dropout"],
    ).to(device)

    # --- Load finetuned checkpoint ---
    resume_path = checkpoint_dir / CONFIG["resume_ckpt"]
    if resume_path.exists():
        print(f"[ckpt] Resuming from {resume_path.name} ...")
        ckpt = torch.load(resume_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing key groups: "
                  f"{set(k.split('.')[0] for k in missing)}")
        prev_epoch = ckpt.get("epoch", 0)
        prev_step  = ckpt.get("step",  0)
        prev_loss  = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed from epoch={prev_epoch}, step={prev_step}, "
              f"val_loss={prev_loss:.4f}")
    else:
        raise FileNotFoundError(
            f"{resume_path} not found. Run finetune/train_text_condition.py first."
        )

    # --- Freeze text encoder ---
    frozen = CONFIG["freeze_text_encoder"]
    for p in model.text_encoder.parameters():
        p.requires_grad = not frozen
    print(f"[model] Text encoder {'frozen' if frozen else 'unfrozen'}.")

    total, trainable = model.count_parameters()
    print(f"[model] {total/1e6:.1f}M total | {trainable/1e6:.1f}M trainable")

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps  = (train_size // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    warmup_steps = CONFIG["warmup_steps"]   # fixed, not ratio-based

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[sched] LR={CONFIG['learning_rate']}  "
          f"total_steps={total_steps:,}  warmup={warmup_steps}")
    print(f"[loss]  eos_weight={CONFIG['eos_weight']}  "
          f"pad_weight={CONFIG['pad_weight']}")

    # --- W&B ---
    run = wandb.init(
        project  = CONFIG["wandb_project"],
        name     = "eos-fix-resume",
        config   = CONFIG,
        reinit   = True,
        settings = wandb.Settings(start_method="thread"),
    )
    print(f"[wandb] {run.url}")

    # --- CSV log ---
    log_path = Path(CONFIG["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_columns = ["epoch", "step", "train_loss", "val_loss", "lr",
                   "tokens_per_sec", "elapsed_min"]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_columns)
    log_writer.writeheader()
    log_file.flush()

    # --- Training loop ---
    global_step        = 0
    best_val_loss      = float("inf")
    last_val_loss      = float("inf")
    patience_counter   = 0
    stopped_early      = False
    t_start            = time.time()
    tokens_per_sec_ema = None

    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'─'*65}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for batch in progress_bar:
            step_t0             = time.time()
            input_ids           = batch["input_ids"].to(device)
            labels              = batch["labels"].to(device)
            timesteps           = batch["timesteps"].to(device)
            text_input_ids      = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            text_padding_mask   = (text_attention_mask == 0)

            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)
            logits      = model(input_ids, timesteps, text_embeds, text_padding_mask)
            loss        = diffusion_loss(
                logits, labels, timesteps,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_weight=CONFIG["pad_weight"],
                eos_weight=CONFIG["eos_weight"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            global_step += 1

            step_time = max(time.time() - step_t0, 1e-6)
            tps = input_ids.numel() / step_time
            tokens_per_sec_ema = (
                tps if tokens_per_sec_ema is None
                else 0.05 * tps + 0.95 * tokens_per_sec_ema
            )

            if global_step % CONFIG["log_every"] == 0:
                current_lr  = scheduler.get_last_lr()[0]
                elapsed_min = (time.time() - t_start) / 60
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}"
                )
                wandb.log({"train/loss": loss.item(), "train/lr": current_lr},
                          step=global_step)
                log_writer.writerow({
                    "epoch": epoch + 1, "step": global_step,
                    "train_loss": round(loss.item(), 6), "val_loss": "",
                    "lr": round(current_lr, 8),
                    "tokens_per_sec": round(tokens_per_sec_ema, 1),
                    "elapsed_min": round(elapsed_min, 2),
                })
                log_file.flush()

            if global_step % CONFIG["val_every"] == 0:
                val_loss      = run_validation(model, val_loader, device, tokenizer)
                last_val_loss = val_loss
                print(f"  → Val loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss}, step=global_step)

                generate_samples(model, tokenizer, hf_tokenizer, device, global_step)

                elapsed_min = (time.time() - t_start) / 60
                log_writer.writerow({
                    "epoch": epoch + 1, "step": global_step,
                    "train_loss": "", "val_loss": round(val_loss, 6),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "tokens_per_sec": round(tokens_per_sec_ema or 0, 1),
                    "elapsed_min": round(elapsed_min, 2),
                })
                log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    patience_counter = 0
                    save_path = checkpoint_dir / CONFIG["save_ckpt"]
                    torch.save({
                        "epoch"    : epoch + 1,
                        "step"     : global_step,
                        "model"    : model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_loss" : val_loss,
                        "config"   : CONFIG,
                    }, save_path)
                    print(f"  ✓ New best → {save_path.name}  (val_loss={val_loss:.4f})")

                    artifact = wandb.Artifact(
                        name=f"eos-fix-valloss{val_loss:.4f}",
                        type="model",
                        metadata={"val_loss": val_loss, "step": global_step},
                    )
                    artifact.add_file(str(save_path))
                    wandb.log_artifact(artifact)
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{CONFIG['early_stop_patience']})")
                    if patience_counter >= CONFIG["early_stop_patience"]:
                        print("Early stopping triggered.")
                        stopped_early = True
                        break

        # Periodic checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            save_path = checkpoint_dir / f"eos_fix_epoch_{epoch+1}.pt"
            torch.save({
                "epoch"    : epoch + 1,
                "step"     : global_step,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss" : last_val_loss,
                "config"   : CONFIG,
            }, save_path)
            print(f"  Periodic checkpoint → {save_path.name}")

        if stopped_early:
            break

    log_file.close()
    print(f"\nResume training complete.{' (early stop)' if stopped_early else ''}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Log → {log_path}")

    save_plots(str(log_path), CONFIG["plot_dir"])
    save_cfg_samples(
        model, tokenizer, hf_tokenizer, device,
        out_path=str(project_root / "outputs" / "finetune_eos_fix_samples.txt"),
    )
    wandb.finish()


if __name__ == "__main__":
    train()
