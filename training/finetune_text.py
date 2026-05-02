"""
training/finetune_text.py
--------------------------
Fine-tune pretrained Morpheus on CheBI-20 with text conditioning.

KEY FEATURES:
  - Cross-attention to text embeddings (zero-init out_proj for stable start)
  - Optional contrastive text encoder (--contrastive_ckpt)
  - NO EMA (per user preference)
  - Differential LR (new modules vs pretrained)
  - EOS-weighted loss (eos_weight=5.0, pad_weight=0.05)
  - revealPad strategy
  - RoPE handles arbitrary max_length — pretrained at 74 works at 256

Usage:
    # Plain (frozen HF text encoder)
    python training/finetune_text.py --checkpoint checkpoints/best_model.pt

    # With contrastive aligner (RECOMMENDED)
    python training/finetune_text.py \
        --checkpoint checkpoints/best_model.pt \
        --contrastive_ckpt checkpoints/contrastive_aligner.pt
"""

import sys
import argparse
import random
import numpy as np
import torch
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModel
import wandb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.datasets import TextConditionedDataset
from training.textDiffusionCollator import TextDiffusionCollator
from training.losses import diffusion_loss
from model.molecularDiffusionModel import MolecularDiffusionModel


CONFIG = {
    # Model architecture (must match pretrained checkpoint)
    "vocab_size":  110,
    "hidden_size": 512,
    "num_heads":   8,
    "ffn_dim":     1024,
    "num_layers":  8,
    # IMPORTANT: max_length can DIFFER from pretraining because RoPE
    # is parameter-free and generalizes to any sequence length.
    # For CheBI-20 fine-tuning, you typically want this larger (e.g. 128).
    "max_length":  128,
    "dropout":     0.1,

    # Text
    "text_model":   "BAAI/bge-large-en-v1.5",
    "max_text_len": 128,
    "uncond_prob":  0.1,

    # Training
    "batch_size":             32,
    "gradient_accumulation":  1,
    "new_lr":                 2e-4,    # cross-attn, text_proj, null_token
    "pretrained_lr":          2e-5,    # self-attn, FFN, embeddings
    "weight_decay":           0.01,
    "max_grad_norm":          1.0,
    "num_epochs":             20,
    "warmup_steps":           500,

    # Loss weights
    "eos_weight": 5.0,
    "pad_weight": 0.05,

    # Validation
    "val_every":      500,
    "val_max_batches": 100,
    "save_every":     2000,

    # Generation samples during training
    "gen_steps":       50,
    "gen_temperature": 1.0,
    "gen_cfg_scale":   3.0,

    "num_workers": 2,
    "seed":        42,

    "data_dir":       str(ROOT / "data"),
    "checkpoint_dir": str(ROOT / "checkpoints"),
    "wandb_project":  "morpheus-text-finetune",
}


# =============================================================================
# CONTRASTIVE ENCODER LOADER
# =============================================================================

def load_contrastive_text_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    text_model_name = cfg["text_model"]

    text_state = {
        k[len("text_model."):]: v
        for k, v in ckpt["model_state"].items()
        if k.startswith("text_model.")
    }

    encoder = AutoModel.from_pretrained(text_model_name)
    encoder.load_state_dict(text_state, strict=True)
    print(f"[contrastive] {ckpt_path.name}  text_model={text_model_name}  "
          f"epoch={ckpt.get('best_epoch', '?')}")
    return encoder.to(device), text_model_name


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, tokenizer, max_batches=100):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            text_pad = batch["text_padding_mask"].to(device)

            text_embeds = model.get_text_embeddings(text_ids, text_mask)
            logits = model(input_ids, timesteps, text_embeds, text_pad)
            loss = diffusion_loss(logits, labels, tokenizer,
                                   CONFIG["eos_weight"], CONFIG["pad_weight"])
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


# =============================================================================
# GENERATION (CFG)
# =============================================================================

@torch.no_grad()
def generate_cfg(model, tokenizer, text_tokenizer, device, prompts,
                 cfg_scale=3.0, num_steps=50, temperature=1.0):
    model.eval()
    B = len(prompts)
    max_length = CONFIG["max_length"]

    text_inputs = text_tokenizer(prompts, padding=True, truncation=True,
                                  max_length=CONFIG["max_text_len"], return_tensors="pt").to(device)
    text_pad = (text_inputs["attention_mask"] == 0)
    text_embeds = model.get_text_embeddings(text_inputs["input_ids"],
                                              text_inputs["attention_mask"])
    null_embeds = model.null_token.expand(B, text_embeds.size(1), -1)

    input_ids = torch.full((B, max_length), tokenizer.mask_token_id,
                            dtype=torch.long, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    for step_idx, t_val in enumerate(t_vals):
        step_t = t_val.repeat(B).unsqueeze(-1)
        cond_l = model(input_ids, step_t, text_embeds, text_pad)
        uncond_l = model(input_ids, step_t, null_embeds, text_pad)
        logits = uncond_l + cfg_scale * (cond_l - uncond_l)

        if step_idx < num_steps - 1:
            logits[:, :, tokenizer.mask_token_id] = float("-inf")

        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.distributions.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

        alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
        n_mask = int((1.0 - alpha_t) * max_length)
        if n_mask > 0 and step_idx < num_steps - 1:
            _, mi = torch.topk(confidence, n_mask, dim=-1, largest=False)
            sampled.scatter_(1, mi, tokenizer.mask_token_id)

        input_ids = sampled

    model.train()
    return input_ids.cpu().tolist()


def log_generation(model, tokenizer, text_tokenizer, device, step, val_df=None):
    if val_df is not None and len(val_df) >= 4:
        sample = val_df.sample(n=4, random_state=step)
        prompts = sample["description"].tolist()
        gt = sample["selfies"].tolist()
    else:
        prompts = ["The molecule is a member of the class of pyrimidines.",
                    "The molecule is a primary alcohol.",
                    "The molecule contains a benzene ring.",
                    "The molecule is a simple amino acid."]
        gt = [None] * 4

    raw = generate_cfg(model, tokenizer, text_tokenizer, device, prompts,
                       CONFIG["gen_cfg_scale"], CONFIG["gen_steps"], CONFIG["gen_temperature"])

    print(f"\n  GEN (step {step}) ────────────────────────────")
    for i, ids in enumerate(raw):
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        ids = [t for t in ids if t not in (tokenizer.pad_token_id,
                                              tokenizer.mask_token_id)]
        s = tokenizer.decode(ids)
        try:
            mol = Chem.MolFromSmiles(sf.decoder(s))
            status = f"✅ {Chem.MolToSmiles(mol)[:55]}" if mol else "❌"
        except Exception:
            status = "❌"
        print(f"    [{i}] {status}")
        print(f"        Prompt: {prompts[i][:60]}")
        if gt[i]:
            try:
                gt_mol = Chem.MolFromSmiles(sf.decoder(gt[i]))
                if gt_mol:
                    print(f"        GT:     {Chem.MolToSmiles(gt_mol)[:55]}")
            except Exception:
                pass


# =============================================================================
# MAIN
# =============================================================================

def train_main(args):
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    mol_tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))

    # Decide text encoder
    if args.contrastive_ckpt:
        custom_encoder, text_model_name = load_contrastive_text_encoder(
            Path(args.contrastive_ckpt), device)
        CONFIG["text_model"] = text_model_name
    else:
        custom_encoder = None

    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    # Data
    data_dir = Path(CONFIG["data_dir"])
    train_df = pd.read_csv(data_dir / "chebi20_train.csv")
    val_df = pd.read_csv(data_dir / "chebi20_val.csv")
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    train_dataset = TextConditionedDataset(
        train_df, mol_tokenizer, max_length=CONFIG["max_length"])
    val_dataset = TextConditionedDataset(
        val_df, mol_tokenizer, max_length=CONFIG["max_length"])

    collator = TextDiffusionCollator(
        mol_tokenizer, text_tokenizer=text_tokenizer,
        max_text_len=CONFIG["max_text_len"], use_precomputed=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                                shuffle=True, collate_fn=collator,
                                num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=False, collate_fn=collator,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

    # Build model
    # NOTE: max_length here can be LARGER than the pretrained max_length.
    # RoPE generalizes to any length without retraining position embeddings.
    model = MolecularDiffusionModel(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["hidden_size"],
        num_heads=CONFIG["num_heads"], ffn_dim=CONFIG["ffn_dim"],
        num_layers=CONFIG["num_layers"], max_length=CONFIG["max_length"],
        pad_token_id=mol_tokenizer.pad_token_id,
        text_model_name=CONFIG["text_model"],
        uncond_prob=CONFIG["uncond_prob"],
        dropout=CONFIG["dropout"],
        load_text_encoder=(custom_encoder is None),
    ).to(device)

    if custom_encoder is not None:
        model.set_text_encoder(custom_encoder)

    # Load pretrained weights (strict=False because cross-attention is new)
    print(f"Loading pretrained: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Filter out keys that don't match (e.g. if positional embedding existed in old ckpt)
    pretrained_state = ckpt["model"]
    model_state = model.state_dict()
    filtered = {}
    for k, v in pretrained_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
        else:
            print(f"  Skipping {k}: not in new model")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Loaded: {len(filtered)}/{len(pretrained_state)} keys")
    print(f"  Missing in checkpoint (new modules): {len(missing)}")
    cross_loaded = sum(1 for k in filtered if "cross_attention" in k)
    print(f"  Cross-attn pre-loaded: {cross_loaded} (zero-init since not in pretrain)")

    # Differential LR
    new_kw = ["cross_attention", "text_proj", "null_token", "norm_cross", "encoder_proj"]
    new_params, pre_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in new_kw):
            new_params.append(p)
        else:
            pre_params.append(p)
    print(f"New: {sum(p.numel() for p in new_params):,}  "
          f"Pretrained: {sum(p.numel() for p in pre_params):,}")

    optimizer = AdamW([
        {"params": new_params, "lr": CONFIG["new_lr"]},
        {"params": pre_params, "lr": CONFIG["pretrained_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    acc = CONFIG["gradient_accumulation"]
    total_steps = (len(train_loader) // acc) * CONFIG["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps)
    print(f"Total steps: {total_steps:,}  Warmup: {CONFIG['warmup_steps']}")

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    global_step, best_val = 0, float("inf")
    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            text_pad = batch["text_padding_mask"].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                text_embeds = model.get_text_embeddings(text_ids, text_mask)
                logits = model(input_ids, timesteps, text_embeds, text_pad)
                loss = diffusion_loss(logits, labels, mol_tokenizer,
                                       CONFIG["eos_weight"], CONFIG["pad_weight"])
                loss = loss / acc
            loss.backward()

            if (step_idx + 1) % acc == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    CONFIG["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    lr_n = optimizer.param_groups[0]["lr"]
                    lr_p = optimizer.param_groups[1]["lr"]
                    print(f"S {global_step:>5} | loss {loss.item() * acc:.4f} | "
                          f"gn {gn:.3f} | lr_n {lr_n:.2e} | lr_p {lr_p:.2e}")
                    wandb.log({"train/loss": loss.item() * acc,
                                "train/grad_norm": float(gn),
                                "train/lr_new": lr_n,
                                "train/lr_pre": lr_p}, step=global_step)

                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, val_loader, device,
                                                mol_tokenizer, CONFIG["val_max_batches"])
                    print(f"  → Val: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss}, step=global_step)
                    log_generation(model, mol_tokenizer, text_tokenizer,
                                    device, global_step, val_df)

                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({
                            "step": global_step, "epoch": epoch + 1,
                            "model": model.state_dict(),
                            "val_loss": val_loss, "config": CONFIG,
                        }, checkpoint_dir / "best_finetuned_model.pt")
                        print(f"  ✅ New best (val: {val_loss:.4f})")

                if global_step % CONFIG["save_every"] == 0:
                    torch.save({
                        "step": global_step, "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": CONFIG,
                    }, checkpoint_dir / f"finetune_step{global_step}.pt")
                    print(f"  💾 Saved step {global_step}")

    print(f"\nDone. Best val: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contrastive_ckpt", default=None)
    parser.add_argument("--max_length", type=int, default=None,
                         help="Override max_length (RoPE works at any length)")
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()
    if args.max_length:
        CONFIG["max_length"] = args.max_length
    if args.num_epochs:
        CONFIG["num_epochs"] = args.num_epochs
    train_main(args)