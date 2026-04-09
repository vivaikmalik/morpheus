"""
training/finetune_text.py — v3 with EMA + regularization
"""

import sys, copy, argparse, random
import numpy as np, torch, torch.nn.functional as F, pandas as pd, selfies as sf
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

CONFIG = {
    "vocab_size": 110, "hidden_size": 1024, "num_heads": 16,
    "ffn_dim": 4096, "num_layers": 12, "max_length": 256,
    "dropout": 0.15,
    "text_model": "allenai/scibert_scivocab_uncased",
    "max_text_len": 256, "uncond_prob": 0.2,
    "batch_size": 256, "gradient_accumulation": 4,
    "new_lr": 5e-4, "pretrained_lr": 1e-5,
    "weight_decay": 0.07, "max_grad_norm": 1.1,
    "num_epochs": 420, "warmup_steps": 1000,
    "ema_decay": 0.9999,
    "num_workers": 12, "seed": 42, "use_precomputed": True,
    "log_every": 50, "val_every": 1000, "save_every": 5000,
    "gen_steps": 50, "gen_temperature": 1.0, "gen_num_mols": 4, "cfg_scale": 3.0,
    "pretrained_checkpoint": str(ROOT / "checkpoints" / "best_model_expanded.pt"),
    "data_dir": str(ROOT / "data"),
    "checkpoint_dir": str(ROOT / "checkpoints"),
    "wandb_project": "morpheus-text-finetune",
}

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1-self.decay)
    def apply_to(self, model):
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup
    def restore(self, model, backup):
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}
    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

def diffusion_loss(logits, labels, pad_token_id=0):
    B, seq_len, vocab_size = logits.shape
    vocab_weights = torch.ones(vocab_size, device=logits.device)
    vocab_weights[pad_token_id] = 0.1
    raw = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1),
                          weight=vocab_weights, ignore_index=-100, reduction='none')
    valid = (labels.view(-1) != -100).float()
    return (raw * valid).sum() / valid.sum().clamp(min=1e-8)

def run_validation(model, ema, val_loader, device, use_precomputed, max_batches=100):
    backup = ema.apply_to(model)
    model.eval()
    total_loss, total_steps = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            if use_precomputed:
                text_embeds = model.project_text_embeddings(batch["scibert_states"].to(device))
                text_pad = batch["text_padding_mask"].to(device)
            else:
                text_embeds = model.get_text_embeddings(
                    batch["text_input_ids"].to(device), batch["text_attention_mask"].to(device))
                text_pad = batch["text_padding_mask"].to(device)
            logits = model(input_ids, timesteps, text_embeds, text_pad)
            total_loss += diffusion_loss(logits, labels).item()
            total_steps += 1
    ema.restore(model, backup)
    model.train()
    return total_loss / max(total_steps, 1)

def generate_with_cfg(model, tokenizer, text_tokenizer, device,
                      prompts, cfg_scale=3.0, num_steps=50, temperature=1.0):
    model.eval()
    B = len(prompts)
    max_length = CONFIG["max_length"]
    if model.text_encoder is None:
        from transformers import AutoModel
        text_encoder = AutoModel.from_pretrained(CONFIG["text_model"]).to(device).eval()
    else:
        text_encoder = model.text_encoder
    text_enc = text_tokenizer(prompts, padding=True, truncation=True,
                               max_length=CONFIG["max_text_len"], return_tensors="pt").to(device)
    with torch.no_grad():
        scibert_states = text_encoder(input_ids=text_enc["input_ids"],
                                      attention_mask=text_enc["attention_mask"]).last_hidden_state
        text_embeds = model.project_text_embeddings(scibert_states)
    text_pad = (text_enc["attention_mask"] == 0)
    if model.text_encoder is None:
        del text_encoder; torch.cuda.empty_cache()
    input_ids = torch.full((B, max_length), tokenizer.mask_token_id, dtype=torch.long, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)
    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(B).unsqueeze(-1)
            logits_c = model(input_ids, step_t, text_embeds, text_pad)
            logits_u = model(input_ids, step_t, None, None)
            logits = logits_u + cfg_scale * (logits_c - logits_u)
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
    model.train()
    return input_ids

def generate_and_log(model, ema, tokenizer, text_tokenizer, device, step, val_df=None):
    if val_df is not None and len(val_df) >= CONFIG["gen_num_mols"]:
        sample_df = val_df.sample(n=CONFIG["gen_num_mols"], random_state=step)
        prompts, gt_selfies = sample_df["description"].tolist(), sample_df["selfies"].tolist()
    else:
        prompts = ["The molecule is a member of the class of pyrimidines.",
                    "The molecule is a primary alcohol with a hydroxyl group.",
                    "The molecule contains a benzene ring with a carboxylic acid group.",
                    "The molecule is a simple amino acid."][:CONFIG["gen_num_mols"]]
        gt_selfies = [None] * len(prompts)
    backup = ema.apply_to(model)
    raw_ids = generate_with_cfg(model, tokenizer, text_tokenizer, device,
                                 prompts, cfg_scale=CONFIG["cfg_scale"],
                                 num_steps=CONFIG["gen_steps"], temperature=CONFIG["gen_temperature"])
    ema.restore(model, backup)
    print(f"\n{'='*70}\n  GENERATED MOLECULES (EMA) — Step {step}\n{'='*70}")
    valid_count = 0
    for i in range(len(prompts)):
        ids = raw_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids: ids = ids[:ids.index(tokenizer.eos_token_id)]
        selfies_str = tokenizer.decode(ids)
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                valid_count += 1
                print(f"  [{i}] ✅ LogP={Descriptors.MolLogP(mol):+.2f}  Heavy={mol.GetNumHeavyAtoms():>2}  {canonical[:50]}")
            else: print(f"  [{i}] ❌ RDKit rejected: {smiles[:50]}")
        except Exception as e: print(f"  [{i}] ❌ Error: {e}")
        print(f"       Prompt: {prompts[i][:70]}{'...' if len(prompts[i])>70 else ''}")
        if gt_selfies[i]:
            try:
                gt_mol = Chem.MolFromSmiles(sf.decoder(gt_selfies[i]))
                if gt_mol: print(f"       GT:     {Chem.MolToSmiles(gt_mol)[:50]}")
            except: pass
        print()
    print(f"  Valid: {valid_count}/{len(prompts)}\n{'='*70}\n")

def load_pretrained(model, checkpoint_path, device):
    print(f"Loading pretrained checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  Trained for {ckpt.get('step','?'):,} steps (epoch {ckpt.get('epoch','?')})")
    print(f"  Loaded: {len(ckpt['model'])-len(unexpected)}  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    cross = [k for k in ckpt["model"] if "cross_attention" in k and k not in unexpected]
    print(f"  Cross-attn loaded: {len(cross)}  Text/proj new: {len([k for k in missing if 'text_' in k or 'null_token' in k])}")
    return ckpt.get("config", {})

def train(args):
    random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"]); torch.cuda.manual_seed_all(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    checkpoint_dir = Path(CONFIG["checkpoint_dir"]); checkpoint_dir.mkdir(exist_ok=True)
    mol_tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
    data_dir = Path(CONFIG["data_dir"])
    print("Loading CheBI-20 data...")
    train_df = pd.read_csv(data_dir / "chebi20_train.csv")
    val_df = pd.read_csv(data_dir / "chebi20_val.csv")
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    use_precomputed = CONFIG["use_precomputed"]
    train_scibert, val_scibert = None, None
    if use_precomputed:
        print("Loading pre-computed SciBERT embeddings...")
        train_scibert = torch.load(data_dir / "chebi20_train_scibert.pt", weights_only=False)
        val_scibert = torch.load(data_dir / "chebi20_val_scibert.pt", weights_only=False)
        print(f"  Train: {len(train_scibert):,}  Val: {len(val_scibert):,}  SciBERT OFF GPU ✅")
    train_dataset = TextConditionedDataset(train_df, mol_tokenizer, scibert_states=train_scibert)
    val_dataset = TextConditionedDataset(val_df, mol_tokenizer, scibert_states=val_scibert)
    collator = TextDiffusionCollator(mol_tokenizer,
        text_tokenizer=text_tokenizer if not use_precomputed else None,
        max_text_len=CONFIG["max_text_len"], use_precomputed=use_precomputed)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, collate_fn=collator, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=collator, num_workers=CONFIG["num_workers"], pin_memory=True)
    print(f"\nBuilding model...")
    model = MolecularDiffusionModel(
        vocab_size=CONFIG["vocab_size"], hidden_size=CONFIG["hidden_size"],
        num_heads=CONFIG["num_heads"], ffn_dim=CONFIG["ffn_dim"],
        num_layers=CONFIG["num_layers"], max_length=CONFIG["max_length"],
        pad_token_id=mol_tokenizer.pad_token_id, text_model_name=CONFIG["text_model"],
        uncond_prob=CONFIG["uncond_prob"], dropout=CONFIG["dropout"],
        load_text_encoder=not use_precomputed).to(device)
    load_pretrained(model, args.checkpoint, device)
    model = torch.compile(model)
    ema = EMA(model, decay=CONFIG["ema_decay"])
    print(f"  EMA initialized (decay={CONFIG['ema_decay']}, {len(ema.shadow)} tensors)")
    acc_steps = CONFIG["gradient_accumulation"]
    steps_per_epoch = len(train_loader) // acc_steps
    total_steps = steps_per_epoch * CONFIG["num_epochs"]
    new_kw = ["cross_attention", "text_proj", "null_token", "norm_cross", "pos_embedding"]
    new_params, pre_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (new_params if any(k in name for k in new_kw) else pre_params).append(p)
    print(f"  New: {sum(p.numel() for p in new_params):,}  Pre: {sum(p.numel() for p in pre_params):,}")
    optimizer = AdamW([{"params": new_params, "lr": CONFIG["new_lr"]},
                        {"params": pre_params, "lr": CONFIG["pretrained_lr"]}],
                       weight_decay=CONFIG["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=CONFIG["warmup_steps"], num_training_steps=total_steps)
    print(f"  Steps/epoch: {steps_per_epoch}  Total: {total_steps}  Warmup: {CONFIG['warmup_steps']}")
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
                if use_precomputed:
                    text_embeds = model.project_text_embeddings(batch["scibert_states"].to(device))
                    text_pad = batch["text_padding_mask"].to(device)
                else:
                    text_embeds = model.get_text_embeddings(
                        batch["text_input_ids"].to(device), batch["text_attention_mask"].to(device))
                    text_pad = batch["text_padding_mask"].to(device)
                logits = model(input_ids, timesteps, text_embeds, text_pad)
                loss = diffusion_loss(logits, labels) / acc_steps
            loss.backward()
            if (step_idx + 1) % acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], CONFIG["max_grad_norm"])
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                ema.update(model)
                global_step += 1
                if global_step % CONFIG["log_every"] == 0:
                    lr_n, lr_p = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
                    print(f"Step {global_step:>6} | loss: {loss.item()*acc_steps:.4f} | "
                          f"grad: {grad_norm:.4f} | lr_new: {lr_n:.2e} | lr_pre: {lr_p:.2e}")
                    wandb.log({"train/loss": loss.item()*acc_steps, "train/grad_norm": float(grad_norm),
                               "train/lr_new": lr_n, "train/lr_pre": lr_p,
                               "train/epoch": epoch + step_idx/len(train_loader)}, step=global_step)
                if global_step % CONFIG["val_every"] == 0:
                    val_loss = run_validation(model, ema, val_loader, device, use_precomputed)
                    print(f"  → Val loss (EMA): {val_loss:.4f}")
                    wandb.log({"val/loss_ema": val_loss}, step=global_step)
                    generate_and_log(model, ema, mol_tokenizer, text_tokenizer, device, global_step, val_df)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        backup = ema.apply_to(model)
                        torch.save({"step": global_step, "epoch": epoch+1,
                                    "model": model.state_dict(), "val_loss": val_loss,
                                    "config": CONFIG}, checkpoint_dir / "best_finetuned_model.pt")
                        ema.restore(model, backup)
                        print(f"  ✅ New best EMA model (val_loss: {val_loss:.4f})")
                if global_step % CONFIG["save_every"] == 0:
                    torch.save({"step": global_step, "epoch": epoch+1,
                                "model": model.state_dict(), "ema": ema.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "config": CONFIG}, checkpoint_dir / f"finetune_step{global_step}.pt")
                    print(f"  💾 Checkpoint saved at step {global_step}")
    print(f"\nTraining complete. Best val loss (EMA): {best_val_loss:.4f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CONFIG["pretrained_checkpoint"])
    parser.add_argument("--new_lr", type=float, default=None)
    parser.add_argument("--pretrained_lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    if args.new_lr: CONFIG["new_lr"] = args.new_lr
    if args.pretrained_lr: CONFIG["pretrained_lr"] = args.pretrained_lr
    if args.epochs: CONFIG["num_epochs"] = args.epochs
    if args.batch_size: CONFIG["batch_size"] = args.batch_size
    train(args)