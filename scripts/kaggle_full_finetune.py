"""
kaggle_full_finetune.py
-----------------------
Fresh fine-tune of the 27M Morpheus model on ChEBI-20 from the pretrained
zinc_17M_pretrain.pt checkpoint. Self-contained for Kaggle T4.

Paste this file into a Kaggle notebook cell and run:
    !python kaggle_full_finetune.py
    !python kaggle_full_finetune.py --resume   # auto-resume from latest epoch ckpt

Kaggle dataset expected at /kaggle/input/datasets/ift6390robli/morpheus-molgen/
containing:
  - zinc_17M_pretrain.pt          (pretrain checkpoint, loaded strict=False)
  - chemical_tokenizer.json
  - train.csv   (columns: prompt, response)
  - val.csv     (columns: prompt, response)

Outputs written to /kaggle/working/
  - best_finetuned_eos_correct.pt      (best val-loss model)
  - molinst_27M_frozen_epoch{N}.pt     (full state per epoch, for resume)
  - finetune_full_log.csv
  - plots_full/loss_curves.png, lr_schedule.png
  - finetune_full_samples.txt
"""

import argparse, csv, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("[wandb] not installed — W&B logging disabled")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR  = Path("/kaggle/input/datasets/ift6390robli/morpheus-molgen")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_CKPT  = INPUT_DIR  / "zinc_17M_pretrain.pt"
TOKENIZER_PATH = INPUT_DIR  / "chemical_tokenizer.json"
TRAIN_CSV      = INPUT_DIR  / "train.csv"
VAL_CSV        = INPUT_DIR  / "val.csv"

CKPT_SAVE      = OUTPUT_DIR / "molinst_27M_frozen.pt"
LOG_PATH       = OUTPUT_DIR / "finetune_full_log.csv"
PLOT_DIR       = OUTPUT_DIR / "plots_full"
SAMPLES_PATH   = OUTPUT_DIR / "finetune_full_samples.txt"

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "vocab_size"  : 110,
    "hidden_size" : 512,
    "num_heads"   : 8,
    "ffn_dim"     : 1024,
    "num_layers"  : 8,
    "max_length"  : 74,
    "dropout"     : 0.1,
    "text_model"  : "BAAI/bge-large-en-v1.5",
    "uncond_prob" : 0.1,

    "batch_size"      : 64,
    "learning_rate"   : 2e-4,
    "weight_decay"    : 0.01,
    "max_grad_norm"   : 1.0,
    "num_epochs"      : 5,
    "warmup_steps"    : 1000,

    "eos_weight"  : 5.0,
    "pad_weight"  : 0.05,

    "num_workers"     : 2,
    "seed"            : 42,

    "log_every"       : 50,
    "val_every"       : 1500,
    "val_max_batches" : 200,
    "early_stop_patience": 5,

    "gen_steps"       : 32,
    "gen_temperature" : 1.2,

    "freeze_text_encoder": True,
    "wandb_project"   : "morpheus-diffusion-finetune",
}

# =============================================================================
# TOKENIZER  (inlined from tokenizer/chemicalTokenizer.py)
# =============================================================================
class ChemicalTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, "r") as f:
            data = json.load(f)
        self.token_to_id       = data["token_to_id"]
        self.id_to_token       = {int(k): v for k, v in data["id_to_token"].items()}
        self.pad_token_id      = data["pad_token_id"]
        self.mask_token_id     = data["mask_token_id"]
        self.eos_token_id      = data["eos_token_id"]
        self.vocab_size        = data["vocab_size"]
        self.chemical_start_id = data["chemical_start_id"]

    def encode(self, selfies_str):
        tokens = list(sf.split_selfies(selfies_str))
        return [self.token_to_id.get(t, self.pad_token_id) for t in tokens]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids
                  if i not in (self.pad_token_id, self.mask_token_id, self.eos_token_id)]
        return "".join(tokens)

# =============================================================================
# EMBEDDINGS  (inlined from model/embeddings.py)
# =============================================================================
class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def sinusoidal_features(self, t):
        device = t.device
        half   = self.hidden_size // 2
        freqs  = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1))
        )
        angles = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, t):
        return self.mlp(self.sinusoidal_features(t))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)

# =============================================================================
# TRANSFORMER BLOCK  (inlined from model/transformerBlock.py)
# =============================================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj    = nn.Linear(hidden_size, hidden_size)
        self.k_proj    = nn.Linear(hidden_size, hidden_size)
        self.v_proj    = nn.Linear(hidden_size, hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.scale     = math.sqrt(self.head_dim)

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, L), dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, -1))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj    = nn.Linear(hidden_size, hidden_size)
        self.k_proj    = nn.Linear(hidden_size, hidden_size)
        self.v_proj    = nn.Linear(hidden_size, hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_embeds, text_padding_mask=None):
        B, L, _   = x.shape
        Bt, Lt, _ = text_embeds.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(text_embeds).view(Bt, Lt, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(text_embeds).view(Bt, Lt, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if text_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, Lt), dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(text_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, -1))


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1           = nn.LayerNorm(hidden_size)
        self.attention       = SelfAttention(hidden_size, num_heads, dropout)
        self.norm_cross      = nn.LayerNorm(hidden_size)
        self.cross_attention = CrossAttention(hidden_size, num_heads, dropout)
        self.norm2           = nn.LayerNorm(hidden_size)
        self.ffn             = FeedForward(hidden_size, ffn_dim, dropout)
        self.dropout         = nn.Dropout(dropout)

    def forward(self, x, text_embeds, padding_mask=None, text_padding_mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), padding_mask))
        x = x + self.dropout(self.cross_attention(self.norm_cross(x), text_embeds, text_padding_mask))
        x = x + self.ffn(self.norm2(x))
        return x

# =============================================================================
# MODEL  (inlined from model/molecularDiffusionModel.py)
# =============================================================================
class MolecularDiffusionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, ffn_dim, num_layers,
                 max_length, pad_token_id, text_model_name="BAAI/bge-large-en-v1.5",
                 uncond_prob=0.1, dropout=0.1):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.uncond_prob  = uncond_prob

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.token_embedding    = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embedding      = PositionalEmbedding(max_length, hidden_size)
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm         = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith("text_encoder"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        for block in self.blocks:
            nn.init.zeros_(block.cross_attention.out_proj.weight)
            nn.init.zeros_(block.cross_attention.out_proj.bias)

    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            out = self.text_encoder(input_ids=text_input_ids,
                                    attention_mask=text_attention_mask)
        return self.text_proj(out.last_hidden_state)

    def forward(self, input_ids, timesteps, text_embeds, text_padding_mask=None):
        B, L = input_ids.shape
        if self.training:
            keep = (torch.rand(B, device=input_ids.device) > self.uncond_prob).view(B, 1, 1)
            null = self.null_token.expand(B, text_embeds.size(1), -1)
            text_embeds = torch.where(keep, text_embeds, null)

        pad_mask = (input_ids == self.pad_token_id)
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding(L, input_ids.device)
        x = x + self.timestep_embedding(timesteps).unsqueeze(1)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x, text_embeds, pad_mask, text_padding_mask)
        return self.lm_head(self.output_norm(x))

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# =============================================================================
# DATASET  (inlined from finetune/conditionalSELFIESDataset.py)
# =============================================================================
class ConditionalSELFIESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=74):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row         = self.df.iloc[idx]
        selfies_str = str(row["response"])
        prompt      = str(row.get("prompt", "A chemical molecule"))
        token_ids   = self.tokenizer.encode(selfies_str)[:(self.max_length - 1)]
        token_ids   = token_ids + [self.tokenizer.eos_token_id]
        pad_len     = self.max_length - len(token_ids)
        token_ids   = token_ids + [self.tokenizer.pad_token_id] * pad_len
        return {"input_ids": torch.tensor(token_ids, dtype=torch.long), "prompt": prompt}

# =============================================================================
# COLLATOR  (inlined from finetune/diffusionCollatorPrompt.py)
# =============================================================================
class ConditionalDiffusionCollator:
    def __init__(self, tokenizer, hf_tokenizer):
        self.tokenizer    = tokenizer
        self.hf_tokenizer = hf_tokenizer
        self.mask_id      = tokenizer.mask_token_id
        self.pad_id       = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        prompts   = [f.get("prompt", "A chemical molecule") for f in features]

        text_inputs = self.hf_tokenizer(
            prompts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        labels      = input_ids.clone()
        t           = torch.rand(input_ids.shape[0])
        alpha_t     = torch.cos(t * math.pi / 2) ** 2
        noise_probs = torch.rand_like(input_ids.float())
        mask_map    = noise_probs > alpha_t.unsqueeze(-1)

        input_ids[mask_map] = self.mask_id
        labels[~mask_map]   = -100

        return {
            "input_ids":           input_ids,
            "labels":              labels,
            "timesteps":           t.unsqueeze(-1),      # [B, 1]
            "text_input_ids":      text_inputs["input_ids"],
            "text_attention_mask": text_inputs["attention_mask"],
        }

# =============================================================================
# LOSS  (eos_weight=5.0, pad_weight=0.05 from the start)
# =============================================================================
def diffusion_loss(logits, labels, timesteps,
                   pad_token_id=0, eos_token_id=2,
                   pad_weight=0.05, eos_weight=5.0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights               = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id] = pad_weight
    vocab_weights[eos_token_id] = eos_weight

    raw_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction="none",
    )
    # timesteps: [B, 1] → expand to [B, seq_len, 1] → [B*seq_len]
    weights = (1.0 - timesteps).unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)
    valid   = (labels.view(-1) != -100).float()
    return (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)

# =============================================================================
# RESUME: find latest epoch checkpoint in OUTPUT_DIR
# =============================================================================
def find_latest_epoch_ckpt():
    """Return (path, epoch_num) of the highest-numbered molinst_27M_frozen_epoch*.pt, or (None, 0)."""
    candidates = sorted(OUTPUT_DIR.glob("molinst_27M_frozen_epoch*.pt"))
    if not candidates:
        return None, 0
    latest = candidates[-1]
    try:
        epoch_num = int(latest.stem.split("_")[-1])
    except ValueError:
        epoch_num = 0
    return latest, epoch_num

# =============================================================================
# VALIDATION
# =============================================================================
def run_validation(model, val_loader, device, tokenizer):
    model.eval()
    total_loss, n = 0.0, 0
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
            total_loss += loss.item()
            n          += 1
            if n >= CONFIG["val_max_batches"]:
                break
    model.train()
    return total_loss / max(n, 1)

# =============================================================================
# GENERATION (mid-training CFG samples)
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

        for si, t_val in enumerate(t_vals):
            step_t        = t_val.repeat(num_mols).unsqueeze(-1)
            cond_logits   = model(input_ids, step_t, text_embeds,  text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds,  text_padding_mask)
            logits        = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            if si < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")
            probs      = torch.softmax(logits / temperature, dim=-1)
            sampled    = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            alpha_t    = (torch.cos(t_val * math.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)
            if num_to_mask > 0 and si < num_steps - 1:
                _, mi = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mi, tokenizer.mask_token_id)
            input_ids = sampled

    print(f"\n{'='*60}\n  CFG SAMPLES — Step {step}\n{'='*60}")
    lengths = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        lengths.append(len(ids))
        sel = tokenizer.decode(ids)
        try:
            mol = Chem.MolFromSmiles(sf.decoder(sel))
            status = f"✓ {Chem.MolToSmiles(mol)[:55]}" if mol else "✗ invalid"
        except Exception:
            status = "✗ error"
        print(f"  [{i}] {len(ids)} tok | {prompts[i][:40]}\n      {status}")
    print(f"  Avg tok len: {sum(lengths)/len(lengths):.1f}  (target ~43)\n{'='*60}\n")
    model.train()

# =============================================================================
# CFG QUALITATIVE SAMPLES (end-of-training)
# =============================================================================
def save_cfg_samples(model, tokenizer, hf_tokenizer, device):
    model.eval()
    max_length  = CONFIG["max_length"]
    cfg_scale   = 3.0
    num_steps   = 32
    temperature = 1.2

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

        for si, t_val in enumerate(t_vals):
            step_t        = t_val.repeat(num_mols).unsqueeze(-1)
            cond_logits   = model(input_ids, step_t, text_embeds,  text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds,  text_padding_mask)
            logits        = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            if si < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")
            probs      = torch.softmax(logits / temperature, dim=-1)
            sampled    = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            alpha_t    = (torch.cos(t_val * math.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)
            if num_to_mask > 0 and si < num_steps - 1:
                _, mi = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mi, tokenizer.mask_token_id)
            input_ids = sampled

    results = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        sel   = tokenizer.decode(ids)
        entry = {"prompt": prompts[i], "selfies": sel, "token_count": len(ids),
                 "valid": False, "smiles": "", "mw": None, "qed": None}
        try:
            mol = Chem.MolFromSmiles(sf.decoder(sel))
            if mol:
                entry.update({
                    "valid": True,
                    "smiles": Chem.MolToSmiles(mol),
                    "mw": round(Descriptors.MolWt(mol), 1),
                    "qed": round(QED.qed(mol), 3),
                })
        except Exception:
            pass
        results.append(entry)

    unique_prompts = list(dict.fromkeys(prompts))
    valid_count    = sum(r["valid"] for r in results)
    avg_tok        = sum(r["token_count"] for r in results) / len(results)

    with open(SAMPLES_PATH, "w") as f:
        f.write("CFG GENERATION SAMPLES — Full Fresh Fine-tune (eos_weight=5.0)\n")
        f.write(f"cfg_scale={cfg_scale}  temperature={temperature}  steps={num_steps}\n")
        f.write(f"Valid: {valid_count}/{num_mols}  |  Avg token length: {avg_tok:.1f} (target ~43)\n")
        f.write("=" * 70 + "\n\n")
        for prompt in unique_prompts:
            f.write(f"PROMPT: {prompt}\n" + "-" * 70 + "\n")
            for j, r in enumerate([r for r in results if r["prompt"] == prompt]):
                status = "VALID" if r["valid"] else "INVALID"
                f.write(f"  [{j+1}] {status}  ({r['token_count']} tokens)\n")
                if r["valid"]:
                    f.write(f"       SMILES : {r['smiles']}\n")
                    f.write(f"       MW={r['mw']}  QED={r['qed']}\n")
                else:
                    f.write(f"       SELFIES: {r['selfies'][:80]}\n")
            f.write("\n")

    print(f"\n{'='*70}\nValid: {valid_count}/{num_mols}  |  Avg tok: {avg_tok:.1f} (target ~43)")
    for prompt in unique_prompts:
        print(f"\n{prompt}")
        for r in [r for r in results if r["prompt"] == prompt]:
            tag  = "✓" if r["valid"] else "✗"
            body = r["smiles"] if r["valid"] else r["selfies"]
            extra = f"  MW={r['mw']} QED={r['qed']}" if r["valid"] else ""
            print(f"  {tag} {r['token_count']} tok | {body[:60]}{extra}")
    print(f"\nSaved → {SAMPLES_PATH}")
    model.train()

# =============================================================================
# PLOTS
# =============================================================================
def save_plots():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOG_PATH)
    if df.empty:
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
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Full Fine-tune (eos_weight=5.0) — Loss Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_curves.png", dpi=300)
    plt.close(fig)

    if not train_rows.empty and "lr" in train_rows.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_rows["step"], train_rows["lr"], color="#4CAF50", **style)
        ax.set_xlabel("Step"); ax.set_ylabel("LR")
        ax.set_title("LR Schedule")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "lr_schedule.png", dpi=300)
        plt.close(fig)

    print(f"[plot] Saved → {PLOT_DIR}/")

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train(do_resume: bool):
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- Tokenizers ---
    tokenizer    = ChemicalTokenizer(TOKENIZER_PATH)
    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    # --- Data ---
    print(f"[data] Loading {TRAIN_CSV} ...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    if "prompt" not in train_df.columns: train_df["prompt"] = "A chemical molecule"
    if "prompt" not in val_df.columns:   val_df["prompt"]   = "A chemical molecule"

    train_dataset = ConditionalSELFIESDataset(train_df, tokenizer)
    val_dataset   = ConditionalSELFIESDataset(val_df,   tokenizer)
    train_size    = len(train_dataset)
    print(f"[data] train={train_size:,}  val={len(val_dataset):,}")

    collator     = ConditionalDiffusionCollator(tokenizer, hf_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True,  collate_fn=collator,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False, collate_fn=collator,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

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

    # --- Optimizer & scheduler (built before loading state dicts) ---
    total_steps = (train_size // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    optimizer   = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )

    # --- Checkpoint loading ---
    start_epoch  = 0
    global_step  = 0
    best_val_loss = float("inf")
    last_val_loss = float("inf")

    resume_path, resume_epoch = find_latest_epoch_ckpt()

    if do_resume and resume_path is not None:
        # ── Resume from latest epoch checkpoint ──────────────────────────────
        print(f"[resume] Loading epoch checkpoint: {resume_path.name}")
        ckpt = torch.load(resume_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"]          # resume from the NEXT epoch
        global_step   = ckpt["step"]
        last_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_loss = ckpt.get("best_val_loss", last_val_loss)
        print(f"  Resuming from epoch {start_epoch}, step {global_step}, "
              f"best_val_loss={best_val_loss:.4f}")
        if missing:    print(f"  Missing keys  : {len(missing)}")
        if unexpected: print(f"  Unexpected keys: {len(unexpected)}")
    elif do_resume and resume_path is None:
        print("[resume] No epoch checkpoints found in /kaggle/working/ — "
              "starting fresh from pretrain checkpoint.")
        do_resume = False  # fall through to fresh load

    if not do_resume:
        # ── Fresh fine-tune from pretrain checkpoint ──────────────────────────
        print(f"[ckpt] Loading pretrain checkpoint: {PRETRAIN_CKPT} ...")
        state = torch.load(PRETRAIN_CKPT, map_location=device)
        # Support both raw state-dict and wrapped {'model_state_dict': ...} formats
        if isinstance(state, dict):
            state = (state.get("model_state_dict")
                     or state.get("model")
                     or state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys  : {len(missing)}  (new params — will train from init)")
        print(f"  Unexpected keys: {len(unexpected)}")

    # --- Freeze text encoder ---
    for p in model.text_encoder.parameters():
        p.requires_grad = not CONFIG["freeze_text_encoder"]
    total_params, trainable_params = model.count_parameters()
    print(f"[model] {total_params/1e6:.1f}M total | {trainable_params/1e6:.1f}M trainable")
    print(f"[sched] total_steps={total_steps:,}  warmup={CONFIG['warmup_steps']}  "
          f"lr={CONFIG['learning_rate']}")
    print(f"[loss]  eos_weight={CONFIG['eos_weight']}  pad_weight={CONFIG['pad_weight']}")

    # --- W&B ---
    if USE_WANDB:
        run_name = f"full-finetune-resume-e{start_epoch}" if start_epoch else "full-finetune-fresh"
        run = wandb.init(project=CONFIG["wandb_project"], name=run_name,
                         config=CONFIG, reinit=True,
                         settings=wandb.Settings(start_method="thread"))
        print(f"[wandb] {run.url}")

    # --- CSV log (append if resuming) ---
    log_columns = ["epoch", "step", "train_loss", "val_loss", "lr",
                   "tokens_per_sec", "elapsed_min"]
    log_mode    = "a" if (start_epoch > 0 and LOG_PATH.exists()) else "w"
    log_file    = open(LOG_PATH, log_mode, newline="")
    log_writer  = csv.DictWriter(log_file, fieldnames=log_columns)
    if log_mode == "w":
        log_writer.writeheader()
    log_file.flush()

    patience_counter   = 0
    stopped_early      = False
    t_start            = time.time()
    tokens_per_sec_ema = None

    model.train()

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        print(f"\n{'─'*65}\n  Epoch {epoch+1}/{CONFIG['num_epochs']}\n{'─'*65}")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for batch in progress_bar:
            t0                  = time.time()
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

            step_time = max(time.time() - t0, 1e-6)
            tps = input_ids.numel() / step_time
            tokens_per_sec_ema = (tps if tokens_per_sec_ema is None
                                  else 0.05 * tps + 0.95 * tokens_per_sec_ema)

            if global_step % CONFIG["log_every"] == 0:
                lr          = scheduler.get_last_lr()[0]
                elapsed_min = (time.time() - t_start) / 60
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                if USE_WANDB:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)
                log_writer.writerow({
                    "epoch": epoch+1, "step": global_step,
                    "train_loss": round(loss.item(), 6), "val_loss": "",
                    "lr": round(lr, 8),
                    "tokens_per_sec": round(tokens_per_sec_ema, 1),
                    "elapsed_min": round(elapsed_min, 2),
                })
                log_file.flush()

            if global_step % CONFIG["val_every"] == 0:
                val_loss      = run_validation(model, val_loader, device, tokenizer)
                last_val_loss = val_loss
                elapsed_min   = (time.time() - t_start) / 60
                print(f"  → Val loss: {val_loss:.4f}")
                if USE_WANDB:
                    wandb.log({"val/loss": val_loss}, step=global_step)
                log_writer.writerow({
                    "epoch": epoch+1, "step": global_step,
                    "train_loss": "", "val_loss": round(val_loss, 6),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "tokens_per_sec": round(tokens_per_sec_ema or 0, 1),
                    "elapsed_min": round(elapsed_min, 2),
                })
                log_file.flush()

                generate_samples(model, tokenizer, hf_tokenizer, device, global_step)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    torch.save({
                        "epoch"         : epoch + 1,
                        "step"          : global_step,
                        "model"         : model.state_dict(),
                        "optimizer"     : optimizer.state_dict(),
                        "scheduler"     : scheduler.state_dict(),
                        "val_loss"      : val_loss,
                        "best_val_loss" : best_val_loss,
                        "config"        : CONFIG,
                    }, CKPT_SAVE)
                    print(f"  ✓ New best → {CKPT_SAVE.name}  (val_loss={val_loss:.4f})")
                    if USE_WANDB:
                        artifact = wandb.Artifact(
                            name=f"full-finetune-valloss{val_loss:.4f}", type="model",
                            metadata={"val_loss": val_loss, "step": global_step},
                        )
                        artifact.add_file(str(CKPT_SAVE))
                        wandb.log_artifact(artifact)
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{CONFIG['early_stop_patience']})")
                    if patience_counter >= CONFIG["early_stop_patience"]:
                        print("Early stopping triggered.")
                        stopped_early = True
                        break

        if stopped_early:
            break

        # ── Periodic checkpoint every epoch (enables Kaggle disconnect resume) ──
        epoch_ckpt_path = OUTPUT_DIR / f"molinst_27M_frozen_epoch{epoch+1}.pt"
        torch.save({
            "epoch"         : epoch + 1,
            "step"          : global_step,
            "model"         : model.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            "val_loss"      : last_val_loss,
            "best_val_loss" : best_val_loss,
            "config"        : CONFIG,
        }, epoch_ckpt_path)
        print(f"  Epoch checkpoint → {epoch_ckpt_path.name}  "
              f"(val_loss={last_val_loss:.4f})")

    log_file.close()
    print(f"\nTraining complete.{' (early stop)' if stopped_early else ''}")
    print(f"Best val loss: {best_val_loss:.4f}")
    save_plots()
    save_cfg_samples(model, tokenizer, hf_tokenizer, device)
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest molinst_27M_frozen_epoch*.pt in /kaggle/working/")
    args = parser.parse_args()
    train(do_resume=args.resume)
