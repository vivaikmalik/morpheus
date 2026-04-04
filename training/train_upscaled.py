"""
training/train_upscaled.py
--------------------------
Stage 1 pretraining of the upscaled (~27M param, Config C) diffusion model
on ZINC250K. Unconditional — no text conditioning.

The checkpoint format is compatible with finetune/train_text_condition.py:
load with strict=False and the cross-attention / text-projector layers will
be kept at their zero-initialized values, ready for Stage 2 fine-tuning.

Usage:
    python training/train_upscaled.py

To switch to batch_size=64 if you hit OOM on 16 GB:
    edit CONFIG["batch_size"] = 64
"""

import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding, PositionalEmbedding
from model.transformerBlock import SelfAttention, FeedForward
from tokenizer.chemicalTokenizer import ChemicalTokenizer

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # Model (Config C — ~27M trainable params)
    "vocab_size"  : 110,
    "hidden_size" : 512,
    "num_heads"   : 8,       # head_dim = 64, power-of-2, MPS-efficient
    "ffn_dim"     : 1024,
    "num_layers"  : 8,
    "max_length"  : 74,      # 73 content tokens + 1 EOS
    "dropout"     : 0.1,

    # Training
    "batch_size"      : 128,   # reduce to 64 if 16 GB MPS hits OOM
    "learning_rate"   : 4e-4,
    "weight_decay"    : 0.01,
    "max_grad_norm"   : 1.0,
    "num_epochs"      : 15,
    "warmup_steps"    : 1000,

    # Loss token weights
    "eos_weight"      : 5.0,   # emphasise learning sequence termination
    "pad_weight"      : 0.05,

    # Data
    "val_fraction"    : 0.10,
    "num_workers"     : 0,
    "seed"            : 42,

    # Logging
    "log_every"       : 100,   # print + write train_loss every N steps

    # Paths
    "project_root"    : str(Path(__file__).parent.parent),
    "checkpoint_dir"  : str(Path(__file__).parent.parent / "checkpoints"),
    "log_path"        : str(Path(__file__).parent.parent / "outputs" / "pretrain_upscaled_log.csv"),
    "plot_dir"        : str(Path(__file__).parent.parent / "outputs" / "plots" / "pretrain_upscaled"),
}

# =============================================================================
# MODEL
# =============================================================================

class PretrainBlock(nn.Module):
    """
    Self-attention + FFN block with NO cross-attention.

    Module names match the corresponding components inside TransformerBlock
    (norm1, attention, norm2, ffn) so that strict=False loading in the
    fine-tuning script correctly populates all shared weights.

    Keys intentionally absent here (kept at init by finetune's strict=False):
        blocks.{i}.norm_cross.*
        blocks.{i}.cross_attention.*
    """

    def __init__(self, hidden_size: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, num_heads, dropout)
        self.norm2     = nn.LayerNorm(hidden_size)
        self.ffn       = FeedForward(hidden_size, ffn_dim, dropout)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attention(x, padding_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class PretrainDiffusionModel(nn.Module):
    """
    Unconditional masked-diffusion model for SELFIES.

    All module names are chosen to match their counterparts in
    MolecularDiffusionModel so that fine-tuning with strict=False works
    without any key renaming:

        token_embedding  ← identical
        pos_embedding    ← identical
        timestep_embedding ← identical
        input_norm       ← identical
        blocks.{i}.norm1 / .attention / .norm2 / .ffn ← identical
        output_norm      ← identical
        lm_head          ← identical (weight-tied with token_embedding)

    NOT present here (MolecularDiffusionModel adds them fresh at finetune):
        text_encoder, text_proj, null_token,
        blocks.{i}.norm_cross, blocks.{i}.cross_attention
    """

    def __init__(
        self,
        vocab_size:  int,
        hidden_size: int,
        num_heads:   int,
        ffn_dim:     int,
        num_layers:  int,
        max_length:  int,
        pad_token_id: int,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.pad_token_id = pad_token_id

        self.token_embedding    = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embedding      = PositionalEmbedding(max_length, hidden_size)
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm         = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            PretrainBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()
        # Weight tying: lm_head shares token_embedding weights (0 extra params)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        B, seq_len = input_ids.shape

        padding_mask = (input_ids == self.pad_token_id)

        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding(seq_len, device)
        x = x + self.timestep_embedding(timesteps).unsqueeze(1)
        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.output_norm(x)
        return self.lm_head(x)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# DATASET
# =============================================================================

class ZINCDataset(Dataset):
    """
    Loads ZINC250K from HuggingFace, converts SMILES → SELFIES,
    filters to molecules the tokenizer can handle, and stores
    padded + EOS-terminated token sequences.
    """

    def __init__(self, tokenizer: ChemicalTokenizer, max_length: int = 74):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []   # list of LongTensors, shape (max_length,)
        self._load_and_filter()

    def _load_and_filter(self):
        print("[data] Downloading ZINC250K from HuggingFace …")
        df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
        print(f"[data] Loaded {len(df):,} rows. Columns: {list(df.columns)}")

        # Prefer the pre-computed selfies column; only convert from SMILES as fallback
        if "selfies" in df.columns:
            smiles_list  = None
            selfies_list = df["selfies"].tolist()
            print("[data] Using pre-computed SELFIES column.")
        elif "smiles" in df.columns:
            smiles_list  = df["smiles"].tolist()
            selfies_list = None
            print("[data] Converting SMILES → SELFIES …")
        else:
            raise ValueError(f"Expected 'selfies' or 'smiles' column, got: {list(df.columns)}")

        vocab         = set(self.tokenizer.token_to_id.keys())
        max_content   = self.max_length - 1   # reserve 1 slot for EOS
        eos_id        = self.tokenizer.eos_token_id
        pad_id        = self.tokenizer.pad_token_id

        n_fail_conv   = 0
        n_fail_vocab  = 0
        n_fail_length = 0

        total = len(df)
        for i in range(total):
            if smiles_list is not None:
                try:
                    selfies_str = sf.encoder(smiles_list[i].strip())
                except Exception:
                    n_fail_conv += 1
                    continue
            else:
                selfies_str = selfies_list[i].strip()

            try:
                tokens = list(sf.split_selfies(selfies_str))
            except Exception:
                n_fail_conv += 1
                continue

            if any(t not in vocab for t in tokens):
                n_fail_vocab += 1
                continue

            if len(tokens) > max_content:
                n_fail_length += 1
                continue

            # Encode: chemical tokens + EOS + PAD
            ids = self.tokenizer.encode(selfies_str)   # no EOS from tokenizer
            ids = ids[:max_content] + [eos_id]          # append EOS
            pad_len = self.max_length - len(ids)
            ids = ids + [pad_id] * pad_len

            self.samples.append(torch.tensor(ids, dtype=torch.long))

        print(f"[data] Filter summary:")
        print(f"  Total raw                : {total:>8,}")
        print(f"  Failed SMILES→SELFIES    : {n_fail_conv:>8,}")
        print(f"  Unknown vocab tokens     : {n_fail_vocab:>8,}")
        print(f"  Exceeded max length      : {n_fail_length:>8,}")
        print(f"  Kept                     : {len(self.samples):>8,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return {"input_ids": self.samples[idx]}


# =============================================================================
# COLLATOR
# =============================================================================

class DiffusionCollator:
    """
    Applies cosine-schedule masking for unconditional masked diffusion.
    Returns input_ids (masked), labels (targets at masked positions), timesteps.
    """

    def __init__(self, tokenizer: ChemicalTokenizer):
        self.mask_id = tokenizer.mask_token_id
        self.pad_id  = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])   # (B, L)
        labels    = input_ids.clone()

        t       = torch.rand(input_ids.shape[0])                       # (B,)
        alpha_t = torch.cos(t * math.pi / 2) ** 2                     # (B,)

        noise_probs = torch.rand_like(input_ids.float())
        mask_map    = noise_probs > alpha_t.unsqueeze(-1)              # (B, L)

        # Don't mask PAD positions — they should stay as PAD
        pad_positions      = (input_ids == self.pad_id)
        mask_map[pad_positions] = False

        input_ids[mask_map] = self.mask_id
        labels[~mask_map]   = -100   # only supervise masked positions

        return {
            "input_ids": input_ids,
            "labels":    labels,
            "timesteps": t.unsqueeze(-1),   # (B, 1)
        }


# =============================================================================
# LOSS
# =============================================================================

def diffusion_loss(logits, labels, timesteps, pad_id=0, eos_id=2,
                   pad_weight=0.05, eos_weight=5.0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights              = torch.ones(vocab_size, device=device)
    vocab_weights[pad_id]      = pad_weight
    vocab_weights[eos_id]      = eos_weight

    raw_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction="none",
    )   # (B*L,)

    # Weight by (1 - t): emphasise low-noise steps (t≈0 = almost clean)
    weights = (1.0 - timesteps)                          # (B, 1)
    weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)

    valid = (labels.view(-1) != -100).float()
    loss  = (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)
    return loss


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, use_amp=False):
    model.eval()
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(input_ids, timesteps)
                loss   = diffusion_loss(
                    logits, labels, timesteps,
                    pad_id=model.pad_token_id,
                    eos_id=CONFIG["vocab_size"] - 1,  # resolved properly below
                )

            total_loss += loss.item()
            n_steps    += 1

    model.train()
    return total_loss / max(n_steps, 1)


# =============================================================================
# GENERATION (unconditional, for sanity-checking during training)
# =============================================================================

def generate_samples(model, tokenizer, device, step, num_mols=6):
    model.eval()
    max_length  = CONFIG["max_length"]
    num_steps   = 32
    temperature = 1.2

    input_ids = torch.full(
        (num_mols, max_length), tokenizer.mask_token_id, device=device
    )
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(num_mols).unsqueeze(-1)
            logits = model(input_ids, step_t)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")

            probs      = torch.softmax(logits / temperature, dim=-1)
            sampled    = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            alpha_t     = (torch.cos(t_val * math.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    print(f"\n{'='*60}")
    print(f"  GENERATED MOLECULES — Step {step}")
    print(f"{'='*60}")
    valid_count = 0
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
                print(f"  [{i}] ✓ {Chem.MolToSmiles(mol)[:65]}")
            else:
                print(f"  [{i}] ✗ invalid structure")
        except Exception:
            print(f"  [{i}] ✗ decode error: {selfies_str[:40]}")
    print(f"\n  Valid: {valid_count}/{num_mols} ({100*valid_count/num_mols:.0f}%)")
    print(f"{'='*60}\n")
    model.train()


# =============================================================================
# PLOTS
# =============================================================================

def save_plots(log_path: str, plot_dir: str):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    if df.empty:
        print("[plot] Log file is empty, skipping plots.")
        return

    train_rows = df[df["train_loss"].notna()].copy()
    val_rows   = df[df["val_loss"].notna()].copy()

    style = {"linewidth": 1.5, "alpha": 0.9}
    dpi   = 300

    # --- a) Loss curves ---
    fig, ax = plt.subplots(figsize=(9, 5))
    if not train_rows.empty:
        ax.plot(train_rows["step"], train_rows["train_loss"],
                label="Train loss", color="#2196F3", **style)
    if not val_rows.empty:
        ax.plot(val_rows["step"], val_rows["val_loss"],
                label="Val loss", color="#F44336", linestyle="--", **style)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Pretraining Loss (27M model, ZINC250K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", dpi=dpi)
    plt.close(fig)

    # --- b) Learning rate schedule ---
    if not train_rows.empty and "lr" in train_rows.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_rows["step"], train_rows["lr"],
                color="#4CAF50", **style)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (cosine + warmup)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "lr_schedule.png", dpi=dpi)
        plt.close(fig)

    # --- c) Throughput ---
    if not train_rows.empty and "tokens_per_sec" in train_rows.columns:
        tps = train_rows["tokens_per_sec"].dropna()
        if not tps.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(train_rows.loc[tps.index, "step"], tps,
                    color="#FF9800", **style)
            ax.set_xlabel("Step")
            ax.set_ylabel("Tokens / sec")
            ax.set_title("Training Throughput")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "throughput.png", dpi=dpi)
            plt.close(fig)

    print(f"[plot] Saved 3 plots → {plot_dir}/")


# =============================================================================
# TRAINING
# =============================================================================

def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # --- Device ---
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        use_amp = True
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device  = torch.device("mps")
        use_amp = False   # torch.amp not supported on MPS
        print("[device] Apple MPS")
    else:
        device  = torch.device("cpu")
        use_amp = False
        print("[device] CPU")

    project_root   = Path(CONFIG["project_root"])
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    Path(CONFIG["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["plot_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = ChemicalTokenizer(project_root / "chemical_tokenizer.json")

    # --- Dataset ---
    full_dataset = ZINCDataset(tokenizer, max_length=CONFIG["max_length"])

    val_size   = int(len(full_dataset) * CONFIG["val_fraction"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"]),
    )
    print(f"[data] Train: {train_size:,} | Val: {val_size:,}")

    collator = DiffusionCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type != "cpu"),
    )

    # --- Model ---
    model = PretrainDiffusionModel(
        vocab_size   = CONFIG["vocab_size"],
        hidden_size  = CONFIG["hidden_size"],
        num_heads    = CONFIG["num_heads"],
        ffn_dim      = CONFIG["ffn_dim"],
        num_layers   = CONFIG["num_layers"],
        max_length   = CONFIG["max_length"],
        pad_token_id = tokenizer.pad_token_id,
        dropout      = CONFIG["dropout"],
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"[model] {total:,} total params | {trainable:,} trainable")
    print(f"[model] hidden={CONFIG['hidden_size']} | ffn={CONFIG['ffn_dim']} "
          f"| layers={CONFIG['num_layers']} | heads={CONFIG['num_heads']} "
          f"| head_dim={CONFIG['hidden_size']//CONFIG['num_heads']}")

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    steps_per_epoch = math.ceil(train_size / CONFIG["batch_size"])
    total_steps     = steps_per_epoch * CONFIG["num_epochs"]
    warmup_steps    = CONFIG["warmup_steps"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.GradScaler() if use_amp else None

    print(f"[sched] {total_steps:,} total steps | {warmup_steps} warmup "
          f"| {steps_per_epoch} steps/epoch")

    # --- CSV log ---
    log_path    = Path(CONFIG["log_path"])
    log_columns = ["epoch", "step", "train_loss", "val_loss", "lr",
                   "tokens_per_sec", "elapsed_min"]
    log_file  = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_columns)
    log_writer.writeheader()
    log_file.flush()

    # --- Training loop ---
    global_step   = 0
    best_val_loss = float("inf")
    t_start       = time.time()
    tokens_per_sec_ema = None   # exponential moving average

    model.train()

    checkpoint_config = {
        "vocab_size"  : CONFIG["vocab_size"],
        "hidden_size" : CONFIG["hidden_size"],
        "num_heads"   : CONFIG["num_heads"],
        "ffn_dim"     : CONFIG["ffn_dim"],
        "num_layers"  : CONFIG["num_layers"],
        "max_length"  : CONFIG["max_length"],
    }

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"{'─'*70}")

        for batch in train_loader:
            step_t0   = time.time()
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(input_ids, timesteps)
                    loss   = diffusion_loss(
                        logits, labels, timesteps,
                        pad_id=tokenizer.pad_token_id,
                        eos_id=tokenizer.eos_token_id,
                        pad_weight=CONFIG["pad_weight"],
                        eos_weight=CONFIG["eos_weight"],
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, timesteps)
                loss   = diffusion_loss(
                    logits, labels, timesteps,
                    pad_id=tokenizer.pad_token_id,
                    eos_id=tokenizer.eos_token_id,
                    pad_weight=CONFIG["pad_weight"],
                    eos_weight=CONFIG["eos_weight"],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            global_step += 1

            # Throughput (tokens processed this step)
            step_time  = max(time.time() - step_t0, 1e-6)
            n_tokens   = input_ids.numel()
            tps        = n_tokens / step_time
            alpha      = 0.05   # EMA smoothing
            tokens_per_sec_ema = (
                tps if tokens_per_sec_ema is None
                else alpha * tps + (1 - alpha) * tokens_per_sec_ema
            )

            if global_step % CONFIG["log_every"] == 0:
                elapsed_min  = (time.time() - t_start) / 60
                lr           = scheduler.get_last_lr()[0]
                steps_left   = total_steps - global_step
                eta_min      = (elapsed_min / global_step) * steps_left if global_step > 0 else 0

                print(
                    f"  step {global_step:>6}/{total_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tokens_per_sec_ema:,.0f} tok/s | "
                    f"elapsed {elapsed_min:.1f}m | "
                    f"ETA {eta_min:.1f}m"
                )

                log_writer.writerow({
                    "epoch"        : epoch + 1,
                    "step"         : global_step,
                    "train_loss"   : round(loss.item(), 6),
                    "val_loss"     : "",
                    "lr"           : round(lr, 8),
                    "tokens_per_sec": round(tokens_per_sec_ema, 1),
                    "elapsed_min"  : round(elapsed_min, 2),
                })
                log_file.flush()

        # --- End-of-epoch validation ---
        print(f"\n[epoch {epoch+1}] Running validation …")
        val_loss    = run_validation(model, val_loader, device, use_amp=use_amp)
        elapsed_min = (time.time() - t_start) / 60
        lr          = scheduler.get_last_lr()[0]

        print(f"[epoch {epoch+1}] val_loss = {val_loss:.4f}  "
              f"(best so far: {best_val_loss:.4f})")

        log_writer.writerow({
            "epoch"         : epoch + 1,
            "step"          : global_step,
            "train_loss"    : "",
            "val_loss"      : round(val_loss, 6),
            "lr"            : round(lr, 8),
            "tokens_per_sec": round(tokens_per_sec_ema or 0, 1),
            "elapsed_min"   : round(elapsed_min, 2),
        })
        log_file.flush()

        # Generate a few molecules for sanity checking
        generate_samples(model, tokenizer, device, global_step)

        # --- Best-model checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path     = checkpoint_dir / "best_model_upscaled.pt"
            torch.save({
                "model"    : model.state_dict(),
                "config"   : checkpoint_config,
                "optimizer": optimizer.state_dict(),
                "epoch"    : epoch + 1,
                "step"     : global_step,
                "val_loss" : val_loss,
            }, save_path)
            print(f"  ✓ New best → {save_path.name}  (val_loss={val_loss:.4f})")

        # --- Periodic checkpoint every 3 epochs ---
        if (epoch + 1) % 3 == 0:
            save_path = checkpoint_dir / f"upscaled_epoch_{epoch+1}.pt"
            torch.save({
                "model"    : model.state_dict(),
                "config"   : checkpoint_config,
                "optimizer": optimizer.state_dict(),
                "epoch"    : epoch + 1,
                "step"     : global_step,
                "val_loss" : val_loss,
            }, save_path)
            print(f"  💾 Periodic checkpoint → {save_path.name}")

    log_file.close()
    print(f"\n{'='*70}")
    print(f"  Pretraining complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Log saved     : {log_path}")
    print(f"  Checkpoint    : {checkpoint_dir}/best_model_upscaled.pt")
    print(f"{'='*70}")

    # --- Save plots ---
    save_plots(str(log_path), CONFIG["plot_dir"])


if __name__ == "__main__":
    train()
