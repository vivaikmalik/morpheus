"""
finetune/train_chebi20.py
--------------------------
Fine-tunes the 27M Morpheus model on ChEBI-20 for direct comparison
with TGM-DLM (AAAI 2024).

Pipeline:
  1. Download ChEBI-20 (liupf/ChEBI-20-MM) via HuggingFace datasets.
  2. Convert SMILES → SELFIES, filter for tokenizer vocab + max length.
  3. Fine-tune from checkpoints/best_model_upscaled.pt (Stage 1 pretrain).
  4. After training, auto-evaluate on test set with TGM-DLM metrics.

Usage:
    python finetune/train_chebi20.py

Outputs:
    data/chebi20_train.csv, data/chebi20_val.csv, data/chebi20_test.csv
    checkpoints/best_chebi20_model.pt
    outputs/chebi20_log.csv
    outputs/plots/chebi20/loss_curves.png, lr_schedule.png
    outputs/chebi20_eval.csv
    outputs/chebi20_eval_summary.txt
"""

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.conditionalSELFIESDataset import ConditionalSELFIESDataset
from finetune.diffusionCollatorPrompt import ConditionalDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel
from tokenizer.chemicalTokenizer import ChemicalTokenizer

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

CONFIG = {
    # Model architecture (must match best_model_upscaled.pt)
    "vocab_size"  : 110,
    "hidden_size" : 512,
    "num_heads"   : 8,
    "ffn_dim"     : 1024,
    "num_layers"  : 8,
    "max_length"  : 74,
    "dropout"     : 0.1,
    "text_model"  : "BAAI/bge-large-en-v1.5",
    "uncond_prob" : 0.1,

    # Training
    "batch_size"           : 32,
    "learning_rate"        : 2e-4,
    "weight_decay"         : 0.01,
    "max_grad_norm"        : 1.0,
    "num_epochs"           : 10,
    "warmup_steps"         : 500,      # fixed steps, not ratio

    # Loss weights (EOS fix applied from the start)
    "eos_weight"  : 5.0,
    "pad_weight"  : 0.05,

    # Validation
    "val_every"           : 500,
    "val_max_batches"     : 100,
    "early_stop_patience" : 5,

    # Generation (periodic samples during training)
    "gen_steps"       : 32,
    "gen_temperature" : 1.0,
    "gen_cfg_scale"   : 3.0,

    # Eval (test set, after training)
    "eval_cfg_scale"   : 3.0,
    "eval_temperature" : 1.0,
    "eval_steps"       : 50,
    "eval_batch_size"  : 32,

    # Data
    "hf_dataset_id"   : "liupf/ChEBI-20-MM",
    "max_selfies_len" : 73,    # +1 EOS = 74
    "data_dir"        : str(PROJECT_ROOT / "data"),
    "train_csv"       : str(PROJECT_ROOT / "data" / "chebi20_train.csv"),
    "val_csv"         : str(PROJECT_ROOT / "data" / "chebi20_val.csv"),
    "test_csv"        : str(PROJECT_ROOT / "data" / "chebi20_test.csv"),

    # Paths
    "pretrain_ckpt"   : str(PROJECT_ROOT / "checkpoints" / "best_model_upscaled.pt"),
    "save_ckpt"       : str(PROJECT_ROOT / "checkpoints" / "best_chebi20_model.pt"),
    "periodic_prefix" : str(PROJECT_ROOT / "checkpoints" / "chebi20_epoch"),
    "log_path"        : str(PROJECT_ROOT / "outputs" / "chebi20_log.csv"),
    "plot_dir"        : str(PROJECT_ROOT / "outputs" / "plots" / "chebi20"),
    "eval_csv"        : str(PROJECT_ROOT / "outputs" / "chebi20_eval.csv"),
    "eval_summary"    : str(PROJECT_ROOT / "outputs" / "chebi20_eval_summary.txt"),

    "num_workers" : 0,
    "seed"        : 42,
    "log_every"   : 50,
    "freeze_text_encoder": True,
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def _detect_columns(ds_split):
    """Return (description_col, smiles_col) from a HuggingFace dataset split."""
    cols = ds_split.column_names
    desc_candidates  = ["description", "text", "caption", "input", "Description"]
    smiles_candidates = ["SMILES", "smiles", "canonical_smiles", "output", "molecule"]

    desc_col = next((c for c in desc_candidates if c in cols), None)
    smi_col  = next((c for c in smiles_candidates if c in cols), None)

    if desc_col is None or smi_col is None:
        raise ValueError(
            f"Could not detect description/SMILES columns in {cols}. "
            "Set them manually in _detect_columns()."
        )
    return desc_col, smi_col


def prepare_chebi20(tokenizer_path: Path, data_dir: Path) -> tuple:
    """
    Download ChEBI-20, convert SMILES → SELFIES, filter, save CSVs.
    Returns (train_csv, val_csv, test_csv) paths.
    Skips download if all three CSVs already exist.
    """
    train_csv = data_dir / "chebi20_train.csv"
    val_csv   = data_dir / "chebi20_val.csv"
    test_csv  = data_dir / "chebi20_test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        counts = {p.stem: len(pd.read_csv(p)) for p in (train_csv, val_csv, test_csv)}
        print(f"[data] ChEBI-20 CSVs already exist — "
              f"train={counts['chebi20_train']:,}  "
              f"val={counts['chebi20_val']:,}  "
              f"test={counts['chebi20_test']:,}")
        return train_csv, val_csv, test_csv

    # Load vocab
    with open(tokenizer_path, "r") as f:
        vocab = set(json.load(f)["token_to_id"].keys())

    # Download via HuggingFace datasets
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "pip install datasets  (HuggingFace datasets library required for ChEBI-20 download)"
        )

    print(f"[data] Downloading {CONFIG['hf_dataset_id']} ...")
    dataset = load_dataset(CONFIG["hf_dataset_id"])
    print(f"[data] Available splits: {list(dataset.keys())}")

    split_map = {
        "train"     : "train",
        "validation": "validation",
        "val"       : "validation",
        "test"      : "test",
    }
    out_paths = {}

    for out_name, hf_name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        if hf_name not in dataset:
            # Some versions use "valid" instead of "validation"
            hf_name = "valid" if "valid" in dataset else list(dataset.keys())[1]
        split = dataset[hf_name]
        desc_col, smi_col = _detect_columns(split)
        print(f"[data] {out_name}: {len(split):,} rows  "
              f"(description='{desc_col}', smiles='{smi_col}')")

        rows = []
        n_invalid, n_vocab, n_len = 0, 0, 0

        for item in tqdm(split, desc=f"  Converting {out_name}", leave=False):
            smiles = item[smi_col]
            if not smiles or not isinstance(smiles, str):
                n_invalid += 1
                continue

            smiles = smiles.strip()
            # Convert SMILES → SELFIES
            try:
                selfies_str = sf.encoder(smiles)
            except Exception:
                n_invalid += 1
                continue

            if not selfies_str:
                n_invalid += 1
                continue

            # Token-level filters
            try:
                tokens = list(sf.split_selfies(selfies_str))
            except Exception:
                n_invalid += 1
                continue

            if any(t not in vocab for t in tokens):
                n_vocab += 1
                continue
            if len(tokens) > CONFIG["max_selfies_len"]:
                n_len += 1
                continue

            desc = item[desc_col]
            if not desc or not isinstance(desc, str):
                desc = "A chemical molecule."
            rows.append({"prompt": desc.strip(), "response": selfies_str})

        df = pd.DataFrame(rows)
        out_path = data_dir / f"chebi20_{out_name}.csv"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        out_paths[out_name] = out_path
        print(f"  → {out_path.name}: {len(df):,} rows kept  "
              f"(invalid={n_invalid}, vocab={n_vocab}, length={n_len})")

    return out_paths["train"], out_paths["val"], out_paths["test"]


# =============================================================================
# LOSS FUNCTION (eos_weight=5.0, pad_weight=0.05)
# =============================================================================

def diffusion_loss(logits, labels, timesteps, tokenizer,
                   pad_weight=0.05, eos_weight=5.0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[tokenizer.pad_token_id] = pad_weight
    vocab_weights[tokenizer.eos_token_id] = eos_weight

    raw_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction="none",
    )

    # Low-noise timesteps (t≈0) should be predicted more accurately
    timesteps = timesteps.squeeze()
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    if timesteps.dim() == 1:
        timesteps = timesteps.unsqueeze(1)
    weights = (1.0 - timesteps).expand(-1, seq_len).reshape(-1)
    valid   = (labels.view(-1) != -100).float()

    return (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(model, val_loader, device, tokenizer):
    model.eval()
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False,
                          total=min(CONFIG["val_max_batches"], len(val_loader))):
            input_ids           = batch["input_ids"].to(device)
            labels              = batch["labels"].to(device)
            timesteps           = batch["timesteps"].to(device)
            text_input_ids      = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)

            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)
            logits = model(input_ids, timesteps, text_embeds,
                           (text_attention_mask == 0))
            loss = diffusion_loss(logits, labels, timesteps, tokenizer,
                                  CONFIG["pad_weight"], CONFIG["eos_weight"])
            total_loss += loss.item()
            n_steps    += 1
            if n_steps >= CONFIG["val_max_batches"]:
                break

    model.train()
    return total_loss / max(n_steps, 1)


# =============================================================================
# CFG GENERATION
# =============================================================================

@torch.no_grad()
def _generate_cfg_batch(model, tokenizer, hf_tokenizer, prompts, device,
                        cfg_scale, temperature, num_steps):
    max_length = CONFIG["max_length"]
    num_mols   = len(prompts)

    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    ).to(device)
    text_embeds = model.get_text_embeddings(
        text_inputs["input_ids"], text_inputs["attention_mask"]
    )
    null_embeds       = model.null_token.expand(num_mols, text_embeds.size(1), -1)
    text_padding_mask = (text_inputs["attention_mask"] == 0)

    input_ids = torch.full((num_mols, max_length), tokenizer.mask_token_id,
                           dtype=torch.long, device=device)
    t_vals    = torch.linspace(1.0, 0.0, num_steps, device=device)

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

    return input_ids.cpu().tolist()


def _decode_ids(ids, tokenizer):
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    ids = [i for i in ids if i not in
           (tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id)]
    return tokenizer.decode(ids)


def generate_samples(model, tokenizer, hf_tokenizer, device, step):
    """Print a few CFG samples during training — ChEBI-20 style prompts."""
    model.eval()
    prompts = [
        "The molecule is a member of the class of pyrimidines.",
        "The molecule is a monocarboxylic acid with antibacterial activity.",
        "The molecule is an amino acid derivative with a benzene ring.",
        "The molecule is a flavonoid with antioxidant properties.",
    ]
    gen_ids = _generate_cfg_batch(
        model, tokenizer, hf_tokenizer, prompts, device,
        cfg_scale=CONFIG["gen_cfg_scale"],
        temperature=CONFIG["gen_temperature"],
        num_steps=CONFIG["gen_steps"],
    )
    print(f"\n  CFG samples — step {step}")
    for i, (prompt, ids) in enumerate(zip(prompts, gen_ids)):
        selfies_str = _decode_ids(ids, tokenizer)
        try:
            smiles = sf.decoder(selfies_str)
            mol    = Chem.MolFromSmiles(smiles)
            status = f"VALID  {Chem.MolToSmiles(mol)[:60]}" if mol else "INVALID"
        except Exception:
            status = "DECODE ERROR"
        print(f"    [{i}] {status}")
    model.train()


# =============================================================================
# TEST-SET EVALUATION (TGM-DLM metrics)
# =============================================================================

def _bleu_n(ref_tokens, hyp_tokens, n):
    """Clip-count n-gram precision (no brevity penalty)."""
    if not ref_tokens or not hyp_tokens or len(hyp_tokens) < n:
        return 0.0
    ref_ng = {}
    for i in range(len(ref_tokens) - n + 1):
        g = tuple(ref_tokens[i:i+n])
        ref_ng[g] = ref_ng.get(g, 0) + 1
    hyp_ng = {}
    for i in range(len(hyp_tokens) - n + 1):
        g = tuple(hyp_tokens[i:i+n])
        hyp_ng[g] = hyp_ng.get(g, 0) + 1
    clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
    return clipped / sum(hyp_ng.values())


def _sentence_bleu(ref_tokens, hyp_tokens, max_n=4):
    """Geometric mean of 1-to-max_n gram precisions + brevity penalty."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precs = [_bleu_n(ref_tokens, hyp_tokens, n) for n in range(1, max_n + 1)]
    if all(p == 0 for p in precs):
        return 0.0
    log_avg = sum(math.log(p) if p > 0 else -1e9 for p in precs) / max_n
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return bp * math.exp(log_avg)


def _lev_sim(a: str, b: str) -> float:
    """Levenshtein similarity = 1 - normalised edit distance (on chars)."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp  = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return 1.0 - dp[n] / max(m, n)


def _fp_tanimoto(mol_a, mol_b, fp_type: str) -> float:
    if mol_a is None or mol_b is None:
        return 0.0
    if fp_type == "morgan":
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    elif fp_type == "maccs":
        fp_a = MACCSkeys.GenMACCSKeys(mol_a)
        fp_b = MACCSkeys.GenMACCSKeys(mol_b)
    elif fp_type == "rdk":
        fp_a = Chem.RDKFingerprint(mol_a)
        fp_b = Chem.RDKFingerprint(mol_b)
    else:
        raise ValueError(fp_type)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def evaluate_test_set(model, tokenizer, hf_tokenizer, device, test_csv: Path):
    """
    Generate molecules for all ChEBI-20 test prompts and compute TGM-DLM metrics.
    Saves per-row CSV + summary text file.
    """
    print(f"\n{'='*65}")
    print(f"  TEST SET EVALUATION — {test_csv.name}")
    print(f"  cfg={CONFIG['eval_cfg_scale']}  T={CONFIG['eval_temperature']}  "
          f"steps={CONFIG['eval_steps']}")
    print(f"{'='*65}")

    df_test = pd.read_csv(test_csv)
    prompts    = df_test["prompt"].tolist()
    gt_selfies = df_test["response"].tolist()
    n_total    = len(prompts)
    bs         = CONFIG["eval_batch_size"]

    model.eval()

    # Generate all molecules in batches
    all_gen_ids = []
    for b in tqdm(range(math.ceil(n_total / bs)), desc="  Generating"):
        batch_ids = _generate_cfg_batch(
            model, tokenizer, hf_tokenizer,
            prompts[b*bs : (b+1)*bs], device,
            cfg_scale=CONFIG["eval_cfg_scale"],
            temperature=CONFIG["eval_temperature"],
            num_steps=CONFIG["eval_steps"],
        )
        all_gen_ids.extend(batch_ids)

    # Compute per-molecule metrics
    rows = []
    for prompt, gt_sf, gen_ids in tqdm(
        zip(prompts, gt_selfies, all_gen_ids), total=n_total, desc="  Scoring"
    ):
        gen_sf  = _decode_ids(gen_ids, tokenizer)

        # SMILES for both (needed for TGM-DLM metrics on SMILES level)
        gt_smiles, gt_mol = "", None
        try:
            gt_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(sf.decoder(gt_sf)))
            gt_mol    = Chem.MolFromSmiles(gt_smiles)
        except Exception:
            pass

        gen_smiles, gen_mol = "", None
        valid = False
        try:
            raw = sf.decoder(gen_sf)
            gen_mol = Chem.MolFromSmiles(raw)
            if gen_mol is not None:
                gen_smiles = Chem.MolToSmiles(gen_mol)
                valid = True
        except Exception:
            pass

        # BLEU-2 and BLEU-4 on SMILES characters (TGM-DLM convention)
        ref_chars = list(gt_smiles)
        hyp_chars = list(gen_smiles)
        bleu2 = _sentence_bleu(ref_chars, hyp_chars, max_n=2)
        bleu4 = _sentence_bleu(ref_chars, hyp_chars, max_n=4)

        # Levenshtein similarity on canonical SMILES characters
        lev_sim = _lev_sim(gt_smiles, gen_smiles) if gt_smiles and gen_smiles else 0.0

        # Exact match (canonical SMILES)
        exact = (gt_smiles == gen_smiles) if gt_smiles and gen_smiles else False

        # Fingerprint similarities (only for valid pairs)
        tan_morgan = _fp_tanimoto(gen_mol, gt_mol, "morgan") if valid and gt_mol else 0.0
        tan_maccs  = _fp_tanimoto(gen_mol, gt_mol, "maccs")  if valid and gt_mol else 0.0
        tan_rdk    = _fp_tanimoto(gen_mol, gt_mol, "rdk")    if valid and gt_mol else 0.0

        rows.append({
            "prompt"      : prompt,
            "gt_smiles"   : gt_smiles,
            "gen_smiles"  : gen_smiles,
            "valid"       : valid,
            "exact_match" : exact,
            "bleu2"       : round(bleu2,  4),
            "bleu4"       : round(bleu4,  4),
            "lev_sim"     : round(lev_sim, 4),
            "morgan_sim"  : round(tan_morgan, 4),
            "maccs_sim"   : round(tan_maccs,  4),
            "rdk_sim"     : round(tan_rdk,    4),
        })

    df_out = pd.DataFrame(rows)

    # Save per-row CSV
    out_csv = Path(CONFIG["eval_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n  Per-row results → {out_csv}")

    # Aggregate
    n       = len(df_out)
    n_valid = df_out["valid"].sum()
    valid_rows = df_out[df_out["valid"]]

    validity_pct  = 100.0 * n_valid / n
    exact_pct     = 100.0 * df_out["exact_match"].mean()
    bleu2_avg     = df_out["bleu2"].mean()
    bleu4_avg     = df_out["bleu4"].mean()
    lev_avg       = df_out["lev_sim"].mean()
    # Fingerprint metrics on valid-only rows (same convention as TGM-DLM)
    morgan_avg = valid_rows["morgan_sim"].mean() if n_valid else float("nan")
    maccs_avg  = valid_rows["maccs_sim"].mean()  if n_valid else float("nan")
    rdk_avg    = valid_rows["rdk_sim"].mean()    if n_valid else float("nan")

    summary_lines = [
        "=" * 65,
        "  ChEBI-20 EVALUATION — Morpheus 27M",
        f"  Checkpoint : {Path(CONFIG['save_ckpt']).name}",
        f"  Test set   : {n} molecules",
        f"  CFG scale  : {CONFIG['eval_cfg_scale']}  "
        f"Temperature: {CONFIG['eval_temperature']}  "
        f"Steps: {CONFIG['eval_steps']}",
        "=" * 65,
        f"  Validity (%)                  {validity_pct:>8.1f}%",
        f"  Exact Match (%)               {exact_pct:>8.1f}%",
        f"  BLEU-2                        {bleu2_avg:>8.4f}",
        f"  BLEU-4                        {bleu4_avg:>8.4f}",
        f"  Levenshtein Similarity        {lev_avg:>8.4f}",
        f"  Morgan FP Tanimoto (valid)    {morgan_avg:>8.4f}",
        f"  MACCS FP Tanimoto  (valid)    {maccs_avg:>8.4f}",
        f"  RDK FP Tanimoto    (valid)    {rdk_avg:>8.4f}",
        "=" * 65,
        "",
        "Reference: TGM-DLM (AAAI 2024) on ChEBI-20",
        "  Validity: ~100%  BLEU-2: 0.564  BLEU-4: 0.470",
        "  Levenshtein: 0.758  Morgan: 0.609  MACCS: 0.784  RDK: 0.677",
        "=" * 65,
    ]

    print("\n" + "\n".join(summary_lines))

    out_txt = Path(CONFIG["eval_summary"])
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n  Summary → {out_txt}")

    return df_out


# =============================================================================
# PLOTS
# =============================================================================

def save_plots():
    log_path = Path(CONFIG["log_path"])
    plot_dir = Path(CONFIG["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    if df.empty:
        print("[plot] Empty log — skipping.")
        return

    train_rows = df[df["train_loss"].notna()].copy()
    val_rows   = df[df["val_loss"].notna()].copy()
    style = {"linewidth": 1.5, "alpha": 0.9}
    dpi   = 300

    fig, ax = plt.subplots(figsize=(9, 5))
    if not train_rows.empty:
        ax.plot(train_rows["step"], train_rows["train_loss"],
                label="Train loss", color="#2196F3", **style)
    if not val_rows.empty:
        ax.plot(val_rows["step"], val_rows["val_loss"],
                label="Val loss", color="#F44336", linestyle="--", **style)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("ChEBI-20 Fine-tuning Loss (27M Morpheus)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", dpi=dpi)
    plt.close(fig)

    if not train_rows.empty and "lr" in train_rows.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(train_rows["step"], train_rows["lr"], color="#4CAF50", **style)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (cosine + warmup)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "lr_schedule.png", dpi=dpi)
        plt.close(fig)

    print(f"[plot] Saved → {plot_dir}/")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    tokenizer_path = PROJECT_ROOT / "chemical_tokenizer.json"
    data_dir       = Path(CONFIG["data_dir"])

    train_csv, val_csv, test_csv = prepare_chebi20(tokenizer_path, data_dir)

    tokenizer    = ChemicalTokenizer(tokenizer_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    print(f"[data] train={len(train_df):,}  val={len(val_df):,}  "
          f"test={len(pd.read_csv(test_csv)):,}")

    collator      = ConditionalDiffusionCollator(tokenizer, hf_tokenizer)
    train_dataset = ConditionalSELFIESDataset(train_df, tokenizer)
    val_dataset   = ConditionalSELFIESDataset(val_df,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, collate_fn=collator,
                              num_workers=CONFIG["num_workers"])
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False,  collate_fn=collator,
                              num_workers=CONFIG["num_workers"])

    # ── Model ──────────────────────────────────────────────────────────────────
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

    total, trainable = model.count_parameters()
    print(f"[model] {total:,} total params | {trainable:,} trainable")

    # Load Stage 1 pretrained weights
    pretrain_ckpt = Path(CONFIG["pretrain_ckpt"])
    if pretrain_ckpt.exists():
        print(f"[ckpt]  Loading pretrained weights from {pretrain_ckpt.name} ...")
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"        missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        print(f"[ckpt]  WARNING: {pretrain_ckpt} not found — training from scratch.")

    # Freeze / unfreeze text encoder
    frozen = CONFIG["freeze_text_encoder"]
    for p in model.text_encoder.parameters():
        p.requires_grad = not frozen
    print(f"[model] Text encoder {'frozen' if frozen else 'unfrozen'}.")

    # ── Optimizer / Scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"],
    )

    train_size  = len(train_dataset)
    total_steps = (train_size // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = CONFIG["warmup_steps"],
        num_training_steps= total_steps,
    )
    print(f"[sched] {total_steps:,} total steps | {CONFIG['warmup_steps']} warmup steps")
    print(f"[loss]  eos_weight={CONFIG['eos_weight']}  pad_weight={CONFIG['pad_weight']}")

    # ── CSV log ────────────────────────────────────────────────────────────────
    log_path = Path(CONFIG["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_columns = ["epoch", "step", "train_loss", "val_loss", "lr",
                   "tokens_per_sec", "elapsed_min"]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_columns)
    log_writer.writeheader()
    log_file.flush()

    Path(CONFIG["checkpoint_dir"] if "checkpoint_dir" in CONFIG
         else str(PROJECT_ROOT / "checkpoints")).mkdir(exist_ok=True)
    ckpt_dir = Path(CONFIG["save_ckpt"]).parent
    ckpt_dir.mkdir(exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────────────
    global_step       = 0
    best_val_loss     = float("inf")
    last_val_loss     = float("inf")
    patience_counter  = 0
    stopped_early     = False
    t_start           = time.time()
    tokens_per_sec_ema = None

    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'─'*65}\n  Epoch {epoch+1}/{CONFIG['num_epochs']}\n{'─'*65}")
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for batch in bar:
            step_t0 = time.time()

            input_ids           = batch["input_ids"].to(device)
            labels              = batch["labels"].to(device)
            timesteps           = batch["timesteps"].to(device)
            text_input_ids      = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)

            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)
            logits = model(input_ids, timesteps, text_embeds,
                           (text_attention_mask == 0))
            loss = diffusion_loss(logits, labels, timesteps, tokenizer,
                                  CONFIG["pad_weight"], CONFIG["eos_weight"])

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
                lr          = scheduler.get_last_lr()[0]
                elapsed_min = (time.time() - t_start) / 60
                bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
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
                print(f"\n  → Val loss: {val_loss:.4f}  (step {global_step})")
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
                        "epoch"    : epoch + 1,
                        "step"     : global_step,
                        "model"    : model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_loss" : val_loss,
                        "config"   : CONFIG,
                    }, CONFIG["save_ckpt"])
                    print(f"  ✓ New best → {Path(CONFIG['save_ckpt']).name}  "
                          f"(val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement "
                          f"({patience_counter}/{CONFIG['early_stop_patience']})")
                    if patience_counter >= CONFIG["early_stop_patience"]:
                        print("  Early stopping triggered.")
                        stopped_early = True
                        break

        # Periodic checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            periodic_path = f"{CONFIG['periodic_prefix']}_{epoch+1}.pt"
            torch.save({
                "epoch": epoch+1, "step": global_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "val_loss": last_val_loss,
                "config": CONFIG,
            }, periodic_path)
            print(f"  Periodic checkpoint → {Path(periodic_path).name}")

        if stopped_early:
            break

    log_file.close()
    print(f"\nTraining complete.{' (early stop)' if stopped_early else ''}")
    print(f"Best val loss: {best_val_loss:.4f}")

    save_plots()

    # ── Test-set evaluation ────────────────────────────────────────────────────
    # Reload best checkpoint for eval
    best_ckpt_path = Path(CONFIG["save_ckpt"])
    if best_ckpt_path.exists():
        print(f"\n[eval] Reloading best checkpoint for test evaluation ...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        print("[eval] WARNING: best checkpoint not found, evaluating current weights.")

    evaluate_test_set(model, tokenizer, hf_tokenizer, device, test_csv)


if __name__ == "__main__":
    train()
