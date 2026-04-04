"""
inference/evaluate_prompted.py
-------------------------------
Evaluates the text-conditioned finetuned model on a test CSV.

For each prompt, generates one molecule via MaskGIT + CFG, then computes:
  - Validity, exact match, scaffold match
  - BLEU (SELFIES tokens as words)
  - Levenshtein distance (normalised and raw)
  - Tanimoto similarity (Morgan FP, radius=2, 2048 bits)
  - Token retention rate
  - QED, LogP, MW of generated molecule

Usage:
    python inference/evaluate_prompted.py
    python inference/evaluate_prompted.py --cfg 2.0 --temp 0.8 --steps 50
    python inference/evaluate_prompted.py --checkpoint checkpoints/best_chebi20_model.pt \\
                                          --test_csv data/chebi20_test.csv
    python inference/evaluate_prompted.py --checkpoint checkpoints/best_chebi20_model.pt \\
                                          --test_csv data/chebi20_test.csv --eos_truncate

Output CSV: outputs/eval_{ckpt_stem}_cfg{X}_temp{Y}_steps{Z}[_eost].csv
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.molecularDiffusionModel import MolecularDiffusionModel
from tokenizer.chemicalTokenizer import ChemicalTokenizer

# =============================================================================
# PATHS & CONFIG
# =============================================================================

PROJECT_ROOT       = Path(__file__).parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_finetuned_model.pt"
TOKENIZER_PATH     = PROJECT_ROOT / "chemical_tokenizer.json"
DEFAULT_TEST_CSV   = PROJECT_ROOT / "data" / "test.csv"
GEN_CFG = {
    "cfg_scale"  : 3.0,
    "temperature": 1.2,
    "num_steps"  : 32,
    "max_length" : 74,
    "batch_size" : 32,   # prompts per forward-pass batch
}

# =============================================================================
# METRICS HELPERS
# =============================================================================

def selfies_tokens(s: str) -> list:
    try:
        return list(sf.split_selfies(s))
    except Exception:
        return list(s)   # fallback: character-level


def bleu_score(reference_tokens: list, hypothesis_tokens: list, max_n: int = 4) -> float:
    """Sentence BLEU with up to 4-gram precision, uniform weights, no brevity penalty."""
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams  = _ngram_counts(reference_tokens,  n)
        hyp_ngrams  = _ngram_counts(hypothesis_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        clipped = sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())
        precisions.append(clipped / sum(hyp_ngrams.values()))

    if all(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(
        math.log(p) if p > 0 else -1e9
        for p in precisions
    ) / max_n

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    return bp * math.exp(log_avg)


def _ngram_counts(tokens: list, n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def levenshtein(a: list, b: list) -> int:
    """Edit distance between two token sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def tanimoto(mol_a, mol_b) -> float:
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def murcko_scaffold(mol) -> str:
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core) if core else ""
    except Exception:
        return ""


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(device, checkpoint_path: Path):
    print(f"[ckpt] Loading {checkpoint_path.name} ...")
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    print(f"  epoch={config.get('num_epochs','?')}  "
          f"step={ckpt.get('step','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    print(f"  hidden={config['hidden_size']}  ffn={config['ffn_dim']}  "
          f"layers={config['num_layers']}  heads={config['num_heads']}")

    model = MolecularDiffusionModel(
        vocab_size      = config["vocab_size"],
        hidden_size     = config["hidden_size"],
        num_heads       = config["num_heads"],
        ffn_dim         = config["ffn_dim"],
        num_layers      = config["num_layers"],
        max_length      = config["max_length"],
        pad_token_id    = 0,
        text_model_name = config.get("text_model", "BAAI/bge-large-en-v1.5"),
        uncond_prob     = 0.0,   # no dropout at inference
        dropout         = 0.0,
    )

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  strict=False: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {total/1e6:.1f}M total params | {trainable/1e6:.1f}M trainable")

    return model, config


# =============================================================================
# CFG GENERATION (batch of prompts)
# =============================================================================

@torch.no_grad()
def generate_cfg_batch(model, tokenizer, hf_tokenizer, prompts: list, device) -> list:
    """
    Generate one molecule per prompt using classifier-free guidance.
    Returns a list of token-ID lists (pre-EOS-truncation).
    """
    cfg_scale   = GEN_CFG["cfg_scale"]
    temperature = GEN_CFG["temperature"]
    num_steps   = GEN_CFG["num_steps"]
    max_length  = GEN_CFG["max_length"]
    num_mols    = len(prompts)

    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    ).to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)

    text_embeds = model.get_text_embeddings(
        text_inputs["input_ids"], text_inputs["attention_mask"]
    )
    null_embeds = model.null_token.expand(num_mols, text_embeds.size(1), -1)

    input_ids = torch.full(
        (num_mols, max_length), tokenizer.mask_token_id,
        dtype=torch.long, device=device
    )
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

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
            _, mask_indices = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
            sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

        input_ids = sampled

    return input_ids.cpu().tolist()


# =============================================================================
# POST-HOC EOS TRUNCATION
# =============================================================================

@torch.no_grad()
def apply_eos_truncation(model, tokenizer, hf_tokenizer,
                         all_gen_ids: list, prompts: list, device) -> list:
    """
    Run one extra forward pass at t=0 for each batch, find the position with
    the highest EOS probability, and truncate there.  Positions already holding
    EOS or PAD are skipped.  Returns a new list of token-ID lists.
    """
    max_length = GEN_CFG["max_length"]
    batch_size = GEN_CFG["batch_size"]
    n_total    = len(prompts)
    truncated  = []

    for b_start in range(0, n_total, batch_size):
        b_end    = min(b_start + batch_size, n_total)
        b_ids    = all_gen_ids[b_start:b_end]
        b_prompts = prompts[b_start:b_end]
        num_mols  = len(b_prompts)

        input_ids = torch.tensor(b_ids, dtype=torch.long, device=device)

        text_inputs = hf_tokenizer(
            b_prompts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        text_padding_mask = (text_inputs["attention_mask"] == 0)
        text_embeds       = model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )

        # t=0 forward pass (fully denoised signal)
        step_t  = torch.zeros(num_mols, 1, device=device)
        logits  = model(input_ids, step_t, text_embeds, text_padding_mask)
        eos_probs = torch.softmax(logits, dim=-1)[:, :, tokenizer.eos_token_id]
        # [num_mols, max_length]

        for mol_idx in range(num_mols):
            ids      = b_ids[mol_idx][:]
            probs    = eos_probs[mol_idx].cpu().tolist()

            # Ignore positions that are already EOS/PAD/MASK — only consider
            # positions that carry a real chemical token
            best_pos, best_prob = -1, -1.0
            for pos, (tok, p) in enumerate(zip(ids, probs)):
                if tok in (tokenizer.eos_token_id,
                           tokenizer.pad_token_id,
                           tokenizer.mask_token_id):
                    continue
                if p > best_prob:
                    best_prob, best_pos = p, pos

            if best_pos >= 0:
                ids = ids[:best_pos]   # truncate before that position

            truncated.append(ids)

    return truncated


# =============================================================================
# PER-MOLECULE METRICS
# =============================================================================

def compute_row_metrics(prompt: str, gt_selfies: str,
                        gen_ids: list, tokenizer) -> dict:
    # Decode generated tokens
    ids = gen_ids[:]
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    ids = [i for i in ids if i not in
           (tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id)]

    gen_selfies = tokenizer.decode(ids)
    gen_tokens  = selfies_tokens(gen_selfies)
    gt_tokens   = selfies_tokens(gt_selfies)

    row = {
        "prompt"           : prompt,
        "gt_selfies"       : gt_selfies,
        "gen_selfies"      : gen_selfies,
        "gen_token_count"  : len(gen_tokens),
        # validity
        "valid"            : False,
        "gen_smiles"       : "",
        # similarity to reference
        "exact_match"      : gen_selfies.strip() == gt_selfies.strip(),
        "bleu"             : bleu_score(gt_tokens, gen_tokens),
        "levenshtein"      : levenshtein(gt_tokens, gen_tokens),
        "norm_levenshtein" : 0.0,
        "tanimoto"         : 0.0,
        "scaffold_match"   : False,
        "token_retention"  : None,
        # generated mol properties
        "qed"              : None,
        "logp"             : None,
        "mw"               : None,
    }

    max_len = max(len(gt_tokens), len(gen_tokens), 1)
    row["norm_levenshtein"] = round(row["levenshtein"] / max_len, 4)

    # Parse ground-truth mol
    gt_mol = None
    try:
        gt_smiles = sf.decoder(gt_selfies)
        gt_mol    = Chem.MolFromSmiles(gt_smiles)
    except Exception:
        pass

    # Parse generated mol
    if not gen_tokens:
        return row

    try:
        gen_smiles_raw = sf.decoder(gen_selfies)
        gen_mol        = Chem.MolFromSmiles(gen_smiles_raw)
        if gen_mol is None:
            return row

        gen_smiles = Chem.MolToSmiles(gen_mol)
        row["valid"]     = True
        row["gen_smiles"] = gen_smiles
        row["qed"]        = round(QED.qed(gen_mol), 4)
        row["logp"]       = round(Descriptors.MolLogP(gen_mol), 4)
        row["mw"]         = round(Descriptors.MolWt(gen_mol), 2)

        # Token retention: re-encode canonical SMILES, compare token counts
        try:
            reenc        = sf.encoder(gen_smiles)
            reenc_ntok   = len(list(sf.split_selfies(reenc)))
            row["token_retention"] = round(reenc_ntok / max(len(gen_tokens), 1), 4)
        except Exception:
            pass

        # Tanimoto vs ground truth
        row["tanimoto"] = round(tanimoto(gen_mol, gt_mol), 4)

        # Scaffold match
        gen_scaffold = murcko_scaffold(gen_mol)
        gt_scaffold  = murcko_scaffold(gt_mol)
        row["scaffold_match"] = (
            bool(gen_scaffold) and bool(gt_scaffold)
            and gen_scaffold == gt_scaffold
        )

    except Exception:
        pass

    return row


# =============================================================================
# AGGREGATE SUMMARY
# =============================================================================

def print_summary(df: pd.DataFrame):
    n     = len(df)
    valid = df[df["valid"]]
    valid_pairs = df[df["valid"] & df["gt_selfies"].notna()]  # both gen and gt parseable

    def pct(x): return f"{x:.1f}%"
    def avg(series): return f"{series.mean():.4f}" if len(series) else "N/A"

    validity       = 100 * len(valid) / n
    exact_pct      = 100 * df["exact_match"].mean()
    scaffold_pct   = (100 * valid_pairs["scaffold_match"].mean()
                      if len(valid_pairs) else float("nan"))
    bleu_avg       = df["bleu"].mean()
    lev_avg        = df["levenshtein"].mean()
    norm_lev_avg   = df["norm_levenshtein"].mean()
    tan_all        = df["tanimoto"].mean()
    tan_valid      = valid_pairs["tanimoto"].mean() if len(valid_pairs) else float("nan")
    tok_ret_avg    = df["token_retention"].dropna().mean()

    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION — test.csv")
    print(f"{'='*60}")
    print(f"  Validity (%)                              {validity:.1f}%")
    print(f"  Exact Match (%)                           {exact_pct:.1f}%")
    print(f"  Scaffold Match — valid pairs (%)          {scaffold_pct:.1f}%")
    print(f"  BLEU Score (avg)                          {bleu_avg:.4f}")
    print(f"  Levenshtein Distance (avg)                {lev_avg:.2f}")
    print(f"  Normalised Levenshtein (avg)              {norm_lev_avg:.4f}")
    print(f"  Tanimoto Similarity (avg, all)            {tan_all:.4f}")
    print(f"  Tanimoto Similarity (avg, valid pairs)    {tan_valid:.4f}")
    print(f"  Token Retention Rate (avg, generated)     {tok_ret_avg:.4f}")
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-conditioned diffusion model")
    parser.add_argument("--cfg",   type=float, default=GEN_CFG["cfg_scale"],
                        help="CFG guidance scale (default: 3.0)")
    parser.add_argument("--temp",  type=float, default=GEN_CFG["temperature"],
                        help="Sampling temperature (default: 1.2)")
    parser.add_argument("--steps", type=int,   default=GEN_CFG["num_steps"],
                        help="MaskGIT denoising steps (default: 32)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file "
                             "(default: checkpoints/best_finetuned_model.pt)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to test CSV with 'prompt' and 'response' columns "
                             "(default: data/test.csv)")
    parser.add_argument("--eos_truncate", action="store_true",
                        help="Post-hoc EOS truncation: run one forward pass at t=0 "
                             "and truncate each sequence at the position with highest "
                             "EOS probability")
    args = parser.parse_args()

    checkpoint_path = (Path(args.checkpoint) if args.checkpoint
                       else DEFAULT_CHECKPOINT)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    test_csv_path = (Path(args.test_csv) if args.test_csv else DEFAULT_TEST_CSV)
    if not test_csv_path.is_absolute():
        test_csv_path = PROJECT_ROOT / test_csv_path

    GEN_CFG["cfg_scale"]   = args.cfg
    GEN_CFG["temperature"] = args.temp
    GEN_CFG["num_steps"]   = args.steps

    ckpt_stem  = checkpoint_path.stem
    eost_tag   = "_eost" if args.eos_truncate else ""
    output_csv = (PROJECT_ROOT / "outputs" /
                  f"eval_{ckpt_stem}_cfg{args.cfg}_temp{args.temp}"
                  f"_steps{args.steps}{eost_tag}.csv")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")
    print(f"[ckpt]   {checkpoint_path}")
    print(f"[data]   {test_csv_path}")
    print(f"[config] cfg={args.cfg}  temp={args.temp}  steps={args.steps}  "
          f"eos_truncate={args.eos_truncate}")
    print(f"[output] {output_csv.name}")

    # Tokenizers
    tokenizer    = ChemicalTokenizer(TOKENIZER_PATH)
    hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    # Model
    model, _ = load_model(device, checkpoint_path)

    # Test data
    print(f"\n[data] Loading {test_csv_path} ...")
    df_test = pd.read_csv(test_csv_path)
    print(f"  {len(df_test):,} rows  |  columns: {list(df_test.columns)}")
    prompts    = df_test["prompt"].tolist()
    gt_selfies = df_test["response"].tolist()
    n_total    = len(prompts)

    # Generate in batches
    batch_size  = GEN_CFG["batch_size"]
    all_gen_ids = []
    n_batches   = math.ceil(n_total / batch_size)

    print(f"\n[gen] Generating {n_total} molecules "
          f"(bs={batch_size}, cfg={GEN_CFG['cfg_scale']}, "
          f"T={GEN_CFG['temperature']}, steps={GEN_CFG['num_steps']}) ...")
    t0 = time.time()

    for b in tqdm(range(n_batches), desc="Generating"):
        start = b * batch_size
        end   = min(start + batch_size, n_total)
        batch_ids = generate_cfg_batch(
            model, tokenizer, hf_tokenizer, prompts[start:end], device
        )
        all_gen_ids.extend(batch_ids)

    gen_time = time.time() - t0
    print(f"  Done in {gen_time:.1f}s  ({gen_time/n_total:.2f}s/mol)")

    # Optional post-hoc EOS truncation
    if args.eos_truncate:
        print(f"\n[eos_truncate] Running t=0 forward pass to truncate sequences ...")
        all_gen_ids = apply_eos_truncation(
            model, tokenizer, hf_tokenizer, all_gen_ids, prompts, device
        )

    # Compute per-molecule metrics
    print(f"\n[metrics] Computing metrics for {n_total} molecules ...")
    rows = []
    for prompt, gt, gen_ids in tqdm(
        zip(prompts, gt_selfies, all_gen_ids), total=n_total, desc="Scoring"
    ):
        rows.append(compute_row_metrics(prompt, gt, gen_ids, tokenizer))

    df_out = pd.DataFrame(rows)

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[out] Saved {len(df_out)} rows → {output_csv}")

    # Summary
    print_summary(df_out)


if __name__ == "__main__":
    main()
