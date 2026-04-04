"""
inference/evaluate_chebi20.py
------------------------------
Evaluates the text-conditioned Morpheus model on CheBI-20 test set
using the same metrics reported by TGM-DLM (AAAI 2024):

  - Validity:    fraction of generated SELFIES that decode to valid molecules
  - BLEU:        sentence-level BLEU between generated and ground truth SELFIES tokens
  - MACCS FTS:   Tanimoto similarity using MACCS fingerprints
  - Morgan FTS:  Tanimoto similarity using Morgan fingerprints (radius=2, 2048 bits)
  - Exact Match: fraction where generated canonical SMILES == ground truth
  - Levenshtein: normalized edit distance on SELFIES token sequences

Usage:
    python inference/evaluate_chebi20.py
    python inference/evaluate_chebi20.py --checkpoint checkpoints/best_finetuned_model.pt
    python inference/evaluate_chebi20.py --cfg 2.0 --temp 0.8 --steps 50

Output:
    outputs/chebi20_eval_cfg{X}_temp{Y}_steps{Z}.csv     (per-molecule results)
    outputs/chebi20_eval_summary.txt                       (aggregated metrics)
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import selfies as sf
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# METRICS
# =============================================================================

def selfies_tokens(s):
    """Split a SELFIES string into its constituent tokens."""
    try:
        return list(sf.split_selfies(s))
    except Exception:
        return list(s)


def ngram_counts(tokens, n):
    """Count n-grams in a token list."""
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i+n])] += 1
    return counts


def sentence_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    """
    Sentence-level BLEU with uniform weights up to max_n-grams.
    Includes brevity penalty.
    """
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))

    # n-gram precisions
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ng  = ngram_counts(reference_tokens, n)
        hyp_ng  = ngram_counts(hypothesis_tokens, n)

        if not hyp_ng:
            log_precisions.append(float('-inf'))
            continue

        # Clipped counts
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total   = sum(hyp_ng.values())

        precision = clipped / total if total > 0 else 0.0
        # Smoothing: add epsilon to avoid log(0)
        log_precisions.append(np.log(precision + 1e-12))

    # Geometric mean with uniform weights
    avg_log_precision = np.mean(log_precisions)
    return bp * np.exp(avg_log_precision)


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,          # insertion
                prev_row[j + 1] + 1,      # deletion
                prev_row[j] + cost,        # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


def normalized_levenshtein(ref_tokens, hyp_tokens):
    """Levenshtein distance normalized by max length."""
    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    max_len = max(len(ref_tokens), len(hyp_tokens), 1)
    return 1.0 - dist / max_len   # higher = more similar


def maccs_tanimoto(mol1, mol2):
    """MACCS fingerprint Tanimoto similarity."""
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def morgan_tanimoto(mol1, mol2, radius=2, n_bits=2048):
    """Morgan fingerprint Tanimoto similarity."""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_scaffold(mol):
    """Get Bemis-Murcko scaffold SMILES."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# =============================================================================
# GENERATION WITH CFG
# =============================================================================

def generate_batch_cfg(model, mol_tokenizer, text_tokenizer, device,
                       prompts, cfg_scale, num_steps, temperature, max_length,
                       max_text_len):
    """
    Generate molecules for a batch of text prompts using CFG.
    Returns raw token ID tensor (B, max_length).
    """
    B = len(prompts)

    # Encode text
    text_enc = text_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=max_text_len, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            text_enc["input_ids"], text_enc["attention_mask"]
        )
    text_pad = (text_enc["attention_mask"] == 0)

    # Start fully masked
    input_ids = torch.full(
        (B, max_length), mol_tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(B).unsqueeze(-1)

            # Conditional + unconditional forward passes
            logits_cond   = model(input_ids, step_t, text_embeds, text_pad)
            logits_uncond = model(input_ids, step_t, None, None)

            # CFG interpolation
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            # Block MASK token (except on last step)
            if step_idx < num_steps - 1:
                logits[:, :, mol_tokenizer.mask_token_id] = float('-inf')

            # Sample
            probs   = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()

            # Confidence-based re-masking
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(
                    confidence, num_to_mask, dim=-1, largest=False
                )
                sampled.scatter_(1, mask_indices, mol_tokenizer.mask_token_id)

            input_ids = sampled

    return input_ids


def decode_molecule(ids, tokenizer):
    """
    Decode token IDs to SELFIES string and canonical SMILES.
    Returns (selfies_str, canonical_smiles, mol) or (selfies_str, None, None).
    """
    ids = list(ids)

    # Truncate at EOS
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]

    selfies_str = tokenizer.decode(ids)

    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return selfies_str, Chem.MolToSmiles(mol), mol
    except Exception:
        pass

    return selfies_str, None, None


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_finetuned_model(checkpoint_path, mol_tokenizer, device):
    """Load fine-tuned model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = MolecularDiffusionModel(
        vocab_size      = config["vocab_size"],
        hidden_size     = config["hidden_size"],
        num_heads       = config["num_heads"],
        ffn_dim         = config["ffn_dim"],
        num_layers      = config["num_layers"],
        max_length      = config["max_length"],
        pad_token_id    = mol_tokenizer.pad_token_id,
        text_model_name = config.get("text_model", "allenai/scibert_scivocab_uncased"),
        uncond_prob     = 0.0,   # no dropout at inference
        dropout         = 0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"  Step: {step}  |  Val loss: {val_loss}")

    total, trainable, frozen = model.count_parameters()
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")

    return model, config


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # --- Tokenizers ---
    mol_tokenizer  = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.text_model or "allenai/scibert_scivocab_uncased"
    )

    # --- Load model ---
    model, config = load_finetuned_model(args.checkpoint, mol_tokenizer, device)
    max_length  = config["max_length"]
    max_text_len = config.get("max_text_len", 256)

    # --- Load test data ---
    test_path = ROOT / "data" / "chebi20_test.csv"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run: python scripts/prepare_chebi20.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"\nTest set: {len(test_df):,} molecules")

    # --- Generate molecules for all test prompts ---
    print(f"\nGenerating with CFG={args.cfg}  temp={args.temp}  steps={args.steps}...")
    start_time = time.time()

    all_results = []
    batch_size = args.batch_size

    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Generating"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]

        prompts = batch_df["description"].tolist()

        raw_ids = generate_batch_cfg(
            model, mol_tokenizer, text_tokenizer, device,
            prompts, args.cfg, args.steps, args.temp,
            max_length, max_text_len,
        )

        for i, row in enumerate(batch_df.itertuples()):
            ids = raw_ids[i].cpu().tolist()
            gen_selfies, gen_smiles, gen_mol = decode_molecule(ids, mol_tokenizer)

            gt_selfies = row.selfies
            gt_smiles  = row.smiles if hasattr(row, "smiles") else None

            # Ground truth mol
            gt_mol = None
            gt_canonical = None
            if gt_smiles:
                gt_mol = Chem.MolFromSmiles(gt_smiles)
                if gt_mol:
                    gt_canonical = Chem.MolToSmiles(gt_mol)

            # --- Compute metrics ---
            valid = gen_mol is not None

            # BLEU (SELFIES token-level)
            ref_tokens = selfies_tokens(gt_selfies)
            hyp_tokens = selfies_tokens(gen_selfies) if gen_selfies else []
            bleu = sentence_bleu(ref_tokens, hyp_tokens)

            # Levenshtein similarity
            lev_sim = normalized_levenshtein(ref_tokens, hyp_tokens)

            # Fingerprint similarities (only if both molecules are valid)
            maccs_sim = None
            morgan_sim = None
            if valid and gt_mol is not None:
                try:
                    maccs_sim  = maccs_tanimoto(gen_mol, gt_mol)
                    morgan_sim = morgan_tanimoto(gen_mol, gt_mol)
                except Exception:
                    pass

            # Exact match
            exact_match = False
            if valid and gt_canonical:
                exact_match = (gen_smiles == gt_canonical)

            # Scaffold match
            scaffold_match = False
            if valid and gt_mol is not None:
                gen_scaffold = get_scaffold(gen_mol)
                gt_scaffold  = get_scaffold(gt_mol)
                if gen_scaffold and gt_scaffold:
                    scaffold_match = (gen_scaffold == gt_scaffold)

            # Molecular properties
            qed_val  = None
            logp_val = None
            mw_val   = None
            if valid:
                try:
                    qed_val  = round(QED.qed(gen_mol), 4)
                    logp_val = round(Descriptors.MolLogP(gen_mol), 3)
                    mw_val   = round(Descriptors.ExactMolWt(gen_mol), 2)
                except Exception:
                    pass

            all_results.append({
                "idx":            row.Index,
                "prompt":         row.description[:100],
                "gt_smiles":      gt_canonical,
                "gen_smiles":     gen_smiles,
                "gen_selfies":    gen_selfies,
                "valid":          valid,
                "exact_match":    exact_match,
                "scaffold_match": scaffold_match,
                "bleu":           round(bleu, 4),
                "maccs_fts":      round(maccs_sim, 4) if maccs_sim is not None else None,
                "morgan_fts":     round(morgan_sim, 4) if morgan_sim is not None else None,
                "levenshtein":    round(lev_sim, 4),
                "qed":            qed_val,
                "logp":           logp_val,
                "mw":             mw_val,
            })

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s "
          f"({elapsed/len(test_df):.2f}s per molecule)")

    # --- Save per-molecule results ---
    results_df = pd.DataFrame(all_results)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    csv_name = f"chebi20_eval_cfg{args.cfg}_temp{args.temp}_steps{args.steps}.csv"
    csv_path = output_dir / csv_name
    results_df.to_csv(csv_path, index=False)
    print(f"\nPer-molecule results: {csv_path}")

    # --- Compute aggregated metrics ---
    n_total    = len(results_df)
    n_valid    = results_df["valid"].sum()
    valid_df   = results_df[results_df["valid"]]

    validity      = n_valid / n_total
    exact_match   = results_df["exact_match"].mean()
    scaffold_match = results_df["scaffold_match"].mean()
    mean_bleu     = results_df["bleu"].mean()
    mean_lev      = results_df["levenshtein"].mean()

    # FTS metrics (only over valid pairs where GT is also valid)
    fts_df = valid_df.dropna(subset=["maccs_fts", "morgan_fts"])
    mean_maccs  = fts_df["maccs_fts"].mean()  if len(fts_df) > 0 else 0.0
    mean_morgan = fts_df["morgan_fts"].mean() if len(fts_df) > 0 else 0.0

    # Property stats
    mean_qed  = valid_df["qed"].mean()  if len(valid_df) > 0 else 0.0
    mean_logp = valid_df["logp"].mean() if len(valid_df) > 0 else 0.0
    mean_mw   = valid_df["mw"].mean()   if len(valid_df) > 0 else 0.0

    # Uniqueness (among valid)
    unique_smiles = valid_df["gen_smiles"].nunique() if len(valid_df) > 0 else 0
    uniqueness    = unique_smiles / n_valid if n_valid > 0 else 0.0

    # --- Print summary ---
    summary_lines = [
        "=" * 60,
        "  CheBI-20 EVALUATION RESULTS",
        f"  CFG={args.cfg}  temp={args.temp}  steps={args.steps}",
        "=" * 60,
        "",
        "  TGM-DLM Benchmark Metrics:",
        f"    Validity        : {validity:.4f}  ({n_valid}/{n_total})",
        f"    BLEU            : {mean_bleu:.4f}",
        f"    MACCS FTS       : {mean_maccs:.4f}",
        f"    Morgan FTS      : {mean_morgan:.4f}",
        "",
        "  Additional Metrics:",
        f"    Exact Match     : {exact_match:.4f}",
        f"    Scaffold Match  : {scaffold_match:.4f}",
        f"    Levenshtein Sim : {mean_lev:.4f}",
        f"    Uniqueness      : {uniqueness:.4f}  ({unique_smiles}/{n_valid})",
        "",
        "  Property Distributions (valid only):",
        f"    Mean QED        : {mean_qed:.4f}",
        f"    Mean LogP       : {mean_logp:.3f}",
        f"    Mean MW         : {mean_mw:.1f}",
        "",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    # Save summary
    summary_path = output_dir / "chebi20_eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Summary saved: {summary_path}")

    # --- Print comparison table format (for paper) ---
    print(f"\n  Paper table row:")
    print(f"  Model     | BLEU   | MACCS FTS | Morgan FTS | Validity")
    print(f"  Morpheus  | {mean_bleu:.3f}  | {mean_maccs:.3f}     "
          f"| {mean_morgan:.3f}      | {validity:.3f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate text-conditioned Morpheus on CheBI-20 test set"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "best_finetuned_model.pt"),
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="CFG scale (default: 3.0)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Generation batch size")
    parser.add_argument("--text_model", type=str, default=None,
                        help="Override text model name")

    args = parser.parse_args()
    evaluate(args)