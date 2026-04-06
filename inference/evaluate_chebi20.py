"""
inference/evaluate_chebi20.py
------------------------------
Evaluates the text-conditioned Morpheus model on CheBI-20 test set.

Reports TWO sets of metrics:
  1. "Ours" — SELFIES token-level BLEU, hashed Morgan FP, normalized Levenshtein
  2. "TGM-DLM compatible" — character-level corpus BLEU on SMILES, unhashed Morgan FP,
     raw Levenshtein on SMILES, InChI exact match (matches their ev.py exactly)

Usage:
    python inference/evaluate_chebi20.py
    python inference/evaluate_chebi20.py --checkpoint checkpoints/best_finetuned_model.pt
    python inference/evaluate_chebi20.py --cfg 2.0 --temp 0.8 --steps 50
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
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer

# TGM-DLM uses nltk corpus_bleu
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev_distance

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# OUR METRICS (SELFIES token-level)
# =============================================================================

def selfies_tokens(s):
    try:
        return list(sf.split_selfies(s))
    except Exception:
        return list(s)


def ngram_counts(tokens, n):
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i+n])] += 1
    return counts


def sentence_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    if not hypothesis_tokens or not reference_tokens:
        return 0.0
    bp = min(1.0, np.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ng  = ngram_counts(reference_tokens, n)
        hyp_ng  = ngram_counts(hypothesis_tokens, n)
        if not hyp_ng:
            log_precisions.append(float('-inf'))
            continue
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total   = sum(hyp_ng.values())
        precision = clipped / total if total > 0 else 0.0
        log_precisions.append(np.log(precision + 1e-12))
    avg_log_precision = np.mean(log_precisions)
    return bp * np.exp(avg_log_precision)


def levenshtein_dist(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def normalized_levenshtein(ref_tokens, hyp_tokens):
    dist = levenshtein_dist(ref_tokens, hyp_tokens)
    max_len = max(len(ref_tokens), len(hyp_tokens), 1)
    return 1.0 - dist / max_len


def maccs_tanimoto(mol1, mol2):
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def morgan_tanimoto_hashed(mol1, mol2, radius=2, n_bits=2048):
    """Our metric: hashed bit vector Morgan FP."""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# =============================================================================
# TGM-DLM COMPATIBLE METRICS (matches their ev.py exactly)
# =============================================================================

def morgan_tanimoto_unhashed(mol1, mol2, radius=2):
    """TGM-DLM metric: unhashed count-based Morgan FP (no bit collisions)."""
    fp1 = AllChem.GetMorganFingerprint(mol1, radius)
    fp2 = AllChem.GetMorganFingerprint(mol2, radius)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def rdk_tanimoto(mol1, mol2):
    """TGM-DLM metric: RDK fingerprint Tanimoto."""
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def inchi_exact_match(mol1, mol2):
    """TGM-DLM metric: exact match via InChI comparison."""
    try:
        return inchi.MolToInchi(mol1) == inchi.MolToInchi(mol2)
    except Exception:
        return False


def compute_tgmdlm_corpus_bleu(gt_smiles_list, gen_smiles_list):
    """
    TGM-DLM metric: character-level corpus BLEU on SMILES.
    Exactly matches their ev.py:
        gt_tokens = [c for c in gt]
        corpus_bleu(references, hypotheses)
    """
    references = []
    hypotheses = []
    for gt, gen in zip(gt_smiles_list, gen_smiles_list):
        if gt is None or gen is None:
            continue
        gt_tokens = [c for c in gt]
        gen_tokens = [c for c in gen]
        references.append([gt_tokens])
        hypotheses.append(gen_tokens)

    if not hypotheses:
        return 0.0
    return corpus_bleu(references, hypotheses)


def compute_tgmdlm_levenshtein(gt_smiles_list, gen_smiles_list):
    """
    TGM-DLM metric: raw character-level Levenshtein distance on SMILES.
    They report mean raw distance, not normalized.
    """
    levs = []
    for gt, gen in zip(gt_smiles_list, gen_smiles_list):
        if gt is None or gen is None:
            continue
        levs.append(lev_distance(gen, gt))
    return np.mean(levs) if levs else 0.0


def get_scaffold(mol):
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
    B = len(prompts)

    text_enc = text_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=max_text_len, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            text_enc["input_ids"], text_enc["attention_mask"]
        )
    text_pad = (text_enc["attention_mask"] == 0)

    input_ids = torch.full(
        (B, max_length), mol_tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(B).unsqueeze(-1)

            logits_cond   = model(input_ids, step_t, text_embeds, text_pad)
            logits_uncond = model(input_ids, step_t, None, None)

            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            if step_idx < num_steps - 1:
                logits[:, :, mol_tokenizer.mask_token_id] = float('-inf')

            probs   = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()

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
    ids = list(ids)
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
        uncond_prob     = 0.0,
        dropout         = 0.0,
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

    mol_tokenizer  = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.text_model or "allenai/scibert_scivocab_uncased"
    )

    model, config = load_finetuned_model(args.checkpoint, mol_tokenizer, device)
    max_length  = config["max_length"]
    max_text_len = config.get("max_text_len", 256)

    test_path = ROOT / "data" / "chebi20_test.csv"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run: python scripts/prepare_chebi20.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"\nTest set: {len(test_df):,} molecules")

    print(f"\nGenerating with CFG={args.cfg}  temp={args.temp}  steps={args.steps}...")
    start_time = time.time()

    # ================================================================= #
    # GENERATION                                                         #
    # ================================================================= #

    all_results = []
    # Collect for TGM-DLM corpus-level metrics
    all_gt_smiles  = []
    all_gen_smiles = []

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

            gt_mol = None
            gt_canonical = None
            if gt_smiles:
                gt_mol = Chem.MolFromSmiles(gt_smiles)
                if gt_mol:
                    gt_canonical = Chem.MolToSmiles(gt_mol)

            valid = gen_mol is not None

            # --- OUR METRICS ---

            # BLEU (SELFIES token-level, sentence-level)
            ref_tokens = selfies_tokens(gt_selfies)
            hyp_tokens = selfies_tokens(gen_selfies) if gen_selfies else []
            bleu_ours = sentence_bleu(ref_tokens, hyp_tokens)

            # Levenshtein (normalized, SELFIES token-level)
            lev_ours = normalized_levenshtein(ref_tokens, hyp_tokens)

            # Fingerprint similarities
            maccs_sim = None
            morgan_hashed = None
            morgan_unhashed = None
            rdk_sim = None
            if valid and gt_mol is not None:
                try:
                    maccs_sim       = maccs_tanimoto(gen_mol, gt_mol)
                    morgan_hashed   = morgan_tanimoto_hashed(gen_mol, gt_mol)
                    morgan_unhashed = morgan_tanimoto_unhashed(gen_mol, gt_mol)
                    rdk_sim         = rdk_tanimoto(gen_mol, gt_mol)
                except Exception:
                    pass

            # Exact match (canonical SMILES)
            exact_match_smiles = False
            if valid and gt_canonical:
                exact_match_smiles = (gen_smiles == gt_canonical)

            # Exact match (InChI — TGM-DLM method)
            exact_match_inchi = False
            if valid and gt_mol is not None:
                exact_match_inchi = inchi_exact_match(gen_mol, gt_mol)

            # Scaffold match
            scaffold_match = False
            if valid and gt_mol is not None:
                gen_scaffold = get_scaffold(gen_mol)
                gt_scaffold  = get_scaffold(gt_mol)
                if gen_scaffold and gt_scaffold:
                    scaffold_match = (gen_scaffold == gt_scaffold)

            # Molecular properties
            qed_val = logp_val = mw_val = None
            if valid:
                try:
                    qed_val  = round(QED.qed(gen_mol), 4)
                    logp_val = round(Descriptors.MolLogP(gen_mol), 3)
                    mw_val   = round(Descriptors.ExactMolWt(gen_mol), 2)
                except Exception:
                    pass

            # Collect SMILES for corpus-level BLEU
            all_gt_smiles.append(gt_canonical)
            all_gen_smiles.append(gen_smiles)

            all_results.append({
                "idx":              row.Index,
                "prompt":           row.description[:100],
                "gt_smiles":        gt_canonical,
                "gen_smiles":       gen_smiles,
                "gen_selfies":      gen_selfies,
                "valid":            valid,
                "exact_match":      exact_match_smiles,
                "exact_match_inchi": exact_match_inchi,
                "scaffold_match":   scaffold_match,
                "bleu_ours":        round(bleu_ours, 4),
                "maccs_fts":        round(maccs_sim, 4) if maccs_sim is not None else None,
                "morgan_hashed":    round(morgan_hashed, 4) if morgan_hashed is not None else None,
                "morgan_unhashed":  round(morgan_unhashed, 4) if morgan_unhashed is not None else None,
                "rdk_fts":          round(rdk_sim, 4) if rdk_sim is not None else None,
                "levenshtein_ours": round(lev_ours, 4),
                "qed":              qed_val,
                "logp":             logp_val,
                "mw":               mw_val,
            })

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s "
          f"({elapsed/len(test_df):.2f}s per molecule)")

    # ================================================================= #
    # COMPUTE AGGREGATED METRICS                                         #
    # ================================================================= #

    results_df = pd.DataFrame(all_results)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    csv_name = f"chebi20_eval_cfg{args.cfg}_temp{args.temp}_steps{args.steps}.csv"
    csv_path = output_dir / csv_name
    results_df.to_csv(csv_path, index=False)
    print(f"\nPer-molecule results: {csv_path}")

    n_total = len(results_df)
    n_valid = results_df["valid"].sum()
    valid_df = results_df[results_df["valid"]]

    validity = n_valid / n_total

    # --- OUR METRICS (aggregated) ---
    mean_bleu_ours = results_df["bleu_ours"].mean()
    mean_lev_ours  = results_df["levenshtein_ours"].mean()
    exact_match_smiles = results_df["exact_match"].mean()
    scaffold_match = results_df["scaffold_match"].mean()

    fts_df = valid_df.dropna(subset=["maccs_fts", "morgan_hashed"])
    mean_maccs         = fts_df["maccs_fts"].mean() if len(fts_df) > 0 else 0.0
    mean_morgan_hashed = fts_df["morgan_hashed"].mean() if len(fts_df) > 0 else 0.0

    # --- TGM-DLM COMPATIBLE METRICS (corpus-level) ---
    tgm_bleu = compute_tgmdlm_corpus_bleu(all_gt_smiles, all_gen_smiles)
    tgm_lev  = compute_tgmdlm_levenshtein(all_gt_smiles, all_gen_smiles)
    exact_match_inchi = results_df["exact_match_inchi"].mean()

    fts_df2 = valid_df.dropna(subset=["morgan_unhashed", "rdk_fts"])
    mean_morgan_unhashed = fts_df2["morgan_unhashed"].mean() if len(fts_df2) > 0 else 0.0
    mean_rdk             = fts_df2["rdk_fts"].mean() if len(fts_df2) > 0 else 0.0

    # --- PROPERTY STATS ---
    mean_qed  = valid_df["qed"].mean()  if len(valid_df) > 0 else 0.0
    mean_logp = valid_df["logp"].mean() if len(valid_df) > 0 else 0.0
    mean_mw   = valid_df["mw"].mean()   if len(valid_df) > 0 else 0.0

    unique_smiles = valid_df["gen_smiles"].nunique() if len(valid_df) > 0 else 0
    uniqueness = unique_smiles / n_valid if n_valid > 0 else 0.0

    # ================================================================= #
    # PRINT SUMMARY                                                      #
    # ================================================================= #

    summary_lines = [
        "=" * 65,
        "  CheBI-20 EVALUATION RESULTS",
        f"  CFG={args.cfg}  temp={args.temp}  steps={args.steps}",
        "=" * 65,
        "",
        "  TGM-DLM COMPATIBLE METRICS (matches their ev.py):",
        f"    BLEU (char-level corpus)  : {tgm_bleu:.4f}",
        f"    MACCS FTS                 : {mean_maccs:.4f}",
        f"    Morgan FTS (unhashed)     : {mean_morgan_unhashed:.4f}",
        f"    RDK FTS                   : {mean_rdk:.4f}",
        f"    Exact Match (InChI)       : {exact_match_inchi:.4f}",
        f"    Levenshtein (raw SMILES)  : {tgm_lev:.2f}",
        f"    Validity                  : {validity:.4f}",
        "",
        "  OUR METRICS:",
        f"    BLEU (SELFIES token, sent): {mean_bleu_ours:.4f}",
        f"    Morgan FTS (hashed 2048)  : {mean_morgan_hashed:.4f}",
        f"    Levenshtein (norm SELFIES): {mean_lev_ours:.4f}",
        f"    Exact Match (SMILES)      : {exact_match_smiles:.4f}",
        f"    Scaffold Match            : {scaffold_match:.4f}",
        f"    Uniqueness                : {uniqueness:.4f}  ({unique_smiles}/{n_valid})",
        "",
        "  PROPERTY DISTRIBUTIONS (valid only):",
        f"    Mean QED  : {mean_qed:.4f}",
        f"    Mean LogP : {mean_logp:.3f}",
        f"    Mean MW   : {mean_mw:.1f}",
        "",
        "=" * 65,
        "",
        "  COMPARISON TABLE (TGM-DLM compatible metrics):",
        "  Model        | BLEU   | MACCS  | Morgan | RDK    | Exact  | Validity",
        f"  Morpheus     | {tgm_bleu:.3f}  | {mean_maccs:.3f}  | {mean_morgan_unhashed:.3f}  "
        f"| {mean_rdk:.3f}  | {exact_match_inchi:.3f}  | {validity:.3f}",
        f"  TGM-DLM      | 0.828  | 0.874  | 0.609  | 0.677  | 0.082  | 0.789",
        "",
        "=" * 65,
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    summary_path = output_dir / "chebi20_eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Summary saved: {summary_path}")


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
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_model", type=str, default=None)

    args = parser.parse_args()
    evaluate(args)