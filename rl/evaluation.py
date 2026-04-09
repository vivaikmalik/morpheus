"""
ChEBI-20 evaluation suite for text-conditioned molecular generation.

Metrics (matching the finetuning notebook):
  - Validity (%)
  - SMILES Exact Match (%)
  - SMILES BLEU (character-level, method1 smoothing)
  - SMILES Levenshtein distance (raw + normalised)
  - MACCS FTS (MACCS-keys Tanimoto)
  - RDK FTS (RDKit topological Tanimoto)
  - Morgan FTS (Morgan r=2 Tanimoto)
  - FCD (Fréchet ChemNet Distance — lower is better)
  - Token Retention Rate
"""

import os
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import selfies as sf
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein as lev

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl.generation import generate_batch_cfg_no_grad

_smoother = SmoothingFunction().method1


# =============================================================================
# CHEBI-20 DATA
# =============================================================================

def load_chebi20_test(cache_dir=None):
    """
    Download ChEBI-20 test split (MolT5 format).
    Returns DataFrame with columns: prompt, response (SMILES).
    """
    import requests

    base_url = (
        "https://raw.githubusercontent.com/blender-nlp/MolT5"
        "/main/ChEBI-20_data"
    )
    fname = "test.txt"
    url = f"{base_url}/{fname}"

    # Check cache
    if cache_dir:
        cached = Path(cache_dir) / "chebi20_test.csv"
        if cached.exists():
            return pd.read_csv(cached)

    print(f"Downloading ChEBI-20 test split …")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text), sep="\t", header=0,
        names=["cid", "smiles", "description"],
    ).dropna(subset=["smiles", "description"])

    result = pd.DataFrame({
        "prompt": df["description"].values,
        "response": df["smiles"].values,
    })

    if cache_dir:
        cached = Path(cache_dir) / "chebi20_test.csv"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        result.to_csv(cached, index=False)
        print(f"  Cached → {cached}")

    print(f"  ChEBI-20 test: {len(result):,} samples")
    return result


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _canonical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _selfies_to_smiles(selfies_str):
    try:
        raw = sf.decoder(selfies_str)
        return _canonical(raw)
    except Exception:
        return None


def _smiles_to_selfies(smiles):
    try:
        can = _canonical(smiles)
        return sf.encoder(can) if can else None
    except Exception:
        return None


def _maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return MACCSkeys.GenMACCSKeys(mol) if mol else None


def _rdk_fp(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return RDKFingerprint(mol) if mol else None


def _morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits) if mol else None


def _tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2) if (fp1 and fp2) else 0.0


def _smiles_bleu(ref, gen):
    """Character-level BLEU on SMILES strings."""
    return sentence_bleu(
        [list(ref)], list(gen) if gen else [],
        smoothing_function=_smoother,
    )


def _token_retention(ref_smiles, gen_selfies):
    """Fraction of GT SELFIES tokens present in generated SELFIES."""
    try:
        ref_sf = _smiles_to_selfies(ref_smiles)
        if not ref_sf:
            return 0.0
        ref_tok = list(sf.split_selfies(ref_sf))
        gen_tok = list(sf.split_selfies(gen_selfies))
        if not ref_tok:
            return 0.0
        gen_tok_set = set(gen_tok)
        return sum(1 for t in ref_tok if t in gen_tok_set) / len(ref_tok)
    except Exception:
        return 0.0


# =============================================================================
# GENERATION HELPER
# =============================================================================

def _generate_selfies_batch(model, tokenizer, hf_tokenizer, device,
                            model_config, prompts, cfg_scale, num_steps,
                            temperature):
    """Generate SELFIES strings for a batch of prompts using CFG."""
    n = len(prompts)
    max_length = model_config["max_length"]

    enc = hf_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(device)
    pad_mask = (enc["attention_mask"] == 0)

    with torch.no_grad():
        text_embeds = model.get_text_embeddings(
            enc["input_ids"], enc["attention_mask"]
        )

    gen_ids = generate_batch_cfg_no_grad(
        model, tokenizer, device, text_embeds, pad_mask,
        model_config, num_steps=num_steps,
        temperature=temperature, cfg_scale=cfg_scale,
    )

    results = []
    for i in range(n):
        tok = gen_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in tok:
            tok = tok[:tok.index(tokenizer.eos_token_id)]
        results.append(tokenizer.decode(tok))
    return results


# =============================================================================
# FULL CHEBI-20 EVALUATION
# =============================================================================

def run_chebi20_eval(model, tokenizer, hf_tokenizer, device, model_config,
                     eval_df=None, batch_size=32, cfg_scale=3.0,
                     num_steps=32, temperature=1.0, max_rows=None,
                     cache_dir=None, compute_fcd=True):
    """
    Run the full ChEBI-20 evaluation suite.

    Parameters
    ----------
    model, tokenizer, hf_tokenizer, device, model_config : standard
    eval_df    : DataFrame with 'prompt' and 'response' (SMILES). If None,
                 downloads ChEBI-20 test split.
    batch_size : generation batch size
    cfg_scale, num_steps, temperature : generation hyperparameters
    max_rows   : limit evaluation to first N rows (for debugging)
    cache_dir  : directory to cache ChEBI-20 download
    compute_fcd: whether to compute FCD (requires fcd_torch)

    Returns
    -------
    metrics : dict of aggregate metrics
    results_df : DataFrame with per-sample results
    """
    model.eval()

    if eval_df is None:
        eval_df = load_chebi20_test(cache_dir=cache_dir)

    if max_rows:
        eval_df = eval_df.head(max_rows)

    prompts_list = eval_df["prompt"].tolist()
    ref_smiles_list = eval_df["response"].tolist()

    # ── Generate ──────────────────────────────────────────────────────
    gen_selfies_list = []
    for start in range(0, len(prompts_list), batch_size):
        batch = prompts_list[start:start + batch_size]
        gen_selfies_list.extend(
            _generate_selfies_batch(
                model, tokenizer, hf_tokenizer, device, model_config,
                batch, cfg_scale, num_steps, temperature,
            )
        )

    # ── Per-sample metrics ────────────────────────────────────────────
    records = []
    for ref_smi, gen_sf_str in zip(ref_smiles_list, gen_selfies_list):
        ref_can = _canonical(ref_smi) or ref_smi
        gen_smi = _selfies_to_smiles(gen_sf_str)

        valid = gen_smi is not None
        exact = valid and (gen_smi == ref_can)

        records.append({
            "reference_smiles": ref_can,
            "generated_selfies": gen_sf_str,
            "generated_smiles": gen_smi or "",
            "valid": valid,
            "smiles_exact_match": exact,
            "smiles_bleu": _smiles_bleu(ref_can, gen_smi) if valid else 0.0,
            "smiles_levenshtein": lev.distance(gen_smi or "", ref_can),
            "smiles_norm_levenshtein": (
                lev.distance(gen_smi or "", ref_can)
                / max(len(gen_smi or ""), len(ref_can), 1)
            ),
            "maccs_fts": _tanimoto(_maccs_fp(ref_can), _maccs_fp(gen_smi)),
            "rdk_fts": _tanimoto(_rdk_fp(ref_can), _rdk_fp(gen_smi)),
            "morgan_fts": _tanimoto(_morgan_fp(ref_can), _morgan_fp(gen_smi)),
            "token_retention_rate": _token_retention(ref_can, gen_sf_str),
        })

    results_df = pd.DataFrame(records)
    valid_mask = results_df["valid"]

    # ── FCD ───────────────────────────────────────────────────────────
    fcd_value = float("nan")
    if compute_fcd:
        try:
            from fcd_torch import FCD as _FCD

            def _fcd_safe(smiles):
                return (bool(smiles) and len(smiles) >= 2
                        and Chem.MolFromSmiles(smiles) is not None)

            pairs = [
                (r["generated_smiles"], r["reference_smiles"])
                for r in records
                if r["valid"] and _fcd_safe(r["generated_smiles"])
                and _fcd_safe(r["reference_smiles"])
            ]
            if len(pairs) >= 2:
                gen_valid, ref_valid = zip(*pairs)
                _fcd_scorer = _FCD(device=str(device), n_jobs=0)
                fcd_value = _fcd_scorer(list(gen_valid), list(ref_valid))
                print(f"  FCD: {fcd_value:.4f} ({len(pairs):,} pairs)")
            else:
                print(f"  FCD skipped — too few valid pairs ({len(pairs)})")
        except ImportError:
            print("  FCD skipped — fcd_torch not installed")
        except Exception as e:
            print(f"  FCD failed: {e}")

    # ── Aggregate metrics ─────────────────────────────────────────────
    metrics = {
        "validity": valid_mask.mean() * 100,
        "smiles_exact_match": results_df["smiles_exact_match"].mean() * 100,
        "smiles_bleu": (
            results_df.loc[valid_mask, "smiles_bleu"].mean()
            if valid_mask.any() else 0.0
        ),
        "smiles_levenshtein": results_df["smiles_levenshtein"].mean(),
        "smiles_norm_levenshtein": results_df["smiles_norm_levenshtein"].mean(),
        "maccs_fts": results_df["maccs_fts"].mean(),
        "rdk_fts": results_df["rdk_fts"].mean(),
        "morgan_fts": results_df["morgan_fts"].mean(),
        "token_retention_rate": results_df["token_retention_rate"].mean(),
        "fcd": fcd_value,
    }

    model.train()
    return metrics, results_df


def print_chebi20_metrics(metrics, num_samples=None, num_valid=None):
    """Pretty-print the ChEBI-20 metric dict."""
    def _fmt(v):
        return f"{v:>9.4f}" if not (isinstance(v, float) and np.isnan(v)) else "      N/A"

    print()
    print("=" * 68)
    print("  ChEBI-20 EVALUATION")
    print("=" * 68)
    print(f"  {'Validity (%)':<42s}  {_fmt(metrics['validity'])}")
    print(f"  {'SMILES Exact Match (%)':<42s}  {_fmt(metrics['smiles_exact_match'])}")
    print(f"  {'SMILES BLEU (avg, valid only)':<42s}  {_fmt(metrics['smiles_bleu'])}")
    print(f"  {'SMILES Levenshtein (avg)':<42s}  {_fmt(metrics['smiles_levenshtein'])}")
    print(f"  {'SMILES Norm. Levenshtein (avg)':<42s}  {_fmt(metrics['smiles_norm_levenshtein'])}")
    print(f"  {'MACCS FTS (avg)':<42s}  {_fmt(metrics['maccs_fts'])}")
    print(f"  {'RDK FTS (avg)':<42s}  {_fmt(metrics['rdk_fts'])}")
    print(f"  {'Morgan FTS (avg)':<42s}  {_fmt(metrics['morgan_fts'])}")
    print(f"  {'Token Retention Rate (avg)':<42s}  {_fmt(metrics['token_retention_rate'])}")
    print(f"  {'FCD (lower is better)':<42s}  {_fmt(metrics['fcd'])}")
    print("-" * 68)
    if num_samples is not None:
        print(f"  Total samples   : {num_samples:>8,}")
    if num_valid is not None and num_samples is not None:
        print(f"  Valid molecules : {num_valid:>8,} / {num_samples:,}")
    print("=" * 68)
