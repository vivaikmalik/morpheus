"""
inference/evaluate_chebi20.py
------------------------------
Comprehensive evaluation with:
  - Standard MaskGIT denoising with CFG
  - ITERATIVE REFINEMENT: re-mask low-confidence tokens, denoise again
  - EOS TRUNCATION: post-hoc forward pass at t=0, truncate at max-EOS position
  - BEST-OF-N RERANKING: generate N candidates, pick best via contrastive scorer
  - TGM-DLM compatible metrics (corpus BLEU, MACCS, Morgan unhashed, InChI exact)

Usage:
    # Basic
    python inference/evaluate_chebi20.py \\
        --checkpoint checkpoints/best_finetuned_model.pt --cfg 3.0

    # With all enhancements
    python inference/evaluate_chebi20.py \\
        --checkpoint checkpoints/best_finetuned_model.pt \\
        --contrastive_ckpt checkpoints/contrastive_aligner.pt \\
        --cfg 3.0 --refine_passes 3 --rerank_n 8 --eos_truncate
"""

import argparse
import math
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, QED, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import AutoTokenizer, AutoModel
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev_distance
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# CONTRASTIVE SCORER (for best-of-N reranking)
# =============================================================================

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def load_contrastive_scorer(ckpt_path, device):
    """Load full ContrastiveAligner for reranking."""
    from contrastive.train_contrastive import (
        load_molecule_backbone, ContrastiveAligner)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    text_model_name = cfg["text_model"]

    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    tokenizer, mol_encoder, mol_config = load_molecule_backbone(
        Path("checkpoints/best_model.pt"), Path("chemical_tokenizer.json"), device)

    aligner = ContrastiveAligner(
        text_model=text_model, molecule_encoder=mol_encoder,
        text_hidden=text_model.config.hidden_size,
        mol_hidden=mol_config["hidden_size"],
        proj_dim=cfg["proj_dim"]).to(device)
    aligner.load_state_dict(ckpt["model_state"], strict=True)
    aligner.eval()

    text_tok = AutoTokenizer.from_pretrained(text_model_name)
    print(f"[scorer] loaded {ckpt_path.name}  text={text_model_name}")
    return aligner, text_tok, mol_config["max_length"]


# =============================================================================
# GENERATION WITH ITERATIVE REFINEMENT
# =============================================================================

@torch.no_grad()
def generate_cfg_with_refinement(model, tokenizer, text_tok, prompts, device,
                                   cfg_scale=3.0, num_steps=50, temperature=1.0,
                                   refine_passes=1, refine_mask_ratio=0.2):
    """Generate with optional iterative refinement passes."""
    max_length = model.token_embedding.num_embeddings  # cheap proxy; actual max from config
    # Get max_length from any block's RoPE cache
    max_length = 74  # fallback; passed via config externally
    B = len(prompts)

    text_inputs = text_tok(prompts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
    text_pad = (text_inputs["attention_mask"] == 0)
    text_embeds = model.get_text_embeddings(text_inputs["input_ids"],
                                              text_inputs["attention_mask"])
    null_embeds = model.null_token.expand(B, text_embeds.size(1), -1)

    input_ids = torch.full((B, max_length), tokenizer.mask_token_id,
                            dtype=torch.long, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    # Main denoising
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

        alpha_t = (torch.cos(t_val * math.pi / 2) ** 2).item()
        n_mask = int((1.0 - alpha_t) * max_length)
        if n_mask > 0 and step_idx < num_steps - 1:
            _, mi = torch.topk(confidence, n_mask, dim=-1, largest=False)
            sampled.scatter_(1, mi, tokenizer.mask_token_id)

        input_ids = sampled

    # Iterative refinement passes
    refine_steps = max(10, num_steps // 5)
    for _ in range(refine_passes - 1):
        # Score committed tokens at t=0
        step_zero = torch.zeros(B, 1, device=device)
        cond_l = model(input_ids, step_zero, text_embeds, text_pad)
        uncond_l = model(input_ids, step_zero, null_embeds, text_pad)
        logits_r = uncond_l + cfg_scale * (cond_l - uncond_l)
        probs_r = torch.softmax(logits_r / temperature, dim=-1)
        confidence = torch.gather(probs_r, 2, input_ids.unsqueeze(-1)).squeeze(-1)

        # Protect PAD positions (don't re-mask)
        pad_mask = (input_ids == tokenizer.pad_token_id)
        confidence = confidence.masked_fill(pad_mask, 1.0)

        # Re-mask the bottom refine_mask_ratio
        n_remask = max(1, int(refine_mask_ratio * max_length))
        _, low_idx = torch.topk(confidence, n_remask, dim=-1, largest=False)
        input_ids = input_ids.scatter(1, low_idx, tokenizer.mask_token_id)

        # Short denoise from t=refine_mask_ratio → 0
        t_vals_r = torch.linspace(refine_mask_ratio, 0.0, refine_steps, device=device)
        for step_idx, t_val in enumerate(t_vals_r):
            is_masked = (input_ids == tokenizer.mask_token_id)
            step_t = t_val.repeat(B).unsqueeze(-1)
            cond_l = model(input_ids, step_t, text_embeds, text_pad)
            uncond_l = model(input_ids, step_t, null_embeds, text_pad)
            logits = uncond_l + cfg_scale * (cond_l - uncond_l)

            if step_idx < refine_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float("-inf")

            probs = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            input_ids = torch.where(is_masked, sampled, input_ids)

    return input_ids.cpu().tolist()


# =============================================================================
# EOS POST-HOC TRUNCATION
# =============================================================================

@torch.no_grad()
def apply_eos_truncation(model, tokenizer, text_tok, all_ids, prompts, device,
                          batch_size=32):
    """Forward pass at t=0, find max-EOS-prob position, truncate."""
    truncated = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        b_ids = all_ids[start:end]
        b_prompts = prompts[start:end]
        B = len(b_prompts)

        max_len = max(len(x) for x in b_ids)
        ids_pad = torch.full((B, max_len), tokenizer.pad_token_id,
                              dtype=torch.long, device=device)
        for i, x in enumerate(b_ids):
            ids_pad[i, :len(x)] = torch.tensor(x, device=device)

        text_inputs = text_tok(b_prompts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
        text_pad = (text_inputs["attention_mask"] == 0)
        text_embeds = model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"])

        step_t = torch.zeros(B, 1, device=device)
        logits = model(ids_pad, step_t, text_embeds, text_pad)
        eos_probs = torch.softmax(logits, dim=-1)[:, :, tokenizer.eos_token_id]

        for mol_idx in range(B):
            ids = b_ids[mol_idx][:]
            probs = eos_probs[mol_idx].cpu().tolist()
            best_pos, best_prob = -1, -1.0
            for pos, (tok, p) in enumerate(zip(ids, probs)):
                if tok in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                            tokenizer.mask_token_id):
                    continue
                if p > best_prob:
                    best_prob, best_pos = p, pos
            if best_pos >= 0:
                ids = ids[:best_pos]
            truncated.append(ids)
    return truncated


# =============================================================================
# BEST-OF-N RERANKING
# =============================================================================

@torch.no_grad()
def rerank_candidates(scorer, scorer_tok, all_candidates, prompts, pad_id, device,
                       max_length):
    """Pick best of N candidates per prompt by contrastive similarity."""
    n_total = len(prompts)
    rerank_n = len(all_candidates)

    text_inputs = scorer_tok(prompts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
    z_text = scorer.encode_text(text_inputs)

    best = []
    for i in range(n_total):
        mol_ids = torch.full((rerank_n, max_length), pad_id, dtype=torch.long, device=device)
        for k, cands in enumerate(all_candidates):
            ids = cands[i][:max_length]
            mol_ids[k, :len(ids)] = torch.tensor(ids, device=device)

        z_mol = scorer.encode_molecule(mol_ids, pad_id)
        sims = (z_text[i:i + 1] * z_mol).sum(dim=-1)
        best_k = int(sims.argmax())
        best.append(all_candidates[best_k][i])

    return best


# =============================================================================
# METRICS
# =============================================================================

def maccs_t(m1, m2):
    return DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(m1), MACCSkeys.GenMACCSKeys(m2))


def morgan_unhashed(m1, m2, r=2):
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprint(m1, r), AllChem.GetMorganFingerprint(m2, r))


def rdk_t(m1, m2):
    return DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(m1), Chem.RDKFingerprint(m2))


def get_scaffold(mol):
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model with correct text encoder
    if args.contrastive_ckpt:
        text_model_ckpt = torch.load(args.contrastive_ckpt, map_location="cpu",
                                       weights_only=False)
        text_model_name = text_model_ckpt["config"]["text_model"]
    else:
        text_model_name = config.get("text_model", "BAAI/bge-large-en-v1.5")

    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
        num_heads=config["num_heads"], ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"], max_length=config["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=text_model_name,
        uncond_prob=0.0, dropout=0.0,
        load_text_encoder=True).to(device)

    if args.contrastive_ckpt:
        # Replace encoder with contrastively-trained one
        from training.finetune_text import load_contrastive_text_encoder
        encoder, _ = load_contrastive_text_encoder(Path(args.contrastive_ckpt), device)
        model.set_text_encoder(encoder)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Model loaded. Step {ckpt.get('step', '?')} val {ckpt.get('val_loss', '?')}")

    text_tok = AutoTokenizer.from_pretrained(text_model_name)

    # Load contrastive scorer for reranking (separate from text encoder)
    scorer, scorer_tok, mol_max = (None, None, config["max_length"])
    if args.rerank_n > 1 and args.contrastive_ckpt:
        scorer, scorer_tok, mol_max = load_contrastive_scorer(
            Path(args.contrastive_ckpt), device)

    # Test data
    test_df = pd.read_csv(ROOT / "data" / "chebi20_test.csv")
    print(f"Test: {len(test_df):,}")

    start = time.time()
    all_results, all_gt_smi, all_gen_smi = [], [], []

    for si in tqdm(range(0, len(test_df), args.batch_size), desc="Generating"):
        ei = min(si + args.batch_size, len(test_df))
        batch = test_df.iloc[si:ei]
        prompts = batch["description"].tolist()
        gts = batch["selfies"].tolist()

        # Generate N candidates per prompt
        all_candidates = []
        for _ in range(args.rerank_n):
            cand = generate_cfg_with_refinement(
                model, tokenizer, text_tok, prompts, device,
                cfg_scale=args.cfg, num_steps=args.steps, temperature=args.temp,
                refine_passes=args.refine_passes, refine_mask_ratio=0.2)
            all_candidates.append(cand)

        # Best-of-N reranking
        if args.rerank_n > 1 and scorer is not None:
            gen_ids = rerank_candidates(scorer, scorer_tok, all_candidates,
                                          prompts, tokenizer.pad_token_id,
                                          device, mol_max)
        else:
            gen_ids = all_candidates[0]

        # Post-hoc EOS truncation
        if args.eos_truncate:
            gen_ids = apply_eos_truncation(model, tokenizer, text_tok,
                                             gen_ids, prompts, device,
                                             batch_size=args.batch_size)

        # Decode and score
        for i, (gt_selfies, ids) in enumerate(zip(gts, gen_ids)):
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            ids = [t for t in ids if t not in (tokenizer.pad_token_id,
                                                  tokenizer.mask_token_id,
                                                  tokenizer.eos_token_id)]
            gen_selfies = tokenizer.decode(ids)
            try:
                gen_smi = sf.decoder(gen_selfies)
                gen_mol = Chem.MolFromSmiles(gen_smi)
                gen_canonical = Chem.MolToSmiles(gen_mol) if gen_mol else None
            except Exception:
                gen_mol, gen_canonical = None, None

            try:
                gt_smi = sf.decoder(gt_selfies)
                gt_mol = Chem.MolFromSmiles(gt_smi)
                gt_canonical = Chem.MolToSmiles(gt_mol) if gt_mol else None
            except Exception:
                gt_mol, gt_canonical = None, None

            valid = gen_mol is not None
            row = {"valid": valid, "gen_smiles": gen_canonical, "gt_smiles": gt_canonical}

            if valid and gt_mol:
                try:
                    row["maccs"] = round(maccs_t(gen_mol, gt_mol), 4)
                    row["morgan_u"] = round(morgan_unhashed(gen_mol, gt_mol), 4)
                    row["rdk"] = round(rdk_t(gen_mol, gt_mol), 4)
                except Exception:
                    pass
                try:
                    row["exact_inchi"] = inchi.MolToInchi(gen_mol) == inchi.MolToInchi(gt_mol)
                except Exception:
                    row["exact_inchi"] = False
                gs, ts = get_scaffold(gen_mol), get_scaffold(gt_mol)
                row["scaffold"] = gs == ts if gs and ts else False
                row["exact_smi"] = gen_canonical == gt_canonical
            else:
                row.update({"maccs": None, "morgan_u": None, "rdk": None,
                             "exact_inchi": False, "scaffold": False, "exact_smi": False})

            all_results.append(row)
            all_gt_smi.append(gt_canonical)
            all_gen_smi.append(gen_canonical)

    elapsed = time.time() - start
    print(f"\nTook {elapsed:.1f}s ({elapsed / len(test_df):.2f}s/mol)")

    df = pd.DataFrame(all_results)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / f"eval_cfg{args.cfg}_t{args.temp}_s{args.steps}_r{args.rerank_n}.csv",
               index=False)

    # Aggregate
    n = len(df)
    nv = df["valid"].sum()
    vdf = df[df["valid"]].dropna(subset=["maccs"])

    # TGM-DLM corpus BLEU on character-level SMILES
    refs, hyps = [], []
    for gt, gen in zip(all_gt_smi, all_gen_smi):
        if gt and gen:
            refs.append([list(gt)])
            hyps.append(list(gen))
    tgm_bleu = corpus_bleu(refs, hyps) if hyps else 0.0
    levs = [lev_distance(g, t) for g, t in zip(all_gen_smi, all_gt_smi) if g and t]
    tgm_lev = np.mean(levs) if levs else 0.0

    summary = f"""
{'=' * 65}
  CheBI-20 EVALUATION
  CFG={args.cfg}  temp={args.temp}  steps={args.steps}  refine={args.refine_passes}
  rerank_n={args.rerank_n}  eos_truncate={args.eos_truncate}
{'=' * 65}

  TGM-DLM COMPATIBLE METRICS:
    BLEU (char corpus)    : {tgm_bleu:.4f}
    MACCS FTS             : {vdf['maccs'].mean():.4f}
    Morgan FTS (unhashed) : {vdf['morgan_u'].mean():.4f}
    RDK FTS               : {vdf['rdk'].mean():.4f}
    Exact Match (InChI)   : {df['exact_inchi'].mean():.4f}
    Levenshtein           : {tgm_lev:.2f}
    Validity              : {nv / n:.4f}

  COMPARISON:
  Model        | BLEU   | MACCS  | Morgan | RDK    | Exact  | Valid
  Morpheus     | {tgm_bleu:.3f}  | {vdf['maccs'].mean():.3f}  | {vdf['morgan_u'].mean():.3f}  | {vdf['rdk'].mean():.3f}  | {df['exact_inchi'].mean():.3f}  | {nv / n:.3f}
  TGM-DLM      | 0.828  | 0.874  | 0.609  | 0.677  | 0.082  | 0.789
{'=' * 65}
"""
    print(summary)
    (out_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--contrastive_ckpt", default=None,
                     help="For both encoder swap and best-of-N reranking")
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--refine_passes", type=int, default=1,
                     help="Number of iterative refinement passes (1 = none)")
    p.add_argument("--rerank_n", type=int, default=1,
                     help="Generate N candidates and rerank (1 = no reranking)")
    p.add_argument("--eos_truncate", action="store_true",
                     help="Apply post-hoc EOS truncation")
    args = p.parse_args()
    evaluate(args)