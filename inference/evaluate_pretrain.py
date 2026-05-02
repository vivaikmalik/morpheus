"""
inference/evaluate_pretrain.py
-------------------------------
Evaluate the pretrained (unconditional) Morpheus model on ZINC250k metrics.

Generates N molecules from scratch and computes:
  - Validity (% that parse as valid SMILES)
  - Uniqueness (% unique among valid)
  - Novelty (% not in training set)
  - Property distributions (LogP, MW, QED, Rings, etc.)
  - Lipinski pass rate
  - Token retention (re-encode → SELFIES roundtrip)

Usage:
    python inference/evaluate_pretrain.py \
        --checkpoint checkpoints/best_model.pt \
        --num_molecules 1000 --num_steps 50 --temperature 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(checkpoint_path, tokenizer, device):
    print(f"Loading: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    print(f"  Step {ckpt.get('step', '?')}, val_loss {ckpt.get('val_loss', '?')}")

    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
        num_heads=config["num_heads"], ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"], max_length=config["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=None,  # unconditional
        dropout=0.0,
    ).to(device)

    # Filter keys (pretrained checkpoint won't have cross-attn keys)
    state = ckpt["model"]
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                  if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")
    return model, config


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate_batch(model, tokenizer, device, batch_size, max_length,
                    num_steps=50, temperature=1.0):
    input_ids = torch.full((batch_size, max_length), tokenizer.mask_token_id,
                            dtype=torch.long, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    for step_idx, t_val in enumerate(t_vals):
        is_last = (step_idx == num_steps - 1)
        step_t = t_val.repeat(batch_size).unsqueeze(-1)
        logits = model(input_ids, step_t)

        # Always block MASK from being a final output
        logits[:, :, tokenizer.mask_token_id] = float('-inf')

        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.distributions.Categorical(probs=probs).sample()

        if is_last:
            input_ids = sampled
            break

        confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
        alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
        n_mask = int((1.0 - alpha_t) * max_length)
        if n_mask > 0:
            _, mi = torch.topk(confidence, n_mask, dim=-1, largest=False)
            sampled.scatter_(1, mi, tokenizer.mask_token_id)

        input_ids = sampled

    return input_ids.cpu().tolist()


# =============================================================================
# MOLECULE ANALYSIS
# =============================================================================

def analyze(raw_ids, tokenizer):
    # Truncate at first EOS
    if tokenizer.eos_token_id in raw_ids:
        raw_ids = raw_ids[:raw_ids.index(tokenizer.eos_token_id)]
    # Strip remaining specials
    raw_ids = [i for i in raw_ids if i not in
               (tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id)]

    token_count = len(raw_ids)
    selfies_str = tokenizer.decode(raw_ids)

    result = {
        "selfies": selfies_str, "token_count": token_count,
        "valid": False, "smiles": "", "canonical_smiles": "",
        "logp": None, "mw": None, "heavy_atoms": None, "rings": None,
        "aromatic_rings": None, "hbd": None, "hba": None, "rot_bonds": None,
        "qed": None, "lipinski": None, "token_retention": None,
    }

    if token_count == 0:
        return result

    try:
        smiles = sf.decoder(selfies_str)
        result["smiles"] = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result

        canonical = Chem.MolToSmiles(mol)
        result["canonical_smiles"] = canonical
        result["valid"] = True

        # Token retention: re-encode canonical SMILES → SELFIES → tokens
        try:
            re_encoded = sf.encoder(canonical)
            re_tokens = len(list(sf.split_selfies(re_encoded)))
            result["token_retention"] = round(re_tokens / max(token_count, 1), 3)
        except Exception:
            pass

        result["logp"] = round(Descriptors.MolLogP(mol), 3)
        result["mw"] = round(Descriptors.MolWt(mol), 2)
        result["heavy_atoms"] = mol.GetNumHeavyAtoms()
        result["rings"] = rdMolDescriptors.CalcNumRings(mol)
        result["aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        result["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
        result["hba"] = rdMolDescriptors.CalcNumHBA(mol)
        result["rot_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        result["qed"] = round(QED.qed(mol), 3)
        result["lipinski"] = (result["mw"] <= 500 and result["logp"] <= 5
                                and result["hbd"] <= 5 and result["hba"] <= 10)
    except Exception as e:
        result["smiles"] = f"ERROR: {e}"

    return result


# =============================================================================
# NOVELTY (vs training set)
# =============================================================================

def compute_novelty(generated_smiles, training_smiles_set):
    """% of generated molecules not in training set."""
    if not generated_smiles:
        return 0.0
    novel = sum(1 for s in generated_smiles if s and s not in training_smiles_set)
    return novel / len(generated_smiles)


# =============================================================================
# DIVERSITY (Tanimoto pairwise on Morgan)
# =============================================================================

def compute_diversity(canonical_smiles, n_sample=200):
    """Mean pairwise Tanimoto distance — higher = more diverse."""
    if len(canonical_smiles) < 2:
        return 0.0

    sample = canonical_smiles[:n_sample] if len(canonical_smiles) > n_sample else canonical_smiles
    fps = []
    for s in sample:
        m = Chem.MolFromSmiles(s)
        if m:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

    if len(fps) < 2:
        return 0.0

    from rdkit import DataStructs
    n = len(fps)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)
    return float(np.mean(distances))


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results, training_smiles_set=None):
    valid = [r for r in results if r["valid"]]
    invalid = len(results) - len(valid)

    print(f"\n{'=' * 70}")
    print(f"  PRETRAINED MODEL EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Total generated: {len(results):,}")
    print(f"  Valid molecules: {len(valid):,} ({100 * len(valid) / len(results):.2f}%)")
    print(f"  Invalid:         {invalid:,} ({100 * invalid / len(results):.2f}%)")

    if not valid:
        print(f"{'=' * 70}")
        return

    canonical = [r["canonical_smiles"] for r in valid]
    unique = set(canonical)
    print(f"  Unique among valid: {len(unique):,} ({100 * len(unique) / len(valid):.2f}%)")

    if training_smiles_set is not None:
        novelty = compute_novelty(canonical, training_smiles_set)
        print(f"  Novelty (vs ZINC250k): {100 * novelty:.2f}%")

    diversity = compute_diversity(canonical)
    print(f"  Diversity (1-Tanimoto): {diversity:.4f}")

    print(f"\n  PROPERTY DISTRIBUTIONS (valid molecules)")
    print(f"  {'-' * 60}")
    props = {
        "LogP":            [r["logp"] for r in valid if r["logp"] is not None],
        "MW":              [r["mw"] for r in valid if r["mw"] is not None],
        "Heavy atoms":     [r["heavy_atoms"] for r in valid if r["heavy_atoms"] is not None],
        "Rings":           [r["rings"] for r in valid if r["rings"] is not None],
        "Aromatic rings":  [r["aromatic_rings"] for r in valid if r["aromatic_rings"] is not None],
        "HBD":             [r["hbd"] for r in valid if r["hbd"] is not None],
        "HBA":             [r["hba"] for r in valid if r["hba"] is not None],
        "Rot bonds":       [r["rot_bonds"] for r in valid if r["rot_bonds"] is not None],
        "QED":             [r["qed"] for r in valid if r["qed"] is not None],
        "Token retention": [r["token_retention"] for r in valid if r["token_retention"] is not None],
        "Token count":     [r["token_count"] for r in valid],
    }
    for name, values in props.items():
        if values:
            print(f"  {name:<18}: mean={np.mean(values):>7.2f}  "
                  f"std={np.std(values):>6.2f}  "
                  f"min={np.min(values):>6.2f}  max={np.max(values):>6.2f}")

    lipinski_pass = sum(1 for r in valid if r["lipinski"])
    print(f"\n  Lipinski pass: {lipinski_pass}/{len(valid)} "
          f"({100 * lipinski_pass / len(valid):.1f}%)")

    print(f"\n  SAMPLE VALID MOLECULES")
    print(f"  {'-' * 60}")
    for r in valid[:8]:
        print(f"  QED={r['qed']:.3f}  LogP={r['logp']:+.2f}  "
              f"MW={r['mw']:>6.1f}  Rings={r['rings']}  "
              f"Ret={r['token_retention']:.0%}  {r['canonical_smiles'][:50]}")

    print(f"{'=' * 70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--num_molecules", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/pretrained_eval.csv")
    parser.add_argument("--load_training_set", action="store_true",
                         help="Load ZINC250k for novelty check (slow first time)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    model, config = load_model(Path(args.checkpoint), tokenizer, device)
    max_length = config["max_length"]

    # Load training SMILES for novelty check
    training_smiles_set = None
    if args.load_training_set:
        print("Loading ZINC250k for novelty check...")
        zinc = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
        # Canonicalize
        training_smiles_set = set()
        for s in zinc["smiles"].tolist():
            try:
                m = Chem.MolFromSmiles(s)
                if m:
                    training_smiles_set.add(Chem.MolToSmiles(m))
            except Exception:
                pass
        print(f"  Loaded {len(training_smiles_set):,} canonical SMILES")

    # Generate
    print(f"\nGenerating {args.num_molecules:,} molecules "
          f"(steps={args.num_steps}, T={args.temperature})...")
    start = time.time()
    all_raw = []
    for batch_start in tqdm(range(0, args.num_molecules, args.batch_size),
                              desc="Batches"):
        bs = min(args.batch_size, args.num_molecules - batch_start)
        raw = generate_batch(model, tokenizer, device, bs, max_length,
                              args.num_steps, args.temperature)
        all_raw.extend(raw)
    elapsed = time.time() - start
    print(f"  Generated in {elapsed:.1f}s ({elapsed / args.num_molecules:.3f}s/mol)")

    # Analyze
    print(f"\nAnalyzing {len(all_raw):,} molecules...")
    results = [analyze(ids, tokenizer) for ids in tqdm(all_raw, desc="Analyzing")]

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary
    print_summary(results, training_smiles_set)


if __name__ == "__main__":
    main()