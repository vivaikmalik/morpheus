import sys
import torch
import argparse
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel

# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
TOKENIZER_PATH  = Path(__file__).parent.parent / "chemical_tokenizer.json"
OUTPUT_PATH     = Path(__file__).parent.parent / "outputs" / "generated_molecules.csv"

GEN_CONFIG = {
    "num_molecules" : 100,    # total molecules to generate
    "batch_size"    : 32,     # generate in batches to avoid OOM
    "num_steps"     : 50,     # MaskGIT denoising steps — more = better quality
    "temperature"   : 1.2,    # > 1.0 = more diverse, < 1.0 = more greedy
    "max_length"    : 74,     # must match training max_length
}

# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_model(checkpoint_path, tokenizer, device):
    print(f"Loading checkpoint: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint["config"]

    print(f"  Trained for {checkpoint['step']:,} steps "
          f"(epoch {checkpoint['epoch']})")
    print(f"  Val loss at save: {checkpoint.get('val_loss', 'N/A')}")

    model = MolecularDiffusionModel(
        vocab_size   = config["vocab_size"],
        hidden_size  = config["hidden_size"],
        num_heads    = config["num_heads"],
        ffn_dim      = config["ffn_dim"],
        num_layers   = config["num_layers"],
        max_length   = config["max_length"],
        pad_token_id = tokenizer.pad_token_id,
        dropout      = 0.0,   # disable dropout at inference
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total:,}")

    return model, config

# =============================================================================
# REVERSE DIFFUSION (MASKGIT SAMPLING LOOP)
# =============================================================================

def generate_batch(model, tokenizer, device, batch_size, config):
    """
    Run the full MaskGIT reverse diffusion loop for one batch.

    Returns:
        List of raw token ID lists (before EOS truncation)
    """
    max_length  = config["max_length"]
    num_steps   = GEN_CONFIG["num_steps"]
    temperature = GEN_CONFIG["temperature"]

    # --- Step 1: Initialize — fully masked at t=1.0 ---
    input_ids = torch.full(
        (batch_size, max_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
        device=device
    )

    # Linearly spaced timesteps from 1.0 down to 0.0
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            is_last_step = (step_idx == num_steps - 1)

            # --- Step 2: Predict — forward pass ---
            step_t = t_val.repeat(batch_size).unsqueeze(-1)   # (B, 1)
            logits = model(input_ids, step_t)                  # (B, max_length, vocab_size)

            # --- Step 3: Filter vocabulary ---

            # Block MASK at all steps including the last
            # The model should never output MASK as a final token
            logits[:, :, tokenizer.mask_token_id] = float('-inf')

            # EOS is always allowed — model can terminate naturally

            # --- Temperature scaling ---
            scaled_logits = logits / temperature

            # --- Sample from distribution ---
            probs   = torch.softmax(scaled_logits, dim=-1)     # (B, max_length, vocab_size)
            sampled = torch.distributions.Categorical(
                probs=probs
            ).sample()                                          # (B, max_length)

            if is_last_step:
                # Final step — lock everything in, no re-masking
                input_ids = sampled
                break

            # --- Step 3: Confidence of each sampled token ---
            confidence = torch.gather(
                probs, 2, sampled.unsqueeze(-1)
            ).squeeze(-1)                                       # (B, max_length)

            # --- Step 4: Lock & Mask ---
            # How many tokens should remain masked at this timestep?
            # alpha_t → 1 means almost everything masked (early steps)
            # alpha_t → 0 means almost nothing masked (late steps)
            alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            # Keep most confident tokens locked in
            # Re-mask the least confident positions
            if num_to_mask > 0:
                _, mask_indices = torch.topk(
                    confidence,
                    k=min(num_to_mask, max_length),
                    dim=-1,
                    largest=False   # least confident = smallest probability
                )
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            # --- Step 5: Iterate ---
            input_ids = sampled

    return input_ids.cpu().tolist()

# =============================================================================
# MOLECULE DECODING & ANALYSIS
# =============================================================================

def decode_and_analyze(raw_ids, tokenizer):
    """
    Decode a single token ID sequence into a molecule and compute properties.

    Returns a dict with all computed properties.
    """
    # --- Truncate at first EOS ---
    if tokenizer.eos_token_id in raw_ids:
        raw_ids = raw_ids[:raw_ids.index(tokenizer.eos_token_id)]

    # --- Remove any remaining special tokens ---
    raw_ids = [
        id for id in raw_ids
        if id not in (
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.eos_token_id
        )
    ]

    token_count = len(raw_ids)
    selfies_str = tokenizer.decode(raw_ids)

    result = {
        "selfies"        : selfies_str,
        "token_count"    : token_count,
        "valid"          : False,
        "smiles"         : "",
        "canonical_smiles": "",
        "logp"           : None,
        "mw"             : None,
        "heavy_atoms"    : None,
        "rings"          : None,
        "aromatic_rings" : None,
        "hbd"            : None,
        "hba"            : None,
        "rot_bonds"      : None,
        "qed"            : None,
        "lipinski"       : None,
        "token_retention": None,
    }

    if token_count == 0:
        return result

    try:
        # --- Decode SELFIES → SMILES ---
        smiles = sf.decoder(selfies_str)
        result["smiles"] = smiles

        # --- Validate with RDKit ---
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result

        canonical = Chem.MolToSmiles(mol)
        result["canonical_smiles"] = canonical
        result["valid"] = True

        # --- Token retention ---
        # Re-encode the canonical SMILES back to SELFIES
        # Measures how much of our generated sequence was chemically meaningful
        re_encoded   = sf.encoder(canonical)
        reenc_tokens = len(list(sf.split_selfies(re_encoded)))
        result["token_retention"] = round(reenc_tokens / max(token_count, 1), 3)

        # --- Physicochemical properties ---
        result["logp"]          = round(Descriptors.MolLogP(mol), 3)
        result["mw"]            = round(Descriptors.MolWt(mol), 2)
        result["heavy_atoms"]   = mol.GetNumHeavyAtoms()
        result["rings"]         = rdMolDescriptors.CalcNumRings(mol)
        result["aromatic_rings"]= rdMolDescriptors.CalcNumAromaticRings(mol)
        result["hbd"]           = rdMolDescriptors.CalcNumHBD(mol)
        result["hba"]           = rdMolDescriptors.CalcNumHBA(mol)
        result["rot_bonds"]     = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # --- Drug-likeness ---
        result["qed"] = round(QED.qed(mol), 3)

        # Lipinski's Rule of Five
        result["lipinski"] = (
            result["mw"]  <= 500 and
            result["logp"] <= 5   and
            result["hbd"] <= 5   and
            result["hba"] <= 10
        )

    except Exception as e:
        result["smiles"] = f"ERROR: {e}"

    return result

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(results):
    valid   = [r for r in results if r["valid"]]
    invalid = [r for r in results if not r["valid"]]

    print(f"\n{'='*65}")
    print(f"  GENERATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Total generated  : {len(results)}")
    print(f"  Valid molecules  : {len(valid)} ({100*len(valid)/len(results):.1f}%)")
    print(f"  Invalid          : {len(invalid)} ({100*len(invalid)/len(results):.1f}%)")

    if not valid:
        print("  No valid molecules to analyze.")
        print(f"{'='*65}\n")
        return

    # Uniqueness — how many unique canonical SMILES?
    unique_smiles = set(r["canonical_smiles"] for r in valid)
    print(f"  Unique valid     : {len(unique_smiles)} "
          f"({100*len(unique_smiles)/len(valid):.1f}% of valid)")

    print(f"\n  --- Property distributions (valid molecules) ---")

    props = {
        "LogP"           : [r["logp"]           for r in valid if r["logp"]           is not None],
        "MW"             : [r["mw"]             for r in valid if r["mw"]             is not None],
        "Heavy atoms"    : [r["heavy_atoms"]     for r in valid if r["heavy_atoms"]    is not None],
        "Rings"          : [r["rings"]           for r in valid if r["rings"]          is not None],
        "QED"            : [r["qed"]             for r in valid if r["qed"]            is not None],
        "Token retention": [r["token_retention"] for r in valid if r["token_retention"]is not None],
        "Token count"    : [r["token_count"]     for r in valid],
    }

    for name, values in props.items():
        if values:
            print(f"  {name:<18}: "
                  f"mean={np.mean(values):.2f}  "
                  f"std={np.std(values):.2f}  "
                  f"min={np.min(values):.2f}  "
                  f"max={np.max(values):.2f}")

    lipinski_pass = sum(1 for r in valid if r["lipinski"])
    print(f"\n  Lipinski pass    : {lipinski_pass}/{len(valid)} "
          f"({100*lipinski_pass/len(valid):.1f}%)")

    print(f"\n  --- Sample valid molecules ---")
    for r in valid[:5]:
        print(f"  LogP: {r['logp']:+.2f} | "
              f"MW: {r['mw']:>6.1f} | "
              f"QED: {r['qed']:.3f} | "
              f"Rings: {r['rings']} | "
              f"Retention: {r['token_retention']:.0%} | "
              f"{r['canonical_smiles'][:50]}")

    print(f"{'='*65}\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Morpheus — Molecular Generation")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                        help="Path to model checkpoint")
    parser.add_argument("--num_molecules", type=int, default=GEN_CONFIG["num_molecules"],
                        help="Number of molecules to generate")
    parser.add_argument("--num_steps", type=int, default=GEN_CONFIG["num_steps"],
                        help="MaskGIT denoising steps")
    parser.add_argument("--temperature", type=float, default=GEN_CONFIG["temperature"],
                        help="Sampling temperature")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output CSV path")
    args = parser.parse_args()

    # Apply CLI overrides
    GEN_CONFIG["num_molecules"] = args.num_molecules
    GEN_CONFIG["num_steps"]     = args.num_steps
    GEN_CONFIG["temperature"]   = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Generating {GEN_CONFIG['num_molecules']} molecules "
          f"| steps={GEN_CONFIG['num_steps']} "
          f"| temperature={GEN_CONFIG['temperature']}\n")

    # --- Load tokenizer and model ---
    tokenizer = ChemicalTokenizer(TOKENIZER_PATH)
    checkpoint_path = Path(args.checkpoint)
    model, config = load_model(checkpoint_path, tokenizer, device)

    # Override max_length from checkpoint config
    GEN_CONFIG["max_length"] = config["max_length"]

    # --- Generate in batches ---
    all_raw_ids = []
    num_mols    = GEN_CONFIG["num_molecules"]
    batch_size  = GEN_CONFIG["batch_size"]
    num_batches = (num_mols + batch_size - 1) // batch_size

    print(f"Generating {num_batches} batches of {batch_size}...")

    for batch_idx in range(num_batches):
        current_batch_size = min(
            batch_size,
            num_mols - batch_idx * batch_size
        )

        print(f"  Batch {batch_idx + 1}/{num_batches} "
              f"({current_batch_size} molecules)...", end=" ")

        raw_ids = generate_batch(
            model, tokenizer, device,
            current_batch_size, config
        )
        all_raw_ids.extend(raw_ids)
        print("done")

    # --- Decode and analyze ---
    print(f"\nDecoding and analyzing {len(all_raw_ids)} molecules...")
    results = [decode_and_analyze(ids, tokenizer) for ids in all_raw_ids]

    # --- Print summary ---
    print_summary(results)

    # --- Save to CSV ---
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Columns: {list(df.columns)}\n")


if __name__ == "__main__":
    main()
