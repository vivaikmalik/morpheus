"""
novelty_check.py
Computes the novelty of generated_molecules.csv against the ZINC250k training set.
Novelty = fraction of valid generated molecules whose canonical SMILES does NOT
appear anywhere in the (canonicalized) ZINC250k dataset.
"""

import pandas as pd
from rdkit import Chem
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.parent
GENERATED_PATH = PROJECT_ROOT / "outputs" / "generated_molecules.csv"

# ---------------------------------------------------------------------------
# Step 1 — Load ZINC250k and build a set of canonical SMILES
# ---------------------------------------------------------------------------
print("Loading ZINC250k from HuggingFace...")
zinc = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
print(f"  {len(zinc):,} molecules loaded")

print("Canonicalizing ZINC250k SMILES (this may take a moment)...")
training_set = set()
failed = 0
for smi in zinc["smiles"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        training_set.add(Chem.MolToSmiles(mol))
    else:
        failed += 1

print(f"  {len(training_set):,} canonical SMILES in training set")
if failed:
    print(f"  {failed} ZINC250k SMILES failed RDKit parsing (skipped)")

# ---------------------------------------------------------------------------
# Step 2 — Load generated molecules
# ---------------------------------------------------------------------------
print(f"\nLoading generated molecules from {GENERATED_PATH.name}...")
gen = pd.read_csv(GENERATED_PATH)
print(f"  {len(gen):,} total rows")

valid = gen[gen["valid"] == True].copy()
print(f"  {len(valid):,} valid molecules")

if len(valid) == 0:
    print("No valid molecules to evaluate.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Step 3 — Check novelty
# ---------------------------------------------------------------------------
# canonical_smiles column was computed by archive/old_code/inference_evaluate_v1.py via RDKit
novel     = valid[~valid["canonical_smiles"].isin(training_set)]
not_novel = valid[ valid["canonical_smiles"].isin(training_set)]

novelty_pct = 100.0 * len(novel) / len(valid)

# ---------------------------------------------------------------------------
# Step 4 — Report
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  NOVELTY REPORT")
print(f"{'='*55}")
print(f"  Total generated         : {len(gen)}")
print(f"  Valid                   : {len(valid)} ({100*len(valid)/len(gen):.1f}%)")
print(f"  Novel (not in ZINC250k) : {len(novel)}")
print(f"  In training set         : {len(not_novel)}")
print(f"\n  Novelty                 : {novelty_pct:.1f}%")
print(f"{'='*55}\n")

if len(not_novel) > 0:
    print("Molecules found in training set:")
    for smi in not_novel["canonical_smiles"].tolist():
        print(f"  {smi}")
