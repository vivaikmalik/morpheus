"""
diversity_check.py
Computes chemical diversity metrics for generated_molecules.csv:
  1. Average pairwise Tanimoto similarity (sampled)
  2. Internal diversity (1 - avg similarity)
  3. Unique Bemis-Murcko scaffolds
  4. Scaffold diversity (unique scaffolds / total valid)
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

GENERATED_PATH = Path(__file__).parent.parent / "outputs" / "generated_molecules.csv"
SAMPLE_PAIRS   = 5000
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load valid molecules
# ---------------------------------------------------------------------------
print(f"Loading {GENERATED_PATH.name}...")
gen   = pd.read_csv(GENERATED_PATH)
valid = gen[gen["valid"] == True]["canonical_smiles"].tolist()
print(f"  {len(gen)} total rows, {len(valid)} valid molecules\n")

if len(valid) == 0:
    print("No valid molecules to evaluate.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Build RDKit mol objects and Morgan fingerprints
# ---------------------------------------------------------------------------
print("Computing Morgan fingerprints (radius=2, 2048 bits)...")
mols = []
fps  = []
failed = 0
for smi in valid:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed += 1
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    mols.append(mol)
    fps.append(fp)

print(f"  {len(fps)} fingerprints computed ({failed} failed)")

# ---------------------------------------------------------------------------
# 1 & 2: Average pairwise Tanimoto similarity + internal diversity
# ---------------------------------------------------------------------------
n = len(fps)
max_pairs = n * (n - 1) // 2

if max_pairs <= SAMPLE_PAIRS:
    # Compute all pairs exactly
    print(f"\nComputing all {max_pairs} pairwise Tanimoto similarities...")
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    mode = "exact"
else:
    # Sample random pairs
    print(f"\nSampling {SAMPLE_PAIRS} random pairs from {max_pairs} possible...")
    indices = [(i, j)
               for i in range(n)
               for j in range(i + 1, n)]
    sampled = random.sample(indices, SAMPLE_PAIRS)
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in sampled]
    mode = f"sampled {SAMPLE_PAIRS}/{max_pairs}"

avg_sim       = float(np.mean(sims))
int_diversity = 1.0 - avg_sim

# ---------------------------------------------------------------------------
# 3 & 4: Bemis-Murcko scaffolds
# ---------------------------------------------------------------------------
print("Computing Bemis-Murcko scaffolds...")
scaffolds = []
for mol in mols:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(Chem.MolToSmiles(scaffold))
    except Exception:
        scaffolds.append("")   # molecule with no rings gets empty scaffold

unique_scaffolds  = set(scaffolds)
scaffold_diversity = len(unique_scaffolds) / len(mols)

# Break down: molecules with no ring system get "" as scaffold
empty_scaffold = sum(1 for s in scaffolds if s == "")
ring_scaffolds = unique_scaffolds - {""}

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  DIVERSITY REPORT  ({len(mols)} valid molecules)")
print(f"{'='*55}")
print(f"\n  --- Fingerprint similarity ({mode}) ---")
print(f"  Avg pairwise Tanimoto  : {avg_sim:.4f}")
print(f"  Std                    : {float(np.std(sims)):.4f}")
print(f"  Min                    : {float(np.min(sims)):.4f}")
print(f"  Max                    : {float(np.max(sims)):.4f}")
print(f"\n  Internal diversity     : {int_diversity:.4f}")

print(f"\n  --- Bemis-Murcko scaffolds ---")
print(f"  Total molecules        : {len(mols)}")
print(f"  Unique scaffolds       : {len(unique_scaffolds)}")
print(f"    of which ring-based  : {len(ring_scaffolds)}")
print(f"    acyclic (no rings)   : {empty_scaffold}")
print(f"  Scaffold diversity     : {scaffold_diversity:.4f}  "
      f"({len(unique_scaffolds)}/{len(mols)})")

print(f"\n  --- Top 10 most common scaffolds ---")
from collections import Counter
scaffold_counts = Counter(scaffolds)
for scaffold, count in scaffold_counts.most_common(10):
    label = scaffold if scaffold else "(acyclic)"
    print(f"  {count:>4}x  {label}")

print(f"\n{'='*55}\n")
