"""
scripts/build_smiles_vocab.py
------------------------------
Builds a SMILES vocabulary from ZINC250k + CheBI-20 datasets.

Collects all unique SMILES tokens using the same regex as TGM-DLM,
then saves as smiles_vocab.json.

Usage:
    python scripts/build_smiles_vocab.py

Output:
    smiles_vocab.json  (in project root)
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.smilesTokenizer import SmilesTokenizer


def main():
    all_smiles = []

    # 1. Load ZINC250k
    print("Loading ZINC250k...")
    try:
        zinc_df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
        all_smiles.extend(zinc_df["smiles"].dropna().tolist())
        print(f"  ZINC250k: {len(zinc_df):,} molecules")
    except Exception as e:
        print(f"  WARNING: Could not load ZINC250k: {e}")

    # 2. Load CheBI-20 (all splits)
    print("Loading CheBI-20...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("liupf/ChEBI-20-MM")
        for split in ["train", "validation", "test"]:
            if split in dataset:
                df = dataset[split].to_pandas()
                smiles_col = None
                for c in ["SMILES", "smiles", "Smiles"]:
                    if c in df.columns:
                        smiles_col = c
                        break
                if smiles_col:
                    all_smiles.extend(df[smiles_col].dropna().tolist())
                    print(f"  CheBI-20 {split}: {len(df):,} molecules")
    except Exception as e:
        print(f"  WARNING: Could not load CheBI-20: {e}")

    print(f"\nTotal SMILES collected: {len(all_smiles):,}")

    # 3. Build vocab
    save_path = ROOT / "smiles_vocab.json"
    SmilesTokenizer.build_vocab(all_smiles, str(save_path))

    # 4. Quick test
    tokenizer = SmilesTokenizer(str(save_path))
    test_smiles = ["CC(=O)O", "c1ccccc1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
    print(f"\nQuick test:")
    for smi in test_smiles:
        ids = tokenizer.encode(smi)
        back = tokenizer.decode(ids)
        match = "✅" if smi == back else "❌"
        print(f"  {match} {smi} → {ids} → {back}")


if __name__ == "__main__":
    main()