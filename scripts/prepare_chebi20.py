"""
scripts/prepare_chebi20.py
--------------------------
Downloads CheBI-20 from HuggingFace, converts SMILES → SELFIES,
filters for tokenizer compatibility, and saves train/val/test CSVs.

Usage:
    python scripts/prepare_chebi20.py

Output:
    data/chebi20_train.csv
    data/chebi20_val.csv
    data/chebi20_test.csv

Each CSV has columns: selfies, description, smiles
"""

import sys
import json
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer


def smiles_to_selfies(smiles):
    """Convert SMILES to SELFIES. Returns None if conversion fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        selfies = sf.encoder(canonical)
        if selfies is None:
            return None
        return selfies
    except Exception:
        return None


def fits_tokenizer(selfies_str, tokenizer, max_length=74):
    """
    Check if every token in the SELFIES string exists in our vocabulary
    and the sequence fits within max_length (including EOS).
    """
    try:
        tokens = list(sf.split_selfies(selfies_str))
    except Exception:
        return False

    # Check all tokens are in vocab
    for tok in tokens:
        if tok not in tokenizer.token_to_id:
            return False

    # Check length: tokens + EOS must fit in max_length
    if len(tokens) + 1 > max_length:
        return False

    return True


def process_split(df, tokenizer, max_length, split_name):
    """Process one split: convert SMILES → SELFIES, filter, return clean df."""
    print(f"\n--- Processing {split_name} split ---")
    print(f"  Raw rows: {len(df):,}")

    # Find the SMILES and description columns
    # CheBI-20 uses 'SMILES' and 'description' (or 'text')
    smiles_col = None
    for candidate in ["SMILES", "smiles", "Smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break

    desc_col = None
    for candidate in ["description", "text", "Description"]:
        if candidate in df.columns:
            desc_col = candidate
            break

    if smiles_col is None or desc_col is None:
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError(f"Could not find SMILES column ({smiles_col}) "
                         f"or description column ({desc_col})")

    # Convert SMILES → SELFIES
    df = df.copy()
    df["selfies"] = df[smiles_col].apply(smiles_to_selfies)

    # Drop failed conversions
    before = len(df)
    df = df.dropna(subset=["selfies"])
    print(f"  After SMILES→SELFIES conversion: {len(df):,} "
          f"(dropped {before - len(df):,})")

    # Filter for tokenizer compatibility
    mask = df["selfies"].apply(lambda s: fits_tokenizer(s, tokenizer, max_length))
    before = len(df)
    df = df[mask]
    print(f"  After vocab/length filter: {len(df):,} "
          f"(dropped {before - len(df):,})")

    # Rename columns to standard format
    result = pd.DataFrame({
        "selfies":     df["selfies"].values,
        "description": df[desc_col].values,
        "smiles":      df[smiles_col].values,
    })

    return result


def main():
    # Load tokenizer
    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    max_length = 74

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Max sequence length:  {max_length}")

    # Download CheBI-20 from HuggingFace
    print("\nDownloading CheBI-20 from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("liupf/ChEBI-20-MM")
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets --break-system-packages")
        sys.exit(1)

    # Create output directory
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Process each split
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            print(f"  WARNING: split '{split_name}' not found, skipping")
            continue

        df = dataset[split_name].to_pandas()
        result = process_split(df, tokenizer, max_length, split_name)

        # Save
        out_name = split_name.replace("validation", "val")
        out_path = data_dir / f"chebi20_{out_name}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(result):,} rows)")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for f in sorted(data_dir.glob("chebi20_*.csv")):
        df = pd.read_csv(f)
        selfies_lens = df["selfies"].apply(
            lambda s: len(list(sf.split_selfies(s)))
        )
        print(f"  {f.name}: {len(df):,} rows, "
              f"median SELFIES length: {selfies_lens.median():.0f}, "
              f"max: {selfies_lens.max()}")


if __name__ == "__main__":
    main()