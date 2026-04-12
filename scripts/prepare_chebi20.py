"""
scripts/prepare_chebi20.py
--------------------------
Downloads CheBI-20 from HuggingFace and saves train/val/test CSVs
with SMILES (no SELFIES conversion).

Filters out molecules whose SMILES exceed max_length tokens.

Usage:
    python scripts/prepare_chebi20.py

Output:
    data/chebi20_train.csv
    data/chebi20_val.csv
    data/chebi20_test.csv

Each CSV has columns: smiles, description
"""

import sys
import pandas as pd
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.smilesTokenizer import SmilesTokenizer


def process_split(df, tokenizer, max_length, split_name):
    print(f"\n--- Processing {split_name} ---")
    print(f"  Raw rows: {len(df):,}")

    # Find columns
    smiles_col = next((c for c in ["SMILES", "smiles"] if c in df.columns), None)
    desc_col = next((c for c in ["description", "text"] if c in df.columns), None)
    if not smiles_col or not desc_col:
        raise ValueError(f"Missing columns. Found: {list(df.columns)}")

    df = df.copy()

    # Validate SMILES with RDKit
    valid = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            tokens = tokenizer.encode(canonical)
            # +1 for EOS token
            if len(tokens) + 1 <= max_length:
                # Check all tokens are in vocab
                if len(tokens) == len(tokenizer.tokenize(canonical)):
                    valid.append({
                        "smiles": canonical,
                        "description": row[desc_col],
                    })

    result = pd.DataFrame(valid)
    print(f"  After validation + length filter: {len(result):,} "
          f"(dropped {len(df) - len(result):,})")

    return result


def main():
    vocab_path = ROOT / "smiles_vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: {vocab_path} not found.")
        print("Run: python scripts/build_smiles_vocab.py")
        sys.exit(1)

    tokenizer = SmilesTokenizer(str(vocab_path))
    max_length = 256  # match TGM-DLM

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max sequence length: {max_length}")

    print("\nDownloading CheBI-20...")
    from datasets import load_dataset
    dataset = load_dataset("liupf/ChEBI-20-MM")

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        df = dataset[split_name].to_pandas()
        result = process_split(df, tokenizer, max_length, split_name)

        out_name = split_name.replace("validation", "val")
        out_path = data_dir / f"chebi20_{out_name}.csv"
        result.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(result):,} rows)")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    import re
    regex = re.compile(SmilesTokenizer.PATTERN)
    for f in sorted(data_dir.glob("chebi20_*.csv")):
        df = pd.read_csv(f)
        lens = df["smiles"].apply(lambda s: len(regex.findall(s)) + 1)  # +EOS
        print(f"  {f.name}: {len(df):,} rows, "
              f"median tokens: {lens.median():.0f}, max: {lens.max()}")


if __name__ == "__main__":
    main()