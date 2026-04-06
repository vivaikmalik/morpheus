"""
prepare_data.py
---------------
Downloads Mol-Instructions (description_guided_molecule_design split),
filters to molecules our tokenizer can handle, and writes:
    data/train.csv
    data/val.csv
    data/test.csv

Each CSV has two columns: prompt, response (SELFIES string).

Usage:
    python data/prepare_data.py
"""

import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent          # molgen/data/
PROJECT_ROOT = SCRIPT_DIR.parent              # molgen/
RAW_DIR      = SCRIPT_DIR / "raw"

ZIP_URL  = (
    "https://huggingface.co/datasets/zjunlp/Mol-Instructions"
    "/resolve/main/data/Molecule-oriented_Instructions.zip"
)
ZIP_PATH      = RAW_DIR / "Molecule-oriented_Instructions.zip"
TARGET_FILE   = "description_guided_molecule_design.json"
JSON_PATH     = RAW_DIR / TARGET_FILE
TOKENIZER_PATH = PROJECT_ROOT / "chemical_tokenizer.json"

MAX_CONTENT_TOKENS = 73   # +1 EOS = 74 total
VAL_FRACTION       = 0.10
RANDOM_SEED        = 42

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if JSON_PATH.exists():
        print(f"[download] {TARGET_FILE} already present, skipping download.")
        return

    if not ZIP_PATH.exists():
        print(f"[download] Fetching {ZIP_URL} ...")
        def _progress(count, block, total):
            pct = min(count * block / total * 100, 100)
            print(f"\r  {pct:5.1f}%", end="", flush=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=_progress)
        print()
        print(f"[download] Saved → {ZIP_PATH}")
    else:
        print(f"[download] Zip already present at {ZIP_PATH}, skipping fetch.")

    print(f"[download] Extracting {TARGET_FILE} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # The target file may sit inside a subdirectory inside the zip
        matches = [n for n in zf.namelist() if n.endswith(TARGET_FILE)]
        if not matches:
            raise FileNotFoundError(
                f"{TARGET_FILE} not found inside zip. "
                f"Available: {zf.namelist()[:10]}"
            )
        member = matches[0]
        # Extract directly to RAW_DIR with a flat name
        with zf.open(member) as src, open(JSON_PATH, "wb") as dst:
            dst.write(src.read())
    print(f"[download] Extracted → {JSON_PATH}")


# ---------------------------------------------------------------------------
# 2. Parse JSON
# ---------------------------------------------------------------------------
def parse(json_path: Path) -> pd.DataFrame:
    print(f"[parse] Loading {json_path} ...")
    with open(json_path, "r") as f:
        entries = json.load(f)
    print(f"[parse] Total entries in JSON: {len(entries):,}")

    rows = []
    for e in entries:
        instruction = e.get("instruction", "").strip()
        inp         = e.get("input", "").strip()
        output      = e.get("output", "")
        metadata    = e.get("metadata", {})
        split       = metadata.get("split", "train") if isinstance(metadata, dict) else "train"

        if inp:
            prompt = f"{instruction}\nInput: {inp}"
        else:
            prompt = instruction

        rows.append({"prompt": prompt, "response": output, "split": split})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Filter
# ---------------------------------------------------------------------------
def load_vocab(tokenizer_path: Path) -> set:
    with open(tokenizer_path, "r") as f:
        data = json.load(f)
    return set(data["token_to_id"].keys())


def filter_df(df: pd.DataFrame, vocab: set):
    n_start = len(df)

    # Drop empty / NaN responses
    df = df[df["response"].notna() & (df["response"].str.strip() != "")]
    n_after_empty = len(df)
    print(f"[filter] Dropped {n_start - n_after_empty:,} empty/NaN responses")

    # Tokenise and filter
    unknown_drop = []
    length_drop  = []
    keep_mask    = []

    for idx, row in df.iterrows():
        try:
            tokens = list(sf.split_selfies(row["response"]))
        except Exception:
            unknown_drop.append(idx)
            keep_mask.append(False)
            continue

        if any(t not in vocab for t in tokens):
            unknown_drop.append(idx)
            keep_mask.append(False)
        elif len(tokens) > MAX_CONTENT_TOKENS:
            length_drop.append(idx)
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    df = df[keep_mask]

    # Report unknown-token drops with examples
    print(f"[filter] Dropped {len(unknown_drop):,} rows (unknown tokens or bad SELFIES)")
    if unknown_drop:
        examples = df.index.isin(unknown_drop[:3])  # won't match — use raw indices
        # Re-fetch from pre-filter df for display
        pass   # examples printed below separately

    print(f"[filter] Dropped {len(length_drop):,}  rows (>{MAX_CONTENT_TOKENS} tokens)")
    print(f"[filter] Remaining: {len(df):,} rows")

    return df, unknown_drop, length_drop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 0. Sanity-check dependencies
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    # 1. Download
    download()

    # 2. Parse
    df = parse(JSON_PATH)

    # 3. Filter — work on a copy that still has the original indices for reporting
    vocab    = load_vocab(TOKENIZER_PATH)
    df_raw   = df.copy().reset_index(drop=True)
    n_total  = len(df_raw)

    # Drop empty
    df_clean = df_raw[df_raw["response"].notna() & (df_raw["response"].str.strip() != "")].copy()
    n_empty  = n_total - len(df_clean)

    # Token filtering
    unknown_examples = []
    unknown_count    = 0
    length_count     = 0
    keep_indices     = []

    for i, row in df_clean.iterrows():
        try:
            tokens = list(sf.split_selfies(row["response"]))
        except Exception:
            unknown_count += 1
            if len(unknown_examples) < 3:
                unknown_examples.append(row["response"][:80])
            continue

        unk = [t for t in tokens if t not in vocab]
        if unk:
            unknown_count += 1
            if len(unknown_examples) < 3:
                unknown_examples.append(
                    f"{row['response'][:60]}  ← unknown: {unk[:3]}"
                )
        elif len(tokens) > MAX_CONTENT_TOKENS:
            length_count += 1
        else:
            keep_indices.append(i)

    df_filtered = df_clean.loc[keep_indices].copy()

    # 4. Split by metadata split field, then carve val from train
    df_test  = df_filtered[df_filtered["split"] == "test"].copy()
    df_train_full = df_filtered[df_filtered["split"] != "test"].copy()

    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
    )

    # Drop the helper split column before saving
    for frame in (df_train, df_val, df_test):
        frame.drop(columns=["split"], inplace=True)

    # 5. Save
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(SCRIPT_DIR / "train.csv", index=False)
    df_val.to_csv(SCRIPT_DIR  / "val.csv",   index=False)
    df_test.to_csv(SCRIPT_DIR / "test.csv",  index=False)

    # 6. Print stats
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total entries in JSON          : {n_total:>8,}")
    print(f"Dropped (empty/NaN response)   : {n_empty:>8,}")
    print(f"Dropped (unknown tokens / bad) : {unknown_count:>8,}")
    if unknown_examples:
        for ex in unknown_examples:
            print(f"    e.g. {ex}")
    print(f"Dropped (length > {MAX_CONTENT_TOKENS} tokens)   : {length_count:>8,}")
    print(f"Remaining after filter         : {len(df_filtered):>8,}")
    print()
    print(f"train.csv                      : {len(df_train):>8,} rows")
    print(f"val.csv                        : {len(df_val):>8,} rows")
    print(f"test.csv                       : {len(df_test):>8,} rows")
    print()
    print("─" * 60)
    print("Sample rows from train.csv (prompt | response):")
    print("─" * 60)
    sample = df_train.sample(n=min(3, len(df_train)), random_state=RANDOM_SEED)
    for _, row in sample.iterrows():
        prompt_preview   = row["prompt"][:120].replace("\n", " ↵ ")
        response_preview = row["response"][:80]
        print(f"\n  PROMPT  : {prompt_preview}")
        print(f"  RESPONSE: {response_preview}")
    print()
    print(f"Saved to {SCRIPT_DIR}/")


if __name__ == "__main__":
    main()
