"""
scripts/recompute_atom_bleu.py
-------------------------------
Compute atom-level BLEU-2 and BLEU-4 from existing eval CSVs that were
generated before atom BLEU was added to the evaluation pipeline.

Usage:
    # Print summary table for all outputs/*.csv
    python3 scripts/recompute_atom_bleu.py

    # Specific files
    python3 scripts/recompute_atom_bleu.py outputs/chebi20_27M_contrastive_eval_cfg1.5_t0.6_s50.csv

    # Also write atom_bleu2 / atom_bleu4 columns back to each CSV
    python3 scripts/recompute_atom_bleu.py --add-columns
"""

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Atom-level SMILES tokenizer (TGM-DLM convention)
# ---------------------------------------------------------------------------
_ATOM_PATTERN = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p'
    r'|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)

def _atom_tokenize(smiles: str) -> list:
    return _ATOM_PATTERN.findall(smiles) if isinstance(smiles, str) else []


# ---------------------------------------------------------------------------
# BLEU helpers (self-contained, no project imports)
# ---------------------------------------------------------------------------
def _bleu_n(ref_tokens, hyp_tokens, n):
    if not ref_tokens or not hyp_tokens or len(hyp_tokens) < n:
        return 0.0
    ref_ng = {}
    for i in range(len(ref_tokens) - n + 1):
        g = tuple(ref_tokens[i:i+n])
        ref_ng[g] = ref_ng.get(g, 0) + 1
    hyp_ng = {}
    for i in range(len(hyp_tokens) - n + 1):
        g = tuple(hyp_tokens[i:i+n])
        hyp_ng[g] = hyp_ng.get(g, 0) + 1
    clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
    return clipped / sum(hyp_ng.values())


def _sentence_bleu(ref_tokens, hyp_tokens, max_n=4):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precs = [_bleu_n(ref_tokens, hyp_tokens, n) for n in range(1, max_n + 1)]
    if all(p == 0 for p in precs):
        return 0.0
    log_avg = sum(math.log(p) if p > 0 else -1e9 for p in precs) / max_n
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return bp * math.exp(log_avg)


def compute_atom_bleu(df: pd.DataFrame):
    """Return (atom_bleu2_series, atom_bleu4_series) for a DataFrame."""
    ab2, ab4 = [], []
    for _, row in df.iterrows():
        ref = _atom_tokenize(str(row.get("gt_smiles", "") or ""))
        hyp = _atom_tokenize(str(row.get("gen_smiles", "") or ""))
        ab2.append(round(_sentence_bleu(ref, hyp, max_n=2), 4))
        ab4.append(round(_sentence_bleu(ref, hyp, max_n=4), 4))
    return ab2, ab4


def process_file(path: Path, add_columns: bool) -> dict | None:
    df = pd.read_csv(path)
    if "gt_smiles" not in df.columns or "gen_smiles" not in df.columns:
        print(f"  SKIP {path.name}: missing gt_smiles or gen_smiles columns")
        return None

    ab2, ab4 = compute_atom_bleu(df)

    if add_columns:
        df["atom_bleu2"] = ab2
        df["atom_bleu4"] = ab4
        df.to_csv(path, index=False)

    # Also read existing char BLEU if present
    char_b2 = df["bleu2"].mean() if "bleu2" in df.columns else float("nan")
    char_b4 = df["bleu4"].mean() if "bleu4" in df.columns else float("nan")

    import statistics
    return {
        "file"        : path.name,
        "n"           : len(df),
        "char_bleu2"  : round(char_b2, 4),
        "char_bleu4"  : round(char_b4, 4),
        "atom_bleu2"  : round(sum(ab2) / len(ab2), 4),
        "atom_bleu4"  : round(sum(ab4) / len(ab4), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute atom-level BLEU for eval CSVs")
    parser.add_argument("files", nargs="*",
                        help="CSV files to process (default: outputs/chebi20_*_eval_*.csv)")
    parser.add_argument("--add-columns", action="store_true",
                        help="Write atom_bleu2 and atom_bleu4 columns back to each CSV (in-place)")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else \
            sorted(Path("outputs").glob("chebi20_*_eval_*.csv"))

    if not paths:
        print("No CSV files found.")
        sys.exit(1)

    results = []
    for p in paths:
        r = process_file(p, args.add_columns)
        if r:
            results.append(r)
            flag = " [+cols]" if args.add_columns else ""
            print(f"  processed: {p.name}{flag}")

    if not results:
        print("No valid files processed.")
        sys.exit(1)

    # Print summary table
    col_w = max(len(r["file"]) for r in results)
    header = f"{'File':<{col_w}}  {'N':>5}  {'charB2':>8}  {'charB4':>8}  {'atomB2':>8}  {'atomB4':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['file']:<{col_w}}  {r['n']:>5}  "
              f"{r['char_bleu2']:>8.4f}  {r['char_bleu4']:>8.4f}  "
              f"{r['atom_bleu2']:>8.4f}  {r['atom_bleu4']:>8.4f}")
    print("=" * len(header))
    print()
    print("Reference — TGM-DLM (AAAI 2024): atom BLEU-2 ≈ 0.826")
    if args.add_columns:
        print("atom_bleu2 / atom_bleu4 columns written back to each CSV.")


if __name__ == "__main__":
    main()
