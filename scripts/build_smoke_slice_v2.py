"""
Generalized smoke-slice builder for dep_aware replication experiments.
Same strata/logic as build_smoke_slice.py, but seed and output paths are
controlled via CLI so we can build reproducibly different draws.

Usage:
    python scripts/build_smoke_slice_v2.py --seed 43
Outputs:
    data/chebi20_smoke_hard40_seed{S}.csv
    data/chebi20_smoke_hard40_seed{S}_meta.csv
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATIFIED_CSV = REPO_ROOT / "outputs" / "stratified_analysis_best_model.csv"
TEST_CSV       = REPO_ROOT / "data" / "chebi20_test.csv"

MORGAN_LOW  = 0.10
MORGAN_HIGH = 0.40

STRATA = [
    {"name": "3+r/3+b (hardest)", "rings": "3+",  "branches": "3+",  "n": 20},
    {"name": "1-2r/3+b",          "rings": "1-2", "branches": "3+",  "n": 10},
    {"name": "3+r/1-2b",          "rings": "3+",  "branches": "1-2", "n":  5},
    {"name": "0r/3+b (CONTROL)",  "rings": "0",   "branches": "3+",  "n":  5},
]

def ring_to_bin(rc):
    if rc == 0:    return "0"
    if rc <= 2:    return "1-2"
    return "3+"

def branch_to_bin(bc):
    if bc == 0:    return "0"
    if bc <= 2:    return "1-2"
    return "3+"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="RNG seed for molecule selection")
    args = parser.parse_args()
    S = args.seed

    OUTPUT_CSV = REPO_ROOT / "data" / f"chebi20_smoke_hard40_seed{S}.csv"
    META_CSV   = REPO_ROOT / "data" / f"chebi20_smoke_hard40_seed{S}_meta.csv"

    if not STRATIFIED_CSV.exists():
        print(f"ERROR: {STRATIFIED_CSV} not found.", file=sys.stderr); sys.exit(1)
    if not TEST_CSV.exists():
        print(f"ERROR: {TEST_CSV} not found.", file=sys.stderr); sys.exit(1)

    df      = pd.read_csv(STRATIFIED_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"[seed={S}] Loaded {len(df)} rows from stratified analysis")
    print(f"[seed={S}] Loaded {len(test_df)} rows from canonical test CSV")

    df["gt_branches"] = df["gt_smiles"].fillna("").str.count(r"\(")
    df["ring_bin"]    = df["ring_count"].apply(ring_to_bin)
    df["branch_bin"]  = df["gt_branches"].apply(branch_to_bin)

    in_range   = (df["morgan_sim"] >= MORGAN_LOW) & (df["morgan_sim"] <= MORGAN_HIGH)
    candidates = df[in_range].copy()
    print(f"[seed={S}] In Morgan range [{MORGAN_LOW}, {MORGAN_HIGH}]: {len(candidates)}")

    rng = np.random.default_rng(S)
    picked_rows = []

    for stratum in STRATA:
        sub = candidates[
            (candidates["ring_bin"]   == stratum["rings"]) &
            (candidates["branch_bin"] == stratum["branches"])
        ]
        n_avail = len(sub)
        n_pick  = min(stratum["n"], n_avail)
        if n_pick < stratum["n"]:
            print(f"  WARNING: stratum {stratum['name']} only has {n_avail} "
                  f"candidates, wanted {stratum['n']}")
        if n_pick == 0:
            print(f"  SKIP {stratum['name']}: 0 candidates"); continue
        idx    = rng.choice(sub.index, size=n_pick, replace=False)
        picked = sub.loc[idx].copy()
        picked["stratum"] = stratum["name"]
        picked_rows.append(picked)
        print(f"  {stratum['name']:30s}: picked {n_pick}/{stratum['n']}  "
              f"mean_morgan={picked['morgan_sim'].mean():.3f}")

    if not picked_rows:
        print("ERROR: no molecules picked.", file=sys.stderr); sys.exit(1)

    picked_df = pd.concat(picked_rows, ignore_index=True)
    print(f"\n[seed={S}] Total picked: {len(picked_df)}")
    print("  Stratum counts:")
    for name, cnt in picked_df["stratum"].value_counts().items():
        print(f"    {name}: {cnt}")

    test_lookup = test_df.set_index("prompt")["response"].to_dict()
    out_rows = []
    missing  = 0
    for _, row in picked_df.iterrows():
        prompt = row["prompt"]
        if prompt not in test_lookup:
            missing += 1; continue
        out_rows.append({"prompt": prompt, "response": test_lookup[prompt]})

    if missing:
        print(f"  WARNING: {missing} prompts not found in canonical test CSV")

    if not out_rows:
        print("ERROR: no rows matched.", file=sys.stderr); sys.exit(1)

    out_df = pd.DataFrame(out_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[seed={S}] Wrote {len(out_df)} rows to {OUTPUT_CSV}")

    picked_df[["prompt", "gt_smiles", "ring_count", "gt_branches",
               "ring_bin", "branch_bin", "stratum", "morgan_sim"]].to_csv(META_CSV, index=False)
    print(f"[seed={S}] Wrote metadata to {META_CSV}")

if __name__ == "__main__":
    main()
