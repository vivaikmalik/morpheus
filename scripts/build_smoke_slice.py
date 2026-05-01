"""
Build the 40-molecule smoke test slice for dependency-aware decoding.

Picks molecules from the stratified analysis CSV based on (ring_count, branch_count)
strata, looks up their SELFIES from the canonical test CSV, and writes a small
test CSV in the prompt,response format that train_chebi20.py expects.

Strata picked:
  - 20 from (3+ rings, 3+ branches) - the worst cell, where 38% of Morgan loss lives.
                                       Hypothesis predicts the biggest improvement here.
  - 10 from (1-2 rings, 3+ branches) - second worst cell.
  -  5 from (3+ rings, 1-2 branches) - fewer branches, still polycyclic.
  -  5 from (0 rings, 3+ branches)   - CONTROL. Hypothesis predicts NO improvement
                                       here (no ring count tokens). If we see one,
                                       something is wrong.

Within each stratum, picks molecules where baseline Morgan is in [0.10, 0.40] -
room to improve, not catastrophic failures with no useful signal.

Reproducible: fixed random seed.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATIFIED_CSV = REPO_ROOT / "outputs" / "stratified_analysis_best_model.csv"
TEST_CSV = REPO_ROOT / "data" / "chebi20_test.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "chebi20_smoke_hard40.csv"

SEED = 42
MORGAN_LOW = 0.10
MORGAN_HIGH = 0.40

STRATA = [
    {"name": "3+r/3+b (hardest)",   "rings": "3+", "branches": "3+", "n": 20},
    {"name": "1-2r/3+b",             "rings": "1-2","branches": "3+", "n": 10},
    {"name": "3+r/1-2b",             "rings": "3+", "branches": "1-2","n": 5},
    {"name": "0r/3+b (CONTROL)",     "rings": "0",  "branches": "3+", "n": 5},
]

def ring_to_bin(rc):
    if rc == 0: return "0"
    if rc <= 2: return "1-2"
    return "3+"

def branch_to_bin(bc):
    if bc == 0: return "0"
    if bc <= 2: return "1-2"
    return "3+"

def main():
    if not STRATIFIED_CSV.exists():
        print(f"ERROR: {STRATIFIED_CSV} not found.", file=sys.stderr)
        print("This script needs the per-molecule diagnostic CSV from your audit work.",
              file=sys.stderr)
        sys.exit(1)
    if not TEST_CSV.exists():
        print(f"ERROR: {TEST_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(STRATIFIED_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Loaded {len(df)} rows from stratified analysis")
    print(f"Loaded {len(test_df)} rows from canonical test CSV")

    df["gt_branches"] = df["gt_smiles"].fillna("").str.count(r"\(")
    df["ring_bin"] = df["ring_count"].apply(ring_to_bin)
    df["branch_bin"] = df["gt_branches"].apply(branch_to_bin)

    in_range = (df["morgan_sim"] >= MORGAN_LOW) & (df["morgan_sim"] <= MORGAN_HIGH)
    candidates = df[in_range].copy()
    print(f"In Morgan range [{MORGAN_LOW}, {MORGAN_HIGH}]: {len(candidates)}")

    rng = np.random.default_rng(SEED)
    picked_rows = []

    for stratum in STRATA:
        sub = candidates[
            (candidates["ring_bin"] == stratum["rings"]) &
            (candidates["branch_bin"] == stratum["branches"])
        ]
        n_avail = len(sub)
        n_pick = min(stratum["n"], n_avail)
        if n_pick < stratum["n"]:
            print(f"WARNING: stratum {stratum['name']} only has {n_avail} candidates, "
                  f"wanted {stratum['n']}")
        if n_pick == 0:
            print(f"  SKIP {stratum['name']}: 0 candidates")
            continue
        idx = rng.choice(sub.index, size=n_pick, replace=False)
        picked = sub.loc[idx].copy()
        picked["stratum"] = stratum["name"]
        picked_rows.append(picked)
        print(f"  {stratum['name']}: picked {n_pick}/{stratum['n']} "
              f"(mean baseline Morgan = {picked['morgan_sim'].mean():.3f})")

    if not picked_rows:
        print("ERROR: no molecules picked. Check stratified CSV contents.", file=sys.stderr)
        sys.exit(1)

    picked_df = pd.concat(picked_rows, ignore_index=True)
    print(f"\nTotal picked: {len(picked_df)}")

    test_lookup = test_df.set_index("prompt")["response"].to_dict()

    out_rows = []
    missing = 0
    for _, row in picked_df.iterrows():
        prompt = row["prompt"]
        if prompt not in test_lookup:
            missing += 1
            continue
        out_rows.append({"prompt": prompt, "response": test_lookup[prompt]})

    if missing > 0:
        print(f"WARNING: {missing} prompts not found in canonical test CSV "
              f"(likely whitespace/encoding mismatch). Output has {len(out_rows)} rows.")

    if not out_rows:
        print("ERROR: no rows could be matched to canonical test CSV.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(out_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUTPUT_CSV}")
    print(f"Columns: {list(out_df.columns)}")

    META_CSV = OUTPUT_CSV.with_name("chebi20_smoke_hard40_meta.csv")
    picked_df[["prompt", "gt_smiles", "ring_count", "gt_branches",
               "ring_bin", "branch_bin", "stratum", "morgan_sim"]].to_csv(META_CSV, index=False)
    print(f"Wrote metadata to {META_CSV}")

if __name__ == "__main__":
    main()
