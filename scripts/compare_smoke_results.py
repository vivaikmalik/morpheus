"""
Compare two smoke-test eval CSVs (standard vs dep_aware) and report:
  - Per-stratum Morgan deltas
  - Per-molecule diff for the (3+ rings, 3+ branches) cell
  - Validation check: control cell (0 rings, 3+ branches) should be ~unchanged

Usage:
  python scripts/compare_smoke_results.py \
      <standard_eval_csv> <dep_aware_eval_csv>
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
META_CSV = REPO_ROOT / "data" / "chebi20_smoke_hard40_meta.csv"

def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    std_path = Path(sys.argv[1])
    dep_path = Path(sys.argv[2])

    if not std_path.exists():
        print(f"ERROR: {std_path} not found", file=sys.stderr)
        sys.exit(1)
    if not dep_path.exists():
        print(f"ERROR: {dep_path} not found", file=sys.stderr)
        sys.exit(1)
    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found. Run build_smoke_slice.py first.",
              file=sys.stderr)
        sys.exit(1)

    std = pd.read_csv(std_path)
    dep = pd.read_csv(dep_path)
    meta = pd.read_csv(META_CSV)

    print(f"Standard:  {len(std)} rows from {std_path.name}")
    print(f"Dep-aware: {len(dep)} rows from {dep_path.name}")
    print(f"Metadata:  {len(meta)} rows")
    print()

    std_m = std.merge(meta[["prompt", "stratum", "ring_count", "gt_branches"]],
                       on="prompt", how="inner")
    dep_m = dep.merge(meta[["prompt", "stratum", "ring_count", "gt_branches"]],
                       on="prompt", how="inner")

    print(f"After meta join: standard={len(std_m)}, dep_aware={len(dep_m)}")
    print()

    print("=" * 70)
    print("OVERALL (n = {})".format(len(std_m)))
    print("=" * 70)
    for metric in ["morgan_sim", "maccs_sim", "rdk_sim", "atom_bleu2"]:
        if metric not in std.columns:
            continue
        s = std_m[metric].mean()
        d = dep_m[metric].mean()
        delta = d - s
        arrow = "UP" if delta > 0 else "DOWN" if delta < 0 else "="
        print(f"  {metric:15s}: standard={s:.4f}  dep_aware={d:.4f}  "
              f"delta={delta:+.4f} {arrow}")
    print()

    print("=" * 70)
    print("PER STRATUM (Morgan)")
    print("=" * 70)
    for stratum in std_m["stratum"].unique():
        s_sub = std_m[std_m["stratum"] == stratum]
        d_sub = dep_m[dep_m["stratum"] == stratum]
        n = len(s_sub)
        s = s_sub["morgan_sim"].mean()
        d = d_sub["morgan_sim"].mean()
        delta = d - s
        arrow = "UP" if delta > 0 else "DOWN" if delta < 0 else "="
        flag = ""
        if "CONTROL" in stratum and abs(delta) > 0.05:
            flag = "  [WARN] control should be ~flat"
        print(f"  {stratum:30s}  n={n:3d}  std={s:.3f}  dep={d:.3f}  "
              f"delta={delta:+.3f} {arrow}{flag}")
    print()

    print("=" * 70)
    print("PER-MOLECULE DETAIL: (3+ rings, 3+ branches)")
    print("=" * 70)
    cell = "3+r/3+b (hardest)"
    s_cell = std_m[std_m["stratum"] == cell].set_index("prompt")
    d_cell = dep_m[dep_m["stratum"] == cell].set_index("prompt")
    common = sorted(set(s_cell.index) & set(d_cell.index))

    print(f"{'morgan_std':>11s} {'morgan_dep':>11s} {'delta':>9s}  prompt[:60]")
    n_better, n_worse, n_same = 0, 0, 0
    for prompt in common:
        s = s_cell.loc[prompt, "morgan_sim"]
        d = d_cell.loc[prompt, "morgan_sim"]
        delta = d - s
        if delta > 0.01: n_better += 1
        elif delta < -0.01: n_worse += 1
        else: n_same += 1
        print(f"  {s:.4f}      {d:.4f}    {delta:+.4f}   {prompt[:60]}")
    print()
    print(f"  Better (delta > +0.01): {n_better}")
    print(f"  Worse  (delta < -0.01): {n_worse}")
    print(f"  Same   (|delta| <= 0.01): {n_same}")
    print()

    print("=" * 70)
    print("VERDICT HEURISTIC")
    print("=" * 70)
    overall_delta = dep_m["morgan_sim"].mean() - std_m["morgan_sim"].mean()
    hard_cell_std = std_m[std_m["stratum"] == cell]["morgan_sim"].mean()
    hard_cell_dep = dep_m[dep_m["stratum"] == cell]["morgan_sim"].mean()
    hard_delta = hard_cell_dep - hard_cell_std

    control_strata = [s for s in std_m["stratum"].unique() if "CONTROL" in s]
    if control_strata:
        ctrl_std = std_m[std_m["stratum"].isin(control_strata)]["morgan_sim"].mean()
        ctrl_dep = dep_m[dep_m["stratum"].isin(control_strata)]["morgan_sim"].mean()
        ctrl_delta = ctrl_dep - ctrl_std
    else:
        ctrl_delta = 0.0

    print(f"  Overall Morgan delta:         {overall_delta:+.4f}")
    print(f"  Hard cell (3+r/3+b) delta:    {hard_delta:+.4f}")
    print(f"  Control cell delta:           {ctrl_delta:+.4f} (should be ~0)")
    print()

    if abs(ctrl_delta) > 0.05:
        print("  [WARN] CONTROL CELL MOVED: investigate before trusting result.")
    elif hard_delta > 0.03:
        print("  [PASS] POSITIVE SIGNAL: hard cell improved meaningfully.")
        print("    Recommend: run full eval with --commit_strategy dep_aware.")
    elif hard_delta > 0.01:
        print("  [WEAK] WEAK SIGNAL: small improvement on hard cell.")
        print("    Recommend: discuss whether to add case (b) or run full eval.")
    elif abs(hard_delta) <= 0.01:
        print("  [FAIL] NO SIGNAL: hard cell unchanged.")
        print("    Recommend: hypothesis may be wrong, or case (a) insufficient.")
        print("    Discuss next step before more work.")
    else:
        print("  [BAD]  NEGATIVE SIGNAL: hard cell got worse.")
        print("    Recommend: investigate, likely a bug in dep_aware logic.")

if __name__ == "__main__":
    main()
