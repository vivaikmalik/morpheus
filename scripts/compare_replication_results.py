"""
Compare replication experiment results: 3 seeds x 2 strategies.
Reads outputs/replication_seed{43,44,45}_{standard,dep_aware}.csv
and the corresponding seed{S}_meta.csv files.

Produces outputs/replication_results.md with:
  - Per-seed table
  - Per-stratum table (averaged across seeds)
  - Comparison to original smoke test
  - Verdict heuristic
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "outputs"
DATA_DIR  = REPO_ROOT / "data"

SEEDS     = [43, 44, 45]
STRATS    = ["standard", "dep_aware"]
STRATA    = ["3+r/3+b (hardest)", "1-2r/3+b", "3+r/1-2b", "0r/3+b (CONTROL)"]

# Original smoke test deltas (seed=42) for reference
ORIG_DELTAS = {
    "3+r/3+b (hardest)": (+0.023, 20),
    "1-2r/3+b":           (+0.011, 10),
    "3+r/1-2b":           (+0.001,  5),
    "0r/3+b (CONTROL)":   (+0.000,  5),
}

METRICS = ["morgan_sim", "maccs_sim", "atom_bleu2"]


def load_eval(seed, strategy):
    p = OUT_DIR / f"replication_seed{seed}_{strategy}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_meta(seed):
    p = DATA_DIR / f"chebi20_smoke_hard40_seed{seed}_meta.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def per_seed_row(seed, std_df, dep_df):
    row = {"seed": seed, "n": len(std_df)}
    for m in METRICS:
        s = std_df[m].mean() if m in std_df.columns else float("nan")
        d = dep_df[m].mean() if m in dep_df.columns else float("nan")
        row[f"{m}_std"] = s
        row[f"{m}_dep"] = d
        row[f"{m}_delta"] = d - s
    return row


def per_stratum_rows(seed, std_df, dep_df, meta_df):
    std_m = std_df.merge(meta_df[["prompt", "stratum"]], on="prompt", how="inner")
    dep_m = dep_df.merge(meta_df[["prompt", "stratum"]], on="prompt", how="inner")
    rows = []
    for stratum in STRATA:
        s_sub = std_m[std_m["stratum"] == stratum]
        d_sub = dep_m[dep_m["stratum"] == stratum]
        if len(s_sub) == 0:
            continue
        s = s_sub["morgan_sim"].mean()
        d = d_sub["morgan_sim"].mean()
        rows.append({
            "seed": seed,
            "stratum": stratum,
            "n": len(s_sub),
            "morgan_std": s,
            "morgan_dep": d,
            "morgan_delta": d - s,
        })
    return rows


def fmt(v, decimals=4):
    if np.isnan(v): return "n/a"
    return f"{v:.{decimals}f}"


def fmt_delta(v, decimals=4):
    if np.isnan(v): return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


def main():
    # ── Load all data ─────────────────────────────────────────────────────────
    seed_rows = []
    stratum_rows = []
    missing = []

    for seed in SEEDS:
        std_df = load_eval(seed, "standard")
        dep_df = load_eval(seed, "dep_aware")
        meta_df = load_meta(seed)

        if std_df is None or dep_df is None or meta_df is None:
            missing.append(seed)
            continue

        seed_rows.append(per_seed_row(seed, std_df, dep_df))
        stratum_rows.extend(per_stratum_rows(seed, std_df, dep_df, meta_df))

    if missing:
        print(f"WARNING: seeds {missing} have missing files. They will be absent from report.")

    if not seed_rows:
        print("ERROR: no data found. Run run_replication_experiment.sh first.", file=sys.stderr)
        sys.exit(1)

    seed_df    = pd.DataFrame(seed_rows)
    stratum_df = pd.DataFrame(stratum_rows)

    # ── Build markdown ────────────────────────────────────────────────────────
    lines = []
    lines.append("# Replication Experiment Results")
    lines.append("")
    lines.append("**Setup:** 3 new molecule draws (seeds 43/44/45) × 2 strategies × 40 molecules each.")
    lines.append("**Goal:** Verify whether the original smoke test signal (seed=42, +0.023 Morgan on 3+r/3+b) was real.")
    lines.append("")

    # Per-seed table
    lines.append("## Per-seed Summary")
    lines.append("")
    header = ("| Seed | n | Morgan std | Morgan dep | Morgan Δ | "
              "MACCS std | MACCS dep | MACCS Δ | atom_bleu2 std | atom_bleu2 dep | atom_bleu2 Δ |")
    sep    = ("|------|---|-----------|-----------|---------|"
              "----------|----------|--------|---------------|---------------|-------------|")
    lines.append(header)
    lines.append(sep)
    for r in seed_rows:
        lines.append(
            f"| {r['seed']} | {r['n']} "
            f"| {fmt(r['morgan_sim_std'])} | {fmt(r['morgan_sim_dep'])} | {fmt_delta(r['morgan_sim_delta'])} "
            f"| {fmt(r['maccs_sim_std'])} | {fmt(r['maccs_sim_dep'])} | {fmt_delta(r['maccs_sim_delta'])} "
            f"| {fmt(r['atom_bleu2_std'])} | {fmt(r['atom_bleu2_dep'])} | {fmt_delta(r['atom_bleu2_delta'])} |"
        )
    # Mean row
    if len(seed_rows) > 1:
        lines.append(
            f"| **mean** | - "
            f"| {fmt(seed_df['morgan_sim_std'].mean())} | {fmt(seed_df['morgan_sim_dep'].mean())} | {fmt_delta(seed_df['morgan_sim_delta'].mean())} "
            f"| {fmt(seed_df['maccs_sim_std'].mean())} | {fmt(seed_df['maccs_sim_dep'].mean())} | {fmt_delta(seed_df['maccs_sim_delta'].mean())} "
            f"| {fmt(seed_df['atom_bleu2_std'].mean())} | {fmt(seed_df['atom_bleu2_dep'].mean())} | {fmt_delta(seed_df['atom_bleu2_delta'].mean())} |"
        )
    lines.append("")

    # Per-stratum table
    lines.append("## Per-stratum Results (averaged across seeds)")
    lines.append("")
    lines.append("*Std dev of Morgan Δ across seeds = noise estimate for each stratum.*")
    lines.append("")
    lines.append("| Stratum | n (total) | Morgan std (mean) | Morgan dep (mean) | Morgan Δ (mean) | Δ std-dev |")
    lines.append("|---------|-----------|------------------|------------------|-----------------|-----------|")

    for stratum in STRATA:
        sub = stratum_df[stratum_df["stratum"] == stratum]
        if len(sub) == 0:
            lines.append(f"| {stratum} | 0 | n/a | n/a | n/a | n/a |")
            continue
        n_total   = sub["n"].sum()
        m_std     = sub["morgan_std"].mean()
        m_dep     = sub["morgan_dep"].mean()
        m_delta   = sub["morgan_delta"].mean()
        m_deltastd = sub["morgan_delta"].std() if len(sub) > 1 else float("nan")
        lines.append(
            f"| {stratum} | {n_total} "
            f"| {fmt(m_std)} | {fmt(m_dep)} | {fmt_delta(m_delta)} | {fmt(m_deltastd)} |"
        )
    lines.append("")

    # Comparison to original
    lines.append("## Comparison to Original Smoke Test (seed=42)")
    lines.append("")
    lines.append("| Stratum | Original Δ (n) | Replication Δ (mean) | Replication Δ std |")
    lines.append("|---------|---------------|---------------------|------------------|")
    for stratum in STRATA:
        orig_delta, orig_n = ORIG_DELTAS.get(stratum, (float("nan"), 0))
        sub = stratum_df[stratum_df["stratum"] == stratum]
        if len(sub) == 0:
            rep_delta = float("nan")
            rep_std   = float("nan")
        else:
            rep_delta = sub["morgan_delta"].mean()
            rep_std   = sub["morgan_delta"].std() if len(sub) > 1 else float("nan")
        lines.append(
            f"| {stratum} | {fmt_delta(orig_delta, 3)} (n={orig_n}) "
            f"| {fmt_delta(rep_delta)} | {fmt(rep_std)} |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")

    hard_sub = stratum_df[stratum_df["stratum"] == "3+r/3+b (hardest)"]
    if len(hard_sub) == 0:
        verdict = "UNKNOWN: no data for hard cell."
        verdict_tag = "UNKNOWN"
    else:
        mean_delta = hard_sub["morgan_delta"].mean()
        std_delta  = hard_sub["morgan_delta"].std() if len(hard_sub) > 1 else float("nan")
        all_positive = (hard_sub["morgan_delta"] > 0).all()

        if all_positive and mean_delta > 0.01 and (np.isnan(std_delta) or std_delta < abs(mean_delta)):
            verdict_tag = "REPLICATION CONFIRMED"
            verdict = (
                f"**REPLICATION CONFIRMED:** All {len(hard_sub)} replication seeds show positive Morgan "
                f"delta on the hard 3+r/3+b cell (mean Δ = {fmt_delta(mean_delta)}, "
                f"std = {fmt(std_delta)}). "
                "The original smoke test signal is reproducible on similar molecule populations."
            )
        elif mean_delta < 0.005 or (not np.isnan(std_delta) and std_delta > abs(mean_delta) + 0.001):
            verdict_tag = "REPLICATION FAILED"
            verdict = (
                f"**REPLICATION FAILED:** The {len(hard_sub)} replication seeds show inconsistent or "
                f"near-zero deltas on the hard 3+r/3+b cell (mean Δ = {fmt_delta(mean_delta)}, "
                f"std = {fmt(std_delta)}). "
                "The original smoke test result was within sample variance and not a real signal."
            )
        else:
            verdict_tag = "MIXED"
            verdict = (
                f"**MIXED:** Some seeds show positive deltas, others show flat or negative "
                f"(mean Δ = {fmt_delta(mean_delta)}, std = {fmt(std_delta)}). "
                "The signal is unreliable on samples this small."
            )

    lines.append(verdict)
    lines.append("")

    # Write output
    md_path = OUT_DIR / "replication_results.md"
    md_path.write_text("\n".join(lines))
    print(f"[output] Markdown written: {md_path}")
    print(f"[verdict] {verdict_tag}")
    print(f"  Hard cell (3+r/3+b) deltas across seeds:")
    if len(hard_sub) > 0:
        for _, r in hard_sub.iterrows():
            print(f"    seed={int(r['seed'])}  delta={fmt_delta(r['morgan_delta'])}")
    else:
        print("    (no data)")


if __name__ == "__main__":
    main()
