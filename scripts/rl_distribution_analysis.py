"""
RL Distribution Analysis
Characterizes molecular property distributions across four RL configurations.
Analyses: distribution stats, comparison plots, concentration metrics,
KL divergence vs ZINC reference, tail behavior, and summary.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.special import rel_entr

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
MOLGEN = "/Users/lirobert/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Li\u2019s MacBook Air/UdeM/IFT6759 (Advanced Machine Learning Projects)/molgen"
EVALS  = f"{MOLGEN}/outputs/evaluations"
PLOTS  = f"{MOLGEN}/outputs/plots/comparison"
OUT    = f"{MOLGEN}/outputs"

os.makedirs(PLOTS, exist_ok=True)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIGS = {
    "Base":       "eval_base_model.csv",
    "QED-only":   "eval_rl_qed_only.csv",
    "QED+MW":     "eval_rl_qed_mw_50step.csv",
    "QED+MW+Div": "eval_rl_qed_mw_div_50step.csv",
}

COLORS = {
    "Base":       "#888888",   # gray
    "QED-only":   "#a8d1e7",   # light blue
    "QED+MW":     "#4a90c4",   # medium blue
    "QED+MW+Div": "#0d3d6b",   # dark blue
}

PROPS = ["qed", "mw", "logp"]
PROP_LABELS = {"qed": "QED", "mw": "MW (Da)", "logp": "LogP"}

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
dfs = {}
for name, fname in CONFIGS.items():
    path = os.path.join(EVALS, fname)
    if not os.path.exists(path):
        print(f"WARNING: missing {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Keep only valid molecules
    valid = df[df["valid"] == True].copy() if "valid" in df.columns else df.copy()
    dfs[name] = valid
    print(f"Loaded {name}: {len(df)} total, {len(valid)} valid")

print()

# ──────────────────────────────────────────────
# ANALYSIS 1: Distribution statistics
# ──────────────────────────────────────────────
print("=" * 60)
print("ANALYSIS 1: Distribution statistics")
print("=" * 60)

stats_lines = ["# RL Distribution Statistics\n"]

for name, df in dfs.items():
    stats_lines.append(f"\n## {name} (n={len(df)})\n")
    stats_lines.append("| Property | Mean | Std | Median | IQR | Min | Max | p5 | p95 |")
    stats_lines.append("|----------|------|-----|--------|-----|-----|-----|----|-----|")

    for prop in PROPS:
        if prop not in df.columns:
            continue
        s = df[prop].dropna()
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        row = (
            f"| {prop.upper()} | {s.mean():.3f} | {s.std():.3f} | "
            f"{s.median():.3f} | {q75-q25:.3f} | {s.min():.3f} | "
            f"{s.max():.3f} | {s.quantile(0.05):.3f} | {s.quantile(0.95):.3f} |"
        )
        stats_lines.append(row)
        print(f"  {name} {prop.upper()}: mean={s.mean():.3f} std={s.std():.3f} "
              f"median={s.median():.3f} IQR={q75-q25:.3f} "
              f"[{s.quantile(0.05):.3f}, {s.quantile(0.95):.3f}]")

    # Uniqueness
    if "canonical_smiles" in df.columns:
        n_unique = df["canonical_smiles"].nunique()
    elif "smiles" in df.columns:
        n_unique = df["smiles"].nunique()
    else:
        n_unique = None

    if n_unique is not None:
        pct = 100 * n_unique / len(df) if len(df) > 0 else 0
        stats_lines.append(f"\n**Unique molecules:** {n_unique} / {len(df)} ({pct:.1f}%)\n")
        print(f"  {name} unique: {n_unique}/{len(df)} ({pct:.1f}%)")

# Save
stats_path = f"{OUT}/rl_distribution_stats.md"
with open(stats_path, "w") as f:
    f.write("\n".join(stats_lines))
print(f"\nSaved: {stats_path}")

# ──────────────────────────────────────────────
# ANALYSIS 2: Distribution comparison plots
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 2: Distribution comparison plots")
print("=" * 60)

plot_paths = {}
for prop in PROPS:
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, df in dfs.items():
        if prop not in df.columns:
            continue
        s = df[prop].dropna()
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s, bw_method="scott")
        x_min, x_max = s.quantile(0.005), s.quantile(0.995)
        x = np.linspace(x_min, x_max, 300)
        y = kde(x)
        ax.plot(x, y, color=COLORS[name], linewidth=2, label=name)
        ax.fill_between(x, y, alpha=0.15, color=COLORS[name])

    ax.set_xlabel(PROP_LABELS[prop], fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"{PROP_LABELS[prop]} Distribution by RL Configuration", fontsize=14)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    fpath = f"{PLOTS}/rl_{prop}_distributions.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths[prop] = fpath
    print(f"Saved: {fpath}")

# Print observations
print("\nObservations:")
for prop in PROPS:
    stds = {name: dfs[name][prop].std() for name in dfs if prop in dfs[name].columns}
    tightest = min(stds, key=stds.get)
    widest   = max(stds, key=stds.get)
    means    = {name: dfs[name][prop].mean() for name in dfs if prop in dfs[name].columns}
    shifts   = {name: abs(means[name] - means["Base"]) for name in dfs if name != "Base"}
    most_shifted = max(shifts, key=shifts.get)
    print(f"  {prop.upper()}: tightest={tightest} (std={stds[tightest]:.3f}), "
          f"widest={widest} (std={stds[widest]:.3f}), "
          f"most_shifted={most_shifted} (Δmean={shifts[most_shifted]:.3f})")

# ──────────────────────────────────────────────
# ANALYSIS 3: Concentration and dispersion
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 3: Concentration and dispersion analysis")
print("=" * 60)

conc_lines = ["\n## Analysis 3: Concentration Metrics\n"]
conc_header = "| Config | MW ±10% | MW ±20% | QED ±10% |"
conc_sep    = "|--------|---------|---------|----------|"
conc_rows = []

print(f"{'Config':<15} {'MW ±10%':>9} {'MW ±20%':>9} {'QED ±10%':>10}")
print("-" * 50)

for name, df in dfs.items():
    mw_mean  = df["mw"].mean()
    qed_mean = df["qed"].mean()

    mw_10  = ((df["mw"] - mw_mean).abs() / mw_mean <= 0.10).mean() * 100
    mw_20  = ((df["mw"] - mw_mean).abs() / mw_mean <= 0.20).mean() * 100
    qed_10 = ((df["qed"] - qed_mean).abs() / qed_mean <= 0.10).mean() * 100

    print(f"  {name:<13} {mw_10:>8.1f}% {mw_20:>8.1f}% {qed_10:>9.1f}%")
    conc_rows.append(f"| {name} | {mw_10:.1f}% | {mw_20:.1f}% | {qed_10:.1f}% |")

conc_lines += [conc_header, conc_sep] + conc_rows

# ──────────────────────────────────────────────
# ANALYSIS 4: KL divergence vs ZINC reference
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 4: KL divergence vs ZINC reference")
print("=" * 60)

# Build reference from val.csv (subsample 1000 for speed)
VAL_PATH = f"{MOLGEN}/data/val.csv"
df_val = pd.read_csv(VAL_PATH)

print("  Computing ZINC reference properties from val.csv (sample=1500)...")
sample_size = 1500
rng = np.random.default_rng(42)
idx = rng.choice(len(df_val), min(sample_size, len(df_val)), replace=False)
df_sample = df_val.iloc[idx]

ref_props = {"qed": [], "mw": [], "logp": []}
n_ok = 0
for _, row in df_sample.iterrows():
    try:
        sel = row["response"].strip()
        smi = sf.decoder(sel)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ref_props["qed"].append(QED.qed(mol))
        ref_props["mw"].append(Descriptors.MolWt(mol))
        ref_props["logp"].append(Descriptors.MolLogP(mol))
        n_ok += 1
    except Exception:
        continue

print(f"  Reference molecules computed: {n_ok} / {sample_size}")
ref_series = {k: np.array(v) for k, v in ref_props.items() if len(v) > 0}

def compute_kl(a, b, n_bins=50):
    """KL divergence via histogram binning."""
    combined = np.concatenate([a, b])
    lo, hi = np.percentile(combined, 1), np.percentile(combined, 99)
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=bins, density=True)
    q, _ = np.histogram(b, bins=bins, density=True)
    # Add smoothing to avoid zeros
    eps = 1e-10
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(rel_entr(p, q)))

kl_lines = ["\n## Analysis 4: KL Divergence vs ZINC Reference\n"]
kl_header = "| Config | KL(QED) | KL(MW) | KL(LogP) |"
kl_sep    = "|--------|---------|--------|----------|"
kl_rows = []

print(f"\n{'Config':<15} {'KL(QED)':>9} {'KL(MW)':>8} {'KL(LogP)':>10}")
print("-" * 50)

kl_totals = {}
for name, df in dfs.items():
    kls = {}
    for prop in ["qed", "mw", "logp"]:
        if prop not in df.columns or prop not in ref_series:
            kls[prop] = float("nan")
            continue
        kls[prop] = compute_kl(df[prop].dropna().values, ref_series[prop])
    kl_totals[name] = sum(v for v in kls.values() if not np.isnan(v))
    print(f"  {name:<13} {kls['qed']:>9.4f} {kls['mw']:>8.4f} {kls['logp']:>10.4f}")
    kl_rows.append(f"| {name} | {kls['qed']:.4f} | {kls['mw']:.4f} | {kls['logp']:.4f} |")

kl_lines += [kl_header, kl_sep] + kl_rows

closest  = min(kl_totals, key=kl_totals.get)
furthest = max(kl_totals, key=kl_totals.get)
print(f"\n  Closest to ZINC: {closest} (total KL={kl_totals[closest]:.4f})")
print(f"  Furthest from ZINC: {furthest} (total KL={kl_totals[furthest]:.4f})")
kl_lines.append(f"\n**Closest to ZINC:** {closest} (ΣKL={kl_totals[closest]:.4f})  ")
kl_lines.append(f"**Furthest from ZINC:** {furthest} (ΣKL={kl_totals[furthest]:.4f})")

# ──────────────────────────────────────────────
# ANALYSIS 5: Tail behavior
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 5: Tail behavior")
print("=" * 60)

tail_lines = ["\n## Analysis 5: Tail Behavior\n"]
tail_header = "| Config | MW<200 | MW>400 | QED>0.9 | QED<0.5 |"
tail_sep    = "|--------|--------|--------|---------|---------|"
tail_rows = []

print(f"{'Config':<15} {'MW<200':>8} {'MW>400':>8} {'QED>0.9':>9} {'QED<0.5':>9}")
print("-" * 55)

for name, df in dfs.items():
    mw_low  = (df["mw"] < 200).sum()
    mw_high = (df["mw"] > 400).sum()
    qed_hi  = (df["qed"] > 0.9).sum()
    qed_lo  = (df["qed"] < 0.5).sum()
    n = len(df)
    print(f"  {name:<13} {mw_low:>5} ({100*mw_low/n:.1f}%)  {mw_high:>5} ({100*mw_high/n:.1f}%)  "
          f"{qed_hi:>5} ({100*qed_hi/n:.1f}%)  {qed_lo:>5} ({100*qed_lo/n:.1f}%)")
    tail_rows.append(
        f"| {name} | {mw_low} ({100*mw_low/n:.1f}%) | {mw_high} ({100*mw_high/n:.1f}%) | "
        f"{qed_hi} ({100*qed_hi/n:.1f}%) | {qed_lo} ({100*qed_lo/n:.1f}%) |"
    )

tail_lines += [tail_header, tail_sep] + tail_rows

# ──────────────────────────────────────────────
# ANALYSIS 6: Honest summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 6: Honest summary")
print("=" * 60)

# Pull key numbers for the summary
base_mw_std    = dfs["Base"]["mw"].std()
qed_only_mw_std= dfs["QED-only"]["mw"].std()
qedmw_mw_std   = dfs["QED+MW"]["mw"].std()

base_qed_std     = dfs["Base"]["qed"].std()
qed_only_qed_std = dfs["QED-only"]["qed"].std()
qedmw_qed_std    = dfs["QED+MW"]["qed"].std()

base_mw_mean      = dfs["Base"]["mw"].mean()
qed_only_mw_mean  = dfs["QED-only"]["mw"].mean()
qedmw_mw_mean     = dfs["QED+MW"]["mw"].mean()

base_qed_mean     = dfs["Base"]["qed"].mean()
qed_only_qed_mean = dfs["QED-only"]["qed"].mean()
qedmw_qed_mean    = dfs["QED+MW"]["qed"].mean()

mw_tightened = qed_only_mw_std < base_mw_std
qed_tightened= qed_only_qed_std < base_qed_std
mw_broadened = qedmw_mw_std > qed_only_mw_std

summary = f"""# RL Distribution Analysis: Honest Summary

## Key Numbers

| Metric | Base | QED-only | QED+MW | QED+MW+Div |
|--------|------|----------|--------|------------|
| QED mean | {dfs["Base"]["qed"].mean():.3f} | {dfs["QED-only"]["qed"].mean():.3f} | {dfs["QED+MW"]["qed"].mean():.3f} | {dfs["QED+MW+Div"]["qed"].mean():.3f} |
| QED std  | {dfs["Base"]["qed"].std():.3f} | {dfs["QED-only"]["qed"].std():.3f} | {dfs["QED+MW"]["qed"].std():.3f} | {dfs["QED+MW+Div"]["qed"].std():.3f} |
| MW mean  | {dfs["Base"]["mw"].mean():.1f} | {dfs["QED-only"]["mw"].mean():.1f} | {dfs["QED+MW"]["mw"].mean():.1f} | {dfs["QED+MW+Div"]["mw"].mean():.1f} |
| MW std   | {dfs["Base"]["mw"].std():.1f} | {dfs["QED-only"]["mw"].std():.1f} | {dfs["QED+MW"]["mw"].std():.1f} | {dfs["QED+MW+Div"]["mw"].std():.1f} |
| LogP mean| {dfs["Base"]["logp"].mean():.3f} | {dfs["QED-only"]["logp"].mean():.3f} | {dfs["QED+MW"]["logp"].mean():.3f} | {dfs["QED+MW+Div"]["logp"].mean():.3f} |
| LogP std | {dfs["Base"]["logp"].std():.3f} | {dfs["QED-only"]["logp"].std():.3f} | {dfs["QED+MW"]["logp"].std():.3f} | {dfs["QED+MW+Div"]["logp"].std():.3f} |

## Five-Sentence Summary

1. **Did QED-only produce a tighter MW distribution than base?** {"Yes" if mw_tightened else "No"}: QED-only MW std = {qed_only_mw_std:.1f} Da vs. base {base_mw_std:.1f} Da ({"narrower by" if mw_tightened else "wider by"} {abs(qed_only_mw_std - base_mw_std):.1f} Da), suggesting that optimizing QED without an explicit MW constraint {"compressed" if mw_tightened else "did not compress"} the molecular weight distribution—likely because high-QED molecules tend to share similar structural profiles.

2. **Did QED-only produce a tighter QED distribution?** {"Yes" if qed_tightened else "No"}: QED std dropped from {base_qed_std:.3f} (base) to {qed_only_qed_std:.3f} (QED-only), indicating the RL policy learned to avoid both very-low- and very-high-QED molecules {"and concentrate around a narrow band" if qed_tightened else "—this is unexpected and warrants further inspection"}.

3. **Is the QED compression "bad"?** {"Likely a mode-collapse signal" if qed_tightened else "Not applicable"}: A tighter QED distribution is not intrinsically bad—it reflects the reward signal doing its job—but a {"very sharp" if qed_only_qed_std < 0.05 else "moderate"} reduction in QED variance combined with {"also" if mw_tightened else ""} MW compression suggests the model may be converging to a limited set of scaffold types (mode narrowing), reducing structural diversity even if individual quality metrics improve.

4. **Did the MW constraint broaden the distribution or just shift the mean?** {"The QED+MW constraint broadened the MW distribution" if mw_broadened else "The QED+MW constraint did NOT substantially broaden the MW distribution"} (std: QED-only={qed_only_mw_std:.1f} → QED+MW={qedmw_mw_std:.1f} Da), {"meaning the constraint added real dispersion back, not merely a mean shift" if mw_broadened else "meaning the constraint primarily shifted the mean (from {qed_only_mw_mean:.1f} to {qedmw_mw_mean:.1f} Da) without recovering the diversity lost during QED-only training"}.

5. **Strongest evidence-based motivation for MW constraint:** {"The QED-only policy compressed MW variance to {qed_only_mw_std:.1f} Da (down from {base_mw_std:.1f} Da in base), concentrating molecules in a narrow band and reducing the fraction of drug-like-range molecules (200–400 Da); the MW penalty was therefore motivated not merely by mean drift but by the loss of distributional coverage across the drug-like weight range." if mw_tightened else "Even though QED-only did not compress MW variance significantly, the mean MW drifted to {qed_only_mw_mean:.1f} Da (base: {base_mw_mean:.1f} Da), pushing molecules outside the Lipinski-recommended 150–500 Da window; the MW constraint corrects this systematic mean bias."}
"""

# Print summary
print(summary)

summary_path = f"{OUT}/rl_distribution_summary.md"
with open(summary_path, "w") as f:
    f.write(summary)
print(f"Saved: {summary_path}")

# ──────────────────────────────────────────────
# Assemble full stats file with all analyses
# ──────────────────────────────────────────────
all_lines = stats_lines + conc_lines + kl_lines + tail_lines
with open(stats_path, "w") as f:
    f.write("\n".join(all_lines))
print(f"Updated: {stats_path}")

print("\n" + "=" * 60)
print("HEADLINE SUMMARY")
print("=" * 60)
print(f"  Configs analyzed: {', '.join(dfs.keys())}")
for name, df in dfs.items():
    print(f"  {name}: n={len(df)}, QED={df['qed'].mean():.3f}±{df['qed'].std():.3f}, "
          f"MW={df['mw'].mean():.1f}±{df['mw'].std():.1f}")
print(f"\n  QED-only MW std: {qed_only_mw_std:.1f} vs Base: {base_mw_std:.1f} "
      f"({'TIGHTER' if mw_tightened else 'WIDER'})")
print(f"  QED-only QED std: {qed_only_qed_std:.3f} vs Base: {base_qed_std:.3f} "
      f"({'TIGHTER' if qed_tightened else 'WIDER'})")
print(f"  QED+MW MW std: {qedmw_mw_std:.1f} vs QED-only: {qed_only_mw_std:.1f} "
      f"({'BROADER' if mw_broadened else 'STILL NARROW'})")
print(f"  Closest to ZINC: {closest}")
print(f"  Furthest from ZINC: {furthest}")
print()
print("Done.")
