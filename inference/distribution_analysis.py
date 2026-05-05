"""
distribution_analysis.py
Compares property distributions between ZINC250k (training) and generated molecules.
Produces overlaid histograms and KL divergence for QED, LogP, and MW.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import gaussian_kde, entropy
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

GENERATED_PATH = Path(__file__).parent.parent / "outputs" / "generated_molecules.csv"
OUTPUT_DIR     = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kl_divergence(p_samples, q_samples, n_bins=100):
    """
    Estimate KL(P || Q) by fitting a KDE to both sample sets,
    evaluating on a shared grid, and computing the discrete KL.
    Using KDE avoids zero-probability bins that break entropy().
    """
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    grid = np.linspace(lo, hi, 1000)

    kde_p = gaussian_kde(p_samples, bw_method="scott")
    kde_q = gaussian_kde(q_samples, bw_method="scott")

    p_pdf = kde_p(grid) + 1e-10   # small epsilon to avoid log(0)
    q_pdf = kde_q(grid) + 1e-10

    # Normalise to valid probability distributions
    p_pdf /= p_pdf.sum()
    q_pdf /= q_pdf.sum()

    return float(entropy(p_pdf, q_pdf))   # scipy entropy = KL(p || q)


def compute_properties_rdkit(smiles_list):
    """Compute QED, LogP, MW for a list of SMILES strings via RDKit."""
    qeds, logps, mws = [], [], []
    failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        qeds.append(QED.qed(mol))
        logps.append(Descriptors.MolLogP(mol))
        mws.append(Descriptors.MolWt(mol))
    return np.array(qeds), np.array(logps), np.array(mws), failed


# ---------------------------------------------------------------------------
# Step 1 — Load & compute ZINC250k properties
# ---------------------------------------------------------------------------
print("Loading ZINC250k from HuggingFace...")
zinc = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
print(f"  {len(zinc):,} molecules loaded")

print("Computing QED / LogP / MW for ZINC250k (this takes ~2 min)...")
train_qed, train_logp, train_mw, train_failed = compute_properties_rdkit(
    zinc["smiles"].tolist()
)
print(f"  Done. {len(train_qed):,} computed, {train_failed} failed.")

# ---------------------------------------------------------------------------
# Step 2 — Load generated molecules
# ---------------------------------------------------------------------------
print(f"\nLoading {GENERATED_PATH.name}...")
gen   = pd.read_csv(GENERATED_PATH)
valid = gen[gen["valid"] == True].copy()
print(f"  {len(valid)} valid molecules")

gen_qed  = valid["qed"].dropna().values.astype(float)
gen_logp = valid["logp"].dropna().values.astype(float)
gen_mw   = valid["mw"].dropna().values.astype(float)

# ---------------------------------------------------------------------------
# Step 3 — KL divergences
# ---------------------------------------------------------------------------
print("\nComputing KL divergences...")
kl_qed  = kl_divergence(train_qed,  gen_qed)
kl_logp = kl_divergence(train_logp, gen_logp)
kl_mw   = kl_divergence(train_mw,   gen_mw)

print(f"  KL(train || gen)  QED : {kl_qed:.4f}")
print(f"  KL(train || gen) LogP : {kl_logp:.4f}")
print(f"  KL(train || gen)   MW : {kl_mw:.4f}")

# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------
PROPS = [
    {
        "name"      : "QED",
        "train"     : train_qed,
        "gen"       : gen_qed,
        "kl"        : kl_qed,
        "xlabel"    : "QED (Drug-likeness)",
        "bins"      : np.linspace(0, 1, 41),
        "filename"  : "dist_qed.png",
    },
    {
        "name"      : "LogP",
        "train"     : train_logp,
        "gen"       : gen_logp,
        "kl"        : kl_logp,
        "xlabel"    : "LogP",
        "bins"      : np.linspace(-5, 12, 51),
        "filename"  : "dist_logp.png",
    },
    {
        "name"      : "MW",
        "train"     : train_mw,
        "gen"       : gen_mw,
        "kl"        : kl_mw,
        "xlabel"    : "Molecular Weight (Da)",
        "bins"      : np.linspace(0, 700, 51),
        "filename"  : "dist_mw.png",
    },
]

TRAIN_COLOR = "#4C72B0"
GEN_COLOR   = "#DD8452"
ALPHA       = 0.55

for prop in PROPS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"{prop['name']} Distribution: ZINC250k vs Generated",
                 fontsize=14, fontweight="bold", y=1.01)

    # --- Left: overlaid histograms (density) ---
    ax = axes[0]
    ax.hist(prop["train"], bins=prop["bins"], density=True,
            color=TRAIN_COLOR, alpha=ALPHA,
            label=f"ZINC250k  (n={len(prop['train']):,})")
    ax.hist(prop["gen"], bins=prop["bins"], density=True,
            color=GEN_COLOR, alpha=ALPHA,
            label=f"Generated  (n={len(prop['gen'])})")

    # KDE curves on top
    grid = np.linspace(prop["bins"][0], prop["bins"][-1], 500)
    ax.plot(grid, gaussian_kde(prop["train"], bw_method="scott")(grid),
            color=TRAIN_COLOR, linewidth=2)
    ax.plot(grid, gaussian_kde(prop["gen"], bw_method="scott")(grid),
            color=GEN_COLOR, linewidth=2)

    ax.set_xlabel(prop["xlabel"], fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(prop["bins"][0], prop["bins"][-1])
    ax.grid(True, alpha=0.3)

    # KL annotation
    ax.text(0.97, 0.95,
            f"KL(train ‖ gen) = {prop['kl']:.4f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    # --- Right: summary statistics table ---
    ax2 = axes[1]
    ax2.axis("off")

    stats = {
        "Statistic": ["Mean", "Median", "Std", "Min", "Max"],
        "ZINC250k": [
            f"{np.mean(prop['train']):.3f}",
            f"{np.median(prop['train']):.3f}",
            f"{np.std(prop['train']):.3f}",
            f"{np.min(prop['train']):.3f}",
            f"{np.max(prop['train']):.3f}",
        ],
        "Generated": [
            f"{np.mean(prop['gen']):.3f}",
            f"{np.median(prop['gen']):.3f}",
            f"{np.std(prop['gen']):.3f}",
            f"{np.min(prop['gen']):.3f}",
            f"{np.max(prop['gen']):.3f}",
        ],
    }
    tbl = ax2.table(
        cellText=list(zip(stats["Statistic"], stats["ZINC250k"], stats["Generated"])),
        colLabels=["", "ZINC250k", "Generated"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Colour header row
    for col in range(3):
        tbl[0, col].set_facecolor("#2d2d2d")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    out_path = OUTPUT_DIR / prop["filename"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Combined 3-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("ZINC250k vs Generated — Property Distributions",
             fontsize=14, fontweight="bold")

for ax, prop in zip(axes, PROPS):
    ax.hist(prop["train"], bins=prop["bins"], density=True,
            color=TRAIN_COLOR, alpha=ALPHA,
            label=f"ZINC250k")
    ax.hist(prop["gen"], bins=prop["bins"], density=True,
            color=GEN_COLOR, alpha=ALPHA,
            label=f"Generated")

    grid = np.linspace(prop["bins"][0], prop["bins"][-1], 500)
    ax.plot(grid, gaussian_kde(prop["train"], bw_method="scott")(grid),
            color=TRAIN_COLOR, linewidth=2)
    ax.plot(grid, gaussian_kde(prop["gen"], bw_method="scott")(grid),
            color=GEN_COLOR, linewidth=2)

    ax.set_title(prop["name"], fontsize=12, fontweight="bold")
    ax.set_xlabel(prop["xlabel"], fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(prop["bins"][0], prop["bins"][-1])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95,
            f"KL = {prop['kl']:.4f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

fig.tight_layout()
combined_path = OUTPUT_DIR / "dist_combined.png"
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {combined_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  DISTRIBUTION SUMMARY")
print(f"{'='*55}")
print(f"  {'Property':<8}  {'KL(train‖gen)':>14}  {'Δmean':>8}  {'Δstd':>8}")
print(f"  {'-'*44}")
for prop in PROPS:
    delta_mean = np.mean(prop["gen"]) - np.mean(prop["train"])
    delta_std  = np.std(prop["gen"])  - np.std(prop["train"])
    print(f"  {prop['name']:<8}  {prop['kl']:>14.4f}  {delta_mean:>+8.3f}  {delta_std:>+8.3f}")
print(f"{'='*55}\n")
print("Note: KL < 0.1 = excellent, 0.1–0.3 = good, > 0.5 = notable shift")
