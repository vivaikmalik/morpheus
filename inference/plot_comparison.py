"""
Side-by-side training curve comparison:
  (a) QED-only run — 500 steps, unconstrained
  (b) QED + MW + Diversity run — 300 steps, multi-objective

Usage
-----
    python inference/plot_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

ROOT      = Path(__file__).parent.parent
OUTPUT    = ROOT / "outputs" / "plots" / "comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

LOG_A = ROOT / "outputs" / "rl" / "logs" / "rl_log_v2.csv"
LOG_B = ROOT / "outputs" / "rl" / "logs" / "rl_log_v2_qed_mw_div_50steps.csv"

WINDOW    = 20
BASE_QED  = 0.731
Y_LIM     = (0.3, 0.85)

# ── Load data ──────────────────────────────────────────────────────────────────
df_a = pd.read_csv(LOG_A)
df_b = pd.read_csv(LOG_B)

def smooth(series):
    return series.rolling(WINDOW, min_periods=1, center=True).mean()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
fig.suptitle(
    "REINFORCE Training Stability: Unconstrained vs Multi-Objective",
    fontsize=13, fontweight="bold", y=1.02,
)

for ax, df, title in [
    (ax_a, df_a, "(a) QED only — unconstrained"),
    (ax_b, df_b, "(b) QED + MW + Diversity — multi-objective"),
]:
    steps  = df["step"]
    qed    = df["mean_qed"]
    qed_sm = smooth(qed)

    ax.plot(steps, qed,    color="#aec6e8", linewidth=0.8, alpha=0.5)
    ax.plot(steps, qed_sm, color="#1f77b4", linewidth=2.0, label="Mean QED (smoothed)")
    ax.axhline(BASE_QED, color="gray", linestyle="--", linewidth=1.4,
               label=f"Base model ({BASE_QED})")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("RL Step", fontsize=11)
    ax.set_ylim(*Y_LIM)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

ax_a.set_ylabel("Mean QED", fontsize=11)

fig.tight_layout()
out_path = OUTPUT / "training_curves_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
