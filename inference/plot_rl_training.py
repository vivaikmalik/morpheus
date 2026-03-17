"""
Plot RL training curves: QED and Diversity over steps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

OUTPUT_DIR = Path("outputs/rl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "rl_log_v2.csv"
SMOOTH_WINDOW = 20
BASE_QED = 0.731
BASE_DIVERSITY = 0.851

print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows, columns: {list(df.columns)}")

steps = df["step"]
qed_raw = df["mean_qed"]
div_raw = df["mean_diversity"]

qed_smooth = qed_raw.rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
div_smooth = div_raw.rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()

# --- Plot 1: QED ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, qed_raw, color="#aec6e8", linewidth=0.8, alpha=0.6, label="_nolegend_")
ax.plot(steps, qed_smooth, color="#1f77b4", linewidth=2.0, label="Mean QED (smoothed)")
ax.axhline(BASE_QED, color="gray", linestyle="--", linewidth=1.4, label=f"Base model ({BASE_QED})")
ax.set_xlabel("RL Step", fontsize=12)
ax.set_ylabel("Mean QED", fontsize=12)
ax.set_title("REINFORCE QED Optimization", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_qed = OUTPUT_DIR / "qed_training_curve.png"
fig.savefig(out_qed, dpi=150)
plt.close(fig)
print(f"  Saved: {out_qed}")

# --- Plot 2: Diversity ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, div_raw, color="#b5d4b0", linewidth=0.8, alpha=0.6, label="_nolegend_")
ax.plot(steps, div_smooth, color="#2ca02c", linewidth=2.0, label="Internal Diversity (smoothed)")
ax.axhline(BASE_DIVERSITY, color="gray", linestyle="--", linewidth=1.4,
           label=f"Base model ({BASE_DIVERSITY})")
ax.set_xlabel("RL Step", fontsize=12)
ax.set_ylabel("Internal Diversity", fontsize=12)
ax.set_title("REINFORCE Diversity over Training", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_div = OUTPUT_DIR / "diversity_training_curve.png"
fig.savefig(out_div, dpi=150)
plt.close(fig)
print(f"  Saved: {out_div}")
