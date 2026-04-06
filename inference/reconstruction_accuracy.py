"""
Reconstruction accuracy vs noise level.

Loads the best checkpoint and ZINC250K validation set, then for each noise
level t in [0.1 … 0.9]:
  - takes 200 real molecules
  - masks them using the cosine schedule at that fixed t
  - runs one forward pass
  - computes token-level accuracy on the masked (non-pad) positions
Saves a plot of accuracy vs t.
"""

import sys
from pathlib import Path

import math
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.conditionalSELFIESDataset import ConditionalSELFIESDataset
from model.molecularDiffusionModel import MolecularDiffusionModel

# ── constants ────────────────────────────────────────────────────────────────
CHECKPOINT   = ROOT / "checkpoints" / "zinc_5M_pretrain.pt"
TOKENIZER    = ROOT / "chemical_tokenizer.json"
OUTPUT_PLOT  = ROOT / "outputs" / "reconstruction_accuracy.png"
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_MOLECULES  = 200
VAL_FRACTION = 0.05
SEED         = 42


def cosine_alpha(t: float) -> float:
    """Fraction of tokens that survive (are NOT masked) at noise level t."""
    return math.cos(t * math.pi / 2) ** 2


def mask_batch(input_ids: torch.Tensor, t: float, mask_id: int, pad_id: int):
    """
    Apply the cosine-schedule mask at a fixed noise level t.

    Returns
    -------
    masked_ids : (B, L) – input with some tokens replaced by mask_id
    is_masked  : (B, L) bool – True where a non-pad token was masked
    """
    B, L = input_ids.shape
    alpha_t = cosine_alpha(t)

    # Per-token Bernoulli: mask if uniform > alpha_t
    noise = torch.rand(B, L, device=input_ids.device)
    is_real   = input_ids != pad_id          # (B, L) – exclude padding
    is_masked = (noise > alpha_t) & is_real  # (B, L)

    masked_ids = input_ids.clone()
    masked_ids[is_masked] = mask_id
    return masked_ids, is_masked


def load_model(checkpoint_path: Path, tokenizer: ChemicalTokenizer, device: torch.device):
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model  = MolecularDiffusionModel(
        vocab_size   = config["vocab_size"],
        hidden_size  = config["hidden_size"],
        num_heads    = config["num_heads"],
        ffn_dim      = config["ffn_dim"],
        num_layers   = config["num_layers"],
        max_length   = config["max_length"],
        pad_token_id = tokenizer.pad_token_id,
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  : {checkpoint_path.name}")
    print(f"  step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    return model


def build_val_dataset(tokenizer: ChemicalTokenizer):
    print("Loading ZINC250K …")
    df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
    full_ds  = ConditionalSELFIESDataset(df, tokenizer)
    val_size = int(len(full_ds) * VAL_FRACTION)
    _, val_ds = random_split(
        full_ds,
        [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Validation set size: {len(val_ds)}")
    return val_ds


def eval_noise_level(
    model:     MolecularDiffusionModel,
    input_ids: torch.Tensor,          # (N, L) on device
    t:         float,
    mask_id:   int,
    pad_id:    int,
    device:    torch.device,
) -> float:
    """Return token-level accuracy for masked tokens at noise level t."""
    masked_ids, is_masked = mask_batch(input_ids, t, mask_id, pad_id)

    n_masked = is_masked.sum().item()
    if n_masked == 0:
        return float("nan")

    timesteps = torch.full((input_ids.shape[0], 1), t, device=device)

    with torch.no_grad():
        logits = model(masked_ids, timesteps)   # (B, L, V)

    preds   = logits.argmax(dim=-1)             # (B, L)
    correct = ((preds == input_ids) & is_masked).sum().item()
    return correct / n_masked


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = ChemicalTokenizer(str(TOKENIZER))
    model     = load_model(CHECKPOINT, tokenizer, device)
    val_ds    = build_val_dataset(tokenizer)

    # ── sample N_MOLECULES deterministically from the validation set ─────────
    rng     = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(val_ds), generator=rng)[:N_MOLECULES].tolist()
    samples = [val_ds[i]["input_ids"] for i in indices]
    input_ids = torch.stack(samples).to(device)   # (N, L)
    print(f"Using {input_ids.shape[0]} molecules, seq_len={input_ids.shape[1]}")

    # ── sweep noise levels ────────────────────────────────────────────────────
    accuracies = []
    for t in NOISE_LEVELS:
        alpha = cosine_alpha(t)
        acc   = eval_noise_level(
            model, input_ids, t,
            mask_id=tokenizer.mask_token_id,
            pad_id=tokenizer.pad_token_id,
            device=device,
        )
        # Expected fraction masked = 1 – alpha_t (for non-pad tokens)
        exp_masked_frac = 1.0 - alpha
        print(f"  t={t:.1f}  alpha={alpha:.3f}  ~{exp_masked_frac*100:.0f}% masked  "
              f"accuracy={acc*100:.1f}%")
        accuracies.append(acc * 100)

    # ── plot ──────────────────────────────────────────────────────────────────
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(NOISE_LEVELS, accuracies, marker="o", linewidth=2, color="#2563EB",
            label="Reconstruction accuracy")
    ax.set_xlabel("Noise level  t", fontsize=12)
    ax.set_ylabel("Token-level accuracy on masked tokens (%)", fontsize=12)
    ax.set_title("Reconstruction Accuracy vs Noise Level\n(cosine schedule, 200 val molecules)",
                 fontsize=13)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    # Annotate each point
    for t, acc in zip(NOISE_LEVELS, accuracies):
        ax.annotate(f"{acc:.1f}%", (t, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#1e3a8a")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nPlot saved → {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
