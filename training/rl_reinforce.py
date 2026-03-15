"""
REINFORCE fine-tuning for molecular property optimisation (QED + diversity).

Usage
-----
    python training/rl_reinforce.py \\
        --checkpoint checkpoints/best_model.pt \\
        --num_steps  500   \\
        --batch_size 32    \\
        --diversity_weight 0.3 \\
        --temperature 0.9
"""

import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdMolDescriptors, Descriptors
import selfies as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
# import evaluate module so we can mutate its GEN_CONFIG before each call
import inference.evaluate as ev
from inference.evaluate import generate_batch, decode_and_analyze, load_model


# =============================================================================
# REWARD
# =============================================================================

def _decode_molecule(ids, tokenizer):
    """Strip specials; return (selfies_str, canonical_smiles | None, mol | None)."""
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    specials = {tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id}
    clean_ids = [i for i in ids if i not in specials]
    selfies_str = tokenizer.decode(clean_ids)
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return selfies_str, Chem.MolToSmiles(mol), mol
    except Exception:
        pass
    return selfies_str, None, None


def compute_rewards(raw_ids_list, tokenizer, diversity_weight):
    """
    QED score minus a diversity penalty for each molecule.

    Parameters
    ----------
    raw_ids_list      : list of list[int], one entry per molecule
    tokenizer         : ChemicalTokenizer
    diversity_weight  : float  (0 = pure QED, >0 penalises within-batch similarity)

    Returns
    -------
    rewards          : list[float]
    qed_scores       : list[float]   (0.0 for invalid molecules)
    mean_sims        : list[float]   mean Tanimoto similarity to the rest of the batch
    canonical_smiles : list[str|None]
    """
    qed_scores, canonical_smiles, mols = [], [], []

    for ids in raw_ids_list:
        _, smi, mol = _decode_molecule(ids, tokenizer)
        canonical_smiles.append(smi)
        if mol is not None:
            qed_scores.append(float(QED.qed(mol)))
            mols.append(mol)
        else:
            qed_scores.append(0.0)
            mols.append(None)

    # Morgan fingerprints (radius=2, 2048 bits)
    fps = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if mol is not None else None
        for mol in mols
    ]

    # Mean Tanimoto similarity per molecule to the rest of the batch
    mean_sims = []
    for i, fp_i in enumerate(fps):
        if fp_i is None:
            mean_sims.append(0.0)
            continue
        sims = [
            DataStructs.TanimotoSimilarity(fp_i, fp_j)
            for j, fp_j in enumerate(fps)
            if j != i and fp_j is not None
        ]
        mean_sims.append(float(np.mean(sims)) if sims else 0.0)

    rewards = [q - diversity_weight * s for q, s in zip(qed_scores, mean_sims)]
    return rewards, qed_scores, mean_sims, canonical_smiles


# =============================================================================
# LOG-PROBABILITY
# =============================================================================

def compute_log_probs(model, ref_model, input_ids, tokenizer, device):
    """
    Forward passes at t=0 through both the policy and the frozen reference model.

    For each molecule, sums log P(token_i) over every position that is not
    a PAD, EOS, or MASK token.

    Parameters
    ----------
    model      : MolecularDiffusionModel  (trainable, gradients flow)
    ref_model  : MolecularDiffusionModel  (frozen reference, no gradients)
    input_ids  : (B, L) LongTensor on device
    tokenizer  : ChemicalTokenizer

    Returns
    -------
    policy_log_probs : (B,) FloatTensor with gradients
    ref_log_probs    : (B,) FloatTensor without gradients
    """
    B, L = input_ids.shape
    t_zero = torch.zeros(B, 1, device=device)

    # Mask out special tokens — they carry no meaningful signal
    specials = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id]
    is_real  = torch.ones(B, L, dtype=torch.bool, device=device)
    for sid in specials:
        is_real &= (input_ids != sid)

    # Policy forward pass (with gradients)
    logits   = model(input_ids, t_zero)                                          # (B, L, V)
    log_prob = F.log_softmax(logits, dim=-1)                                     # (B, L, V)
    token_lp = torch.gather(log_prob, 2, input_ids.unsqueeze(-1)).squeeze(-1)   # (B, L)
    policy_log_probs = (token_lp * is_real.float()).sum(dim=-1)                 # (B,)

    # Reference forward pass (no gradients)
    with torch.no_grad():
        ref_logits   = ref_model(input_ids, t_zero)                                      # (B, L, V)
        ref_log_prob = F.log_softmax(ref_logits, dim=-1)                                 # (B, L, V)
        ref_token_lp = torch.gather(ref_log_prob, 2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)
        ref_log_probs = (ref_token_lp * is_real.float()).sum(dim=-1)                     # (B,)

    return policy_log_probs, ref_log_probs


# =============================================================================
# PERIODIC FULL EVALUATION
# =============================================================================

def full_eval_metrics(model, tokenizer, device, batch_size, model_config,
                      temperature, gen_steps):
    """
    Generate 100 molecules and return a dict of quality metrics.
    Novelty is omitted (would require loading ZINC250k on each call).
    """
    model.eval()
    ev.GEN_CONFIG["num_steps"]   = gen_steps
    ev.GEN_CONFIG["temperature"] = temperature

    all_raw, remaining = [], 100
    while remaining > 0:
        bs = min(batch_size, remaining)
        all_raw.extend(generate_batch(model, tokenizer, device, bs, model_config))
        remaining -= bs

    results = [decode_and_analyze(ids, tokenizer) for ids in all_raw]
    valid   = [r for r in results if r["valid"]]

    validity   = len(valid) / len(results)
    uniqueness = len({r["canonical_smiles"] for r in valid}) / len(valid) if valid else 0.0
    qeds       = [r["qed"] for r in valid if r["qed"]  is not None]
    mws        = [r["mw"]  for r in valid if r["mw"]   is not None]

    # Pairwise Tanimoto diversity (capped at 50 molecules to stay fast)
    subset_fps = []
    for r in valid[:50]:
        mol = Chem.MolFromSmiles(r["canonical_smiles"])
        if mol:
            subset_fps.append(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            )

    mean_diversity = 0.0
    if len(subset_fps) >= 2:
        pair_sims = [
            DataStructs.TanimotoSimilarity(subset_fps[i], subset_fps[j])
            for i in range(len(subset_fps))
            for j in range(i + 1, len(subset_fps))
        ]
        mean_diversity = 1.0 - float(np.mean(pair_sims))

    model.train()
    return {
        "validity"  : validity,
        "uniqueness": uniqueness,
        "novelty"   : float("nan"),   # skipped — loading ZINC250k each eval is too slow
        "mean_qed"  : float(np.mean(qeds)) if qeds else 0.0,
        "diversity" : mean_diversity,
        "mean_mw"   : float(np.mean(mws)) if mws else 0.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE fine-tuning for molecular QED optimisation"
    )
    parser.add_argument("--checkpoint",       default="checkpoints/best_model.pt",
                        help="Path to pretrained checkpoint (relative to project root)")
    parser.add_argument("--num_steps",        type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--diversity_weight", type=float, default=0.0,
                        help="Weight on within-batch similarity penalty (0 = pure QED)")
    parser.add_argument("--temperature",      type=float, default=0.9)
    parser.add_argument("--gen_steps",        type=int,   default=32,
                        help="MaskGIT denoising steps used during generation")
    parser.add_argument("--beta",             type=float, default=0.1,
                        help="KL penalty weight (0 = no regularisation)")
    parser.add_argument("--output_dir",       default="outputs/rl")
    args = parser.parse_args()

    # ── paths ──────────────────────────────────────────────────────────────────
    checkpoint_path = ROOT / args.checkpoint
    output_dir      = ROOT / args.output_dir
    checkpoint_dir  = ROOT / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "rl_log.csv"

    # ── device ─────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

    # ── model ──────────────────────────────────────────────────────────────────
    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    model, model_config = load_model(checkpoint_path, tokenizer, device)
    model.train()

    # Frozen reference model — loaded once, never updated
    ref_model, _ = load_model(checkpoint_path, tokenizer, device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Pre-configure the generation settings used by generate_batch
    ev.GEN_CONFIG["num_steps"]   = args.gen_steps
    ev.GEN_CONFIG["temperature"] = args.temperature

    # ── CSV header ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "mean_qed", "mean_reward", "mean_diversity",
            "loss", "mean_kl", "validity",
        ])

    print(
        f"\nStarting REINFORCE\n"
        f"  steps={args.num_steps}  batch={args.batch_size}  lr={args.lr}\n"
        f"  diversity_weight={args.diversity_weight}  beta={args.beta}  "
        f"temperature={args.temperature}  gen_steps={args.gen_steps}\n"
    )

    # ── training loop ──────────────────────────────────────────────────────────
    for step in range(1, args.num_steps + 1):

        # 1. Generate batch with MaskGIT (no_grad is applied inside generate_batch)
        model.eval()
        raw_ids_list = generate_batch(
            model, tokenizer, device, args.batch_size, model_config
        )
        model.train()

        # 2. Compute rewards (QED − diversity_weight * mean_similarity)
        rewards, qed_scores, mean_sims, canonical_smiles = compute_rewards(
            raw_ids_list, tokenizer, args.diversity_weight
        )
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 3. Pack token IDs into a (B, L) tensor for the log-prob forward pass
        input_ids = torch.tensor(raw_ids_list, dtype=torch.long, device=device)

        # 4. Compute policy and reference log-probs at t=0
        policy_log_probs, ref_log_probs = compute_log_probs(
            model, ref_model, input_ids, tokenizer, device
        )  # both (B,)

        # 5. REINFORCE loss + KL penalty
        advantage  = (rewards_t - rewards_t.mean()).clamp(-2, 2)
        kl_penalty = (policy_log_probs - ref_log_probs).clamp(min=0)  # (B,) one-sided
        loss       = -(advantage * policy_log_probs).mean() + args.beta * kl_penalty.mean()

        # 6. Gradient step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 7. Scalar metrics for this step
        mean_qed       = float(np.mean(qed_scores))
        mean_reward    = float(np.mean(rewards))
        mean_diversity = float(np.mean([1.0 - s for s in mean_sims]))
        mean_kl        = float(kl_penalty.mean().item())
        validity       = sum(s is not None for s in canonical_smiles) / len(canonical_smiles)

        # 8. Console + CSV logging (every step)
        print(
            f"Step {step:>4}  |  "
            f"QED={mean_qed:.4f}  "
            f"reward={mean_reward:.4f}  "
            f"diversity={mean_diversity:.4f}  "
            f"KL={mean_kl:.4f}  "
            f"loss={loss.item():.4f}  "
            f"valid={validity*100:.1f}%"
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step,
                f"{mean_qed:.4f}",
                f"{mean_reward:.4f}",
                f"{mean_diversity:.4f}",
                f"{loss.item():.4f}",
                f"{mean_kl:.4f}",
                f"{validity:.4f}",
            ])

        # 9. Full evaluation every 50 steps (100 molecules, full metric suite)
        if step % 50 == 0:
            print(f"\n{'─'*60}")
            print(f"  Full evaluation @ step {step}")
            m = full_eval_metrics(
                model, tokenizer, device, args.batch_size,
                model_config, args.temperature, args.gen_steps,
            )
            print(
                f"  validity={m['validity']*100:.1f}%  "
                f"uniqueness={m['uniqueness']*100:.1f}%  "
                f"QED={m['mean_qed']:.4f}  "
                f"diversity={m['diversity']:.4f}  "
                f"MW={m['mean_mw']:.1f}"
            )
            print(f"{'─'*60}\n")
            # Restore generation settings (full_eval_metrics may touch GEN_CONFIG)
            ev.GEN_CONFIG["num_steps"]   = args.gen_steps
            ev.GEN_CONFIG["temperature"] = args.temperature

        # 10. Checkpoint every 100 steps
        if step % 100 == 0:
            ckpt_path = checkpoint_dir / f"rl_step_{step}.pt"
            torch.save({
                "step"     : step,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config"   : model_config,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path.name}")

    print(f"\nDone.  Log: {csv_path}")


if __name__ == "__main__":
    main()
