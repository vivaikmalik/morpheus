"""
REINFORCE v2 — correct per-step log-probability accumulation.

Key difference from v1: log-probabilities are accumulated during the MaskGIT
generation loop itself.  At each denoising step we record log P(token) for
every position that is locked in *at that step* (was MASK before, becomes
non-MASK after confidence-based re-masking).  This gives the true log-
probability of the generated sequence under the multi-step policy, making
the REINFORCE gradient theoretically sound.

Usage
-----
    python training/rl_reinforce_v2.py \\
        --checkpoint checkpoints/best_model.pt \\
        --num_steps  200  \\
        --batch_size 32   \\
        --gen_steps  15   \\
        --beta       0.1  \\
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
from rdkit.Chem import QED, rdMolDescriptors
import selfies as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
import inference.evaluate as ev
from inference.evaluate import generate_batch, decode_and_analyze, load_model


# =============================================================================
# GENERATION WITH LOG-PROB ACCUMULATION
# =============================================================================

def generate_batch_with_log_probs(model, tokenizer, device, batch_size,
                                  model_config, num_steps=15, temperature=0.9):
    """
    MaskGIT decoding with gradients enabled.  Log-probabilities are accumulated
    step-by-step: at each denoising step, for every position that transitions
    from MASK → non-MASK (i.e., is locked in at this step), we record
    log P(chosen_token | context, t).

    This gives the correct sum of conditional log-probs over the generation
    trajectory, suitable for the REINFORCE policy gradient.

    Parameters
    ----------
    model        : MolecularDiffusionModel  (train mode, gradients enabled)
    tokenizer    : ChemicalTokenizer
    device       : torch.device
    batch_size   : int
    model_config : dict  (must contain "max_length")
    num_steps    : int   number of MaskGIT denoising steps (default 15)
    temperature  : float sampling temperature

    Returns
    -------
    input_ids       : (B, L) LongTensor  — final generated token IDs
    total_log_probs : (B,)   FloatTensor — accumulated log-prob **with gradients**
    """
    max_length = model_config["max_length"]
    B          = batch_size

    # Start fully masked
    input_ids = torch.full(
        (B, max_length), tokenizer.mask_token_id,
        dtype=torch.long, device=device,
    )

    t_vals          = torch.linspace(1.0, 0.0, num_steps, device=device)
    total_log_probs = torch.zeros(B, device=device)   # accumulates across steps

    for step_idx, t_val in enumerate(t_vals):
        is_last_step = (step_idx == num_steps - 1)

        # Which positions are still masked entering this step?
        was_masked = (input_ids == tokenizer.mask_token_id)  # (B, L)  bool

        # ── Forward pass (gradients flow) ────────────────────────────────────
        step_t = t_val.repeat(B).unsqueeze(-1)          # (B, 1)
        logits = model(input_ids, step_t)                # (B, L, V)

        # Block MASK from ever being a final output token
        logits[:, :, tokenizer.mask_token_id] = float("-inf")

        scaled_logits = logits / temperature
        log_p  = F.log_softmax(scaled_logits, dim=-1)   # (B, L, V)  — has grad
        probs  = torch.exp(log_p)                        # (B, L, V)

        # ── Sample ───────────────────────────────────────────────────────────
        # Discrete sampling: no gradient through sample(), grad lives in log_p
        sampled = torch.distributions.Categorical(probs=probs).sample()  # (B, L)

        if is_last_step:
            # Lock in all remaining masked positions
            newly_locked = was_masked                              # (B, L)
            token_lp     = torch.gather(
                log_p, 2, sampled.unsqueeze(-1)
            ).squeeze(-1)                                          # (B, L)
            # masked_fill avoids -inf * 0 = NaN for already-locked positions
            total_log_probs = total_log_probs + token_lp.masked_fill(~newly_locked, 0.0).sum(dim=-1)
            input_ids = sampled
            break

        # ── Confidence-based re-masking ───────────────────────────────────────
        confidence = torch.gather(
            probs, 2, sampled.unsqueeze(-1)
        ).squeeze(-1)                                              # (B, L)

        alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
        num_to_mask = int((1.0 - alpha_t) * max_length)

        if num_to_mask > 0:
            _, mask_indices = torch.topk(
                confidence,
                k=min(num_to_mask, max_length),
                dim=-1,
                largest=False,   # re-mask least confident
            )
            sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

        # ── Record log-probs for newly locked tokens ──────────────────────────
        # Newly locked = was MASK before the step, is non-MASK after re-masking.
        # For these positions sampled[i,j] is the actual locked token
        # (positions that got re-masked were scattered back to mask_token_id).
        newly_locked = was_masked & (sampled != tokenizer.mask_token_id)  # (B, L)

        # Use the post-scatter sampled: for newly_locked positions sampled[i,j]
        # is the actual locked token; for re-masked positions sampled[i,j] is
        # mask_token_id whose log_p = -inf.  masked_fill replaces those -inf
        # values with 0.0 before summing, preventing -inf * 0 = NaN.
        token_lp = torch.gather(
            log_p, 2, sampled.unsqueeze(-1)
        ).squeeze(-1)                                              # (B, L)
        total_log_probs = total_log_probs + token_lp.masked_fill(~newly_locked, 0.0).sum(dim=-1)

        input_ids = sampled

    return input_ids, total_log_probs   # (B, L), (B,) with grad


# =============================================================================
# REWARD  (unchanged from v1)
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

    Returns
    -------
    rewards, qed_scores, mean_sims, canonical_smiles
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

    fps = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if mol is not None else None
        for mol in mols
    ]

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
# REFERENCE LOG-PROBABILITY  (single t=0 pass, no gradients)
# =============================================================================

def compute_ref_log_probs(ref_model, input_ids, tokenizer, device):
    """
    Single forward pass through the frozen reference model at t=0.
    Returns (B,) FloatTensor without gradients.

    The reference log-prob is used as a baseline for the KL penalty.
    It is computed at t=0 (a single-step reconstruction probability), which
    is an approximation to the multi-step reference policy — but sufficient
    as an anchoring signal.
    """
    B, L  = input_ids.shape
    t_zero = torch.zeros(B, 1, device=device)

    specials = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id]
    is_real  = torch.ones(B, L, dtype=torch.bool, device=device)
    for sid in specials:
        is_real &= (input_ids != sid)

    with torch.no_grad():
        ref_logits = ref_model(input_ids, t_zero)                                       # (B, L, V)
        ref_lp     = F.log_softmax(ref_logits, dim=-1)                                  # (B, L, V)
        token_lp   = torch.gather(ref_lp, 2, input_ids.unsqueeze(-1)).squeeze(-1)      # (B, L)

    return (token_lp * is_real.float()).sum(dim=-1)   # (B,)


# =============================================================================
# PERIODIC FULL EVALUATION  (unchanged from v1, uses 50-step generate_batch)
# =============================================================================

def full_eval_metrics(model, tokenizer, device, batch_size, model_config, temperature):
    """Generate 100 molecules with 50 denoising steps and return quality metrics."""
    model.eval()
    ev.GEN_CONFIG["num_steps"]   = 50
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
    qeds       = [r["qed"] for r in valid if r["qed"] is not None]
    mws        = [r["mw"]  for r in valid if r["mw"]  is not None]

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
        "mean_qed"  : float(np.mean(qeds)) if qeds else 0.0,
        "diversity" : mean_diversity,
        "mean_mw"   : float(np.mean(mws)) if mws else 0.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE v2 — per-step log-prob accumulation"
    )
    parser.add_argument("--checkpoint",       default="checkpoints/best_model.pt",
                        help="Path to pretrained checkpoint (relative to project root)")
    parser.add_argument("--num_steps",        type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--diversity_weight", type=float, default=0.0,
                        help="Weight on within-batch similarity penalty (0 = pure QED)")
    parser.add_argument("--temperature",      type=float, default=0.9)
    parser.add_argument("--gen_steps",        type=int,   default=15,
                        help="MaskGIT denoising steps during RL training (default 15)")
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
    csv_path = output_dir / "rl_log_v2.csv"

    # ── device ─────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

    # ── models ─────────────────────────────────────────────────────────────────
    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))

    model, model_config = load_model(checkpoint_path, tokenizer, device)
    model.train()

    # Frozen reference model — loaded once, never updated
    ref_model, _ = load_model(checkpoint_path, tokenizer, device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── CSV header ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "mean_qed", "mean_reward", "mean_diversity",
            "loss", "mean_kl", "validity",
        ])

    print(
        f"\nStarting REINFORCE v2\n"
        f"  steps={args.num_steps}  batch={args.batch_size}  lr={args.lr}\n"
        f"  diversity_weight={args.diversity_weight}  beta={args.beta}  "
        f"temperature={args.temperature}  gen_steps={args.gen_steps}\n"
    )

    # ── training loop ──────────────────────────────────────────────────────────
    for step in range(1, args.num_steps + 1):

        # 1. Generate batch with per-step log-prob accumulation (gradients on)
        input_ids_t, policy_log_probs = generate_batch_with_log_probs(
            model, tokenizer, device, args.batch_size,
            model_config, num_steps=args.gen_steps, temperature=args.temperature,
        )   # input_ids_t: (B, L);  policy_log_probs: (B,) with grad

        # 2. Decode to list-of-lists for reward computation
        raw_ids_list = input_ids_t.detach().cpu().tolist()

        # 3. Rewards
        rewards, qed_scores, mean_sims, canonical_smiles = compute_rewards(
            raw_ids_list, tokenizer, args.diversity_weight
        )
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 4. Reference log-probs at t=0 (no gradients)
        ref_log_probs = compute_ref_log_probs(
            ref_model, input_ids_t, tokenizer, device
        )   # (B,)

        # 5. REINFORCE loss + one-sided KL penalty
        advantage  = (rewards_t - rewards_t.mean()).clamp(-2, 2)
        kl_penalty = (policy_log_probs - ref_log_probs).clamp(min=0)   # (B,)
        loss       = -(advantage * policy_log_probs).mean() + args.beta * kl_penalty.mean()

        # 6. Gradient step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 7. Scalar metrics
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

        # 9. Full evaluation every 50 steps (100 molecules, 50-step generate_batch)
        if step % 50 == 0:
            print(f"\n{'─'*60}")
            print(f"  Full evaluation @ step {step}")
            m = full_eval_metrics(
                model, tokenizer, device, args.batch_size,
                model_config, args.temperature,
            )
            print(
                f"  validity={m['validity']*100:.1f}%  "
                f"uniqueness={m['uniqueness']*100:.1f}%  "
                f"QED={m['mean_qed']:.4f}  "
                f"diversity={m['diversity']:.4f}  "
                f"MW={m['mean_mw']:.1f}"
            )
            print(f"{'─'*60}\n")

        # 10. Checkpoint every 100 steps
        if step % 100 == 0:
            ckpt_path = checkpoint_dir / f"rl_v2_step_{step}.pt"
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
