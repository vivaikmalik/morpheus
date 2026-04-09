"""
RL post-training for prompt adherence in text-conditioned molecular diffusion.

Optimises a weighted combination of Tanimoto similarity and BLEU-2/4 scores
between generated molecules and ground-truth molecules from the dataset,
conditioned on the same natural-language prompt.

The RL algorithm is pluggable via --algorithm (reinforce, ppo, grpo).
Full ChEBI-20 evaluation runs at the end and logs to wandb.

Usage
-----
    python rl/train_prompt_adherence.py \
        --algorithm  reinforce \
        --checkpoint checkpoints/best_finetuned_model.pt \
        --data_path  train.csv \
        --num_steps  500 \
        --batch_size 16 \
        --cfg_scale  3.0 \
        --lr         1e-5 \
        --beta       0.1 \
        --run_name   prompt_adherence_v1

    # GRPO example (4 samples per prompt, 4 prompts per batch = 16 total):
    python rl/train_prompt_adherence.py \
        --algorithm grpo --group_size 4 \
        --batch_size 16 ...

    # PPO example (4 optimisation epochs per generation batch):
    python rl/train_prompt_adherence.py \
        --algorithm ppo --ppo_epochs 4 \
        --batch_size 16 ...
"""

import sys
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import wandb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from transformers import AutoTokenizer

from rl.algorithms import get_algorithm, list_algorithms
from rl.generation import generate_batch_cfg_with_log_probs, generate_batch_cfg_no_grad
from rl.rewards import compute_rewards
from rl.utils import load_model, compute_ref_log_probs
from rl.evaluation import run_chebi20_eval, print_chebi20_metrics


# =============================================================================
# PERIODIC EVALUATION (quick, on training data)
# =============================================================================

def full_eval(model, hf_tokenizer, tokenizer, device, model_config,
              eval_df, batch_size=16, num_steps=32, temperature=0.9,
              cfg_scale=3.0, w_tanimoto=1.0, w_bleu2=0.5, w_bleu4=0.5,
              num_eval=100):
    """Generate molecules for a subset of eval prompts and compute metrics."""
    model.eval()

    eval_subset = eval_df.sample(n=min(num_eval, len(eval_df)), replace=False)
    prompts = eval_subset["prompt"].tolist()
    ref_selfies = eval_subset["response"].tolist()

    all_gen_ids = []
    all_ref_selfies = []

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_refs = ref_selfies[start:end]

        text_inputs = hf_tokenizer(
            batch_prompts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        text_padding_mask = (text_inputs["attention_mask"] == 0)

        with torch.no_grad():
            text_embeds = model.get_text_embeddings(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            )

        gen_ids = generate_batch_cfg_no_grad(
            model, tokenizer, device, text_embeds, text_padding_mask,
            model_config, num_steps=num_steps,
            temperature=temperature, cfg_scale=cfg_scale,
        )

        all_gen_ids.extend(gen_ids.cpu().tolist())
        all_ref_selfies.extend(batch_refs)

    rewards, tan_scores, b2_scores, b4_scores, valid_flags = compute_rewards(
        all_gen_ids, all_ref_selfies, tokenizer,
        w_tanimoto=w_tanimoto, w_bleu2=w_bleu2, w_bleu4=w_bleu4,
    )

    model.train()
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_tanimoto": float(np.mean(tan_scores)),
        "mean_bleu2": float(np.mean(b2_scores)),
        "mean_bleu4": float(np.mean(b4_scores)),
        "validity": float(np.mean(valid_flags)),
    }


# =============================================================================
# SINGLE RL STEP (shared across all algorithms)
# =============================================================================

def run_rl_step(
    algo, model, ref_model, optimizer, trainable_params,
    tokenizer, hf_tokenizer, device, model_config,
    prompts, ref_selfies_list, args,
):
    """
    Generate, compute reward, compute loss via the algorithm, update weights.
    Returns (reward_metrics, algo_metrics).
    """
    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    ).to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)

    text_embeds = model.get_text_embeddings(
        text_inputs["input_ids"], text_inputs["attention_mask"]
    )
    with torch.no_grad():
        ref_text_embeds = ref_model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )

    input_ids_t, policy_log_probs = generate_batch_cfg_with_log_probs(
        model, tokenizer, device, text_embeds, text_padding_mask,
        model_config, num_steps=args.gen_steps,
        temperature=args.temperature, cfg_scale=args.cfg_scale,
    )

    raw_ids_list = input_ids_t.detach().cpu().tolist()
    rewards, tan_scores, b2_scores, b4_scores, valid_flags = compute_rewards(
        raw_ids_list, ref_selfies_list, tokenizer,
        w_tanimoto=args.w_tanimoto, w_bleu2=args.w_bleu2,
        w_bleu4=args.w_bleu4, validity_bonus=args.validity_bonus,
    )
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    ref_log_probs = compute_ref_log_probs(
        ref_model, input_ids_t, ref_text_embeds, text_padding_mask,
        tokenizer, device, cfg_scale=args.cfg_scale,
    )

    step_output = algo.compute_loss(
        policy_log_probs=policy_log_probs,
        rewards=rewards_t,
        ref_log_probs=ref_log_probs,
    )

    optimizer.zero_grad()
    step_output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()

    reward_metrics = {
        "mean_reward": float(np.mean(rewards)),
        "mean_tanimoto": float(np.mean(tan_scores)),
        "mean_bleu2": float(np.mean(b2_scores)),
        "mean_bleu4": float(np.mean(b4_scores)),
        "validity": float(np.mean(valid_flags)),
        "loss": step_output.loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "mean_policy_log_prob": policy_log_probs.mean().item(),
        "std_reward": float(np.std(rewards)),
    }
    return reward_metrics, step_output.metrics


def run_ppo_step(
    algo, model, ref_model, optimizer, trainable_params,
    tokenizer, hf_tokenizer, device, model_config,
    prompts, ref_selfies_list, args,
):
    """
    PPO-specific: generate once, then run multiple optimisation epochs
    on the same batch using the clipped surrogate objective.
    """
    text_inputs = hf_tokenizer(
        prompts, padding=True, truncation=True,
        max_length=128, return_tensors="pt",
    ).to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)

    text_embeds = model.get_text_embeddings(
        text_inputs["input_ids"], text_inputs["attention_mask"]
    )
    with torch.no_grad():
        ref_text_embeds = ref_model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )

    input_ids_t, first_log_probs = generate_batch_cfg_with_log_probs(
        model, tokenizer, device, text_embeds, text_padding_mask,
        model_config, num_steps=args.gen_steps,
        temperature=args.temperature, cfg_scale=args.cfg_scale,
    )

    raw_ids_list = input_ids_t.detach().cpu().tolist()
    rewards, tan_scores, b2_scores, b4_scores, valid_flags = compute_rewards(
        raw_ids_list, ref_selfies_list, tokenizer,
        w_tanimoto=args.w_tanimoto, w_bleu2=args.w_bleu2,
        w_bleu4=args.w_bleu4, validity_bonus=args.validity_bonus,
    )
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    ref_log_probs = compute_ref_log_probs(
        ref_model, input_ids_t, ref_text_embeds, text_padding_mask,
        tokenizer, device, cfg_scale=args.cfg_scale,
    )

    # First epoch
    step_output = algo.compute_loss(
        policy_log_probs=first_log_probs,
        rewards=rewards_t,
        ref_log_probs=ref_log_probs,
    )
    optimizer.zero_grad()
    step_output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()

    # Subsequent epochs
    for _ in range(1, algo.ppo_epochs):
        text_embeds = model.get_text_embeddings(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        B, L = input_ids_t.shape
        t_zero = torch.zeros(B, 1, device=device)
        null_embeds = model.null_token.expand(B, text_embeds.size(1), -1)

        cond_logits = model(input_ids_t, t_zero, text_embeds, text_padding_mask)
        uncond_logits = model(input_ids_t, t_zero, null_embeds, text_padding_mask)
        cfg_logits = uncond_logits + args.cfg_scale * (cond_logits - uncond_logits)

        log_p = torch.nn.functional.log_softmax(cfg_logits, dim=-1)
        specials = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id]
        is_real = torch.ones(B, L, dtype=torch.bool, device=device)
        for sid in specials:
            is_real &= (input_ids_t != sid)
        token_lp = torch.gather(log_p, 2, input_ids_t.unsqueeze(-1)).squeeze(-1)
        new_log_probs = (token_lp * is_real.float()).sum(dim=-1)

        step_output = algo.compute_loss(
            policy_log_probs=new_log_probs,
            rewards=rewards_t,
            ref_log_probs=ref_log_probs,
        )
        optimizer.zero_grad()
        step_output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

    reward_metrics = {
        "mean_reward": float(np.mean(rewards)),
        "mean_tanimoto": float(np.mean(tan_scores)),
        "mean_bleu2": float(np.mean(b2_scores)),
        "mean_bleu4": float(np.mean(b4_scores)),
        "validity": float(np.mean(valid_flags)),
        "loss": step_output.loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "mean_policy_log_prob": first_log_probs.mean().item(),
        "std_reward": float(np.std(rewards)),
    }
    return reward_metrics, step_output.metrics


# =============================================================================
# GRPO DATA PREPARATION
# =============================================================================

def expand_for_grpo(prompts, ref_selfies_list, group_size):
    """Repeat each prompt G times for group-relative ranking."""
    expanded_prompts = []
    expanded_refs = []
    for p, r in zip(prompts, ref_selfies_list):
        expanded_prompts.extend([p] * group_size)
        expanded_refs.extend([r] * group_size)
    return expanded_prompts, expanded_refs


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL post-training for prompt adherence (pluggable algorithm)"
    )
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, default="reinforce",
                        choices=list_algorithms(),
                        help=f"RL algorithm: {', '.join(list_algorithms())}")

    # Model / data
    parser.add_argument("--checkpoint", default="checkpoints/best_finetuned_model.pt")
    parser.add_argument("--data_path", default="train.csv")
    parser.add_argument("--text_model", default="BAAI/bge-large-en-v1.5")

    # Training
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)

    # Generation
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--gen_steps", type=int, default=15)
    parser.add_argument("--eval_gen_steps", type=int, default=32)

    # Algorithm hyperparameters (shared)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty weight")

    # REINFORCE-specific
    parser.add_argument("--max_adv", type=float, default=3.0,
                        help="[reinforce] advantage clamp magnitude")

    # PPO-specific
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="[ppo/grpo] clipping epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="[ppo] optimisation epochs per batch")

    # GRPO-specific
    parser.add_argument("--group_size", type=int, default=4,
                        help="[grpo] samples per prompt for group ranking")

    # Reward weights
    parser.add_argument("--w_tanimoto", type=float, default=1.0)
    parser.add_argument("--w_bleu2", type=float, default=0.5)
    parser.add_argument("--w_bleu4", type=float, default=0.5)
    parser.add_argument("--validity_bonus", type=float, default=0.2)

    # Logging / checkpointing
    parser.add_argument("--run_name", type=str, default="prompt_adherence")
    parser.add_argument("--output_dir", default="outputs/rl_prompt")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--num_eval", type=int, default=100)

    # wandb
    parser.add_argument("--wandb_project", type=str, default="morpheus-rl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                        help="Download checkpoint from a wandb artifact instead of --checkpoint. "
                             "Format: 'entity/project/artifact_name:version' "
                             "e.g. 'myteam/morpheus-rl/rl-reinforce_v1-final:latest'")
    parser.add_argument("--wandb_artifact_text_encoder", type=str, default=None,
                        help="Download text encoder from a wandb artifact. "
                             "The artifact should contain a HF model directory. "
                             "Overrides --text_model when provided.")

    # Final ChEBI-20 evaluation
    parser.add_argument("--chebi20_eval", action="store_true", default=True,
                        help="Run ChEBI-20 eval at end of training")
    parser.add_argument("--no_chebi20_eval", action="store_true",
                        help="Skip ChEBI-20 eval at end of training")
    parser.add_argument("--chebi20_max_rows", type=int, default=None,
                        help="Limit ChEBI-20 eval rows (for debugging)")

    args = parser.parse_args()
    if args.no_chebi20_eval:
        args.chebi20_eval = False

    # ── Instantiate algorithm ─────────────────────────────────────────
    algo_kwargs = {"beta": args.beta}
    if args.algorithm == "reinforce":
        algo_kwargs["max_adv"] = args.max_adv
    elif args.algorithm == "ppo":
        algo_kwargs["clip_eps"] = args.clip_eps
        algo_kwargs["ppo_epochs"] = args.ppo_epochs
        algo_kwargs["max_adv"] = args.max_adv
    elif args.algorithm == "grpo":
        algo_kwargs["clip_eps"] = args.clip_eps
        algo_kwargs["group_size"] = args.group_size

    algo = get_algorithm(args.algorithm, **algo_kwargs)
    print(f"Algorithm: {args.algorithm.upper()} ({type(algo).__name__})")

    # ── Paths ─────────────────────────────────────────────────────────
    output_dir = ROOT / args.output_dir
    checkpoint_dir = ROOT / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"rl_log_{args.run_name}.csv"

    # ── Checkpoint: wandb artifact or local file ───────────────────────
    if args.wandb_artifact:
        print(f"Downloading wandb artifact: {args.wandb_artifact}")
        api = wandb.Api()
        artifact = api.artifact(args.wandb_artifact, type="model")
        # Prefer SLURM_TMPDIR (fast local SSD, ~800GB) → SCRATCH → fallback to checkpoints/
        import os as _os
        _artifact_root = (
            _os.environ.get("SLURM_TMPDIR")
            or _os.environ.get("SCRATCH")
            or str(checkpoint_dir / "_artifacts")
        )
        artifact_dir = Path(artifact.download(root=_artifact_root))
        pt_files = list(artifact_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in artifact {args.wandb_artifact}")
        checkpoint_path = pt_files[0]
        print(f"  Downloaded → {checkpoint_path}")
    else:
        checkpoint_path = ROOT / args.checkpoint

    # ── Text encoder: wandb artifact or HF hub ─────────────────────────
    text_encoder_ckpt = None  # path to contrastive .pt checkpoint (if any)
    if args.wandb_artifact_text_encoder:
        print(f"Downloading text encoder artifact: {args.wandb_artifact_text_encoder}")
        api = wandb.Api() if not args.wandb_artifact else api  # reuse if already created
        te_artifact = api.artifact(args.wandb_artifact_text_encoder, type="model")
        import os as _os
        _te_root = (
            _os.environ.get("SLURM_TMPDIR")
            or _os.environ.get("SCRATCH")
            or str(checkpoint_dir / "_artifacts")
        )
        te_dir = Path(te_artifact.download(root=_te_root))
        pt_files = list(te_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in text encoder artifact {args.wandb_artifact_text_encoder}")
        text_encoder_ckpt = pt_files[0]
        print(f"  Text encoder checkpoint → {text_encoder_ckpt}")

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_path)
    if "response" not in df.columns or "prompt" not in df.columns:
        raise ValueError("CSV must have 'prompt' and 'response' columns.")
    print(f"Dataset: {len(df):,} rows from {args.data_path}")

    # ── Resolve contrastive text model name (for tokenizer only) ────────
    # The diffusion checkpoint was trained with the default text model (e.g. BGE-1024),
    # so load_model must use that to match text_proj dims.  The contrastive encoder
    # (e.g. scibert-768) is swapped in afterwards via set_text_encoder, which creates
    # encoder_proj (768→1024) to bridge the dimension gap.
    contrastive_text_model = None
    if text_encoder_ckpt is not None:
        _te_ckpt = torch.load(text_encoder_ckpt, map_location="cpu", weights_only=False)
        contrastive_text_model = _te_ckpt.get("config", {}).get("text_model")
        if contrastive_text_model:
            print(f"  Contrastive text model: {contrastive_text_model}")
        del _te_ckpt

    # ── Tokenizers ────────────────────────────────────────────────────
    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    # Use the contrastive encoder's tokenizer when swapping encoders
    hf_tokenizer = AutoTokenizer.from_pretrained(contrastive_text_model or args.text_model)

    # ── Policy model ──────────────────────────────────────────────────
    print("\n--- Policy model ---")
    model, model_config = load_model(
        checkpoint_path, tokenizer, device, args.text_model
    )
    # Swap in contrastive text encoder after loading (set_text_encoder handles dim mismatch)
    if text_encoder_ckpt is not None:
        from finetune.train_chebi20 import _load_contrastive_text_encoder
        contrastive_encoder = _load_contrastive_text_encoder(text_encoder_ckpt, device)
        model.set_text_encoder(contrastive_encoder)
    model.train()

    # ── Reference model (frozen) ──────────────────────────────────────
    print("\n--- Reference model (frozen) ---")
    ref_model, _ = load_model(
        checkpoint_path, tokenizer, device, args.text_model
    )
    if text_encoder_ckpt is not None:
        ref_encoder = _load_contrastive_text_encoder(text_encoder_ckpt, device)
        ref_model.set_text_encoder(ref_encoder)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # ── Optimiser ─────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"Optimising {sum(p.numel() for p in trainable_params):,} trainable params\n")

    # ── wandb init ────────────────────────────────────────────────────
    wandb_config = {
        "algorithm": args.algorithm,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "cfg_scale": args.cfg_scale,
        "temperature": args.temperature,
        "gen_steps": args.gen_steps,
        "eval_gen_steps": args.eval_gen_steps,
        "beta": args.beta,
        "w_tanimoto": args.w_tanimoto,
        "w_bleu2": args.w_bleu2,
        "w_bleu4": args.w_bleu4,
        "validity_bonus": args.validity_bonus,
        "text_model": args.text_model,
        "max_adv": args.max_adv,
        "clip_eps": args.clip_eps,
        "ppo_epochs": args.ppo_epochs,
        "group_size": args.group_size,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "total_params": sum(p.numel() for p in model.parameters()),
        "model_config": model_config,
    }

    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=wandb_config,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )
        print(f"W&B run: {run.url}")
    else:
        wandb.init(mode="disabled")
        print("W&B disabled")

    # ── CSV log header ────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "mean_reward", "mean_tanimoto", "mean_bleu2", "mean_bleu4",
            "validity", "loss", "mean_kl",
        ])

    print(
        f"\nStarting RL [{args.algorithm.upper()}] for prompt adherence "
        f"[{args.run_name}]\n"
        f"  steps={args.num_steps}  batch={args.batch_size}  lr={args.lr}\n"
        f"  cfg_scale={args.cfg_scale}  temperature={args.temperature}  "
        f"gen_steps={args.gen_steps}\n"
        f"  w_tanimoto={args.w_tanimoto}  w_bleu2={args.w_bleu2}  "
        f"w_bleu4={args.w_bleu4}  validity_bonus={args.validity_bonus}\n"
        f"  beta={args.beta}\n"
    )

    best_reward = -float("inf")

    # ── Training loop ─────────────────────────────────────────────────
    for step in range(1, args.num_steps + 1):
        algo.pre_step(step)

        # Sample prompts
        if args.algorithm == "grpo":
            num_prompts = args.batch_size // args.group_size
            batch_df = df.sample(n=num_prompts, replace=False)
            prompts, ref_selfies_list = expand_for_grpo(
                batch_df["prompt"].tolist(),
                batch_df["response"].tolist(),
                args.group_size,
            )
        else:
            batch_df = df.sample(n=args.batch_size, replace=False)
            prompts = batch_df["prompt"].tolist()
            ref_selfies_list = batch_df["response"].tolist()

        # Dispatch to the right step function
        if algo.needs_multiple_generations:
            reward_metrics, algo_metrics = run_ppo_step(
                algo, model, ref_model, optimizer, trainable_params,
                tokenizer, hf_tokenizer, device, model_config,
                prompts, ref_selfies_list, args,
            )
        else:
            reward_metrics, algo_metrics = run_rl_step(
                algo, model, ref_model, optimizer, trainable_params,
                tokenizer, hf_tokenizer, device, model_config,
                prompts, ref_selfies_list, args,
            )

        algo.post_step(step)

        # ── wandb: per-step training metrics ──────────────────────────
        mean_kl = algo_metrics.get("mean_kl", 0.0)
        wandb.log({
            "train/loss": reward_metrics["loss"],
            "train/mean_reward": reward_metrics["mean_reward"],
            "train/std_reward": reward_metrics.get("std_reward", 0.0),
            "train/mean_tanimoto": reward_metrics["mean_tanimoto"],
            "train/mean_bleu2": reward_metrics["mean_bleu2"],
            "train/mean_bleu4": reward_metrics["mean_bleu4"],
            "train/validity": reward_metrics["validity"],
            "train/mean_kl": mean_kl,
            "train/grad_norm": reward_metrics.get("grad_norm", 0.0),
            "train/mean_policy_log_prob": reward_metrics.get("mean_policy_log_prob", 0.0),
            # Algorithm-specific metrics (advantage stats, clip fraction, etc.)
            **{f"train/{k}": v for k, v in algo_metrics.items()},
        }, step=step)

        # Console logging
        print(
            f"Step {step:>4}  |  "
            f"reward={reward_metrics['mean_reward']:.4f}  "
            f"tan={reward_metrics['mean_tanimoto']:.4f}  "
            f"bleu2={reward_metrics['mean_bleu2']:.4f}  "
            f"bleu4={reward_metrics['mean_bleu4']:.4f}  "
            f"valid={reward_metrics['validity']*100:.1f}%  "
            f"KL={mean_kl:.4f}  "
            f"loss={reward_metrics['loss']:.4f}"
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step,
                f"{reward_metrics['mean_reward']:.4f}",
                f"{reward_metrics['mean_tanimoto']:.4f}",
                f"{reward_metrics['mean_bleu2']:.4f}",
                f"{reward_metrics['mean_bleu4']:.4f}",
                f"{reward_metrics['validity']:.4f}",
                f"{reward_metrics['loss']:.4f}",
                f"{mean_kl:.4f}",
            ])

        # ── Periodic evaluation ───────────────────────────────────────
        if step % args.eval_every == 0:
            print(f"\n{'─' * 60}")
            print(f"  Full evaluation @ step {step}")
            m = full_eval(
                model, hf_tokenizer, tokenizer, device, model_config,
                df, batch_size=args.batch_size,
                num_steps=args.eval_gen_steps, temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                w_tanimoto=args.w_tanimoto, w_bleu2=args.w_bleu2,
                w_bleu4=args.w_bleu4, num_eval=args.num_eval,
            )
            print(
                f"  reward={m['mean_reward']:.4f}  "
                f"tanimoto={m['mean_tanimoto']:.4f}  "
                f"bleu2={m['mean_bleu2']:.4f}  "
                f"bleu4={m['mean_bleu4']:.4f}  "
                f"validity={m['validity'] * 100:.1f}%"
            )
            print(f"{'─' * 60}\n")

            # wandb: eval metrics
            wandb.log({
                "eval/mean_reward": m["mean_reward"],
                "eval/mean_tanimoto": m["mean_tanimoto"],
                "eval/mean_bleu2": m["mean_bleu2"],
                "eval/mean_bleu4": m["mean_bleu4"],
                "eval/validity": m["validity"],
            }, step=step)

            # Track best reward
            if m["mean_reward"] > best_reward:
                best_reward = m["mean_reward"]
                wandb.run.summary["best_eval_reward"] = best_reward
                wandb.run.summary["best_eval_step"] = step

        # ── Checkpoint ────────────────────────────────────────────────
        if step % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"rl_prompt_{args.run_name}_step_{step}.pt"
            torch.save({
                "step": step,
                "algorithm": args.algorithm,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_config,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path.name}")

            # wandb: save model artifact
            if not args.no_wandb:
                artifact = wandb.Artifact(
                    name=f"rl-{args.run_name}-step{step}",
                    type="model",
                    metadata={
                        "step": step,
                        "algorithm": args.algorithm,
                        "mean_reward": reward_metrics["mean_reward"],
                    },
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

    # ══════════════════════════════════════════════════════════════════
    # FINAL CHEBI-20 EVALUATION
    # ══════════════════════════════════════════════════════════════════
    if args.chebi20_eval:
        print(f"\n{'═' * 68}")
        print("  FINAL ChEBI-20 EVALUATION")
        print(f"{'═' * 68}")

        chebi20_metrics, chebi20_results_df = run_chebi20_eval(
            model, tokenizer, hf_tokenizer, device, model_config,
            batch_size=args.batch_size,
            cfg_scale=args.cfg_scale,
            num_steps=args.eval_gen_steps,
            temperature=args.temperature,
            max_rows=args.chebi20_max_rows,
            cache_dir=str(output_dir),
            compute_fcd=True,
        )

        num_valid = int(chebi20_results_df["valid"].sum())
        print_chebi20_metrics(
            chebi20_metrics,
            num_samples=len(chebi20_results_df),
            num_valid=num_valid,
        )

        # Save per-sample results
        chebi20_csv = output_dir / f"chebi20_eval_{args.run_name}.csv"
        chebi20_results_df.to_csv(chebi20_csv, index=False)
        print(f"Per-sample results → {chebi20_csv}")

        # wandb: log ChEBI-20 metrics
        chebi20_wandb = {}
        for k, v in chebi20_metrics.items():
            if not (isinstance(v, float) and np.isnan(v)):
                chebi20_wandb[f"chebi20/{k}"] = v
        wandb.log(chebi20_wandb, step=args.num_steps)

        # wandb: summary table
        wandb.run.summary.update({
            f"chebi20/{k}": v for k, v in chebi20_metrics.items()
            if not (isinstance(v, float) and np.isnan(v))
        })

        # wandb: upload per-sample results as table
        if not args.no_wandb:
            sample_cols = [
                "reference_smiles", "generated_smiles", "valid",
                "smiles_exact_match", "smiles_bleu", "morgan_fts",
                "maccs_fts", "rdk_fts", "token_retention_rate",
            ]
            table = wandb.Table(
                dataframe=chebi20_results_df[sample_cols].head(500)
            )
            wandb.log({"chebi20/samples": table}, step=args.num_steps)

            # Upload full CSV as artifact
            artifact = wandb.Artifact(
                name=f"chebi20-eval-{args.run_name}",
                type="evaluation",
                metadata=chebi20_metrics,
            )
            artifact.add_file(str(chebi20_csv))
            wandb.log_artifact(artifact)

    # ── Final save ────────────────────────────────────────────────────
    final_ckpt = checkpoint_dir / f"rl_prompt_{args.run_name}_final.pt"
    torch.save({
        "step": args.num_steps,
        "algorithm": args.algorithm,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config,
    }, final_ckpt)
    print(f"\nFinal checkpoint → {final_ckpt.name}")

    if not args.no_wandb:
        artifact = wandb.Artifact(
            name=f"rl-{args.run_name}-final",
            type="model",
            metadata={"step": args.num_steps, "algorithm": args.algorithm},
        )
        artifact.add_file(str(final_ckpt))
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f"Done. Log: {csv_path}")


if __name__ == "__main__":
    main()
