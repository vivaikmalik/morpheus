"""
Standalone ChEBI-20 evaluation for a diffusion model checkpoint.

Runs the same full evaluation as train_chebi20.py's evaluate_test_set,
using the RL module's infrastructure for model loading and generation.

Usage
-----
    python rl/eval_checkpoint.py \
        --wandb_artifact entity/project/artifact:version \
        --cfg_scale 3.0 --num_steps 32 --temperature 0.9

    # With contrastive text encoder:
    python rl/eval_checkpoint.py \
        --wandb_artifact entity/project/artifact:version \
        --wandb_artifact_text_encoder entity/project/text_encoder:version

    # From a local checkpoint:
    python rl/eval_checkpoint.py \
        --checkpoint checkpoints/best_finetuned_model.pt
"""

import sys
import argparse
import torch
import numpy as np
import wandb
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from transformers import AutoTokenizer
from rl.utils import load_model
from rl.evaluation import run_chebi20_eval, print_chebi20_metrics


def main():
    parser = argparse.ArgumentParser(description="ChEBI-20 evaluation for a checkpoint")

    # Model source
    parser.add_argument("--checkpoint", default="checkpoints/best_finetuned_model.pt")
    parser.add_argument("--text_model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                        help="Download diffusion checkpoint from wandb artifact")
    parser.add_argument("--wandb_artifact_text_encoder", type=str, default=None,
                        help="Download contrastive text encoder from wandb artifact")

    # Generation
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)

    # Eval options
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit to first N rows (for debugging)")
    parser.add_argument("--no_fcd", action="store_true", help="Skip FCD computation")
    parser.add_argument("--output_dir", default="outputs/eval")

    # wandb logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ── Checkpoint: wandb artifact or local ──────────────────────────
    if args.wandb_artifact:
        print(f"Downloading wandb artifact: {args.wandb_artifact}")
        api = wandb.Api()
        artifact = api.artifact(args.wandb_artifact, type="model")
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

    # ── Text encoder: wandb artifact or default ──────────────────────
    text_encoder_ckpt = None
    if args.wandb_artifact_text_encoder:
        print(f"Downloading text encoder artifact: {args.wandb_artifact_text_encoder}")
        api = wandb.Api() if not args.wandb_artifact else api
        te_artifact = api.artifact(args.wandb_artifact_text_encoder, type="model")
        import os as _os
        _te_root = (
            _os.environ.get("SLURM_TMPDIR")
            or _os.environ.get("SCRATCH")
            or str(checkpoint_dir / "_artifacts")
        )
        _te_root = str(Path(_te_root) / "text_encoder")
        te_dir = Path(te_artifact.download(root=_te_root))
        pt_files = list(te_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt file found in text encoder artifact {args.wandb_artifact_text_encoder}"
            )
        text_encoder_ckpt = pt_files[0]
        print(f"  Text encoder checkpoint → {text_encoder_ckpt}")

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Load contrastive text encoder if provided ────────────────────
    contrastive_encoder = None
    if text_encoder_ckpt is not None:
        from finetune.train_chebi20 import _load_contrastive_text_encoder
        contrastive_encoder = _load_contrastive_text_encoder(text_encoder_ckpt, device)

    # ── Tokenizers ───────────────────────────────────────────────────
    tokenizer = ChemicalTokenizer(str(ROOT / "chemical_tokenizer.json"))
    hf_tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # ── Load model ───────────────────────────────────────────────────
    print("\n--- Loading model ---")
    model, model_config = load_model(
        checkpoint_path, tokenizer, device, args.text_model,
        text_encoder=contrastive_encoder,
    )
    model.eval()

    # ── wandb ────────────────────────────────────────────────────────
    if not args.no_wandb and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            job_type="eval",
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        wandb.init(mode="disabled")

    # ── Run evaluation ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  ChEBI-20 EVALUATION")
    print(f"  cfg={args.cfg_scale}  T={args.temperature}  steps={args.num_steps}")
    print(f"{'='*68}")

    metrics, results_df = run_chebi20_eval(
        model, tokenizer, hf_tokenizer, device, model_config,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        temperature=args.temperature,
        max_rows=args.max_rows,
        cache_dir=str(output_dir),
        compute_fcd=not args.no_fcd,
    )

    num_valid = int(results_df["valid"].sum())
    print_chebi20_metrics(metrics, num_samples=len(results_df), num_valid=num_valid)

    # ── Save results ─────────────────────────────────────────────────
    tag = f"cfg{args.cfg_scale}_t{args.temperature}_s{args.num_steps}"
    csv_path = output_dir / f"chebi20_eval_{tag}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nPer-sample results → {csv_path}")

    # ── Log to wandb ─────────────────────────────────────────────────
    chebi20_wandb = {}
    for k, v in metrics.items():
        if not (isinstance(v, float) and np.isnan(v)):
            chebi20_wandb[f"chebi20/{k}"] = v
    wandb.log(chebi20_wandb)
    wandb.run.summary.update(chebi20_wandb)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
