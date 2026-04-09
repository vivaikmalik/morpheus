"""
Shared utilities for RL post-training: model loading, molecule decoding,
and reference log-probability computation.
"""

import glob as _glob
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem
from transformers import AutoModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.molecularDiffusionModel import MolecularDiffusionModel
from tokenizer.chemicalTokenizer import ChemicalTokenizer


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path, tokenizer, device,
               text_model_name="BAAI/bge-large-en-v1.5"):
    """Load a text-conditioned diffusion model from a checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    resolved_text_model = config.get("text_model", text_model_name)
    if resolved_text_model != text_model_name:
        print(f"  Text model from checkpoint: {resolved_text_model} (overrides CLI default)")

    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=resolved_text_model,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model"], strict=False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")
    return model, config


def load_text_encoder_from_artifact(artifact_path, text_model_name, device):
    """
    Download a contrastive wandb artifact and extract its text encoder weights.

    The contrastive checkpoint is expected to have a ``model_state`` key whose
    entries are prefixed with ``text_model.`` (as saved by ContrastiveAligner).

    Parameters
    ----------
    artifact_path : str
        Full wandb artifact path, e.g.
        ``'entity/project/artifact_name:version'``.
    text_model_name : str
        HuggingFace model name used to instantiate the encoder architecture
        (must match the one used during contrastive training).
    device : torch.device

    Returns
    -------
    encoder : AutoModel  (on *device*, eval mode, frozen)
    """
    import wandb

    print(f"Downloading text-encoder artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    _artifact_root = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("SCRATCH")
        or str(ROOT / "checkpoints" / "_artifacts_text_enc")
    )
    artifact_dir = Path(artifact.download(root=_artifact_root))
    pt_files = sorted(_glob.glob(str(artifact_dir / "**" / "*.pt"), recursive=True))
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt file found in text-encoder artifact {artifact_path}"
        )
    ckpt_path = pt_files[0]
    print(f"  Downloaded → {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    text_state = {
        k[len("text_model."):]: v
        for k, v in ckpt["model_state"].items()
        if k.startswith("text_model.")
    }
    if not text_state:
        raise ValueError(
            f"No 'text_model.*' keys found in contrastive checkpoint {ckpt_path}. "
            "Expected a ContrastiveAligner checkpoint."
        )

    encoder = AutoModel.from_pretrained(text_model_name)
    encoder.load_state_dict(text_state, strict=True)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"  Contrastive text encoder loaded (epoch={epoch}, val_loss={val_loss:.4f})")
    return encoder


# =============================================================================
# MOLECULE DECODING
# =============================================================================

def decode_molecule(ids, tokenizer):
    """
    Strip special tokens and decode to SELFIES / SMILES.

    Returns
    -------
    selfies_str     : str
    selfies_tokens  : list[str]
    canonical_smiles: str | None
    mol             : rdkit.Chem.Mol | None
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    specials = {tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id}
    clean_ids = [i for i in ids if i not in specials]

    selfies_str = tokenizer.decode(clean_ids)
    selfies_tokens = list(sf.split_selfies(selfies_str)) if selfies_str else []

    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return selfies_str, selfies_tokens, Chem.MolToSmiles(mol), mol
    except Exception:
        pass
    return selfies_str, selfies_tokens, None, None


# =============================================================================
# REFERENCE LOG-PROBABILITY (CFG, single t=0 pass, no gradients)
# =============================================================================

def compute_ref_log_probs(ref_model, input_ids, text_embeds, text_padding_mask,
                          tokenizer, device, cfg_scale=3.0):
    """
    Single-step reference log-probability at t=0 under the CFG distribution.

    Used as the anchor for the KL penalty in REINFORCE.  Computing the full
    multi-step reference log-prob would require re-running generation through
    the reference model; the t=0 approximation is standard and sufficient for
    anchoring.

    Parameters
    ----------
    ref_model       : frozen MolecularDiffusionModel
    input_ids       : (B, L) generated token IDs
    text_embeds     : (B, T, D) text embeddings from the *reference* model
    text_padding_mask : (B, T)
    tokenizer       : ChemicalTokenizer
    device          : torch.device
    cfg_scale       : float

    Returns
    -------
    ref_log_probs : (B,) FloatTensor, no gradients
    """
    B, L = input_ids.shape
    t_zero = torch.zeros(B, 1, device=device)

    specials = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.mask_token_id]
    is_real = torch.ones(B, L, dtype=torch.bool, device=device)
    for sid in specials:
        is_real &= (input_ids != sid)

    null_embeds = ref_model.null_token.expand(B, text_embeds.size(1), -1)

    with torch.no_grad():
        cond_logits = ref_model(input_ids, t_zero, text_embeds, text_padding_mask)
        uncond_logits = ref_model(input_ids, t_zero, null_embeds, text_padding_mask)
        cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

        log_p = F.log_softmax(cfg_logits, dim=-1)
        token_lp = torch.gather(log_p, 2, input_ids.unsqueeze(-1)).squeeze(-1)

    return (token_lp * is_real.float()).sum(dim=-1)  # (B,)
