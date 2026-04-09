"""
Shared utilities for RL post-training: model loading, molecule decoding,
and reference log-probability computation.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.molecularDiffusionModel import MolecularDiffusionModel
from tokenizer.chemicalTokenizer import ChemicalTokenizer


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path, tokenizer, device,
               text_model_name="BAAI/bge-large-en-v1.5",
               text_encoder=None):
    """Load a text-conditioned diffusion model from a checkpoint file.

    Args:
        text_encoder: Fallback text encoder (e.g. contrastive scibert).
            ``set_text_encoder`` is called before ``load_state_dict`` to
            register ``encoder_proj``.  If the diffusion checkpoint already
            contains ``encoder_proj`` + ``text_encoder`` weights they are
            loaded from it; otherwise the fallback encoder's weights are
            kept.
    """
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    resolved_text_model = config.get("text_model", text_model_name)
    if resolved_text_model != text_model_name:
        print(f"  Text model from checkpoint: {resolved_text_model} (overrides CLI default)")

    if text_encoder is not None:
        # Build the model with the default BGE architecture so text_proj
        # dimensions (1024 input) match the checkpoint, then swap the
        # encoder afterwards.  set_text_encoder will create encoder_proj
        # to bridge the dimension gap (e.g. scibert 768 → 1024).
        init_text_model = text_model_name  # BGE default
        print(f"  Initialising with {init_text_model} (will swap text encoder after)")
    else:
        init_text_model = resolved_text_model

    model = MolecularDiffusionModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
        pad_token_id=tokenizer.pad_token_id,
        text_model_name=init_text_model,
        dropout=0.0,
    ).to(device)

    if text_encoder is not None:
        model.set_text_encoder(text_encoder)

    missing, _ = model.load_state_dict(checkpoint["model"], strict=False)

    # If the diffusion checkpoint had encoder_proj + text_encoder weights,
    # they're already loaded.  Otherwise the fallback encoder weights are kept.
    if text_encoder is not None:
        has_encoder = not any(k.startswith("text_encoder.") for k in missing)
        has_proj = not any(k.startswith("encoder_proj.") for k in missing)
        if has_encoder and has_proj:
            print(f"  Text encoder + encoder_proj loaded from diffusion checkpoint")
        else:
            print(f"  Using fallback contrastive text encoder"
                  f" (missing from checkpoint: encoder={not has_encoder}, proj={not has_proj})")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")
    return model, config


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
