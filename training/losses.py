"""
training/losses.py
------------------
Loss functions for masked diffusion training.

Key insight: cross-entropy with vocabulary-weighted classes lets us upweight
the EOS token (5×) and downweight PAD (0.05×). This fixes the "molecule too
short" failure mode where the model learns to predict PAD everywhere because
PAD is the most common token in the dataset.
"""

import torch
import torch.nn.functional as F


def diffusion_loss(logits, labels, tokenizer,
                   eos_weight=5.0, pad_weight=0.05,
                   timesteps=None, use_timestep_weighting=False):
    """
    Cross-entropy loss with EOS upweight + PAD downweight.

    Args:
        logits: (B, seq_len, vocab_size)
        labels: (B, seq_len) with -100 at positions to ignore
        tokenizer: chemical tokenizer with eos_token_id, pad_token_id
        eos_weight: multiplier for EOS class (default 5.0)
        pad_weight: multiplier for PAD class (default 0.05)
        timesteps: (B, 1) optional, for timestep-based weighting
        use_timestep_weighting: if True, weight loss by (1-t) so low-t (clean)
                                positions matter more
    """
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[tokenizer.pad_token_id] = pad_weight
    vocab_weights[tokenizer.eos_token_id] = eos_weight

    raw_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction='none'
    )

    valid = (labels.view(-1) != -100).float()

    if use_timestep_weighting and timesteps is not None:
        weights = (1.0 - timesteps).unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)
        loss = (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)
    else:
        loss = (raw_loss * valid).sum() / valid.sum().clamp(min=1e-8)

    return loss