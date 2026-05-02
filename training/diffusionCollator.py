"""
training/diffusionCollator.py
------------------------------
Pretraining collator with revealPad strategy.

revealPad: PAD positions are MASKED just like real tokens.
The model must learn that beyond the EOS, positions should be predicted as PAD.
This teaches it where molecules should end.
"""

import torch


class DiffusionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = input_ids.clone()

        t = torch.rand(input_ids.shape[0])
        alpha_t = torch.cos(t * torch.pi / 2) ** 2
        noise_probs = torch.rand_like(input_ids.float())

        # revealPad: all positions can be masked (including PAD)
        mask_map = noise_probs > alpha_t.unsqueeze(-1)
        input_ids[mask_map] = self.mask_id
        labels[~mask_map] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "timesteps": t.unsqueeze(-1),
        }


# Alias for backward compat with old code
ConditionalDiffusionCollator = DiffusionCollator