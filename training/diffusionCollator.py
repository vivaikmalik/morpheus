import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class ConditionalDiffusionCollator:
    def __init__(self, tokenizer, hf_tokenizer):
        self.tokenizer = tokenizer
        self.hf_tokenizer = hf_tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.pad_id  = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])  # (B, L)
        prompts   = [f["prompt"] for f in features]

        labels = input_ids.clone()

        t       = torch.rand(input_ids.shape[0])           # (B,)
        alpha_t = torch.cos(t * torch.pi / 2) ** 2        # (B,)

        noise_probs = torch.rand_like(input_ids.float())
        mask_map    = noise_probs > alpha_t.unsqueeze(-1)  # True where masked

        input_ids[mask_map] = self.mask_id
        labels[~mask_map]   = -100                         # only predict masked tokens
        text_encoding = self.hf_tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        )

        return {
            "input_ids":          input_ids,                          # (B, L)
            "labels":             labels,                             # (B, L)
            "timesteps":          t.unsqueeze(-1),                    # (B, 1)
            "text_input_ids":     text_encoding["input_ids"],         # (B, T)
            "text_attention_mask": text_encoding["attention_mask"],   # (B, T)
        }
