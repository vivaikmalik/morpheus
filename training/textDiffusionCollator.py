"""
training/textDiffusionCollator.py
----------------------------------
Fine-tuning collator with text conditioning + revealPad + EOS.
"""

import torch


class TextDiffusionCollator:
    def __init__(self, mol_tokenizer, text_tokenizer=None, max_text_len=128,
                 use_precomputed=False):
        self.mol_tokenizer = mol_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_len = max_text_len
        self.use_precomputed = use_precomputed
        self.mask_id = mol_tokenizer.mask_token_id
        self.pad_id = mol_tokenizer.pad_token_id

    def __call__(self, features):
        # ----- Molecule side: revealPad masking -----
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = input_ids.clone()

        t = torch.rand(input_ids.shape[0])
        alpha_t = torch.cos(t * torch.pi / 2) ** 2
        noise_probs = torch.rand_like(input_ids.float())
        mask_map = noise_probs > alpha_t.unsqueeze(-1)
        input_ids[mask_map] = self.mask_id
        labels[~mask_map] = -100

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "timesteps": t.unsqueeze(-1),
        }

        # ----- Text side -----
        if self.use_precomputed:
            states = [f["scibert_states"] for f in features]
            masks = [f["scibert_mask"] for f in features]
            max_tlen = max(s.shape[0] for s in states)
            hidden_dim = states[0].shape[1]

            B = len(features)
            padded_states = torch.zeros(B, max_tlen, hidden_dim)
            padded_mask = torch.ones(B, max_tlen, dtype=torch.bool)
            for i, (s, m) in enumerate(zip(states, masks)):
                padded_states[i, :s.shape[0]] = s
                padded_mask[i, :m.shape[0]] = m

            batch["scibert_states"] = padded_states
            batch["text_padding_mask"] = padded_mask
        else:
            prompts = [f["prompt"] for f in features]
            enc = self.text_tokenizer(
                prompts, padding=True, truncation=True,
                max_length=self.max_text_len, return_tensors="pt")
            batch["text_input_ids"] = enc["input_ids"]
            batch["text_attention_mask"] = enc["attention_mask"]
            batch["text_padding_mask"] = (enc["attention_mask"] == 0)

        return batch