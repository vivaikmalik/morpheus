import torch


class TextDiffusionCollator:
    """
    Collator for text-conditioned masked diffusion training.

    Combines two jobs:
      1. Masked diffusion noise (revealPad style — PADs can be masked)
      2. SciBERT tokenization of text prompts (batched for efficiency)

    Parameters
    ----------
    mol_tokenizer  : ChemicalTokenizer  (SELFIES tokenizer)
    text_tokenizer : transformers.AutoTokenizer  (SciBERT tokenizer)
    max_text_len   : int  maximum text token length (SciBERT max is 512)
    """
    def __init__(self, mol_tokenizer, text_tokenizer, max_text_len=256):
        self.mol_tokenizer  = mol_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_len   = max_text_len
        self.mask_id        = mol_tokenizer.mask_token_id   # 1
        self.pad_id         = mol_tokenizer.pad_token_id    # 0

    def __call__(self, features):
        # ================================================================= #
        # PART 1: MASKED DIFFUSION (same as revealPad collator)             #
        # ================================================================= #

        input_ids = torch.stack([f["input_ids"] for f in features])   # (B, 74)
        labels    = input_ids.clone()

        # Sample random timestep t for each molecule
        t = torch.rand(input_ids.shape[0])                            # (B,)

        # Cosine schedule: alpha_t close to 1 at t=0, close to 0 at t=1
        alpha_t = torch.cos(t * torch.pi / 2) ** 2                   # (B,)

        # Decide which tokens get masked
        # revealPad: PADs ARE maskable — model must learn to predict them
        noise_probs = torch.rand_like(input_ids.float())              # (B, 74)
        mask_map = noise_probs > alpha_t.unsqueeze(-1)                # (B, 74)

        # Apply mask
        input_ids[mask_map] = self.mask_id

        # Labels: only compute loss on masked positions
        labels[~mask_map] = -100

        # ================================================================= #
        # PART 2: TEXT TOKENIZATION (SciBERT)                               #
        # ================================================================= #

        prompts = [f["prompt"] for f in features]

        text_enc = self.text_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # text_padding_mask: True where PAD (for cross-attention masking)
        # SciBERT attention_mask is 1=real, 0=pad → invert for our convention
        text_padding_mask = (text_enc["attention_mask"] == 0)

        return {
            "input_ids":          input_ids,                      # (B, 74)
            "labels":             labels,                         # (B, 74)
            "timesteps":          t.unsqueeze(-1),                # (B, 1)
            "text_input_ids":     text_enc["input_ids"],          # (B, text_len)
            "text_attention_mask": text_enc["attention_mask"],     # (B, text_len)
            "text_padding_mask":  text_padding_mask,              # (B, text_len)
        }