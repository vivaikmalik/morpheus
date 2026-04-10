import torch


class TextDiffusionCollator:
    """
    Collator for text-conditioned masked diffusion training.

    Two modes:
      1. Pre-computed SciBERT: pads scibert_states/mask to max text_len in batch
      2. Legacy: tokenizes raw text prompts with SciBERT tokenizer

    Parameters
    ----------
    mol_tokenizer  : ChemicalTokenizer
    text_tokenizer : transformers.AutoTokenizer or None (None if pre-computed)
    max_text_len   : int
    use_precomputed : bool, if True, expects scibert_states in features
    """
    def __init__(self, mol_tokenizer, text_tokenizer=None,
                 max_text_len=74, use_precomputed=False):
        self.mol_tokenizer  = mol_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_len   = max_text_len
        self.use_precomputed = use_precomputed
        self.mask_id        = mol_tokenizer.mask_token_id   # 1
        self.pad_id         = mol_tokenizer.pad_token_id    # 0

    def __call__(self, features):
        # ================================================================= #
        # PART 1: MASKED DIFFUSION (same as before)                         #
        # ================================================================= #

        input_ids = torch.stack([f["input_ids"] for f in features])   # (B, 74)
        labels    = input_ids.clone()

        t = torch.rand(input_ids.shape[0])
        alpha_t = torch.cos(t * torch.pi / 2) ** 2

        noise_probs = torch.rand_like(input_ids.float())
        mask_map = noise_probs > alpha_t.unsqueeze(-1)

        input_ids[mask_map] = self.mask_id
        labels[~mask_map] = -100

        batch = {
            "input_ids": input_ids,
            "labels":    labels,
            "timesteps": t.unsqueeze(-1),
        }

        # ================================================================= #
        # PART 2: TEXT EMBEDDINGS                                            #
        # ================================================================= #

        if self.use_precomputed:
            # Pre-computed SciBERT hidden states — just pad to max len in batch
            states_list = [f["scibert_states"] for f in features]
            masks_list  = [f["scibert_mask"] for f in features]

            # Find max text length in this batch
            max_tlen = max(s.shape[0] for s in states_list)
            hidden_dim = states_list[0].shape[1]  # 768

            B = len(features)
            padded_states = torch.zeros(B, max_tlen, hidden_dim)
            padded_mask   = torch.ones(B, max_tlen, dtype=torch.bool)  # True=PAD

            for i, (s, m) in enumerate(zip(states_list, masks_list)):
                tlen = s.shape[0]
                padded_states[i, :tlen] = s
                padded_mask[i, :tlen]   = m

            batch["scibert_states"]     = padded_states   # (B, max_tlen, 768)
            batch["text_padding_mask"]  = padded_mask     # (B, max_tlen)

        else:
            # Legacy: tokenize text on the fly
            prompts = [f["prompt"] for f in features]
            text_enc = self.text_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            batch["text_input_ids"]     = text_enc["input_ids"]
            batch["text_attention_mask"] = text_enc["attention_mask"]
            batch["text_padding_mask"]  = (text_enc["attention_mask"] == 0)

        return batch