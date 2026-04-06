import torch
from torch.utils.data import Dataset


class TextConditionedDataset(Dataset):
    """
    Dataset for text → molecule fine-tuning.

    Two modes:
      1. Pre-computed SciBERT: loads hidden states from .pt file (fast, no SciBERT on GPU)
      2. Raw text: returns description string for collator to tokenize (legacy)

    Parameters
    ----------
    df : DataFrame with 'selfies' and 'description' columns
    tokenizer : ChemicalTokenizer (SELFIES)
    max_length : int, max SELFIES sequence length
    scibert_states : dict or None, pre-computed SciBERT hidden states
                     keyed by DataFrame row index → {states, mask}
    """
    def __init__(self, df, tokenizer, max_length=74, scibert_states=None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.selfies    = df["selfies"].tolist()
        self.prompts    = df["description"].tolist()
        self.scibert_states = scibert_states

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        selfies_str = self.selfies[idx]

        # Tokenize SELFIES → token IDs + EOS
        ids = self.tokenizer.encode(selfies_str)
        ids = ids + [self.tokenizer.eos_token_id]

        # Pad to max_length
        input_ids = torch.tensor(ids, dtype=torch.long)
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            padding = torch.full(
                (self.max_length - seq_len,),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[:self.max_length]

        result = {
            "input_ids": input_ids,        # (max_length,)
            "prompt":    self.prompts[idx], # raw text string (for legacy/eval)
        }

        # If pre-computed SciBERT states are available, include them
        if self.scibert_states is not None:
            entry = self.scibert_states[idx]
            result["scibert_states"] = entry["states"]  # (text_len, 768)
            result["scibert_mask"]   = entry["mask"]    # (text_len,) True=PAD

        return result