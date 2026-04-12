import torch
from torch.utils.data import Dataset


class TextConditionedDataset(Dataset):
    """
    Dataset for text → molecule fine-tuning with SMILES.

    Two modes:
      1. Pre-computed SciBERT: loads hidden states from .pt file
      2. Raw text: returns description string for collator to tokenize
    """
    def __init__(self, df, tokenizer, max_length=256, scibert_states=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = df["smiles"].tolist()
        self.prompts = df["description"].tolist()
        self.scibert_states = scibert_states

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        ids = self.tokenizer.encode(smiles)
        ids = ids + [self.tokenizer.eos_token_id]

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
            "input_ids": input_ids,
            "prompt": self.prompts[idx],
        }

        if self.scibert_states is not None:
            entry = self.scibert_states[idx]
            result["scibert_states"] = entry["states"]
            result["scibert_mask"] = entry["mask"]

        return result