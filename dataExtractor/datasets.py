"""
dataExtractor/datasets.py
-------------------------
Datasets for SELFIES diffusion pipeline:
  - SELFIESDataset: pretraining on raw SELFIES strings
  - TextConditionedDataset: fine-tuning with descriptions
"""

import torch
from torch.utils.data import Dataset


class SELFIESDataset(Dataset):
    """Pretraining dataset: SELFIES strings only."""

    def __init__(self, df, tokenizer, max_length=74, selfies_column="selfies"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selfies_list = df[selfies_column].tolist()

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.selfies_list[idx])
        ids = ids[:self.max_length - 1]   # leave room for EOS
        ids.append(self.tokenizer.eos_token_id)

        input_ids = torch.tensor(ids, dtype=torch.long)
        if input_ids.shape[0] < self.max_length:
            pad = torch.full((self.max_length - input_ids.shape[0],),
                             self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])

        return {"input_ids": input_ids}


class TextConditionedDataset(Dataset):
    """Fine-tuning dataset: SELFIES + description.

    Two modes:
      1. Pre-computed text embeddings (scibert_states dict)
      2. Raw text (returns prompt string)
    """

    def __init__(self, df, tokenizer, max_length=74,
                 selfies_column="selfies", text_column="description",
                 scibert_states=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selfies = df[selfies_column].tolist()
        self.prompts = df[text_column].tolist()
        self.scibert_states = scibert_states

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.selfies[idx])
        ids = ids[:self.max_length - 1]
        ids.append(self.tokenizer.eos_token_id)

        input_ids = torch.tensor(ids, dtype=torch.long)
        if input_ids.shape[0] < self.max_length:
            pad = torch.full((self.max_length - input_ids.shape[0],),
                             self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])

        result = {"input_ids": input_ids, "prompt": self.prompts[idx]}

        if self.scibert_states is not None:
            entry = self.scibert_states[idx]
            result["scibert_states"] = entry["states"]
            result["scibert_mask"] = entry["mask"]

        return result