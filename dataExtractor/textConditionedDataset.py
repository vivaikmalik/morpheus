import torch
from torch.utils.data import Dataset


class TextConditionedDataset(Dataset):
    """
    Dataset for text → molecule fine-tuning.

    Expects a DataFrame with columns:
        - 'selfies': the SELFIES string of the molecule
        - 'description': the natural language description (text prompt)

    Returns tokenized SELFIES (padded to max_length) and the raw text string.
    Text tokenization is deferred to the collator for efficient batching.
    """
    def __init__(self, df, tokenizer, max_length=74):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.selfies    = df["selfies"].tolist()
        self.prompts    = df["description"].tolist()

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
            # Truncate if somehow longer (shouldn't happen with filtered data)
            input_ids = input_ids[:self.max_length]

        return {
            "input_ids": input_ids,        # (max_length,) SELFIES token IDs
            "prompt":    self.prompts[idx], # raw text string
        }