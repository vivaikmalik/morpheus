"""ChEBI-20 (prompt, SELFIES) dataset with EOS-then-pad encoding to a fixed length."""
import torch
from torch.utils.data import Dataset


class ConditionalSELFIESDataset(Dataset):
    """Tokenizes the SELFIES response, truncates to max_length-1, appends EOS, then pads.

    Returns input_ids and the raw prompt string per sample; text tokenization happens in
    the collator."""

    def __init__(self, df, tokenizer, max_length=74):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        selfies_str = str(row["response"])
        prompt = str(row.get("prompt", "A chemical molecule"))

        token_ids = self.tokenizer.encode(selfies_str)[:(self.max_length - 1)]
        token_ids = token_ids + [self.tokenizer.eos_token_id]
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "prompt": prompt,
        }
