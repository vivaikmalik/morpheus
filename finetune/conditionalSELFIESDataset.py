import torch
from torch.utils.data import Dataset


class ConditionalSELFIESDataset(Dataset):
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

        token_ids = self.tokenizer.encode(selfies_str)
        token_ids = token_ids[:self.max_length]
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "prompt": prompt,
        }
