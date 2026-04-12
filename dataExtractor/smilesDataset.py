import torch
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    """
    Dataset for SMILES-based pretraining.
    Reads 'smiles' column from DataFrame, tokenizes with SmilesTokenizer.
    """
    def __init__(self, df, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles_list = df["smiles"].tolist()

        # Optional: property column for conditional generation
        self.properties = None
        if "logP" in df.columns:
            self.properties = df["logP"].tolist()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
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

        result = {"input_ids": input_ids}

        if self.properties is not None:
            result["cond_value"] = torch.tensor(float(self.properties[idx]))

        return result