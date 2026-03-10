import torch
from torch.utils.data import Dataset
import selfies as sf

import sys
from pathlib import Path

# Add the project root to Python's search path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer

class ConditionalSELFIESDataset(Dataset):
    def __init__(self, df, tokenizer, property_column="logP", max_length=74):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selfies_list = df["selfies"].tolist()
        self.properties = df[property_column].tolist()

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        # --- 1. Tokenize the SELFIES string ---
        selfies_str = self.selfies_list[idx]
        ids = self.tokenizer.encode(selfies_str)
        ids = ids + [self.tokenizer.eos_token_id]   # append EOS after real tokens
        input_ids = torch.tensor(ids, dtype=torch.long)

        # --- 3. Pad to max_length ---
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            padding = torch.full(
                (self.max_length - seq_len,),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding])

        # --- 4. Return input_ids and the LogP value ---
        return {
            "input_ids": input_ids,                              # shape: (max_length,)
            "cond_value": torch.tensor(float(self.properties[idx]))  # scalar
        }

"""
import pandas as pd

df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
project_root = Path(__file__).parent.parent
tokenizer = ChemicalTokenizer("chemical_tokenizer.json")
dataset = ConditionalSELFIESDataset(df, tokenizer)

# Test a few samples
for i in [0, 1, 100, 999]:
    sample = dataset[i]
    input_ids = sample["input_ids"]
    cond_value = sample["cond_value"]
    
    # Count real tokens (non-padding)
    real_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
    
    print(f"Sample {i}:")
    print(f"  SELFIES   : {df['selfies'].iloc[i][:60]}...")
    print(f"  input_ids : {input_ids[:10].tolist()}... (shape: {input_ids.shape})")
    print(f"  real tokens: {real_tokens} / {len(input_ids)}")
    print(f"  LogP      : {cond_value.item():.3f}")
    print()

# Verify all sequences are exactly max_length
print("Checking all lengths...")
for i in range(len(dataset)):
    assert dataset[i]["input_ids"].shape[0] == 74, f"Wrong length at index {i}"
print("✅ All sequences are length 72")

"""