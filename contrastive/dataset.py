import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def read_pairs_from_csv(csv_path, text_column, selfies_column, max_rows: Optional[int] = None):
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    pairs = []
    for _, row in df.iterrows():
        text = str(row[text_column]) if pd.notna(row[text_column]) else ''
        selfies = str(row[selfies_column]) if pd.notna(row[selfies_column]) else ''
        if text.strip() and selfies.strip():
            pairs.append((text, selfies))
    return pairs

def train_val_split(pairs, val_fraction=0.1, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    split = int(len(pairs) * (1.0 - val_fraction))
    train_pairs = [pairs[i] for i in idx[:split]]
    val_pairs = [pairs[i] for i in idx[split:]]
    return train_pairs, val_pairs
