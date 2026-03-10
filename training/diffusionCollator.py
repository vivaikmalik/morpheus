import torch

import sys
from pathlib import Path

# Add the project root to Python's search path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ConditionalDiffusionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id   # 1
        self.pad_id  = tokenizer.pad_token_id    # 0

    def __call__(self, features):
        # --- 1. Stack all samples into batch tensors ---
        input_ids   = torch.stack([f["input_ids"]  for f in features])  # (B, 72)
        cond_values = torch.stack([f["cond_value"]  for f in features])  # (B,)

        # --- 2. Save a clean copy as the answer key ---
        labels = input_ids.clone()  # (B, 72)

        # --- 3. Sample a random timestep t for each molecule ---
        t = torch.rand(input_ids.shape[0])  # (B,)  values between 0 and 1

        # --- 4. Compute how much of the sequence survives unmasked ---
        alpha_t = torch.cos(t * torch.pi / 2) ** 2  # (B,)  between 0 and 1

        # --- 5. Decide which tokens get masked ---
        noise_probs = torch.rand_like(input_ids.float())  # (B, 72) random per token

        # PAD tokens should never be masked
        pad_mask = (input_ids == self.pad_id)            # (B, 72) True where PAD
        noise_probs.masked_fill_(pad_mask, 0.0)          # PAD positions won't be masked

        # A token gets masked if its random number exceeds alpha_t
        # High alpha_t (t≈0, clean) → most tokens survive
        # Low alpha_t  (t≈1, noisy) → most tokens get masked
        mask_map = noise_probs > alpha_t.unsqueeze(-1)   # (B, 72) True where masked

        # --- 6. Apply the mask ---
        input_ids[mask_map] = self.mask_id               # replace with [MASK] token

        # --- 7. Build the labels ---
        # Only compute loss on positions that were actually masked
        labels[~mask_map] = -100    # not masked → ignore
        labels[pad_mask]  = -100    # PAD → always ignore

        return {
            "input_ids":   input_ids,                    # (B, 72)  noisy input
            "labels":      labels,                       # (B, 72)  answer key
            "timesteps":   t.unsqueeze(-1),              # (B, 1)   noise level
            "cond_values": cond_values.unsqueeze(-1)     # (B, 1)   LogP targets
        }


"""
from torch.utils.data import DataLoader

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from dataExtractor.conditionalSELFIESDataset import ConditionalSELFIESDataset
import pandas as pd

project_root = Path(__file__).parent.parent

tokenizer = ChemicalTokenizer("chemical_tokenizer.json")
df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")
dataset = ConditionalSELFIESDataset(df, tokenizer)

collator = ConditionalDiffusionCollator(tokenizer)
loader   = DataLoader(dataset, batch_size=4, collate_fn=collator)

# Grab one batch
batch = next(iter(loader))

print("Batch keys:", list(batch.keys()))
print(f"\ninput_ids   shape: {batch['input_ids'].shape}")
print(f"labels      shape: {batch['labels'].shape}")
print(f"timesteps   shape: {batch['timesteps'].shape}")
print(f"cond_values shape: {batch['cond_values'].shape}")

print(f"\nTimesteps (t):  {batch['timesteps'].squeeze().tolist()}")
print(f"LogP targets:   {batch['cond_values'].squeeze().tolist()}")

# Inspect molecule 0 in detail
print(f"\n--- Molecule 0 detail ---")
ids    = batch['input_ids'][0]
labs   = batch['labels'][0]
t_val  = batch['timesteps'][0].item()
alpha  = (torch.cos(torch.tensor(t_val) * torch.pi / 2) ** 2).item()

real_tokens   = (ids != tokenizer.pad_token_id).sum().item()
masked_tokens = (ids == tokenizer.mask_token_id).sum().item()
active_labels = (labs != -100).sum().item()

print(f"t = {t_val:.3f}  →  alpha_t = {alpha:.3f}")
print(f"Real tokens   : {real_tokens}")
print(f"Masked tokens : {masked_tokens}  ({masked_tokens/real_tokens*100:.0f}% of real tokens)")
print(f"Active labels : {active_labels}  (should equal masked tokens: {masked_tokens})")
print(f"\ninput_ids : {ids[:15].tolist()}...")
print(f"labels    : {labs[:15].tolist()}...")


"""