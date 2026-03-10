import selfies as sf
import pandas as pd
import json
from collections import Counter
from pathlib import Path

# --- Step 1: Collect all unique tokens from the dataset ---
df = pd.read_csv("hf://datasets/edmanft/zinc250k/zinc250k_selfies.csv")

all_tokens = []
for selfies_str in df["selfies"]:
    tokens = list(sf.split_selfies(selfies_str))
    all_tokens.extend(tokens)

# Get unique tokens, sorted by frequency (most common first)
token_counts = Counter(all_tokens)
selfies_tokens = [tok for tok, _ in token_counts.most_common()]

print(f"Found {len(selfies_tokens)} unique SELFIES tokens")

# --- Step 2: Build vocabulary ---
# Special tokens always go first at fixed IDs
PAD_TOKEN  = "[PAD]"   # ID 0
MASK_TOKEN = "[MASK]"  # ID 1
EOS_TOKEN  = "[EOS]"   # ID 2

special_tokens = [PAD_TOKEN, MASK_TOKEN, EOS_TOKEN]

# Full vocabulary = special tokens + all SELFIES tokens
vocab = special_tokens + selfies_tokens

# --- Step 3: Build lookup dictionaries ---
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print(f"Total vocabulary size: {len(vocab)}")
print(f"\nSpecial token IDs:")
print(f"  PAD  = {token_to_id[PAD_TOKEN]}")
print(f"  MASK = {token_to_id[MASK_TOKEN]}")
print(f"  EOS  = {token_to_id[EOS_TOKEN]}")
print(f"\nFirst 5 SELFIES tokens:")
for tok in selfies_tokens[:5]:
    print(f"  {tok} = {token_to_id[tok]}")
print(f"\nChemical tokens start at ID: {token_to_id[selfies_tokens[0]]}")

# --- Step 4: Save the tokenizer ---
tokenizer_data = {
    "token_to_id": token_to_id,
    "id_to_token": {str(k): v for k, v in id_to_token.items()},  # JSON requires string keys
    "pad_token": PAD_TOKEN,
    "mask_token": MASK_TOKEN,
    "eos_token": EOS_TOKEN,
    "pad_token_id": token_to_id[PAD_TOKEN],
    "mask_token_id": token_to_id[MASK_TOKEN],
    "eos_token_id": token_to_id[EOS_TOKEN],
    "vocab_size": len(vocab),
    "chemical_start_id": token_to_id[selfies_tokens[0]]
}

save_path = Path("chemical_tokenizer.json")
with open(save_path, "w") as f:
    json.dump(tokenizer_data, f, indent=2)

print(f"\n✅ Tokenizer saved to {save_path}")