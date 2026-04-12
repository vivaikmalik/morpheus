"""
scripts/precompute_scibert.py
------------------------------
Pre-computes SciBERT hidden states for all CheBI-20 descriptions.
Saves them as a dict keyed by DataFrame index → {states, mask}.

This eliminates SciBERT from the GPU during training (~440MB VRAM saved,
~2x faster per step). TGM-DLM does exactly this.

Usage:
    python scripts/precompute_scibert.py

Output:
    data/chebi20_train_scibert.pt
    data/chebi20_val_scibert.pt
    data/chebi20_test_scibert.pt
"""

import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MAX_TEXT_LEN = 256
BATCH_SIZE = 64
MODEL_NAME = "/content/morpheus/scibert_local"


def precompute_split(df, tokenizer, model, device, save_path):
    """Pre-compute SciBERT hidden states for one split."""
    model.eval()
    all_states = {}

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc=f"  {save_path.stem}"):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        descriptions = batch_df["description"].tolist()

        enc = tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            hidden_states = outputs.last_hidden_state.cpu()  # (B, text_len, 768)
            attention_mask = enc["attention_mask"].cpu()       # (B, text_len)

        for i, row_idx in enumerate(range(start_idx, end_idx)):
            # Store per-sample: hidden states and padding mask
            seq_len = attention_mask[i].sum().item()  # actual text length
            all_states[row_idx] = {
                "states": hidden_states[i],          # (text_len, 768)
                "mask": (attention_mask[i] == 0),     # True where PAD
            }

    torch.save(all_states, save_path)
    print(f"  Saved: {save_path} ({len(all_states)} samples)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = ROOT / "data"

    print(f"\nLoading SciBERT ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    for split in ["train", "val", "test"]:
        csv_path = data_dir / f"chebi20_{split}.csv"
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        save_path = data_dir / f"chebi20_{split}_scibert.pt"
        print(f"\n  {split}: {len(df):,} samples")
        precompute_split(df, tokenizer, model, device, save_path)

    print("\nDone! SciBERT embeddings saved.")
    print("You can now train without loading SciBERT on the GPU.")


if __name__ == "__main__":
    main()