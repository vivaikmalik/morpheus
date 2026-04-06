import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import os

def precompute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device).eval()
    
    os.makedirs("data", exist_ok=True)
    
    for split in ["train", "val"]:
        df = pd.read_csv(f"data/chebi20_{split}.csv")
        embeddings = {}
        
        print(f"Pre-computing {split} embeddings...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            inputs = tokenizer(row['description'], return_tensors="pt", 
                               padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                # Extract the 768-dim last hidden state
                state = model(**inputs).last_hidden_state.cpu() 
            
            # Store by index to match TextConditionedDataset access
            embeddings[i] = {
                "state": state.squeeze(0), # (seq_len, 768)
                "mask": (inputs["attention_mask"] == 0).cpu().squeeze(0) # (seq_len)
            }
            
        torch.save(embeddings, f"data/chebi20_{split}_embeddings.pt")

if __name__ == "__main__":
    precompute()