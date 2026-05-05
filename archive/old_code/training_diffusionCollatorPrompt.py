# NOTE: Not imported by any training/ script (training/ uses diffusionCollator.py instead).
# Canonical prompt-collator: finetune/diffusionCollatorPrompt.py
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class ConditionalDiffusionCollator:
    def __init__(self, tokenizer, hf_tokenizer):
        self.tokenizer = tokenizer
        self.hf_tokenizer = hf_tokenizer # HuggingFace Text Tokenizer
        self.mask_id = tokenizer.mask_token_id   
        self.pad_id  = tokenizer.pad_token_id    

    def __call__(self, features):
        input_ids   = torch.stack([f["input_ids"]  for f in features])
        
        # Assume your dataset now outputs a "prompt" natural language string
        prompts = [f.get("prompt", "A chemical molecule") for f in features]

        # Tokenize the natural language prompts
        text_inputs = self.hf_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        labels = input_ids.clone()
        t = torch.rand(input_ids.shape[0]) 

        alpha_t = torch.cos(t * torch.pi / 2) ** 2  
        noise_probs = torch.rand_like(input_ids.float())  

        mask_map = noise_probs > alpha_t.unsqueeze(-1)   
        input_ids[mask_map] = self.mask_id               
        labels[~mask_map] = -100    

        return {
            "input_ids":           input_ids,
            "labels":              labels,
            "timesteps":           t.unsqueeze(-1),
            "text_input_ids":      text_inputs["input_ids"],
            "text_attention_mask": text_inputs["attention_mask"],
        }