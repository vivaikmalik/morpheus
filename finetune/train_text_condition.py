import sys
import os
import torch
import random
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModel
import wandb
import glob as _glob
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from finetune.conditionalSELFIESDataset import ConditionalSELFIESDataset
from finetune.diffusionCollatorPrompt import ConditionalDiffusionCollator
from model.molecularDiffusionModel import MolecularDiffusionModel

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "vocab_size"  : 110,
    "hidden_size" : 256,
    "num_heads"   : 8,
    "ffn_dim"     : 512,
    "num_layers"  : 8,
    "max_length"  : 74,
    "dropout"     : 0.1,
    "text_model"  : "allenai/scibert_scivocab_uncased",
    "uncond_prob" : 0.1,

    "batch_size"           : 256,
    "learning_rate"        : 2e-4, 
    "weight_decay"         : 0.01,
    "max_grad_norm"        : 1.0,
    "num_epochs"           : 10,
    "warmup_ratio"         : 0.06,  

    "val_fraction" : 0.05,
    "num_workers"  : 0,
    "seed"         : 42,

    "log_every"  : 50,
    "val_every"  : 500,

    "gen_steps"       : 32,    
    "gen_temperature" : 1.2,   
    "gen_num_mols"    : 4,     

    "data_path"       : "train.csv",
    "val_data_path"   : None,          
    "early_stop_patience" : 5,         

    "project_root"  : str(Path(__file__).parent.parent),
    "checkpoint_dir": str(Path(__file__).parent.parent / "checkpoints"),
    "wandb_project"  : "morpheus-diffusion-finetune",
    "wandb_run_name" : None,   # set from notebook, e.g. "contrastive-enc-lr2e4"
    "contrastive_ckpt_path" : None,   # local path to contrastive_text_selfies.pt
    "contrastive_artifact"  : None,   # W&B artifact

    "freeze_text_encoder"   : True,
}

# CONTRASTIVE TEXT ENCODER LOADER
def _load_contrastive_text_encoder(config, device):
    """Extract the fine-tuned text_model from a ContrastiveAligner checkpoint."""
    ckpt_path = config.get("contrastive_ckpt_path")
    artifact  = config.get("contrastive_artifact")

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading contrastive text encoder from: {ckpt_path}")
    elif artifact:
        print(f"Downloading contrastive artifact: {artifact}")
        _run = wandb.init(project="morpheus-contrastive-load", job_type="load", resume="allow")
        _art = _run.use_artifact(artifact, type="model")
        _dir = _art.download(root=os.path.join(config["project_root"], "wandb_contrastive"))
        _pts = _glob.glob(os.path.join(_dir, "**", "*.pt"), recursive=True)
        assert _pts, f"No .pt file found in artifact at {_dir}"
        ckpt_path = sorted(_pts)[0]
        _run.finish()
        print(f"Artifact downloaded → {ckpt_path}")
    else:
        raise ValueError(
            "Set CONFIG['contrastive_ckpt_path'] or CONFIG['contrastive_artifact']."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    text_state = {
        k[len("text_model."):]: v
        for k, v in ckpt["model_state"].items()
        if k.startswith("text_model.")
    }
    assert text_state, "No 'text_model.*' keys in contrastive checkpoint."

    encoder = AutoModel.from_pretrained(config["text_model"])
    encoder.load_state_dict(text_state, strict=True)
    print(f"Contrastive text encoder loaded (epoch={ckpt.get('epoch','?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f})")
    return encoder.to(device)


# LOSS FUNCTION
def diffusion_loss(logits, labels, timesteps, pad_token_id=0):
    B, seq_len, vocab_size = logits.shape
    device = logits.device

    vocab_weights = torch.ones(vocab_size, device=device)
    vocab_weights[pad_token_id] = 0.1

    raw_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        weight=vocab_weights,
        ignore_index=-100,
        reduction='none'
    )

    weights = (1.0 - timesteps)
    weights = weights.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1)
    valid = (labels.view(-1) != -100).float()

    loss = (raw_loss * weights * valid).sum() / (weights * valid).sum().clamp(min=1e-8)
    return loss

# VALIDATION
def run_validation(model, val_loader, device):
    model.eval()
    total_loss  = 0.0
    total_steps = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            
            # Text inputs
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            # Convert HF attention mask (1=real, 0=pad)
            text_padding_mask = (text_attention_mask == 0)

            # Get text embeddings
            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)

            logits = model(input_ids, timesteps, text_embeds, text_padding_mask)
            loss   = diffusion_loss(logits, labels, timesteps)

            total_loss  += loss.item()
            total_steps += 1

    model.train()
    return total_loss / total_steps

# GENERATION (CFG)
def generate_samples(model, tokenizer, hf_tokenizer, device, step, cfg_scale=3.0):
    model.eval()

    max_length  = CONFIG["max_length"]
    num_steps   = CONFIG["gen_steps"]
    temperature = CONFIG["gen_temperature"]

    prompts = [
        "A highly soluble molecule with a benzene ring.",
        "A small fragment molecule.",
        "A molecule containing fluorine.",
        "A complex organic compound with multiple rings."
    ]
    num_mols = len(prompts)
    
    # Prepare Text condition
    text_inputs = hf_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_padding_mask = (text_inputs["attention_mask"] == 0)
    
    # Prepare Diffusion Tensors
    input_ids = torch.full((num_mols, max_length), tokenizer.mask_token_id, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    with torch.no_grad():
        # Compute text embeddings ONCE
        text_embeds = model.get_text_embeddings(text_inputs["input_ids"], text_inputs["attention_mask"])
        # Null embeddings for unconditional pass
        null_embeds = model.null_token.expand(num_mols, text_embeds.size(1), -1)

        for step_idx, t_val in enumerate(t_vals):
            step_t = t_val.repeat(num_mols).unsqueeze(-1)
            
            cond_logits = model(input_ids, step_t, text_embeds, text_padding_mask)
            uncond_logits = model(input_ids, step_t, null_embeds, text_padding_mask)
            
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            if step_idx < num_steps - 1:
                logits[:, :, tokenizer.mask_token_id] = float('-inf')

            scaled_logits = logits / temperature
            probs         = torch.softmax(scaled_logits, dim=-1)
            sampled       = torch.distributions.Categorical(probs=probs).sample()
            confidence    = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            alpha_t     = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < num_steps - 1:
                _, mask_indices = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    results = []
    for i in range(num_mols):
        ids = input_ids[i].cpu().tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]

        selfies_str = tokenizer.decode(ids)
        row = {"idx": i, "prompt": prompts[i], "selfies": selfies_str, "valid": False, "smiles": ""}
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                row.update({"valid": True, "smiles": Chem.MolToSmiles(mol)})
            else:
                row["smiles"] = "Invalid Structure"
        except:
            row["smiles"] = "Decode Error"
        results.append(row)

    print(f"\n{'='*60}")
    print(f"  CFG GENERATED MOLECULES — Step {step}")
    for r in results:
        status = "Valid" if r["valid"] else "Invalid"
        print(f"  [{r['idx']}] {status} Prompt: {r['prompt']}\n      SMILES: {r['smiles']}")
    print(f"{'='*60}\n")

    model.train()

# TRAINING LOOP
def train():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    project_root   = Path(CONFIG["project_root"])
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    tokenizer = ChemicalTokenizer(project_root / "chemical_tokenizer.json")
    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])

    print(f"Loading dataset from {CONFIG['data_path']}...")
    df = pd.read_csv(CONFIG["data_path"])
    
    if "response" not in df.columns:
        raise ValueError("CSV must have a 'response' column with SELFIES strings.")
    if "prompt" not in df.columns:
        print("WARNING: 'prompt' column not found in CSV. Falling back to default.")
        df["prompt"] = "A chemical molecule"

    train_dataset = ConditionalSELFIESDataset(df, tokenizer)

    if CONFIG["val_data_path"] and os.path.exists(CONFIG["val_data_path"]):
        val_df = pd.read_csv(CONFIG["val_data_path"])
        if "response" not in val_df.columns:
            raise ValueError("val CSV must also have a 'response' column.")
        if "prompt" not in val_df.columns:
            val_df["prompt"] = "A chemical molecule"
        val_dataset = ConditionalSELFIESDataset(val_df, tokenizer)
        train_size  = len(train_dataset)
        print(f"Using separate val CSV: {CONFIG['val_data_path']} ({len(val_dataset):,} rows)")
    else:
        val_size      = int(len(train_dataset) * CONFIG["val_fraction"])
        train_size    = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print(f"Random split: {train_size:,} train / {val_size:,} val")

    collator = ConditionalDiffusionCollator(tokenizer, hf_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collator)

    model = MolecularDiffusionModel(
        vocab_size      = CONFIG["vocab_size"],
        hidden_size     = CONFIG["hidden_size"],
        num_heads       = CONFIG["num_heads"],
        ffn_dim         = CONFIG["ffn_dim"],
        num_layers      = CONFIG["num_layers"],
        max_length      = CONFIG["max_length"],
        pad_token_id    = tokenizer.pad_token_id,
        text_model_name = CONFIG["text_model"],
        uncond_prob     = CONFIG["uncond_prob"],
        dropout         = CONFIG["dropout"],
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total | {trainable:,} trainable (Text encoder is frozen)")

    # LOAD BEST PRE-TRAINED MODEL
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading pre-trained weights from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        
        print(f"Missing keys (expected - new layers): {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
    else:
        print("WARNING: best_model.pt not found. Training from scratch.")

    if CONFIG.get("contrastive_ckpt_path") or CONFIG.get("contrastive_artifact"):
        model.text_encoder = _load_contrastive_text_encoder(CONFIG, device)

    frozen = CONFIG.get("freeze_text_encoder", True)
    for p in model.text_encoder.parameters():
        p.requires_grad = not frozen
    print(f"Text encoder {'frozen' if frozen else 'unfrozen (trainable)'}.")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    total_steps   = (train_size // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"LR schedule: {total_steps} total steps, {warmup_steps} warmup ({CONFIG['warmup_ratio']*100:.0f}%)")

    run = wandb.init(
        project  = CONFIG["wandb_project"],
        name     = CONFIG.get("wandb_run_name"),
        config   = CONFIG,
        reinit   = True,
        settings = wandb.Settings(start_method="thread"),
    )
    print(f"W&B run started: {run.url}")

    global_step      = 0
    best_val_loss    = float('inf')
    patience_counter = 0
    stopped_early    = False

    model.train()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            timesteps = batch["timesteps"].to(device)
            
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            text_padding_mask = (text_attention_mask == 0)

            text_embeds = model.get_text_embeddings(text_input_ids, text_attention_mask)

            logits = model(input_ids, timesteps, text_embeds, text_padding_mask)
            loss   = diffusion_loss(logits, labels, timesteps)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % CONFIG["log_every"] == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
                wandb.log({"train/loss": loss.item(), "train/lr": current_lr}, step=global_step)

            if global_step % CONFIG["val_every"] == 0:
                val_loss = run_validation(model, val_loader, device)
                print(f"  → Val loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss}, step=global_step)

                generate_samples(model, tokenizer, hf_tokenizer, device, global_step)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    ckpt_path = checkpoint_dir / "best_finetuned_model.pt"
                    torch.save({
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": CONFIG,
                    }, ckpt_path)
                    print("Saved best finetuned model")

                    artifact = wandb.Artifact(
                        name     = f"finetuned-model-valloss{val_loss:.4f}",
                        type     = "model",
                        metadata = {"val_loss": val_loss, "step": global_step},
                    )
                    artifact.add_file(str(ckpt_path))
                    wandb.log_artifact(artifact)
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{CONFIG['early_stop_patience']})")
                    if patience_counter >= CONFIG["early_stop_patience"]:
                        print(f"Early stopping: val loss has not improved for "
                              f"{CONFIG['early_stop_patience']} consecutive val checks.")
                        stopped_early = True
                        break

        if stopped_early:
            break

    print("Fine-tuning complete." + (" (early stop)" if stopped_early else ""))
    wandb.finish()

if __name__ == "__main__":
    train()