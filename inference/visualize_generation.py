import sys
import torch
import selfies as sf
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

# Add the project root to Python's search path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.chemicalTokenizer import ChemicalTokenizer
from model.molecularDiffusionModel import MolecularDiffusionModel

# =============================================================================
# SETTINGS
# =============================================================================
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
TOKENIZER_PATH  = Path(__file__).parent.parent / "chemical_tokenizer.json"

STEPS = 32
TEMPERATURE = 1.2

# =============================================================================
# HELPER: VISUAL DECODER
# =============================================================================
def get_visual_string(ids_tensor, tokenizer):
    """
    Decodes the token IDs but KEEPS the [MASK], [PAD], and [EOS] tokens visible 
    so we can watch the diffusion process happen in the terminal.
    """
    ids_list = ids_tensor.cpu().tolist()
    # Pull raw strings directly from the dictionary to avoid the stripping logic
    tokens = [tokenizer.id_to_token[i] for i in ids_list]
    return "".join(tokens)

# =============================================================================
# VISUAL INFERENCE LOOP
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Loading model on {device}...\n")

    # 1. Load Tokenizer
    tokenizer = ChemicalTokenizer(TOKENIZER_PATH)

    # 2. Load Model weights and config
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config     = checkpoint["config"]
    max_length = config["max_length"]

    model = MolecularDiffusionModel(
        vocab_size   = tokenizer.vocab_size, # Dynamic vocab size!
        hidden_size  = config["hidden_size"],
        num_heads    = config["num_heads"],
        ffn_dim      = config["ffn_dim"],
        num_layers   = config["num_layers"],
        max_length   = max_length,
        pad_token_id = tokenizer.pad_token_id,
        dropout      = 0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Model loaded. Generating 1 molecule over {STEPS} steps at temp {TEMPERATURE}.\n")
    print("-" * 100)

    # --- INITIALIZE WITH 100% MASKS ---
    input_ids = torch.full(
        (1, max_length), 
        tokenizer.mask_token_id, 
        dtype=torch.long, 
        device=device
    )

    t_vals = torch.linspace(1.0, 0.0, STEPS, device=device)

    with torch.no_grad():
        for step_idx, t_val in enumerate(t_vals):
            # Print current state BEFORE the forward pass
            current_str = get_visual_string(input_ids[0], tokenizer)
            mask_count = (input_ids[0] == tokenizer.mask_token_id).sum().item()
            print(f"Step {step_idx:02d} | Masks: {mask_count:02d} | {current_str}")

            step_t = t_val.unsqueeze(0).unsqueeze(-1)  # (1, 1)
            logits = model(input_ids, step_t)          # (1, max_length, vocab_size)

            # Block MASK at all steps except the initial setup
            if step_idx < STEPS - 1:
                logits[:, :, tokenizer.mask_token_id] = float('-inf')

            # Temperature & Sampling
            scaled_logits = logits / TEMPERATURE
            probs   = torch.softmax(scaled_logits, dim=-1)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            
            # Confidence
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)

            # Schedule: How many tokens should remain masked?
            alpha_t = (torch.cos(t_val * torch.pi / 2) ** 2).item()
            num_to_mask = int((1.0 - alpha_t) * max_length)

            if num_to_mask > 0 and step_idx < STEPS - 1:
                _, mask_indices = torch.topk(
                    confidence, num_to_mask, dim=-1, largest=False
                )
                sampled.scatter_(1, mask_indices, tokenizer.mask_token_id)

            input_ids = sampled

    # Print the absolute final frame
    final_str = get_visual_string(input_ids[0], tokenizer)
    print(f"Step {STEPS:02d} | Masks: 00 | {final_str}")
    print("-" * 100)

    # =========================================================================
    # DECODE AND ANALYZE
    # =========================================================================
    final_ids = input_ids[0].cpu().tolist()
    
    # Slice at EOS
    if tokenizer.eos_token_id in final_ids:
        final_ids = final_ids[:final_ids.index(tokenizer.eos_token_id)]
    
    # Strip any stray specials
    clean_ids = [i for i in final_ids if i not in (tokenizer.pad_token_id, tokenizer.mask_token_id)]
    
    selfies_str = tokenizer.decode(clean_ids)
    print(f"\nFinal SELFIES: {selfies_str}")

    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            logp = Descriptors.MolLogP(mol)
            qed = QED.qed(mol)
            print(f"Final SMILES : {Chem.MolToSmiles(mol)}")
            print(f"Valid RDKit  : ✅ YES")
            print(f"Properties   : LogP = {logp:+.2f} | QED = {qed:.3f} | Heavy Atoms = {mol.GetNumHeavyAtoms()}")
        else:
            print(f"Final SMILES : {smiles}")
            print(f"Valid RDKit  : ❌ NO (Sanitization Failed)")
    except Exception as e:
        print(f"Valid RDKit  : ❌ NO (SELFIES Error: {e})")

    print("\n")

if __name__ == "__main__":
    main()