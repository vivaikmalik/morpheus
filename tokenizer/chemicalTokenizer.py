import json
import selfies as sf

class ChemicalTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, "r") as f:
            data = json.load(f)
        
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.pad_token_id  = data["pad_token_id"]
        self.mask_token_id = data["mask_token_id"]
        self.eos_token_id  = data["eos_token_id"]
        self.vocab_size    = data["vocab_size"]
        self.chemical_start_id = data["chemical_start_id"]

    def encode(self, selfies_str):
        """Convert a SELFIES string to a list of token IDs"""
        tokens = list(sf.split_selfies(selfies_str))
        ids = [self.token_to_id.get(t, self.pad_token_id) for t in tokens]
        return ids

    def decode(self, ids):
        """Convert a list of token IDs back to a SELFIES string"""
        tokens = [self.id_to_token[i] for i in ids 
                  if i not in (self.pad_token_id, self.mask_token_id, self.eos_token_id)]
        return "".join(tokens)

"""
tokenizer = ChemicalTokenizer("chemical_tokenizer.json")

test = "[C][Branch1][C][O][C][=C][C][=C][C][=C][Ring1][=C]"
ids = tokenizer.encode(test)
back = tokenizer.decode(ids)

print(f"Original : {test}")
print(f"Token IDs: {ids}")
print(f"Decoded  : {back}")
print(f"Match    : {test == back}")
print(f"\nVocab size: {tokenizer.vocab_size}")
print(f"MASK token ID: {tokenizer.mask_token_id}")
print(f"Chemical tokens start at: {tokenizer.chemical_start_id}")

"""