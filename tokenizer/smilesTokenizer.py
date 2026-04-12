"""
tokenizer/smilesTokenizer.py
------------------------------
Regex-based SMILES tokenizer matching TGM-DLM's regexTokenizer.

Special tokens:
    0 = [PAD]
    1 = [MASK]   (for masked diffusion)
    2 = [EOS]

Chemical tokens start at ID 3.

Usage:
    tokenizer = SmilesTokenizer("smiles_vocab.json")
    ids = tokenizer.encode("CC(=O)O")       # → [4, 4, 23, 12, 24, 4]
    smi = tokenizer.decode(ids)              # → "CC(=O)O"
"""

import json
import re


class SmilesTokenizer:
    # Standard SMILES regex — same as TGM-DLM
    PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.pad_token_id = data["pad_token_id"]       # 0
        self.mask_token_id = data["mask_token_id"]     # 1
        self.eos_token_id = data["eos_token_id"]       # 2
        self.vocab_size = data["vocab_size"]
        self.chemical_start_id = data["chemical_start_id"]  # 3
        self.regex = re.compile(self.PATTERN)

    def tokenize(self, smiles):
        """Split a SMILES string into tokens."""
        return self.regex.findall(smiles)

    def encode(self, smiles):
        """Convert a SMILES string to a list of token IDs."""
        tokens = self.tokenize(smiles)
        ids = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            # Skip unknown tokens (rare edge case)
        return ids

    def decode(self, ids):
        """Convert a list of token IDs back to a SMILES string."""
        tokens = []
        for i in ids:
            if i in (self.pad_token_id, self.mask_token_id, self.eos_token_id):
                continue
            if i in self.id_to_token:
                tokens.append(self.id_to_token[i])
        return "".join(tokens)

    @classmethod
    def build_vocab(cls, smiles_list, save_path):
        """
        Build vocabulary from a list of SMILES strings and save as JSON.

        Args:
            smiles_list: list of SMILES strings
            save_path: where to save the vocab JSON
        """
        regex = re.compile(cls.PATTERN)
        token_set = set()

        for smi in smiles_list:
            tokens = regex.findall(smi)
            token_set.update(tokens)

        # Sort for reproducibility
        sorted_tokens = sorted(token_set)

        # Build mappings: PAD=0, MASK=1, EOS=2, chemical tokens from 3
        token_to_id = {"[PAD]": 0, "[MASK]": 1, "[EOS]": 2}
        id_to_token = {0: "[PAD]", 1: "[MASK]", 2: "[EOS]"}

        for i, tok in enumerate(sorted_tokens):
            token_to_id[tok] = i + 3
            id_to_token[i + 3] = tok

        vocab_data = {
            "token_to_id": token_to_id,
            "id_to_token": {str(k): v for k, v in id_to_token.items()},
            "pad_token_id": 0,
            "mask_token_id": 1,
            "eos_token_id": 2,
            "chemical_start_id": 3,
            "vocab_size": len(token_to_id),
        }

        with open(save_path, "w") as f:
            json.dump(vocab_data, f, indent=2)

        print(f"Vocab saved: {save_path}")
        print(f"  Total tokens: {len(token_to_id)} (3 special + {len(sorted_tokens)} chemical)")

        return vocab_data