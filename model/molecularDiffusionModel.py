import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding, PositionalEmbedding
from model.transformerBlock import TransformerBlock


class MolecularDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,           # 110
        hidden_size,          # 256
        num_heads,            # 8
        ffn_dim,              # 512
        num_layers,           # 6
        max_length,           # 72
        pad_token_id,         # 0
        dropout=0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size

        # ------------------------------------------------------------------ #
        # 1. TOKEN EMBEDDING                                                   #
        # Maps each token ID to a learned vector of size hidden_size           #
        # ------------------------------------------------------------------ #
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id   # PAD token always maps to zero vector
        )

        # ------------------------------------------------------------------ #
        # 2. POSITIONAL EMBEDDING                                              #
        # Tells the model where each token is in the sequence                  #
        # ------------------------------------------------------------------ #
        self.pos_embedding = PositionalEmbedding(max_length, hidden_size)

        # ------------------------------------------------------------------ #
        # 3. TIMESTEP EMBEDDING                                                #
        # Converts scalar t → rich 256-dim vector                             #
        # ------------------------------------------------------------------ #
        self.timestep_embedding = TimestepEmbedding(hidden_size)

        # ------------------------------------------------------------------ #
        # 5. INPUT LAYER NORM                                                  #
        # Normalizes the combined embedding before transformer layers          #
        # ------------------------------------------------------------------ #
        self.input_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------ #
        # 6. TRANSFORMER BLOCKS                                                #
        # 6 identical blocks stacked                                           #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------ #
        # 7. OUTPUT HEAD                                                       #
        # Projects from hidden_size back to vocab_size to get logits           #
        # ------------------------------------------------------------------ #
        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        # ------------------------------------------------------------------ #
        # 8. WEIGHT INITIALIZATION                                             #
        # ------------------------------------------------------------------ #
        self._init_weights()

        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self):
        """
        Initialize weights carefully.
        Small weights at the start prevent exploding/vanishing gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, timesteps):
        """
        input_ids:   (B, seq_len)  token IDs, some replaced with MASK
        timesteps:   (B, 1)        noise level t between 0 and 1
        cond_values: (B, 1)        target LogP values

        returns:     (B, seq_len, vocab_size)  logits over vocabulary
        """
        device  = input_ids.device
        B, seq_len = input_ids.shape

        # --- Padding mask: True where PAD tokens are ---
        padding_mask = (input_ids == self.pad_token_id)  # (B, seq_len)

        # --- 1. Token embeddings ---
        x = self.token_embedding(input_ids)              # (B, seq_len, hidden_size)

        # --- 2. Positional embeddings ---
        pos = self.pos_embedding(seq_len, device)        # (1, seq_len, hidden_size)
        x = x + pos

        # --- 3. Timestep embedding ---
        t_emb = self.timestep_embedding(timesteps)       # (B, hidden_size)

        x = x + t_emb.unsqueeze(1)

        # --- 5. Normalize combined input ---
        x = self.input_norm(x)                           # (B, seq_len, hidden_size)

        # --- 6. Pass through transformer blocks ---
        for block in self.blocks:
            x = block(x, padding_mask)                   # (B, seq_len, hidden_size)

        # --- 7. Output head ---
        x      = self.output_norm(x)                     # (B, seq_len, hidden_size)
        logits = self.lm_head(x)                         # (B, seq_len, vocab_size)

        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
