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
        hidden_size,          # 768
        num_heads,            # 12
        ffn_dim,              # 3072
        num_layers,           # 12
        max_length,           # 74
        pad_token_id,         # 0
        text_model_name=None, # None during pretraining, "allenai/scibert_scivocab_uncased" during fine-tuning
        uncond_prob=0.1,
        dropout=0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.uncond_prob  = uncond_prob
        self.has_text_encoder = text_model_name is not None

        # ------------------------------------------------------------------ #
        # TEXT ENCODER (only loaded during fine-tuning)                       #
        # During pretraining: text_model_name=None, no SciBERT loaded        #
        # During fine-tuning: text_model_name="allenai/scibert_scivocab_uncased"
        # ------------------------------------------------------------------ #
        if self.has_text_encoder:
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            text_hidden_size = self.text_encoder.config.hidden_size  # 768

            # Projector: maps SciBERT output to model's hidden_size
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_size, ffn_dim),    # 768 → 3072
                nn.SiLU(),
                nn.Linear(ffn_dim, hidden_size),          # 3072 → 768
            )

            # Null token for classifier-free guidance
            self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # ------------------------------------------------------------------ #
        # DIFFUSION COMPONENTS                                                #
        # ------------------------------------------------------------------ #
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.pos_embedding      = PositionalEmbedding(max_length, hidden_size)
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm         = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------ #
        # TRANSFORMER BLOCKS                                                   #
        # Each block has: self-attention → cross-attention → FFN              #
        # During pretraining: cross-attention acts as 2nd self-attention      #
        # During fine-tuning: cross-attention attends to text embeddings      #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------ #
        # OUTPUT HEAD                                                          #
        # ------------------------------------------------------------------ #
        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        # ------------------------------------------------------------------ #
        # INITIALIZATION                                                       #
        # ------------------------------------------------------------------ #
        self._init_weights()
        self.lm_head.weight = self.token_embedding.weight   # weight tying

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith("text_encoder"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------ #
    # TEXT EMBEDDING (only available during fine-tuning)                    #
    # ------------------------------------------------------------------ #

    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        """
        Run frozen SciBERT + learnable projector.
        Only callable when text_model_name was provided at construction.
        """
        assert self.has_text_encoder, "No text encoder — was text_model_name=None?"

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            hidden_states = outputs.last_hidden_state   # (B, text_len, 768)

        return self.text_proj(hidden_states)             # (B, text_len, hidden_size)

    # ------------------------------------------------------------------ #
    # FORWARD PASS                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, input_ids, timesteps, text_embeds=None, text_padding_mask=None):
        """
        input_ids:         (B, seq_len)             masked SELFIES token IDs
        timesteps:         (B, 1)                   noise level t in [0, 1]
        text_embeds:       (B, text_len, hidden_size) projected text embeddings
                           or None → cross-attention uses molecule states (pretraining)
        text_padding_mask: (B, text_len)            True where text is PAD

        returns:           (B, seq_len, vocab_size) logits over vocabulary
        """
        device = input_ids.device
        B, seq_len = input_ids.shape

        # --- Molecule embedding pipeline ---
        padding_mask = (input_ids == self.pad_token_id)

        x = self.token_embedding(input_ids)
        pos = self.pos_embedding(seq_len, device)
        x = x + pos
        t_emb = self.timestep_embedding(timesteps)
        x = x + t_emb.unsqueeze(1)
        x = self.input_norm(x)

        # --- Determine cross-attention input ---
        if text_embeds is not None:
            # Fine-tuning with text: cross-attention attends to text
            cross_input = text_embeds
            cross_mask  = text_padding_mask

            # CFG dropout: randomly replace text with null token during training
            if self.training and self.has_text_encoder and self.uncond_prob > 0:
                drop_mask = torch.rand(B, device=device) < self.uncond_prob
                if drop_mask.any():
                    # When text is dropped, cross-attention falls back to molecule states
                    # (same as pretraining behavior) — NOT a null token
                    for i in range(B):
                        if drop_mask[i]:
                            cross_input[i] = x[i]
                            if cross_mask is not None:
                                cross_mask[i] = padding_mask[i]
        else:
            # Pretraining (no text): cross-attention = 2nd self-attention
            cross_input = x
            cross_mask  = padding_mask

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, cross_input, padding_mask, cross_mask)

        # --- Output head ---
        x      = self.output_norm(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return total, trainable, frozen