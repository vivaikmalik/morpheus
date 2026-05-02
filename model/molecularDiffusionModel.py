"""
model/molecularDiffusionModel.py
--------------------------------
Masked diffusion transformer for molecular generation.

Features:
  - RoPE (rotary position embeddings) instead of learned positional embeddings
  - Cross-attention to text embeddings (zero-init for stable fine-tuning)
  - Optional text encoder (None for pretraining, HF model for fine-tuning)
  - set_text_encoder() to swap in contrastively-trained encoders
  - encoder_proj for narrow encoders (e.g. SciBERT 768 → BGE-aligned 1024)
  - Classifier-free guidance via uncond_prob dropout to null_token
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding
from model.transformerBlock import TransformerBlock


class MolecularDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        ffn_dim,
        num_layers,
        max_length,
        pad_token_id,
        text_model_name: Optional[str] = None,  # None during pretraining
        uncond_prob: float = 0.1,
        dropout: float = 0.1,
        load_text_encoder: bool = True,         # False to skip loading HF weights
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.uncond_prob = uncond_prob
        self.has_text_encoder = text_model_name is not None

        # ------------------------------------------------------------------ #
        # TEXT ENCODER (optional)
        # ------------------------------------------------------------------ #
        self.text_encoder = None
        self.text_proj = None
        self.encoder_proj: Optional[nn.Linear] = None
        self.null_token = None

        if self.has_text_encoder:
            if load_text_encoder:
                from transformers import AutoModel
                self.text_encoder = AutoModel.from_pretrained(text_model_name)
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
                text_hidden_size = self.text_encoder.config.hidden_size
            else:
                # User will set encoder later via set_text_encoder()
                # Default to BGE-large dimension (1024)
                text_hidden_size = 1024

            # Projector: text_hidden → ffn_dim → hidden_size
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden_size, ffn_dim),
                nn.SiLU(),
                nn.Linear(ffn_dim, hidden_size),
            )

            # Null token for classifier-free guidance
            self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # ------------------------------------------------------------------ #
        # DIFFUSION COMPONENTS
        # ------------------------------------------------------------------ #
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # NOTE: NO learned positional embedding — RoPE is applied inside attention
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout,
                             use_rope=True, max_seq_len=max_length)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()
        self.lm_head.weight = self.token_embedding.weight  # weight tying

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

        # Re-zero cross-attention out_proj (might have been overwritten)
        for block in self.blocks:
            nn.init.zeros_(block.cross_attention.out_proj.weight)
            nn.init.zeros_(block.cross_attention.out_proj.bias)

    # ====================================================================== #
    # TEXT ENCODER MANAGEMENT
    # ====================================================================== #

    def set_text_encoder(self, encoder: nn.Module) -> None:
        """
        Replace the text encoder with a (possibly contrastively-trained) one.
        If the new encoder's hidden dim differs from text_proj input, an
        encoder_proj layer is added.

        Must be called BEFORE load_state_dict if loading a checkpoint that was
        saved with this encoder, so the encoder_proj keys exist.
        """
        assert self.has_text_encoder, "Model has no text encoder slot"
        self.text_encoder = encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        enc_dim = encoder.config.hidden_size
        proj_in_dim = self.text_proj[0].in_features

        if enc_dim != proj_in_dim:
            proj = nn.Linear(enc_dim, proj_in_dim).to(next(encoder.parameters()).device)
            nn.init.zeros_(proj.bias)
            nn.init.zeros_(proj.weight)
            with torch.no_grad():
                overlap = min(enc_dim, proj_in_dim)
                proj.weight[:overlap, :overlap] = torch.eye(overlap)
            self.encoder_proj = proj
            print(f"[model] encoder_proj: {enc_dim} → {proj_in_dim}  (trainable)")
        else:
            self.encoder_proj = None
            print(f"[model] encoder_proj not needed ({enc_dim}-dim)")

    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        """Run text encoder + (optional encoder_proj) + text_proj."""
        assert self.text_encoder is not None
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids, attention_mask=text_attention_mask)
            hidden_states = outputs.last_hidden_state

        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)
        return self.text_proj(hidden_states)

    def project_text_embeddings(self, scibert_states):
        """For pre-computed embeddings: skip the encoder, run only the projector."""
        if self.encoder_proj is not None:
            scibert_states = self.encoder_proj(scibert_states)
        return self.text_proj(scibert_states)

    # ====================================================================== #
    # FORWARD
    # ====================================================================== #

    def forward(self, input_ids, timesteps, text_embeds=None, text_padding_mask=None):
        """
        input_ids:   (B, seq_len) — masked token IDs
        timesteps:   (B, 1) — noise level
        text_embeds: (B, text_len, hidden_size) or None for pretraining
        text_padding_mask: (B, text_len) where True = PAD
        """
        device = input_ids.device
        B, seq_len = input_ids.shape
        padding_mask = (input_ids == self.pad_token_id)

        # Apply CFG dropout during training
        if self.training and text_embeds is not None and self.has_text_encoder:
            keep_mask = (torch.rand(B, device=device) > self.uncond_prob).view(B, 1, 1)
            null_seq = self.null_token.expand(B, text_embeds.size(1), -1)
            text_embeds = torch.where(keep_mask, text_embeds, null_seq)

        # Embed tokens (NO positional add — RoPE handles it inside attention)
        x = self.token_embedding(input_ids)
        t_emb = self.timestep_embedding(timesteps)
        x = x + t_emb.unsqueeze(1)
        x = self.input_norm(x)

        # Transformer
        for block in self.blocks:
            x = block(x, text_embeds, padding_mask, text_padding_mask)

        x = self.output_norm(x)
        return self.lm_head(x)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable, total - trainable