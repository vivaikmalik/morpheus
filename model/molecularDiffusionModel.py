import torch
import torch.nn as nn
import sys
from pathlib import Path
from transformers import AutoModel
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding, RotaryEmbedding
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
        text_model_name="BAAI/bge-large-en-v1.5",
        uncond_prob=0.1,
        dropout=0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.uncond_prob  = uncond_prob
        self.max_length   = max_length

        # ------------------------------------------------------------------ #
        # TEXT ENCODER & PROJECTOR (FROZEN)                                  #
        # ------------------------------------------------------------------ #
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        text_hidden_size = self.text_encoder.config.hidden_size

        # MLP projector: text encoder output → model hidden_size.
        # Assumes text_hidden_size == 1024 (BGE-large default).
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size)
        )

        # Optional narrow-encoder projection (set via set_text_encoder).
        # None when encoder hidden_dim matches text_proj input dim (BGE: 1024).
        # Registered as nn.Linear when a narrower encoder is used (e.g. SciBERT: 768),
        # so it is included in state_dict and trained during fine-tuning.
        self.encoder_proj: nn.Linear | None = None

        # Learnable null token for CFG
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # ------------------------------------------------------------------ #
        # DIFFUSION EMBEDDINGS                                               #
        # ------------------------------------------------------------------ #
        self.token_embedding    = nn.Embedding(vocab_size, hidden_size,
                                               padding_idx=pad_token_id)
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        self.input_norm         = nn.LayerNorm(hidden_size)

        # RoPE replaces the old learned PositionalEmbedding. It carries no
        # trainable params (only buffers), and is applied to Q/K inside each
        # SelfAttention. Built with a generous max_length so we never hit the
        # cache rebuild path during typical training/inference.
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_length=max(max_length, 4096),
        )

        # ------------------------------------------------------------------ #
        # TRANSFORMER BLOCKS                                                 #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head     = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith('text_encoder'):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        for block in self.blocks:
            nn.init.zeros_(block.cross_attention.out_proj.weight)
            nn.init.zeros_(block.cross_attention.out_proj.bias)

    def set_text_encoder(self, encoder: nn.Module) -> None:
        """Replace the text encoder and wire up encoder_proj if hidden dims differ.

        Must be called BEFORE load_state_dict when loading a fine-tune checkpoint
        that was saved with a non-default encoder, so that encoder_proj keys are
        present in the model before strict loading.

        When encoder.config.hidden_size == text_proj input dim (1024 for BGE),
        encoder_proj is set to None and the forward path is identical to default.
        """
        self.text_encoder = encoder
        enc_dim     = encoder.config.hidden_size
        proj_in_dim = self.text_proj[0].in_features   # first Linear of text_proj

        if enc_dim != proj_in_dim:
            proj = nn.Linear(enc_dim, proj_in_dim).to(next(encoder.parameters()).device)
            nn.init.zeros_(proj.bias)
            # Near-identity init for the rows that overlap; zero-pad the rest.
            nn.init.zeros_(proj.weight)
            with torch.no_grad():
                overlap = min(enc_dim, proj_in_dim)
                proj.weight[:overlap, :overlap] = torch.eye(overlap)
            self.encoder_proj = proj
            print(f"[model] encoder_proj created: {enc_dim} → {proj_in_dim}  (trainable)")
        else:
            self.encoder_proj = None
            print(f"[model] encoder_proj not needed ({enc_dim}-dim matches text_proj input)")

    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        """Run the text encoder and project embeddings to model hidden_size."""
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            hidden_states = outputs.last_hidden_state   # [B, L, enc_dim]

        # encoder_proj is outside no_grad so gradients flow during fine-tuning.
        if self.encoder_proj is not None:
            hidden_states = self.encoder_proj(hidden_states)   # [B, L, proj_in_dim]

        return self.text_proj(hidden_states)   # [B, L, hidden_size]

    def forward(self, input_ids, timesteps, text_embeds, text_padding_mask=None):
        device  = input_ids.device
        B, seq_len = input_ids.shape

        if self.training:
            # Drop text conditioning with probability `uncond_prob`
            keep_mask = torch.rand(B, device=device) > self.uncond_prob
            keep_mask = keep_mask.view(B, 1, 1)

            null_seq    = self.null_token.expand(B, text_embeds.size(1), -1)
            text_embeds = torch.where(keep_mask, text_embeds, null_seq)

        padding_mask = (input_ids == self.pad_token_id)

        # Token + timestep embeddings (NO additive positional embedding —
        # position is now injected via RoPE inside each self-attention).
        x = self.token_embedding(input_ids)
        t_emb = self.timestep_embedding(timesteps)
        x = x + t_emb.unsqueeze(1)
        x = self.input_norm(x)

        # Compute RoPE cos/sin once for this forward pass and reuse across blocks.
        rope = self.rope(seq_len, device=device, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, text_embeds,
                      padding_mask=padding_mask,
                      text_padding_mask=text_padding_mask,
                      rope=rope)

        x      = self.output_norm(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
