import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import TimestepEmbedding, PositionalEmbedding
from model.transformerBlock import TransformerBlock
from transformers import AutoModel


class MolecularDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,           # 110
        hidden_size,          # 256
        num_heads,            # 8
        ffn_dim,              # 512
        num_layers,           # 8
        max_length,           # 74
        pad_token_id,         # 0
        text_model_name="allenai/scibert_scivocab_uncased",
        uncond_prob=0.1,      # fraction of training samples where text is dropped
        dropout=0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.uncond_prob  = uncond_prob

        # ------------------------------------------------------------------ #
        # TEXT ENCODER (frozen SciBERT)                                       #
        # Produces contextual embeddings of the text description.             #
        # 768-dim output, frozen — no gradients, no weight updates.           #
        # ------------------------------------------------------------------ #
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
 
        text_hidden_size = self.text_encoder.config.hidden_size  # 768

        # ------------------------------------------------------------------ #
        # TEXT PROJECTOR                                                       #
        # Maps SciBERT's 768-dim to model's hidden_size (256).                #
        # This is the only learnable bridge between text and molecule.        #
        # ------------------------------------------------------------------ #
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, ffn_dim),   # 768 → 512
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),         # 512 → 256
        )

         # ------------------------------------------------------------------ #
        # NULL TOKEN (for classifier-free guidance)                           #
        # During training, text embeddings are randomly replaced with this    #
        # null token. At inference, the null token gives the "unconditional"  #
        # logits for CFG interpolation.                                       #
        # ------------------------------------------------------------------ #
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)



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
        for name, module in self.named_modules():
            if name.startswith("text_encoder"):
                continue
            # Don't touch cross-attention out_proj (already zero-init)
            if "cross_attention.out_proj" in name:
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
    # TEXT EMBEDDING HELPERS                                               #
    # ------------------------------------------------------------------ #
 
    def get_text_embeddings(self, text_input_ids, text_attention_mask):
        """
        Run frozen SciBERT + learnable projector.
 
        text_input_ids:    (B, text_len)  SciBERT token IDs
        text_attention_mask: (B, text_len) 1 = real token, 0 = padding
 
        returns: (B, text_len, hidden_size)  projected text embeddings
        """
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            hidden_states = outputs.last_hidden_state   # (B, text_len, 768)
 
        # Learnable projection: 768 → 256
        return self.text_proj(hidden_states)             # (B, text_len, 256)
    
    def forward(self, input_ids, timesteps, text_embeds=None, text_padding_mask=None):
        """
        input_ids:         (B, seq_len)             masked SELFIES token IDs
        timesteps:         (B, 1)                   noise level t in [0, 1]
        text_embeds:       (B, text_len, hidden_size) projected text embeddings
                           or None → uses null token (unconditional)
        text_padding_mask: (B, text_len)            True where text is PAD
 
        returns:           (B, seq_len, vocab_size) logits over vocabulary
        """
        device  = input_ids.device
        B, seq_len = input_ids.shape

        # --- Default to null token if no text is provided (unconditional) ---
        if text_embeds is None:
            text_embeds = self.null_token.expand(B, 1, -1)
            text_padding_mask = None   # single null token, no padding
        
        # --- Classifier-free guidance dropout during training ---
        # With probability uncond_prob, replace text with null token.
        # This teaches the model to generate both with and without text,
        # which is required for CFG at inference time.
        if self.training and self.uncond_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.uncond_prob   # (B,)
            if drop_mask.any():
                null_seq = self.null_token.expand(B, text_embeds.size(1), -1)
                keep = (~drop_mask).float().view(B, 1, 1)
                text_embeds = text_embeds * keep + null_seq * (1 - keep)
                # Also zero out text_padding_mask for dropped samples
                if text_padding_mask is not None:
                    text_padding_mask = text_padding_mask.clone()
                    text_padding_mask[drop_mask] = False
        
        # --- Molecule embedding pipeline (unchanged from pretrained) ---
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
            x = block(x, text_embeds, padding_mask, text_padding_mask)                   # (B, seq_len, hidden_size)

        # --- 7. Output head ---
        x      = self.output_norm(x)                     # (B, seq_len, hidden_size)
        logits = self.lm_head(x)                         # (B, seq_len, vocab_size)

        return logits

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return total, trainable, frozen
