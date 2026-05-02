"""
model/transformerBlock.py
-------------------------
Transformer block with:
  - Self-attention with RoPE (rotary position embeddings)
  - Cross-attention to text embeddings (zero-init for stable fine-tuning)
  - SiLU FFN with pre-LayerNorm

The cross-attention's out_proj is zero-initialized so at the start of fine-tuning
it produces zero output → preserves pretrained behavior. As training progresses
it learns to inject text conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import RoPE


# =============================================================================
# SELF-ATTENTION WITH ROPE
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, use_rope=True, max_seq_len=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        if use_rope:
            self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, padding_mask=None):
        B, seq_len, _ = x.shape

        Q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            Q, K = self.rope(Q, K)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, seq_len), dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attended = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        return self.out_proj(attended)


# =============================================================================
# CROSS-ATTENTION (text conditioning)
# =============================================================================

class CrossAttention(nn.Module):
    """
    Q from molecule hidden states, K & V from text embeddings.
    out_proj is zero-initialized so cross-attention starts as a no-op,
    preserving pretrained behavior at the start of fine-tuning.

    NO RoPE here — text and molecule have independent position systems.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Critical: zero-init out_proj so initial output is zero
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_embeds, text_padding_mask=None):
        B, seq_len, _ = x.shape
        _, text_len, _ = text_embeds.shape

        Q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(text_embeds).view(B, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(text_embeds).view(B, text_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if text_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, text_len), dtype=x.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(
                text_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attended = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        return self.out_proj(attended)


# =============================================================================
# FFN
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# TRANSFORMER BLOCK (self-attn → cross-attn → FFN, all pre-norm + residual)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1,
                 use_rope=True, max_seq_len=2048):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, num_heads, dropout,
                                         use_rope=use_rope, max_seq_len=max_seq_len)

        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attention = CrossAttention(hidden_size, num_heads, dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_embeds=None, padding_mask=None, text_padding_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, padding_mask)
        x = self.dropout(x)
        x = residual + x

        # Cross-attention (skip if no text)
        if text_embeds is not None:
            residual = x
            x = self.norm_cross(x)
            x = self.cross_attention(x, text_embeds, text_padding_mask)
            x = self.dropout(x)
            x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x