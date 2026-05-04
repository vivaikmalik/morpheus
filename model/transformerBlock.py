import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from model.embeddings import apply_rope


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads  # e.g. 512 // 8 = 64

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.head_dim)

    def forward(self, x, padding_mask=None, rope=None):
        """
        x:            (B, seq_len, hidden_size)
        padding_mask: (B, seq_len) True where PAD tokens are
        rope:         optional (cos, sin) tuple from RotaryEmbedding.forward
        """
        B, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ----- RoPE (applied to Q and K only; V is untouched) -----
        if rope is not None:
            cos, sin = rope
            Q, K = apply_rope(Q, K, cos, sin)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, seq_len), dtype=x.dtype, device=x.device)
            mask_bool = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.masked_fill(mask_bool, float('-inf'))

        attended = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(B, seq_len, self.hidden_size)

        return self.out_proj(attended)


class CrossAttention(nn.Module):
    """SELFIES queries → text key/value. RoPE is intentionally NOT applied here:
    the text K,V already carry their own positional information from the HF
    encoder, and rotating SELFIES Q without rotating K together breaks the
    inner-product structure RoPE relies on."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Q from SELFIES, K & V from text embeddings
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Critical to preserve the unconditional pre-training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_embeds, text_padding_mask=None):
        B, seq_len, _ = x.shape
        _, text_len, _ = text_embeds.shape

        Q = self.q_proj(x)
        K = self.k_proj(text_embeds)
        V = self.v_proj(text_embeds)

        Q = Q.view(B, seq_len,  self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, text_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if text_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, text_len), dtype=x.dtype, device=x.device)
            mask_bool = text_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.masked_fill(mask_bool, float('-inf'))

        attended = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(B, seq_len, self.hidden_size)

        return self.out_proj(attended)


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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.norm1     = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, num_heads, dropout)

        self.norm_cross      = nn.LayerNorm(hidden_size)
        self.cross_attention = CrossAttention(hidden_size, num_heads, dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn   = FeedForward(hidden_size, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_embeds, padding_mask=None,
                text_padding_mask=None, rope=None):
        # --- Self-Attention (with RoPE) ---
        residual = x
        x = self.norm1(x)
        x = self.attention(x, padding_mask, rope=rope)
        x = self.dropout(x)
        x = residual + x

        # --- Cross-Attention (no RoPE) ---
        residual = x
        x = self.norm_cross(x)
        x = self.cross_attention(x, text_embeds, text_padding_mask)
        x = self.dropout(x)
        x = residual + x

        # --- FFN ---
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
