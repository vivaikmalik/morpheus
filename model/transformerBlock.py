import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads  # 512 // 8 = 64

        # These three linear layers create the Q, K, V matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Final projection after attention
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.head_dim)  # scaling factor

    def forward(self, x, padding_mask=None):
        """
        x:            (B, seq_len, hidden_size)
        padding_mask: (B, seq_len) True where PAD tokens are
        returns:      (B, seq_len, hidden_size)
        """
        B, seq_len, _ = x.shape

        # --- 1. Project input into Q, K, V ---
        Q = self.q_proj(x)  # (B, seq_len, hidden_size)
        K = self.k_proj(x)  # (B, seq_len, hidden_size)
        V = self.v_proj(x)  # (B, seq_len, hidden_size)

        # --- 2. Split into multiple heads ---
        # Reshape to (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = torch.zeros((B, 1, 1, seq_len), dtype=x.dtype, device=x.device)
            mask_bool = padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, seq_len)
            attn_mask = attn_mask.masked_fill(mask_bool, float('-inf'))

        attended = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # --- 7. Merge heads back together ---
        attended = attended.transpose(1, 2).contiguous()  # (B, seq_len, num_heads, head_dim)
        attended = attended.view(B, seq_len, self.hidden_size)  # (B, seq_len, hidden_size)

        # --- 8. Final projection ---
        return self.out_proj(attended)  # (B, seq_len, hidden_size)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),  # expand: 512 → 2048
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),  # contract: 2048 → 512
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)  # (B, seq_len, hidden_size)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.norm1     = nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, num_heads, dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn   = FeedForward(hidden_size, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        x:            (B, seq_len, hidden_size)
        padding_mask: (B, seq_len) True where PAD tokens are
        returns:      (B, seq_len, hidden_size)
        """
        # --- Attention with residual ---
        residual = x
        x = self.norm1(x)                        # normalize first
        x = self.attention(x, padding_mask)      # attend
        x = self.dropout(x)
        x = residual + x                         # add residual

        # --- FFN with residual ---
        residual = x
        x = self.norm2(x)                        # normalize first
        x = self.ffn(x)                          # transform
        x = residual + x                         # add residual

        return x