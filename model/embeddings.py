"""
model/embeddings.py
-------------------
Embeddings used in Morpheus diffusion model:
  - TimestepEmbedding: scalar t → sinusoidal → MLP → hidden_size
  - RoPE (Rotary Position Embedding): applied inside attention
  - PositionalEmbedding: kept for backward compat (legacy learned embeddings)

RoPE encodes positions by *rotating* Q and K vectors in 2D pairs by an angle
proportional to position. Unlike learned embeddings, RoPE generalises to
sequences longer than seen during training and encodes RELATIVE positions.

Reference: https://arxiv.org/abs/2104.09864
"""

import math
import torch
import torch.nn as nn


# =============================================================================
# 1. TIMESTEP EMBEDDING (unchanged)
# =============================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def sinusoidal_features(self, t):
        device = t.device
        half = self.hidden_size // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1))
        )
        angles = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, t):
        return self.mlp(self.sinusoidal_features(t))


# =============================================================================
# 2. ROTARY POSITION EMBEDDING (new - replaces learned PositionalEmbedding)
# =============================================================================

class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    Applied to Q and K inside attention. For each pair of dimensions,
    rotate by an angle θ_i * position_index where θ_i = 10000^(-2i/d).

    Args:
        head_dim: dimension PER ATTENTION HEAD (not full hidden_size)
        max_seq_len: pre-compute rotations up to this length
        base: rotation base frequency (default 10000)
    """
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies for each pair of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)  # (seq_len, head_dim/2)
        # Duplicate each freq so we can apply to (x_even, x_odd) pairs
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        """Rotate the last dim by 90°: (x1, x2) → (-x2, x1)."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        Apply rotary embeddings to Q and K.

        q, k: (B, num_heads, seq_len, head_dim)
        returns: rotated q, k
        """
        seq_len = q.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(q.device)  # (seq_len, head_dim)
        sin = self.sin_cached[:seq_len].to(q.device)

        # Broadcast: (seq_len, head_dim) → (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


# =============================================================================
# 3. LEGACY LEARNED POSITIONAL EMBEDDING (kept for backward compat)
# =============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)