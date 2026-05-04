import torch
import torch.nn as nn
import math

# =============================================================================
# 1. SINUSOIDAL TIMESTEP EMBEDDING  (unchanged)
# =============================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # A small MLP that projects the sinusoidal features to hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def sinusoidal_features(self, t):
        """
        Convert scalar timesteps to sinusoidal feature vectors.

        t: (B, 1) tensor of timestep values between 0 and 1
        returns: (B, hidden_size) tensor
        """
        device = t.device
        half = self.hidden_size // 2

        # Create a bank of frequencies
        # These are exponentially spaced between 1 and 10000
        frequencies = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1))
        )                                               # (hidden_size/2,)

        # Multiply each timestep by each frequency
        angles = t * frequencies.unsqueeze(0)           # (B, hidden_size/2)

        # Concatenate sine and cosine of each angle
        features = torch.cat([torch.sin(angles),
                              torch.cos(angles)], dim=-1)  # (B, hidden_size)

        return features

    def forward(self, t):
        """
        t: (B, 1) timestep values
        returns: (B, hidden_size) rich timestep representation
        """
        # 1. Convert scalar t to sinusoidal features
        features = self.sinusoidal_features(t)      # (B, hidden_size)

        # 2. Pass through MLP for learnable refinement
        return self.mlp(features)                   # (B, hidden_size)


# =============================================================================
# 2. ROTARY POSITIONAL EMBEDDING (RoPE)
# =============================================================================
#
# Replaces the learned absolute PositionalEmbedding. RoPE is applied
# multiplicatively to Q and K inside self-attention (see transformerBlock.py),
# not added to the residual stream. It gives:
#   - relative-position-aware attention without extra parameters
#   - clean length extrapolation (the cache just grows)
#   - no positional drift across diffusion timesteps
#
# Important: do NOT apply RoPE to cross-attention. The text K,V already carry
# their own position info from the HuggingFace text encoder.
#
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoFormer / LLaMA style).

    Caches cos/sin tables up to `max_length`. If a longer sequence comes in,
    the cache transparently grows.
    """

    def __init__(self, head_dim, max_length=4096, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {head_dim}"
            )

        self.head_dim = head_dim
        self.base = base

        # inv_freq: (head_dim/2,)  →  rotation rates per frequency band
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cos/sin caches built on demand
        self._cached_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_length)

    def _build_cache(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # (seq_len, head_dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate so each pair of dims gets the same angle: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Reshape for broadcasting against (B, num_heads, seq_len, head_dim)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self._cached_len = seq_len

    def forward(self, seq_len, device, dtype):
        """Return (cos, sin) tables of shape (1, 1, seq_len, head_dim)."""
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x):
    """Rotate the second half of the head dim by 90 degrees: [a, b] -> [-b, a]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE to query and key tensors.

    q, k:      (B, num_heads, seq_len, head_dim)
    cos, sin:  (1, 1, seq_len, head_dim)  from RotaryEmbedding.forward
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
