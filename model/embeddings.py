import torch
import torch.nn as nn
import math

# =============================================================================
# 1. SINUSOIDAL TIMESTEP EMBEDDING
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



### For now, nothing 

# =============================================================================
# 3. POSITIONAL EMBEDDING
# =============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, hidden_size):
        super().__init__()

        # One learned vector per position
        self.embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, seq_len, device):
        """
        seq_len: integer, length of the current sequence
        returns: (1, seq_len, hidden_size) positional vectors
                 the 1 in dim 0 will broadcast across the batch
        """
        positions = torch.arange(seq_len, device=device)  # [0, 1, 2, ..., 71]
        return self.embedding(positions).unsqueeze(0)      # (1, seq_len, hidden_size)
