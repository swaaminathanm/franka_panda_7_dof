import torch
import torch.nn as nn
from diffusers.models.embeddings import get_timestep_embedding


class SinusoidalPositionalEncoding(nn.Module):
    """Continuous ODE step time encoder for Flow Matching using diffusers get_timestep_embedding.

    Maps scalar time t in range [0, 1] to a 256D continuous condition embedding.
    """

    def __init__(self, frequency_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.output_dim = output_dim

        # 2-layer MLP to project frequency embedding to output condition space
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for continuous time embedding.

        Args:
            t: Tensor of shape (Batch_Size,) or (Batch_Size, 1) containing float time values in [0, 1].

        Returns:
            Time embedding tensor of shape (Batch_Size, output_dim).
        """
        if t.dim() > 1:
            t = t.squeeze(-1)

        # Compute continuous sinusoidal timestep embedding using diffusers utility
        freq_emb = get_timestep_embedding(t * 1000.0, embedding_dim=self.frequency_dim)  # (Batch_Size, frequency_dim)

        # Project through 2-layer MLP to condition dimension
        time_emb = self.mlp(freq_emb)  # (Batch_Size, output_dim)
        return time_emb


class StateEncoder(nn.Module):
    """Encoder mapping 34D observation states to 256D condition embeddings.

    Uses a 3-layer MLP with Mish activation functions.
    """

    def __init__(self, state_dim: int = 34, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass for state embedding.

        Args:
            state: Tensor of shape (Batch_Size, 34)

        Returns:
            State embedding tensor of shape (Batch_Size, output_dim).
        """
        return self.mlp(state)


class ConditionEmbedding(nn.Module):
    """Unified Condition Embedding module combining StateEncoder and SinusoidalPositionalEncoding.

    Outputs condition vector c = state_embedding + time_embedding of shape (Batch_Size, output_dim).
    """

    def __init__(self, state_dim: int = 34, frequency_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim=state_dim, hidden_dim=output_dim, output_dim=output_dim)
        self.time_encoder = SinusoidalPositionalEncoding(frequency_dim=frequency_dim, output_dim=output_dim)

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass combining state and time embeddings.

        Args:
            state: Tensor of shape (Batch_Size, 34)
            t: Tensor of shape (Batch_Size,) or (Batch_Size, 1) in range [0, 1]

        Returns:
            Condition embedding tensor c of shape (Batch_Size, output_dim).
        """
        state_emb = self.state_encoder(state)
        time_emb = self.time_encoder(t)
        condition_emb = state_emb + time_emb
        return condition_emb


if __name__ == "__main__":
    # Unit tests and shape verification
    cond_embedding = ConditionEmbedding(state_dim=34, frequency_dim=128, output_dim=256)

    batch_size = 4
    dummy_t = torch.rand(batch_size, 1)  # Continuous ODE steps in [0, 1]
    dummy_state = torch.randn(batch_size, 34)  # 34D observation vector

    condition_emb = cond_embedding(dummy_state, dummy_t)

    print(f"Condition embedding shape: {condition_emb.shape}")
    assert condition_emb.shape == (batch_size, 256), f"Expected (4, 256), got {condition_emb.shape}"
    assert not torch.isnan(condition_emb).any(), "Condition embedding contains NaN!"

    print("[SUCCESS] policy/embeddings.py ConditionEmbedding test passed successfully!")
