import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy.embeddings import ConditionEmbedding


class FiLMGenerator(nn.Module):
    """Generates scale (gamma) and shift (beta) parameters from condition embedding.

    Maps condition vector c of shape (Batch_Size, cond_dim) to gamma and beta
    vectors of shape (Batch_Size, out_channels).
    """

    def __init__(self, cond_dim: int = 256, out_channels: int = 128):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * out_channels)

    def forward(self, condition: torch.Tensor):
        """Generates gamma and beta modulation parameters.

        Args:
            condition: Tensor of shape (Batch_Size, cond_dim)

        Returns:
            gamma: Tensor of shape (Batch_Size, out_channels, 1)
            beta: Tensor of shape (Batch_Size, out_channels, 1)
        """
        # Linear projection to 2 * out_channels
        scale_shift = self.proj(condition)  # (Batch_Size, 2 * out_channels)

        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(scale_shift, 2, dim=-1)

        # Unsqueeze temporal length dimension for broadcasting over (Batch_Size, Channels, Length)
        gamma = gamma.unsqueeze(-1)  # (Batch_Size, out_channels, 1)
        beta = beta.unsqueeze(-1)  # (Batch_Size, out_channels, 1)

        return gamma, beta


class FiLM1DResBlock(nn.Module):
    """1D Residual Convolution Block with Feature-wise Linear Modulation (FiLM).

    Applies Conv1d -> GroupNorm -> FiLM Modulation (gamma * x + beta) -> Mish -> Conv1d -> Skip Addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 256,
        kernel_size: int = 5,
        num_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2  # Keep temporal length T_pred constant

        # 1. First Conv + Norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # FiLM parameter generator
        self.film_gen = FiLMGenerator(cond_dim=cond_dim, out_channels=out_channels)

        # Activation
        self.act = nn.Mish()

        # 2. Second Conv + Norm
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # Residual skip connection projection if channel dimensions mismatch
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass for FiLM 1D ResBlock.

        Args:
            x: Action trajectory feature map of shape (Batch_Size, in_channels, T_pred)
            condition: Condition embedding of shape (Batch_Size, cond_dim)

        Returns:
            Output feature map of shape (Batch_Size, out_channels, T_pred).
        """
        residual = self.residual_proj(x)

        # 1. First Conv & GroupNorm
        h = self.conv1(x)
        h = self.norm1(h)

        # 2. Apply FiLM Modulation: gamma * h + beta
        gamma, beta = self.film_gen(condition)
        h = (1.0 + gamma) * h + beta  # Residual scale format (1 + gamma) for training stability

        # 3. Activationm
        h = self.act(h)

        # 4. Second Conv & GroupNorm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        # 5. Skip connection addition
        return h + residual


class FlowMatching1DCNN(nn.Module):
    """1D ResNet FiLM Network for Flow Matching Policy.

    Predicts velocity field v_theta(x_t, t, c) of shape (Batch_Size, 4, T_pred) given:
      - x_t: Noisy action trajectory of shape (Batch_Size, 4, T_pred)
      - state: Observation state of shape (Batch_Size, 34)
      - t: Continuous ODE time step scalar in [0, 1] of shape (Batch_Size,) or (Batch_Size, 1)
    """

    def __init__(
        self,
        action_dim: int = 4,
        state_dim: int = 34,
        cond_dim: int = 256,
        channels: list = [128, 256, 128],
        kernel_size: int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 1. Condition Embedding (State + Time)
        self.cond_embedding = ConditionEmbedding(state_dim=state_dim, frequency_dim=128, output_dim=cond_dim)

        # 2. Input Action Projection: (B, 4, T_pred) -> (B, channels[0], T_pred)
        self.in_proj = nn.Conv1d(action_dim, channels[0], kernel_size=kernel_size, padding=kernel_size // 2)

        # 3. FiLM 1D ResNet Backbone
        self.blocks = nn.ModuleList()
        in_c = channels[0]
        for out_c in channels:
            self.blocks.append(
                FiLM1DResBlock(in_channels=in_c, out_channels=out_c, cond_dim=cond_dim, kernel_size=kernel_size)
            )
            in_c = out_c

        # 4. Output Velocity Projection: (B, channels[-1], T_pred) -> (B, 4, T_pred)
        self.out_proj = nn.Conv1d(channels[-1], action_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        # Initialize final conv weights near zero so initial vector field predictions are stable
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting velocity vector field.

        Args:
            x_t: Noisy action sample of shape (Batch_Size, 4, T_pred) or (Batch_Size, T_pred, 4)
            state: Observation state of shape (Batch_Size, 34)
            t: Flow step scalar in [0, 1] of shape (Batch_Size,) or (Batch_Size, 1)

        Returns:
            Velocity prediction v_theta of shape (Batch_Size, 4, T_pred).
        """
        # Ensure x_t channel layout is (Batch_Size, Channels=4, Length=T_pred)
        if x_t.shape[1] != self.action_dim and x_t.shape[2] == self.action_dim:
            x_t = x_t.transpose(1, 2)  # Convert (B, T_pred, 4) -> (B, 4, T_pred)

        # 1. Compute unified condition embedding c
        c = self.cond_embedding(state, t)  # (Batch_Size, cond_dim)

        # 2. Input projection
        h = self.in_proj(x_t)  # (Batch_Size, channels[0], T_pred)

        # 3. FiLM ResNet blocks
        for block in self.blocks:
            h = block(h, c)

        # 4. Output velocity field projection
        v_theta = self.out_proj(h)  # (Batch_Size, 4, T_pred)
        return v_theta


if __name__ == "__main__":
    # Unit tests and shape assertions for 1D CNN FiLM Backbone
    model = FlowMatching1DCNN(action_dim=4, state_dim=34, cond_dim=256)

    batch_size = 4
    dummy_xt = torch.randn(batch_size, 4, 16)  # Noisy action sample
    dummy_state = torch.randn(batch_size, 34)  # 34D observation state
    dummy_t = torch.rand(batch_size, 1)  # Flow time step scalar in [0, 1]

    velocity_pred = model(dummy_xt, dummy_state, dummy_t)

    print(f"Input x_t shape: {dummy_xt.shape}")
    print(f"Output v_theta shape: {velocity_pred.shape}")

    assert velocity_pred.shape == (batch_size, 4, 16), f"Expected (4, 4, 16), got {velocity_pred.shape}"
    assert not torch.isnan(velocity_pred).any(), "Predicted velocity contains NaN!"

    print("[SUCCESS] policy/cnn_1d.py FlowMatching1DCNN test passed successfully!")
