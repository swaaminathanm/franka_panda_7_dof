import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy.cnn_1d import FlowMatching1DCNN
from policy.normalizer import LinearNormalizer


class FlowMatchingPolicy(nn.Module):
    """Optimal Transport Flow Matching (OT-CFM) Policy with 1D CNN Backbone.

    Implements:
      1. Training Loss: Rectified Flow Velocity Matching Loss (MSE between v_theta and x_1 - x_0).
      2. Inference Sampling: Euler ODE integration solver generating action trajectory chunks.
    """

    def __init__(
        self,
        action_dim: int = 4,
        state_dim: int = 30,
        pred_horizon: int = 16,
        cond_dim: int = 256,
        stats_path: str = "data/lerobot/meta/stats.json",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.pred_horizon = pred_horizon

        # 1. Backbone Vector Field Network
        self.model = FlowMatching1DCNN(
            action_dim=action_dim,
            state_dim=state_dim,
            cond_dim=cond_dim,
        )

        # 2. Linear Normalizer
        self.normalizer = LinearNormalizer(state_dim=state_dim, action_dim=action_dim)
        if os.path.exists(stats_path):
            self.normalizer.load_from_stats_json(stats_path)

    def compute_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Computes OT-CFM Flow Matching MSE Training Loss.

        Args:
            state: Raw observation state tensor of shape (Batch_Size, 30)
            action: Raw ground truth action chunk tensor of shape (Batch_Size, 16, 4)

        Returns:
            Scalar MSE loss tensor.
        """
        # 1. Normalize state and target action trajectory to [-1, 1]
        norm_state = self.normalizer.normalize_state(state)  # (Batch_Size, 30)
        x_1 = self.normalizer.normalize_action(action)  # (Batch_Size, 16, 4)

        # Ensure x_1 has shape (Batch_Size, 4, 16) for 1D CNN layout
        if x_1.shape[1] != self.action_dim and x_1.shape[2] == self.action_dim:
            x_1 = x_1.transpose(1, 2)  # (Batch_Size, 4, 16)

        batch_size = x_1.shape[0]
        device = x_1.device

        # 2. Sample continuous ODE time t ~ Uniform(0, 1)
        t = torch.rand(batch_size, 1, device=device)  # (Batch_Size, 1)

        # 3. Sample starting Gaussian noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)  # (Batch_Size, 4, 16)

        # 4. Straight-line optimal transport flow path: x_t = (1 - t) * x_0 + t * x_1
        # Expand t for 3D broadcasting over (Batch_Size, 4, 16)
        t_broadcast = t.unsqueeze(-1)  # (Batch_Size, 1, 1)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * x_1  # (Batch_Size, 4, 16)

        # 5. Target velocity field vector: u_t = dx_t / dt = x_1 - x_0
        target_velocity = x_1 - x_0  # (Batch_Size, 4, 16)

        # 6. Predict velocity field v_theta(x_t, norm_state, t)
        v_theta = self.model(x_t, norm_state, t)  # (Batch_Size, 4, 16)

        # 7. Compute Mean Squared Error Loss
        loss = F.mse_loss(v_theta, target_velocity)
        return loss

    @torch.no_grad()
    def sample_actions(self, state: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Inference ODE Euler solver generating action trajectory chunks.

        Args:
            state: Raw observation state of shape (30,) or (Batch_Size, 30)
            num_steps: Number of Euler integration steps (default: 10)

        Returns:
            Unnormalized predicted action chunk trajectory of shape (Batch_Size, 16, 4).
        """
        # Ensure batch dimension
        is_single = False
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, 30)
            is_single = True

        batch_size = state.shape[0]
        device = state.device

        # 1. Normalize observation state
        norm_state = self.normalizer.normalize_state(state)  # (Batch_Size, 30)

        # 2. Sample initial noise x_0 ~ N(0, I)
        x_t = torch.randn(batch_size, self.action_dim, self.pred_horizon, device=device)  # (Batch_Size, 4, 16)

        # 3. Euler ODE Integration over t in [0, 1]
        dt = 1.0 / float(num_steps)

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((batch_size, 1), t_val, device=device, dtype=torch.float32)

            # Predict velocity field v_theta
            v_pred = self.model(x_t, norm_state, t)  # (Batch_Size, 4, 16)

            # Euler step update: x_(t + dt) = x_t + dt * v_pred
            x_t = x_t + dt * v_pred

        # 4. Transpose back to (Batch_Size, 16, 4) layout
        x_1 = x_t.transpose(1, 2)  # (Batch_Size, 16, 4)

        # 5. Unnormalize predicted action trajectory back to physical simulation units
        raw_actions = self.normalizer.unnormalize_action(x_1)  # (Batch_Size, 16, 4)

        if is_single:
            raw_actions = raw_actions.squeeze(0)  # (16, 4)

        return raw_actions


if __name__ == "__main__":
    # Unit tests for FlowMatchingPolicy
    pred_horizon = 16

    policy = FlowMatchingPolicy(action_dim=4, state_dim=30, pred_horizon=pred_horizon, cond_dim=256)

    batch_size = 4
    dummy_state = torch.randn(batch_size, 30)
    dummy_action = torch.randn(batch_size, pred_horizon, 4)

    # 1. Test Training Loss Computation
    loss = policy.compute_loss(dummy_state, dummy_action)
    print(f"Training Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"

    # 2. Test Inference Action Sampling (10-step Euler ODE)
    sampled_actions = policy.sample_actions(dummy_state, num_steps=10)
    print(f"Sampled Actions shape: {sampled_actions.shape}")
    assert sampled_actions.shape == (batch_size, 16, 4), f"Expected (4, 16, 4), got {sampled_actions.shape}"
    assert not torch.isnan(sampled_actions).any(), "Sampled actions contain NaN!"

    # 3. Test Single Vector Input (30,)
    single_state = torch.randn(30)
    single_actions = policy.sample_actions(single_state, num_steps=10)
    print(f"Single State Sampled Actions shape: {single_actions.shape}")
    assert single_actions.shape == (16, 4), f"Expected (16, 4), got {single_actions.shape}"

    print("[SUCCESS] policy/flow_matching.py FlowMatchingPolicy test passed successfully!")
