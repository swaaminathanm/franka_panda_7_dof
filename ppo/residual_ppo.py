import os
import sys
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy.cnn_1d import FlowMatching1DCNN
from policy.embeddings import StateEncoder


class ResidualFlowMatchingActor(nn.Module):
    """Trainable PPO Residual Actor using FlowMatching1DCNN backbone architecture."""

    def __init__(self, action_dim: int = 4, state_dim: int = 30, cond_dim: int = 256, pred_horizon: int = 16):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # 1. FlowMatching1DCNN backbone network (trainable for residual learning)
        self.cnn_backbone = FlowMatching1DCNN(
            action_dim=action_dim,
            state_dim=state_dim,
            cond_dim=cond_dim,
        )

        # 2. Trainable log std matrix for PPO Gaussian exploration over 16 steps
        # Initialized to -2.3 so initial exploration std is exp(-2.3) ~ 0.1
        self.log_std = nn.Parameter(torch.full((pred_horizon, action_dim), -2.3))

    def forward(self, a_base: torch.Tensor, state: torch.Tensor):
        """
        Predicts 16-step residual action chunk delta_a given base trajectory and state.

        Args:
            a_base: Base action trajectory chunk of shape (Batch_Size, T, 4)
            state: Observation state of shape (Batch_Size, 30)

        Returns:
            mu: Predicted mean of residual action chunk of shape (Batch_Size, T, 4)
            std: Predicted standard deviation of residual action chunk of shape (T, 4)
        """
        t_eval = torch.ones(state.shape[0], device=state.device)

        # FlowMatching1DCNN takes (a_base, state, t) and returns (Batch, 4, T)
        res_features = self.cnn_backbone(a_base, state, t_eval)

        # Transpose from (Batch, 4, T) -> (Batch, T, 4)
        mu = res_features.transpose(1, 2)

        # Bound residual offsets to safe range [-0.2, +0.2]
        mu = torch.tanh(mu) * 0.2

        horizon = a_base.shape[1]

        std = torch.exp(self.log_std[:horizon])
        
        return mu, std


class ValueCritic(nn.Module):
    """Trainable State Value Critic V(s)."""

    def __init__(self, state_dim: int = 30, cond_dim: int = 256):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim=state_dim, output_dim=cond_dim)
        self.value_head = nn.Linear(cond_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.state_encoder(state)
        return self.value_head(feat)


class ResidualFlowPolicy(nn.Module):
    """Wrapper composing trained actor and critic into a unified policy."""

    def __init__(self, flow_policy, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

        # Frozen Expert Base Policy
        self.flow_policy = flow_policy
        self.flow_policy.eval()
        for p in self.flow_policy.parameters():
            p.requires_grad_(False)

    def get_action(self, state: torch.Tensor, deterministic: bool = False, k_exec: int = 16):
        """Generates 16-step combined action chunk and PPO telemetry."""
        # 1. Compute base action using expert frozen policy
        with torch.no_grad():
            a_base = self.flow_policy.sample_actions(state, num_steps=10)  # (1, 16, 4)

        a_base = a_base[:, :k_exec, :]  # (1, k_exec, 4)

        # 2. Compute residual offsets
        mu, std = self.actor(a_base, state)  # (1, k_exec, 4), (k_exec, 4)
        dist = Normal(mu, std)

        if deterministic:
            delta_a = mu
        else:
            delta_a = dist.sample()  # (1, k_exec, 4)

        # 3. Sum log probs across k_exec timesteps and 4 action dimensions
        log_prob = dist.log_prob(delta_a).sum(dim=(-2, -1))  # (1,)

        # 4. Combine base action + residual offset
        a_final = torch.clamp(a_base + delta_a, min=-1.0, max=1.0)  # (1, k_exec, 4)

        # 5. Query Critic
        v_value = self.critic(state)  # (1, 1)

        return a_final, a_base, delta_a, log_prob, v_value
