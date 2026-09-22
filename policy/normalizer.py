import json
import os
import torch
import torch.nn as nn


class LinearNormalizer(nn.Module):
    """Linear Min-Max Normalizer for Flow Matching Policy.

    Normalizes 34D observation states and 4D actions to [-1, 1] range.
    Supports loading statistics directly from LeRobot metadata (stats.json).
    Min/Max statistics are registered as PyTorch buffers so they automatically
    move with the model to CPU/GPU and are saved in state_dict checkpoints.
    """

    def __init__(self, state_dim: int = 34, action_dim: int = 4, eps: float = 1e-8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = eps

        # Register persistent buffers (saved in state_dict, moved to device)
        self.register_buffer("state_min", torch.zeros(state_dim))
        self.register_buffer("state_max", torch.ones(state_dim))
        self.register_buffer("action_min", torch.zeros(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))

        self.is_fit = False

    def load_from_stats_json(self, stats_json_path: str = "data/lerobot/meta/stats.json"):
        """Loads min/max statistics directly from LeRobot dataset stats.json."""
        if not os.path.exists(stats_json_path):
            raise FileNotFoundError(f"Stats file not found at: {stats_json_path}")

        with open(stats_json_path, "r") as f:
            stats = json.load(f)

        s_min = torch.tensor(stats["observation.state"]["min"], dtype=torch.float32)
        s_max = torch.tensor(stats["observation.state"]["max"], dtype=torch.float32)
        a_min = torch.tensor(stats["action"]["min"], dtype=torch.float32)
        a_max = torch.tensor(stats["action"]["max"], dtype=torch.float32)

        self._set_stats(s_min, s_max, a_min, a_max)
        print(f"[Normalizer] Loaded statistics from: {stats_json_path}")

    def _set_stats(self, s_min: torch.Tensor, s_max: torch.Tensor, a_min: torch.Tensor, a_max: torch.Tensor):
        """Validates zero-range safety and copies statistics into PyTorch buffers."""
        range_s = s_max - s_min
        s_max = torch.where(range_s < self.eps, s_min + 1.0, s_max)

        range_a = a_max - a_min
        a_max = torch.where(range_a < self.eps, a_min + 1.0, a_max)

        self.state_min.copy_(s_min)
        self.state_max.copy_(s_max)
        self.action_min.copy_(a_min)
        self.action_max.copy_(a_max)

        self.is_fit = True

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalizes state tensor from raw physical units to [-1, 1]."""
        state_min = self.state_min.to(state.device)
        state_max = self.state_max.to(state.device)
        return 2.0 * (state - state_min) / (state_max - state_min + self.eps) - 1.0

    def unnormalize_state(self, norm_state: torch.Tensor) -> torch.Tensor:
        """Unnormalizes state tensor from [-1, 1] back to raw physical units."""
        state_min = self.state_min.to(norm_state.device)
        state_max = self.state_max.to(norm_state.device)
        return 0.5 * (norm_state + 1.0) * (state_max - state_min + self.eps) + state_min

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalizes action tensor from physical units to [-1, 1]."""
        action_min = self.action_min.to(action.device)
        action_max = self.action_max.to(action.device)
        return 2.0 * (action - action_min) / (action_max - action_min + self.eps) - 1.0

    def unnormalize_action(self, norm_action: torch.Tensor) -> torch.Tensor:
        """Unnormalizes action tensor from [-1, 1] back to physical simulation units."""
        action_min = self.action_min.to(norm_action.device)
        action_max = self.action_max.to(norm_action.device)
        return 0.5 * (norm_action + 1.0) * (action_max - action_min + self.eps) + action_min

    def save(self, filepath: str):
        """Saves normalizer state dict independently."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        """Loads normalizer state dict."""
        self.load_state_dict(torch.load(filepath))
        self.is_fit = True


if __name__ == "__main__":
    normalizer = LinearNormalizer(state_dim=34, action_dim=4)
    stats_path = "data/lerobot_all/meta/stats.json"

    if os.path.exists(stats_path):
        normalizer.load_from_stats_json(stats_path)
        dummy_state = torch.randn(2, 34)
        dummy_action = torch.randn(2, 16, 4)

        norm_s = normalizer.normalize_state(dummy_state)
        rec_s = normalizer.unnormalize_state(norm_s)

        norm_a = normalizer.normalize_action(dummy_action)
        rec_a = normalizer.unnormalize_action(norm_a)

        assert torch.allclose(dummy_state, rec_s, atol=1e-4)
        assert torch.allclose(dummy_action, rec_a, atol=1e-4)
        print("[SUCCESS] LinearNormalizer test passed successfully!")
    else:
        print(f"Stats path {stats_path} not found for testing.")
