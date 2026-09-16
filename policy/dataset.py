import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy.normalizer import LinearNormalizer


class FrankaLeRobotDataset(Dataset):
    """PyTorch Dataset for Franka Panda Pick-and-Place Demonstration Trajectories.

    Loads demonstration data from LeRobot parquet format (or raw .npz files) and
    implements a sliding-window trajectory sampler with boundary clamping to achieve
    100% frame coverage across all episodes.
    """

    def __init__(
        self,
        data_dir: str = "data/lerobot",
        pred_horizon: int = 16,
        normalizer: LinearNormalizer = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.pred_horizon = pred_horizon
        self.normalizer = normalizer

        # 1. Load episode data into memory
        self.episodes = self._load_data(data_dir)
        print(f"[Dataset] Loaded {len(self.episodes)} episodes with total {sum(len(e['states']) for e in self.episodes)} frames.")

        # 2. Build full episode sliding-window sample map
        self.sample_indices = self._build_sample_indices()
        print(f"[Dataset] Constructed {len(self.sample_indices)} valid training samples (100% frame coverage).")

    def _load_data(self, data_dir: str):
        """Loads states and actions grouped by episode from Parquet files."""
        episodes = []

        # Load from LeRobot parquet files
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "data", "**", "*.parquet"), recursive=True))
        if parquet_files:
            dfs = [pd.read_parquet(pf) for pf in parquet_files]
            full_df = pd.concat(dfs, ignore_index=True)

            for ep_idx, group in full_df.groupby("episode_index"):
                states = np.stack(group["observation.state"].values).astype(np.float32)
                actions = np.stack(group["action"].values).astype(np.float32)
                episodes.append({"states": states, "actions": actions})
            return episodes

        raise FileNotFoundError(f"No dataset found in {data_dir}.")

    def _build_sample_indices(self):
        """Builds index map covering every single frame across all episodes with boundary clamping."""
        indices = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep["states"])
            for step in range(ep_len):
                indices.append((ep_idx, step))
        return indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        ep_idx, step = self.sample_indices[idx]
        ep = self.episodes[ep_idx]
        ep_len = len(ep["states"])

        # --- 1. Extract Current State (30,) ---
        state_seq = torch.from_numpy(ep["states"][step])  # (30,)

        # --- 2. Extract Action Target Trajectory with End-Clamping (T_pred, 4) ---
        action_indices = [min(ep_len - 1, step + i) for i in range(self.pred_horizon)]
        action_seq = torch.from_numpy(ep["actions"][action_indices])  # (T_pred, 4)

        # --- 3. Normalize if normalizer is provided ---
        if self.normalizer is not None:
            state_seq = self.normalizer.normalize_state(state_seq)
            action_seq = self.normalizer.normalize_action(action_seq)

        return {
            "state": state_seq,
            "action": action_seq,
        }


if __name__ == "__main__":
    # Test dataset loading and shape verification
    normalizer = LinearNormalizer(state_dim=30, action_dim=4)
    stats_path = "data/lerobot/meta/stats.json"
    if os.path.exists(stats_path):
        normalizer.load_from_stats_json(stats_path)

    dataset = FrankaLeRobotDataset(
        data_dir="data/lerobot",
        pred_horizon=16,
        normalizer=normalizer,
    )

    print(f"Total dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample 0 state shape: {sample['state'].shape}")
    print(f"Sample 0 action shape: {sample['action'].shape}")

    assert sample["state"].shape == (30,), f"Expected (30,), got {sample['state'].shape}"
    assert sample["action"].shape == (16, 4), f"Expected (16, 4), got {sample['action'].shape}"
    assert not torch.isnan(sample["state"]).any(), "State contains NaN!"
    assert not torch.isnan(sample["action"]).any(), "Action contains NaN!"

    # Test last sample boundary clamping
    last_sample = dataset[len(dataset) - 1]
    assert last_sample["action"].shape == (16, 4)

    print("[SUCCESS] FrankaLeRobotDataset test passed successfully!")
