"""Offline Reward Validation & Diagnostic Profiler.

Tests and verifies reward function integrity by profiling:
  1. Real clean expert trajectories from data/raw_all
  2. Synthetic stalling / hovering policy trajectories
  3. Synthetic wall collision trajectories

Ensures expert return exceeds bad policies by a large safety margin without needing training or eval.
"""

import glob
import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env_utils import make_franka_env
from ppo.reward_utils import RichRewardFrankaWrapper


def run_reward_validation(raw_dir: str = "data/raw_all"):
    print("=" * 60)
    print("=== Offline Reward Validation & Trajectory Profiling ===")
    print("=" * 60 + "\n")

    env = make_franka_env()
    wrapped_env = RichRewardFrankaWrapper(env)

    # 1. Profile Clean Expert Trajectory
    clean_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    expert_total = 0.0
    expert_steps = 0

    if clean_files:
        f_clean = clean_files[0]
        data = np.load(f_clean, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]

        obs, info = wrapped_env.reset()
        expert_rewards = []

        for t in range(len(states)):
            act = actions[t]
            r = wrapped_env.compute_rich_reward(obs, act, info)
            expert_rewards.append(r)

            obs, _, term, trunc, info = wrapped_env.step(act)
            if term or trunc:
                break

        expert_total = sum(expert_rewards)
        expert_steps = len(expert_rewards)
        print(f"1. Clean Expert Trajectory Return ({expert_steps} steps): {expert_total:.2f}")
    else:
        print(f"[Warning] No .npz files found in {raw_dir}. Skipping expert replay.")

    # 2. Profile Synthetic Hovering / Stalling Policy
    obs, info = wrapped_env.reset()
    hover_rewards = []
    hover_act = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # Closed grip, zero motion

    for t in range(300):
        r = wrapped_env.compute_rich_reward(obs, hover_act, info)
        hover_rewards.append(r)
        obs, _, term, trunc, info = wrapped_env.step(hover_act)
        if term or trunc:
            break

    hover_total = sum(hover_rewards)
    print(f"2. Synthetic Hovering Policy Return ({len(hover_rewards)} steps): {hover_total:.2f}")

    # 3. Profile Synthetic Wall Collision Policy
    obs, info = wrapped_env.reset()
    wall_rewards = []
    wall_act = np.array([0.1, 0.0, 0.0, -1.0], dtype=np.float32)  # Drive into wall

    for t in range(300):
        r = wrapped_env.compute_rich_reward(obs, wall_act, info)
        wall_rewards.append(r)
        obs, _, term, trunc, info = wrapped_env.step(wall_act)
        if term or trunc:
            break

    wall_total = sum(wall_rewards)
    print(f"3. Synthetic Wall Collision Policy Return ({len(wall_rewards)} steps): {wall_total:.2f}\n")

    # 4. Diagnostic Safety Margins
    print("=" * 60)
    print("=== Validation Safety Margins ===")
    print("=" * 60)
    margin_hover = expert_total - hover_total
    margin_wall = expert_total - wall_total

    print(f"• Expert Margin vs Hovering:   +{margin_hover:.2f} pts")
    print(f"• Expert Margin vs Wall Hits:  +{margin_wall:.2f} pts\n")

    assert expert_total > hover_total + 150, "EXPLOIT RISK: Expert return is not significantly higher than hovering!"
    assert expert_total > wall_total + 150, "EXPLOIT RISK: Expert return is not significantly higher than wall hits!"

    print("[SUCCESS] Reward function passed offline validation with healthy safety margins!\n")
    env.close()


if __name__ == "__main__":
    run_reward_validation()
