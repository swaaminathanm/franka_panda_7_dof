# Implementation Plan: Residual Flow-PPO with Rich Rewards

This document outlines the architecture, reward design, policy definitions, training loop, and evaluation pipeline for **Residual Flow-PPO**.

---

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   30D State Vector (obs)    |
                      +--------------+--------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
+-----------------------+                         +-----------------------+
|  Flow Matching Policy |                         | Residual Flow-PPO     |
|   (FROZEN Expert)     |                         | Actor (TRAINABLE)     |
+----------+------------+                         +-----------+-----------+
           |                                                   |
           v Base Action Chunk (a_base)                        v Residual Offset Chunk (delta_a)
           +-------------------------+-------------------------+
                                     |
                                     v
                  a_exec_chunk = Clip(a_base + delta_a)
                                     |
                                     v
                       +---------------------------+
                       | Rich Reward Franka Env    |
                       | (Obstacle Avoidance,      |
                       |  Reaching, Grasping, Goal)|
                       +---------------------------+
```

---

## 2. Environment Reward Wrapper: `reward_utils.py`

```python
import numpy as np
import gymnasium as gym


class RichRewardFrankaWrapper(gym.Wrapper):
    """Gymnasium Wrapper augmenting PandaPickAndPlace-v3 with a multi-phase dense reward signal.

    Phases & Reward Terms:
      1. Reaching: -dist(gripper, object)
      2. Grasping & Lifting: +3.0 when object in hand (dist < 3.5cm and z_obj > 0.02m)
      3. Placement Goal Progress: -2.0 * dist(object, goal) when object in hand
      4. Obstacle Collision Penalty: -20.0 if PyBullet detects contact with partition wall
      5. Task Success Bonus: +100.0 if dist(object, goal) < 0.05m
      6. Smooth Action Effort Penalty: -0.01 * sum(action^2)
    """

    def __init__(self, env, obstacle_id: int = None):
        super().__init__(env)
        self.obstacle_id = obstacle_id

    def set_obstacle_id(self, obstacle_id: int):
        self.obstacle_id = obstacle_id

    def compute_rich_reward(self, obs, action, info) -> float:
        sim = self.env.unwrapped.sim
        bullet_p = sim.physics_client

        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"][:3]
        goal_pos = obs["desired_goal"][:3]

        dist_ee_obj = float(np.linalg.norm(obj_pos - ee_pos))
        dist_obj_goal = float(np.linalg.norm(goal_pos - obj_pos))

        # 1. Reaching reward
        r_reach = -dist_ee_obj

        # 2. Grasping & Lifting reward
        is_grasped = (dist_ee_obj < 0.035) and (obj_pos[2] > 0.02)
        r_grasp = 3.0 if is_grasped else 0.0

        # 3. Placement goal progress (active when object is in hand)
        r_place = -2.0 * dist_obj_goal if is_grasped else 0.0

        # 4. Obstacle collision penalty
        p_collision = 0.0
        if self.obstacle_id is not None:
            contacts = bullet_p.getContactPoints(bodyA=self.obstacle_id)
            if len(contacts) > 0:
                p_collision = -20.0

        # 5. Task success bonus
        is_success = bool(info.get("is_success", False)) or (dist_obj_goal < 0.05)
        r_success = 100.0 if is_success else 0.0

        # 6. Action effort penalty
        p_effort = -0.01 * float(np.sum(np.square(action)))

        total_reward = r_reach + r_grasp + r_place + p_collision + r_success + p_effort
        return float(total_reward)

    def step(self, action):
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        rich_reward = self.compute_rich_reward(obs, action, info)
        return obs, rich_reward, terminated, truncated, info
```

---

## 3. Policy Definitions: `policy/residual_ppo.py`

```python
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
        self.log_std = nn.Parameter(torch.zeros(pred_horizon, action_dim))

    def forward(self, a_base: torch.Tensor, state: torch.Tensor):
        """Predicts 16-step residual action chunk delta_a given base trajectory and state."""
        t_eval = torch.ones(state.shape[0], device=state.device)

        # FlowMatching1DCNN takes (a_base, state, t) and returns (Batch, 4, 16)
        res_features = self.cnn_backbone(a_base, state, t_eval)

        # Transpose from (Batch, 4, 16) -> (Batch, 16, 4)
        mu = res_features.transpose(1, 2)

        # Bound residual offsets to safe range [-0.2, +0.2]
        mu = torch.tanh(mu) * 0.2

        std = torch.exp(self.log_std)
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
    """Unified Policy combining Frozen Flow Matching Expert + Trainable Residual Flow-PPO Actor."""

    def __init__(self, flow_policy, residual_actor, critic):
        super().__init__()
        # 1. Frozen Expert Base Policy (FlowMatchingPolicy)
        self.flow_policy = flow_policy
        self.flow_policy.eval()
        for p in self.flow_policy.parameters():
            p.requires_grad = False

        # 2. Trainable Residual PPO Actor (ResidualFlowMatchingActor)
        self.residual_actor = residual_actor

        # 3. Trainable Value Critic (ValueCritic)
        self.critic = critic

    def get_action(self, state_tensor: torch.Tensor, deterministic: bool = False):
        """Generates a 16-step combined action chunk and PPO telemetry."""
        # Step 1: Query Frozen Base Flow Policy for 16-step expert action chunk
        with torch.no_grad():
            a_base = self.flow_policy.sample_actions(state_tensor, num_steps=10)  # (1, 16, 4)

        # Step 2: Query Residual Actor for residual mean offset chunk and std
        mu, std = self.residual_actor(a_base, state_tensor)  # (1, 16, 4), (16, 4)
        dist = Normal(mu, std)

        if deterministic:
            delta_a = mu
        else:
            delta_a = dist.sample()  # (1, 16, 4)

        # Sum log probs across 16 timesteps and 4 action dimensions
        log_prob = dist.log_prob(delta_a).sum(dim=(-2, -1))  # (1,)

        # Step 3: Combine base action chunk + residual offset chunk
        a_exec_chunk = torch.clamp(a_base + delta_a, min=-1.0, max=1.0)  # (1, 16, 4)

        # Step 4: Query Value Critic
        v_value = self.critic(state_tensor)  # (1, 1)

        return a_exec_chunk, a_base, delta_a, log_prob, v_value
```

---

## 4. Training Loop: `train_flow_ppo.py`

```python
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env_utils import make_franka_env, setup_scene_physics, extract_state
from reward_utils import RichRewardFrankaWrapper
from policy.flow_matching import FlowMatchingPolicy
from policy.residual_ppo import ResidualFlowMatchingActor, ValueCritic, ResidualFlowPolicy


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0.0
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def train_ppo(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[PPO Train] Using device: {device}")

    # 1. Load Pre-trained Flow Matching Policy
    flow_policy = FlowMatchingPolicy(
        action_dim=4,
        state_dim=30,
        pred_horizon=16,
        cond_dim=args.cond_dim,
        stats_path=os.path.join(args.data_dir, "meta", "stats.json"),
    ).to(device)

    if os.path.exists(args.flow_checkpoint):
        ckpt = torch.load(args.flow_checkpoint, map_location=device)
        flow_policy.load_state_dict(ckpt["model_state_dict"])
        print(f"[PPO Train] Loaded base Flow Matching policy from: {args.flow_checkpoint}")
    else:
        print(f"[PPO Train] Warning: Flow checkpoint {args.flow_checkpoint} not found!")

    # 2. Setup Residual Actor & Critic
    residual_actor = ResidualFlowMatchingActor(action_dim=4, state_dim=30, cond_dim=args.cond_dim).to(device)
    critic = ValueCritic(state_dim=30, cond_dim=args.cond_dim).to(device)
    residual_policy = ResidualFlowPolicy(flow_policy, residual_actor, critic)

    optimizer = optim.AdamW(
        list(residual_actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # 3. Create Gym Environment with Rich Reward
    base_env = make_franka_env(max_episode_steps=args.max_steps)
    env = RichRewardFrankaWrapper(base_env)

    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")
    start_time = time.time()

    print(f"\n[PPO Train] Starting Residual Flow-PPO Training ({args.episodes} episodes)...")
    print(f"            Learning Rate: {args.lr} | K_exec: {args.k_exec}\n")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        obstacle_id = setup_scene_physics(env)
        env.set_obstacle_id(obstacle_id)

        # Episode rollout buffers
        states_b, a_base_b, delta_a_b, log_probs_b, rewards_b, values_b, dones_b = [], [], [], [], [], [], []
        action_buffer = []

        ep_step = 0
        ep_reward = 0.0

        # Step A: Rollout collection using current updated model
        while ep_step < args.max_steps:
            state_vec = extract_state(obs)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)

            if len(action_buffer) == 0:
                with torch.no_grad():
                    a_exec_chunk, a_base_chunk, delta_a_chunk, log_prob, v_val = residual_policy.get_action(state_tensor)
                    a_exec_chunk = a_exec_chunk.squeeze(0).cpu().numpy()  # (16, 4)
                    a_base_chunk = a_base_chunk.squeeze(0).cpu().numpy()  # (16, 4)
                    delta_a_chunk = delta_a_chunk.squeeze(0).cpu().numpy()  # (16, 4)

                action_buffer = list(a_exec_chunk[: args.k_exec])
                step_a_base = a_base_chunk[0]
                step_delta_a = delta_a_chunk[0]
                step_log_prob = log_prob.item()
                step_v_val = v_val.item()

            action = action_buffer.pop(0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_step += 1
            ep_reward += reward

            states_b.append(state_vec)
            a_base_b.append(step_a_base)
            delta_a_b.append(step_delta_a)
            log_probs_b.append(step_log_prob)
            rewards_b.append(reward)
            values_b.append(step_v_val)
            dones_b.append(done)

            obs = next_obs
            if done:
                break

        # Step B: Compute GAE Advantages
        last_state = extract_state(obs)
        with torch.no_grad():
            next_value = critic(torch.from_numpy(last_state).unsqueeze(0).to(device)).item()

        advantages, returns = compute_gae(rewards_b, values_b, dones_b, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step C: PPO Gradient Optimization (Mutates residual_actor & critic weights in-place)
        t_states = torch.from_numpy(np.array(states_b)).to(device)
        t_a_base = torch.from_numpy(np.array(a_base_b)).to(device)
        t_delta_a = torch.from_numpy(np.array(delta_a_b)).to(device)
        t_old_log_probs = torch.tensor(log_probs_b, dtype=torch.float32).to(device)
        t_advantages = torch.from_numpy(advantages).to(device)
        t_returns = torch.from_numpy(returns).to(device)

        for ppo_epoch in range(args.ppo_epochs):
            mu, std = residual_actor(t_a_base.unsqueeze(1), t_states)  # (N, 16, 4)
            dist = torch.distributions.Normal(mu, std)

            t_delta_expand = t_delta_a.unsqueeze(1).expand(-1, 16, -1)
            new_log_probs = dist.log_prob(t_delta_expand).sum(dim=(-2, -1))
            entropy = dist.entropy().sum(dim=(-2, -1)).mean()

            ratios = torch.exp(new_log_probs - t_old_log_probs)
            surr1 = ratios * t_advantages
            surr2 = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * t_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            v_pred = critic(t_states).squeeze(-1)
            critic_loss = torch.mean((v_pred - t_returns) ** 2)

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(residual_actor.parameters()) + list(critic.parameters()), 0.5)
            optimizer.step()  # In-place parameter mutation

        print(f"Episode [{ep:3d}/{args.episodes:3d}] | Total Reward: {ep_reward:7.2f} | Steps: {ep_step:3d}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_ckpt = os.path.join(args.save_dir, "residual_ppo_best.pt")
            torch.save({"actor": residual_actor.state_dict(), "critic": critic.state_dict()}, best_ckpt)
            print(f"   [Checkpoint] New best PPO model saved -> {best_ckpt}")

    print(f"\n[PPO Train] Completed in {time.time() - start_time:.1f}s!")


def main():
    parser = argparse.ArgumentParser(description="Train Residual Flow-PPO Policy")
    parser.add_argument("--flow-checkpoint", type=str, default="checkpoints/flow_policy_best.pt")
    parser.add_argument("--data-dir", type=str, default="data/lerobot")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k-exec", type=int, default=4)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(args)


if __name__ == "__main__":
    main()
```

---

## 5. Verification Plan

1. **Unit Test**: Test `reward_utils.py` and `policy/residual_ppo.py`.
2. **Train**: Run `python train_flow_ppo.py --episodes 50`.
3. **Eval**: Evaluate with `python eval_flow_ppo.py --episodes 10 --save-video`.
