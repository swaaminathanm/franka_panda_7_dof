import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Add workspace root and ppo directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env_utils import make_franka_env, setup_scene_physics, extract_state
from ppo.reward_utils import RichRewardFrankaWrapper
from policy.flow_matching import FlowMatchingPolicy
from ppo.residual_ppo import ResidualFlowMatchingActor, ValueCritic, ResidualFlowPolicy


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


def evaluate_policy(env, policy, episodes=5, max_steps=300, k_exec=4):
    """Evaluates policy deterministically across evaluation episodes."""
    total_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)
        obstacle_id = setup_scene_physics(env)
        env.set_obstacle_id(obstacle_id)
        ep_reward = 0.0
        action_buffer = []

        step = 0
        while step < max_steps:
            state_vec = extract_state(obs) # (30,)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0) # (1, 30)

            if len(action_buffer) == 0:
                with torch.no_grad():
                    a_final_chunk, _, _, _, _ = policy.get_action(state_tensor, deterministic=True, k_exec=k_exec)
                    a_final_chunk = a_final_chunk.squeeze(0).cpu().numpy() # (k_exec, 4)
                action_buffer = list(a_final_chunk)

            action = action_buffer.pop(0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))


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

    # 3. Initialize Optimizer
    optimizer = optim.AdamW(
        list(residual_actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # 4. Create Gym Environment with Rich Reward
    base_env = make_franka_env(max_episode_steps=args.max_steps)
    env = RichRewardFrankaWrapper(base_env)

    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")
    start_time = time.time()

    print(f"\n[PPO Train] Starting Chunk-Level Residual Flow-PPO Training ({args.episodes} episodes)...")
    print(f"            Learning Rate: {args.lr} | K_exec: {args.k_exec} | Horizon: 16\n")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        obstacle_id = setup_scene_physics(env)
        env.set_obstacle_id(obstacle_id)

        # Chunk-level Rollout Buffers
        states_b = []       # List of (30,) state vectors at chunk start
        a_base_b = []       # List of (16, 4) base trajectory chunks
        delta_a_b = []      # List of (16, 4) residual trajectory offset chunks
        log_probs_b = []    # List of scalar log probabilities for full (16, 4) chunks
        rewards_b = []      # List of cumulative rewards over K_exec execution steps
        values_b = []       # List of scalar value estimates V(s) at chunk start
        dones_b = []        # List of termination boolean flags

        ep_step = 0
        ep_reward = 0.0

        # Step A: Chunk-level Rollout Collection
        while ep_step < args.max_steps:
            state_vec = extract_state(obs)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)  # (1, 30)

            # Query Residual Flow Policy for full 16-step trajectory chunk
            with torch.no_grad():
                a_final_chunk, a_base_chunk, delta_a_chunk, log_prob, v_val = residual_policy.get_action(state_tensor, deterministic=False, k_exec=args.k_exec)
            
            a_final_np = a_final_chunk.squeeze(0).cpu().numpy()  # (k_exec, 4)
            a_base_np = a_base_chunk.squeeze(0).cpu().numpy()    # (k_exec, 4)
            delta_a_np = delta_a_chunk.squeeze(0).cpu().numpy()  # (k_exec, 4)

            action_queue = list(a_final_np)
                
            chunk_reward = 0.0
            chunk_executed = 0

            for action in action_queue:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                chunk_reward += reward
                chunk_executed += 1
                obs = next_obs
                if done:
                    break

            ep_step += chunk_executed
            ep_reward += chunk_reward

            # Record chunk transition
            states_b.append(state_vec)
            a_base_b.append(a_base_np)
            delta_a_b.append(delta_a_np)
            log_probs_b.append(log_prob.item())
            rewards_b.append(chunk_reward)
            values_b.append(v_val.item())
            dones_b.append(done)

            if done:
                break

        # Step B: Compute Generalized Advantage Estimation (GAE) over chunks
        last_state = extract_state(obs)
        with torch.no_grad():
            next_value = critic(torch.from_numpy(last_state).unsqueeze(0).to(device)).item()

        advantages, returns = compute_gae(rewards_b, values_b, dones_b, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step C: Chunk-Level PPO Gradient Optimization
        t_states = torch.from_numpy(np.array(states_b)).to(device)        # (N_chunks, 30)
        t_a_base = torch.from_numpy(np.array(a_base_b)).to(device)        # (N_chunks, k_exec, 4)
        t_delta_a = torch.from_numpy(np.array(delta_a_b)).to(device)      # (N_chunks, k_exec, 4)
        t_old_log_probs = torch.tensor(log_probs_b, dtype=torch.float32, device=device) # (N_chunks,)
        t_advantages = torch.from_numpy(advantages).to(device)            # (N_chunks,)
        t_returns = torch.from_numpy(returns).to(device)                  # (N_chunks,)

        for ppo_epoch in range(args.ppo_epochs):
            # Forward pass over full (N_chunks, k_exec, 4) trajectory chunks
            mu, std = residual_actor(t_a_base, t_states)                  # mu: (N_chunks, k_exec, 4), std: (k_exec, 4)
            dist = torch.distributions.Normal(mu, std)

            # Compute new log probabilities across full k_exec trajectory horizon
            new_log_probs = dist.log_prob(t_delta_a).sum(dim=(-2, -1))     # (N_chunks,)
            entropy = -dist.entropy().sum(dim=(-2, -1)).mean()

            # PPO Clipped Surrogate Loss
            ratios = torch.exp(new_log_probs - t_old_log_probs)
            surr1 = ratios * t_advantages
            surr2 = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * t_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            v_pred = critic(t_states).squeeze(-1)                         # (N_chunks,)
            critic_loss = F.mse_loss(v_pred, t_returns)

            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy

            # Gradient Update
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(residual_actor.parameters()) + list(critic.parameters()), max_norm=0.5)
            optimizer.step()

        print(f"Episode [{ep:3d}/{args.episodes:3d}] | Total Reward: {ep_reward:7.2f} | Chunks: {len(states_b):2d} | Steps: {ep_step:3d}")

        if ep % args.eval_interval == 0:
            eval_env = RichRewardFrankaWrapper(make_franka_env())
            eval_reward = evaluate_policy(eval_env, residual_policy, episodes=3, max_steps=args.max_steps, k_exec=args.k_exec)
            print(f"   [PPO Eval Ep {ep}] Avg Reward: {eval_reward:.2f} | Best: {best_reward:.2f}")

            if eval_reward > best_reward:
                best_reward = eval_reward
                best_ckpt = os.path.join(args.save_dir, "residual_ppo_best.pt")
                torch.save({"actor": residual_actor.state_dict(), "critic": critic.state_dict()}, best_ckpt)
                print(f"   [Checkpoint] Saved best residual model -> {best_ckpt}")

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
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(args)


if __name__ == "__main__":
    main()