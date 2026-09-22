import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

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

        device = next(policy.parameters()).device if list(policy.parameters()) else torch.device("cpu")
        step = 0
        while step < max_steps:
            state_vec = extract_state(obs)  # (34,)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)  # (1, 34)

            if len(action_buffer) == 0:
                with torch.no_grad():
                    a_final_chunk, _, _, _, _ = policy.get_action(state_tensor, deterministic=True, k_exec=k_exec)
                    a_final_chunk = a_final_chunk.squeeze(0).cpu().numpy()  # (k_exec, 4)
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

    # Initialize TensorBoard SummaryWriter
    writer = None
    if HAS_TENSORBOARD and not args.no_tensorboard:
        run_name = f"ppo_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(args.log_dir, run_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[PPO Train] TensorBoard logging enabled -> {log_dir}")
    else:
        print("[PPO Train] TensorBoard logging disabled or package not available.")

    # 1. Load Pre-trained Flow Matching Policy
    flow_policy = FlowMatchingPolicy(
        action_dim=4,
        state_dim=34,
        pred_horizon=16,
        cond_dim=args.cond_dim,
        stats_path=os.path.join(args.data_dir, "meta", "stats.json"),
    ).to(device)

    if not args.flow_checkpoint or not os.path.exists(args.flow_checkpoint):
        raise FileNotFoundError(f"Flow checkpoint file not found at: '{args.flow_checkpoint}'")

    ckpt = torch.load(args.flow_checkpoint, map_location=device)
    flow_policy.load_state_dict(ckpt["model_state_dict"])
    print(f"[PPO Train] Loaded base Flow Matching policy from: {args.flow_checkpoint}")

    # 2. Setup Residual Actor & Critic
    residual_actor = ResidualFlowMatchingActor(action_dim=4, state_dim=34, cond_dim=args.cond_dim).to(device)
    critic = ValueCritic(state_dim=34, cond_dim=args.cond_dim).to(device)

    if args.ppo_checkpoint:
        if not os.path.exists(args.ppo_checkpoint):
            raise FileNotFoundError(f"PPO checkpoint file not found at: '{args.ppo_checkpoint}'")
        ppo_ckpt = torch.load(args.ppo_checkpoint, map_location=device)
        if "actor" in ppo_ckpt:
            residual_actor.load_state_dict(ppo_ckpt["actor"])
            if "critic" in ppo_ckpt:
                critic.load_state_dict(ppo_ckpt["critic"])
        else:
            residual_actor.load_state_dict(ppo_ckpt)
        print(f"[PPO Train] Resuming training from residual PPO checkpoint: {args.ppo_checkpoint}")

    residual_policy = ResidualFlowPolicy(flow_policy, residual_actor, critic)

    # 3. Initialize Optimizer & Learning Rate Scheduler
    optimizer = optim.AdamW(
        list(residual_actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    total_ppo_updates = max(1, args.episodes // args.batch_episodes)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_ppo_updates,
        eta_min=args.lr_min,
    )

    # 4. Create Gym Environment with Rich Reward
    base_env = make_franka_env(
        max_episode_steps=args.max_steps,
        corners_only=args.corners_only,
        near_obstacle=args.near_obstacle,
    )
    env = RichRewardFrankaWrapper(base_env)

    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")
    start_time = time.time()

    print(f"\n[PPO Train] Starting Chunk-Level Residual Flow-PPO Training ({args.episodes} episodes)...")
    print(f"            Learning Rate: {args.lr} | K_exec: {args.k_exec} | Horizon: 16\n")

    # Multi-Episode Batch Buffers
    batch_states = []
    batch_a_base = []
    batch_delta_a = []
    batch_log_probs = []
    batch_advantages = []
    batch_returns = []

    for ep in range(1, args.episodes + 1):
        # Update exploration noise schedule (decays log_std from -2.3 ~ 0.10 to -3.5 ~ 0.03 over 70% of episodes)
        residual_actor.update_noise_std(
            ep,
            args.episodes,
            init_log_std=args.init_log_std,
            min_log_std=args.min_log_std,
        )

        obs, info = env.reset(seed=args.seed + ep)
        obstacle_id = setup_scene_physics(env)
        env.set_obstacle_id(obstacle_id)

        # Episode Rollout Buffers
        states_ep = []
        a_base_ep = []
        delta_a_ep = []
        log_probs_ep = []
        rewards_ep = []
        values_ep = []
        dones_ep = []

        ep_step = 0
        ep_reward = 0.0

        # Step A: Rollout Collection for 1 Episode
        while ep_step < args.max_steps:
            state_vec = extract_state(obs)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)

            with torch.no_grad():
                a_final_chunk, a_base_chunk, delta_a_chunk, log_prob, v_val = residual_policy.get_action(state_tensor, deterministic=False, k_exec=args.k_exec)

            a_final_np = a_final_chunk.squeeze(0).cpu().numpy()
            a_base_np = a_base_chunk.squeeze(0).cpu().numpy()
            delta_a_np = delta_a_chunk.squeeze(0).cpu().numpy()

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

            states_ep.append(state_vec)
            a_base_ep.append(a_base_np)
            delta_a_ep.append(delta_a_np)
            log_probs_ep.append(log_prob.item())
            rewards_ep.append(chunk_reward)
            values_ep.append(v_val.item())
            dones_ep.append(done)

            if done:
                break

        # Step B: GAE Calculation for Episode
        last_state = extract_state(obs)
        with torch.no_grad():
            next_value = critic(torch.from_numpy(last_state).unsqueeze(0).to(device)).item()

        adv_ep, ret_ep = compute_gae(rewards_ep, values_ep, dones_ep, next_value)

        # Accumulate Episode into Batch
        batch_states.extend(states_ep)
        batch_a_base.extend(a_base_ep)
        batch_delta_a.extend(delta_a_ep)
        batch_log_probs.extend(log_probs_ep)
        batch_advantages.extend(adv_ep)
        batch_returns.extend(ret_ep)

        current_batch_chunks = len(batch_states)
        did_ppo_update = False
        current_lr = optimizer.param_groups[0]["lr"]

        # Log Episode Level Metrics to TensorBoard
        if writer:
            writer.add_scalar("train/episode_reward", ep_reward, ep)
            writer.add_scalar("train/episode_steps", ep_step, ep)
            writer.add_scalar("train/learning_rate", current_lr, ep)

        # Step C: PPO Gradient Optimization over Multi-Episode Batch
        epoch_tot_losses = []
        epoch_act_losses = []
        epoch_val_losses = []
        epoch_entropies = []
        epoch_stds = []
        epoch_kls = []
        epoch_clips = []

        if (ep % args.batch_episodes == 0) or (ep == args.episodes):
            did_ppo_update = True
            t_states = torch.from_numpy(np.array(batch_states)).to(device)
            t_a_base = torch.from_numpy(np.array(batch_a_base)).to(device)
            t_delta_a = torch.from_numpy(np.array(batch_delta_a)).to(device)
            t_old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)
            t_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
            t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)
            t_returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)

            for ppo_epoch in range(args.ppo_epochs):
                mu, std = residual_actor(t_a_base, t_states)
                dist = torch.distributions.Normal(mu, std)

                new_log_probs = dist.log_prob(t_delta_a).sum(dim=(-2, -1))
                entropy = -dist.entropy().sum(dim=(-2, -1)).mean()

                ratios = torch.exp(new_log_probs - t_old_log_probs)
                surr1 = ratios * t_advantages
                surr2 = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * t_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                v_pred = critic(t_states).squeeze(-1)
                critic_loss = F.huber_loss(v_pred, t_returns, delta=10.0)

                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(residual_actor.parameters()) + list(critic.parameters()), max_norm=0.5)
                optimizer.step()

                # Diagnostic Metrics
                with torch.no_grad():
                    approx_kl = ((ratios - 1.0) - torch.log(ratios)).mean().item()
                    clip_frac = ((ratios - 1.0).abs() > args.clip_eps).float().mean().item()

                epoch_tot_losses.append(total_loss.item())
                epoch_act_losses.append(actor_loss.item())
                epoch_val_losses.append(critic_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_stds.append(std.mean().item())
                epoch_kls.append(approx_kl)
                epoch_clips.append(clip_frac)

            scheduler.step()  # Stepped right after optimizer.step()

            # Mean PPO update metrics
            avg_tot_loss = float(np.mean(epoch_tot_losses))
            avg_act_loss = float(np.mean(epoch_act_losses))
            avg_val_loss = float(np.mean(epoch_val_losses))
            avg_ent = float(np.mean(epoch_entropies))
            avg_std = float(np.mean(epoch_stds))
            avg_kl = float(np.mean(epoch_kls))
            avg_clip = float(np.mean(epoch_clips))

            if writer:
                writer.add_scalar("train/ppo/total_loss", avg_tot_loss, ep)
                writer.add_scalar("train/ppo/actor_loss", avg_act_loss, ep)
                writer.add_scalar("train/ppo/critic_loss", avg_val_loss, ep)
                writer.add_scalar("train/ppo/entropy", avg_ent, ep)
                writer.add_scalar("train/ppo/action_std", avg_std, ep)
                writer.add_scalar("train/ppo/approx_kl", avg_kl, ep)
                writer.add_scalar("train/ppo/clip_fraction", avg_clip, ep)

            # Clear batch buffers after update
            batch_states.clear()
            batch_a_base.clear()
            batch_delta_a.clear()
            batch_log_probs.clear()
            batch_advantages.clear()
            batch_returns.clear()

        # Terminal Progress Printing
        print(f"Episode [{ep:3d}/{args.episodes:3d}] | Total Reward: {ep_reward:7.2f} | LR: {current_lr:.6f} | Ep Chunks: {len(states_ep):2d} | Batch Chunks: {current_batch_chunks:3d} | Steps: {ep_step:3d}")
        if did_ppo_update:
            print(f"   |- [PPO UPDATE] Loss: {avg_tot_loss:.4f} | Act: {avg_act_loss:.4f} | Val: {avg_val_loss:.4f} | Ent: {avg_ent:.4f} | Std: {avg_std:.4f} | KL: {avg_kl:.4f} | Clip: {avg_clip*100:.1f}%")

        if ep % args.eval_interval == 0:
            eval_env = RichRewardFrankaWrapper(
                make_franka_env(
                    max_episode_steps=args.max_steps,
                    corners_only=args.corners_only,
                    near_obstacle=args.near_obstacle,
                )
            )
            eval_reward = evaluate_policy(eval_env, residual_policy, episodes=3, max_steps=args.max_steps, k_exec=args.k_exec)
            print(f"   [PPO Eval Ep {ep}] Avg Reward: {eval_reward:.2f} | Best: {best_reward:.2f}")

            if writer:
                writer.add_scalar("eval/avg_reward", eval_reward, ep)

            if eval_reward > best_reward:
                best_reward = eval_reward
                best_ckpt = args.out_checkpoint if args.out_checkpoint else os.path.join(args.save_dir, "residual_ppo_best.pt")
                os.makedirs(os.path.dirname(best_ckpt) or ".", exist_ok=True)
                torch.save({"actor": residual_actor.state_dict(), "critic": critic.state_dict()}, best_ckpt)
                print(f"   [Checkpoint] Saved best residual model -> {best_ckpt}")

    if writer:
        writer.close()

    print(f"\n[PPO Train] Completed in {time.time() - start_time:.1f}s!")


def main():
    parser = argparse.ArgumentParser(description="Train Residual Flow-PPO Policy")
    parser.add_argument("--flow-checkpoint", type=str, required=True, help="Path to trained base Flow Matching checkpoint (.pt file)")
    parser.add_argument("--ppo-checkpoint", type=str, default=None, help="Path to residual PPO checkpoint to resume training from (default: None)")
    parser.add_argument("--out-checkpoint", type=str, default="checkpoints/residual_ppo_tuned.pt", help="Custom output path to save best fine-tuned residual PPO checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_all")
    parser.add_argument("--episodes", type=int, default=600, help="Total PPO training episodes (default: 600)")
    parser.add_argument("--batch-episodes", type=int, default=4, help="Number of rollout episodes to accumulate per PPO update (default: 4)")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate for CosineAnnealingLR scheduler")
    parser.add_argument("--init-log-std", type=float, default=-2.3, help="Initial log_std (-2.3 -> std ~ 0.10, -2.7 -> std ~ 0.067, -3.0 -> std ~ 0.05)")
    parser.add_argument("--min-log-std", type=float, default=-3.2, help="Minimum log_std floor (-3.2 -> std ~ 0.04)")
    parser.add_argument("--k-exec", type=int, default=4)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=8, help="Number of PPO optimization epochs per batch (default: 8)")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--corners-only", action="store_true", help="Restrict goal target strictly to table corners. Default: False.")
    parser.add_argument("--near-obstacle", action="store_true", help="Place block close to obstacle wall (X: -0.12..-0.09). Default: False.")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for TensorBoard log runs (default: runs)")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(args)


if __name__ == "__main__":
    main()