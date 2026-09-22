import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env_utils import make_franka_env, setup_scene_physics, extract_state
from dashboard import render_multi_camera_dashboard, close_dashboard
from policy.flow_matching import FlowMatchingPolicy
from ppo.residual_ppo import ResidualFlowMatchingActor, ValueCritic, ResidualFlowPolicy
from ppo.reward_utils import RichRewardFrankaWrapper


def evaluate_ppo_policy(args):
    """Evaluates Residual Flow-PPO Policy in PyBullet Franka Panda Gym environment with optional dashboard rendering & MP4 video recording."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n[PPO Eval] Using device: {device}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"[PPO Eval] Video recording enabled. Saving to: {args.video_dir}/")

    # 1. Load Pre-trained Base Flow Matching Policy & Auto-detect matching dataset stats
    if not os.path.exists(args.flow_checkpoint):
        raise FileNotFoundError(
            f"[PPO Eval Error] Base flow checkpoint '{args.flow_checkpoint}' was not found on disk! "
            f"Please verify the filename path or complete base model training before running evaluation."
        )

    stats_dir = args.data_dir
    flow_ckpt_data = torch.load(args.flow_checkpoint, map_location=device)
    ckpt_args = flow_ckpt_data.get("args", {})
    if isinstance(ckpt_args, dict) and "data_dir" in ckpt_args:
        ckpt_data_dir = ckpt_args["data_dir"]
        if not os.path.exists(ckpt_data_dir):
            raise FileNotFoundError(
                f"[PPO Eval Error] Base flow checkpoint '{args.flow_checkpoint}' was trained on dataset directory '{ckpt_data_dir}', "
                f"but this dataset directory was not found on disk! Please ensure '{ckpt_data_dir}' exists."
            )
        stats_dir = ckpt_data_dir
        print(f"[PPO Eval] Auto-detected matching dataset directory from base checkpoint: {stats_dir}")

    stats_path = os.path.join(stats_dir, "meta", "stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"[PPO Eval Error] Normalization stats file not found at '{stats_path}'! "
            f"Cannot evaluate policy without valid dataset statistics."
        )
    print(f"[PPO Eval] Using normalization stats: {stats_path}")

    flow_policy = FlowMatchingPolicy(
        action_dim=4,
        state_dim=34,
        pred_horizon=args.pred_horizon,
        cond_dim=args.cond_dim,
        stats_path=stats_path,
    ).to(device)

    flow_policy.load_state_dict(flow_ckpt_data["model_state_dict"])
    print(f"[PPO Eval] Loaded base Flow Matching policy from: {args.flow_checkpoint}")

    # 2. Setup Residual Actor & Critic
    residual_actor = ResidualFlowMatchingActor(
        action_dim=4,
        state_dim=34,
        cond_dim=args.cond_dim,
        pred_horizon=args.pred_horizon,
    ).to(device)

    critic = ValueCritic(state_dim=34, cond_dim=args.cond_dim).to(device)

    if not os.path.exists(args.ppo_checkpoint):
        raise FileNotFoundError(
            f"[PPO Eval Error] Specified Residual PPO checkpoint '{args.ppo_checkpoint}' was not found on disk! "
            f"Please verify the filename path or complete PPO training before running evaluation."
        )

    ppo_ckpt = torch.load(args.ppo_checkpoint, map_location=device)
    if "actor" in ppo_ckpt:
        residual_actor.load_state_dict(ppo_ckpt["actor"])
        if "critic" in ppo_ckpt:
            critic.load_state_dict(ppo_ckpt["critic"])
    else:
        residual_actor.load_state_dict(ppo_ckpt)
    print(f"[PPO Eval] Loaded Residual PPO checkpoint from: {args.ppo_checkpoint}")

    residual_policy = ResidualFlowPolicy(flow_policy, residual_actor, critic)
    residual_policy.eval()

    # 3. Create Gym Environment with Rich Reward Wrapper
    base_env = make_franka_env(max_episode_steps=args.max_steps, corners_only=args.corners_only, near_obstacle=args.near_obstacle)
    env = RichRewardFrankaWrapper(base_env)

    sim = env.unwrapped.sim
    bullet_p = sim.physics_client

    print(f"\n[PPO Eval] Starting Evaluation Rollouts ({args.episodes} episodes)...")
    print(f"           Mode: Deterministic (Greedy Action Selection)")
    print(f"           Horizon (K_exec): {args.k_exec}\n")

    success_count = 0
    total_rewards_all = []
    total_steps_all = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        obstacle_id = setup_scene_physics(env)
        env.set_obstacle_id(obstacle_id)

        video_writer = None
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"ppo_eval_ep_{ep:02d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Multi-camera dashboard size: 960x720
            video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (960, 720))

        ep_step = 0
        ep_reward = 0.0
        ep_success = False
        action_queue = []
        last_delta_a_str = "Delta: [0,0,0,0]"
        start_time = time.time()

        while ep_step < args.max_steps:
            state_vec = extract_state(obs)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)

            if len(action_queue) == 0:
                with torch.no_grad():
                    a_final_chunk, a_base_chunk, delta_a_chunk, _, _ = residual_policy.get_action(
                        state_tensor,
                        deterministic=True,
                        k_exec=args.k_exec,
                    )
                a_final_np = a_final_chunk.squeeze(0).cpu().numpy()  # (k_exec, 4)
                delta_a_np = delta_a_chunk.squeeze(0).cpu().numpy()  # (k_exec, 4)
                action_queue = list(a_final_np)
                last_delta_a_str = f"Delta: {np.round(delta_a_np[0], 3)}"

            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_step += 1
            ep_reward += reward
            obs = next_obs

            contacts_a = bullet_p.getContactPoints(bodyA=obstacle_id)
            contacts_b = bullet_p.getContactPoints(bodyB=obstacle_id)
            ee_pos = obs["observation"][:3]
            geom_collision = abs(ee_pos[0] - (-0.02)) < 0.035 and abs(ee_pos[1]) < 0.16 and ee_pos[2] < 0.26
            is_collision = len(contacts_a) > 0 or len(contacts_b) > 0 or geom_collision
            gripper_width = env.unwrapped.robot.get_fingers_width()

            obj_pos = obs["achieved_goal"][:3]
            goal_pos = obs["desired_goal"][:3]
            dist_obj_to_goal = float(np.linalg.norm(goal_pos - obj_pos))
            is_success = bool(info.get("is_success", False)) or (dist_obj_to_goal < 0.05)

            if args.render or args.save_video:
                key, frame = render_multi_camera_dashboard(
                    bullet_p,
                    sim=sim,
                    step=ep_step,
                    max_steps=args.max_steps,
                    episode=ep,
                    mode="PPO_EVAL",
                    action_label=last_delta_a_str,
                    gripper_width=gripper_width,
                    is_collision=is_collision,
                    is_success=is_success,
                    wait_ms=20 if args.render else 1,
                    show_window=args.render,
                )

                if args.save_video and video_writer is not None:
                    video_writer.write(frame)

            if is_success:
                ep_success = True
                break

            if terminated or truncated:
                break

        if video_writer is not None:
            video_writer.release()

        if ep_success:
            success_count += 1

        total_rewards_all.append(ep_reward)
        total_steps_all.append(ep_step)
        status_str = "SUCCESS" if ep_success else "FAILED"
        elapsed_ep = time.time() - start_time
        video_note = f" -> {video_path}" if args.save_video else ""
        print(f"  Episode [{ep:2d}/{args.episodes:2d}] | Result: {status_str:7s} | Reward: {ep_reward:7.2f} | Steps: {ep_step:3d} | Time: {elapsed_ep:.1f}s{video_note}")

    env.close()
    if args.render:
        close_dashboard()

    success_rate = (success_count / args.episodes) * 100.0
    avg_reward = np.mean(total_rewards_all)
    avg_steps = np.mean(total_steps_all)

    print(f"\n[PPO Eval] Evaluation Completed!")
    print(f"           Success Rate: {success_count}/{args.episodes} ({success_rate:.1f}%)")
    print(f"           Average Reward: {avg_reward:.2f}")
    print(f"           Average Steps per Episode: {avg_steps:.1f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Residual Flow-PPO Policy in Franka Panda Gym Sim")
    parser.add_argument("--ppo-checkpoint", type=str, default="checkpoints/residual_ppo_best.pt", help="Path to residual PPO model checkpoint")
    parser.add_argument("--flow-checkpoint", type=str, default="checkpoints/flow_policy_best.pt", help="Path to base Flow Matching checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_all", help="Path to dataset metadata")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode (default: 300)")
    parser.add_argument("--pred-horizon", type=int, default=16, help="Prediction horizon (default: 16)")
    parser.add_argument("--k-exec", type=int, default=4, help="Execution horizon steps per chunk (default: 4)")
    parser.add_argument("--corners-only", action="store_true", help="Restrict goal target strictly to table corners")
    parser.add_argument("--near-obstacle", action="store_true", help="Place pickup block right next to the obstacle wall (X = -0.07..-0.04m)")
    parser.add_argument("--cond-dim", type=int, default=256, help="Condition embedding dimension (default: 256)")
    parser.add_argument("--render", action="store_true", help="Render PyBullet visual window & multi-camera dashboard")
    parser.add_argument("--save-video", action="store_true", help="Save MP4 video recordings of evaluation rollouts")
    parser.add_argument("--video-dir", type=str, default="videos/ppo", help="Directory to save MP4 videos (default: videos/ppo)")
    parser.add_argument("--video-fps", type=int, default=30, help="Video frame rate FPS (default: 30)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--seed", type=int, default=100, help="Random seed (default: 100)")
    args = parser.parse_args()

    evaluate_ppo_policy(args)


if __name__ == "__main__":
    main()
