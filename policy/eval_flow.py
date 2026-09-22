import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env_utils import make_franka_env, setup_scene_physics
from controllers import FlowMatchingController
from dashboard import render_multi_camera_dashboard, close_dashboard
from policy.flow_matching import FlowMatchingPolicy


def evaluate_policy(args):
    """Evaluates trained Flow Matching Policy in PyBullet Panda simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Eval] Using device: {device}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"[Eval] Video recording enabled. Saving to: {args.video_dir}/")

    # 1. Load Trained Policy Checkpoint & Auto-detect matching dataset stats
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"[Eval Error] Specified checkpoint file '{args.checkpoint}' was not found on disk! "
            f"Please verify the filename path or complete training before running evaluation."
        )

    stats_dir = args.data_dir
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint_data.get("args", {})
    if isinstance(ckpt_args, dict) and "data_dir" in ckpt_args:
        ckpt_data_dir = ckpt_args["data_dir"]
        if not os.path.exists(ckpt_data_dir):
            raise FileNotFoundError(
                f"[Eval Error] Checkpoint '{args.checkpoint}' was trained on dataset directory '{ckpt_data_dir}', "
                f"but this dataset directory was not found on disk! Please ensure '{ckpt_data_dir}' exists."
            )
        stats_dir = ckpt_data_dir
        print(f"[Eval] Auto-detected matching dataset directory from checkpoint: {stats_dir}")

    stats_path = os.path.join(stats_dir, "meta", "stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"[Eval Error] Normalization stats file not found at '{stats_path}'! "
            f"Cannot evaluate policy without valid dataset statistics."
        )
    print(f"[Eval] Using normalization stats: {stats_path}")

    policy = FlowMatchingPolicy(
        action_dim=4,
        state_dim=34,
        pred_horizon=args.pred_horizon,
        cond_dim=args.cond_dim,
        stats_path=stats_path,
    ).to(device)

    policy.load_state_dict(checkpoint_data["model_state_dict"])
    print(f"[Eval] Loaded checkpoint from: {args.checkpoint} (Epoch {checkpoint_data.get('epoch', '?')})")

    policy.eval()

    # 2. Setup FlowMatchingController
    controller = FlowMatchingController(
        policy=policy,
        k_exec=args.k_exec,
        ode_steps=args.ode_steps,
    )

    # 3. Create Gymnasium Panda Environment
    env = make_franka_env(max_episode_steps=args.max_steps, corners_only=args.corners_only, near_obstacle=args.near_obstacle)

    # Access underlying PyBullet physics client
    sim = env.unwrapped.sim
    bullet_p = sim.physics_client

    print(f"\n[Eval] Starting Evaluation Rollouts ({args.episodes} episodes)...")
    print(f"       Prediction Horizon (T_pred): {args.pred_horizon}")
    print(f"       Execution Horizon  (K_exec): {args.k_exec}")
    print(f"       ODE Solver Steps   (N_ode):  {args.ode_steps}\n")

    success_count = 0
    total_steps_all_episodes = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)

        # Setup custom obstacle wall, goal marker, and gripper dynamics
        obstacle_id = setup_scene_physics(env)
        controller.reset_buffer()

        video_writer = None
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"eval_ep_{ep:02d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Multi-camera dashboard dimensions: 960x720
            video_writer = cv2.VideoWriter(video_path, fourcc, args.video_fps, (960, 720))

        ep_step = 0
        ep_success = False
        start_time = time.time()

        while ep_step < args.max_steps:
            # Query FlowMatchingController for action using Receding Horizon Control
            action, action_label, _, _, _ = controller.get_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_step += 1

            contacts_a = bullet_p.getContactPoints(bodyA=obstacle_id)
            contacts_b = bullet_p.getContactPoints(bodyB=obstacle_id)
            ee_pos = obs["observation"][:3]
            geom_collision = abs(ee_pos[0] - (-0.02)) < 0.035 and abs(ee_pos[1]) < 0.16 and ee_pos[2] < 0.26
            is_collision = len(contacts_a) > 0 or len(contacts_b) > 0 or geom_collision
            gripper_width = env.unwrapped.robot.get_fingers_width()

            # Check distance to goal for success determination
            obj_pos = obs["achieved_goal"][:3]
            goal_pos = obs["desired_goal"][:3]
            dist_obj_to_goal = np.linalg.norm(goal_pos - obj_pos)
            is_success = bool(info.get("is_success", False)) or (dist_obj_to_goal < 0.05)

            # Render dashboard / capture video frame
            if args.render or args.save_video:
                key, frame = render_multi_camera_dashboard(
                    bullet_p,
                    sim=sim,
                    step=ep_step,
                    max_steps=args.max_steps,
                    episode=ep,
                    mode="EVAL",
                    action_label=action_label,
                    gripper_width=gripper_width,
                    is_collision=is_collision,
                    is_success=is_success,
                    wait_ms=20 if args.render else 1,
                    show_window=args.render,
                )

                if args.save_video:
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

        total_steps_all_episodes.append(ep_step)
        status_str = "SUCCESS" if ep_success else "FAILED"
        elapsed_ep = time.time() - start_time
        video_note = f" -> {video_path}" if args.save_video else ""
        print(f"  Episode [{ep:2d}/{args.episodes:2d}] | Result: {status_str} | Steps: {ep_step:3d} | Time: {elapsed_ep:.1f}s{video_note}")

    env.close()
    if args.render:
        close_dashboard()

    success_rate = (success_count / args.episodes) * 100.0
    avg_steps = np.mean(total_steps_all_episodes)
    print(f"\n[Eval] Evaluation Completed!")
    print(f"       Success Rate: {success_count}/{args.episodes} ({success_rate:.1f}%)")
    print(f"       Average Steps per Episode: {avg_steps:.1f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching Policy in Franka Panda Gym Sim")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flow_policy_best.pt", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_all", help="Path to LeRobot dataset")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode (default: 300)")
    parser.add_argument("--pred-horizon", type=int, default=16, help="Action prediction chunk length (default: 16)")
    parser.add_argument("--k-exec", type=int, default=4, help="Receding Horizon execution steps (default: 4)")
    parser.add_argument("--corners-only", action="store_true", help="Restrict goal target strictly to table corners")
    parser.add_argument("--near-obstacle", action="store_true", help="Place pickup block right next to the obstacle wall (X = -0.07..-0.04m)")
    parser.add_argument("--ode-steps", type=int, default=10, help="Number of Euler ODE inference steps (default: 10)")
    parser.add_argument("--cond-dim", type=int, default=256, help="Condition embedding dimension (default: 256)")
    parser.add_argument("--render", action="store_true", help="Render PyBullet visual window & camera dashboard")
    parser.add_argument("--save-video", action="store_true", help="Save MP4 video of evaluation rollouts to disk")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save evaluation rollout videos (default: videos)")
    parser.add_argument("--video-fps", type=int, default=30, help="Video frame rate FPS (default: 30)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for evaluation (default: 100)")
    args = parser.parse_args()

    evaluate_policy(args)


if __name__ == "__main__":
    main()
