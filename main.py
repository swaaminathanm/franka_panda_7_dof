import argparse
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym
import panda_gym

from controllers import ManualController, FlowMatchingController
from dashboard import close_dashboard, render_multi_camera_dashboard
from recorder import EpisodeRecorder
from env_utils import (
    create_obstacle,
    set_goal_color,
    setup_block_obj_goal_pos,
    setup_grasp_physics,
    setup_scene_physics,
    extract_state,
    make_franka_env,
)


def main():
    parser = argparse.ArgumentParser(description="Franka Panda 7-DoF Simulation")
    parser.add_argument(
        "--mode",
        choices=["MANUAL", "AGENT"],
        default="MANUAL",
        help="Operating mode: MANUAL (keyboard teleoperation) or AGENT (Flow Matching policy)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max episode steps (default: 500).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/raw",
        help="Directory to save demonstration episodes (default: data/raw)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/flow_policy_best.pt",
        help="Policy checkpoint for AGENT mode (default: checkpoints/flow_policy_best.pt)",
    )
    parser.add_argument(
        "--k-exec",
        type=int,
        default=4,
        help="Receding Horizon Control execution steps for AGENT mode (default: 4)",
    )
    parser.add_argument(
        "--record-idle",
        action="store_true",
        help="Record idle frames in MANUAL mode (when no key is pressed). Default: False.",
    )
    args = parser.parse_args()
    mode = args.mode.upper()

    # Create Gym Environment
    if mode == "MANUAL":
        env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array").unwrapped
        max_steps = None
    else:
        max_steps = args.max_steps
        env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps)

    setup_block_obj_goal_pos(env)
    obs, info = env.reset()

    sim = env.unwrapped.sim
    p = sim.physics_client

    obstacle_id = setup_scene_physics(env)
    action_dim = env.action_space.shape[0]

    # Setup Controller based on Mode
    if mode == "MANUAL":
        controller = ManualController(step_size=0.4, gripper_speed=1.0)
    else:
        from policy.flow_matching import FlowMatchingPolicy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = FlowMatchingPolicy(
            action_dim=4,
            state_dim=30,
            pred_horizon=16,
            cond_dim=256,
            stats_path="data/lerobot/meta/stats.json",
        ).to(device)

        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(ckpt["model_state_dict"])
            print(f"[AGENT Mode] Loaded trained policy from: {args.checkpoint} (Epoch {ckpt.get('epoch', '?')})")
        else:
            print(f"[AGENT Mode] Warning: Checkpoint {args.checkpoint} not found! Running un-trained policy.")
        policy.eval()

        controller = FlowMatchingController(policy=policy, k_exec=args.k_exec)

    recorder = EpisodeRecorder(save_dir=args.save_dir, fps=50)

    print("\n" + "=" * 60)
    print(f" FRANKA PANDA SIMULATION — MODE: {mode}")
    if mode == "MANUAL":
        print(' Step termination: DISABLED (Unlimited steps, reset with "R")')
        print(f" Demonstrations saved in : {args.save_dir}/ (Existing: {recorder.saved_count})")
    else:
        print(f" Max Steps per Episode: {max_steps}")
        print(f" Execution Horizon    : {args.k_exec} steps (Receding Horizon Control)")
    print("=" * 60)

    if mode == "MANUAL":
        print(' Focus on the "Multi-Camera Dashboard" window to operate:')
        print("   W / S       : Move End-Effector Forward (+X) / Backward (-X)")
        print("   A / D       : Move End-Effector Left (+Y) / Right (-Y)")
        print("   E / Q       : Move End-Effector Up (+Z) / Down (-Z)")
        print("   O / C       : Open / Close Gripper")
        print("   SPACE       : Hold position (zero action)")
        print("   B / T       : Start / Pause Recording (Toggle)")
        print("   ENTER       : Save demonstration episode")
        print("   R           : Reset / Discard current episode")
        print("   ESC         : Exit simulation")
        print("=" * 60 + "\n")

    episode = 1
    step = 0
    action_label = "IDLE" if mode == "MANUAL" else "AGENT (Flow)"
    is_success = False
    is_recording = False

    def reset_episode_state():
        nonlocal obs, info, obstacle_id, episode, step, is_success, is_recording
        obs, info = env.reset()
        obstacle_id = setup_scene_physics(env)
        if hasattr(controller, "reset_gripper"):
            controller.reset_gripper()
        if hasattr(controller, "reset_buffer"):
            controller.reset_buffer()
        episode += 1
        step = 0
        is_success = False
        is_recording = False

    try:
        while True:
            # Check contact with obstacle
            contacts = p.getContactPoints(bodyA=obstacle_id)
            is_collision = len(contacts) > 0

            # Gripper opening width (in meters)
            gripper_width = env.unwrapped.robot.get_fingers_width()

            key, _ = render_multi_camera_dashboard(
                p,
                sim=sim,
                step=step,
                max_steps=max_steps,
                episode=episode,
                mode=mode,
                action_label=action_label,
                gripper_width=gripper_width,
                is_collision=is_collision,
                is_success=is_success,
                demos_saved=recorder.saved_count,
                rec_steps=recorder.current_length,
                is_recording=is_recording,
                wait_ms=20,
            )

            if key == 27:  # ESC
                print("Simulation stopped by user.")
                break

            # Unified Controller Call
            (
                action,
                action_label,
                reset_requested,
                save_requested,
                record_toggle_requested,
            ) = controller.get_action(key=key, obs=obs, action_dim=action_dim)

            if record_toggle_requested:
                is_recording = not is_recording
                state_str = "STARTED" if is_recording else "PAUSED"
                print(
                    f">>> [Recorder] Recording {state_str} at step {step} ({recorder.current_length} frames buffered)."
                )
                continue

            if save_requested:
                if recorder.current_length == 0:
                    print("[Recorder] Buffer is empty; nothing to save.")
                else:
                    saved_path = recorder.save_episode(
                        task_name="pick_and_place_around_obstacle"
                    )
                    if saved_path:
                        print(f"Demonstration saved to: {saved_path}")
                reset_episode_state()
                continue

            if reset_requested:
                if recorder.current_length > 0:
                    print(
                        f"Manual reset triggered at step {step}. Discarded {recorder.current_length} buffered frames."
                    )
                recorder.clear_buffer()
                reset_episode_state()
                continue

            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            is_success = bool(info.get("is_success", False))

            if mode == "MANUAL":
                truncated = False

            # Buffer step transition into recorder if in MANUAL recording mode
            should_record = is_recording if mode == "MANUAL" else False
            if mode == "MANUAL" and not args.record_idle and action_label == "IDLE":
                should_record = False

            if should_record:
                recorder.record_step(
                    obs=prev_obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    is_collision=is_collision,
                )

            if terminated or truncated:
                status = (
                    "SUCCESS"
                    if is_success
                    else ("TIMEOUT" if truncated else "TERMINATED")
                )
                print(
                    f"Episode {episode} finished [{status}] at step {step}. Resetting..."
                )
                if mode == "MANUAL" and is_success:
                    saved_path = recorder.save_episode(
                        task_name="pick_and_place_around_obstacle"
                    )
                    if saved_path:
                        print(f"Demonstration saved to: {saved_path}")
                else:
                    recorder.clear_buffer()

                reset_episode_state()

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        if recorder.current_length > 0:
            print(
                f"[Recorder] Exiting simulation. {recorder.current_length} un-saved frames in buffer discarded."
            )
            recorder.clear_buffer()
        close_dashboard()
        env.close()


if __name__ == "__main__":
    main()
