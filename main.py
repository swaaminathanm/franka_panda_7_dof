import argparse

import gymnasium as gym
import numpy as np
import panda_gym

from controllers import ManualController
from dashboard import close_dashboard, render_multi_camera_dashboard
from recorder import EpisodeRecorder


def create_obstacle(p):
    wall_thickness = 0.02  # 2 cm thick along X
    wall_width = 0.28  # 28 cm wide in the center (table is 70 cm wide)
    wall_height = 0.24  # 24 cm tall (slightly higher than the EE default resting pos at z = 0.20 m)
    table_z = 0.0

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_thickness / 2.0, wall_width / 2.0, wall_height / 2.0],
        rgbaColor=[0.2, 0.4, 0.85, 1.0],  # Blue partition wall
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_thickness / 2.0, wall_width / 2.0, wall_height / 2.0],
    )
    obstacle_id = p.createMultiBody(
        baseMass=0,  # Static / immovable
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[-0.02, 0.0, table_z + wall_height / 2.0],
    )
    return obstacle_id


def set_goal_color(p, rgba_color=None):
    """Set the goal target color (default: vibrant red)."""
    if rgba_color is None:
        rgba_color = [1.0, 0.15, 0.15, 1.0]
    for i in range(p.getNumBodies()):
        v_data = p.getVisualShapeData(i)
        if v_data and len(v_data) > 0:
            rgba = v_data[0][7]
            # In panda-gym, the goal body has initial alpha < 0.5
            if rgba[3] < 0.5:
                p.changeVisualShape(i, -1, rgbaColor=rgba_color)


def setup_block_obj_goal_pos(env):
    task = env.unwrapped.task
    task.obj_range_low = np.array([-0.25, -0.10, 0.0])
    task.obj_range_high = np.array([-0.15, 0.10, 0.0])
    task.goal_range_low = np.array([0.15, -0.10, 0.0])
    task.goal_range_high = np.array([0.25, 0.10, 0.0])
    return task


def setup_grasp_physics(p, sim):
    """Enhances contact friction on gripper finger pads for firm, slip-free grasping."""
    robot_id = sim._bodies_idx.get("panda", 0)
    # Apply rubber-pad dynamics ONLY to the two gripper fingers (links 9 and 10)
    for finger_link in (9, 10):
        p.changeDynamics(
            robot_id,
            finger_link,
            lateralFriction=3.0,
            spinningFriction=0.1,
            rollingFriction=0.1,
            frictionAnchor=1,
        )


def main():
    parser = argparse.ArgumentParser(description="Franka Panda 7-DoF Simulation")
    parser.add_argument(
        "--mode",
        choices=["MANUAL", "AGENT"],
        default="MANUAL",
        help="Operating mode: MANUAL (teleoperation, unlimited steps) or AGENT",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max episode steps for AGENT mode (default: 200). Ignored in MANUAL mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/raw",
        help="Directory to save demonstration episodes (default: data/raw)",
    )
    parser.add_argument(
        "--record-idle",
        action="store_true",
        help="Record idle frames in MANUAL mode (when no key is pressed). Default: False.",
    )
    args = parser.parse_args()
    mode = args.mode.upper()

    # In AGENT mode: Explicitly pass max_episode_steps to gym.make (enforced by TimeLimit wrapper)
    # In MANUAL mode: Unwrap the environment to completely disable the TimeLimit wrapper
    if mode == "MANUAL":
        env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array").unwrapped
        max_steps = None
    else:
        max_steps = args.max_steps
        env = gym.make(
            "PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps
        )

    setup_block_obj_goal_pos(env)

    obs, info = env.reset()
    print("Environment reset successful!")
    print("Initial observation keys:", list(obs.keys()))
    print("Action space:", env.action_space)

    sim = env.unwrapped.sim
    p = sim.physics_client

    obstacle_id = create_obstacle(p)
    set_goal_color(p)
    setup_grasp_physics(p, sim)

    controller = ManualController(step_size=0.4, gripper_speed=1.0)
    recorder = EpisodeRecorder(save_dir=args.save_dir, fps=50)
    action_dim = env.action_space.shape[0]

    print("\n" + "=" * 60)
    print(f" FRANKA PANDA SIMULATION — MODE: {mode}")
    if mode == "MANUAL":
        print(' Step termination: DISABLED (Unlimited steps, reset with "R")')
    print(
        f" Demonstrations saved in : {args.save_dir}/ (Existing: {recorder.saved_count})"
    )
    print("=" * 60)
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
    action_label = "IDLE"
    is_success = False
    is_recording = False

    try:
        while True:
            # Check contact with obstacle
            contacts = p.getContactPoints(bodyA=obstacle_id)
            is_collision = len(contacts) > 0

            # Gripper opening width (in meters)
            gripper_width = env.unwrapped.robot.get_fingers_width()

            key = render_multi_camera_dashboard(
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

            (
                action,
                action_label,
                reset_requested,
                save_requested,
                record_toggle_requested,
            ) = controller.get_action(key, action_dim=action_dim)

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
                obs, info = env.reset()
                set_goal_color(p)
                setup_grasp_physics(p, sim)
                controller.reset_gripper()
                episode += 1
                step = 0
                is_success = False
                is_recording = False
                continue

            if reset_requested:
                if recorder.current_length > 0:
                    print(
                        f"Manual reset triggered at step {step}. Discarded {recorder.current_length} buffered frames."
                    )
                recorder.clear_buffer()
                obs, info = env.reset()
                set_goal_color(p)
                setup_grasp_physics(p, sim)
                controller.reset_gripper()
                episode += 1
                step = 0
                is_success = False
                is_recording = False
                continue

            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            is_success = bool(info.get("is_success", False))

            # In MANUAL mode: Disable step-limit / timeout termination
            if mode == "MANUAL":
                truncated = False

            # Buffer step transition into recorder
            # In MANUAL mode, record only when recording has been started by user (is_recording is True); other modes False
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
                if is_success:
                    saved_path = recorder.save_episode(
                        task_name="pick_and_place_around_obstacle"
                    )
                    if saved_path:
                        print(f"Demonstration saved to: {saved_path}")
                else:
                    recorder.clear_buffer()

                obs, info = env.reset()
                set_goal_color(p)
                setup_grasp_physics(p, sim)
                controller.reset_gripper()
                episode += 1
                step = 0
                is_success = False
                is_recording = False


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
