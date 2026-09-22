"""Converter script: Transforms data/raw demonstrations (.npz) into Hugging Face LeRobotDataset format.

Features:
  - observation.state: (30,) float32 vector containing EE poses, velocities, gripper state,
                       object poses, goals, and relative vectors.
  - action: (4,) float32 vector containing [dx, dy, dz, gripper_action].
  - Standard LeRobot metadata (timestamp, frame_index, episode_index, tasks, stats).
"""

import argparse
import glob
import json
import os
import shutil
import numpy as np


def get_lerobot_features():
    """Defines the feature specification schema conforming to LeRobot standards (34D wall-aware state)."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (34,),
            "names": [
                # End-effector 3D position & linear velocity (6)
                "ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz",
                # Gripper width (1)
                "gripper_width",
                # Object position, rotation, linear velocity & angular velocity (12)
                "obj_x", "obj_y", "obj_z", "obj_roll", "obj_pitch", "obj_yaw",
                "obj_vx", "obj_vy", "obj_vz", "obj_wx", "obj_wy", "obj_wz",
                # Target desired goal (3)
                "goal_x", "goal_y", "goal_z",
                # Relative displacement vectors (6)
                "rel_ee_to_obj_x", "rel_ee_to_obj_y", "rel_ee_to_obj_z",
                "rel_obj_to_goal_x", "rel_obj_to_goal_y", "rel_obj_to_goal_z",
                # Scalar Euclidean distances (2)
                "dist_gripper_to_obj",
                "dist_obj_to_goal",
                # Obstacle wall relative displacement & scalar distance (4)
                "rel_gripper_to_wall_x", "rel_gripper_to_wall_y", "rel_gripper_to_wall_z",
                "dist_gripper_to_wall",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }


def convert_raw_to_lerobot(
    raw_dir="data/raw",
    output_dir="data/lerobot",
    repo_id="local/franka_panda_pick_and_place",
    fps=50,
    only_success=False,
):
    """Converts recorded .npz demonstration files into LeRobotDataset format."""
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"Error: Missing required packages: {e}")
        print("Please install them with: pip install lerobot torch pyarrow")
        return

    from env_utils import append_wall_features_to_30d

    npz_files = sorted(glob.glob(os.path.join(raw_dir, "episode_*.npz")))
    if not npz_files:
        print(f"[Converter] No .npz files found in {raw_dir}. Nothing to convert.")
        return

    print(f"[Converter] Found {len(npz_files)} raw episodes in {raw_dir}.")

    # Clean existing destination if needed
    if os.path.exists(output_dir):
        print(f"[Converter] Cleaning previous LeRobot output directory: {output_dir}")
        shutil.rmtree(output_dir)

    features = get_lerobot_features()

    print(f"[Converter] Initializing LeRobotDataset with repo_id='{repo_id}' at {output_dir}...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="panda",
        features=features,
        root=output_dir,
        use_videos=False,
    )

    task_description = "Pick up object and place it around obstacle onto the target goal"
    total_episodes = 0
    total_frames = 0

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))

        is_success = meta.get("final_success", False)
        if only_success and not is_success:
            print(f"  [Skip] {os.path.basename(npz_path)} (unsuccessful, final_success=False)")
            continue

        states = data["states"]    # (T, 30) or (T, 34)
        actions = data["actions"]  # (T, 4)
        num_frames = len(states)

        for t in range(num_frames):
            state_tensor = torch.from_numpy(states[t]).float()
            state_34 = append_wall_features_to_30d(state_tensor)
            frame = {
                "observation.state": state_34,
                "action": torch.from_numpy(actions[t]).float(),
                "task": task_description,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        total_episodes += 1
        total_frames += num_frames
        status_tag = "SUCCESS" if is_success else "RECORDED"
        print(f"  [Added] {os.path.basename(npz_path)}: {num_frames} frames [{status_tag}]")

    print(f"\n[Converter] Finalizing {total_episodes} episodes ({total_frames} frames)...")
    dataset.finalize()
    print(f"[Converter] Done! LeRobot dataset created successfully at: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert raw .npz demos to LeRobotDataset format")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw recordings folder")
    parser.add_argument("--output-dir", type=str, default="data/lerobot", help="Destination folder for LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local/franka_panda_pick_and_place", help="Dataset identifier")
    parser.add_argument("--fps", type=int, default=50, help="Recording FPS (default: 50)")
    parser.add_argument("--only-success", action="store_true", help="Convert only successful demonstrations")
    args = parser.parse_args()

    convert_raw_to_lerobot(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        only_success=args.only_success,
    )


if __name__ == "__main__":
    main()
