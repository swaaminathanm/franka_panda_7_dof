import json
import os

import numpy as np


def extract_state(obs):
    """Extracts an enriched 30-dimensional state vector from the Gymnasium observation dict.

    Feature Breakdown:
      1. observation (19-d):
         - End-effector position (3) & linear velocity (3)
         - Gripper fingers width (1)
         - Object position (3), rotation roll-pitch-yaw (3), linear vel (3), angular vel (3)
      2. desired_goal (3-d):
         - Target goal position (x, y, z)
      3. rel_gripper_to_obj (3-d):
         - 3D relative displacement vector from gripper EE to object: (obj_pos - ee_pos)
      4. rel_obj_to_goal (3-d):
         - 3D relative displacement vector from object to target goal: (goal_pos - obj_pos)
      5. dist_gripper_to_obj (1-d):
         - Euclidean scalar distance between gripper EE and object
      6. dist_obj_to_goal (1-d):
         - Euclidean scalar distance between object and target goal

    Total Dimension: 19 + 3 + 3 + 3 + 1 + 1 = 30
    """
    observation = np.asarray(obs["observation"], dtype=np.float32)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)

    gripper_pos = observation[:3]
    obj_pos = achieved_goal
    goal_pos = desired_goal

    # Relative 3D displacement vectors (direct guides for XYZ delta actions)
    rel_gripper_to_obj = (obj_pos - gripper_pos).astype(np.float32)
    rel_obj_to_goal = (goal_pos - obj_pos).astype(np.float32)

    # Scalar Euclidean distances (direct indicators for phase transitions)
    dist_gripper_to_obj = np.array([np.linalg.norm(rel_gripper_to_obj)], dtype=np.float32)
    dist_obj_to_goal = np.array([np.linalg.norm(rel_obj_to_goal)], dtype=np.float32)

    return np.concatenate([
        observation,
        desired_goal,
        rel_gripper_to_obj,
        rel_obj_to_goal,
        dist_gripper_to_obj,
        dist_obj_to_goal,
    ], axis=0).astype(np.float32)


class EpisodeRecorder:
    """Records, buffers, and persists robot demonstration trajectories to disk."""

    def __init__(self, save_dir="data/raw", fps=50):
        self.save_dir = save_dir
        self.fps = fps
        self.dt = 1.0 / fps
        os.makedirs(self.save_dir, exist_ok=True)

        self.current_buffer = []
        self.saved_count = self._count_existing_episodes()

    def _count_existing_episodes(self):
        """Finds the next index for saved episodes in save_dir to prevent overwrites."""
        if not os.path.exists(self.save_dir):
            return 0
        indices = []
        for f in os.listdir(self.save_dir):
            if f.startswith("episode_") and f.endswith(".npz"):
                part = f[len("episode_"):-len(".npz")]
                if part.isdigit():
                    indices.append(int(part))
        return max(indices) + 1 if indices else 0

    def record_step(self, obs, action, reward, terminated, truncated, info, is_collision=False):
        """Buffers a single simulation step transition with enriched state and metrics."""
        step_idx = len(self.current_buffer)

        state = extract_state(obs)

        desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)
        gripper_pos = np.asarray(obs["observation"][:3], dtype=np.float32)

        dist_gripper_to_obj = float(np.linalg.norm(achieved_goal - gripper_pos))
        dist_obj_to_goal = float(np.linalg.norm(desired_goal - achieved_goal))

        transition = {
            "step": step_idx,
            "timestamp": float(step_idx * self.dt),
            "state": state,
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "is_success": bool(info.get("is_success", False)),
            "is_collision": bool(is_collision),
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
            "dist_gripper_to_obj": dist_gripper_to_obj,
            "dist_obj_to_goal": dist_obj_to_goal,
        }
        self.current_buffer.append(transition)

    def save_episode(self, task_name="pick_and_place_around_obstacle"):
        """Saves the buffered episode transitions to a compressed .npz archive and companion JSON."""
        if len(self.current_buffer) == 0:
            print("[Recorder] Warning: Current buffer is empty, nothing to save.")
            return None

        states = np.stack([t["state"] for t in self.current_buffer])
        actions = np.stack([t["action"] for t in self.current_buffer])
        rewards = np.array([t["reward"] for t in self.current_buffer], dtype=np.float32)
        dones = np.array([t["terminated"] or t["truncated"] for t in self.current_buffer], dtype=bool)
        successes = np.array([t["is_success"] for t in self.current_buffer], dtype=bool)
        collisions = np.array([t["is_collision"] for t in self.current_buffer], dtype=bool)
        timestamps = np.array([t["timestamp"] for t in self.current_buffer], dtype=np.float32)
        desired_goals = np.stack([t["desired_goal"] for t in self.current_buffer])
        achieved_goals = np.stack([t["achieved_goal"] for t in self.current_buffer])
        dists_gripper_to_obj = np.array([t["dist_gripper_to_obj"] for t in self.current_buffer], dtype=np.float32)
        dists_obj_to_goal = np.array([t["dist_obj_to_goal"] for t in self.current_buffer], dtype=np.float32)

        episode_idx = self.saved_count
        filename = f"episode_{episode_idx:06d}.npz"
        filepath = os.path.join(self.save_dir, filename)
        json_path = os.path.join(self.save_dir, f"episode_{episode_idx:06d}.json")

        metadata = {
            "episode_index": episode_idx,
            "total_frames": len(self.current_buffer),
            "fps": self.fps,
            "duration_seconds": round(float(len(self.current_buffer) * self.dt), 3),
            "task_name": task_name,
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "has_collision": bool(np.any(collisions)),
            "final_success": bool(successes[-1]) if len(successes) > 0 else False,
            "min_dist_gripper_to_obj": round(float(np.min(dists_gripper_to_obj)), 4) if len(dists_gripper_to_obj) > 0 else None,
            "final_dist_obj_to_goal": round(float(dists_obj_to_goal[-1]), 4) if len(dists_obj_to_goal) > 0 else None,
            "desired_goal": desired_goals[-1].tolist() if len(desired_goals) > 0 else [],
        }

        # Save compressed numpy archive
        np.savez_compressed(
            filepath,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            successes=successes,
            collisions=collisions,
            timestamps=timestamps,
            desired_goals=desired_goals,
            achieved_goals=achieved_goals,
            dists_gripper_to_obj=dists_gripper_to_obj,
            dists_obj_to_goal=dists_obj_to_goal,
            metadata=json.dumps(metadata),
        )

        # Save human-readable JSON metadata
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.saved_count += 1
        num_frames = len(self.current_buffer)
        self.clear_buffer()

        print(f"\n>>> [RECORDER] Saved demonstration #{episode_idx} ({num_frames} frames) to: {filepath}\n")
        return filepath

    def clear_buffer(self):
        """Clears the active transition buffer without saving."""
        self.current_buffer.clear()

    @property
    def current_length(self):
        """Returns the number of buffered transitions in the current episode."""
        return len(self.current_buffer)
