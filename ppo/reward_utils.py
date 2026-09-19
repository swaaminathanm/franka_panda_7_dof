import numpy as np
import gymnasium as gym


class RichRewardFrankaWrapper(gym.Wrapper):
    """Gymnasium Wrapper augmenting PandaPickAndPlace-v3 with progress & obstacle clearance rewards.

    Fixes:
      1. Hover-Farming: Progress-based reward (reward actual movement toward goal).
      2. Edge-Grasping: Centered grasp threshold (< 2.2cm).
      3. Obstacle Wall Lock: Adds Vertical Wall Clearance & Escape Upward reward (forces +dz lift to clear 24cm wall).
    """

    def __init__(self, env, obstacle_id: int = None):
        super().__init__(env)
        self.obstacle_id = obstacle_id
        self.prev_dist_obj_goal = None
        self.lift_bonus_given = False
        self.wall_x = -0.02
        self.wall_height = 0.24  # 24 cm tall partition wall

    def set_obstacle_id(self, obstacle_id: int):
        self.obstacle_id = obstacle_id

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obj_pos = obs["achieved_goal"][:3]
        goal_pos = obs["desired_goal"][:3]
        self.prev_dist_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))
        self.lift_bonus_given = False
        return obs, info

    def compute_rich_reward(self, obs, action, info) -> float:
        """Compute progress-driven reward signal with wall clearance & escape vectors."""
        sim = self.env.unwrapped.sim
        bullet_p = sim.physics_client

        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"][:3]
        goal_pos = obs["desired_goal"][:3]

        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))

        if self.prev_dist_obj_goal is None:
            self.prev_dist_obj_goal = dist_obj_goal

        reward = 0.0

        # 1. Reaching: -dist(gripper, object)
        reward -= dist_ee_obj

        is_obs_lifted = obj_pos[2] > 0.02
        is_obs_grasped = (dist_ee_obj < 0.022) and is_obs_lifted
        is_success = bool(info.get("is_success", False)) or (dist_obj_goal < 0.05)

        # 2. One-time Lifting Bonus (+15.0)
        if is_obs_grasped and not self.lift_bonus_given:
            reward += 15.0
            self.lift_bonus_given = True

        # 3. Goal Progress Reward (Reward actual movement TOWARD goal)
        goal_progress = self.prev_dist_obj_goal - dist_obj_goal
        if is_obs_grasped:
            if goal_progress > 0:
                reward += 20.0 * goal_progress  # Reward moving closer to goal
            else:
                reward -= 0.5  # Penalize hovering / standing still

            reward -= 1.0 * dist_obj_goal

        self.prev_dist_obj_goal = dist_obj_goal

        # 4. Obstacle Wall Clearance & Vertical Lift Shaping
        dist_to_wall_x = abs(ee_pos[0] - self.wall_x)
        if dist_to_wall_x < 0.08:  # Near partition wall region
            if ee_pos[2] >= self.wall_height:
                reward += 3.0  # Reward clearing top of wall (z >= 24cm)
            else:
                reward -= 5.0 * (self.wall_height - ee_pos[2])  # Penalize low altitude near wall

        # 5. Obstacle Collision & Escape Reward
        if self.obstacle_id is not None:
            contacts = bullet_p.getContactPoints(bodyA=self.obstacle_id)
            if len(contacts) > 0:
                reward -= 2.0  # Capped step penalty for wall contact
                # Reward UPWARD action vector (+dz) to force lifting out of wall contact
                if action[2] > 0:
                    reward += 5.0 * action[2]  # Reward lifting UP to escape wall contact
                else:
                    reward -= 2.0  # Penalize trying to push horizontally/downward into wall

        # 6. Task Success Bonus (+150.0)
        if is_success:
            reward += 150.0

        # 7. Action Effort Penalty
        reward -= 0.01 * np.sum(action ** 2)

        return float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.compute_rich_reward(obs, action, info)
        return obs, reward, terminated, truncated, info