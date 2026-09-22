import numpy as np
import gymnasium as gym


class RichRewardFrankaWrapper(gym.Wrapper):
    """Gymnasium Wrapper augmenting PandaPickAndPlace-v3 with progress, alignment & re-grasp recovery.
    """

    def __init__(self, env, obstacle_id: int = None):
        super().__init__(env)
        self.obstacle_id = obstacle_id
        self.prev_dist_obj_goal = None
        self.lift_bonus_given = False
        self.wall_x = -0.02
        self.wall_height = 0.24  # 24 cm tall partition wall
        self.required_clearance = 0.285  # 28.5 cm EE height required to clear 24cm wall cleanly

    def set_obstacle_id(self, obstacle_id: int):
        self.obstacle_id = obstacle_id

    def set_wall_height(self, wall_height: float):
        self.wall_height = float(wall_height)
        self.required_clearance = float(wall_height + 0.045)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obj_pos = obs["achieved_goal"][:3]
        goal_pos = obs["desired_goal"][:3]
        self.prev_dist_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))
        self.lift_bonus_given = False
        return obs, info

    def compute_rich_reward(self, obs, action, info) -> float:
        """Compute progress-driven reward signal with re-grasp attraction & wall clearance."""
        sim = self.env.unwrapped.sim
        bullet_p = sim.physics_client

        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"][:3]
        goal_pos = obs["desired_goal"][:3]

        # 3D relative positions
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_xy_ee_obj = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        dist_z_ee_obj = float(abs(ee_pos[2] - obj_pos[2]))
        dist_obj_goal = float(np.linalg.norm(obj_pos - goal_pos))

        if self.prev_dist_obj_goal is None:
            self.prev_dist_obj_goal = dist_obj_goal

        reward = 0.0

        # Physical Contact Check via PyBullet
        robot_id = sim._bodies_idx.get("panda", 0)
        obj_id = sim._bodies_idx.get("object", 1)
        contacts_obj = bullet_p.getContactPoints(bodyA=robot_id, bodyB=obj_id)
        has_physical_contact = len(contacts_obj) > 0

        # Perception & Grasp State Classification
        is_obs_grasped = (dist_ee_obj < 0.035) and has_physical_contact
        is_obs_lifted = is_obs_grasped and (obj_pos[2] > 0.03)
        is_gripper_aligned_low = (dist_xy_ee_obj < 0.035) and (dist_z_ee_obj < 0.025)
        is_success = bool(info.get("is_success", False)) or (dist_obj_goal < 0.05)
        is_out_of_bounds = (
            abs(ee_pos[0]) > 0.32 or
            abs(ee_pos[1]) > 0.30 or
            ee_pos[2] > 0.42 or
            ee_pos[2] < 0.00
        )
        
        # 1. Reaching & Initial Alignment Phase
        if not is_obs_grasped:
            reward -= 5.0 * dist_ee_obj  # Pull EE toward block
            # Encourage closing fingers when aligned low over block
            if is_gripper_aligned_low and action[3] < -0.1:
                reward += 3.0

        # 2. Carrying & Lifting Phase
        if is_obs_lifted:
            if not self.lift_bonus_given:
                reward += 15.0  # Awarded ONCE per episode
                self.lift_bonus_given = True

            # Penalize relaxing grip / opening fingers while carrying
            if action[3] >= -0.1:
                reward -= 6.0

        # 3. Goal Progress Reward & Direction Alignment (Symmetric +25 / -25)
        if is_obs_grasped:
            goal_progress = self.prev_dist_obj_goal - dist_obj_goal
            if goal_progress > 0:
                reward += 25.0 * goal_progress  # Reward moving closer
            else:
                reward -= 25.0 * abs(goal_progress)  # Symmetric penalty for moving away

        self.prev_dist_obj_goal = dist_obj_goal

        # 4. Out-of-Bounds / High-Air Wandering Penalty
        if is_out_of_bounds:
            reward -= 15.0

        # 5. Wall-Crossing Altitude Shaping (enforced across X in [-0.15, 0.15])
        if is_obs_grasped and (-0.15 < obj_pos[0] < 0.15):
            clearance_gap = self.required_clearance - ee_pos[2]
            if clearance_gap > 0:
                reward -= 10.0 * clearance_gap  # Smooth penalty for flying below 28.5cm
            else:
                reward += 2.0  # Continuous safe clearance bonus

        # 6. Obstacle Wall Collision Penalty (Pure PyBullet Contact Detection)
        is_wall_collision = False
        if self.obstacle_id is not None:
            contacts = bullet_p.getContactPoints(bodyA=self.obstacle_id)
            if len(contacts) > 0:
                is_wall_collision = True

        if is_wall_collision:
            reward -= 20.0  # Strict wall collision penalty

        # 7. Task Success Bonus (+150.0)
        if is_success:
            reward += 150.0

        # 8. Per-Step Time Penalty (encourages minimal step completion)
        reward -= 0.5

        return float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.compute_rich_reward(obs, action, info)
        return obs, reward, terminated, truncated, info