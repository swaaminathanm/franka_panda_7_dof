import numpy as np
import gymnasium as gym
import panda_gym


def create_obstacle(p):
    """Creates a static 3D obstacle partition wall in PyBullet."""
    wall_thickness = 0.02  # 2 cm thick along X
    wall_width = 0.28  # 28 cm wide in the center (table is 70 cm wide)
    wall_height = 0.24  # Static 24 cm tall
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
            if rgba[3] < 0.5:
                p.changeVisualShape(i, -1, rgbaColor=rgba_color)


def setup_block_obj_goal_pos(env, corners_only: bool = False, near_obstacle: bool = False):
    """Restricts initial object and goal random sampling regions.
    
    Args:
        env: Gymnasium environment
        corners_only: If True, restricts goal sampling strictly to table edge corner regions [X: 0.23..0.28, Y: -0.14..0.14].
        near_obstacle: If True, places green block right next to the obstacle wall [X: -0.07..-0.04].
    """
    task = env.unwrapped.task
    if near_obstacle:
        # Place block close to obstacle wall (wall at X = -0.02m) with kinematic finger clearance [X: -0.12..-0.09]
        task.obj_range_low = np.array([-0.12, -0.08, 0.0])
        task.obj_range_high = np.array([-0.09, 0.08, 0.0])
    else:
        task.obj_range_low = np.array([-0.25, -0.10, 0.0])
        task.obj_range_high = np.array([-0.15, 0.10, 0.0])

    if corners_only:
        def sample_corner_goal():
            # Pick randomly between Far-Right (Top Right) and Near-Right (Bottom Right) corners
            # Table edge is at X = 0.25m. Setting max X = 0.225m ensures block (4cm) stays 100% on the table surface.
            corner_choice = env.unwrapped.np_random.integers(0, 2)
            if corner_choice == 0:    # 1. Far-Right Corner (Top Right in image)
                x = env.unwrapped.np_random.uniform(0.18, 0.225)
                y = env.unwrapped.np_random.uniform(0.06, 0.11)   # Positive Y (Top Right)
            else:                     # 2. Near-Right Corner (Bottom Right in image)
                x = env.unwrapped.np_random.uniform(0.18, 0.225)
                y = env.unwrapped.np_random.uniform(-0.11, -0.06)  # Negative Y (Bottom Right)
            return np.array([x, y, task.object_size / 2.0])

        task._sample_goal = sample_corner_goal
    else:
        # Standard table region (safely inside X <= 0.225m)
        task.goal_range_low = np.array([0.15, -0.10, 0.0])
        task.goal_range_high = np.array([0.225, 0.10, 0.0])

    return task


def setup_scene_physics(env):
    """Re-applies obstacle wall and red goal color for PyBullet."""
    sim = env.unwrapped.sim
    p = sim.physics_client
    obstacle_id = create_obstacle(p)
    set_goal_color(p)
    return obstacle_id


def extract_state(obs):
    """Extracts an enriched 34-dimensional wall-aware state vector from the Gymnasium observation dict."""
    observation = np.asarray(obs["observation"], dtype=np.float32)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)

    gripper_pos = observation[:3]
    obj_pos = achieved_goal
    goal_pos = desired_goal
    wall_pos = np.array([-0.02, 0.0, 0.12], dtype=np.float32)

    # Relative 3D displacement vectors
    rel_gripper_to_obj = (obj_pos - gripper_pos).astype(np.float32)
    rel_obj_to_goal = (goal_pos - obj_pos).astype(np.float32)
    rel_gripper_to_wall = (wall_pos - gripper_pos).astype(np.float32)

    # Scalar Euclidean distances
    dist_gripper_to_obj = np.array([np.linalg.norm(rel_gripper_to_obj)], dtype=np.float32)
    dist_obj_to_goal = np.array([np.linalg.norm(rel_obj_to_goal)], dtype=np.float32)
    dist_gripper_to_wall = np.array([np.linalg.norm(rel_gripper_to_wall)], dtype=np.float32)

    return np.concatenate([
        observation,
        desired_goal,
        rel_gripper_to_obj,
        rel_obj_to_goal,
        dist_gripper_to_obj,
        dist_obj_to_goal,
        rel_gripper_to_wall,
        dist_gripper_to_wall,
    ], axis=0).astype(np.float32)


def append_wall_features_to_30d(state_tensor):
    """Dynamically appends 4 wall features to a (..., 30) state tensor if needed."""
    import torch
    if state_tensor.shape[-1] == 34:
        return state_tensor
    if state_tensor.shape[-1] == 30:
        ee_pos = state_tensor[..., :3]
        wall_pos = torch.tensor([-0.02, 0.0, 0.12], dtype=state_tensor.dtype, device=state_tensor.device)
        rel_wall = wall_pos - ee_pos
        dist_wall = torch.norm(rel_wall, dim=-1, keepdim=True)
        return torch.cat([state_tensor, rel_wall, dist_wall], dim=-1)
    return state_tensor


def make_franka_env(max_episode_steps: int = 300, corners_only: bool = False, near_obstacle: bool = False):
    """Factory creating configured PandaPickAndPlace-v3 environment.
    
    Args:
        max_episode_steps: Max steps per episode (default: 300)
        corners_only: If True, restricts red goal target strictly to table corners.
        near_obstacle: If True, places green block right next to obstacle wall.
    """
    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_episode_steps)
    setup_block_obj_goal_pos(env, corners_only=corners_only, near_obstacle=near_obstacle)
    return env


if __name__ == "__main__":
    # Test env creation and state extraction
    env = make_franka_env()
    obs, info = env.reset()
    setup_scene_physics(env)
    state = extract_state(obs)

    print(f"Extracted state shape: {state.shape}")
    assert state.shape == (34,), f"Expected (34,), got {state.shape}"
    assert not np.isnan(state).any(), "State contains NaN!"

    env.close()
    print("[SUCCESS] env_utils.py test passed successfully!")
