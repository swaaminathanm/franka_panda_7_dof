import numpy as np
import gymnasium as gym
import panda_gym


def create_obstacle(p):
    """Creates a static 3D obstacle partition wall in PyBullet."""
    wall_thickness = 0.02  # 2 cm thick along X
    wall_width = 0.28  # 28 cm wide in the center (table is 70 cm wide)
    wall_height = 0.24  # 24 cm tall
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


def setup_block_obj_goal_pos(env):
    """Restricts initial object and goal random sampling regions."""
    task = env.unwrapped.task
    task.obj_range_low = np.array([-0.25, -0.10, 0.0])
    task.obj_range_high = np.array([-0.15, 0.10, 0.0])
    task.goal_range_low = np.array([0.15, -0.10, 0.0])
    task.goal_range_high = np.array([0.25, 0.10, 0.0])
    return task


def setup_grasp_physics(p, sim):
    """Enhances contact friction on gripper finger pads for firm, slip-free grasping."""
    robot_id = sim._bodies_idx.get("panda", 0)
    for finger_link in (9, 10):
        p.changeDynamics(
            robot_id,
            finger_link,
            lateralFriction=3.0,
            spinningFriction=0.1,
            rollingFriction=0.1,
            frictionAnchor=1,
        )


def setup_scene_physics(env):
    """Re-applies obstacle wall, red goal color, and rubber gripper physics for PyBullet."""
    sim = env.unwrapped.sim
    p = sim.physics_client
    obstacle_id = create_obstacle(p)
    set_goal_color(p)
    setup_grasp_physics(p, sim)
    return obstacle_id


def extract_state(obs):
    """Extracts an enriched 30-dimensional state vector from the Gymnasium observation dict."""
    observation = np.asarray(obs["observation"], dtype=np.float32)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)

    gripper_pos = observation[:3]
    obj_pos = achieved_goal
    goal_pos = desired_goal

    # Relative 3D displacement vectors
    rel_gripper_to_obj = (obj_pos - gripper_pos).astype(np.float32)
    rel_obj_to_goal = (goal_pos - obj_pos).astype(np.float32)

    # Scalar Euclidean distances
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


def make_franka_env(max_episode_steps: int = 300):
    """Factory creating configured PandaPickAndPlace-v3 environment."""
    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_episode_steps)
    setup_block_obj_goal_pos(env)
    return env


if __name__ == "__main__":
    # Test env creation and state extraction
    env = make_franka_env()
    obs, info = env.reset()
    setup_scene_physics(env)
    state = extract_state(obs)

    print(f"Extracted state shape: {state.shape}")
    assert state.shape == (30,), f"Expected (30,), got {state.shape}"
    assert not np.isnan(state).any(), "State contains NaN!"

    env.close()
    print("[SUCCESS] env_utils.py test passed successfully!")
