import cv2
import numpy as np


def capture_camera(p, yaw, pitch, distance, target_pos, label='', width=480, height=360):
    """Captures an RGB view matrix from PyBullet and converts it to BGR for OpenCV."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.1,
        farVal=3.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    if label:
        cv2.putText(
            bgr,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return bgr


def capture_wrist_camera(p, sim, label='2. Wrist View', width=480, height=360):
    """Captures a first-person eye-in-hand / wrist camera view mounted on the robot gripper."""
    # Link 11 is panda_grasptarget (centered between the fingertips)
    pos = np.array(sim.get_link_position('panda', 11))
    orn = sim.get_link_orientation('panda', 11)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    # Place camera slightly behind and above fingers looking forward/downward
    eye = pos + rot @ np.array([0.09, 0.0, -0.07])
    target = pos + rot @ np.array([0.0, 0.0, 0.10])
    up = rot[:, 0]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=70.0,
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=3.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    if label:
        cv2.putText(
            bgr,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return bgr


def render_multi_camera_dashboard(
    p,
    sim=None,
    target_pos=None,
    width=480,
    height=360,
    step=None,
    max_steps=None,
    episode=None,
    mode="MANUAL",
    action_label="IDLE",
    gripper_width=None,
    is_collision=False,
    is_success=False,
    demos_saved=None,
    rec_steps=None,
    is_recording=False,
    wait_ms=20,
    show_window=True,
):
    """Renders a 2x2 multi-camera grid with telemetry overlays and captures keyboard input."""
    if target_pos is None:
        target_pos = [0.0, 0.0, 0.0]

    img_back = capture_camera(p, yaw=180.0, pitch=-20.0, distance=0.8, target_pos=target_pos, label='1. Back View', width=width, height=height)
    img_wrist = capture_wrist_camera(p, sim, label='2. Wrist View', width=width, height=height)

    row1 = np.hstack([img_back, img_wrist])

    img_front = capture_camera(p, yaw=0.0, pitch=-20.0, distance=0.8, target_pos=target_pos, label='3. Front View', width=width, height=height)
    img_iso = capture_camera(p, yaw=30.0, pitch=-35.0, distance=1.0, target_pos=target_pos, label='4. Isometric', width=width, height=height)
    row2 = np.hstack([img_front, img_iso])

    dashboard = np.vstack([row1, row2])

    # Top Status & Controls Header Bar
    header_h = 44
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], header_h), (25, 25, 25), -1)

    # Mode Badge
    cv2.putText(dashboard, f"[{mode}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2, cv2.LINE_AA)

    # Action Indicator
    action_color = (0, 255, 255) if action_label != "IDLE" else (160, 160, 160)
    cv2.putText(dashboard, f"Act: {action_label}", (115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color, 1, cv2.LINE_AA)

    # Gripper Width
    if gripper_width is not None:
        grip_state = "OPEN" if gripper_width > 0.03 else "CLOSED"
        cv2.putText(dashboard, f"Grip: {gripper_width * 100:.1f}cm ({grip_state})", (505, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Collision Warning
    obs_text = "HIT!" if is_collision else "SAFE"
    obs_color = (0, 0, 255) if is_collision else (0, 220, 0)
    cv2.putText(dashboard, f"Obstacle: {obs_text}", (675, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, obs_color, 2, cv2.LINE_AA)

    # Demos Saved & Recording Telemetry
    if demos_saved is not None:
        cv2.putText(dashboard, f"Demos: {demos_saved}", (810, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 120, 255), 1, cv2.LINE_AA)

    rec_count = rec_steps if rec_steps is not None else 0
    if is_recording:
        # Pulsing / bright red dot with active frame count
        cv2.circle(dashboard, (826, 23), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(dashboard, f"REC: {rec_count}", (840, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
    elif rec_count > 0:
        # Paused recording state
        cv2.putText(dashboard, f"PAUSED ({rec_count})", (820, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2, cv2.LINE_AA)
    elif mode == "MANUAL":
        # Standby cue for the operator
        cv2.putText(dashboard, "REC: OFF [B]", (820, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    # Bottom Footer Bar
    if step is not None:
        status_text = f"Step: {step}"
        if max_steps is not None:
            status_text += f"/{max_steps}"
        if episode is not None:
            status_text = f"Ep: {episode} | " + status_text

        cv2.rectangle(dashboard, (10, dashboard.shape[0] - 42), (320, dashboard.shape[0] - 10), (20, 20, 20), -1)
        cv2.putText(
            dashboard,
            status_text,
            (18, dashboard.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Success Banner
    if is_success:
        cv2.rectangle(dashboard, (330, dashboard.shape[0] - 42), (650, dashboard.shape[0] - 10), (0, 140, 0), -1)
        cv2.putText(
            dashboard,
            "SUCCESS! [ENTER: Save | R: Discard]",
            (336, dashboard.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Color legend on bottom-right of the dashboard
    legend_text = 'Obj: Green | Goal: Red | Wall: Blue'
    badge_w = 295
    cv2.rectangle(
        dashboard,
        (dashboard.shape[1] - badge_w - 10, dashboard.shape[0] - 42),
        (dashboard.shape[1] - 10, dashboard.shape[0] - 10),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        dashboard,
        legend_text,
        (dashboard.shape[1] - badge_w, dashboard.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    key = -1
    if show_window:
        cv2.namedWindow('Multi-Camera Dashboard', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Multi-Camera Dashboard', dashboard)
        key = cv2.waitKey(wait_ms) & 0xFF

    return key, dashboard


def close_dashboard():
    """Destroys all OpenCV windows."""
    cv2.destroyAllWindows()
