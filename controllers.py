import numpy as np


class BaseController:
    """Abstract base class for Panda arm controllers."""

    def get_action(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement get_action().")


class ManualController(BaseController):
    """Manual keyboard teleoperation controller for Franka Panda.

    Key mappings:
      W / S       : Move End-Effector +X / -X (Forward / Backward)
      A / D       : Move End-Effector +Y / -Y (Left / Right)
      E / Q       : Move End-Effector +Z / -Z (Up / Down)
      O           : Open Gripper (+1.0)
      C           : Close Gripper (-1.0)
      SPACE       : Hold Position / Zero Action
      B / T       : Start / Pause Recording (Toggle)
      ENTER       : Save Demonstration Episode
      R           : Reset Episode
      ESC         : Quit Application
    """

    def __init__(self, step_size=0.4, gripper_speed=1.0):
        self.step_size = step_size
        self.gripper_speed = gripper_speed
        self.gripper_state = 1.0  # Start with gripper open (+1.0)

    def reset_gripper(self):
        """Resets the persistent gripper state to open."""
        self.gripper_state = 1.0

    def get_action(self, key, action_dim=4):
        """Translates keyboard keycode into a continuous action array, status label, reset flag, save flag, and record toggle flag.

        Args:
            key (int): OpenCV waitKey ASCII keycode.
            action_dim (int): Action dimension (3 for XYZ only, 4 for XYZ + gripper).

        Returns:
            tuple[np.ndarray, str, bool, bool, bool]: (action, label, reset_requested, save_requested, record_toggle_requested)
        """
        action = np.zeros(action_dim, dtype=np.float32)
        label = "IDLE"
        reset_requested = False
        save_requested = False
        record_toggle_requested = False

        # Maintain persistent grip state so fingers continuously squeeze the object during motion
        if action_dim >= 4:
            action[3] = self.gripper_state

        if key in (ord('w'), ord('W')):
            action[0] = +self.step_size
            label = "FWD (+X)"
        elif key in (ord('s'), ord('S')):
            action[0] = -self.step_size
            label = "BACK (-X)"
        elif key in (ord('a'), ord('A')):
            action[1] = +self.step_size
            label = "LEFT (+Y)"
        elif key in (ord('d'), ord('D')):
            action[1] = -self.step_size
            label = "RIGHT (-Y)"
        elif key in (ord('e'), ord('E')):
            action[2] = +self.step_size
            label = "UP (+Z)"
        elif key in (ord('q'), ord('Q')):
            action[2] = -self.step_size
            label = "DOWN (-Z)"
        elif action_dim >= 4 and key in (ord('o'), ord('O')):
            self.gripper_state = +self.gripper_speed
            action[3] = self.gripper_state
            label = "GRIP OPEN"
        elif action_dim >= 4 and key in (ord('c'), ord('C')):
            self.gripper_state = -self.gripper_speed
            action[3] = self.gripper_state
            label = "GRIP CLOSE"
        elif key in (ord('r'), ord('R'), ord('z'), ord('Z')):
            reset_requested = True
            label = "RESET"
        elif key in (ord('b'), ord('B'), ord('t'), ord('T')):
            record_toggle_requested = True
            label = "REC TOGGLE"
        elif key in (10, 13):  # ENTER key
            save_requested = True
            label = "SAVE"
        elif key == ord(' '):
            label = "HOLD"

        return action, label, reset_requested, save_requested, record_toggle_requested


class AgentController(BaseController):
    """Placeholder for autonomous policies (e.g. Flow Matching, DAgger)."""

    def __init__(self, policy_model=None):
        self.policy_model = policy_model

    def get_action(self, obs, key=None, action_dim=4):
        raise NotImplementedError("Agent controller is not implemented yet.")
