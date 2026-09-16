import numpy as np
import torch

from env_utils import extract_state


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

    def get_action(self, key, obs=None, action_dim=4):
        """Translates keyboard keycode into a continuous action array, status label, reset flag, save flag, and record toggle flag.

        Args:
            key (int): OpenCV waitKey ASCII keycode.
            obs (dict, optional): Observation dict (unused in manual mode).
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
    """Abstract controller for autonomous policy rollouts."""

    def __init__(self, policy=None):
        self.policy = policy

    def get_action(self, obs, key=None, action_dim=4):
        raise NotImplementedError("Subclasses must implement get_action().")


class FlowMatchingController(AgentController):
    """Autonomous Flow Matching policy controller using Receding Horizon Control (RHC)."""

    def __init__(self, policy, k_exec: int = 4, ode_steps: int = 10):
        super().__init__(policy=policy)
        self.k_exec = k_exec
        self.ode_steps = ode_steps
        self.action_buffer = []

    def reset_buffer(self):
        """Clears the Receding Horizon action buffer upon episode reset."""
        self.action_buffer = []

    def get_action(self, obs, key=None, action_dim=4):
        """Generates policy actions using Receding Horizon Control (RHC).

        Args:
            obs (dict): Current Gym observation dictionary.
            key (int, optional): OpenCV waitKey code (used for ESC detection).
            action_dim (int): Action dimension.

        Returns:
            tuple[np.ndarray, str, bool, bool, bool]: (action, label, reset_requested, save_requested, record_toggle_requested)
        """
        reset_requested = False
        save_requested = False
        record_toggle_requested = False

        # If buffer is empty, query policy for new 16-step action chunk
        if len(self.action_buffer) == 0:
            state_vec = extract_state(obs)
            device = next(self.policy.parameters()).device
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)

            with torch.no_grad():
                chunk = self.policy.sample_actions(state_tensor, num_steps=self.ode_steps)
                chunk = chunk.squeeze(0).cpu().numpy()  # (16, 4)

            num_exec = min(self.k_exec, chunk.shape[0])
            self.action_buffer = list(chunk[:num_exec])

        action = self.action_buffer.pop(0)
        label = "AGENT (Flow)"

        return action, label, reset_requested, save_requested, record_toggle_requested
