# Franka Panda 7-DoF Simulation & Data Collection

## 1. Setup & Installation

```powershell
# 1. Create a virtual environment (Python 3.10 recommended)
python -m venv venv

# 2. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** Ensure `opencv-python` (GUI version) is installed. Do **not** install `opencv-python-headless`, as it disables the camera dashboard window.

---

## 2. Running the Simulation

```powershell
python main.py
```

### Optional Arguments:
- `--mode MANUAL`: Teleoperation with unlimited steps (default).
- `--save-dir data/raw`: Folder where demonstration `.npz` files are saved.
- `--record-idle`: Record frames even when no keys are pressed (default: `False`).

---

## 3. Teleoperation Controls

Focus on the **"Multi-Camera Dashboard"** window to operate:

| Key | Action |
| :---: | :--- |
| **W / S** | Move End-Effector Forward / Backward ($\pm X$) |
| **A / D** | Move End-Effector Left / Right ($\pm Y$) |
| **E / Q** | Move End-Effector Up / Down ($\pm Z$) |
| **O / C** | Open / Close Gripper (continuous squeeze lock) |
| **SPACE** | Hold Position (zero action) |
| **B / T** | **Start / Pause Recording** (Toggle) |
| **ENTER** | **Save Demonstration Episode** |
| **R** | **Reset Episode** (discards unsaved buffer) |
| **ESC** | **Exit Simulation** |

---

## 4. Converting to LeRobot Dataset

After gathering demonstrations in `data/raw/`, convert them into Hugging Face `LeRobotDataset` format (Apache Parquet + normalization stats):

```powershell
python convert_to_lerobot.py
```

- Output directory: `data/lerobot/`
- Add `--only-success` to convert only successful episodes (`final_success=True`).
