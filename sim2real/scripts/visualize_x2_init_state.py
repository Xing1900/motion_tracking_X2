"""
Visualize the X2_INIT_STATE default pose in MuJoCo before training.

Usage:
    cd /home/liuguoxing/Documents/motion_tracking/sim2real
    uv run python scripts/visualize_x2_init_state.py

The script loads assets/x2/x2_ultra.xml, applies the joint positions from
active_adaptation.assets.X2.humanoid.X2_INIT_STATE, and opens a MuJoCo viewer.
"""
import argparse
import sys
import re
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Add project roots to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM2REAL_ROOT = PROJECT_ROOT / "sim2real"
ASSETS_DIR = SIM2REAL_ROOT / "assets"
sys.path.insert(0, str(SIM2REAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Stub out imports that humanoid.py needs only for type/spec definitions,
# so we can load X2_INIT_STATE / X2_JOINT_ORDER without installing mjlab/torch.
import types

class _AnyArgs:
    def __init__(self, *args, **kwargs):
        pass

_stub_mod = types.ModuleType("mjlab")
_stub_mod.actuator = types.ModuleType("mjlab.actuator")
_stub_mod.actuator.BuiltinPositionActuatorCfg = _AnyArgs
_stub_mod.entity = types.ModuleType("mjlab.entity")
_stub_mod.entity.EntityArticulationInfoCfg = _AnyArgs
class _EntityCfg:
    class InitialStateCfg:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
    def __init__(self, *args, **kwargs):
        self.data = types.SimpleNamespace(default_joint_pos=None)

_stub_mod.entity.EntityCfg = _EntityCfg
_stub_mod.utils = types.ModuleType("mjlab.utils")
_stub_mod.utils.os = types.ModuleType("mjlab.utils.os")
_stub_mod.utils.os.update_assets = lambda *args, **kwargs: None
sys.modules["mjlab"] = _stub_mod
sys.modules["mjlab.actuator"] = _stub_mod.actuator
sys.modules["mjlab.entity"] = _stub_mod.entity
sys.modules["mjlab.utils"] = _stub_mod.utils
sys.modules["mjlab.utils.os"] = _stub_mod.utils.os

# Load humanoid.py directly to avoid pulling in heavy dependencies (torch, etc.)
import importlib.util
_humanoid_path = PROJECT_ROOT / "active_adaptation" / "assets" / "X2" / "humanoid.py"
_spec = importlib.util.spec_from_file_location("x2_humanoid", str(_humanoid_path))
_x2_humanoid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_x2_humanoid)  # type: ignore
X2_INIT_STATE = _x2_humanoid.X2_INIT_STATE
X2_JOINT_ORDER = _x2_humanoid.X2_JOINT_ORDER


def resolve_init_state_to_array(joint_order: list[str], init_state) -> np.ndarray:
    """Resolve regex-based joint_pos dict from X2_INIT_STATE to a numeric array."""
    qpos = np.zeros(len(joint_order), dtype=np.float64)

    # Process patterns in reverse priority order so specific names override wildcard
    patterns = list(init_state.joint_pos.items())
    patterns.reverse()

    for pattern, value in patterns:
        regex = re.compile(pattern)
        for i, name in enumerate(joint_order):
            if regex.fullmatch(name):
                qpos[i] = float(value)

    return qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_image", type=str, default=None,
                        help="Save a rendered image to this path and exit (headless mode).")
    parser.add_argument("--azimuth", type=float, default=135.0,
                        help="Camera azimuth angle in degrees.")
    parser.add_argument("--elevation", type=float, default=-20.0,
                        help="Camera elevation angle in degrees.")
    args = parser.parse_args()

    xml_path = ASSETS_DIR / "x2" / "x2_ultra.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Floating base: qpos[:7], actuated joints: qpos[7:7+n_actuated]
    default_qpos = resolve_init_state_to_array(X2_JOINT_ORDER, X2_INIT_STATE)
    data.qpos[:3] = X2_INIT_STATE.pos
    data.qpos[3:7] = getattr(X2_INIT_STATE, "rot", (1.0, 0.0, 0.0, 0.0))
    data.qpos[7 : 7 + len(default_qpos)] = default_qpos

    mujoco.mj_forward(model, data)

    print("Current default qpos (actuated joints):")
    for name, val in zip(X2_JOINT_ORDER, default_qpos):
        print(f"  {name}: {val:.4f}")

    if args.save_image:
        renderer = mujoco.Renderer(model, height=480, width=640)
        # Configure a default camera looking at the robot from the front/side
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if cam.trackbodyid < 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.0
        cam.azimuth = args.azimuth
        cam.elevation = args.elevation
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        import cv2
        cv2.imwrite(args.save_image, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved image to {args.save_image}")
        return

    print("MuJoCo viewer opened.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
