from __future__ import annotations

from pathlib import Path
import re

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets


ASSET_PATH = Path(__file__).resolve().parent
GP02_XML = ASSET_PATH / "gp02_v3.xml"


def _get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, ASSET_PATH / "meshes", meshdir)
    return assets


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GP02_XML))
    # The source GMR XML contains torque motors for standalone replay. MJLab
    # attaches the position actuators configured below when building a training
    # scene, so retaining both sets creates duplicate actuator names.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    spec.assets = _get_assets(spec.meshdir)
    return spec


# Dataset/controller order. This must remain identical to the 24-DoF GP02
# order emitted by scripts/data_process/gmr_pkl_to_motion_tracking_npz.py.
GP02_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint",
)


JOINT_SYMMETRY_MAP = {
    "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
    "right_hip_pitch_joint": (1, "left_hip_pitch_joint"),
    "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
    "right_hip_roll_joint": (-1, "left_hip_roll_joint"),
    "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
    "right_hip_yaw_joint": (-1, "left_hip_yaw_joint"),
    "left_knee_joint": (1, "right_knee_joint"),
    "right_knee_joint": (1, "left_knee_joint"),
    "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
    "right_ankle_pitch_joint": (1, "left_ankle_pitch_joint"),
    "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
    "right_ankle_roll_joint": (-1, "left_ankle_roll_joint"),
    "waist_yaw_joint": (-1, "waist_yaw_joint"),
    "waist_roll_joint": (-1, "waist_roll_joint"),
    "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
    "right_shoulder_pitch_joint": (1, "left_shoulder_pitch_joint"),
    "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
    "right_shoulder_roll_joint": (-1, "left_shoulder_roll_joint"),
    "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
    "right_shoulder_yaw_joint": (-1, "left_shoulder_yaw_joint"),
    "left_elbow_joint": (1, "right_elbow_joint"),
    "right_elbow_joint": (1, "left_elbow_joint"),
    "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
    "right_wrist_yaw_joint": (-1, "left_wrist_yaw_joint"),
}


SPATIAL_SYMMETRY_MAP = {
    "pelvis": "pelvis",
    "waist_yaw_link": "waist_yaw_link",
    "torso_link": "torso_link",
}
for _left, _right in (
    ("left_hip_pitch_link", "right_hip_pitch_link"),
    ("left_hip_roll_link", "right_hip_roll_link"),
    ("left_hip_yaw_link", "right_hip_yaw_link"),
    ("left_knee_link", "right_knee_link"),
    ("left_ankle_pitch_link", "right_ankle_pitch_link"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
    ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
    ("left_shoulder_roll_link", "right_shoulder_roll_link"),
    ("left_shoulder_yaw_link", "right_shoulder_yaw_link"),
    ("left_elbow_link", "right_elbow_link"),
    ("left_wrist_yaw_link", "right_wrist_yaw_link"),
):
    SPATIAL_SYMMETRY_MAP[_left] = _right
    SPATIAL_SYMMETRY_MAP[_right] = _left


# The user-confirmed stable standing pose is zero for all joints. The root
# height comes from the GP02 MJCF's qpos0.
GP02_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.732850403921914),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


def _position_actuator(
    *patterns: str,
    stiffness: float,
    damping: float,
    effort_limit: float,
) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=patterns,
        armature=0.01,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
    )


# Leg/waist gains are the stable-standing values supplied for GP02. Arm gains
# use the conservative 40/5 setting found in the local GP02 controller config;
# they still need hardware validation before sim-to-real deployment.
GP02_ACTUATORS = (
    _position_actuator(".*_hip_pitch_joint", stiffness=100.0, damping=4.0, effort_limit=139.0),
    _position_actuator(".*_hip_roll_joint", stiffness=100.0, damping=3.0, effort_limit=88.0),
    _position_actuator(".*_hip_yaw_joint", stiffness=100.0, damping=3.0, effort_limit=88.0),
    _position_actuator(".*_knee_joint", stiffness=150.0, damping=5.0, effort_limit=139.0),
    _position_actuator(".*_ankle_pitch_joint", stiffness=40.0, damping=3.0, effort_limit=80.0),
    _position_actuator(".*_ankle_roll_joint", stiffness=30.0, damping=2.0, effort_limit=80.0),
    _position_actuator("waist_yaw_joint", stiffness=40.0, damping=5.0, effort_limit=88.0),
    _position_actuator("waist_roll_joint", stiffness=40.0, damping=5.0, effort_limit=50.0),
    _position_actuator(
        ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_yaw_joint",
        stiffness=40.0, damping=5.0, effort_limit=25.0,
    ),
)


def _validate_static_configuration() -> None:
    mapped = set()
    for actuator in GP02_ACTUATORS:
        patterns = actuator.target_names_expr
        for name in GP02_JOINT_ORDER:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                if name in mapped:
                    raise ValueError(f"GP02 joint matched by more than one actuator: {name}")
                mapped.add(name)
    if mapped != set(GP02_JOINT_ORDER):
        raise ValueError(f"GP02 actuator mapping incomplete: {sorted(set(GP02_JOINT_ORDER) - mapped)}")
    if set(JOINT_SYMMETRY_MAP) != set(GP02_JOINT_ORDER):
        raise ValueError("GP02 joint symmetry map must cover all canonical joints")


_validate_static_configuration()

GP02_ARTICULATION = EntityArticulationInfoCfg(
    actuators=GP02_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

GP02_CFG = EntityCfg(
    init_state=GP02_INIT_STATE,
    spec_fn=_get_spec,
    articulation=GP02_ARTICULATION,
)
GP02_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
GP02_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
GP02_CFG.joint_name_order = GP02_JOINT_ORDER

# Keep aliases consistent with the G1/X2 asset API.
GP02_V3 = GP02_CFG
GP02_V3_SELF = GP02_CFG
