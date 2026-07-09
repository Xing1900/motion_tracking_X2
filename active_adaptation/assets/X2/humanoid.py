from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

ASSET_PATH = os.path.dirname(__file__)
X2_XML = Path(ASSET_PATH) / "x2.xml"


def _get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, X2_XML.parent / "meshes", meshdir)
    return assets


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(X2_XML))
    spec.assets = _get_assets(spec.meshdir)
    return spec


# Manual symmetry maps (explicit control).
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
    "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    "right_wrist_pitch_joint": (1, "left_wrist_pitch_joint"),
    "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
    "right_wrist_roll_joint": (-1, "left_wrist_roll_joint"),
    "waist_yaw_joint": (-1, "waist_yaw_joint"),
    "waist_pitch_joint": (1, "waist_pitch_joint"),
    "waist_roll_joint": (-1, "waist_roll_joint"),
}

SPATIAL_SYMMETRY_MAP = {
    "left_hip_pitch_link": "right_hip_pitch_link",
    "right_hip_pitch_link": "left_hip_pitch_link",
    "left_hip_roll_link": "right_hip_roll_link",
    "right_hip_roll_link": "left_hip_roll_link",
    "left_hip_yaw_link": "right_hip_yaw_link",
    "right_hip_yaw_link": "left_hip_yaw_link",
    "left_knee_link": "right_knee_link",
    "right_knee_link": "left_knee_link",
    "left_ankle_pitch_link": "right_ankle_pitch_link",
    "right_ankle_pitch_link": "left_ankle_pitch_link",
    "left_ankle_roll_link": "right_ankle_roll_link",
    "right_ankle_roll_link": "left_ankle_roll_link",
    "left_toe_link": "right_toe_link",
    "right_toe_link": "left_toe_link",
    "pelvis": "pelvis",
    "waist_yaw_link": "waist_yaw_link",
    "waist_pitch_link": "waist_pitch_link",
    "torso_link": "torso_link",
    "left_shoulder_pitch_link": "right_shoulder_pitch_link",
    "right_shoulder_pitch_link": "left_shoulder_pitch_link",
    "left_shoulder_roll_link": "right_shoulder_roll_link",
    "right_shoulder_roll_link": "left_shoulder_roll_link",
    "left_shoulder_yaw_link": "right_shoulder_yaw_link",
    "right_shoulder_yaw_link": "left_shoulder_yaw_link",
    "left_elbow_link": "right_elbow_link",
    "right_elbow_link": "left_elbow_link",
    "left_wrist_yaw_link": "right_wrist_yaw_link",
    "right_wrist_yaw_link": "left_wrist_yaw_link",
    "left_wrist_pitch_link": "right_wrist_pitch_link",
    "right_wrist_pitch_link": "left_wrist_pitch_link",
    "left_wrist_roll_link": "right_wrist_roll_link",
    "right_wrist_roll_link": "left_wrist_roll_link",
    "head_yaw_link": "head_yaw_link",
    "head_pitch_link": "head_pitch_link",
}

X2_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.78),
    joint_pos={
        ".*_hip_pitch_joint": -0.28,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": -0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
        ".*_wrist_yaw_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_roll_joint": 0.0,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Actuator parameters are initial guesses based on G1 tuning and robot structure.
# Fine-tuning may be required for best sim-to-real transfer.
X2_ACTUATOR_UPPER = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
    ),
    armature=0.003609725,
    stiffness=14.25062309787429,
    damping=0.907222843292423,
    effort_limit=36.0,
)
X2_ACTUATOR_HIP_YAW_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    armature=0.010177520,
    stiffness=40.17923847137318,
    damping=2.5578897650279457,
    effort_limit=120.0,
)
X2_ACTUATOR_HIP_PITCH_ROLL_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
    armature=0.025101925,
    stiffness=99.09842777666113,
    damping=6.3088018534966395,
    effort_limit=120.0,
)
X2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=48.0,
)
X2_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=36.0,
)

X2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        X2_ACTUATOR_UPPER,
        X2_ACTUATOR_HIP_PITCH_ROLL_KNEE,
        X2_ACTUATOR_HIP_YAW_WAIST_YAW,
        X2_ACTUATOR_WAIST,
        X2_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# Actuated joint order in qpos (after freejoint), must match x2.xml.
X2_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)

X2_CFG = EntityCfg(
    init_state=X2_INIT_STATE,
    spec_fn=_get_spec,
    articulation=X2_ARTICULATION,
)

X2_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
X2_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
X2_CFG.joint_name_order = X2_JOINT_ORDER

X2_ULTRA = X2_CFG
X2_ULTRA_SELF = X2_CFG
