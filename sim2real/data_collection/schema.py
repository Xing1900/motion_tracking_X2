"""Shared schema constants for X2 VR demonstration data.

The ordering below deliberately mirrors ``BaseConfig.seq`` in
``x1-motion-control/x1_digit_mc/src/rl_module/cfg/rl/rl_tracking.yaml``.
Do not derive it from the legacy teleop NPZ recorder: that file labels waist
and wrist axes in a different order.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


RAW_DATASET_SCHEMA_VERSION = "x2-vr-raw-v1"
TELEOP_TAP_SCHEMA_VERSION = 1

BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES = (
    "actual_human_height",
    "gmr_max_iter",
    "lookback_ms",
    "min_link_height",
    "min_link_height_align_strategy",
    "min_link_height_bootstrap_frames",
)


def normalize_bridge_runtime_effective_params(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize the bridge settings that affect labels."""

    if not isinstance(payload, Mapping):
        raise ValueError("bridge runtime effective params must be a mapping")
    expected = set(BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "bridge runtime effective params keys mismatch: "
            f"missing={missing}, extra={extra}"
        )

    def finite_float(name: str) -> float:
        value = payload[name]
        if isinstance(value, bool):
            raise ValueError(f"bridge runtime param {name} must be numeric")
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"bridge runtime param {name} must be numeric") from exc
        if not math.isfinite(normalized):
            raise ValueError(f"bridge runtime param {name} must be finite")
        return normalized

    def strict_int(name: str, *, minimum: int) -> int:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"bridge runtime param {name} must be an integer")
        if value < minimum:
            raise ValueError(f"bridge runtime param {name} must be >= {minimum}")
        return int(value)

    actual_human_height = finite_float("actual_human_height")
    if actual_human_height <= 0.0:
        raise ValueError("bridge runtime param actual_human_height must be > 0")
    lookback_ms = finite_float("lookback_ms")
    if lookback_ms < 0.0:
        raise ValueError("bridge runtime param lookback_ms must be >= 0")
    strategy = payload["min_link_height_align_strategy"]
    if not isinstance(strategy, str) or strategy not in {"startup_fixed", "per_frame"}:
        raise ValueError(
            "bridge runtime param min_link_height_align_strategy must be "
            "startup_fixed or per_frame"
        )

    return {
        "actual_human_height": actual_human_height,
        "gmr_max_iter": strict_int("gmr_max_iter", minimum=0),
        "lookback_ms": lookback_ms,
        "min_link_height": finite_float("min_link_height"),
        "min_link_height_align_strategy": strategy,
        "min_link_height_bootstrap_frames": strict_int(
            "min_link_height_bootstrap_frames", minimum=1
        ),
    }


X2_TRACKING_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

X2_HEAD_JOINT_NAMES = ["head_yaw_joint", "head_pitch_joint"]

XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]

REFERENCE_ACTION_NAMES = [
    "reference.root_pos.x",
    "reference.root_pos.y",
    "reference.root_pos.z",
    "reference.root_quat.w",
    "reference.root_quat.x",
    "reference.root_quat.y",
    "reference.root_quat.z",
    *[f"reference.{name}" for name in X2_TRACKING_JOINT_NAMES],
]

HAND_ACTION_NAMES = [
    "hand.left_grasp_fraction",
    "hand.right_grasp_fraction",
]

ACTION_NAMES = [*REFERENCE_ACTION_NAMES, *HAND_ACTION_NAMES]

OBSERVATION_STATE_NAMES = [
    *[f"joint_position.{name}" for name in X2_TRACKING_JOINT_NAMES],
    *[f"joint_velocity.{name}" for name in X2_TRACKING_JOINT_NAMES],
    "imu_torso.orientation.x",
    "imu_torso.orientation.y",
    "imu_torso.orientation.z",
    "imu_torso.orientation.w",
    "imu_torso.angular_velocity.x",
    "imu_torso.angular_velocity.y",
    "imu_torso.angular_velocity.z",
    "imu_torso.linear_acceleration.x",
    "imu_torso.linear_acceleration.y",
    "imu_torso.linear_acceleration.z",
]

TIMING_NAMES = [
    "camera_grid_offset_signed_ms",
    "reference_previous_age_ms",
    "hand_command_previous_age_ms",
    "joint_max_age_ms",
    "imu_age_ms",
]

# GR00T N1.7 reference-prediction schema.  The raw telemetry keeps the root
# quaternion so the conversion is lossless; the training dataset uses the
# row-major first two rows of its rotation matrix because that is the
# XYZ_ROT6D convention implemented by GR00T's EndEffectorPose.
ROOT_ROT6D_NAMES = [
    "rotation.row0.x",
    "rotation.row0.y",
    "rotation.row0.z",
    "rotation.row1.x",
    "rotation.row1.y",
    "rotation.row1.z",
]

GROOT_N17_STATE_NAMES = [
    *[f"joint_position.{name}" for name in X2_TRACKING_JOINT_NAMES],
    *[f"joint_velocity.{name}" for name in X2_TRACKING_JOINT_NAMES],
    "root_angular_velocity.x",
    "root_angular_velocity.y",
    "root_angular_velocity.z",
    "projected_gravity.x",
    "projected_gravity.y",
    "projected_gravity.z",
    "current_root_reference.position.x",
    "current_root_reference.position.y",
    "current_root_reference.position.z",
    *[f"current_root_reference.{name}" for name in ROOT_ROT6D_NAMES],
    *[f"current_joint_reference.{name}" for name in X2_TRACKING_JOINT_NAMES],
    "current_grasp.left",
    "current_grasp.right",
]

GROOT_N17_ACTION_NAMES = [
    "root_reference.position.x",
    "root_reference.position.y",
    "root_reference.position.z",
    *[f"root_reference.{name}" for name in ROOT_ROT6D_NAMES],
    *[f"joint_reference.{name}" for name in X2_TRACKING_JOINT_NAMES],
    "grasp.left",
    "grasp.right",
]

GROOT_N17_TIMING_NAMES = [
    "camera_grid_offset_signed_ms",
    "tracking_telemetry_previous_age_ms",
    "hand_command_previous_age_ms",
    "consumed_reference_source_age_ms",
]


assert len(X2_TRACKING_JOINT_NAMES) == 29
assert len(REFERENCE_ACTION_NAMES) == 36
assert len(HAND_ACTION_NAMES) == 2
assert len(ACTION_NAMES) == 38
assert len(OBSERVATION_STATE_NAMES) == 68
assert len(TIMING_NAMES) == 5
assert len(GROOT_N17_STATE_NAMES) == 104
assert len(GROOT_N17_ACTION_NAMES) == 40
assert len(GROOT_N17_TIMING_NAMES) == 4
