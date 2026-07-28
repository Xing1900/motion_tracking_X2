"""Shared schema constants for X2 VR demonstration data.

The ordering below deliberately mirrors ``BaseConfig.seq`` in
``x1-motion-control/x1_digit_mc/src/rl_module/cfg/rl/rl_tracking.yaml``.
Do not derive it from the legacy teleop NPZ recorder: that file labels waist
and wrist axes in a different order.
"""

from __future__ import annotations


RAW_DATASET_SCHEMA_VERSION = "x2-vr-raw-v1"
TELEOP_TAP_SCHEMA_VERSION = 1

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

ACTION_NAMES = [
    "reference.root_pos.x",
    "reference.root_pos.y",
    "reference.root_pos.z",
    "reference.root_quat.w",
    "reference.root_quat.x",
    "reference.root_quat.y",
    "reference.root_quat.z",
    *[f"reference.{name}" for name in X2_TRACKING_JOINT_NAMES],
]

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
    "camera_nearest_age_ms",
    "reference_nearest_age_ms",
    "joint_max_age_ms",
    "imu_age_ms",
]


assert len(X2_TRACKING_JOINT_NAMES) == 29
assert len(ACTION_NAMES) == 36
assert len(OBSERVATION_STATE_NAMES) == 68
