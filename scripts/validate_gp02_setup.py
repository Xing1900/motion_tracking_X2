#!/usr/bin/env python3
"""Preflight checks for the GP02 asset, task config, and motion datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import hydra
import mujoco
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from active_adaptation.assets import get_robot_cfg  # noqa: E402
from active_adaptation.assets.GP02.humanoid import GP02_JOINT_ORDER  # noqa: E402
from active_adaptation.utils.fk_helper import MotionFKHelper  # noqa: E402
from active_adaptation.utils.motion import MotionDataset  # noqa: E402


def _model_names(model: mujoco.MjModel, obj_type, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj_type, i) for i in range(count)]


def _assert_patterns_match(patterns: list[str], names: list[str], context: str) -> None:
    for pattern in patterns:
        if not any(re.fullmatch(pattern, name) for name in names):
            raise AssertionError(f"{context} pattern {pattern!r} matches nothing")


def _assert_names_covered(patterns: list[str], names: list[str], context: str) -> None:
    uncovered = [
        name for name in names
        if not any(re.fullmatch(pattern, name) for pattern in patterns)
    ]
    if uncovered:
        raise AssertionError(f"{context} does not cover: {uncovered}")


def _validate_dataset(name: str, full_scan: bool) -> dict:
    dataset = MotionDataset.create_from_path_lazy(name)
    expected = list(GP02_JOINT_ORDER)
    if dataset.joint_names != expected:
        raise AssertionError(
            f"{name}: joint order mismatch\nexpected={expected}\nactual={dataset.joint_names}"
        )
    if dataset.num_motions <= 0 or dataset.num_steps <= 0:
        raise AssertionError(f"{name}: empty dataset")
    if not torch.all(dataset.starts[1:] == dataset.ends[:-1]):
        raise AssertionError(f"{name}: non-contiguous segment boundaries")
    if int(dataset.starts[0]) != 0 or int(dataset.ends[-1]) != dataset.num_steps:
        raise AssertionError(f"{name}: invalid first/last boundary")

    for field in ("root_pos_w", "root_quat_w", "joint_pos"):
        values = getattr(dataset.data, field)
        if full_scan:
            for start in range(0, dataset.num_steps, 250_000):
                if not torch.isfinite(values[start:start + 250_000]).all():
                    raise AssertionError(f"{name}: {field} contains NaN or Inf near frame {start}")
        else:
            sample_count = min(4096, dataset.num_steps)
            indices = torch.linspace(0, dataset.num_steps - 1, sample_count).long()
            if not torch.isfinite(values[indices]).all():
                raise AssertionError(f"{name}: {field} contains NaN or Inf")

    root = PROJECT_ROOT / "dataset" / name
    report_path = root / "quality_report.json"
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(report["mem_path"]).resolve() != root.resolve():
            raise AssertionError(f"{name}: quality report mem_path is stale: {report['mem_path']}")
        if report["accepted_segments"] != dataset.num_motions:
            raise AssertionError(f"{name}: quality report segment count is stale")
        if report["accepted_frames"] != dataset.num_steps:
            raise AssertionError(f"{name}: quality report frame count is stale")
    return {
        "name": name,
        "motions": dataset.num_motions,
        "frames": dataset.num_steps,
        "quality_report": str(report_path) if report is not None else None,
    }


def _validate_motiontracking_fk(model: mujoco.MjModel, body_names: list[str]) -> float:
    """Compare the runtime FK helper against native MuJoCo on CPU."""
    all_model_body_names = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    body_name_to_id = {name: idx for idx, name in enumerate(all_model_body_names)}
    joint_id_to_name = {
        idx: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
        for idx in range(model.njnt)
    }
    helper = MotionFKHelper._build(
        model=model,
        body_name_to_id=body_name_to_id,
        joint_id_to_name=joint_id_to_name,
        dataset_joint_names=GP02_JOINT_ORDER,
        output_body_names=body_names,
        base_body_name="pelvis",
        device=torch.device("cpu"),
    )

    samples = torch.zeros((3, len(GP02_JOINT_ORDER)), dtype=torch.float32)
    samples[1] = torch.linspace(-0.15, 0.15, len(GP02_JOINT_ORDER))
    samples[2] = -samples[1]
    root_pos = torch.zeros((3, 3), dtype=torch.float32)
    root_quat = torch.zeros((3, 4), dtype=torch.float32)
    root_quat[:, 0] = 1.0
    _, _, helper_pos, _ = helper.body_pose(root_pos, root_quat, samples)

    data = mujoco.MjData(model)
    max_error = 0.0
    for sample_id, joint_pos in enumerate(samples.numpy()):
        data.qpos[:] = np.concatenate(
            [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), joint_pos]
        )
        mujoco.mj_forward(model, data)
        error = np.linalg.norm(helper_pos[sample_id].numpy() - data.xpos[1:], axis=-1)
        max_error = max(max_error, float(error.max()))
    if max_error > 1e-5:
        raise AssertionError(f"MotionTracking FK disagrees with MuJoCo: max error={max_error} m")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-data-scan",
        action="store_true",
        help="Check every memmap value instead of a deterministic 4096-frame sample",
    )
    args = parser.parse_args()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "cfg")):
        cfg = hydra.compose(config_name="train", overrides=["task=GP02/GP02_tracking"])

    if cfg.task.robot.name != "gp02_v3":
        raise AssertionError(f"Unexpected robot config: {cfg.task.robot.name}")
    if cfg.task.action.get("_target_") != "active_adaptation.envs.mdp.action.JointPosition":
        raise AssertionError("GP02 action target is missing or incorrect")
    if cfg.task.command.get("_target_") != (
        "active_adaptation.envs.mdp.commands.motion_tracking.MotionTrackingComplianceCommand"
    ):
        raise AssertionError("GP02 motion-tracking command target is missing or incorrect")
    required_command_keys = {
        "future_steps", "student_future_steps", "init_noise",
        "body_z_terminate_thres", "gravity_terminate_thres",
    }
    missing_command_keys = sorted(required_command_keys - set(cfg.task.command.keys()))
    if missing_command_keys:
        raise AssertionError(f"GP02 command config is incomplete: {missing_command_keys}")
    if list(cfg.task.command.future_steps)[0] != 0 or list(cfg.task.command.student_future_steps)[0] != 0:
        raise AssertionError("GP02 future-step horizons must begin with zero")
    for group in ("policy", "priv", "priv_critic"):
        if group not in cfg.task.observation or not cfg.task.observation[group]:
            raise AssertionError(f"GP02 observation group is missing or empty: {group}")
    if "tracking" not in cfg.task.reward or not cfg.task.reward.tracking:
        raise AssertionError("GP02 tracking rewards are missing")
    for termination_name in ("body_z_termination", "gravity_dir_termination"):
        if termination_name not in cfg.task.termination:
            raise AssertionError(f"GP02 termination is missing: {termination_name}")

    robot_cfg = get_robot_cfg(cfg.task.robot.name)
    spec = robot_cfg.spec_fn()
    model = spec.compile()
    joint_names = _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
    body_names = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]

    if joint_names != list(GP02_JOINT_ORDER):
        raise AssertionError(
            f"MJCF joint order mismatch\nexpected={list(GP02_JOINT_ORDER)}\nactual={joint_names}"
        )
    if model.nu != 0:
        raise AssertionError(
            f"GP02 source actuators must be removed before MJLab attaches its PD actuators, got {model.nu}"
        )

    # Compile the entity through MJLab as training does. This catches duplicate
    # source/config actuator names that a standalone MJCF compile cannot see.
    from mjlab.scene import Scene, SceneCfg

    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.5)
    scene_cfg.entities["robot"] = robot_cfg
    training_model = Scene(scene_cfg, device="cpu").compile()
    if training_model.nu != len(GP02_JOINT_ORDER):
        raise AssertionError(
            f"Expected 24 MJLab position actuators, got {training_model.nu}"
        )
    training_sensor_names = _model_names(
        training_model, mujoco.mjtObj.mjOBJ_SENSOR, training_model.nsensor
    )
    for sensor_name in ("robot/imu_ang_vel", "robot/imu_lin_acc"):
        if sensor_name not in training_sensor_names:
            raise AssertionError(f"Required GP02 training sensor is missing: {sensor_name}")
    model_mass = float(model.body_mass.sum())
    if abs(model_mass - float(cfg.task.robot.mass)) > 1e-5:
        raise AssertionError(f"Configured mass {cfg.task.robot.mass} != MJCF mass {model_mass}")
    fk_max_error = _validate_motiontracking_fk(model, body_names)

    command = cfg.task.command
    _assert_patterns_match(list(command.required_motion_body_patterns), body_names, "required body")
    _assert_patterns_match(list(command.keypoint_patterns), body_names, "keypoint")
    _assert_patterns_match(list(command.upper_force_keypoint_patterns), body_names, "upper-force keypoint")
    _assert_patterns_match(list(command.joint_patterns), joint_names, "tracked joint")
    _assert_patterns_match(list(command.feet_patterns), body_names, "foot")
    locked_joint_patterns = list(command.get("locked_joint_patterns", []))
    locked_joint_names = [
        name for name in joint_names
        if any(re.fullmatch(pattern, name) for pattern in locked_joint_patterns)
    ]
    if locked_joint_names != ["waist_yaw_joint", "waist_roll_joint"]:
        raise AssertionError(f"Unexpected GP02 locked joints: {locked_joint_names}")
    reference_overrides = dict(command.dataset.get("reference_joint_pos_overrides", {}))
    if reference_overrides != {
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
    }:
        raise AssertionError(
            f"GP02 fixed-waist reference projection is missing or incorrect: {reference_overrides}"
        )

    action_joint_patterns = [str(cfg.task.action.get("joint_names", ".*"))]
    action_joint_names = [
        name for name in joint_names
        if any(re.fullmatch(pattern, name) for pattern in action_joint_patterns)
    ]
    if set(action_joint_names) & set(locked_joint_names):
        raise AssertionError("Locked GP02 joints must not be part of the policy action space")
    if len(action_joint_names) != 22:
        raise AssertionError(f"Expected 22 GP02 policy actions, got {len(action_joint_names)}")
    action_scaling_patterns = list(cfg.task.action.action_scaling.keys())
    _assert_patterns_match(action_scaling_patterns, action_joint_names, "action scaling")
    _assert_names_covered(action_scaling_patterns, action_joint_names, "action scaling")

    mem_paths = list(command.dataset.mem_paths)
    weights = list(command.dataset.path_weights)
    if len(mem_paths) != len(weights) or not mem_paths:
        raise AssertionError("Dataset paths and weights must have the same non-zero length")
    if any(float(weight) <= 0 for weight in weights):
        raise AssertionError("All GP02 dataset weights must be positive")

    datasets = [_validate_dataset(name, args.full_data_scan) for name in mem_paths]
    print("GP02 preflight: PASS")
    print(
        f"MJCF: joints={len(joint_names)}, bodies={len(body_names)}, actuators={training_model.nu}, "
        f"mass={model_mass:.6f} kg, FK max error={fk_max_error:.3e} m"
    )
    print(
        f"Control: actions={len(action_joint_names)}, "
        f"locked_joints={locked_joint_names}"
    )
    for item in datasets:
        suffix = f", report={item['quality_report']}" if item["quality_report"] else ""
        print(f"Dataset {item['name']}: motions={item['motions']}, frames={item['frames']}{suffix}")


if __name__ == "__main__":
    main()
