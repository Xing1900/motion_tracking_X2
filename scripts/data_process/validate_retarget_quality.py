"""Quantitative quality check for GMR-retargeted X2 motions.

Loads npz files produced by gmr_pkl_to_motion_tracking_npz.py and reports:
  - NaN/Inf counts
  - joint limit violations (parsed from x2.xml)
  - joint velocity / acceleration spikes
  - root height / roll / pitch stats
  - foot contact quality (height + sliding)
  - body ground penetration
"""
from __future__ import annotations

import argparse
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
CONTACT_HEIGHT = 0.05  # m: considered "on ground"
PENETRATION_THRESHOLD = -0.01  # m


def parse_joint_limits(xml_path: Path) -> dict[str, tuple[float, float]]:
    """Return {joint_name: (lower, upper)} from MuJoCo XML."""
    tree = ET.parse(xml_path)
    limits: dict[str, tuple[float, float]] = {}
    for joint in tree.iter("joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name is None or rng is None:
            continue
        lo, hi = map(float, rng.split())
        limits[name] = (lo, hi)
    return limits


def finite_diff(data: np.ndarray, fps: float) -> np.ndarray:
    """Central difference; endpoints use forward/backward."""
    if data.shape[0] < 2:
        return np.zeros_like(data)
    dt = 1.0 / fps
    out = np.empty_like(data)
    out[0] = (data[1] - data[0]) / dt
    out[-1] = (data[-1] - data[-2]) / dt
    out[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    return out


def check_file(npz_path: Path, limits: dict[str, tuple[float, float]], joint_order: list[str]):
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    local_body_pos = np.asarray(data["local_body_pos"], dtype=np.float32)
    body_names = list(data["body_names"])

    report: dict = {"path": str(npz_path), "frames": root_pos.shape[0]}

    # NaN / Inf
    report["nan_inf"] = int(
        np.isnan(root_pos).any()
        or np.isnan(root_quat_xyzw).any()
        or np.isnan(dof_pos).any()
        or np.isnan(local_body_pos).any()
        or np.isinf(root_pos).any()
        or np.isinf(root_quat_xyzw).any()
        or np.isinf(dof_pos).any()
        or np.isinf(local_body_pos).any()
    )

    # Joint limits
    n_joints = len(joint_order)
    limit_violations = np.zeros(n_joints, dtype=np.int64)
    for i, name in enumerate(joint_order):
        lo, hi = limits[name]
        bad = (dof_pos[:, i] < lo - 1e-6) | (dof_pos[:, i] > hi + 1e-6)
        limit_violations[i] = bad.sum()
    report["joint_limit_violations_total"] = int(limit_violations.sum())
    if limit_violations.sum():
        worst = int(np.argmax(limit_violations))
        report["worst_joint_violation"] = (joint_order[worst], int(limit_violations[worst]))

    # Joint velocities / accelerations
    qvel = finite_diff(dof_pos, fps)
    qacc = finite_diff(qvel, fps)
    report["max_joint_vel"] = float(np.abs(qvel).max())
    report["max_joint_acc"] = float(np.abs(qacc).max())
    report["joint_vel_spike_frames"] = int((np.abs(qvel) > 20.0).any(axis=1).sum())

    # Root stats
    report["root_z_min"] = float(root_pos[:, 2].min())
    report["root_z_max"] = float(root_pos[:, 2].max())
    report["root_xy_path"] = float(np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum())
    report["root_xy_displacement"] = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))

    root_rot = R.from_quat(root_quat_xyzw)  # scipy uses xyzw
    root_euler = root_rot.as_euler("XYZ", degrees=False)
    report["root_roll_max"] = float(np.abs(root_euler[:, 0]).max())
    report["root_pitch_max"] = float(np.abs(root_euler[:, 1]).max())

    # Global body positions
    root_rot_m = root_rot.as_matrix()
    global_body_pos = np.einsum("tij,tbj->tbi", root_rot_m, local_body_pos) + root_pos[:, None, :]
    body_z_min = global_body_pos[..., 2].min(axis=1)
    report["penetration_frames"] = int((body_z_min < PENETRATION_THRESHOLD).sum())
    report["min_body_z"] = float(body_z_min.min())

    # Feet
    foot_idx = [body_names.index(n) for n in FOOT_NAMES]
    foot_global = global_body_pos[:, foot_idx, :]  # (T, 2, 3)
    foot_h = foot_global[..., 2]
    report["foot_min_height"] = float(foot_h.min())
    report["foot_max_height"] = float(foot_h.max())
    report["foot_on_ground_ratio"] = float((foot_h < CONTACT_HEIGHT).mean())

    # Foot sliding: horizontal speed when foot near ground
    foot_xy = foot_global[..., :2]
    foot_xy_vel = finite_diff(foot_xy, fps)  # (T, 2, 2)
    foot_xy_speed = np.linalg.norm(foot_xy_vel, axis=-1)  # (T, 2)
    on_ground = foot_h < CONTACT_HEIGHT
    report["foot_slide_speed_mean"] = float(foot_xy_speed[on_ground].mean()) if on_ground.any() else 0.0
    report["foot_slide_speed_max"] = float(foot_xy_speed[on_ground].max()) if on_ground.any() else 0.0

    return report


def aggregate(reports: list[dict]) -> dict:
    total_frames = sum(r["frames"] for r in reports)
    n_files = len(reports)
    bad_files = [r for r in reports if r["nan_inf"] or r["joint_limit_violations_total"] or r["penetration_frames"]]

    def mean(key: str) -> float:
        vals = [r[key] for r in reports]
        return float(np.mean(vals)) if vals else 0.0

    def max_(key: str) -> float:
        vals = [r[key] for r in reports]
        return float(np.max(vals)) if vals else 0.0

    return {
        "files": n_files,
        "total_frames": total_frames,
        "files_with_issues": len(bad_files),
        "nan_inf_files": sum(r["nan_inf"] for r in reports),
        "joint_limit_violations_total": sum(r["joint_limit_violations_total"] for r in reports),
        "penetration_frames_total": sum(r["penetration_frames"] for r in reports),
        "max_joint_vel": max_("max_joint_vel"),
        "mean_max_joint_vel": mean("max_joint_vel"),
        "max_joint_acc": max_("max_joint_acc"),
        "mean_max_joint_acc": mean("max_joint_acc"),
        "root_z_min": min(r["root_z_min"] for r in reports),
        "root_z_max": max_("root_z_max"),
        "root_roll_max": max_("root_roll_max"),
        "root_pitch_max": max_("root_pitch_max"),
        "foot_min_height": min(r["foot_min_height"] for r in reports),
        "foot_max_height": max_("foot_max_height"),
        "mean_foot_on_ground_ratio": mean("foot_on_ground_ratio"),
        "mean_foot_slide_speed": mean("foot_slide_speed_mean"),
        "max_foot_slide_speed": max_("foot_slide_speed_max"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory of npz files or a single npz")
    parser.add_argument("--xml", type=Path, default=Path(__file__).parent.parent.parent / "active_adaptation/assets/X2/x2.xml")
    parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample N files")
    args = parser.parse_args()

    limits = parse_joint_limits(args.xml)
    # Joint order expected by the npz (matches GMR output and XML actuated order)
    joint_order = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    ]
    for name in joint_order:
        if name not in limits:
            raise RuntimeError(f"Joint limit missing for {name}")

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(args.input.rglob("*.npz"))

    if args.sample and args.sample < len(files):
        rng = np.random.default_rng(0)
        files = rng.choice(files, args.sample, replace=False).tolist()

    reports = []
    for f in files:
        try:
            reports.append(check_file(f, limits, joint_order))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    agg = aggregate(reports)
    print("\n===== Aggregate quality report =====")
    for k, v in agg.items():
        print(f"{k:40s}: {v}")

    print("\n===== Files with issues (first 10) =====")
    issue_files = [r for r in reports if r["nan_inf"] or r["joint_limit_violations_total"] or r["penetration_frames"]]
    for r in issue_files[:10]:
        print(f"  {r['path']}")
        print(f"    frames={r['frames']} nan={r['nan_inf']} limit_viol={r['joint_limit_violations_total']} penetr={r['penetration_frames']} max_vel={r['max_joint_vel']:.2f}")


if __name__ == "__main__":
    main()
