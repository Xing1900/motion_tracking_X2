"""Convert GMR retargeted robot motion (pkl) to motion_tracking npz format.

GMR pkl fields:
    fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list

motion_tracking npz fields:
    fps, root_pos, root_rot, dof_pos, local_body_pos, body_names, joint_names

Also upsamples 30 fps GMR output to target_fps (default 50).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _install_numpy_pickle_compat() -> None:
    """Alias old/new numpy module paths so legacy pickle files can still load."""
    compat_modules = {
        "numpy._core": np.core,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.umath": np.core.umath,
    }
    for name, module in compat_modules.items():
        sys.modules.setdefault(name, module)


def _load_pickle(path: Path):
    _install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


# X2 actuated joint order in GMR qpos (after freejoint).
# This must match the order in gmr/assets/agibot_x2/x2_ultra.xml.
X2_JOINT_NAMES = [
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


def upsample_scalar(data: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Linear interpolation for scalar/position data."""
    n_frames = data.shape[0]
    if n_frames < 2:
        return data
    src_t = np.arange(n_frames) / src_fps
    tgt_t = np.arange(int(np.floor(n_frames / src_fps * tgt_fps))) / tgt_fps
    # Ensure we don't exceed the last source frame.
    tgt_t = tgt_t[tgt_t <= src_t[-1]]
    out = np.zeros((len(tgt_t), *data.shape[1:]), dtype=data.dtype)
    for i in range(data.shape[1] if data.ndim > 1 else 1):
        if data.ndim == 1:
            out[:, 0 if out.ndim > 1 else 0] = np.interp(tgt_t, src_t, data)
        else:
            out[:, i] = np.interp(tgt_t, src_t, data[:, i])
    return out if data.ndim > 1 else out.squeeze(-1) if out.ndim > 1 else out


def upsample_positions(data: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Linear interpolation for 3D positions."""
    n_frames = data.shape[0]
    if n_frames < 2:
        return data
    src_t = np.arange(n_frames) / src_fps
    tgt_t = np.arange(int(np.floor(n_frames / src_fps * tgt_fps))) / tgt_fps
    tgt_t = tgt_t[tgt_t <= src_t[-1]]
    out = np.zeros((len(tgt_t), *data.shape[1:]), dtype=data.dtype)
    for idx in np.ndindex(data.shape[1:]):
        slicer = (slice(None),) + idx
        out[slicer] = np.interp(tgt_t, src_t, data[slicer])
    return out


def upsample_rotations(quat_xyzw: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions (xyzw)."""
    n_frames = quat_xyzw.shape[0]
    if n_frames < 2:
        return quat_xyzw
    src_t = np.arange(n_frames) / src_fps
    tgt_t = np.arange(int(np.floor(n_frames / src_fps * tgt_fps))) / tgt_fps
    tgt_t = tgt_t[tgt_t <= src_t[-1]]

    key_rots = R.from_quat(quat_xyzw)
    slerp = Slerp(src_t, key_rots)
    interp_rots = slerp(tgt_t)
    return interp_rots.as_quat().astype(quat_xyzw.dtype)


def convert_pkl_to_npz(input_path: Path, output_path: Path, target_fps: float = 50.0) -> None:
    motion_data = _load_pickle(input_path)
    if not isinstance(motion_data, dict):
        raise TypeError(f"Expected top-level pickle object to be dict, got {type(motion_data)!r}")

    src_fps = float(motion_data.get("fps", 30.0))
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
    local_body_pos = np.asarray(motion_data["local_body_pos"], dtype=np.float32)
    body_names = motion_data.get("link_body_list", motion_data.get("body_names"))
    if body_names is None:
        raise KeyError(f"Missing body names in {input_path}")
    body_names = [str(n) for n in body_names]

    # Sanity checks.
    n_frames = root_pos.shape[0]
    assert root_rot.shape[0] == n_frames
    assert dof_pos.shape[0] == n_frames
    assert local_body_pos.shape[0] == n_frames
    assert dof_pos.shape[1] == len(X2_JOINT_NAMES), (
        f"dof_pos has {dof_pos.shape[1]} joints, expected {len(X2_JOINT_NAMES)}"
    )
    assert local_body_pos.shape[1] == len(body_names), (
        f"local_body_pos has {local_body_pos.shape[1]} bodies, expected {len(body_names)}"
    )

    if abs(src_fps - target_fps) > 1e-3:
        print(f"[upsample] {input_path.name}: {src_fps} fps -> {target_fps} fps")
        root_pos = upsample_positions(root_pos, src_fps, target_fps)
        root_rot = upsample_rotations(root_rot, src_fps, target_fps)
        dof_pos = upsample_positions(dof_pos, src_fps, target_fps)
        local_body_pos = upsample_positions(local_body_pos, src_fps, target_fps)
    else:
        print(f"[skip] {input_path.name}: already {src_fps} fps")

    npz_payload = {
        "fps": np.array(target_fps, dtype=np.int32),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "body_names": np.array(body_names, dtype=object),
        "joint_names": np.array(X2_JOINT_NAMES, dtype=object),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **npz_payload)
    print(f"Converted: {input_path}")
    print(f"Saved to : {output_path} ({root_pos.shape[0]} frames @ {target_fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GMR pkl motion files to motion_tracking npz.")
    parser.add_argument("input", type=Path, help="Input .pkl file or directory containing .pkl files.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output .npz file or directory.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Target frame rate (default: 50).")
    parser.add_argument("--num-cpus", type=int, default=1, help="Number of parallel workers.")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    assert input_path.exists(), f"Input not found: {input_path}"

    if input_path.is_file():
        pkl_files = [input_path]
    else:
        pkl_files = sorted(input_path.rglob("*.pkl"))

    if args.output is None:
        if input_path.is_file():
            output_paths = [input_path.with_suffix(".npz")]
        else:
            output_dir = input_path.parent / (input_path.name + "_npz")
            output_paths = [
                output_dir / p.relative_to(input_path).with_suffix(".npz")
                for p in pkl_files
            ]
    else:
        output_path = args.output.expanduser().resolve()
        if len(pkl_files) == 1:
            output_paths = [output_path]
        else:
            output_paths = [
                output_path / p.relative_to(input_path).with_suffix(".npz")
                for p in pkl_files
            ]

    if args.num_cpus > 1:
        from multiprocessing import Pool
        with Pool(args.num_cpus) as pool:
            pool.starmap(
                convert_pkl_to_npz,
                [(src, dst, args.target_fps) for src, dst in zip(pkl_files, output_paths)],
            )
    else:
        for src, dst in zip(pkl_files, output_paths):
            convert_pkl_to_npz(src, dst, args.target_fps)


if __name__ == "__main__":
    main()
