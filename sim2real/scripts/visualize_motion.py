"""
Play a retargeted X2 motion clip in MuJoCo for visual inspection.

Usage:
    cd /home/liuguoxing/Documents/motion_tracking/sim2real
    uv run python scripts/visualize_motion.py --npz /path/to/motion.npz

Controls in the MuJoCo viewer window:
    Space : pause / resume
    R     : restart from beginning
    -/=   : decrease / increase playback speed
    Esc   : quit
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

SIM2REAL_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = SIM2REAL_ROOT / "assets"


def load_motion(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    joint_names = data["joint_names"].tolist()
    if isinstance(joint_names[0], (bytes, np.bytes_)):
        joint_names = [n.decode("utf-8") for n in joint_names]
    return {
        "fps": int(data["fps"]),
        "joint_names": joint_names,
        "dof_pos": data["dof_pos"].astype(np.float64),
        "root_pos": data["root_pos"].astype(np.float64),
        "root_rot": data["root_rot"].astype(np.float64),  # xyzw
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=str, help="Path to retargeted X2 motion .npz")
    parser.add_argument("--xml", type=str, default=str(ASSETS_DIR / "x2" / "x2_ultra.xml"))
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    motion = load_motion(args.npz)
    print(f"Loaded: {args.npz}")
    print(f"  fps: {motion['fps']}, frames: {len(motion['dof_pos'])}, duration: {len(motion['dof_pos']) / motion['fps']:.2f}s")
    print(f"  joints: {motion['joint_names'][:5]} ... ({len(motion['joint_names'])} total)")
    print(f"  root height range: {motion['root_pos'][:, 2].min():.3f} ~ {motion['root_pos'][:, 2].max():.3f}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Map motion joint names to MuJoCo qpos indices (skip floating base, qpos[7:])
    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    # skip freejoint / floating base
    actuated_mj_names = [n for n in mj_joint_names if n in motion["joint_names"]]

    if len(actuated_mj_names) != len(motion["joint_names"]):
        print(f"[Warning] MuJoCo has {len(actuated_mj_names)} actuated joints, motion has {len(motion['joint_names'])}")

    mj_qpos_idx = []
    for name in motion["joint_names"]:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model")
        qpos_adr = model.jnt_qposadr[jnt_id]
        mj_qpos_idx.append(qpos_adr)

    # floating base qpos indices
    base_qpos_idx = list(range(7))

    frame_dt = 1.0 / motion["fps"]
    n_frames = len(motion["dof_pos"])
    frame_idx = 0
    paused = False

    def key_callback(keycode):
        nonlocal paused, frame_idx
        if keycode == mujoco.mjtKey.mjKEY_SPACE:
            paused = not paused
        elif keycode == mujoco.mjtKey.mjKEY_R:
            frame_idx = 0
        elif keycode == ord("-"):
            args.speed = max(0.1, args.speed - 0.1)
            print(f"speed: {args.speed:.1f}x")
        elif keycode == ord("="):
            args.speed = min(3.0, args.speed + 0.1)
            print(f"speed: {args.speed:.1f}x")

    print("\nViewer controls:")
    print("  Space : pause / resume")
    print("  R     : restart")
    print("  -/=   : slower / faster")
    print("  Esc   : quit\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_frame_time = time.time()
        while viewer.is_running():
            if not paused:
                # advance frame based on real time and speed
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed >= frame_dt / args.speed:
                    frame_idx = (frame_idx + 1) % n_frames
                    last_frame_time = now

            # write current frame
            q = motion["dof_pos"][frame_idx]
            for qi, adr in zip(q, mj_qpos_idx):
                data.qpos[adr] = qi

            # root: convert root_rot xyzw -> wxyz for MuJoCo
            r = motion["root_rot"][frame_idx]
            data.qpos[0:3] = motion["root_pos"][frame_idx]
            data.qpos[3:7] = [r[3], r[0], r[1], r[2]]

            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
