"""
Select a diverse set of demo motions from the retargeted X2 dataset for sim2real.
Outputs a YAML snippet suitable for sim2real/config/tracking.yaml.
"""
import argparse
import json
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotionInfo:
    path: str
    rel_path: str
    name: str
    n_frames: int
    fps: int
    duration: float
    mean_root_speed: float
    max_root_speed: float
    mean_joint_vel: float
    max_joint_vel: float


def compute_stats(path: str, dataset_root: Path) -> Optional[MotionInfo]:
    try:
        data = np.load(path, allow_pickle=True)
        fps = int(data["fps"])
        root_pos = data["root_pos"].astype(np.float32)
        dof_pos = data["dof_pos"].astype(np.float32)
        n_frames = root_pos.shape[0]
        if n_frames < fps * 2:  # skip < 2s
            return None

        dt = 1.0 / fps
        # root horizontal speed
        diff = root_pos[1:, :2] - root_pos[:-1, :2]
        speeds = np.linalg.norm(diff, axis=1) / dt
        mean_root_speed = float(np.mean(speeds))
        max_root_speed = float(np.max(speeds))

        # joint angular velocity magnitude
        jdiff = dof_pos[1:] - dof_pos[:-1]
        jvels = np.linalg.norm(jdiff, axis=1) / dt
        mean_joint_vel = float(np.mean(jvels))
        max_joint_vel = float(np.max(jvels))

        p = Path(path)
        rel = p.relative_to(dataset_root.resolve())
        rel_str = str(rel).replace("\\", "/")
        stem = p.stem.replace("_stageii", "")
        # build a readable name: subset + subject + action
        parts = rel_str.replace(".npz", "").split("/")
        if len(parts) >= 3:
            name = f"{parts[-3]}_{parts[-2]}_{stem}".replace("-", "_").replace(".", "_")[:80]
        else:
            name = f"{stem}".replace("-", "_").replace(".", "_")[:80]

        return MotionInfo(
            path=path,
            rel_path=rel_str,
            name=name,
            n_frames=n_frames,
            fps=fps,
            duration=n_frames / fps,
            mean_root_speed=mean_root_speed,
            max_root_speed=max_root_speed,
            mean_joint_vel=mean_joint_vel,
            max_joint_vel=max_joint_vel,
        )
    except Exception as e:
        return None


def classify(m: MotionInfo) -> str:
    # Combined heuristic: root displacement + joint activity
    if m.mean_root_speed < 0.15 and m.mean_joint_vel < 2.0:
        return "static"
    if m.mean_root_speed < 0.4 and m.mean_joint_vel > 5.5:
        return "vigorous_in_place"  # jump rope, jumping jacks, treadmill jog
    if m.mean_root_speed < 0.4 and m.mean_joint_vel > 3.5:
        return "dance_or_exercise"
    if m.mean_root_speed < 0.6:
        return "slow_walk"
    if m.mean_root_speed < 1.2:
        return "walk"
    if m.mean_root_speed < 2.5 or m.mean_joint_vel > 5.0:
        return "run_jog"
    return "sprint"


def pick_diverse(candidates: List[MotionInfo], rng: random.Random) -> List[MotionInfo]:
    by_class = {}
    for c in candidates:
        cls = classify(c)
        by_class.setdefault(cls, []).append(c)

    print("\nClass distribution in sample:")
    for cls in sorted(by_class.keys()):
        print(f"  {cls}: {len(by_class[cls])}")

    picked = []
    desires = [
        ("static", 2),
        ("vigorous_in_place", 1),
        ("dance_or_exercise", 2),
        ("slow_walk", 2),
        ("walk", 3),
        ("run_jog", 2),
    ]
    for cls, count in desires:
        pool = by_class.get(cls, [])
        if not pool:
            continue
        # prefer clips around 4-8 seconds
        pool.sort(key=lambda x: abs(x.duration - 6.0))
        picked.extend(pool[:count])

    if len(picked) < 12:
        remaining = [c for c in candidates if c not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: 12 - len(picked)])

    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="/home/liuguoxing/Documents/dataset_new/retarget_x2_npz")
    parser.add_argument("--id-label", type=str, default="/home/liuguoxing/Documents/motion_tracking/dataset/x2_amass_all/id_label.json")
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="/tmp/x2_sim2real_motions.yaml")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)

    with open(args.id_label) as f:
        id_label = json.load(f)
    all_paths = [entry["source_path"] for entry in id_label]
    rng.shuffle(all_paths)
    sample_paths = all_paths[: args.max_samples]
    print(f"Analyzing {len(sample_paths)} motions from {len(all_paths)} total...")

    infos = []
    for i, p in enumerate(sample_paths):
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(sample_paths)} ...")
        info = compute_stats(p, dataset_root)
        if info is not None:
            infos.append(info)

    print(f"Valid motions: {len(infos)}")

    picked = pick_diverse(infos, rng)
    # sort by activity level for readability
    picked.sort(key=lambda x: (x.mean_root_speed, x.mean_joint_vel))

    print(f"\nSelected {len(picked)} motions:")
    print(f"{'name':<70} {'class':<20} {'dur':>6} {'spd':>6} {'jvel':>6}")
    for m in picked:
        print(f"{m.name:<70} {classify(m):<20} {m.duration:6.1f} {m.mean_root_speed:6.2f} {m.mean_joint_vel:6.2f}")

    lines = ["motions:"]
    for m in picked:
        lines.append(f'  - name: "{m.name}"')
        lines.append(f'    path: "{m.rel_path}"')
        lines.append(f"    start: 0")
        lines.append(f"    end: -1")

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote YAML to {args.output}")


if __name__ == "__main__":
    main()
