import argparse
import json
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.motion import MotionDataset

EXCLUDE_LABEL_PATH = Path(__file__).parent / "label.txt"
SEED_KEEP_FILENAMES_PATH = Path("/home/axell/Desktop/dataset_new/retarget_g1/seed/keep_filenames.txt")
EXCLUDED_SEGMENTS: set[tuple[str, int, int]] = set()
SEED_KEEP_FILENAMES = set()
ENABLE_AMASS_FILTER = False
ENABLE_SEED_FILTER = False
VERBOSE_REJECTIONS = False
FILTER_REJECTIONS: Counter[str] = Counter()
MIN_SEGMENT_FRAMES = 250
MAX_ROOT_LINEAR_OR_ANGULAR_SPEED = 10.0
MAX_QUATERNION_NORM_ERROR = 0.1
EXCLUDED_SUBSTRINGS = [
    "CMU/94",
    "CMU/126",
    "chair",
    "HDM05/tr",
    "HDM05/bk",
    "SShapeRL",
    "SShapeLR",
    "CircleCCW",
    "KIT/1226",
]


def preprocess_motion(motion, foot_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]  # (T,3)
    offset_xy = root_pos[0, :2].copy()  # 首帧 x,y
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    z_l = motion["xpos"][:, foot_idx[0], 2]
    z_r = motion["xpos"][:, foot_idx[1], 2]

    if not always_on_ground:
        z_min = float(min(z_l.min(), z_r.min()))
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz
        motion["xpos"][:, :, 2] += dz
    else:
        z_min = np.min(
            np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
            axis=-1,
            keepdims=True,
        )
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz.reshape(-1)
        motion["xpos"][:, :, 2] += dz
    return motion


def none_callback(_ctx, m):
    m["metadata"] = None


def amass_relative_path(path: str | Path) -> str:
    """Return a machine-independent AMASS path used by manual labels."""
    normalized = str(path).replace("\\", "/")
    marker = "/AMASS/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def load_excluded_segments(label_path: Path) -> set[tuple[str, int, int]]:
    """Load exact bad segments instead of excluding a whole source motion."""
    if not label_path.exists():
        return set()
    segments = set()
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Invalid exclusion label line: {line!r}")
            segments.add((amass_relative_path(fields[0]), int(fields[1]), int(fields[2])))
    return segments

def load_keep_filenames(list_path: Path) -> set[str]:
    if not list_path.exists():
        return set()
    names = set()
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def reject_motion(reason: str, path: Path, start_idx: int, end_idx: int) -> bool:
    FILTER_REJECTIONS[reason] += 1
    if VERBOSE_REJECTIONS:
        print(f"Invalid motion ({reason}): {path} [{start_idx}, {end_idx})")
    return False


def check_motion(motion, foot_idx, path, start_idx, end_idx) -> bool:
    """Return False when the motion violates basic physical sanity checks."""

    qvel = motion["qvel"]
    qpos = motion["qpos"]
    xpos = motion["xpos"]

    path_str = str(path)
    if ENABLE_AMASS_FILTER:
        segment_key = (amass_relative_path(path), int(start_idx), int(end_idx))
        if EXCLUDED_SEGMENTS and segment_key in EXCLUDED_SEGMENTS:
            return reject_motion("manual_exclusion", path, start_idx, end_idx)
        if any(s in path_str for s in EXCLUDED_SUBSTRINGS):
            return reject_motion("excluded_motion_family", path, start_idx, end_idx)
    if ENABLE_SEED_FILTER and SEED_KEEP_FILENAMES and path.stem not in SEED_KEEP_FILENAMES:
        return reject_motion("missing_seed_allowlist", path, start_idx, end_idx)

    if qpos.ndim != 2 or qvel.ndim != 2 or xpos.ndim != 3:
        return reject_motion("invalid_array_rank", path, start_idx, end_idx)
    if not (qpos.shape[0] == qvel.shape[0] == xpos.shape[0]):
        return reject_motion("inconsistent_frame_count", path, start_idx, end_idx)
    if qpos.shape[1] < 7 or qvel.shape[1] < 6 or xpos.shape[-1] != 3:
        return reject_motion("invalid_array_shape", path, start_idx, end_idx)
    if not all(np.isfinite(array).all() for array in (qpos, qvel, xpos)):
        return reject_motion("non_finite_value", path, start_idx, end_idx)

    quat_norm_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1.0)
    if np.any(quat_norm_error > MAX_QUATERNION_NORM_ERROR):
        return reject_motion("invalid_root_quaternion", path, start_idx, end_idx)

    if np.any(np.abs(qvel[:, :6]) > MAX_ROOT_LINEAR_OR_ANGULAR_SPEED):
        return reject_motion("root_velocity_spike", path, start_idx, end_idx)
    if qpos.shape[0] < MIN_SEGMENT_FRAMES:
        return reject_motion("short_segment", path, start_idx, end_idx)

    min_body_z = np.min(xpos[:, :, 2], axis=1)
    all_off = min_body_z > 0.2
    fps = int(motion.get("fps", 0))
    if fps <= 0:
        fps = 50
    if np.any(all_off):
        padded = np.concatenate(([0], all_off.astype(np.int8), [0]))
        edges = np.diff(padded)
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        max_run = (run_ends - run_starts).max() if run_starts.size else 0
        if max_run > fps:
            return reject_motion("all_bodies_airborne_over_1s", path, start_idx, end_idx)
    max_body_z = float(np.max(xpos[:, :, 2]))
    if max_body_z <= 0.2:
        return reject_motion("low_max_body_height", path, start_idx, end_idx)
    return True


def write_quality_report(mem_path: Path, dataset_root: Path) -> Path:
    with (mem_path / "meta_motion.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (mem_path / "id_label.json").open("r", encoding="utf-8") as f:
        labels = json.load(f)

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "mem_path": str(mem_path.resolve()),
        "amass_filter_enabled": ENABLE_AMASS_FILTER,
        "seed_filter_enabled": ENABLE_SEED_FILTER,
        "input_npz_files": len(list(dataset_root.rglob("*.npz"))) if dataset_root.is_dir() else 1,
        "accepted_segments": len(labels),
        "accepted_frames": int(sum(meta["ends"][i] - meta["starts"][i] for i in range(len(meta["starts"])))),
        "rejected_segments": int(sum(FILTER_REJECTIONS.values())),
        "rejections_by_reason": dict(sorted(FILTER_REJECTIONS.items())),
        "manual_exclusion_labels_loaded": len(EXCLUDED_SEGMENTS),
        "filter_thresholds": {
            "minimum_segment_frames": MIN_SEGMENT_FRAMES,
            "maximum_root_linear_or_angular_speed": MAX_ROOT_LINEAR_OR_ANGULAR_SPEED,
            "maximum_root_quaternion_norm_error": MAX_QUATERNION_NORM_ERROR,
            "maximum_all_bodies_airborne_seconds": 1.0,
            "minimum_max_body_height": 0.2,
        },
    }
    report_path = mem_path / "quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="NPZ file or directory to convert")
    ap.add_argument("--mem-path", required=True, help="Output memmap directory")
    ap.add_argument("--amass-filter", action="store_true", help="Enable AMASS-specific path/name filters")
    ap.add_argument("--seed-filter", action="store_true", help="Enable seed keep_filenames allowlist filter")
    ap.add_argument("--verbose-rejections", action="store_true", help="Print every rejected segment")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    global EXCLUDED_SEGMENTS, SEED_KEEP_FILENAMES, ENABLE_AMASS_FILTER, ENABLE_SEED_FILTER, VERBOSE_REJECTIONS
    ENABLE_AMASS_FILTER = args.amass_filter
    ENABLE_SEED_FILTER = args.seed_filter
    VERBOSE_REJECTIONS = args.verbose_rejections
    FILTER_REJECTIONS.clear()
    EXCLUDED_SEGMENTS = load_excluded_segments(EXCLUDE_LABEL_PATH) if ENABLE_AMASS_FILTER else set()
    SEED_KEEP_FILENAMES = load_keep_filenames(SEED_KEEP_FILENAMES_PATH) if ENABLE_SEED_FILTER else set()

    MotionDataset.create_from_path(
        str(dataset_root),
        target_fps=50,
        mem_path=args.mem_path,
        callback=none_callback,
        motion_processer=partial(preprocess_motion, always_on_ground=False),
        motion_filter=check_motion,
        segment_len=1000,
        storage_float_dtype=torch.float16,
        storage_int_dtype=torch.int32,
    )
    report_path = write_quality_report(Path(args.mem_path), dataset_root)
    print(f"Quality report: {report_path}")
    print(f"Rejected segments by reason: {dict(sorted(FILTER_REJECTIONS.items()))}")


if __name__ == "__main__":
    main()
