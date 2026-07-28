#!/usr/bin/env python3
"""Synchronize raw X2 VR episodes and write LeRobotDataset v3.

The converter is intentionally offline.  It can be run with ``--dry_run`` in
the GMR/ROS Python 3.10 environment to validate synchronization.  Actual
LeRobot v0.6 conversion should run in a separate Python 3.12 environment.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

try:
    from .raw_episode_writer import event_monotonic_ns, read_jsonl
    from .schema import (
        ACTION_NAMES,
        OBSERVATION_STATE_NAMES,
        RAW_DATASET_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
    )
except ImportError:  # Direct execution from this directory.
    from raw_episode_writer import event_monotonic_ns, read_jsonl
    from schema import (
        ACTION_NAMES,
        OBSERVATION_STATE_NAMES,
        RAW_DATASET_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
    )


@dataclass
class ConvertedSample:
    timestamp_ns: int
    image_path: Path
    state: np.ndarray
    action: np.ndarray
    timing_ms: np.ndarray


@dataclass
class EpisodeSamples:
    episode_dir: Path
    task: str
    samples: list[ConvertedSample]
    candidate_count: int
    skip_counts: Dict[str, int]


def _load_manifest(episode_dir: Path) -> Dict[str, Any]:
    path = episode_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_stream(episode_dir: Path, stream: str) -> list[Dict[str, Any]]:
    events = list(read_jsonl(episode_dir / "streams" / f"{stream}.jsonl"))
    events.sort(key=event_monotonic_ns)
    return events


def _nearest_index(times: Sequence[int], target_ns: int) -> Optional[int]:
    if not times:
        return None
    right = bisect.bisect_left(times, target_ns)
    if right == 0:
        return 0
    if right == len(times):
        return len(times) - 1
    return right if times[right] - target_ns < target_ns - times[right - 1] else right - 1


def _previous_index(times: Sequence[int], target_ns: int) -> Optional[int]:
    index = bisect.bisect_right(times, target_ns) - 1
    return index if index >= 0 else None


def _normalize_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _nlerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    first = _normalize_quat_wxyz(q0)
    second = _normalize_quat_wxyz(q1)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    return _normalize_quat_wxyz((1.0 - alpha) * first + alpha * second)


class ReferenceSeries:
    def __init__(self, events: Iterable[Dict[str, Any]]) -> None:
        samples: list[tuple[int, np.ndarray, float]] = []
        for event in events:
            base_time_ns = event_monotonic_ns(event)
            frame_dt_ns = int(event.get("frame_dt_ns", 20_000_000))
            raw_retarget_age = event.get("retarget_age_ms")
            if raw_retarget_age is None:
                # Direct retarget tap events are fresh at their own receive
                # time.  A reference with no backing retarget is not.
                retarget_age_ms = 0.0 if "qpos_root_xyz_quat_wxyz_dof" in event else math.inf
            else:
                try:
                    retarget_age_ms = float(raw_retarget_age)
                except (TypeError, ValueError):
                    retarget_age_ms = math.inf
            frames = event.get("frames_qpos_root_xyz_quat_wxyz_dof")
            if isinstance(frames, list):
                for frame_index, frame in enumerate(frames):
                    vector = np.asarray(frame, dtype=np.float64).reshape(-1)
                    if vector.shape == (36,) and np.all(np.isfinite(vector)):
                        samples.append(
                            (
                                base_time_ns + frame_index * frame_dt_ns,
                                vector,
                                retarget_age_ms,
                            )
                        )
                continue

            # Fallback for the pre-interpolation retarget tap.
            frame = event.get("qpos_root_xyz_quat_wxyz_dof")
            if frame is not None:
                vector = np.asarray(frame, dtype=np.float64).reshape(-1)
                if vector.shape == (36,) and np.all(np.isfinite(vector)):
                    samples.append((base_time_ns, vector, retarget_age_ms))

        samples.sort(key=lambda item: item[0])
        deduplicated: list[tuple[int, np.ndarray, float]] = []
        for timestamp_ns, vector, retarget_age_ms in samples:
            if deduplicated and deduplicated[-1][0] == timestamp_ns:
                deduplicated[-1] = (timestamp_ns, vector, retarget_age_ms)
            else:
                deduplicated.append((timestamp_ns, vector, retarget_age_ms))
        self.times = [item[0] for item in deduplicated]
        self.values = [item[1] for item in deduplicated]
        self.retarget_ages_ms = [item[2] for item in deduplicated]

    def interpolate(self, target_ns: int) -> tuple[Optional[np.ndarray], float, float, float]:
        if not self.times:
            return None, math.inf, math.inf, math.inf
        right = bisect.bisect_left(self.times, target_ns)
        if right == 0:
            reference_age_ms = abs(self.times[0] - target_ns) / 1e6
            return (
                self.values[0].copy(),
                reference_age_ms,
                self.retarget_ages_ms[0] + reference_age_ms,
                0.0,
            )
        if right == len(self.times):
            reference_age_ms = abs(target_ns - self.times[-1]) / 1e6
            return (
                self.values[-1].copy(),
                reference_age_ms,
                self.retarget_ages_ms[-1] + reference_age_ms,
                0.0,
            )
        if self.times[right] == target_ns:
            return (
                self.values[right].copy(),
                0.0,
                self.retarget_ages_ms[right],
                0.0,
            )

        left = right - 1
        t0, t1 = self.times[left], self.times[right]
        if t1 <= t0:
            reference_age_ms = abs(target_ns - t0) / 1e6
            return (
                self.values[left].copy(),
                reference_age_ms,
                self.retarget_ages_ms[left] + reference_age_ms,
                0.0,
            )
        alpha = float(target_ns - t0) / float(t1 - t0)
        output = (1.0 - alpha) * self.values[left] + alpha * self.values[right]
        output[3:7] = _nlerp_quat_wxyz(self.values[left][3:7], self.values[right][3:7], alpha)
        age_ms = min(abs(target_ns - t0), abs(t1 - target_ns)) / 1e6
        retarget_age_ms = max(
            self.retarget_ages_ms[left] + (target_ns - t0) / 1e6,
            self.retarget_ages_ms[right] + (t1 - target_ns) / 1e6,
        )
        return output, age_ms, retarget_age_ms, (t1 - t0) / 1e6


class JointStateSeries:
    def __init__(self, events: Iterable[Dict[str, Any]]) -> None:
        self.events = sorted(events, key=event_monotonic_ns)
        self.times = [event_monotonic_ns(event) for event in self.events]
        self._cursor = -1
        self._last_target_ns = -1
        self._positions: Dict[str, float] = {}
        self._velocities: Dict[str, float] = {}
        self._position_update_time: Dict[str, int] = {}
        self._velocity_update_time: Dict[str, int] = {}

    def _reset(self) -> None:
        self._cursor = -1
        self._positions.clear()
        self._velocities.clear()
        self._position_update_time.clear()
        self._velocity_update_time.clear()

    def _apply_event(self, event_index: int) -> None:
        event = self.events[event_index]
        names = event.get("name", [])
        q = event.get("position", [])
        dq = event.get("velocity", [])
        timestamp_ns = self.times[event_index]
        if not isinstance(names, list):
            return
        for index, name_value in enumerate(names):
            name = str(name_value)
            if name not in X2_TRACKING_JOINT_NAMES:
                continue
            if index < len(q):
                self._positions[name] = float(q[index])
                self._position_update_time[name] = timestamp_ns
            if index < len(dq):
                self._velocities[name] = float(dq[index])
                self._velocity_update_time[name] = timestamp_ns

    def sample(
        self, target_ns: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, list[str]]:
        if target_ns < self._last_target_ns:
            self._reset()
        self._last_target_ns = target_ns

        while self._cursor + 1 < len(self.events) and self.times[self._cursor + 1] <= target_ns:
            self._cursor += 1
            self._apply_event(self._cursor)

        if self._cursor < 0:
            return None, None, math.inf, list(X2_TRACKING_JOINT_NAMES)

        missing = [name for name in X2_TRACKING_JOINT_NAMES if name not in self._positions]
        missing.extend(
            f"velocity:{name}" for name in X2_TRACKING_JOINT_NAMES if name not in self._velocities
        )
        if missing:
            return None, None, math.inf, missing

        position_vector = np.asarray(
            [self._positions[name] for name in X2_TRACKING_JOINT_NAMES], dtype=np.float64
        )
        velocity_vector = np.asarray(
            [self._velocities[name] for name in X2_TRACKING_JOINT_NAMES], dtype=np.float64
        )
        max_age_ms = max(
            max(
                target_ns - self._position_update_time[name],
                target_ns - self._velocity_update_time[name],
            )
            for name in X2_TRACKING_JOINT_NAMES
        ) / 1e6
        return position_vector, velocity_vector, max_age_ms, []


class PreviousEventSeries:
    def __init__(self, events: Iterable[Dict[str, Any]]) -> None:
        self.events = sorted(events, key=event_monotonic_ns)
        self.times = [event_monotonic_ns(event) for event in self.events]

    def sample(self, target_ns: int) -> tuple[Optional[Dict[str, Any]], float]:
        index = _previous_index(self.times, target_ns)
        if index is None:
            return None, math.inf
        return self.events[index], (target_ns - self.times[index]) / 1e6


def _decode_rgb(image_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to decode raw camera frames") from exc
    encoded = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode camera frame: {image_path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _validate_manifest(manifest: Dict[str, Any], episode_dir: Path) -> None:
    if manifest.get("schema_version") != RAW_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported raw schema in {episode_dir}: {manifest.get('schema_version')!r}"
        )
    if manifest.get("robot_type") != "agibot_x2":
        raise ValueError(f"Unexpected robot_type in {episode_dir}: {manifest.get('robot_type')!r}")
    if manifest.get("joint_order") != X2_TRACKING_JOINT_NAMES:
        raise ValueError(f"Joint order does not match rl_tracking in {episode_dir}")
    if not str(manifest.get("task", "")).strip():
        raise ValueError(f"Episode task is empty: {episode_dir}")


def split_contiguous_samples(
    samples: Sequence[ConvertedSample], *, fps: int, min_frames: int
) -> tuple[list[list[ConvertedSample]], int]:
    """Split at missing target ticks so LeRobot never compresses time."""

    if not samples:
        return [], 0
    step_ns = int(round(1e9 / float(fps)))
    raw_segments: list[list[ConvertedSample]] = [[samples[0]]]
    for sample in samples[1:]:
        if sample.timestamp_ns - raw_segments[-1][-1].timestamp_ns == step_ns:
            raw_segments[-1].append(sample)
        else:
            raw_segments.append([sample])
    kept = [segment for segment in raw_segments if len(segment) >= min_frames]
    dropped = sum(len(segment) for segment in raw_segments if len(segment) < min_frames)
    return kept, dropped


def _validate_segment_images(
    segments: Sequence[tuple[str, Sequence[ConvertedSample]]],
) -> tuple[int, int, int]:
    first_shape: Optional[tuple[int, int, int]] = None
    decoded_paths: set[Path] = set()
    for _, samples in segments:
        for sample in samples:
            if sample.image_path in decoded_paths:
                continue
            rgb = _decode_rgb(sample.image_path)
            shape = tuple(rgb.shape)
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise ValueError(
                    f"Camera shape changed from {first_shape} to {shape}: {sample.image_path}"
                )
            decoded_paths.add(sample.image_path)
    if first_shape is None:
        raise RuntimeError("No images found in synchronized segments")
    return first_shape


def build_episode_samples(
    episode_dir: Path,
    *,
    fps: int,
    max_camera_age_ms: float,
    max_reference_age_ms: float,
    max_reference_gap_ms: float,
    max_retarget_age_ms: float,
    max_joint_age_ms: float,
    max_imu_age_ms: float,
) -> EpisodeSamples:
    manifest = _load_manifest(episode_dir)
    _validate_manifest(manifest, episode_dir)
    recording = manifest.get("recording", {})
    start_ns = int(recording["start_monotonic_ns"])
    stop_value = recording.get("stop_trigger_monotonic_ns")
    if stop_value is None:
        raise ValueError(f"Episode has no stop trigger: {episode_dir}")
    stop_ns = int(stop_value)
    if stop_ns <= start_ns:
        raise ValueError(f"Episode stop is not after start: {episode_dir}")

    reference_events = _load_stream(episode_dir, "reference")
    if not reference_events:
        reference_events = _load_stream(episode_dir, "retarget")
    references = ReferenceSeries(reference_events)
    joints = JointStateSeries(_load_stream(episode_dir, "joint_states"))
    imu = PreviousEventSeries(_load_stream(episode_dir, "imu_torso"))
    camera_events = _load_stream(episode_dir, "camera_head")
    camera_times = [event_monotonic_ns(event) for event in camera_events]

    step_ns = int(round(1e9 / float(fps)))
    target_times = list(range(start_ns, stop_ns + 1, step_ns))
    samples: list[ConvertedSample] = []
    skip_counts: Counter[str] = Counter()

    for target_ns in target_times:
        camera_index = _nearest_index(camera_times, target_ns)
        if camera_index is None:
            skip_counts["camera_missing"] += 1
            continue
        camera_event = camera_events[camera_index]
        camera_age_ms = abs(camera_times[camera_index] - target_ns) / 1e6
        if camera_age_ms > max_camera_age_ms:
            skip_counts["camera_stale"] += 1
            continue

        action, reference_age_ms, retarget_age_ms, reference_gap_ms = references.interpolate(
            target_ns
        )
        if action is None:
            skip_counts["reference_missing"] += 1
            continue
        if reference_age_ms > max_reference_age_ms:
            skip_counts["reference_stale"] += 1
            continue
        if reference_gap_ms > max_reference_gap_ms:
            skip_counts["reference_gap"] += 1
            continue
        if not math.isfinite(retarget_age_ms) or retarget_age_ms > max_retarget_age_ms:
            skip_counts["retarget_stale"] += 1
            continue

        q, dq, joint_age_ms, missing_joints = joints.sample(target_ns)
        if q is None or dq is None:
            skip_counts["joint_missing"] += 1
            if missing_joints:
                skip_counts[f"missing:{','.join(missing_joints)}"] += 1
            continue
        if joint_age_ms > max_joint_age_ms:
            skip_counts["joint_stale"] += 1
            continue

        imu_event, imu_age_ms = imu.sample(target_ns)
        if imu_event is None:
            skip_counts["imu_missing"] += 1
            continue
        if imu_age_ms > max_imu_age_ms:
            skip_counts["imu_stale"] += 1
            continue

        orientation = np.asarray(imu_event.get("orientation_xyzw", []), dtype=np.float64)
        angular_velocity = np.asarray(
            imu_event.get("angular_velocity_xyz", []), dtype=np.float64
        )
        linear_acceleration = np.asarray(
            imu_event.get("linear_acceleration_xyz", []), dtype=np.float64
        )
        if (
            orientation.shape != (4,)
            or angular_velocity.shape != (3,)
            or linear_acceleration.shape != (3,)
        ):
            skip_counts["imu_invalid"] += 1
            continue
        orientation_covariance = imu_event.get("orientation_covariance", [])
        if (
            isinstance(orientation_covariance, list)
            and orientation_covariance
            and float(orientation_covariance[0]) == -1.0
        ):
            skip_counts["imu_orientation_unavailable"] += 1
            continue
        orientation_norm = float(np.linalg.norm(orientation))
        if not np.isfinite(orientation_norm) or orientation_norm < 1e-8:
            skip_counts["imu_orientation_invalid"] += 1
            continue
        orientation = orientation / orientation_norm

        state = np.concatenate([q, dq, orientation, angular_velocity, linear_acceleration]).astype(
            np.float32
        )
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (68,) or action.shape != (36,):
            skip_counts["shape_invalid"] += 1
            continue
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            skip_counts["non_finite"] += 1
            continue

        relative_path = camera_event.get("image_path")
        if not isinstance(relative_path, str):
            skip_counts["camera_path_missing"] += 1
            continue
        image_path = episode_dir / relative_path
        if not image_path.is_file():
            skip_counts["camera_file_missing"] += 1
            continue

        samples.append(
            ConvertedSample(
                timestamp_ns=target_ns,
                image_path=image_path,
                state=state,
                action=action,
                timing_ms=np.asarray(
                    [camera_age_ms, reference_age_ms, joint_age_ms, imu_age_ms],
                    dtype=np.float32,
                ),
            )
        )

    return EpisodeSamples(
        episode_dir=episode_dir,
        task=str(manifest.get("task", "")),
        samples=samples,
        candidate_count=len(target_times),
        skip_counts=dict(skip_counts),
    )


def _discover_episodes(raw_root: Path, require_success: bool) -> list[Path]:
    episodes: list[Path] = []
    for episode_dir in sorted(raw_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        manifest = _load_manifest(episode_dir)
        if manifest.get("status") != "complete":
            continue
        if require_success and manifest.get("success") is not True:
            continue
        episodes.append(episode_dir)
    return episodes


def _create_lerobot_dataset(
    *,
    repo_id: str,
    output_root: Path,
    fps: int,
    image_shape: tuple[int, int, int],
) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not installed. The pinned LeRobot v0.6 requires Python >=3.12; "
            "create a separate converter environment and install 'lerobot[dataset]==0.6.0'. "
            "Use --dry_run first in the current GMR environment."
        ) from exc

    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(OBSERVATION_STATE_NAMES),),
            "names": OBSERVATION_STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="agibot_x2",
        features=features,
        use_videos=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw X2 VR episodes to LeRobotDataset v3")
    parser.add_argument("--raw_root", default="~/Datasets/x2_vr/raw")
    parser.add_argument("--output_root", default="~/Datasets/x2_vr/lerobot_v3")
    parser.add_argument("--repo_id", default="local/x2_vr")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max_camera_age_ms", type=float, default=100.0)
    parser.add_argument("--max_reference_age_ms", type=float, default=60.0)
    parser.add_argument("--max_reference_gap_ms", type=float, default=120.0)
    parser.add_argument("--max_retarget_age_ms", type=float, default=100.0)
    parser.add_argument("--max_joint_age_ms", type=float, default=100.0)
    parser.add_argument("--max_imu_age_ms", type=float, default=100.0)
    parser.add_argument(
        "--min_segment_frames",
        type=int,
        default=2,
        help="Discard shorter fragments after splitting an episode at missing target ticks",
    )
    parser.add_argument("--require_success", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.min_segment_frames <= 0:
        raise ValueError("--min_segment_frames must be positive")

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")
    episode_dirs = _discover_episodes(raw_root, require_success=args.require_success)
    if not episode_dirs:
        raise RuntimeError("No eligible complete episodes found")

    converted: list[EpisodeSamples] = []
    total_candidates = 0
    total_samples = 0
    for episode_dir in episode_dirs:
        result = build_episode_samples(
            episode_dir,
            fps=args.fps,
            max_camera_age_ms=args.max_camera_age_ms,
            max_reference_age_ms=args.max_reference_age_ms,
            max_reference_gap_ms=args.max_reference_gap_ms,
            max_retarget_age_ms=args.max_retarget_age_ms,
            max_joint_age_ms=args.max_joint_age_ms,
            max_imu_age_ms=args.max_imu_age_ms,
        )
        converted.append(result)
        total_candidates += result.candidate_count
        total_samples += len(result.samples)
        print(
            f"[convert] {episode_dir.name}: accepted={len(result.samples)}/"
            f"{result.candidate_count}, skipped={result.skip_counts}"
        )

    print(
        f"[convert] total accepted={total_samples}/{total_candidates} "
        f"({100.0 * total_samples / max(1, total_candidates):.1f}%)"
    )
    output_segments: list[tuple[str, list[ConvertedSample]]] = []
    short_fragment_frames = 0
    for episode in converted:
        segments, dropped_short = split_contiguous_samples(
            episode.samples,
            fps=args.fps,
            min_frames=args.min_segment_frames,
        )
        short_fragment_frames += dropped_short
        output_segments.extend((episode.task, segment) for segment in segments)
    if not output_segments:
        raise RuntimeError(
            "All synchronized episodes are empty; inspect missing/stale counters above"
        )
    print(
        f"[convert] continuous segments={len(output_segments)}, "
        f"usable_frames={sum(len(segment) for _, segment in output_segments)}, "
        f"short_fragment_frames_dropped={short_fragment_frames}"
    )

    first_shape = _validate_segment_images(output_segments)
    if args.dry_run:
        first_sample = output_segments[0][1][0]
        print(
            f"[convert] dry-run OK: image={first_shape}, "
            f"state={first_sample.state.shape}, action={first_sample.action.shape}"
        )
        return

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. "
            "Choose a new path; raw conversion never deletes data."
        )
    if (
        output_root == raw_root
        or output_root in raw_root.parents
        or raw_root in output_root.parents
    ):
        raise ValueError("--output_root and --raw_root must not overlap")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(f".{output_root.name}.partial-{uuid.uuid4().hex}")

    dataset = _create_lerobot_dataset(
        repo_id=args.repo_id,
        output_root=staging_root,
        fps=args.fps,
        image_shape=first_shape,
    )
    try:
        for task, segment in output_segments:
            for sample in segment:
                rgb = _decode_rgb(sample.image_path)
                dataset.add_frame(
                    {
                        "observation.images.head": rgb,
                        "observation.state": sample.state.astype(np.float32, copy=False),
                        "action": sample.action.astype(np.float32, copy=False),
                        "task": task,
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()

    staging_root.replace(output_root)
    print(f"[convert] LeRobotDataset v3 written to {output_root}")


if __name__ == "__main__":
    main()
