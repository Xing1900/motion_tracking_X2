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
import warnings
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
        REFERENCE_ACTION_NAMES,
        TIMING_NAMES,
        X2_TRACKING_JOINT_NAMES,
    )
    from .synchronization import (
        SYNC_TIME_KEY,
        TIME_BASES,
        TIME_BASIS_RECEIVER,
        TIME_BASIS_SOURCE,
        TimedStream,
        synchronization_time_ns,
        timed_stream,
    )
except ImportError:  # Direct execution from this directory.
    from raw_episode_writer import event_monotonic_ns, read_jsonl
    from schema import (
        ACTION_NAMES,
        OBSERVATION_STATE_NAMES,
        RAW_DATASET_SCHEMA_VERSION,
        REFERENCE_ACTION_NAMES,
        TIMING_NAMES,
        X2_TRACKING_JOINT_NAMES,
    )
    from synchronization import (
        SYNC_TIME_KEY,
        TIME_BASES,
        TIME_BASIS_RECEIVER,
        TIME_BASIS_SOURCE,
        TimedStream,
        synchronization_time_ns,
        timed_stream,
    )


@dataclass
class ConvertedSample:
    timestamp_ns: int
    image_path: Path
    state: np.ndarray
    action: np.ndarray
    timing_ms: np.ndarray
    observation_timestamp_ns: Optional[int] = None


@dataclass
class EpisodeSamples:
    episode_dir: Path
    task: str
    samples: list[ConvertedSample]
    candidate_count: int
    skip_counts: Dict[str, int]
    time_basis: str = TIME_BASIS_SOURCE
    synchronization_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None


CONVERSION_REPORT_SCHEMA_VERSION = "x2-vr-lerobot-conversion-v1"


def _load_manifest(episode_dir: Path) -> Dict[str, Any]:
    path = episode_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_stream(episode_dir: Path, stream: str) -> list[Dict[str, Any]]:
    events = list(read_jsonl(episode_dir / "streams" / f"{stream}.jsonl"))
    events.sort(key=event_monotonic_ns)
    return events


def _load_timed_stream(
    episode_dir: Path,
    stream: str,
    *,
    time_basis: str,
) -> TimedStream:
    events = list(read_jsonl(episode_dir / "streams" / f"{stream}.jsonl"))
    return timed_stream(events, stream, time_basis=time_basis)


def _series_time_ns(event: Dict[str, Any]) -> int:
    if SYNC_TIME_KEY in event:
        return synchronization_time_ns(event)
    return event_monotonic_ns(event)


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
    def __init__(
        self,
        events: Iterable[Dict[str, Any]],
        *,
        account_for_scheduled_frame_age: bool = True,
    ) -> None:
        samples: list[tuple[int, np.ndarray, float]] = []
        for event in events:
            base_time_ns = _series_time_ns(event)
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
                    if vector.shape == (len(REFERENCE_ACTION_NAMES),) and np.all(
                        np.isfinite(vector)
                    ):
                        # A chunk is available in one reply but its later
                        # frames are scheduled for later controller ticks.  By
                        # then the GMR input used to construct the chunk is
                        # older by the same frame offset.
                        scheduled_retarget_age_ms = retarget_age_ms
                        if account_for_scheduled_frame_age:
                            scheduled_retarget_age_ms += (
                                frame_index * frame_dt_ns / 1e6
                            )
                        samples.append(
                            (
                                base_time_ns + frame_index * frame_dt_ns,
                                vector,
                                scheduled_retarget_age_ms,
                            )
                        )
                continue

            # Fallback for the pre-interpolation retarget tap.
            frame = event.get("qpos_root_xyz_quat_wxyz_dof")
            if frame is not None:
                vector = np.asarray(frame, dtype=np.float64).reshape(-1)
                if vector.shape == (len(REFERENCE_ACTION_NAMES),) and np.all(
                    np.isfinite(vector)
                ):
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

    def sample_previous(
        self, target_ns: int
    ) -> tuple[Optional[np.ndarray], float, float, float]:
        """Return the command actually available at or before ``target_ns``.

        Reference events are discrete commands sent toward C++.  Causal
        previous-event sampling avoids inventing a command by interpolating
        with a reply that did not yet exist at the camera exposure time.
        """

        index = _previous_index(self.times, target_ns)
        if index is None:
            return None, math.inf, math.inf, math.inf
        command_age_ms = (target_ns - self.times[index]) / 1e6
        next_index = index + 1
        command_gap_ms = (
            (self.times[next_index] - self.times[index]) / 1e6
            if next_index < len(self.times)
            else 0.0
        )
        return (
            self.values[index].copy(),
            command_age_ms,
            self.retarget_ages_ms[index] + command_age_ms,
            command_gap_ms,
        )

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
        self.events = sorted(events, key=_series_time_ns)
        self.times = [_series_time_ns(event) for event in self.events]
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
        self.events = sorted(events, key=_series_time_ns)
        self.times = [_series_time_ns(event) for event in self.events]

    def sample(self, target_ns: int) -> tuple[Optional[Dict[str, Any]], float]:
        index = _previous_index(self.times, target_ns)
        if index is None:
            return None, math.inf
        return self.events[index], (target_ns - self.times[index]) / 1e6


class HandCommandSeries:
    """Causally sample authoritative or legacy high-level hand actions.

    Hand commands are zero-order-held: a target tick may use only the most
    recent event at or before that tick.  Nearest-neighbour sampling would let
    a future grip transition leak into an earlier camera/robot observation.
    """

    LEGACY_GRIP_DEADZONE = 0.10
    LEGACY_GRIP_FULL_SCALE = 0.90
    UINT32_MODULUS = 1 << 32

    def __init__(self, events: Iterable[Dict[str, Any]], *, legacy_controller: bool) -> None:
        self.series = PreviousEventSeries(events)
        self.legacy_controller = bool(legacy_controller)

    @staticmethod
    def _unit_pair(left: Any, right: Any) -> Optional[np.ndarray]:
        try:
            values = np.asarray([float(left), float(right)], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            return None
        if np.any(values < 0.0) or np.any(values > 1.0):
            return None
        return values

    @classmethod
    def _normalize_legacy_grips(cls, values: np.ndarray) -> np.ndarray:
        return np.clip(
            (values - cls.LEGACY_GRIP_DEADZONE)
            / (cls.LEGACY_GRIP_FULL_SCALE - cls.LEGACY_GRIP_DEADZONE),
            0.0,
            1.0,
        )

    @classmethod
    def _uint32_sequence(cls, value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            sequence = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if sequence < 0 or sequence >= cls.UINT32_MODULUS:
            return None
        if isinstance(value, float) and not value.is_integer():
            return None
        return sequence

    def _authoritative_sequence_status(self, index: int, target_ns: int) -> str:
        current_sequence = self._uint32_sequence(
            self.series.events[index].get("sequence")
        )
        if current_sequence is None:
            return "invalid"

        next_index = index + 1
        if next_index >= len(self.series.events):
            return "ok"
        current_time_ns = self.series.times[index]
        next_time_ns = self.series.times[next_index]
        if not current_time_ns < target_ns < next_time_ns:
            # A published event is authoritative at its own timestamp, even
            # when one or more status events were lost before the next event.
            return "ok"

        next_sequence = self._uint32_sequence(
            self.series.events[next_index].get("sequence")
        )
        if next_sequence is None:
            return "invalid"
        sequence_delta = (
            next_sequence - current_sequence
        ) % self.UINT32_MODULUS
        return "sequence_gap" if sequence_delta > 1 else "ok"

    def sample(self, target_ns: int) -> tuple[Optional[np.ndarray], float, str]:
        index = _previous_index(self.series.times, target_ns)
        if index is None:
            return None, math.inf, "missing"
        event = self.series.events[index]
        event_age_ms = (target_ns - self.series.times[index]) / 1e6

        if not self.legacy_controller:
            sequence_status = self._authoritative_sequence_status(index, target_ns)
            if sequence_status != "ok":
                return None, event_age_ms, sequence_status
            active = event.get("active")
            if active is False:
                return None, event_age_ms, "inactive"
            if active is not True:
                return None, event_age_ms, "invalid"
            values = self._unit_pair(event.get("left_grasp"), event.get("right_grasp"))
            if values is None:
                return None, event_age_ms, "invalid"
            return values, event_age_ms, "ok"

        buttons = event.get("controller_buttons")
        if not isinstance(buttons, dict):
            return None, event_age_ms, "invalid"
        values = self._unit_pair(
            buttons.get("left_grip_value"), buttons.get("right_grip_value")
        )
        if values is None:
            return None, event_age_ms, "invalid"

        # Legacy controller events predate the authoritative hand status.  When
        # available, include the XR-source age carried by the bridge instead of
        # treating a newly re-published frozen controller value as fresh.
        raw_source_age_ms = event.get("controller_age_ms")
        if raw_source_age_ms is not None:
            try:
                source_age_ms = float(raw_source_age_ms)
            except (TypeError, ValueError):
                return None, math.inf, "invalid"
            if not math.isfinite(source_age_ms) or source_age_ms < 0.0:
                return None, math.inf, "invalid"
            event_age_ms += source_age_ms

        return self._normalize_legacy_grips(values), event_age_ms, "ok"


def _load_hand_command_series(
    episode_dir: Path,
    *,
    time_basis: str = TIME_BASIS_RECEIVER,
) -> HandCommandSeries:
    hand_command_stream = _load_timed_stream(
        episode_dir, "hand_command", time_basis=time_basis
    )
    if hand_command_stream.input_count:
        return HandCommandSeries(hand_command_stream.events, legacy_controller=False)

    controller_events = _load_timed_stream(
        episode_dir, "controller", time_basis=time_basis
    ).events
    warnings.warn(
        f"{episode_dir}: authoritative hand_command stream is missing; "
        "deriving left/right grasp actions from legacy controller grip values "
        "with deadzone=0.10 and full_scale=0.90",
        RuntimeWarning,
        stacklevel=2,
    )
    return HandCommandSeries(controller_events, legacy_controller=True)


def _decode_rgb(image_path: Path, *, rotation_deg: int = 0) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to decode raw camera frames") from exc
    if rotation_deg not in (0, 180):
        raise ValueError("camera rotation must be 0 or 180 degrees")
    encoded = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode camera frame: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rotation_deg == 180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


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
    segments: Sequence[tuple[EpisodeSamples, Sequence[ConvertedSample]]],
    *,
    camera_rotation_deg: int,
) -> tuple[int, int, int]:
    first_shape: Optional[tuple[int, int, int]] = None
    decoded_paths: set[Path] = set()
    for _, samples in segments:
        for sample in samples:
            if sample.image_path in decoded_paths:
                continue
            rgb = _decode_rgb(
                sample.image_path, rotation_deg=camera_rotation_deg
            )
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


def _conversion_report_payload(
    *,
    raw_root: Path,
    output_root: Path,
    repo_id: str,
    fps: int,
    time_basis: str,
    allow_partial_source_time: bool,
    max_camera_age_ms: float,
    max_reference_age_ms: float,
    max_reference_gap_ms: float,
    max_retarget_age_ms: float,
    max_hand_command_age_ms: float,
    max_joint_age_ms: float,
    max_imu_age_ms: float,
    camera_rotation_deg: int,
    min_segment_frames: int,
    require_success: bool,
    converted: Sequence[EpisodeSamples],
    output_segments: Sequence[tuple[EpisodeSamples, Sequence[ConvertedSample]]],
    short_fragment_frames_dropped: int,
) -> Dict[str, Any]:
    """Build the immutable provenance report stored beside LeRobot output."""

    segment_records: list[Dict[str, Any]] = []
    segment_indices_by_episode: Dict[Path, list[int]] = {}
    for dataset_episode_index, (episode, samples) in enumerate(output_segments):
        if not samples:
            raise ValueError("conversion report cannot describe an empty output segment")
        first = samples[0]
        last = samples[-1]
        if first.observation_timestamp_ns is None or last.observation_timestamp_ns is None:
            raise ValueError("output segment is missing camera synchronization timestamps")
        segment_indices_by_episode.setdefault(episode.episode_dir, []).append(
            dataset_episode_index
        )
        segment_records.append(
            {
                "dataset_episode_index": dataset_episode_index,
                "raw_episode": episode.episode_dir.name,
                "raw_episode_path": str(episode.episode_dir),
                "task": episode.task,
                "frame_count": len(samples),
                "first_target_timestamp_ns": int(first.timestamp_ns),
                "last_target_timestamp_ns": int(last.timestamp_ns),
                "first_camera_timestamp_ns": int(first.observation_timestamp_ns),
                "last_camera_timestamp_ns": int(last.observation_timestamp_ns),
            }
        )

    episode_records: list[Dict[str, Any]] = []
    for episode in converted:
        output_segment_indices = segment_indices_by_episode.get(
            episode.episode_dir, []
        )
        episode_records.append(
            {
                "raw_episode": episode.episode_dir.name,
                "raw_episode_path": str(episode.episode_dir),
                "task": episode.task,
                "candidate_ticks": int(episode.candidate_count),
                "accepted_ticks_before_segmentation": len(episode.samples),
                "skip_counts": episode.skip_counts,
                "synchronization_diagnostics": episode.synchronization_diagnostics,
                "output_segment_indices": output_segment_indices,
                "output_segment_frame_counts": [
                    int(segment_records[index]["frame_count"])
                    for index in output_segment_indices
                ],
            }
        )

    thresholds_ms = {
        "camera": float(max_camera_age_ms),
        "reference": float(max_reference_age_ms),
        "reference_gap": float(max_reference_gap_ms),
        "retarget": float(max_retarget_age_ms),
        "hand_command": float(max_hand_command_age_ms),
        "joint": float(max_joint_age_ms),
        "imu": float(max_imu_age_ms),
    }
    return {
        "schema_version": CONVERSION_REPORT_SCHEMA_VERSION,
        "raw_dataset_schema_version": RAW_DATASET_SCHEMA_VERSION,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "repo_id": str(repo_id),
        "time_basis": str(time_basis),
        "fps": int(fps),
        "conversion": {
            "thresholds_ms": thresholds_ms,
            "camera_rotation_deg": int(camera_rotation_deg),
            "min_segment_frames": int(min_segment_frames),
            "require_success": bool(require_success),
            "allow_partial_source_time": bool(allow_partial_source_time),
        },
        "summary": {
            "raw_episode_count": len(converted),
            "candidate_ticks": sum(
                int(episode.candidate_count) for episode in converted
            ),
            "accepted_ticks_before_segmentation": sum(
                len(episode.samples) for episode in converted
            ),
            "output_segment_count": len(segment_records),
            "output_frame_count": sum(
                int(record["frame_count"]) for record in segment_records
            ),
            "short_fragment_frames_dropped": int(
                short_fragment_frames_dropped
            ),
        },
        "raw_episodes": episode_records,
        "output_segments": segment_records,
        "timestamp_note": (
            "target and camera timestamps are integer nanoseconds in the "
            "recorder-monotonic domain selected by time_basis; each output "
            "segment is contiguous at 1/fps"
        ),
    }


def _write_conversion_report(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def build_episode_samples(
    episode_dir: Path,
    *,
    fps: int,
    max_camera_age_ms: float,
    max_reference_age_ms: float,
    max_reference_gap_ms: float,
    max_retarget_age_ms: float,
    max_hand_command_age_ms: float,
    max_joint_age_ms: float,
    max_imu_age_ms: float,
    time_basis: str = TIME_BASIS_SOURCE,
    allow_partial_source_time: bool = False,
) -> EpisodeSamples:
    if time_basis not in TIME_BASES:
        raise ValueError(f"unsupported time basis: {time_basis!r}")
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

    timed_streams: Dict[str, TimedStream] = {
        name: _load_timed_stream(episode_dir, name, time_basis=time_basis)
        for name in ("reference", "hand_command", "joint_states", "imu_torso", "camera_head")
    }
    reference_stream = timed_streams["reference"]
    reference_source_name = "reference"
    if reference_stream.input_count == 0:
        reference_stream = _load_timed_stream(
            episode_dir, "retarget", time_basis=time_basis
        )
        timed_streams["retarget"] = reference_stream
        reference_source_name = "retarget"
    references = ReferenceSeries(
        reference_stream.events,
        # Source mode describes the age at the scheduled execution time of
        # each frame in a reply chunk.  Receiver mode intentionally preserves
        # the old converter's packet-level age for exact comparison.
        account_for_scheduled_frame_age=(time_basis == TIME_BASIS_SOURCE),
    )

    hand_stream = timed_streams["hand_command"]
    if hand_stream.input_count:
        hand_commands = HandCommandSeries(hand_stream.events, legacy_controller=False)
        hand_source_name = "hand_command"
        selected_hand_stream = hand_stream
    else:
        controller_stream = _load_timed_stream(
            episode_dir, "controller", time_basis=time_basis
        )
        timed_streams["controller"] = controller_stream
        warnings.warn(
            f"{episode_dir}: authoritative hand_command stream is missing; "
            "deriving left/right grasp actions from legacy controller grip values "
            "with deadzone=0.10 and full_scale=0.90",
            RuntimeWarning,
            stacklevel=2,
        )
        hand_commands = HandCommandSeries(controller_stream.events, legacy_controller=True)
        hand_source_name = "controller"
        selected_hand_stream = controller_stream

    if time_basis == TIME_BASIS_SOURCE and not allow_partial_source_time:
        required_timed_streams = {
            "camera_head": timed_streams["camera_head"],
            "joint_states": timed_streams["joint_states"],
            "imu_torso": timed_streams["imu_torso"],
            reference_source_name: reference_stream,
            hand_source_name: selected_hand_stream,
        }
        incomplete = {
            name: (stream.missing_time_count, stream.input_count)
            for name, stream in required_timed_streams.items()
            if stream.missing_time_count > 0
        }
        if incomplete:
            details = ", ".join(
                f"{name}={missing}/{total}"
                for name, (missing, total) in sorted(incomplete.items())
            )
            raise ValueError(
                f"Source-time mapping failed for {episode_dir}: {details}. "
                "Use --allow_partial_source_time only for explicit diagnostic "
                "conversion of incomplete legacy data, or use "
                "--time_basis receiver to reproduce the legacy alignment."
            )
        regressed = {
            name: (
                stream.large_source_regression_count,
                stream.max_source_regression_ms,
            )
            for name, stream in required_timed_streams.items()
            if stream.large_source_regression_count > 0
        }
        if regressed:
            details = ", ".join(
                f"{name}={count} (max={maximum_ms:.3f} ms)"
                for name, (count, maximum_ms) in sorted(regressed.items())
            )
            raise ValueError(
                f"Large source-time regression detected for {episode_dir}: {details}. "
                "This usually indicates a clock reset or corrupt stream order; "
                "do not sort it silently into a training episode."
            )

    joints = JointStateSeries(timed_streams["joint_states"].events)
    imu = PreviousEventSeries(timed_streams["imu_torso"].events)
    camera_events = timed_streams["camera_head"].events
    if time_basis == TIME_BASIS_SOURCE:
        # Pre/post-roll remains in raw storage, but training observations must
        # be captured inside the A-to-X active interval.
        camera_events = [
            event
            for event in camera_events
            if start_ns <= _series_time_ns(event) <= stop_ns
        ]
    camera_times = [_series_time_ns(event) for event in camera_events]

    synchronization_diagnostics = {
        name: {
            "input_events": stream.input_count,
            "timed_events": len(stream.events),
            "missing_time_events": stream.missing_time_count,
            "source_regressions": stream.source_regression_count,
            "large_source_regressions": stream.large_source_regression_count,
            "max_source_regression_ms": stream.max_source_regression_ms,
        }
        for name, stream in sorted(timed_streams.items())
    }
    synchronization_diagnostics["reference_selection"] = {
        "input_events": reference_stream.input_count,
        "timed_events": len(reference_stream.events),
        "missing_time_events": reference_stream.missing_time_count,
        "source_regressions": reference_stream.source_regression_count,
        "large_source_regressions": reference_stream.large_source_regression_count,
        "max_source_regression_ms": reference_stream.max_source_regression_ms,
        "using_reference_stream": int(reference_source_name == "reference"),
    }

    step_ns = int(round(1e9 / float(fps)))
    target_times = list(range(start_ns, stop_ns + 1, step_ns))
    samples: list[ConvertedSample] = []
    skip_counts: Counter[str] = Counter()
    last_camera_index: Optional[int] = None
    last_camera_anchor_ns: Optional[int] = None

    for target_ns in target_times:
        camera_index = _nearest_index(camera_times, target_ns)
        if camera_index is None:
            skip_counts["camera_missing"] += 1
            continue
        camera_event = camera_events[camera_index]
        camera_anchor_ns = camera_times[camera_index]
        camera_grid_offset_ms = (camera_anchor_ns - target_ns) / 1e6
        if abs(camera_grid_offset_ms) > max_camera_age_ms:
            skip_counts["camera_stale"] += 1
            continue

        if time_basis == TIME_BASIS_SOURCE:
            if camera_index == last_camera_index:
                skip_counts["camera_reused"] += 1
                continue
            last_camera_index = camera_index
            if last_camera_anchor_ns is not None and camera_anchor_ns <= last_camera_anchor_ns:
                skip_counts["camera_non_monotonic"] += 1
                continue
            last_camera_anchor_ns = camera_anchor_ns
            association_ns = camera_anchor_ns
            reference_action, reference_age_ms, retarget_age_ms, reference_gap_ms = (
                references.sample_previous(association_ns)
            )
        else:
            # Exact compatibility path: the original receiver-clock converter
            # associated every non-camera stream with the fixed FPS target,
            # allowed one image to serve adjacent targets, and interpolated
            # reference commands around that target.
            association_ns = target_ns
            reference_action, reference_age_ms, retarget_age_ms, reference_gap_ms = (
                references.interpolate(association_ns)
            )

        if reference_action is None:
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

        hand_action, hand_command_age_ms, hand_status = hand_commands.sample(association_ns)
        if hand_action is None:
            skip_counts[f"hand_command_{hand_status}"] += 1
            continue
        if hand_command_age_ms > max_hand_command_age_ms:
            skip_counts["hand_command_stale"] += 1
            continue

        q, dq, joint_age_ms, missing_joints = joints.sample(association_ns)
        if q is None or dq is None:
            skip_counts["joint_missing"] += 1
            if missing_joints:
                skip_counts[f"missing:{','.join(missing_joints)}"] += 1
            continue
        if joint_age_ms > max_joint_age_ms:
            skip_counts["joint_stale"] += 1
            continue

        imu_event, imu_age_ms = imu.sample(association_ns)
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
        action = np.concatenate([reference_action, hand_action]).astype(np.float32)
        if state.shape != (68,) or action.shape != (len(ACTION_NAMES),):
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
                    [
                        camera_grid_offset_ms,
                        reference_age_ms,
                        hand_command_age_ms,
                        joint_age_ms,
                        imu_age_ms,
                    ],
                    dtype=np.float32,
                ),
                observation_timestamp_ns=camera_anchor_ns,
            )
        )

    return EpisodeSamples(
        episode_dir=episode_dir,
        task=str(manifest.get("task", "")),
        samples=samples,
        candidate_count=len(target_times),
        skip_counts=dict(skip_counts),
        time_basis=time_basis,
        synchronization_diagnostics=synchronization_diagnostics,
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
        "sync.timing_ms": {
            "dtype": "float32",
            "shape": (len(TIMING_NAMES),),
            "names": TIMING_NAMES,
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
    parser.add_argument(
        "--time_basis",
        choices=TIME_BASES,
        default=TIME_BASIS_SOURCE,
        help=(
            "Use source/header timestamps (recommended) or reproduce the legacy "
            "recorder-arrival alignment"
        ),
    )
    parser.add_argument(
        "--allow_partial_source_time",
        action="store_true",
        help=(
            "Allow source-time conversion after dropping raw events whose source "
            "clock cannot be mapped. Diagnostic/legacy escape hatch only; strict "
            "source conversion fails by default."
        ),
    )
    parser.add_argument("--max_camera_age_ms", type=float, default=100.0)
    parser.add_argument("--max_reference_age_ms", type=float, default=60.0)
    parser.add_argument("--max_reference_gap_ms", type=float, default=120.0)
    parser.add_argument("--max_retarget_age_ms", type=float, default=100.0)
    parser.add_argument("--max_hand_command_age_ms", type=float, default=100.0)
    parser.add_argument("--max_joint_age_ms", type=float, default=100.0)
    parser.add_argument("--max_imu_age_ms", type=float, default=100.0)
    parser.add_argument(
        "--camera_rotation_deg",
        type=int,
        choices=(0, 180),
        default=180,
        help="Rotate decoded head-camera frames before writing the dataset",
    )
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
    if args.max_hand_command_age_ms <= 0.0:
        raise ValueError("--max_hand_command_age_ms must be positive")

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
            max_hand_command_age_ms=args.max_hand_command_age_ms,
            max_joint_age_ms=args.max_joint_age_ms,
            max_imu_age_ms=args.max_imu_age_ms,
            time_basis=args.time_basis,
            allow_partial_source_time=args.allow_partial_source_time,
        )
        converted.append(result)
        total_candidates += result.candidate_count
        total_samples += len(result.samples)
        print(
            f"[convert] {episode_dir.name}: accepted={len(result.samples)}/"
            f"{result.candidate_count}, time_basis={result.time_basis}, "
            f"skipped={result.skip_counts}, sync={result.synchronization_diagnostics}"
        )

    print(
        f"[convert] total accepted={total_samples}/{total_candidates} "
        f"({100.0 * total_samples / max(1, total_candidates):.1f}%)"
    )
    output_segments: list[tuple[EpisodeSamples, list[ConvertedSample]]] = []
    short_fragment_frames = 0
    for episode in converted:
        segments, dropped_short = split_contiguous_samples(
            episode.samples,
            fps=args.fps,
            min_frames=args.min_segment_frames,
        )
        short_fragment_frames += dropped_short
        output_segments.extend((episode, segment) for segment in segments)
    if not output_segments:
        raise RuntimeError(
            "All synchronized episodes are empty; inspect missing/stale counters above"
        )
    print(
        f"[convert] continuous segments={len(output_segments)}, "
        f"usable_frames={sum(len(segment) for _, segment in output_segments)}, "
        f"short_fragment_frames_dropped={short_fragment_frames}"
    )

    first_shape = _validate_segment_images(
        output_segments,
        camera_rotation_deg=args.camera_rotation_deg,
    )
    if args.dry_run:
        first_sample = output_segments[0][1][0]
        print(
            f"[convert] dry-run OK: image={first_shape}, "
            f"state={first_sample.state.shape}, action={first_sample.action.shape}, "
            f"timing={first_sample.timing_ms.shape}, time_basis={args.time_basis}"
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
        for episode, segment in output_segments:
            for sample in segment:
                rgb = _decode_rgb(
                    sample.image_path,
                    rotation_deg=args.camera_rotation_deg,
                )
                dataset.add_frame(
                    {
                        "observation.images.head": rgb,
                        "observation.state": sample.state.astype(np.float32, copy=False),
                        "sync.timing_ms": sample.timing_ms.astype(
                            np.float32, copy=False
                        ),
                        "action": sample.action.astype(np.float32, copy=False),
                        "task": episode.task,
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()

    conversion_report = _conversion_report_payload(
        raw_root=raw_root,
        output_root=output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        time_basis=args.time_basis,
        allow_partial_source_time=args.allow_partial_source_time,
        max_camera_age_ms=args.max_camera_age_ms,
        max_reference_age_ms=args.max_reference_age_ms,
        max_reference_gap_ms=args.max_reference_gap_ms,
        max_retarget_age_ms=args.max_retarget_age_ms,
        max_hand_command_age_ms=args.max_hand_command_age_ms,
        max_joint_age_ms=args.max_joint_age_ms,
        max_imu_age_ms=args.max_imu_age_ms,
        camera_rotation_deg=args.camera_rotation_deg,
        min_segment_frames=args.min_segment_frames,
        require_success=args.require_success,
        converted=converted,
        output_segments=output_segments,
        short_fragment_frames_dropped=short_fragment_frames,
    )
    _write_conversion_report(
        staging_root / "conversion_report.json", conversion_report
    )

    staging_root.replace(output_root)
    print(
        f"[convert] LeRobotDataset v3 written to {output_root}; "
        f"provenance={output_root / 'conversion_report.json'}"
    )


if __name__ == "__main__":
    main()
