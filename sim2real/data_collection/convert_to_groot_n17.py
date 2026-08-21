#!/usr/bin/env python3
"""Build a GR00T N1.7 X2 reference-prediction dataset.

The authoritative robot sample is the synchronous C++ tracking telemetry, not
the pre-alignment Python/GMR reference tap.  At each 25 Hz camera observation we
causally select the newest controller sample and accepted hand command that are
not newer than the image exposure time.

Rows store C++-consumed references expressed as absolute poses in each output
episode's first-reference frame. GR00T's modality configuration
converts the root and joint-reference action groups to relative actions during
training and restores absolute references during inference.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from .convert_to_lerobot import (
        _decode_rgb,
        _load_manifest,
        _load_timed_stream,
        _nearest_index,
        _series_time_ns,
    )
    from .schema import (
        GROOT_N17_ACTION_NAMES,
        GROOT_N17_STATE_NAMES,
        GROOT_N17_TIMING_NAMES,
        RAW_DATASET_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
        normalize_bridge_runtime_effective_params,
    )
    from .synchronization import TIME_BASIS_SOURCE, TimedStream
except ImportError:  # Direct execution from this directory.
    from convert_to_lerobot import (
        _decode_rgb,
        _load_manifest,
        _load_timed_stream,
        _nearest_index,
        _series_time_ns,
    )
    from schema import (
        GROOT_N17_ACTION_NAMES,
        GROOT_N17_STATE_NAMES,
        GROOT_N17_TIMING_NAMES,
        RAW_DATASET_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
        normalize_bridge_runtime_effective_params,
    )
    from synchronization import TIME_BASIS_SOURCE, TimedStream


CONVERSION_REPORT_SCHEMA_VERSION = "x2-groot-n17-conversion-v2"
EXPECTED_RECORD_PROFILE = "groot_n17"
EXPECTED_TELEMETRY_SCHEMA_VERSION = 1
REFERENCE_AGE_SPLIT_TOLERANCE_MS = 0.5
REQUIRED_CAPTURE_PROVENANCE = {
    "recorder",
    "controller_binary",
    "controller_config",
    "controller_policy",
    "controller_policy_data",
    "teleop_bridge",
    "gmr_config",
    "gmr_runtime",
    "hand_config",
}
EXPECTED_MODALITY_JSON = {
    "state": {
        "joint_position": {"start": 0, "end": 29},
        "joint_velocity": {"start": 29, "end": 58},
        "root_angular_velocity": {"start": 58, "end": 61},
        "projected_gravity": {"start": 61, "end": 64},
        "current_root_reference": {"start": 64, "end": 73},
        "current_joint_reference": {"start": 73, "end": 102},
        "current_grasp": {"start": 102, "end": 104},
    },
    "action": {
        "root_reference": {"start": 0, "end": 9},
        "joint_reference": {"start": 9, "end": 38},
        "grasp": {"start": 38, "end": 40},
    },
    "video": {"ego_view": {"original_key": "observation.images.head"}},
    "annotation": {
        "human.task_description": {"original_key": "task_index"}
    },
}


@dataclass
class GrootSample:
    timestamp_ns: int
    observation_timestamp_ns: int
    image_path: Path
    state: np.ndarray
    action: np.ndarray
    timing_ms: np.ndarray
    telemetry_sequence: int
    tracking_error_sq: float
    command_error_sq: float
    global_root_reference_xyz_rot6d: np.ndarray


@dataclass
class GrootEpisode:
    episode_dir: Path
    task: str
    samples: list[GrootSample]
    candidate_count: int
    skip_counts: Dict[str, int]
    synchronization_diagnostics: Dict[str, Dict[str, Any]]
    telemetry_sequence_gaps: int
    reference_age_diagnostics: Dict[str, Any]


def _validate_modality_json(path: Path) -> str:
    """Require the exact 104/40 X2 modality contract and return its digest."""

    if not path.is_file():
        raise FileNotFoundError(f"modality.json not found: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid modality.json {path}: {exc}") from exc
    if payload != EXPECTED_MODALITY_JSON:
        raise ValueError(
            f"{path} does not match the exact X2 N1.7 104-state/40-action contract"
        )
    return hashlib.sha256(raw).hexdigest()


def _capture_contract(manifest: Dict[str, Any], episode_dir: Path) -> Dict[str, Any]:
    source_config = manifest.get("source_config", {})
    if not isinstance(source_config, dict):
        raise ValueError(f"{episode_dir} has no source_config mapping")
    provenance = source_config.get("capture_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{episode_dir} has no capture_provenance mapping")
    missing = sorted(REQUIRED_CAPTURE_PROVENANCE.difference(provenance))
    if missing:
        raise ValueError(
            f"{episode_dir} is missing capture provenance: {', '.join(missing)}"
        )
    normalized_provenance: Dict[str, Dict[str, Any]] = {}
    for name, record in sorted(provenance.items()):
        if not isinstance(record, dict):
            raise ValueError(f"{episode_dir} provenance {name!r} is invalid")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"{episode_dir} provenance {name!r} is invalid")
        # Paths are retained in each raw manifest for diagnosis but excluded
        # from compatibility identity so the same bits can be copied between
        # robot and workstation without appearing incompatible.
        normalized_provenance[name] = {"sha256": digest, "size_bytes": size}
    try:
        bridge_runtime_effective_params = normalize_bridge_runtime_effective_params(
            source_config.get("bridge_runtime_effective_params")
        )
    except ValueError as exc:
        raise ValueError(
            f"{episode_dir} has invalid bridge runtime effective params: {exc}"
        ) from exc
    return {
        "joint_order": manifest.get("joint_order"),
        "tracking_telemetry_schema_version": source_config.get(
            "tracking_telemetry_schema_version"
        ),
        "tracking_telemetry_delivery_semantics": source_config.get(
            "tracking_telemetry_delivery_semantics"
        ),
        "tracking_telemetry_reference_age_semantics": source_config.get(
            "tracking_telemetry_reference_age_semantics"
        ),
        "hand_status_delivery_semantics": source_config.get(
            "hand_status_delivery_semantics"
        ),
        "head_joint_assumption": source_config.get("head_joint_assumption"),
        "bridge_runtime_effective_params": bridge_runtime_effective_params,
        "capture_provenance": normalized_provenance,
    }


def _strict_vector(event: Dict[str, Any], key: str, size: int) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(event.get(key), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _quat_wxyz_to_rot6d(quaternion: np.ndarray) -> Optional[np.ndarray]:
    """Match GR00T EndEffectorPose: first two rotation-matrix rows."""

    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        return None
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation[:2, :].reshape(6)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of GR00T's row-major rotation-6D representation."""

    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row0_norm = float(np.linalg.norm(rows[0]))
    if not np.isfinite(row0_norm) or row0_norm < 1e-8:
        raise ValueError("invalid first rotation-6D row")
    row0 = rows[0] / row0_norm
    row1 = rows[1] - float(np.dot(row0, rows[1])) * row0
    row1_norm = float(np.linalg.norm(row1))
    if not np.isfinite(row1_norm) or row1_norm < 1e-8:
        raise ValueError("degenerate rotation-6D rows")
    row1 /= row1_norm
    row2 = np.cross(row0, row1)
    return np.vstack([row0, row1, row2])


def localize_segment_root_references(samples: Sequence[GrootSample]) -> None:
    """Express root references in the output episode's first-reference frame.

    This is a rigid coordinate-frame change, not an action delta. Both state
    and action still contain absolute poses within the episode-local frame, so
    GR00T applies its RELATIVE transform exactly once against the current local
    reference. Raw telemetry remains unchanged and globally reconstructible.
    """

    if not samples:
        return
    anchor_pose = np.asarray(samples[0].action[:9], dtype=np.float64)
    anchor_position = anchor_pose[:3]
    anchor_rotation = _rot6d_to_matrix(anchor_pose[3:])
    for sample in samples:
        pose = np.asarray(sample.action[:9], dtype=np.float64)
        rotation = _rot6d_to_matrix(pose[3:])
        local_position = anchor_rotation.T @ (pose[:3] - anchor_position)
        local_rotation = anchor_rotation.T @ rotation
        local_pose = np.concatenate([local_position, local_rotation[:2, :].reshape(6)]).astype(
            np.float32
        )
        sample.action[:9] = local_pose
        sample.state[64:73] = local_pose


class SequenceAwarePreviousSeries:
    """Causally sample a sequenced latest-state stream without hiding loss.

    A received event is authoritative at its own timestamp. If the next
    received event proves that one or more publications were lost (or that the
    publisher restarted), timestamps strictly between those two events are
    rejected: zero-order-holding the older value there would silently attach a
    stale robot state/action to the camera observation.
    """

    def __init__(
        self,
        events: Sequence[Dict[str, Any]],
        *,
        sequence_modulus: int,
        reset_field: Optional[str] = None,
    ) -> None:
        if sequence_modulus <= 1:
            raise ValueError("sequence_modulus must be greater than one")
        self.events = list(events)
        self.times = [_series_time_ns(event) for event in self.events]
        self.sequence_modulus = int(sequence_modulus)
        self.reset_field = reset_field

    def _sequence(self, event: Dict[str, Any]) -> Optional[int]:
        value = event.get("sequence")
        if isinstance(value, bool):
            return None
        try:
            sequence = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if isinstance(value, float) and not value.is_integer():
            return None
        if sequence < 0 or sequence >= self.sequence_modulus:
            return None
        return sequence

    def previous(
        self, target_ns: int
    ) -> tuple[Optional[Dict[str, Any]], float, str]:
        index = bisect.bisect_right(self.times, target_ns) - 1
        if index < 0:
            return None, math.inf, "missing"

        event = self.events[index]
        age_ms = (target_ns - self.times[index]) / 1e6
        current_sequence = self._sequence(event)
        if current_sequence is None:
            return None, age_ms, "sequence_invalid"

        next_index = index + 1
        if next_index >= len(self.events):
            return event, age_ms, "ok"
        current_time_ns = self.times[index]
        next_time_ns = self.times[next_index]
        if not current_time_ns < target_ns < next_time_ns:
            # Do not reject either received endpoint. At the next event's own
            # time, bisect selects that new authoritative state.
            return event, age_ms, "ok"

        next_event = self.events[next_index]
        next_sequence = self._sequence(next_event)
        if next_sequence is None:
            return None, age_ms, "sequence_invalid"

        sequence_delta = (next_sequence - current_sequence) % self.sequence_modulus
        true_wrap = (
            current_sequence == self.sequence_modulus - 1 and next_sequence == 0
        )
        reset_marked = (
            self.reset_field is not None and next_event.get(self.reset_field) is True
        )
        if not true_wrap and (reset_marked or next_sequence < current_sequence):
            return None, age_ms, "sequence_reset"
        if sequence_delta != 1:
            return None, age_ms, "sequence_gap"
        return event, age_ms, "ok"


def _telemetry_payload(event: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    try:
        schema_version = int(event.get("schema_version", -1))
        joint_count = int(event.get("joint_count", -1))
    except (TypeError, ValueError, OverflowError):
        return None
    if schema_version != EXPECTED_TELEMETRY_SCHEMA_VERSION:
        return None
    if joint_count != len(X2_TRACKING_JOINT_NAMES):
        return None
    if event.get("joint_names") != X2_TRACKING_JOINT_NAMES:
        return None
    root_position = _strict_vector(event, "reference_root_position", 3)
    root_quaternion = _strict_vector(event, "reference_root_quaternion_wxyz", 4)
    root_rot6d = None if root_quaternion is None else _quat_wxyz_to_rot6d(root_quaternion)
    values = {
        "root_position": root_position,
        "root_rot6d": root_rot6d,
        "reference_joint": _strict_vector(event, "reference_joint_position", 29),
        "measured_joint": _strict_vector(event, "measured_joint_position", 29),
        "measured_velocity": _strict_vector(event, "measured_joint_velocity", 29),
        "root_angular_velocity": _strict_vector(event, "root_angular_velocity", 3),
        "projected_gravity": _strict_vector(event, "projected_gravity", 3),
        "policy_action": _strict_vector(event, "policy_action", 29),
        "command_joint": _strict_vector(event, "command_joint_position", 29),
    }
    if any(value is None for value in values.values()):
        return None
    return {key: value for key, value in values.items() if value is not None}


def _nonnegative_finite_age(event: Dict[str, Any], key: str) -> Optional[float]:
    """Return one optional age without accepting bools, NaN or negatives."""

    value = event.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        age_ms = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(age_ms) or age_ms < 0.0:
        return None
    return age_ms


def _reference_age_components(
    event: Dict[str, Any],
) -> tuple[Optional[Dict[str, float]], str]:
    """Resolve the additive schema-v1 reference-age split.

    ``reference_source_age_ms`` is retained on the wire as the legacy total
    source-to-policy age.  It must never be used as the 80 ms upstream/GMR
    freshness gate: the controller intentionally consumes a frame several
    future-buffer steps after the bridge selected it.  New telemetry exposes
    both sides of the bridge-reply boundary so those meanings stay separate.
    """

    legacy_total_ms = _nonnegative_finite_age(event, "reference_source_age_ms")
    explicit_total_ms = _nonnegative_finite_age(event, "reference_total_age_ms")
    upstream_ms = _nonnegative_finite_age(
        event, "reference_upstream_age_at_bridge_ms"
    )
    bridge_to_policy_ms = _nonnegative_finite_age(
        event, "reference_bridge_to_policy_age_ms"
    )
    if legacy_total_ms is None and explicit_total_ms is None:
        return None, "reference_total_age_missing"
    if upstream_ms is None or bridge_to_policy_ms is None:
        if legacy_total_ms is not None and explicit_total_ms is None:
            # This is the expected signature of an old raw schema-v1 event.
            return None, "reference_upstream_age_missing_legacy"
        return None, "reference_age_split_missing"
    if legacy_total_ms is None or explicit_total_ms is None:
        return None, "reference_total_age_alias_missing"
    if (
        abs(legacy_total_ms - explicit_total_ms)
        > REFERENCE_AGE_SPLIT_TOLERANCE_MS
    ):
        return None, "reference_total_age_alias_mismatch"

    total_ms = explicit_total_ms
    if (
        abs(total_ms - upstream_ms - bridge_to_policy_ms)
        > REFERENCE_AGE_SPLIT_TOLERANCE_MS
    ):
        return None, "reference_age_split_inconsistent"
    return {
        "upstream_ms": upstream_ms,
        "total_ms": total_ms,
        "bridge_to_policy_ms": bridge_to_policy_ms,
    }, "ok"


def _numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _reference_age_diagnostics(
    events: Sequence[Dict[str, Any]], *, max_upstream_age_ms: float
) -> Dict[str, Any]:
    legacy_total: list[float] = []
    explicit_total: list[float] = []
    upstream: list[float] = []
    bridge_to_policy: list[float] = []
    split_residual: list[float] = []
    complete_split_count = 0
    legacy_only_count = 0
    upstream_over_limit_count = 0
    for event in events:
        legacy = _nonnegative_finite_age(event, "reference_source_age_ms")
        total = _nonnegative_finite_age(event, "reference_total_age_ms")
        upstream_age = _nonnegative_finite_age(
            event, "reference_upstream_age_at_bridge_ms"
        )
        queue_age = _nonnegative_finite_age(
            event, "reference_bridge_to_policy_age_ms"
        )
        if legacy is not None:
            legacy_total.append(legacy)
        if total is not None:
            explicit_total.append(total)
        if upstream_age is not None:
            upstream.append(upstream_age)
            if upstream_age > max_upstream_age_ms:
                upstream_over_limit_count += 1
        if queue_age is not None:
            bridge_to_policy.append(queue_age)
        if legacy is not None and upstream_age is None:
            legacy_only_count += 1
        resolved_total = total if total is not None else legacy
        if (
            resolved_total is not None
            and upstream_age is not None
            and queue_age is not None
        ):
            complete_split_count += 1
            split_residual.append(resolved_total - upstream_age - queue_age)
    return {
        "input_events": len(events),
        "complete_split_events": complete_split_count,
        "legacy_total_only_events": legacy_only_count,
        "upstream_over_limit_events": upstream_over_limit_count,
        "upstream_limit_ms": float(max_upstream_age_ms),
        "legacy_total_consumed_age_ms": _numeric_summary(legacy_total),
        "explicit_total_consumed_age_ms": _numeric_summary(explicit_total),
        "upstream_age_at_bridge_ms": _numeric_summary(upstream),
        "bridge_to_policy_age_ms": _numeric_summary(bridge_to_policy),
        "split_residual_ms": _numeric_summary(split_residual),
    }


def _hand_payload(event: Dict[str, Any]) -> Optional[np.ndarray]:
    if event.get("active") is not True:
        return None
    try:
        values = np.asarray(
            [float(event["left_grasp"]), float(event["right_grasp"])],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        return None
    return values


def _sync_diagnostics(streams: Dict[str, TimedStream]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "input_events": stream.input_count,
            "timed_events": len(stream.events),
            "missing_time_events": stream.missing_time_count,
            "source_regressions": stream.source_regression_count,
            "large_source_regressions": stream.large_source_regression_count,
            "max_source_regression_ms": stream.max_source_regression_ms,
        }
        for name, stream in sorted(streams.items())
    }


def _telemetry_gap_count(events: Sequence[Dict[str, Any]]) -> int:
    sequences: list[int] = []
    for event in events:
        try:
            sequence = int(event["sequence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if sequence >= 0:
            sequences.append(sequence)
    return sum(max(0, current - previous - 1) for previous, current in zip(sequences, sequences[1:]))


def _validate_manifest(manifest: Dict[str, Any], episode_dir: Path) -> None:
    if manifest.get("schema_version") != RAW_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported raw schema in {episode_dir}")
    if manifest.get("robot_type") != "agibot_x2":
        raise ValueError(f"Unexpected robot type in {episode_dir}")
    if manifest.get("joint_order") != X2_TRACKING_JOINT_NAMES:
        raise ValueError(f"Joint order mismatch in {episode_dir}")
    if not str(manifest.get("task", "")).strip():
        raise ValueError(f"Empty task in {episode_dir}")
    source_config = manifest.get("source_config", {})
    if source_config.get("record_profile") != EXPECTED_RECORD_PROFILE:
        raise ValueError(
            f"{episode_dir} was not recorded with --record_profile {EXPECTED_RECORD_PROFILE}"
        )
    if source_config.get("hand_status_delivery_semantics") != "latest_state":
        raise ValueError(
            f"{episode_dir} does not declare latest-state hand delivery semantics"
        )
    if source_config.get("head_joint_assumption") != "fixed_not_recorded":
        raise ValueError(
            f"{episode_dir} does not declare the fixed-head camera assumption"
        )
    if (
        source_config.get("tracking_telemetry_schema_version")
        != EXPECTED_TELEMETRY_SCHEMA_VERSION
    ):
        raise ValueError(f"{episode_dir} has the wrong tracking telemetry schema")
    if (
        source_config.get("tracking_telemetry_delivery_semantics")
        != "bounded_nonblocking_sequence_checked"
    ):
        raise ValueError(f"{episode_dir} has incompatible telemetry delivery semantics")
    _capture_contract(manifest, episode_dir)


def _discover_groot_episodes(raw_root: Path, *, require_success: bool) -> list[Path]:
    """Select only complete episodes recorded with the N1.7 profile.

    A raw root can intentionally contain older ``vla`` or diagnostic episodes;
    they must not make a GR00T conversion batch fail halfway through.
    """

    episodes: list[Path] = []
    for episode_dir in sorted(raw_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        manifest = _load_manifest(episode_dir)
        if manifest.get("status") != "complete":
            continue
        if manifest.get("source_config", {}).get("record_profile") != EXPECTED_RECORD_PROFILE:
            continue
        if require_success and manifest.get("success") is not True:
            continue
        episodes.append(episode_dir)
    return episodes


def build_episode_samples(
    episode_dir: Path,
    *,
    fps: int = 25,
    max_camera_grid_offset_ms: float = 30.0,
    max_tracking_age_ms: float = 50.0,
    max_hand_age_ms: float = 40.0,
    max_reference_upstream_age_ms: float = 80.0,
) -> GrootEpisode:
    manifest = _load_manifest(episode_dir)
    _validate_manifest(manifest, episode_dir)
    recording = manifest.get("recording", {})
    start_ns = int(recording["start_monotonic_ns"])
    stop_ns = int(recording["stop_trigger_monotonic_ns"])
    if stop_ns <= start_ns:
        raise ValueError(f"Invalid A/X interval in {episode_dir}")

    streams = {
        name: _load_timed_stream(episode_dir, name, time_basis=TIME_BASIS_SOURCE)
        for name in ("camera_head", "tracking_telemetry", "hand_command")
    }
    incomplete = {
        name: (stream.missing_time_count, stream.input_count)
        for name, stream in streams.items()
        if stream.missing_time_count > 0
    }
    if incomplete:
        raise ValueError(f"Source-time mapping failed in {episode_dir}: {incomplete}")
    regressed = {
        name: (stream.large_source_regression_count, stream.max_source_regression_ms)
        for name, stream in streams.items()
        if stream.large_source_regression_count > 0
    }
    if regressed:
        raise ValueError(f"Large source-time regression in {episode_dir}: {regressed}")

    camera_events = [
        event
        for event in streams["camera_head"].events
        if start_ns <= _series_time_ns(event) <= stop_ns
    ]
    camera_times = [_series_time_ns(event) for event in camera_events]
    active_telemetry_events = [
        event
        for event in streams["tracking_telemetry"].events
        if start_ns <= _series_time_ns(event) <= stop_ns
    ]
    reference_age_diagnostics = _reference_age_diagnostics(
        active_telemetry_events,
        max_upstream_age_ms=max_reference_upstream_age_ms,
    )
    telemetry = SequenceAwarePreviousSeries(
        streams["tracking_telemetry"].events,
        sequence_modulus=1 << 64,
        reset_field="tracking_telemetry_sequence_reset",
    )
    hands = SequenceAwarePreviousSeries(
        streams["hand_command"].events,
        sequence_modulus=1 << 32,
    )

    step_ns = int(round(1e9 / float(fps)))
    target_times = list(range(start_ns, stop_ns + 1, step_ns))
    samples: list[GrootSample] = []
    skips: Counter[str] = Counter()
    previous_camera_index: Optional[int] = None
    previous_camera_time: Optional[int] = None

    for target_ns in target_times:
        camera_index = _nearest_index(camera_times, target_ns)
        if camera_index is None:
            skips["camera_missing"] += 1
            continue
        camera_time = camera_times[camera_index]
        camera_offset_ms = (camera_time - target_ns) / 1e6
        if abs(camera_offset_ms) > max_camera_grid_offset_ms:
            skips["camera_grid_offset"] += 1
            continue
        if camera_index == previous_camera_index:
            skips["camera_reused"] += 1
            continue
        if previous_camera_time is not None and camera_time <= previous_camera_time:
            skips["camera_non_monotonic"] += 1
            continue
        previous_camera_index = camera_index
        previous_camera_time = camera_time

        telemetry_event, tracking_age_ms, tracking_status = telemetry.previous(
            camera_time
        )
        if tracking_status != "ok":
            skips[f"tracking_{tracking_status}"] += 1
            continue
        assert telemetry_event is not None
        if tracking_age_ms > max_tracking_age_ms:
            skips["tracking_stale"] += 1
            continue
        parsed = _telemetry_payload(telemetry_event)
        if parsed is None:
            skips["tracking_invalid"] += 1
            continue
        if telemetry_event.get("vr_session_active") is not True:
            skips["tracking_inactive"] += 1
            continue
        provenance_flags = (
            telemetry_event.get("reference_is_transition"),
            telemetry_event.get("reference_is_padded"),
            telemetry_event.get("reference_is_fallback"),
        )
        if not all(isinstance(value, bool) for value in provenance_flags):
            skips["reference_provenance_missing"] += 1
            continue
        if telemetry_event.get("reference_source_time_exact") is not True:
            # Production labels require the bridge's absolute robot-local
            # CLOCK_MONOTONIC source stamp. Age-only compatibility telemetry
            # omits bridge->controller queueing and is diagnostics-only.
            skips["reference_source_time_inexact"] += 1
            continue
        if telemetry_event["reference_is_transition"]:
            # The controller-generated A/start blend is safe to execute but is
            # not an operator/VLA action label.
            skips["reference_transition"] += 1
            continue
        if telemetry_event["reference_is_padded"]:
            # This is controller-generated future/low-watermark padding, not
            # an operator reference.  A fresh bridge fallback is represented
            # separately and remains eligible under the source-age gate.
            skips["reference_padded"] += 1
            continue
        reference_ages, reference_age_status = _reference_age_components(
            telemetry_event
        )
        if reference_age_status != "ok":
            skips[reference_age_status] += 1
            continue
        assert reference_ages is not None
        reference_upstream_age_ms = reference_ages["upstream_ms"]
        reference_total_age_ms = reference_ages["total_ms"]
        reference_bridge_to_policy_age_ms = reference_ages[
            "bridge_to_policy_ms"
        ]
        if reference_upstream_age_ms > max_reference_upstream_age_ms:
            skips["reference_upstream_stale"] += 1
            continue

        hand_event, hand_age_ms, hand_status = hands.previous(camera_time)
        if hand_status != "ok":
            skips[f"hand_{hand_status}"] += 1
            continue
        assert hand_event is not None
        if hand_age_ms > max_hand_age_ms:
            skips["hand_stale"] += 1
            continue
        hand = _hand_payload(hand_event)
        if hand is None:
            skips["hand_inactive_or_invalid"] += 1
            continue

        root_reference = np.concatenate([parsed["root_position"], parsed["root_rot6d"]])
        state = np.concatenate(
            [
                parsed["measured_joint"],
                parsed["measured_velocity"],
                parsed["root_angular_velocity"],
                parsed["projected_gravity"],
                root_reference,
                parsed["reference_joint"],
                hand,
            ]
        ).astype(np.float32)
        action = np.concatenate(
            [root_reference, parsed["reference_joint"], hand]
        ).astype(np.float32)
        if state.shape != (len(GROOT_N17_STATE_NAMES),) or action.shape != (
            len(GROOT_N17_ACTION_NAMES),
        ):
            skips["shape_invalid"] += 1
            continue
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            skips["non_finite"] += 1
            continue

        relative_image_path = camera_events[camera_index].get("image_path")
        if not isinstance(relative_image_path, str):
            skips["camera_path_missing"] += 1
            continue
        image_path = episode_dir / relative_image_path
        if not image_path.is_file():
            skips["camera_file_missing"] += 1
            continue
        try:
            sequence = int(telemetry_event["sequence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            skips["tracking_sequence_invalid"] += 1
            continue

        tracking_error = parsed["reference_joint"] - parsed["measured_joint"]
        command_error = parsed["command_joint"] - parsed["measured_joint"]
        samples.append(
            GrootSample(
                timestamp_ns=target_ns,
                observation_timestamp_ns=camera_time,
                image_path=image_path,
                state=state,
                action=action,
                timing_ms=np.asarray(
                    [
                        camera_offset_ms,
                        tracking_age_ms,
                        hand_age_ms,
                        reference_total_age_ms,
                        reference_upstream_age_ms,
                        reference_bridge_to_policy_age_ms,
                    ],
                    dtype=np.float32,
                ),
                telemetry_sequence=sequence,
                tracking_error_sq=float(np.mean(np.square(tracking_error))),
                command_error_sq=float(np.mean(np.square(command_error))),
                global_root_reference_xyz_rot6d=root_reference.astype(
                    np.float32, copy=True
                ),
            )
        )

    return GrootEpisode(
        episode_dir=episode_dir,
        task=str(manifest["task"]),
        samples=samples,
        candidate_count=len(target_times),
        skip_counts=dict(skips),
        synchronization_diagnostics=_sync_diagnostics(streams),
        telemetry_sequence_gaps=_telemetry_gap_count(streams["tracking_telemetry"].events),
        reference_age_diagnostics=reference_age_diagnostics,
    )


def split_contiguous_samples(
    samples: Sequence[GrootSample], *, fps: int, min_frames: int
) -> tuple[list[list[GrootSample]], int]:
    if not samples:
        return [], 0
    step_ns = int(round(1e9 / float(fps)))
    raw_segments: list[list[GrootSample]] = [[samples[0]]]
    for sample in samples[1:]:
        if sample.timestamp_ns - raw_segments[-1][-1].timestamp_ns == step_ns:
            raw_segments[-1].append(sample)
        else:
            raw_segments.append([sample])
    kept = [segment for segment in raw_segments if len(segment) >= min_frames]
    dropped = sum(len(segment) for segment in raw_segments if len(segment) < min_frames)
    return kept, dropped


def _validate_images(
    segments: Sequence[tuple[GrootEpisode, Sequence[GrootSample]]], rotation_deg: int
) -> tuple[int, int, int]:
    shape: Optional[tuple[int, int, int]] = None
    seen: set[Path] = set()
    for _episode, samples in segments:
        for sample in samples:
            if sample.image_path in seen:
                continue
            current = tuple(_decode_rgb(sample.image_path, rotation_deg=rotation_deg).shape)
            if shape is None:
                shape = current
            elif current != shape:
                raise ValueError(f"Camera shape changed from {shape} to {current}")
            seen.add(sample.image_path)
    if shape is None:
        raise RuntimeError("No usable camera images")
    return shape


def _create_dataset(repo_id: str, root: Path, fps: int, image_shape: tuple[int, int, int]) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot is not installed. Use --dry_run in the robot/GMR environment, "
            "then run the real conversion in the pinned LeRobot environment."
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
            "shape": (len(GROOT_N17_STATE_NAMES),),
            "names": GROOT_N17_STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(GROOT_N17_ACTION_NAMES),),
            "names": GROOT_N17_ACTION_NAMES,
        },
        "sync.timing_ms": {
            "dtype": "float32",
            "shape": (len(GROOT_N17_TIMING_NAMES),),
            "names": GROOT_N17_TIMING_NAMES,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="agibot_x2",
        features=features,
        use_videos=True,
    )


def _report(
    *,
    raw_root: Path,
    output_root: Path,
    fps: int,
    args: argparse.Namespace,
    episodes: Sequence[GrootEpisode],
    segments: Sequence[tuple[GrootEpisode, Sequence[GrootSample]]],
    short_frames: int,
    modality_json: Path,
    modality_sha256: str,
    capture_contract: Dict[str, Any],
    capture_contract_sha256: str,
) -> Dict[str, Any]:
    converter_path = Path(__file__).resolve()
    converter_sha256 = hashlib.sha256(converter_path.read_bytes()).hexdigest()
    segment_records = []
    for index, (episode, samples) in enumerate(segments):
        segment_records.append(
            {
                "dataset_episode_index": index,
                "raw_episode": episode.episode_dir.name,
                "frame_count": len(samples),
                "first_grid_timestamp_ns": samples[0].timestamp_ns,
                "last_grid_timestamp_ns": samples[-1].timestamp_ns,
                "first_camera_timestamp_ns": samples[0].observation_timestamp_ns,
                "last_camera_timestamp_ns": samples[-1].observation_timestamp_ns,
                "first_telemetry_sequence": samples[0].telemetry_sequence,
                "last_telemetry_sequence": samples[-1].telemetry_sequence,
                "root_frame_anchor_global_xyz_rot6d": samples[
                    0
                ].global_root_reference_xyz_rot6d.tolist(),
            }
        )
    all_samples = [sample for _episode, samples in segments for sample in samples]
    raw_episode_records = []
    for episode in episodes:
        manifest = _load_manifest(episode.episode_dir)
        raw_episode_records.append(
            {
                "raw_episode": episode.episode_dir.name,
                "task": episode.task,
                "candidate_ticks": episode.candidate_count,
                "accepted_ticks": len(episode.samples),
                "skip_counts": episode.skip_counts,
                "telemetry_sequence_gaps": episode.telemetry_sequence_gaps,
                "synchronization_diagnostics": episode.synchronization_diagnostics,
                "reference_age_diagnostics": episode.reference_age_diagnostics,
                "manifest_status": manifest.get("status"),
                "success": manifest.get("success"),
                "validation": manifest.get("validation"),
                "ingress_drops": manifest.get("ingress_drops", {}),
                "source_config": manifest.get("source_config", {}),
            }
        )
    return {
        "schema_version": CONVERSION_REPORT_SCHEMA_VERSION,
        "raw_dataset_schema_version": RAW_DATASET_SCHEMA_VERSION,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "fps": fps,
        "converter": {
            "path": str(converter_path),
            "sha256": converter_sha256,
        },
        "modality": {
            "path": str(modality_json),
            "sha256": modality_sha256,
            "contract": "x2_n17_state104_action40",
        },
        "capture_contract": capture_contract,
        "capture_contract_sha256": capture_contract_sha256,
        "representation": {
            "state_dim": len(GROOT_N17_STATE_NAMES),
            "action_dim": len(GROOT_N17_ACTION_NAMES),
            "action_horizon": 40,
            "root_rotation": "xyz+rot6d (first two matrix rows)",
            "stored_action": "absolute consumed reference in episode-local frame",
            "root_coordinate_frame": "output-segment first consumed reference",
            "training_transform": "GR00T relative root/joint; absolute grasp",
            "camera_rotation_deg": args.camera_rotation_deg,
            "minimum_segment_frames": args.min_segment_frames,
            "require_success": bool(args.require_success),
            "timing_names": GROOT_N17_TIMING_NAMES,
        },
        "thresholds_ms": {
            "camera_grid_offset": args.max_camera_grid_offset_ms,
            "tracking_telemetry": args.max_tracking_age_ms,
            "hand_command": args.max_hand_age_ms,
            "reference_upstream_age_at_bridge": (
                args.max_reference_upstream_age_ms
            ),
        },
        "summary": {
            "candidate_ticks": sum(episode.candidate_count for episode in episodes),
            "accepted_before_segmentation": sum(len(episode.samples) for episode in episodes),
            "output_frames": len(all_samples),
            "output_segments": len(segments),
            "short_fragment_frames_dropped": short_frames,
            "tracking_reference_rmse_rad": math.sqrt(
                float(np.mean([sample.tracking_error_sq for sample in all_samples]))
            ),
            "command_tracking_rmse_rad": math.sqrt(
                float(np.mean([sample.command_error_sq for sample in all_samples]))
            ),
            "accepted_reference_total_consumed_age_ms": _numeric_summary(
                [float(sample.timing_ms[3]) for sample in all_samples]
            ),
            "accepted_reference_upstream_age_at_bridge_ms": _numeric_summary(
                [float(sample.timing_ms[4]) for sample in all_samples]
            ),
            "accepted_reference_bridge_to_policy_age_ms": _numeric_summary(
                [float(sample.timing_ms[5]) for sample in all_samples]
            ),
        },
        "raw_episodes": raw_episode_records,
        "output_segments": segment_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", default="~/Datasets/x2_vr/raw")
    parser.add_argument("--output_root", default="~/Datasets/x2_vr/groot_n17_lerobot_v3")
    parser.add_argument("--repo_id", default="local/x2_groot_n17")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max_camera_grid_offset_ms", type=float, default=30.0)
    parser.add_argument("--max_tracking_age_ms", type=float, default=50.0)
    parser.add_argument("--max_hand_age_ms", type=float, default=40.0)
    parser.add_argument(
        "--max_reference_upstream_age_ms",
        type=float,
        default=80.0,
        help=(
            "Maximum selected-reference age at the bridge reply boundary; "
            "total controller-consumption age is diagnostic only."
        ),
    )
    parser.add_argument("--min_segment_frames", type=int, default=40)
    parser.add_argument("--camera_rotation_deg", type=int, choices=(0, 180), default=180)
    parser.add_argument("--require_success", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--modality_json",
        default="~/Documents/Isaac-GR00T/examples/X2/modality.json",
        help="N1.7 modality mapping copied into meta/modality.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.min_segment_frames < 40:
        raise ValueError("fps must be positive and min_segment_frames must be >= 40")
    if min(
        args.max_camera_grid_offset_ms,
        args.max_tracking_age_ms,
        args.max_hand_age_ms,
        args.max_reference_upstream_age_ms,
    ) <= 0.0:
        raise ValueError("all synchronization thresholds must be positive")

    raw_root = Path(args.raw_root).expanduser().resolve()
    episode_dirs = _discover_groot_episodes(
        raw_root, require_success=args.require_success
    )
    if not episode_dirs:
        raise RuntimeError("No eligible complete episodes found")

    contracts = [
        _capture_contract(_load_manifest(path), path) for path in episode_dirs
    ]
    canonical_contract_json = json.dumps(
        contracts[0], sort_keys=True, separators=(",", ":")
    )
    for path, contract in zip(episode_dirs[1:], contracts[1:]):
        if json.dumps(contract, sort_keys=True, separators=(",", ":")) != canonical_contract_json:
            raise ValueError(
                f"Capture contract mismatch in {path}; do not mix runtime/config revisions"
            )
    capture_contract_sha256 = hashlib.sha256(
        canonical_contract_json.encode("utf-8")
    ).hexdigest()

    converted = [
        build_episode_samples(
            episode_dir,
            fps=args.fps,
            max_camera_grid_offset_ms=args.max_camera_grid_offset_ms,
            max_tracking_age_ms=args.max_tracking_age_ms,
            max_hand_age_ms=args.max_hand_age_ms,
            max_reference_upstream_age_ms=args.max_reference_upstream_age_ms,
        )
        for episode_dir in episode_dirs
    ]
    for episode in converted:
        print(
            f"[groot] {episode.episode_dir.name}: accepted={len(episode.samples)}/"
            f"{episode.candidate_count}, skipped={episode.skip_counts}, "
            f"telemetry_sequence_gaps={episode.telemetry_sequence_gaps}, "
            f"reference_age={episode.reference_age_diagnostics}"
        )

    output_segments: list[tuple[GrootEpisode, list[GrootSample]]] = []
    short_frames = 0
    for episode in converted:
        segments, dropped = split_contiguous_samples(
            episode.samples, fps=args.fps, min_frames=args.min_segment_frames
        )
        output_segments.extend((episode, segment) for segment in segments)
        for segment in segments:
            localize_segment_root_references(segment)
        short_frames += dropped
    if not output_segments:
        if any(
            episode.skip_counts.get("reference_upstream_age_missing_legacy", 0)
            for episode in converted
        ):
            raise RuntimeError(
                "No production segment: this raw telemetry has only the legacy "
                "total consumed-reference age. It remains readable for the "
                "printed diagnostics, but requires a controller build that "
                "records reference_upstream_age_at_bridge_ms and "
                "reference_bridge_to_policy_age_ms before training conversion."
            )
        raise RuntimeError("No continuous segment is long enough for the 40-step action horizon")

    image_shape = _validate_images(output_segments, args.camera_rotation_deg)
    modality_json = Path(args.modality_json).expanduser().resolve()
    modality_sha256 = _validate_modality_json(modality_json)
    output_frames = sum(len(samples) for _episode, samples in output_segments)
    print(
        f"[groot] segments={len(output_segments)}, frames={output_frames}, "
        f"short_frames_dropped={short_frames}, image={image_shape}, state=104, action=40"
    )
    if args.dry_run:
        return

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if output_root == raw_root or output_root in raw_root.parents or raw_root in output_root.parents:
        raise ValueError("raw and output roots must not overlap")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.partial-{uuid.uuid4().hex}")
    dataset = _create_dataset(args.repo_id, staging, args.fps, image_shape)
    try:
        for episode, samples in output_segments:
            for sample in samples:
                dataset.add_frame(
                    {
                        "observation.images.head": _decode_rgb(
                            sample.image_path, rotation_deg=args.camera_rotation_deg
                        ),
                        "observation.state": sample.state,
                        "action": sample.action,
                        "sync.timing_ms": sample.timing_ms,
                        "task": episode.task,
                    }
                )
            dataset.save_episode()
        dataset.finalize()

        meta_dir = staging / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(modality_json, meta_dir / "modality.json")
        report = _report(
            raw_root=raw_root,
            output_root=output_root,
            fps=args.fps,
            args=args,
            episodes=converted,
            segments=output_segments,
            short_frames=short_frames,
            modality_json=modality_json,
            modality_sha256=modality_sha256,
            capture_contract=contracts[0],
            capture_contract_sha256=capture_contract_sha256,
        )
        with (staging / "conversion_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        staging.replace(output_root)
    except BaseException:
        # Preserve staging for diagnosis; never expose it under the final path.
        raise
    print(f"[groot] dataset written to {output_root}")


if __name__ == "__main__":
    main()
