"""Validation helpers for the X2 controller tracking-telemetry tap.

The controller publishes this stream from a background worker so recording can
observe the exact state/reference pair consumed by the policy without adding
ROS subscriptions or blocking the 50 Hz control loop.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Sequence

try:
    from .schema import X2_TRACKING_JOINT_NAMES
except ImportError:  # Direct execution from this directory.
    from schema import X2_TRACKING_JOINT_NAMES


TRACKING_TELEMETRY_TOPIC = "tracking_telemetry"
TRACKING_TELEMETRY_SCHEMA_VERSION = 1
TRACKING_TELEMETRY_JOINT_COUNT = len(X2_TRACKING_JOINT_NAMES)
UINT64_MAX = (1 << 64) - 1

_VECTOR_DIMS = {
    "reference_root_position": 3,
    "reference_root_quaternion_wxyz": 4,
    "reference_joint_position": TRACKING_TELEMETRY_JOINT_COUNT,
    "measured_joint_position": TRACKING_TELEMETRY_JOINT_COUNT,
    "measured_joint_velocity": TRACKING_TELEMETRY_JOINT_COUNT,
    "root_angular_velocity": 3,
    "projected_gravity": 3,
    "policy_action": TRACKING_TELEMETRY_JOINT_COUNT,
    "command_joint_position": TRACKING_TELEMETRY_JOINT_COUNT,
}


class TrackingTelemetryProtocolError(ValueError):
    """Raised when a controller telemetry message violates the wire schema."""


def _strict_int(payload: Dict[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingTelemetryProtocolError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise TrackingTelemetryProtocolError(
            f"{key} must be in [{minimum}, {maximum}], got {value}"
        )
    return int(value)


def _strict_bool(payload: Dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise TrackingTelemetryProtocolError(f"{key} must be a boolean")
    return value


def _finite_vector(payload: Dict[str, Any], key: str, length: int) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise TrackingTelemetryProtocolError(
            f"{key} must be a JSON array of length {length}, got {actual}"
        )
    normalized: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TrackingTelemetryProtocolError(
                f"{key}[{index}] must be a finite number"
            )
        number = float(item)
        if not math.isfinite(number):
            raise TrackingTelemetryProtocolError(
                f"{key}[{index}] must be finite"
            )
        normalized.append(number)
    return normalized


def _optional_finite_number(payload: Dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingTelemetryProtocolError(f"{key} must be finite or null")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise TrackingTelemetryProtocolError(
            f"{key} must be a non-negative finite number or null"
        )
    return normalized


def _optional_uint64(payload: Dict[str, Any], key: str) -> int | None:
    if payload.get(key) is None:
        return None
    return _strict_int(payload, key, minimum=0, maximum=UINT64_MAX)


def _optional_generation(payload: Dict[str, Any], key: str) -> int | None:
    """Normalize an optional uint64 generation; legacy -1 means unavailable."""

    value = payload.get(key)
    if value is None or value == -1:
        return None
    return _strict_int(payload, key, minimum=0, maximum=UINT64_MAX)


def _optional_bool(payload: Dict[str, Any], key: str) -> bool | None:
    if payload.get(key) is None:
        return None
    return _strict_bool(payload, key)


def parse_tracking_telemetry_parts(parts: Sequence[bytes]) -> Dict[str, Any]:
    """Decode and strictly validate one two-frame telemetry publication.

    The first frame must be the literal ASCII topic ``tracking_telemetry`` and
    the second frame must be a UTF-8 JSON object conforming to schema version
    1.  Numeric arrays are normalized to Python floats after finite checks.
    Recorder receive timestamps are intentionally added by the caller only
    after this function succeeds.
    """

    if len(parts) != 2:
        raise TrackingTelemetryProtocolError(
            f"expected 2 multipart frames, got {len(parts)}"
        )
    topic_bytes, payload_bytes = parts
    try:
        topic = topic_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TrackingTelemetryProtocolError("topic is not ASCII") from exc
    if topic != TRACKING_TELEMETRY_TOPIC:
        raise TrackingTelemetryProtocolError(
            f"unexpected topic {topic!r}; expected {TRACKING_TELEMETRY_TOPIC!r}"
        )

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackingTelemetryProtocolError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingTelemetryProtocolError("payload must be a JSON object")

    schema_version = _strict_int(
        payload,
        "schema_version",
        minimum=0,
        maximum=UINT64_MAX,
    )
    if schema_version != TRACKING_TELEMETRY_SCHEMA_VERSION:
        raise TrackingTelemetryProtocolError(
            "schema_version mismatch: expected "
            f"{TRACKING_TELEMETRY_SCHEMA_VERSION}, got {schema_version}"
        )
    if payload.get("stream") != TRACKING_TELEMETRY_TOPIC:
        raise TrackingTelemetryProtocolError(
            f"stream must equal {TRACKING_TELEMETRY_TOPIC!r}"
        )

    normalized = dict(payload)
    normalized["sequence"] = _strict_int(
        payload, "sequence", minimum=0, maximum=UINT64_MAX
    )
    normalized["sample_monotonic_ns"] = _strict_int(
        payload, "sample_monotonic_ns", minimum=1, maximum=UINT64_MAX
    )
    normalized["sample_wall_time_ns"] = _strict_int(
        payload, "sample_wall_time_ns", minimum=1, maximum=UINT64_MAX
    )
    joint_count = _strict_int(
        payload,
        "joint_count",
        minimum=0,
        maximum=UINT64_MAX,
    )
    if joint_count != TRACKING_TELEMETRY_JOINT_COUNT:
        raise TrackingTelemetryProtocolError(
            f"joint_count must be {TRACKING_TELEMETRY_JOINT_COUNT}, got {joint_count}"
        )
    normalized["joint_count"] = joint_count
    joint_names = payload.get("joint_names")
    if joint_names != X2_TRACKING_JOINT_NAMES:
        raise TrackingTelemetryProtocolError(
            "joint_names do not match the canonical X2 tracking order"
        )
    normalized["joint_names"] = list(joint_names)
    normalized["vr_user_enabled"] = _strict_bool(payload, "vr_user_enabled")
    normalized["vr_session_active"] = _strict_bool(payload, "vr_session_active")
    for key, length in _VECTOR_DIMS.items():
        normalized[key] = _finite_vector(payload, key, length)
    # Added as optional provenance within schema v1 so new controller builds
    # can expose reference freshness without making older v1 recordings
    # unreadable.
    normalized["reference_source_age_ms"] = _optional_finite_number(
        payload, "reference_source_age_ms"
    )
    normalized["reference_source_time_exact"] = _strict_bool(
        payload, "reference_source_time_exact"
    )
    normalized["reference_source_frame_sequence"] = _optional_uint64(
        payload, "reference_source_frame_sequence"
    )
    normalized["reference_is_transition"] = _optional_bool(
        payload, "reference_is_transition"
    )
    normalized["reference_is_padded"] = _optional_bool(
        payload, "reference_is_padded"
    )
    normalized["reference_is_fallback"] = _optional_bool(
        payload, "reference_is_fallback"
    )
    normalized["sensor_generation"] = _optional_generation(
        payload, "sensor_generation"
    )

    # The paired controller wall stamp is the portable synchronization anchor;
    # keep the monotonic stamp as provenance for same-host diagnostics.
    normalized["source_timestamp_ns"] = normalized["sample_wall_time_ns"]
    return normalized


class TrackingTelemetrySequenceTracker:
    """Track loss and publisher restarts in a uint64 telemetry sequence."""

    def __init__(self) -> None:
        self._last: int | None = None

    def observe(self, event: Dict[str, Any]) -> int:
        """Annotate ``event`` and return the number of missing publications."""

        current = int(event["sequence"])
        if self._last is None:
            self._last = current
            return 0
        previous = self._last
        self._last = current
        if current < previous:
            event["tracking_telemetry_sequence_reset"] = True
            return 0
        if current == previous:
            raise TrackingTelemetryProtocolError(
                f"duplicate telemetry sequence {current}"
            )
        gap = current - previous - 1
        if gap:
            event["tracking_telemetry_gap_before"] = gap
        return gap
