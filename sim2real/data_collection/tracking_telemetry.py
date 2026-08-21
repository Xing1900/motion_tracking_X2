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
REFERENCE_AGE_SPLIT_TOLERANCE_MS = 0.5
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
_REFERENCE_AGE_FIELDS = (
    "reference_source_age_ms",
    "reference_total_age_ms",
    "reference_upstream_age_at_bridge_ms",
    "reference_bridge_to_policy_age_ms",
)


class TrackingTelemetryProtocolError(ValueError):
    """Raised when a controller telemetry message violates the wire schema."""


class TrackingTelemetryReferenceAgeError(TrackingTelemetryProtocolError):
    """Raised when production telemetry lacks a self-consistent age split."""

    def __init__(self, message: str, *, drop_reason: str) -> None:
        super().__init__(message)
        self.drop_reason = str(drop_reason)


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
    # ``reference_source_age_ms`` is the schema-v1 legacy name for total
    # source-to-policy-consumption age.  New publishers keep that alias and
    # additionally split it at the bridge reply boundary.  All additions stay
    # optional within schema v1 so an old recording remains readable for
    # explicit diagnostics instead of being misparsed as a new split sample.
    normalized["reference_age_split_fields_present"] = all(
        age_key in payload for age_key in _REFERENCE_AGE_FIELDS
    )
    for age_key in _REFERENCE_AGE_FIELDS:
        normalized[age_key] = _optional_finite_number(payload, age_key)
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


def require_reference_age_split(event: Dict[str, Any]) -> None:
    """Require the additive schema-v1 split used by production recording.

    The wire parser keeps these fields optional so old raw JSON remains
    readable.  A live ``groot_n17`` recorder is stricter: admitting a legacy
    total-only sample would make the offline 80 ms upstream gate ambiguous.
    """

    if event.get("reference_age_split_fields_present") is not True:
        raise TrackingTelemetryReferenceAgeError(
            "reference age split fields are absent (old controller wire contract)",
            drop_reason="reference_age_split_missing",
        )

    values = [event.get(key) for key in _REFERENCE_AGE_FIELDS]
    real_consumed_reference = (
        event.get("vr_session_active") is True
        and event.get("reference_is_transition") is False
        and event.get("reference_is_padded") is False
    )
    if (
        real_consumed_reference
        and event.get("reference_source_time_exact") is not True
    ):
        raise TrackingTelemetryReferenceAgeError(
            "active operator reference has no exact robot-local source stamp",
            drop_reason="reference_age_split_missing",
        )
    numeric_values_required = real_consumed_reference
    if all(value is None for value in values):
        if numeric_values_required:
            raise TrackingTelemetryReferenceAgeError(
                "reference age split is null for an active exact operator reference",
                drop_reason="reference_age_split_missing",
            )
        # A new controller deliberately publishes null values before VR has an
        # exact operator source (and for synthesized transition/padding).  Key
        # presence proves the wire contract without killing the idle recorder.
        return
    if any(value is None for value in values):
        missing = [
            key
            for key, value in zip(_REFERENCE_AGE_FIELDS, values)
            if value is None
        ]
        raise TrackingTelemetryReferenceAgeError(
            "reference age split is partially null: " + ", ".join(missing),
            drop_reason="reference_age_split_inconsistent",
        )

    legacy_total_ms = float(event["reference_source_age_ms"])
    explicit_total_ms = float(event["reference_total_age_ms"])
    upstream_ms = float(event["reference_upstream_age_at_bridge_ms"])
    bridge_to_policy_ms = float(event["reference_bridge_to_policy_age_ms"])
    if (
        abs(legacy_total_ms - explicit_total_ms)
        > REFERENCE_AGE_SPLIT_TOLERANCE_MS
    ):
        raise TrackingTelemetryReferenceAgeError(
            "reference age total alias mismatch: "
            f"legacy={legacy_total_ms:.6f} ms, explicit={explicit_total_ms:.6f} ms",
            drop_reason="reference_age_split_inconsistent",
        )
    residual_ms = explicit_total_ms - upstream_ms - bridge_to_policy_ms
    if abs(residual_ms) > REFERENCE_AGE_SPLIT_TOLERANCE_MS:
        raise TrackingTelemetryReferenceAgeError(
            "reference age split inconsistent: "
            f"total={explicit_total_ms:.6f} ms, upstream={upstream_ms:.6f} ms, "
            f"bridge_to_policy={bridge_to_policy_ms:.6f} ms, "
            f"residual={residual_ms:.6f} ms",
            drop_reason="reference_age_split_inconsistent",
        )


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
