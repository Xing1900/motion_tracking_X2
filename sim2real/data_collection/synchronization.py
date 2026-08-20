"""Timestamp mapping helpers for X2 raw demonstration streams.

Raw events retain both source/header timestamps and recorder arrival times.
The functions here map source timestamps into the recorder monotonic domain
without overwriting either original timestamp.  Converter and visualization
tools should use this module so they cannot silently disagree about time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


TIME_BASIS_SOURCE = "source"
TIME_BASIS_RECEIVER = "receiver"
TIME_BASES = (TIME_BASIS_SOURCE, TIME_BASIS_RECEIVER)

SYNC_TIME_KEY = "_sync_time_ns"
SYNC_TIME_ORIGIN_KEY = "_sync_time_origin"
SYNC_APPARENT_LATENCY_KEY = "_sync_apparent_latency_ms"
DEFAULT_LARGE_SOURCE_REGRESSION_MS = 100.0


@dataclass(frozen=True)
class TimePoint:
    time_ns: int
    origin: str
    apparent_latency_ms: Optional[float] = None


@dataclass
class TimedStream:
    events: list[Dict[str, Any]]
    input_count: int
    missing_time_count: int
    source_regression_count: int
    large_source_regression_count: int
    max_source_regression_ms: float


def _integer(event: Dict[str, Any], key: str) -> Optional[int]:
    value = event.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def arrival_time_ns(event: Dict[str, Any]) -> Optional[int]:
    """Return a recorded arrival timestamp without inventing a fallback."""

    for key in (
        "recorder_recv_monotonic_ns",
        "bridge_sample_monotonic_ns",
        "bridge_recv_monotonic_ns",
        "bridge_enqueue_monotonic_ns",
    ):
        value = _integer(event, key)
        if value is not None and value >= 0:
            return value
    return None


def _recorder_clock_pair(event: Dict[str, Any]) -> Optional[tuple[int, int]]:
    recv_mono_ns = _integer(event, "recorder_recv_monotonic_ns")
    recv_wall_ns = _integer(event, "recorder_recv_wall_time_ns")
    if (
        recv_mono_ns is None
        or recv_wall_ns is None
        or recv_mono_ns < 0
        or recv_wall_ns <= 0
    ):
        return None
    return recv_mono_ns, recv_wall_ns


def map_wall_to_recorder_monotonic(
    source_wall_ns: int,
    event: Dict[str, Any],
    *,
    origin: str,
) -> Optional[TimePoint]:
    """Map a wall/ROS timestamp to recorder monotonic time.

    This is valid when source wall time and recorder wall time share a
    synchronized clock.  The apparent latency is retained for QA; it is not
    treated as the observation timestamp.
    """

    if source_wall_ns <= 0:
        return None
    recorder_pair = _recorder_clock_pair(event)
    if recorder_pair is None:
        return None
    recv_mono_ns, recv_wall_ns = recorder_pair
    return TimePoint(
        time_ns=recv_mono_ns + int(source_wall_ns) - recv_wall_ns,
        origin=origin,
        apparent_latency_ms=(recv_wall_ns - int(source_wall_ns)) / 1e6,
    )


def _bridge_clock_anchor(event: Dict[str, Any]) -> Optional[tuple[int, int, str]]:
    for prefix in ("bridge_enqueue", "bridge_sample", "bridge_recv"):
        mono_ns = _integer(event, f"{prefix}_monotonic_ns")
        wall_ns = _integer(event, f"{prefix}_wall_time_ns")
        if mono_ns is not None and wall_ns is not None and mono_ns >= 0 and wall_ns > 0:
            return mono_ns, wall_ns, prefix
    return None


def map_bridge_monotonic_to_recorder(
    bridge_time_ns: int,
    event: Dict[str, Any],
    *,
    origin: str,
) -> Optional[TimePoint]:
    """Map a bridge-host monotonic timestamp through paired wall clocks."""

    if bridge_time_ns < 0:
        return None
    anchor = _bridge_clock_anchor(event)
    if anchor is None:
        return None
    anchor_mono_ns, anchor_wall_ns, anchor_name = anchor
    bridge_wall_ns = anchor_wall_ns + int(bridge_time_ns) - anchor_mono_ns
    return map_wall_to_recorder_monotonic(
        bridge_wall_ns,
        event,
        origin=f"{origin}:via_{anchor_name}",
    )


def source_time_point(event: Dict[str, Any], stream: str) -> Optional[TimePoint]:
    """Return the physical/command time used for source-based alignment."""

    stream = str(stream)

    # A reference event describes the command replied to C++.  The sample
    # target is the older GMR lookback time, not the action issue time.
    if stream == "reference":
        bridge_recv_ns = _integer(event, "bridge_recv_monotonic_ns")
        if bridge_recv_ns is not None:
            mapped = map_bridge_monotonic_to_recorder(
                bridge_recv_ns,
                event,
                origin="reference_command",
            )
            if mapped is not None:
                return mapped
        bridge_enqueue_ns = _integer(event, "bridge_enqueue_monotonic_ns")
        if bridge_enqueue_ns is not None:
            mapped = map_bridge_monotonic_to_recorder(
                bridge_enqueue_ns,
                event,
                origin="reference_tap_enqueue",
            )
            if mapped is not None:
                return mapped

    # Retarget/XR events are bridge-process diagnostics.  Prefer their bridge
    # receive time; XR SDK timestamps are not assumed to be ROS wall time.
    if stream in {"retarget", "xr"}:
        for key, origin in (
            ("bridge_recv_monotonic_ns", f"{stream}_bridge_receive"),
            ("bridge_enqueue_monotonic_ns", f"{stream}_tap_enqueue"),
        ):
            value = _integer(event, key)
            if value is not None:
                mapped = map_bridge_monotonic_to_recorder(value, event, origin=origin)
                if mapped is not None:
                    return mapped

    # Camera, robot feedback, IMU and hand status carry ROS/header wall time.
    if stream in {"camera_head", "joint_states", "imu_torso", "imu_chest", "hand_command"}:
        source_wall_ns = _integer(event, "source_timestamp_ns")
        if source_wall_ns is not None:
            mapped = map_wall_to_recorder_monotonic(
                source_wall_ns,
                event,
                origin=f"{stream}_source_header",
            )
            if mapped is not None:
                return mapped

    # Legacy controller actions are sampled by the bridge at 50 Hz.  Their XR
    # device timestamp is retained only as provenance because its epoch is not
    # guaranteed to match ROS wall time.
    if stream == "controller":
        bridge_sample_ns = _integer(event, "bridge_sample_monotonic_ns")
        if bridge_sample_ns is not None:
            mapped = map_bridge_monotonic_to_recorder(
                bridge_sample_ns,
                event,
                origin="controller_bridge_sample",
            )
            if mapped is not None:
                return mapped

    return None


def event_time_point(
    event: Dict[str, Any],
    stream: str,
    *,
    time_basis: str,
    max_abs_apparent_latency_ms: float = 10_000.0,
) -> Optional[TimePoint]:
    if time_basis not in TIME_BASES:
        raise ValueError(f"unsupported time basis: {time_basis!r}")
    if time_basis == TIME_BASIS_RECEIVER:
        value = arrival_time_ns(event)
        return None if value is None else TimePoint(value, "recorder_arrival")

    point = source_time_point(event, stream)
    if point is None:
        return None
    latency_ms = point.apparent_latency_ms
    if (
        latency_ms is not None
        and (
            not (-max_abs_apparent_latency_ms <= latency_ms <= max_abs_apparent_latency_ms)
        )
    ):
        return None
    return point


def timed_stream(
    events: Iterable[Dict[str, Any]],
    stream: str,
    *,
    time_basis: str,
    max_abs_apparent_latency_ms: float = 10_000.0,
    large_source_regression_ms: float = DEFAULT_LARGE_SOURCE_REGRESSION_MS,
) -> TimedStream:
    """Copy events and attach a derived synchronization timestamp."""

    output: list[Dict[str, Any]] = []
    missing_time_count = 0
    source_regression_count = 0
    large_source_regression_count = 0
    max_source_regression_ns = 0
    high_watermark_by_source: Dict[str, int] = {}
    input_count = 0
    for raw_event in events:
        input_count += 1
        point = event_time_point(
            raw_event,
            stream,
            time_basis=time_basis,
            max_abs_apparent_latency_ms=max_abs_apparent_latency_ms,
        )
        if point is None:
            missing_time_count += 1
            continue
        source_key = str(raw_event.get("topic") or stream)
        high_watermark_ns = high_watermark_by_source.get(source_key)
        if high_watermark_ns is not None and point.time_ns < high_watermark_ns:
            source_regression_count += 1
            # Compare with the highest value already observed. A clock reset
            # that walks backward in several small steps must not evade the
            # large-regression threshold.
            regression_ns = high_watermark_ns - point.time_ns
            max_source_regression_ns = max(max_source_regression_ns, regression_ns)
            if regression_ns > int(float(large_source_regression_ms) * 1e6):
                large_source_regression_count += 1
        high_watermark_by_source[source_key] = max(
            point.time_ns,
            high_watermark_ns if high_watermark_ns is not None else point.time_ns,
        )
        event = dict(raw_event)
        event[SYNC_TIME_KEY] = int(point.time_ns)
        event[SYNC_TIME_ORIGIN_KEY] = point.origin
        if point.apparent_latency_ms is not None:
            event[SYNC_APPARENT_LATENCY_KEY] = float(point.apparent_latency_ms)
        output.append(event)

    output.sort(key=lambda item: int(item[SYNC_TIME_KEY]))
    return TimedStream(
        events=output,
        input_count=input_count,
        missing_time_count=missing_time_count,
        source_regression_count=source_regression_count,
        large_source_regression_count=large_source_regression_count,
        max_source_regression_ms=max_source_regression_ns / 1e6,
    )


def synchronization_time_ns(event: Dict[str, Any]) -> int:
    value = _integer(event, SYNC_TIME_KEY)
    if value is None:
        raise ValueError("event has no derived synchronization timestamp")
    return value
