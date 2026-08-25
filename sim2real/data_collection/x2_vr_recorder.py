#!/usr/bin/env python3
"""Record X2 VR demonstrations without touching the real-time control path.

The process subscribes to the teleop bridge's independent PUB tap, robot joint
states and IMUs, plus either the robot-side camera TCP tap or the legacy ROS
camera topic.  Right ``key_one`` starts an episode; left ``key_one`` stops it.

This writes a loss-preserving raw format.  Conversion/resampling into a
LeRobotDataset happens offline in ``convert_to_lerobot.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import socket
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

try:
    from .camera_tap_client import CameraTapClient
    from .raw_episode_writer import (
        DEFAULT_CAMERA_WRITER_JOIN_TIMEOUT_S,
        RawEpisodeManager,
        event_monotonic_ns,
        ros_stamp_to_ns,
    )
    from .schema import (
        BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES,
        REFERENCE_DIAGNOSTICS_SCHEMA_VERSION,
        TELEOP_TAP_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
        normalize_bridge_runtime_effective_params,
    )
    from .tracking_telemetry import (
        TRACKING_TELEMETRY_SCHEMA_VERSION,
        TRACKING_TELEMETRY_TOPIC,
        TrackingTelemetryProtocolError,
        TrackingTelemetryReferenceAgeError,
        TrackingTelemetryReferenceDiagnosticsError,
        TrackingTelemetrySequenceTracker,
        parse_tracking_telemetry_parts,
        require_reference_age_split,
        require_reference_diagnostics_contract,
    )
except ImportError:  # Direct execution from this directory.
    from camera_tap_client import CameraTapClient
    from raw_episode_writer import (
        DEFAULT_CAMERA_WRITER_JOIN_TIMEOUT_S,
        RawEpisodeManager,
        event_monotonic_ns,
        ros_stamp_to_ns,
    )
    from schema import (
        BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES,
        REFERENCE_DIAGNOSTICS_SCHEMA_VERSION,
        TELEOP_TAP_SCHEMA_VERSION,
        X2_TRACKING_JOINT_NAMES,
        normalize_bridge_runtime_effective_params,
    )
    from tracking_telemetry import (
        TRACKING_TELEMETRY_SCHEMA_VERSION,
        TRACKING_TELEMETRY_TOPIC,
        TrackingTelemetryProtocolError,
        TrackingTelemetryReferenceAgeError,
        TrackingTelemetryReferenceDiagnosticsError,
        TrackingTelemetrySequenceTracker,
        parse_tracking_telemetry_parts,
        require_reference_age_split,
        require_reference_diagnostics_contract,
    )


DEFAULT_CAMERA_TOPIC = "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed"
DEFAULT_HAND_STATUS_TOPIC = "/vr_hand_controller/status"
DEFAULT_AIMDK_JOINT_TOPICS = [
    "/aima/hal/joint/leg/state",
    "/aima/hal/joint/waist/state",
    "/aima/hal/joint/arm/state",
    "/aima/hal/joint/head/state",
]
DEFAULT_COMPAT_JOINT_TOPICS = [
    "/joint_states/leg",
    "/joint_states/waist",
    "/joint_states/arm",
    "/joint_states/head",
]
DEFAULT_AIMDK_IMU_TOPICS = [
    "/aima/hal/imu/torso/state",
    "/aima/hal/imu/chest/state",
]
DEFAULT_COMPAT_IMU_TOPICS = [
    "/imu/torso/data",
    "/imu/chest/data",
]
SENSOR_PROFILES = {
    "aimdk": {
        "joint_topics": DEFAULT_AIMDK_JOINT_TOPICS,
        "imu_topics": DEFAULT_AIMDK_IMU_TOPICS,
        "joint_message_type": "aimdk_msgs/msg/JointStateArray",
    },
    "compat": {
        "joint_topics": DEFAULT_COMPAT_JOINT_TOPICS,
        "imu_topics": DEFAULT_COMPAT_IMU_TOPICS,
        "joint_message_type": "sensor_msgs/msg/JointState",
    },
}

RECORD_PROFILES = ("full", "vla", "groot_n17")
VLA_TAP_STREAMS = ("controller", "reference")
GROOT_N17_TAP_STREAMS = ("controller",)
DEFAULT_TRACKING_TAP_ADDR = "tcp://127.0.0.1:28707"
DEFAULT_DISPATCHER_JOIN_TIMEOUT_S = 15.0
MIN_DISPATCHER_JOIN_MARGIN_S = 1.0
REQUIRED_GROOT_PROVENANCE_FILES = {
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


def _hash_provenance_files(entries: list[str]) -> Dict[str, Dict[str, Any]]:
    """Hash named runtime artifacts without trusting git/worktree state."""

    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "--provenance_file must use NAME=/absolute/or/relative/path"
            )
        name, raw_path = entry.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid provenance name: {name!r}")
        if name in result:
            raise ValueError(f"duplicate provenance name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance file not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return dict(sorted(result.items()))


def _hash_capture_provenance_files(
    entries: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Hash the running recorder itself together with declared artifacts."""

    return _hash_provenance_files(
        [f"recorder={Path(__file__).resolve()}", *entries]
    )


def _bridge_runtime_effective_params(
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Return the explicitly declared bridge settings used for this capture."""

    payload = {
        name: getattr(args, f"bridge_{name}", None)
        for name in BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES
    }
    provided = {name for name, value in payload.items() if value is not None}
    if not provided:
        if args.record_profile == "groot_n17":
            raise ValueError(
                "--record_profile groot_n17 requires all --bridge_* runtime params: "
                + ", ".join(BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES)
            )
        return None
    missing = sorted(set(BRIDGE_RUNTIME_EFFECTIVE_PARAM_NAMES) - provided)
    if missing:
        raise ValueError(
            "bridge runtime effective params are incomplete; missing: "
            + ", ".join(missing)
        )
    return normalize_bridge_runtime_effective_params(payload)


class TapSequenceGapTracker:
    """Detect recorder-tap transport gaps without confusing topic filters.

    New bridges provide ``tap_topic_seq``, which is checked independently for
    every received topic. Older bridges only provide the global ``tap_seq``;
    full-profile recording can safely use that as a compatibility fallback
    because it subscribes to every topic. A filtered VLA subscription must not
    use the legacy global sequence because skipped topics are intentional.
    """

    def __init__(self, *, allow_legacy_global: bool) -> None:
        self.allow_legacy_global = bool(allow_legacy_global)
        self._last_topic_seq: Dict[str, int] = {}
        self._last_global_seq: Optional[int] = None

    @staticmethod
    def _sequence(value: Any) -> Optional[int]:
        try:
            sequence = int(value)
        except (TypeError, ValueError):
            return None
        return sequence if sequence >= 0 else None

    @staticmethod
    def _gap_and_next(last: Optional[int], current: int) -> tuple[int, int, bool]:
        if last is None:
            return 0, current, False
        if current < last:
            # A bridge restart resets its counters. Re-baseline without
            # inventing a huge gap; expose the reset on the stored event.
            return 0, current, True
        if current == last:
            return 0, last, False
        return max(0, current - last - 1), current, False

    def observe(self, topic: str, event: Dict[str, Any]) -> Dict[str, int]:
        """Annotate ``event`` and return manifest drop counters to increment."""

        topic = str(topic)
        topic_seq = self._sequence(event.get("tap_topic_seq"))
        global_seq = self._sequence(event.get("tap_seq"))

        # Keep the legacy baseline current even while receiving a new bridge.
        # This makes a mixed-version restart degrade cleanly to the fallback.
        if global_seq is None:
            global_gap, global_reset = 0, False
        else:
            global_gap, global_next, global_reset = self._gap_and_next(
                self._last_global_seq, global_seq
            )
            self._last_global_seq = global_next

        if topic_seq is not None:
            gap, next_seq, reset = self._gap_and_next(
                self._last_topic_seq.get(topic), topic_seq
            )
            self._last_topic_seq[topic] = next_seq
            if reset:
                event["tap_topic_sequence_reset"] = True
            if not gap:
                return {}
            event["tap_topic_gap_before"] = gap
            return {f"teleop_tap_transport.{topic}": gap}

        if not self.allow_legacy_global or global_seq is None:
            return {}
        if global_reset:
            event["tap_sequence_reset"] = True
        if not global_gap:
            return {}
        event["tap_gap_before"] = global_gap
        return {"teleop_tap_transport": global_gap}


class LatestEventSlot:
    """Thread-safe single-event slot used for coalesced high-bandwidth data."""

    def __init__(self) -> None:
        self._event: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def replace(self, event: Dict[str, Any]) -> bool:
        """Store ``event`` and return whether an older pending event was replaced."""

        with self._lock:
            replaced = self._event is not None
            self._event = event
            return replaced

    def take(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            event = self._event
            self._event = None
            return event

    def take_if_not_after(self, timestamp_ns: int) -> Optional[Dict[str, Any]]:
        """Take the pending event only when it is no newer than ``timestamp_ns``."""

        with self._lock:
            if self._event is None:
                return None
            if event_monotonic_ns(self._event) > int(timestamp_ns):
                return None
            event = self._event
            self._event = None
            return event

    def pending(self) -> bool:
        with self._lock:
            return self._event is not None


class IngressQueue:
    """Bounded FIFO plus a coalesced latest-only camera slot.

    Joint, IMU, hand and controller traffic retain normal FIFO semantics.  A
    direct ROS callback or TCP camera-tap reader uses
    :meth:`put_latest_camera`, so slow image writes can retain at most one
    not-yet-dispatched compressed frame instead of filling the shared FIFO
    with stale megabyte-sized messages.
    """

    def __init__(self, maxsize: int) -> None:
        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max(1, int(maxsize)))
        self._received: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._peak_size = 0
        self._lock = threading.Lock()
        self._latest_camera = LatestEventSlot()
        self._wake_event = threading.Event()

    def put(self, event: Dict[str, Any]) -> None:
        stream = str(event.get("stream", "unknown"))
        with self._lock:
            self._received[stream] += 1
        inserted = False
        try:
            self.queue.put_nowait(event)
            inserted = True
        except queue.Full:
            if stream == "controller":
                # Preserve the newest button sample (and therefore start/stop
                # edges) by sacrificing one older non-controller event.  Never
                # evict a queued release/press sample to make room for another
                # controller sample: doing so can erase the next rising edge.
                evicted = self._evict_oldest_non_controller()
                if evicted is not None:
                    with self._lock:
                        self._dropped[str(evicted.get("stream", "unknown"))] += 1
                    try:
                        self.queue.put_nowait(event)
                        inserted = True
                    except queue.Full:
                        with self._lock:
                            self._dropped[stream] += 1
                else:
                    with self._lock:
                        self._dropped[stream] += 1
            else:
                with self._lock:
                    self._dropped[stream] += 1
        if inserted:
            self._wake_event.set()
        with self._lock:
            self._peak_size = max(self._peak_size, self.size())

    def _evict_oldest_non_controller(self) -> Optional[Dict[str, Any]]:
        """Remove one queued data event while preserving all button samples."""

        # queue.Queue exposes no selective removal API.  Its documented mutex
        # protects the internal deque and completion counters used here.
        with self.queue.mutex:
            for index, candidate in enumerate(self.queue.queue):
                if str(candidate.get("stream", "unknown")) == "controller":
                    continue
                evicted = self.queue.queue[index]
                del self.queue.queue[index]
                self.queue.unfinished_tasks -= 1
                if self.queue.unfinished_tasks == 0:
                    self.queue.all_tasks_done.notify_all()
                self.queue.not_full.notify()
                return evicted
        return None

    def put_latest_camera(self, event: Dict[str, Any]) -> None:
        """Coalesce a camera event outside the public FIFO."""

        stream = str(event.get("stream", "unknown"))
        with self._lock:
            self._received[stream] += 1
        if self._latest_camera.replace(event):
            # Keep this separate from camera transport/FIFO loss.  It means a
            # valid callback was deliberately superseded before disk dispatch.
            self.note_drop("camera_coalesced")
        self._wake_event.set()
        with self._lock:
            self._peak_size = max(self._peak_size, self.size())

    def _oldest_controller_time_ns(self) -> Optional[int]:
        """Return the oldest queued controller timestamp without removing it."""

        # Camera delivery may overtake ordinary state samples because source
        # timestamps, rather than file-write order, drive offline alignment.
        # It must never overtake an older A/X controller edge, however, or an
        # image could be assigned to the wrong episode boundary.
        with self.queue.mutex:
            for event in self.queue.queue:
                if str(event.get("stream", "unknown")) == "controller":
                    return event_monotonic_ns(event)
        return None

    def _try_get_next(self) -> Optional[tuple[Dict[str, Any], bool]]:
        """Return (event, came_from_fifo), approximately in receive-time order."""

        # The production camera paths now use CameraIngressDispatcher.  Keep
        # this compatibility slot cheap when unused: scanning an 8192-event
        # state FIFO for controller edges on every dequeue would itself burn
        # material CPU on the robot.
        if self._latest_camera.pending():
            oldest_controller_ns = self._oldest_controller_time_ns()
            if oldest_controller_ns is None:
                camera_event = self._latest_camera.take()
                if camera_event is not None:
                    return camera_event, False
            else:
                # State/reference write order is irrelevant to source-time
                # alignment.  Let camera bypass that backlog, while preserving
                # an older (or exactly simultaneous) A/X edge.
                camera_event = self._latest_camera.take_if_not_after(
                    oldest_controller_ns - 1
                )
                if camera_event is not None:
                    return camera_event, False

        try:
            return self.queue.get_nowait(), True
        except queue.Empty:
            # A producer may have changed the FIFO between peek and get.
            camera_event = self._latest_camera.take()
            return None if camera_event is None else (camera_event, False)

    def get_next(self, timeout: float) -> Optional[tuple[Dict[str, Any], bool]]:
        """Wait for either FIFO work or the latest coalesced camera frame."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            item = self._try_get_next()
            if item is not None:
                return item
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            self._wake_event.clear()
            # Close the clear/wait race by checking both sources once more.
            item = self._try_get_next()
            if item is not None:
                return item
            self._wake_event.wait(remaining)

    def empty(self) -> bool:
        return self.queue.empty() and not self._latest_camera.pending()

    def wake(self) -> None:
        self._wake_event.set()

    def note_drop(self, stream: str, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._dropped[str(stream)] += int(count)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._received)

    def drops(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._dropped)

    def size(self) -> int:
        return self.queue.qsize() + int(self._latest_camera.pending())

    def peak_size(self) -> int:
        with self._lock:
            return self._peak_size


class EventDispatcher:
    def __init__(
        self,
        ingress: IngressQueue,
        manager: RawEpisodeManager,
        *,
        join_timeout_s: float = 10.0,
        auxiliary_ingress_ready: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.ingress = ingress
        self.manager = manager
        self.stop_event = threading.Event()
        self.fatal_exception: Optional[BaseException] = None
        self.join_timeout_s = max(0.0, float(join_timeout_s))
        self.auxiliary_ingress_ready = auxiliary_ingress_ready or (lambda: True)
        self.thread = threading.Thread(target=self._run, name="x2-raw-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.ingress.empty():
            item = self.ingress.get_next(timeout=0.05)
            if item is None:
                if not self.auxiliary_ingress_ready():
                    continue
                try:
                    self.manager.tick()
                except BaseException as exc:
                    self.fatal_exception = exc
                    self.stop_event.set()
                    self.manager.request_abort()
                    print(f"[recorder] fatal writer error: {exc}")
                    return
                continue
            event, came_from_fifo = item
            try:
                self.manager.handle_event(event)
            except BaseException as exc:
                self.fatal_exception = exc
                self.stop_event.set()
                self.manager.request_abort()
                print(f"[recorder] fatal writer error: {exc}")
                return
            finally:
                if came_from_fifo:
                    self.ingress.queue.task_done()
            # Do not finalize post-roll while already-received events remain in
            # the writer backlog.  Raw may contain a little extra tail; the
            # converter crops exactly at the stop trigger.
            if self.ingress.empty() and self.auxiliary_ingress_ready():
                try:
                    self.manager.tick()
                except BaseException as exc:
                    self.fatal_exception = exc
                    self.stop_event.set()
                    self.manager.request_abort()
                    print(f"[recorder] fatal writer error: {exc}")
                    return

    def close(self) -> None:
        self.stop_event.set()
        self.ingress.wake()
        self.thread.join(timeout=self.join_timeout_s)
        if self.thread.is_alive():
            self.manager.request_abort()
            if self.fatal_exception is None:
                self.fatal_exception = TimeoutError(
                    "recorder writer did not stop within "
                    f"{self.join_timeout_s:.3f} s"
                )
            print(
                "[recorder] writer did not stop within "
                f"{self.join_timeout_s:.3f} s; preserving partial episode"
            )
        else:
            try:
                if self.fatal_exception is not None:
                    self.manager.abort()
                else:
                    self.manager.close()
            except BaseException as exc:
                if self.fatal_exception is None:
                    self.fatal_exception = exc
                print(f"[recorder] failed to finalize active episode: {exc}")


class CameraIngressDispatcher:
    """Bounded camera-only ingress path into the episode's async writer.

    TCP and ROS camera callbacks only copy/enqueue a compressed frame here.
    The worker performs episode/pre-roll routing independently from the public
    state dispatcher, then ``RawEpisodeWriter`` performs image I/O on its own
    second-stage camera thread.  Both stages are bounded and drop the oldest
    pending image, preserving low latency under overload.
    """

    def __init__(
        self,
        manager: RawEpisodeManager,
        *,
        max_frames: int,
        max_bytes: int,
        join_timeout_s: float = 3.0,
    ) -> None:
        self.manager = manager
        self.max_frames = max(1, int(max_frames))
        self.max_bytes = max(1, int(max_bytes))
        self.join_timeout_s = max(0.0, float(join_timeout_s))
        self._condition = threading.Condition()
        self._queue: Deque[tuple[Dict[str, Any], int]] = deque()
        self._queue_bytes = 0
        self._accepting = True
        self._stop_requested = False
        self._inflight = False
        self._inflight_timestamp_ns: Optional[int] = None
        self._received: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._peak_frames = 0
        self._peak_bytes = 0
        self.fatal_exception: Optional[BaseException] = None
        self.thread = threading.Thread(
            target=self._run,
            name="x2-camera-ingress",
            daemon=True,
        )

    @staticmethod
    def _payload_size(event: Dict[str, Any]) -> int:
        payload = event.get("data")
        return len(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else 0

    def start(self) -> None:
        self.thread.start()

    def put(self, event: Dict[str, Any]) -> None:
        """Enqueue one compressed frame without blocking its transport thread."""

        stream = str(event.get("stream", "unknown"))
        size = self._payload_size(event)
        with self._condition:
            self._received[stream] += 1
            if not self._accepting:
                self._dropped["camera_ingress_closed"] += 1
                return
            if stream != "camera_head" or size <= 0:
                self._dropped["camera_ingress_invalid"] += 1
                return
            if size > self.max_bytes:
                self._dropped["camera_ingress_oversize"] += 1
                return

            while self._queue and (
                len(self._queue) >= self.max_frames
                or self._queue_bytes + size > self.max_bytes
            ):
                _dropped_event, dropped_size = self._queue.popleft()
                self._queue_bytes -= dropped_size
                self._dropped["camera_ingress_coalesced"] += 1

            self._queue.append((event, size))
            self._queue_bytes += size
            self._peak_frames = max(self._peak_frames, len(self._queue))
            self._peak_bytes = max(self._peak_bytes, self._queue_bytes)
            self._condition.notify()

    def note_drop(self, name: str, count: int = 1) -> None:
        if count <= 0:
            return
        with self._condition:
            self._dropped[str(name)] += int(count)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop_requested:
                    self._condition.wait()
                if not self._queue and self._stop_requested:
                    return
                event, size = self._queue.popleft()
                self._queue_bytes -= size
                self._inflight = True
                self._inflight_timestamp_ns = event_monotonic_ns(event)

            failure: Optional[BaseException] = None
            try:
                self.manager.handle_camera_event(event)
            except BaseException as exc:
                failure = exc
            finally:
                # Latch interrupted status before publishing idle/fatal state.
                # Otherwise the state dispatcher could observe an empty camera
                # ingress and finalize the stopped episode as complete in the
                # few instructions before request_abort().
                if failure is not None:
                    self.manager.request_abort()
                with self._condition:
                    self._inflight = False
                    self._inflight_timestamp_ns = None
                    if failure is not None:
                        self.fatal_exception = failure
                        self._accepting = False
                        self._stop_requested = True
                        if self._queue:
                            self._dropped["camera_ingress_after_fatal"] += len(
                                self._queue
                            )
                            self._queue.clear()
                            self._queue_bytes = 0
                    self._condition.notify_all()

            if failure is not None:
                print(f"[recorder] fatal camera ingress error: {failure}")
                return

    def idle(self) -> bool:
        with self._condition:
            return not self._queue and not self._inflight

    def ready_through(self, deadline_ns: Optional[int]) -> bool:
        """Whether no pending camera frame belongs at/before ``deadline_ns``.

        During normal recording (no stop deadline) this retains the strict
        idle rule.  During post-roll, a continuously overloaded camera queue
        may contain only frames newer than the deadline; those frames must not
        prevent episode finalization forever and will become next-episode
        pre-roll after the atomic writer swap.
        """

        with self._condition:
            if deadline_ns is None:
                return not self._queue and not self._inflight
            deadline = int(deadline_ns)
            if (
                self._inflight_timestamp_ns is not None
                and self._inflight_timestamp_ns <= deadline
            ):
                return False
            return not any(
                event_monotonic_ns(event) <= deadline for event, _size in self._queue
            )

    def close(self) -> None:
        with self._condition:
            self._accepting = False
            self._stop_requested = True
            self._condition.notify_all()
        self.thread.join(timeout=self.join_timeout_s)
        if self.thread.is_alive():
            self.manager.request_abort()
            if self.fatal_exception is None:
                self.fatal_exception = TimeoutError(
                    "camera ingress did not drain within "
                    f"{self.join_timeout_s:.3f} s"
                )

    def counts(self) -> Dict[str, int]:
        with self._condition:
            return dict(self._received)

    def drops(self) -> Dict[str, int]:
        with self._condition:
            return dict(self._dropped)

    def size(self) -> int:
        with self._condition:
            return len(self._queue) + int(self._inflight)

    def peak_size(self) -> int:
        with self._condition:
            return self._peak_frames


def _message_receive_times() -> tuple[int, int]:
    return time.monotonic_ns(), time.time_ns()


def _joint_state_payload(message: Any) -> Dict[str, Any]:
    """Normalize AimDK and sensor_msgs joint states into the raw schema."""

    aimdk_joints = getattr(message, "joints", None)
    if aimdk_joints is not None:
        payload: Dict[str, Any] = {
            "name": [str(joint.name) for joint in aimdk_joints],
            "position": [float(joint.position) for joint in aimdk_joints],
            "velocity": [float(joint.velocity) for joint in aimdk_joints],
            "effort": [float(joint.effort) for joint in aimdk_joints],
            "error_code": [int(joint.error_code) for joint in aimdk_joints],
            "source_message_type": "aimdk_msgs/msg/JointStateArray",
        }
        domain_state = getattr(getattr(message, "state", None), "value", None)
        if domain_state is not None:
            payload["domain_state"] = int(domain_state)
        return payload

    return {
        "name": [str(name) for name in message.name],
        "position": [float(value) for value in message.position],
        "velocity": [float(value) for value in message.velocity],
        "effort": [float(value) for value in message.effort],
        "source_message_type": "sensor_msgs/msg/JointState",
    }


def _hand_status_payload(message: Any) -> Dict[str, Any]:
    """Normalize the authoritative high-level hand command status."""

    return {
        "sequence": int(message.sequence),
        "active": bool(message.active),
        "mode": int(message.mode),
        "left_grasp": float(message.left_grasp),
        "right_grasp": float(message.right_grasp),
        "source_message_type": "x1_protocol/msg/VrHandControlStatus",
    }


def _imu_stream_name(topic: str, index: int) -> str:
    topic_lower = str(topic).lower()
    if "torso" in topic_lower:
        return "imu_torso"
    if "chest" in topic_lower:
        return "imu_chest"
    return "imu_torso" if index == 0 else f"imu_{index}"


def _required_joint_topics(topics: list[str]) -> list[str]:
    required_groups = {"leg", "waist", "arm"}
    return [
        str(topic)
        for topic in topics
        if required_groups.intersection(part for part in str(topic).split("/") if part)
    ]


def _profile_topics(
    sensor_profile: str, record_profile: str
) -> tuple[list[str], list[str]]:
    """Return default ROS topics for a recording profile."""

    profile = SENSOR_PROFILES[sensor_profile]
    joint_topics = list(profile["joint_topics"])
    imu_topics = list(profile["imu_topics"])
    if record_profile == "groot_n17":
        # The controller telemetry tap carries one atomic state/reference
        # snapshot. Avoid duplicate high-rate ROS state subscriptions in this
        # low-overhead training profile.
        joint_topics = []
        imu_topics = []
    elif record_profile == "vla":
        joint_topics = _required_joint_topics(joint_topics)
        imu_topics = imu_topics[:1]
    return joint_topics, imu_topics


def _validate_shutdown_timeouts(
    camera_writer_join_timeout_s: float,
    dispatcher_join_timeout_s: float,
) -> None:
    """Ensure the outer watchdog cannot expire before camera finalization."""

    camera_timeout_s = float(camera_writer_join_timeout_s)
    dispatcher_timeout_s = float(dispatcher_join_timeout_s)
    if camera_timeout_s <= 0.0:
        raise ValueError("--camera_writer_join_timeout_s must be positive")
    if dispatcher_timeout_s <= 0.0:
        raise ValueError("--dispatcher_join_timeout_s must be positive")
    minimum_dispatcher_timeout_s = (
        camera_timeout_s + MIN_DISPATCHER_JOIN_MARGIN_S
    )
    if dispatcher_timeout_s <= minimum_dispatcher_timeout_s:
        raise ValueError(
            "--dispatcher_join_timeout_s must be greater than "
            "--camera_writer_join_timeout_s plus at least "
            f"{MIN_DISPATCHER_JOIN_MARGIN_S:.1f} s of finalization margin"
        )


def _build_ros_node(
    ingress: IngressQueue,
    camera_ingress: CameraIngressDispatcher,
    args: argparse.Namespace,
) -> Any:
    try:
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.node import Node
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import CompressedImage, Imu, JointState
    except ImportError as exc:
        raise ImportError(
            "ROS Python packages are unavailable. Source /opt/ros/humble/setup.bash before "
            "running the gmr virtualenv, or pass --disable_ros for a reference-only recording."
        ) from exc

    joint_message_type: Any = JointState
    if args.sensor_profile == "aimdk" and args.joint_topics:
        try:
            from aimdk_msgs.msg import JointStateArray
        except ImportError as exc:
            raise ImportError(
                "AimDK ROS messages are unavailable. Source the built "
                "x1_digit_mc/install/setup.bash after /opt/ros/humble/setup.bash, "
                "or use --sensor_profile compat when compatibility topics exist."
            ) from exc
        joint_message_type = JointStateArray

    hand_status_message_type: Any = None
    if args.hand_status_topic:
        try:
            from x1_protocol.msg import VrHandControlStatus
        except ImportError as exc:
            raise ImportError(
                "x1_protocol/VrHandControlStatus is unavailable. Source the built "
                "x1_digit_mc/install/setup.bash after /opt/ros/humble/setup.bash, "
                "or pass --hand_status_topic '' only for legacy recordings that "
                "do not need authoritative hand actions."
            ) from exc
        hand_status_message_type = VrHandControlStatus

    class X2SensorRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("x2_vr_data_recorder")
            # ``Node`` itself owns an internal ``_subscriptions`` collection.
            # Keep our Python references under a distinct name; shadowing that
            # attribute duplicates entries and makes ``destroy_node()`` fail.
            self._owned_subscriptions = []
            self._callback_group = ReentrantCallbackGroup()
            self._camera_receive_sequence = 0
            self._camera_sequence_lock = threading.Lock()

            reliability = (
                QoSReliabilityPolicy.RELIABLE
                if args.camera_qos_reliability == "reliable"
                else QoSReliabilityPolicy.BEST_EFFORT
            )
            camera_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=reliability,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            hand_status_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=max(1, int(args.hand_status_qos_depth)),
                reliability=(
                    QoSReliabilityPolicy.RELIABLE
                    if args.hand_status_qos_reliability == "reliable"
                    else QoSReliabilityPolicy.BEST_EFFORT
                ),
                durability=QoSDurabilityPolicy.VOLATILE,
            )

            if args.camera_topic and not args.camera_tap_addr:
                self._owned_subscriptions.append(
                    self.create_subscription(
                        CompressedImage,
                        args.camera_topic,
                        self._on_camera,
                        camera_qos,
                        callback_group=self._callback_group,
                    )
                )

            if args.hand_status_topic:
                self._owned_subscriptions.append(
                    self.create_subscription(
                        hand_status_message_type,
                        args.hand_status_topic,
                        self._on_hand_status,
                        hand_status_qos,
                        callback_group=self._callback_group,
                    )
                )

            for topic in args.joint_topics:
                self._owned_subscriptions.append(
                    self.create_subscription(
                        joint_message_type,
                        topic,
                        lambda message, source_topic=topic: self._on_joint_state(
                            source_topic, message
                        ),
                        qos_profile_sensor_data,
                        callback_group=self._callback_group,
                    )
                )

            for index, topic in enumerate(args.imu_topics):
                stream = _imu_stream_name(topic, index)
                self._owned_subscriptions.append(
                    self.create_subscription(
                        Imu,
                        topic,
                        lambda message, source_topic=topic, stream_name=stream: self._on_imu(
                            source_topic, stream_name, message
                        ),
                        qos_profile_sensor_data,
                        callback_group=self._callback_group,
                    )
                )

            camera_source = (
                f"<tcp tap {args.camera_tap_addr}>"
                if args.camera_tap_addr
                else (args.camera_topic or "<disabled>")
            )
            self.get_logger().info(
                f"camera={camera_source}, joint_topics={args.joint_topics}, "
                f"imu_topics={args.imu_topics}, "
                f"hand_status={args.hand_status_topic or '<disabled>'}"
            )

        def _base_event(self, stream: str, source_stamp_ns: int, frame_id: str) -> Dict[str, Any]:
            recv_monotonic_ns, recv_wall_time_ns = _message_receive_times()
            return {
                "stream": stream,
                "source_timestamp_ns": int(source_stamp_ns),
                "recorder_recv_monotonic_ns": recv_monotonic_ns,
                "recorder_recv_wall_time_ns": recv_wall_time_ns,
                "frame_id": str(frame_id),
            }

        def _on_camera(self, message: Any) -> None:
            with self._camera_sequence_lock:
                self._camera_receive_sequence += 1
                camera_receive_sequence = self._camera_receive_sequence
            event = self._base_event(
                "camera_head", ros_stamp_to_ns(message.header.stamp), message.header.frame_id
            )
            event.update(
                {
                    "topic": args.camera_topic,
                    "format": str(message.format),
                    "sequence": camera_receive_sequence,
                    "sequence_source": "recorder_ros_callback",
                    "source_message_type": "sensor_msgs/msg/CompressedImage",
                    "encoded_size_bytes": len(message.data),
                    "data": bytes(message.data),
                }
            )
            camera_ingress.put(event)

        def _on_joint_state(self, topic: str, message: Any) -> None:
            event = self._base_event(
                "joint_states", ros_stamp_to_ns(message.header.stamp), message.header.frame_id
            )
            event.update({"topic": topic, **_joint_state_payload(message)})
            ingress.put(event)

        def _on_hand_status(self, message: Any) -> None:
            event = self._base_event(
                "hand_command",
                ros_stamp_to_ns(message.header.stamp),
                message.header.frame_id,
            )
            event.update({"topic": args.hand_status_topic, **_hand_status_payload(message)})
            ingress.put(event)

        def _on_imu(self, topic: str, stream: str, message: Any) -> None:
            event = self._base_event(
                stream, ros_stamp_to_ns(message.header.stamp), message.header.frame_id
            )
            event.update(
                {
                    "topic": topic,
                    "orientation_xyzw": [
                        float(message.orientation.x),
                        float(message.orientation.y),
                        float(message.orientation.z),
                        float(message.orientation.w),
                    ],
                    "angular_velocity_xyz": [
                        float(message.angular_velocity.x),
                        float(message.angular_velocity.y),
                        float(message.angular_velocity.z),
                    ],
                    "linear_acceleration_xyz": [
                        float(message.linear_acceleration.x),
                        float(message.linear_acceleration.y),
                        float(message.linear_acceleration.z),
                    ],
                    "orientation_covariance": [
                        float(value) for value in message.orientation_covariance
                    ],
                    "angular_velocity_covariance": [
                        float(value) for value in message.angular_velocity_covariance
                    ],
                    "linear_acceleration_covariance": [
                        float(value) for value in message.linear_acceleration_covariance
                    ],
                }
            )
            ingress.put(event)

    return rclpy, X2SensorRecorderNode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record raw X2 VR teleoperation episodes")
    parser.add_argument("--tap_addr", default="tcp://127.0.0.1:28704")
    parser.add_argument(
        "--output_root",
        default="~/Datasets/x2_vr/raw",
        help="Raw episode root (kept outside the git repository by default)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Natural-language description assigned to each recorded episode",
    )
    parser.add_argument("--pre_roll_s", type=float, default=0.5)
    parser.add_argument("--post_roll_s", type=float, default=0.5)
    parser.add_argument("--queue_size", type=int, default=8192)
    parser.add_argument("--status_interval_s", type=float, default=2.0)
    parser.add_argument(
        "--record_profile",
        choices=RECORD_PROFILES,
        default="full",
        help=(
            "'groot_n17' records only controller boundaries, atomic C++ "
            "tracking telemetry, camera and hand command; 'vla' records "
            "controller/reference plus ROS state; 'full' also keeps XR, raw "
            "retarget, head state and chest IMU diagnostics"
        ),
    )
    parser.add_argument(
        "--camera_writer_queue_frames",
        type=int,
        default=24,
        help="Maximum pending JPEG frames in the independent episode writer",
    )
    parser.add_argument(
        "--camera_writer_queue_mib",
        type=float,
        default=64.0,
        help="Maximum pending JPEG payload memory in the independent episode writer",
    )
    parser.add_argument(
        "--camera_writer_join_timeout_s",
        type=float,
        default=DEFAULT_CAMERA_WRITER_JOIN_TIMEOUT_S,
        help="Maximum time allowed to drain the bounded camera writer at finalize",
    )
    parser.add_argument(
        "--dispatcher_join_timeout_s",
        type=float,
        default=DEFAULT_DISPATCHER_JOIN_TIMEOUT_S,
        help=(
            "Outer recorder shutdown watchdog; must exceed the camera writer "
            "timeout plus finalization margin"
        ),
    )
    parser.add_argument(
        "--hand_status_qos_depth",
        type=int,
        default=1,
        help=(
            "Keep-last DDS reader history for hand status. Depth 1 preserves "
            "latest-state semantics and prevents old commands accumulating."
        ),
    )
    parser.add_argument(
        "--hand_status_qos_reliability",
        choices=["reliable", "best_effort"],
        default="best_effort",
        help=(
            "Hand status subscription reliability. best_effort with depth 1 "
            "is the low-latency latest-state default."
        ),
    )
    parser.add_argument("--disable_ros", action="store_true")
    parser.add_argument(
        "--sensor_profile",
        choices=sorted(SENSOR_PROFILES),
        default="aimdk",
        help=(
            "ROS sensor transport: 'aimdk' subscribes directly to the real X2 HAL "
            "topics; 'compat' uses sensor_msgs compatibility relay topics"
        ),
    )
    parser.add_argument("--camera_topic", default=DEFAULT_CAMERA_TOPIC)
    parser.add_argument(
        "--hand_status_topic",
        default=DEFAULT_HAND_STATUS_TOPIC,
        help=(
            "Authoritative mapped hand-command status topic. Pass an empty string "
            "only for legacy/reference-only recording without hand actions."
        ),
    )
    parser.add_argument(
        "--camera_tap_addr",
        default="",
        help=(
            "Optional robot-side compressed-camera TCP tap (tcp://host:port). "
            "When set, the camera ROS subscription is skipped. Other ROS "
            "subscriptions are selected by --record_profile."
        ),
    )
    parser.add_argument(
        "--tracking_tap_addr",
        default=DEFAULT_TRACKING_TAP_ADDR,
        help=(
            "C++ tracking telemetry PUB endpoint used by --record_profile "
            "groot_n17 (multipart tracking_telemetry + JSON)."
        ),
    )
    parser.add_argument(
        "--provenance_file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Runtime artifact to hash into every manifest. The running recorder "
            "file is hashed automatically; groot_n17 additionally requires "
            "controller_binary, controller_config, controller_policy, "
            "controller_policy_data, teleop_bridge, gmr_config, gmr_runtime, and "
            "hand_config. Repeat this option."
        ),
    )
    bridge_runtime = parser.add_argument_group(
        "effective teleop bridge runtime provenance"
    )
    bridge_runtime.add_argument(
        "--bridge_actual_human_height",
        type=float,
        default=None,
        help="Exact --actual_human_height used by the running teleop bridge",
    )
    bridge_runtime.add_argument(
        "--bridge_gmr_max_iter",
        type=int,
        default=None,
        help="Exact --gmr_max_iter used by the running teleop bridge",
    )
    bridge_runtime.add_argument(
        "--bridge_lookback_ms",
        type=float,
        default=None,
        help="Exact --lookback_ms used by the running teleop bridge",
    )
    bridge_runtime.add_argument(
        "--bridge_min_link_height",
        type=float,
        default=None,
        help="Exact --min_link_height used by the running teleop bridge",
    )
    bridge_runtime.add_argument(
        "--bridge_min_link_height_align_strategy",
        choices=["startup_fixed", "per_frame"],
        default=None,
        help="Exact --min_link_height_align_strategy used by the running bridge",
    )
    bridge_runtime.add_argument(
        "--bridge_min_link_height_bootstrap_frames",
        type=int,
        default=None,
        help="Exact --min_link_height_bootstrap_frames used by the running bridge",
    )
    parser.add_argument(
        "--camera_qos_reliability",
        choices=["reliable", "best_effort"],
        default="best_effort",
        help="Camera subscription reliability (sensor streams normally use best_effort)",
    )
    parser.add_argument(
        "--joint_topics",
        nargs="+",
        default=None,
        help="Override the joint topics selected by --sensor_profile",
    )
    parser.add_argument(
        "--imu_topics",
        nargs="+",
        default=None,
        help="Override the IMU topics selected by --sensor_profile (torso first)",
    )
    args = parser.parse_args()
    default_joint_topics, default_imu_topics = _profile_topics(
        args.sensor_profile, args.record_profile
    )
    if args.joint_topics is None:
        args.joint_topics = default_joint_topics
    if args.imu_topics is None:
        args.imu_topics = default_imu_topics
    return args


def main() -> None:
    args = parse_args()
    if not args.task.strip():
        raise ValueError("--task must not be empty")
    if args.pre_roll_s < 0 or args.post_roll_s < 0:
        raise ValueError("pre/post-roll durations must be non-negative")
    if args.camera_writer_queue_frames < 1:
        raise ValueError("--camera_writer_queue_frames must be positive")
    if args.camera_writer_queue_mib <= 0.0:
        raise ValueError("--camera_writer_queue_mib must be positive")
    _validate_shutdown_timeouts(
        args.camera_writer_join_timeout_s,
        args.dispatcher_join_timeout_s,
    )
    if args.hand_status_qos_depth < 1:
        raise ValueError("--hand_status_qos_depth must be positive")
    if args.record_profile == "groot_n17" and not args.tracking_tap_addr:
        raise ValueError(
            "--tracking_tap_addr must not be empty for --record_profile groot_n17"
        )
    bridge_runtime_effective_params = _bridge_runtime_effective_params(args)
    capture_provenance = _hash_capture_provenance_files(args.provenance_file)
    if args.record_profile == "groot_n17":
        missing_provenance = sorted(
            REQUIRED_GROOT_PROVENANCE_FILES.difference(capture_provenance)
        )
        if missing_provenance:
            raise ValueError(
                "--record_profile groot_n17 requires --provenance_file for: "
                + ", ".join(missing_provenance)
            )

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq is required by the X2 VR recorder") from exc

    output_root = Path(args.output_root).expanduser().resolve()
    ingress = IngressQueue(args.queue_size)
    camera_enabled = bool(args.camera_tap_addr) or (
        not args.disable_ros and bool(args.camera_topic)
    )
    if args.record_profile == "groot_n17":
        # These are deliberately fixed rather than conditional on configured
        # transports. A missing camera/hand/telemetry source must make the raw
        # episode invalid instead of silently producing an incomplete dataset.
        required_streams = [
            "controller",
            TRACKING_TELEMETRY_TOPIC,
            "camera_head",
            "hand_command",
        ]
        minimum_stream_counts = {
            "controller": 2,
            TRACKING_TELEMETRY_TOPIC: 3,
            "camera_head": 2,
            "hand_command": 2,
        }
        max_stream_gap_s = {
            "controller": 1.0,
            TRACKING_TELEMETRY_TOPIC: 0.25,
            "camera_head": 0.5,
            "hand_command": 0.25,
        }
        minimum_matching_event_counts = [
            {
                "name": "hand_command_active",
                "stream": "hand_command",
                "field": "active",
                "equals": True,
                "minimum_count": 2,
            }
        ]
    else:
        required_streams = ["controller", "reference"]
        minimum_stream_counts = {"controller": 2, "reference": 2}
        max_stream_gap_s = {"controller": 1.0, "reference": 0.5}
        minimum_matching_event_counts = []
        if camera_enabled:
            required_streams.append("camera_head")
            minimum_stream_counts["camera_head"] = 2
            max_stream_gap_s["camera_head"] = 0.5
        if not args.disable_ros:
            required_streams.extend(["joint_states", "imu_torso"])
            minimum_stream_counts.update({"joint_states": 3, "imu_torso": 2})
            max_stream_gap_s["imu_torso"] = 0.5
            if args.hand_status_topic:
                required_streams.append("hand_command")
                minimum_stream_counts["hand_command"] = 2
                max_stream_gap_s["hand_command"] = 0.5
                minimum_matching_event_counts.append(
                    {
                        "name": "hand_command_active",
                        "stream": "hand_command",
                        "field": "active",
                        "equals": True,
                        "minimum_count": 2,
                    }
                )
    if args.record_profile == "groot_n17":
        tap_streams = list(GROOT_N17_TAP_STREAMS)
    elif args.record_profile == "vla":
        tap_streams = list(VLA_TAP_STREAMS)
    else:
        tap_streams = ["*"]
    source_config = {
        "hostname": socket.gethostname(),
        "record_profile": args.record_profile,
        "capture_provenance": capture_provenance,
        "bridge_runtime_effective_params": bridge_runtime_effective_params,
        "head_joint_assumption": (
            "fixed_not_recorded"
            if args.record_profile == "groot_n17"
            else None
        ),
        "tap_streams": tap_streams,
        "tap_sequence_check": (
            "per_topic_with_legacy_global_fallback"
            if args.record_profile == "full"
            else "per_topic"
        ),
        "tap_addr": args.tap_addr,
        "tracking_tap_addr": (
            args.tracking_tap_addr
            if args.record_profile == "groot_n17"
            else None
        ),
        "tracking_telemetry_schema_version": (
            TRACKING_TELEMETRY_SCHEMA_VERSION
            if args.record_profile == "groot_n17"
            else None
        ),
        "tracking_telemetry_delivery_semantics": (
            "bounded_nonblocking_sequence_checked"
            if args.record_profile == "groot_n17"
            else None
        ),
        "tracking_telemetry_reference_age_semantics": (
            "split_upstream_bridge_to_policy_with_legacy_total"
            if args.record_profile == "groot_n17"
            else None
        ),
        "tracking_telemetry_reference_diagnostics_schema_version": (
            REFERENCE_DIAGNOSTICS_SCHEMA_VERSION
            if args.record_profile == "groot_n17"
            else None
        ),
        "tracking_telemetry_reference_diagnostics_semantics": (
            "bridge_gmr_root_cause_v1"
            if args.record_profile == "groot_n17"
            else None
        ),
        "camera_topic": (
            None if args.disable_ros or args.camera_tap_addr else args.camera_topic
        ),
        "camera_tap_addr": args.camera_tap_addr or None,
        "camera_transport": (
            "tcp_tap"
            if args.camera_tap_addr
            else ("ros" if camera_enabled else "disabled")
        ),
        "camera_qos_reliability": args.camera_qos_reliability,
        "camera_writer_queue_frames": int(args.camera_writer_queue_frames),
        "camera_writer_queue_bytes": int(args.camera_writer_queue_mib * 1024 * 1024),
        "camera_ingress_queue_frames": int(args.camera_writer_queue_frames),
        "camera_ingress_queue_bytes": int(args.camera_writer_queue_mib * 1024 * 1024),
        "camera_ingress_join_timeout_s": float(args.dispatcher_join_timeout_s),
        "camera_writer_join_timeout_s": float(args.camera_writer_join_timeout_s),
        "dispatcher_join_timeout_s": float(args.dispatcher_join_timeout_s),
        "hand_status_qos_depth": int(args.hand_status_qos_depth),
        "hand_status_qos_reliability": args.hand_status_qos_reliability,
        "hand_status_delivery_semantics": (
            "latest_state" if int(args.hand_status_qos_depth) == 1 else "history"
        ),
        "hand_status_topic": (
            None if args.disable_ros or not args.hand_status_topic else args.hand_status_topic
        ),
        "hand_status_message_type": (
            None
            if args.disable_ros or not args.hand_status_topic
            else "x1_protocol/msg/VrHandControlStatus"
        ),
        "sensor_profile": args.sensor_profile,
        "joint_message_type": (
            SENSOR_PROFILES[args.sensor_profile]["joint_message_type"]
            if not args.disable_ros and args.joint_topics
            else None
        ),
        "joint_topics": [] if args.disable_ros else list(args.joint_topics),
        "imu_topics": [] if args.disable_ros else list(args.imu_topics),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", "0"),
        "required_streams": required_streams,
        "minimum_stream_counts": minimum_stream_counts,
        "minimum_matching_event_counts": minimum_matching_event_counts,
        "max_stream_gap_s": max_stream_gap_s,
        "required_joint_names": (
            list(X2_TRACKING_JOINT_NAMES)
            if not args.disable_ros and args.joint_topics
            else []
        ),
        "required_joint_topics": (
            [] if args.disable_ros else _required_joint_topics(args.joint_topics)
        ),
        "max_joint_topic_gap_s": (
            0.5 if not args.disable_ros and args.joint_topics else 0.0
        ),
    }
    camera_ingress: Optional[CameraIngressDispatcher] = None

    def combined_drop_counts() -> Dict[str, int]:
        combined: Counter[str] = Counter(ingress.drops())
        if camera_ingress is not None:
            combined.update(camera_ingress.drops())
        return dict(combined)

    manager = RawEpisodeManager(
        output_root=output_root,
        task=args.task,
        pre_roll_s=args.pre_roll_s,
        post_roll_s=args.post_roll_s,
        source_config=source_config,
        drop_counts=combined_drop_counts,
    )
    camera_ingress = CameraIngressDispatcher(
        manager,
        max_frames=args.camera_writer_queue_frames,
        max_bytes=int(args.camera_writer_queue_mib * 1024 * 1024),
        join_timeout_s=args.dispatcher_join_timeout_s,
    )
    camera_tap = (
        CameraTapClient(
            args.camera_tap_addr,
            on_event=camera_ingress.put,
            note_drop=camera_ingress.note_drop,
        )
        if args.camera_tap_addr
        else None
    )
    dispatcher = EventDispatcher(
        ingress,
        manager,
        join_timeout_s=args.dispatcher_join_timeout_s,
        auxiliary_ingress_ready=lambda: camera_ingress.ready_through(
            manager.finalize_deadline_ns
        ),
    )

    ros_api = None
    ros_node = None
    ros_executor = None
    ros_thread: Optional[threading.Thread] = None
    if not args.disable_ros:
        try:
            import rclpy as ros_api
        except ImportError as exc:
            raise ImportError(
                "ROS Python packages are unavailable. Source /opt/ros/humble/setup.bash before "
                "running the gmr virtualenv, or pass --disable_ros."
            ) from exc

        try:
            ros_api.init(args=None)
            _, ros_node = _build_ros_node(ingress, camera_ingress, args)
            from rclpy.executors import MultiThreadedExecutor

            ros_executor = MultiThreadedExecutor(num_threads=2)
            ros_executor.add_node(ros_node)
            ros_thread = threading.Thread(
                target=ros_executor.spin, name="x2-recorder-ros", daemon=True
            )
        except Exception:
            if ros_node is not None:
                ros_node.destroy_node()
            if ros_api.ok():
                ros_api.shutdown()
            raise

    context = zmq.Context.instance()
    tap_socket = context.socket(zmq.SUB)
    tap_socket.setsockopt(zmq.LINGER, 0)
    tap_socket.setsockopt(zmq.RCVHWM, max(100, args.queue_size))
    if args.record_profile in {"vla", "groot_n17"}:
        filtered_tap_streams = (
            VLA_TAP_STREAMS
            if args.record_profile == "vla"
            else GROOT_N17_TAP_STREAMS
        )
        for stream in filtered_tap_streams:
            tap_socket.setsockopt(zmq.SUBSCRIBE, stream.encode("utf-8"))
    else:
        tap_socket.setsockopt(zmq.SUBSCRIBE, b"")
    tap_socket.connect(args.tap_addr)

    tracking_socket = None
    if args.record_profile == "groot_n17":
        tracking_socket = context.socket(zmq.SUB)
        tracking_socket.setsockopt(zmq.LINGER, 0)
        # Telemetry is only 25 Hz, but bound both libzmq and public ingress so
        # a stalled recorder cannot create an old-state backlog.
        tracking_socket.setsockopt(zmq.RCVHWM, 8)
        tracking_socket.setsockopt(
            zmq.SUBSCRIBE, TRACKING_TELEMETRY_TOPIC.encode("ascii")
        )
        tracking_socket.connect(args.tracking_tap_addr)
    poller = zmq.Poller()
    poller.register(tap_socket, zmq.POLLIN)
    if tracking_socket is not None:
        poller.register(tracking_socket, zmq.POLLIN)

    dispatcher.start()
    camera_ingress.start()
    if ros_thread is not None:
        ros_thread.start()
    if camera_tap is not None:
        camera_tap.start()

    print(f"[recorder] output: {output_root}")
    print(
        f"[recorder] teleop tap: {args.tap_addr} | profile={args.record_profile} | "
        f"stored_tap_streams={source_config['tap_streams']}"
    )
    if tracking_socket is not None:
        print(
            f"[recorder] tracking telemetry: {args.tracking_tap_addr} "
            f"(schema={TRACKING_TELEMETRY_SCHEMA_VERSION}, expected=25 Hz)"
        )
    if camera_tap is not None:
        print(f"[recorder] camera tap: {args.camera_tap_addr} (ROS camera disabled)")
    print("[recorder] waiting: right key_one starts; left key_one stops and saves")
    if args.disable_ros:
        if args.record_profile == "groot_n17":
            stored_tap_streams = "controller/tracking_telemetry"
        elif args.record_profile == "vla":
            stored_tap_streams = "controller/reference"
        else:
            stored_tap_streams = "XR/retarget/reference/controller"
        if camera_tap is not None:
            print(
                f"[recorder] ROS disabled: recording {stored_tap_streams} and "
                "the camera TCP tap; joints/IMUs are disabled"
            )
        else:
            print(
                f"[recorder] ROS disabled: this run records {stored_tap_streams} only"
            )

    next_status_time = time.monotonic() + max(0.1, args.status_interval_s)
    tap_sequence_tracker = TapSequenceGapTracker(
        allow_legacy_global=args.record_profile == "full"
    )
    tracking_sequence_tracker = TrackingTelemetrySequenceTracker()
    writer_failed = False
    shutdown_requested = threading.Event()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: shutdown_requested.set())
    try:
        while not shutdown_requested.is_set():
            if (
                dispatcher.fatal_exception is not None
                or camera_ingress.fatal_exception is not None
            ):
                writer_failed = True
                break
            events = dict(poller.poll(timeout=100))
            if tap_socket in events:
                while True:
                    try:
                        parts = tap_socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(parts) != 2:
                        print(f"[recorder] invalid tap multipart frame count: {len(parts)}")
                        continue
                    topic_bytes, payload_bytes = parts
                    recv_monotonic_ns, recv_wall_time_ns = _message_receive_times()
                    try:
                        event = json.loads(payload_bytes.decode("utf-8"))
                        topic = topic_bytes.decode("utf-8")
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        print(f"[recorder] invalid tap event: {exc}")
                        continue
                    if int(event.get("tap_schema_version", -1)) != TELEOP_TAP_SCHEMA_VERSION:
                        print(
                            "[recorder] tap schema mismatch: "
                            f"expected {TELEOP_TAP_SCHEMA_VERSION}, got "
                            f"{event.get('tap_schema_version')}"
                        )
                        continue
                    event["stream"] = topic
                    event["recorder_recv_monotonic_ns"] = recv_monotonic_ns
                    event["recorder_recv_wall_time_ns"] = recv_wall_time_ns
                    for drop_name, gap in tap_sequence_tracker.observe(
                        topic, event
                    ).items():
                        ingress.note_drop(drop_name, gap)
                    if topic == "controller":
                        # Snapshot at controller receipt, not later when a
                        # backlogged state dispatcher finally processes A.
                        # Independent camera drops after the A timestamp then
                        # remain attributable to the new episode.
                        event["recorder_ingress_drop_snapshot"] = (
                            combined_drop_counts()
                        )
                    ingress.put(event)

            if tracking_socket is not None and tracking_socket in events:
                while True:
                    try:
                        parts = tracking_socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    recv_monotonic_ns, recv_wall_time_ns = _message_receive_times()
                    try:
                        event = parse_tracking_telemetry_parts(parts)
                    except TrackingTelemetryProtocolError as exc:
                        ingress.note_drop("tracking_telemetry_invalid", 1)
                        print(f"[recorder] invalid tracking telemetry: {exc}")
                        continue
                    try:
                        require_reference_diagnostics_contract(event)
                    except TrackingTelemetryReferenceDiagnosticsError as exc:
                        # The raw parser accepts older schema-v1 recordings,
                        # but new production capture must prove that its C++
                        # publisher and active bridge expose the diagnostic-v1
                        # contract. Diagnostic values themselves remain
                        # nullable and never alter the training freshness gate.
                        ingress.note_drop("tracking_telemetry_invalid", 1)
                        ingress.note_drop(exc.drop_reason, 1)
                        print(f"[recorder] invalid tracking telemetry: {exc}")
                        raise RuntimeError(
                            "groot_n17 tracking telemetry bridge/GMR diagnostics "
                            "contract is unavailable; deploy/restart both the "
                            "instrumented bridge and C++ controller before recording"
                        ) from exc
                    try:
                        require_reference_age_split(event)
                    except TrackingTelemetryReferenceAgeError as exc:
                        # Parsing remains backward compatible for offline old
                        # raw inspection, but production capture must never
                        # admit an ambiguous legacy total-only sample.
                        ingress.note_drop("tracking_telemetry_invalid", 1)
                        ingress.note_drop(exc.drop_reason, 1)
                        print(f"[recorder] invalid tracking telemetry: {exc}")
                        raise RuntimeError(
                            "groot_n17 tracking telemetry reference-age contract "
                            "is unavailable; rebuild/restart the C++ controller "
                            "before recording"
                        ) from exc
                    try:
                        gap = tracking_sequence_tracker.observe(event)
                    except TrackingTelemetryProtocolError as exc:
                        ingress.note_drop("tracking_telemetry_invalid", 1)
                        print(f"[recorder] invalid tracking telemetry: {exc}")
                        continue
                    event["recorder_recv_monotonic_ns"] = recv_monotonic_ns
                    event["recorder_recv_wall_time_ns"] = recv_wall_time_ns
                    if gap:
                        ingress.note_drop("tracking_telemetry_transport", gap)
                    ingress.put(event)

            now = time.monotonic()
            if now >= next_status_time:
                camera_status = camera_tap.status() if camera_tap is not None else None
                camera_text = (
                    f", camera_tap="
                    f"{'connected' if camera_status.connected else 'reconnecting'}"
                    f"/frames={camera_status.received_frames}"
                    f"/gaps={camera_status.transport_gaps}"
                    if camera_status is not None
                    else ""
                )
                print(
                    f"[recorder] state={manager.state}, episode={manager.current_episode_index}, "
                    f"queue={ingress.size()}/{args.queue_size} (peak={ingress.peak_size()}), "
                    f"camera_queue={camera_ingress.size()}/"
                    f"{args.camera_writer_queue_frames} "
                    f"(peak={camera_ingress.peak_size()}), "
                    f"received={ingress.counts()}, camera_received="
                    f"{camera_ingress.counts()}, dropped={combined_drop_counts()}"
                    f"{camera_text}"
                )
                next_status_time = now + max(0.1, args.status_interval_s)
    except KeyboardInterrupt:
        print("\n[recorder] stopping")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        poller.unregister(tap_socket)
        tap_socket.close(0)
        if tracking_socket is not None:
            poller.unregister(tracking_socket)
            tracking_socket.close(0)

        if camera_tap is not None:
            camera_tap.close()

        if ros_executor is not None:
            ros_executor.shutdown(timeout_sec=2.0)
        if ros_thread is not None:
            ros_thread.join(timeout=3.0)
        if ros_node is not None:
            ros_node.destroy_node()
        if ros_api is not None and ros_api.ok():
            ros_api.shutdown()

        camera_ingress.close()
        dispatcher.close()
        print(
            f"[recorder] final received={ingress.counts()}, "
            f"camera_received={camera_ingress.counts()}, "
            f"dropped={combined_drop_counts()}"
        )

    if (
        writer_failed
        or dispatcher.fatal_exception is not None
        or camera_ingress.fatal_exception is not None
    ):
        failure = dispatcher.fatal_exception or camera_ingress.fatal_exception
        raise RuntimeError(
            "Raw recorder writer failed; inspect the partial episode and error above"
        ) from failure


if __name__ == "__main__":
    main()
