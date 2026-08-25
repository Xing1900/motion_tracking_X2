"""Crash-tolerant raw episode writer used by the X2 VR recorder."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional

try:
    from .schema import RAW_DATASET_SCHEMA_VERSION, X2_TRACKING_JOINT_NAMES
except ImportError:  # Direct script execution from this directory.
    from schema import RAW_DATASET_SCHEMA_VERSION, X2_TRACKING_JOINT_NAMES


_EPISODE_RE = re.compile(r"^\.?episode_(\d{6})(?:\.partial)?$")
DEFAULT_CAMERA_WRITER_JOIN_TIMEOUT_S = 10.0


def event_monotonic_ns(event: Dict[str, Any]) -> int:
    """Return the recorder-host monotonic timestamp used for synchronization.

    Bridge monotonic clocks are retained in the raw event for diagnostics, but
    they are only comparable to ROS receive times when both processes run on
    the same host.  Recorder receipt time gives every stream one clock domain.
    """

    for key in (
        "recorder_recv_monotonic_ns",
        "bridge_sample_monotonic_ns",
        "bridge_recv_monotonic_ns",
        "bridge_enqueue_monotonic_ns",
    ):
        value = event.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return time.monotonic_ns()


def ros_stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RawEpisodeWriter:
    """Write one episode into a hidden partial directory, then atomically rename it."""

    def __init__(
        self,
        output_root: Path,
        episode_index: int,
        task: str,
        start_monotonic_ns: int,
        start_wall_time_ns: int,
        source_config: Dict[str, Any],
    ) -> None:
        self.output_root = output_root
        self.episode_index = int(episode_index)
        self.task = str(task)
        self.partial_dir = output_root / f".episode_{episode_index:06d}.partial"
        self.final_dir = output_root / f"episode_{episode_index:06d}"
        if self.partial_dir.exists() or self.final_dir.exists():
            raise FileExistsError(f"Episode {episode_index:06d} already exists")

        self.stream_dir = self.partial_dir / "streams"
        self.image_dir = self.partial_dir / "images" / "head_rgb"
        self.stream_dir.mkdir(parents=True)
        self.image_dir.mkdir(parents=True)

        self._handles: Dict[str, Any] = {}
        self._counts: Counter[str] = Counter()
        self._camera_index = 0
        self._closed = False
        self._last_checkpoint_ns = time.monotonic_ns()
        self._start_monotonic_ns = int(start_monotonic_ns)
        self._validation_stop_ns: Optional[int] = None
        self._active_stream_stats: Dict[str, Dict[str, int]] = {}
        self._active_joint_names: set[str] = set()
        self._active_joint_topic_stats: Dict[str, Dict[str, int]] = {}
        effective_source_config = dict(source_config)
        self._camera_queue_max_frames = int(
            effective_source_config.setdefault("camera_writer_queue_frames", 24)
        )
        self._camera_queue_max_bytes = int(
            effective_source_config.setdefault(
                "camera_writer_queue_bytes", 64 * 1024 * 1024
            )
        )
        self._camera_join_timeout_s = float(
            effective_source_config.setdefault(
                "camera_writer_join_timeout_s",
                DEFAULT_CAMERA_WRITER_JOIN_TIMEOUT_S,
            )
        )
        if self._camera_queue_max_frames < 1:
            raise ValueError("camera_writer_queue_frames must be positive")
        if self._camera_queue_max_bytes < 1:
            raise ValueError("camera_writer_queue_bytes must be positive")
        if self._camera_join_timeout_s <= 0.0:
            raise ValueError("camera_writer_join_timeout_s must be positive")

        self._camera_condition = threading.Condition()
        self._camera_queue: Deque[tuple[Dict[str, Any], int, bytes]] = deque()
        self._camera_queue_bytes = 0
        self._camera_accepting = True
        self._camera_stop_requested = False
        self._camera_fatal_exception: Optional[BaseException] = None
        self._camera_drop_counts: Counter[str] = Counter()
        self._persisted_camera_events: list[tuple[int, Dict[str, Any]]] = []
        self._camera_active_stats_finalized = False
        self._camera_thread = threading.Thread(
            target=self._camera_worker_main,
            name=f"x2-camera-writer-{self.episode_index:06d}",
            # A permanently blocked filesystem call must not keep the recorder
            # process alive after the bounded finalize join has failed.  The
            # hidden .partial directory remains the recovery boundary.
            daemon=True,
        )

        self._matching_event_requirements = self._normalize_matching_event_requirements(
            effective_source_config.get("minimum_matching_event_counts", [])
        )
        self._active_matching_event_counts: Counter[str] = Counter()
        self._manifest: Dict[str, Any] = {
            "schema_version": RAW_DATASET_SCHEMA_VERSION,
            "robot_type": "agibot_x2",
            "episode_index": self.episode_index,
            "task": self.task,
            "status": "recording",
            "success": None,
            "joint_order": X2_TRACKING_JOINT_NAMES,
            "recording": {
                "start_monotonic_ns": int(start_monotonic_ns),
                "start_wall_time_ns": int(start_wall_time_ns),
                "stop_trigger_monotonic_ns": None,
                "finalized_monotonic_ns": None,
                "termination": None,
                "pre_roll_s": source_config.get("pre_roll_s"),
                "post_roll_s": source_config.get("post_roll_s"),
            },
            "source_config": effective_source_config,
            "stream_counts": {},
            "ingress_drops": {},
        }
        _write_json(self.partial_dir / "manifest.json", self._manifest)
        self._camera_thread.start()

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _normalize_matching_event_requirements(
        raw_requirements: Any,
    ) -> list[Dict[str, Any]]:
        if not isinstance(raw_requirements, list):
            raise ValueError("minimum_matching_event_counts must be a list")
        requirements: list[Dict[str, Any]] = []
        names: set[str] = set()
        for index, raw_requirement in enumerate(raw_requirements):
            if not isinstance(raw_requirement, dict):
                raise ValueError(
                    f"minimum_matching_event_counts[{index}] must be an object"
                )
            stream = str(raw_requirement.get("stream", "")).strip()
            field = str(raw_requirement.get("field", "")).strip()
            if not stream or not field or "equals" not in raw_requirement:
                raise ValueError(
                    "matching-event requirements need stream, field, and equals"
                )
            minimum_count = int(raw_requirement.get("minimum_count", 1))
            if minimum_count < 1:
                raise ValueError("matching-event minimum_count must be positive")
            name = str(
                raw_requirement.get("name") or f"{stream}.{field}.equals"
            ).strip()
            if not name or name in names:
                raise ValueError(f"duplicate or empty matching-event requirement: {name!r}")
            names.add(name)
            requirements.append(
                {
                    "name": name,
                    "stream": stream,
                    "field": field,
                    "equals": raw_requirement["equals"],
                    "minimum_count": minimum_count,
                }
            )
        return requirements

    @staticmethod
    def _event_field(event: Dict[str, Any], field: str) -> tuple[bool, Any]:
        value: Any = event
        for component in field.split("."):
            if not isinstance(value, dict) or component not in value:
                return False, None
            value = value[component]
        return True, value

    @staticmethod
    def _field_value_matches(actual: Any, expected: Any) -> bool:
        # Python considers True == 1; validation predicates should not.
        if isinstance(expected, bool):
            return actual is expected
        return actual == expected

    def _stream_handle(self, stream: str) -> Any:
        safe_stream = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stream)
        handle = self._handles.get(safe_stream)
        if handle is None:
            handle = (self.stream_dir / f"{safe_stream}.jsonl").open(
                "a", encoding="utf-8", buffering=1024 * 1024
            )
            self._handles[safe_stream] = handle
        return handle

    @staticmethod
    def _update_timing_stats(
        stats_by_key: Dict[str, Dict[str, int]], key: str, timestamp_ns: int
    ) -> None:
        stats = stats_by_key.get(key)
        if stats is None:
            stats_by_key[key] = {
                "count": 1,
                "first_ns": int(timestamp_ns),
                "last_ns": int(timestamp_ns),
                "max_gap_ns": 0,
            }
            return
        stats["count"] += 1
        stats["first_ns"] = min(stats["first_ns"], int(timestamp_ns))
        if timestamp_ns >= stats["last_ns"]:
            stats["max_gap_ns"] = max(
                stats["max_gap_ns"], int(timestamp_ns) - stats["last_ns"]
            )
            stats["last_ns"] = int(timestamp_ns)

    def _track_active_event(
        self, stream: str, event: Dict[str, Any], timestamp_ns: int
    ) -> None:
        if timestamp_ns < self._start_monotonic_ns:
            return
        if (
            self._validation_stop_ns is not None
            and timestamp_ns > self._validation_stop_ns
        ):
            return
        self._update_timing_stats(self._active_stream_stats, stream, timestamp_ns)
        for requirement in self._matching_event_requirements:
            if requirement["stream"] != stream:
                continue
            field_found, actual = self._event_field(event, requirement["field"])
            if field_found and self._field_value_matches(actual, requirement["equals"]):
                self._active_matching_event_counts[requirement["name"]] += 1
        if stream != "joint_states":
            return

        topic = str(event.get("topic", ""))
        if topic:
            self._update_timing_stats(self._active_joint_topic_stats, topic, timestamp_ns)
        names = event.get("name", [])
        positions = event.get("position", [])
        if isinstance(names, list) and isinstance(positions, list):
            for index, name in enumerate(names):
                if index < len(positions):
                    self._active_joint_names.add(str(name))

    def mark_stop_trigger(self, timestamp_ns: int) -> None:
        self._raise_if_camera_writer_failed()
        if self._validation_stop_ns is None:
            self._validation_stop_ns = int(timestamp_ns)

    def _camera_writer_failure(self) -> Optional[BaseException]:
        with self._camera_condition:
            return self._camera_fatal_exception

    def _raise_if_camera_writer_failed(self) -> None:
        failure = self._camera_writer_failure()
        if failure is not None:
            raise RuntimeError("asynchronous camera writer failed") from failure

    def _enqueue_camera_event(
        self,
        serializable: Dict[str, Any],
        timestamp_ns: int,
        image_data: bytes,
    ) -> None:
        image_size = len(image_data)
        with self._camera_condition:
            if self._camera_fatal_exception is not None:
                raise RuntimeError("asynchronous camera writer failed") from (
                    self._camera_fatal_exception
                )
            if not self._camera_accepting:
                raise RuntimeError("camera writer is no longer accepting events")
            if image_size > self._camera_queue_max_bytes:
                self._camera_drop_counts["camera_writer_oversize"] += 1
                return

            while self._camera_queue and (
                len(self._camera_queue) >= self._camera_queue_max_frames
                or self._camera_queue_bytes + image_size > self._camera_queue_max_bytes
            ):
                _, _, dropped_data = self._camera_queue.popleft()
                self._camera_queue_bytes -= len(dropped_data)
                self._camera_drop_counts["camera_writer_coalesced"] += 1

            self._camera_queue.append((serializable, int(timestamp_ns), image_data))
            self._camera_queue_bytes += image_size
            self._camera_condition.notify()

    def _persist_camera_event(
        self,
        handle: Any,
        serializable: Dict[str, Any],
        timestamp_ns: int,
        image_data: bytes,
    ) -> None:
        image_format = str(serializable.get("format", "jpeg")).lower()
        suffix = ".png" if "png" in image_format else ".jpg"
        filename = f"{self._camera_index:08d}_{timestamp_ns}{suffix}"
        image_path = self.image_dir / filename
        with image_path.open("wb") as image_handle:
            image_handle.write(image_data)

        serializable["image_path"] = str(image_path.relative_to(self.partial_dir))
        serializable["encoded_size_bytes"] = len(image_data)
        handle.write(
            json.dumps(
                serializable,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
        )
        handle.write("\n")

        # Counts and active-window accounting must describe frames that really
        # made it through both image and metadata writes, not merely enqueues.
        self._camera_index += 1
        self._counts["camera_head"] += 1
        self._persisted_camera_events.append((int(timestamp_ns), serializable))

    def _camera_worker_main(self) -> None:
        camera_path = self.stream_dir / "camera_head.jsonl"
        handle: Optional[Any] = None
        failure: Optional[BaseException] = None
        last_flush_ns = time.monotonic_ns()
        try:
            while True:
                with self._camera_condition:
                    while not self._camera_queue and not self._camera_stop_requested:
                        self._camera_condition.wait()
                    if not self._camera_queue and self._camera_stop_requested:
                        break
                    serializable, timestamp_ns, image_data = self._camera_queue.popleft()
                    self._camera_queue_bytes -= len(image_data)

                if handle is None:
                    handle = camera_path.open(
                        "a", encoding="utf-8", buffering=1024 * 1024
                    )
                self._persist_camera_event(
                    handle, serializable, timestamp_ns, image_data
                )
                now_ns = time.monotonic_ns()
                if now_ns - last_flush_ns >= 1_000_000_000:
                    handle.flush()
                    last_flush_ns = now_ns
        except BaseException as exc:
            failure = exc
        finally:
            if handle is not None:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                finally:
                    try:
                        handle.close()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc

            with self._camera_condition:
                if failure is not None and self._camera_fatal_exception is None:
                    self._camera_fatal_exception = failure
                if failure is not None and self._camera_queue:
                    self._camera_drop_counts["camera_writer_after_fatal"] += len(
                        self._camera_queue
                    )
                    self._camera_queue.clear()
                    self._camera_queue_bytes = 0
                self._camera_accepting = False
                self._camera_condition.notify_all()

    def _stop_camera_writer(self) -> Optional[BaseException]:
        with self._camera_condition:
            self._camera_accepting = False
            self._camera_stop_requested = True
            self._camera_condition.notify_all()
        self._camera_thread.join(timeout=self._camera_join_timeout_s)
        if self._camera_thread.is_alive():
            timeout_error = TimeoutError(
                "camera writer did not drain within "
                f"{self._camera_join_timeout_s:.3f} s"
            )
            with self._camera_condition:
                if self._camera_fatal_exception is None:
                    self._camera_fatal_exception = timeout_error
            raise timeout_error
        return self._camera_writer_failure()

    def _finalize_camera_active_stats(self) -> None:
        if self._camera_active_stats_finalized:
            return
        for timestamp_ns, event in self._persisted_camera_events:
            self._track_active_event("camera_head", event, timestamp_ns)
        self._camera_active_stats_finalized = True
        self._persisted_camera_events.clear()

    def write_event(self, event: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed episode")
        self._raise_if_camera_writer_failed()
        stream = str(event.get("stream", "unknown"))
        serializable = dict(event)
        timestamp_ns = event_monotonic_ns(serializable)

        if stream == "camera_head":
            image_data = serializable.pop("data", None)
            if not isinstance(image_data, (bytes, bytearray, memoryview)):
                return
            self._enqueue_camera_event(
                serializable, timestamp_ns, bytes(image_data)
            )
            return

        self._track_active_event(stream, serializable, timestamp_ns)

        handle = self._stream_handle(stream)
        handle.write(
            json.dumps(
                serializable,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
        )
        handle.write("\n")
        self._counts[stream] += 1

        now_ns = time.monotonic_ns()
        if now_ns - self._last_checkpoint_ns >= 1_000_000_000:
            # Keep at most roughly one second in Python userspace buffers.  A
            # clean finalize still fsyncs every stream before the atomic rename.
            for stream_handle in self._handles.values():
                stream_handle.flush()
            self._last_checkpoint_ns = now_ns

    def finalize(
        self,
        *,
        status: str,
        stop_trigger_monotonic_ns: Optional[int],
        success: Optional[bool],
        ingress_drops: Dict[str, int],
        termination: Optional[Dict[str, Any]] = None,
        abort_requested: Optional[Callable[[], bool]] = None,
    ) -> Path:
        if self._closed:
            return self.final_dir

        if stop_trigger_monotonic_ns is not None and self._validation_stop_ns is None:
            self._validation_stop_ns = int(stop_trigger_monotonic_ns)
        camera_failure = self._stop_camera_writer()
        abort_was_requested = (
            abort_requested is not None and abort_requested()
        )
        propagate_camera_failure = (
            camera_failure is not None and not abort_was_requested
        )
        self._finalize_camera_active_stats()

        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._handles.clear()

        if camera_failure is not None:
            status = "interrupted"
            success = None
            self._manifest["camera_writer_error"] = {
                "type": type(camera_failure).__name__,
                "message": str(camera_failure),
            }
        if abort_was_requested or (
            abort_requested is not None and abort_requested()
        ):
            status = "interrupted"
            success = None

        self._manifest["status"] = str(status)
        self._manifest["success"] = success
        self._manifest["recording"]["termination"] = (
            None if termination is None else dict(termination)
        )
        self._manifest["stream_counts"] = dict(sorted(self._counts.items()))
        merged_drops: Counter[str] = Counter(
            {
                str(stream): int(count)
                for stream, count in ingress_drops.items()
                if int(count) > 0
            }
        )
        merged_drops.update(self._camera_drop_counts)
        self._manifest["ingress_drops"] = dict(sorted(merged_drops.items()))
        required_streams = [
            str(stream) for stream in self._manifest["source_config"].get("required_streams", [])
        ]
        minimum_stream_counts = {
            str(stream): int(count)
            for stream, count in self._manifest["source_config"]
            .get("minimum_stream_counts", {})
            .items()
        }
        missing_streams = [
            stream
            for stream in required_streams
            if self._active_stream_stats.get(stream, {}).get("count", 0)
            < minimum_stream_counts.get(stream, 1)
        ]
        validation_end_ns = (
            int(stop_trigger_monotonic_ns)
            if stop_trigger_monotonic_ns is not None
            else time.monotonic_ns()
        )
        max_stream_gap_s = {
            str(stream): float(seconds)
            for stream, seconds in self._manifest["source_config"]
            .get("max_stream_gap_s", {})
            .items()
        }
        stream_max_gaps_s: Dict[str, float] = {}
        stale_streams: list[str] = []
        for stream, limit_s in max_stream_gap_s.items():
            stats = self._active_stream_stats.get(stream)
            if stats is None:
                continue
            max_gap_ns = max(
                stats["max_gap_ns"],
                max(0, stats["first_ns"] - self._start_monotonic_ns),
                max(0, validation_end_ns - stats["last_ns"]),
            )
            stream_max_gaps_s[stream] = max_gap_ns / 1e9
            if max_gap_ns > int(limit_s * 1e9):
                stale_streams.append(stream)

        required_joint_names = {
            str(name)
            for name in self._manifest["source_config"].get("required_joint_names", [])
        }
        missing_joint_names = sorted(required_joint_names - self._active_joint_names)
        required_joint_topics = [
            str(topic)
            for topic in self._manifest["source_config"].get("required_joint_topics", [])
        ]
        missing_joint_topics = [
            topic for topic in required_joint_topics if topic not in self._active_joint_topic_stats
        ]
        joint_topic_max_gap_s = float(
            self._manifest["source_config"].get("max_joint_topic_gap_s", 0.0)
        )
        stale_joint_topics: list[str] = []
        joint_topic_gaps_s: Dict[str, float] = {}
        if joint_topic_max_gap_s > 0.0:
            for topic in required_joint_topics:
                stats = self._active_joint_topic_stats.get(topic)
                if stats is None:
                    continue
                max_gap_ns = max(
                    stats["max_gap_ns"],
                    max(0, stats["first_ns"] - self._start_monotonic_ns),
                    max(0, validation_end_ns - stats["last_ns"]),
                )
                joint_topic_gaps_s[topic] = max_gap_ns / 1e9
                if max_gap_ns > int(joint_topic_max_gap_s * 1e9):
                    stale_joint_topics.append(topic)

        matching_event_requirements = []
        failed_matching_event_requirements: list[str] = []
        for requirement in self._matching_event_requirements:
            actual_count = int(
                self._active_matching_event_counts.get(requirement["name"], 0)
            )
            passed = actual_count >= requirement["minimum_count"]
            matching_event_requirements.append(
                {
                    **requirement,
                    "actual_count": actual_count,
                    "passed": passed,
                }
            )
            if not passed:
                failed_matching_event_requirements.append(requirement["name"])

        explicit_invalid_reasons: list[str] = []
        if status == "invalid":
            termination_reason = (
                str(termination.get("reason", "")).strip()
                if isinstance(termination, dict)
                else ""
            )
            explicit_invalid_reasons.append(
                termination_reason or "explicit_invalid_status"
            )
        validation_valid = not any(
            (
                missing_streams,
                stale_streams,
                missing_joint_names,
                missing_joint_topics,
                stale_joint_topics,
                failed_matching_event_requirements,
                explicit_invalid_reasons,
            )
        )
        self._manifest["validation"] = {
            "required_streams": required_streams,
            "missing_streams": missing_streams,
            "active_stream_counts": {
                stream: stats["count"]
                for stream, stats in sorted(self._active_stream_stats.items())
            },
            "stream_max_gaps_s": stream_max_gaps_s,
            "stale_streams": sorted(stale_streams),
            "missing_joint_names": missing_joint_names,
            "missing_joint_topics": missing_joint_topics,
            "joint_topic_max_gaps_s": joint_topic_gaps_s,
            "stale_joint_topics": sorted(stale_joint_topics),
            "matching_event_requirements": matching_event_requirements,
            "failed_matching_event_requirements": failed_matching_event_requirements,
            "explicit_invalid_reasons": explicit_invalid_reasons,
            "valid": validation_valid,
        }
        if status == "complete" and not validation_valid:
            self._manifest["status"] = "invalid"
        self._manifest["recording"]["stop_trigger_monotonic_ns"] = (
            None if stop_trigger_monotonic_ns is None else int(stop_trigger_monotonic_ns)
        )
        self._manifest["recording"]["finalized_monotonic_ns"] = time.monotonic_ns()
        self._manifest["recording"]["finalized_wall_time_ns"] = time.time_ns()
        _write_json(self.partial_dir / "manifest.json", self._manifest)

        # The manifest write itself performs fsync and can block long enough
        # for the dispatcher watchdog to request abort. Re-check immediately
        # before exposing the directory; if needed, atomically rewrite the
        # manifest as interrupted first.
        if (
            abort_requested is not None
            and abort_requested()
            and self._manifest["status"] != "interrupted"
        ):
            self._manifest["status"] = "interrupted"
            self._manifest["success"] = None
            _write_json(self.partial_dir / "manifest.json", self._manifest)

        self.partial_dir.replace(self.final_dir)
        self._closed = True
        if propagate_camera_failure:
            raise RuntimeError(
                "asynchronous camera writer failed; episode saved as interrupted"
            ) from camera_failure
        return self.final_dir


class RawEpisodeManager:
    """Turn controller-button edges and asynchronous sensor events into episodes."""

    def __init__(
        self,
        output_root: Path,
        task: str,
        pre_roll_s: float,
        post_roll_s: float,
        source_config: Dict[str, Any],
        drop_counts: Callable[[], Dict[str, int]],
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.task = str(task)
        self.pre_roll_ns = max(0, int(float(pre_roll_s) * 1e9))
        self.post_roll_ns = max(0, int(float(post_roll_s) * 1e9))
        self.source_config = {
            **source_config,
            "pre_roll_s": float(pre_roll_s),
            "post_roll_s": float(post_roll_s),
        }
        self.drop_counts = drop_counts

        required_streams = {
            str(stream)
            for stream in self.source_config.get("required_streams", [])
        }
        start_max_age_s = self.source_config.get("camera_start_max_age_s")
        stall_timeout_s = self.source_config.get("camera_stall_timeout_s")
        # Old/raw diagnostic callers do not carry these keys.  Preserve their
        # historical behavior unless both guards are explicitly configured.
        self._camera_guard_enabled = (
            "camera_head" in required_streams
            and start_max_age_s is not None
            and stall_timeout_s is not None
        )
        self._camera_start_max_age_ns = 0
        self._camera_stall_timeout_ns = 0
        if self._camera_guard_enabled:
            if float(start_max_age_s) <= 0.0:
                raise ValueError("camera_start_max_age_s must be positive")
            if float(stall_timeout_s) <= 0.0:
                raise ValueError("camera_stall_timeout_s must be positive")
            self._camera_start_max_age_ns = int(float(start_max_age_s) * 1e9)
            self._camera_stall_timeout_ns = int(float(stall_timeout_s) * 1e9)

        self._pre_roll: Deque[Dict[str, Any]] = deque()
        # Camera routing has its own lock/pre-roll so high-rate synchronous
        # state JSON writes under _lock cannot delay the camera ingress worker.
        # Episode start/finalize acquire _lock then _camera_route_lock; the
        # camera path never acquires _lock, so the order cannot deadlock.
        self._camera_pre_roll: Deque[Dict[str, Any]] = deque()
        self._camera_route_lock = threading.Lock()
        # Protected by _camera_route_lock.  This is deliberately recorder
        # receive time, not a camera/source clock that may regress or jump.
        self._last_camera_recv_monotonic_ns: Optional[int] = None
        self._writer: Optional[RawEpisodeWriter] = None
        self._next_episode_index = self._find_next_episode_index()
        self._prev_start = False
        self._prev_stop = False
        self._stop_trigger_ns: Optional[int] = None
        self._finalize_deadline_ns: Optional[int] = None
        self._termination: Optional[Dict[str, Any]] = None
        self._drop_baseline: Dict[str, int] = {}
        self._lock = threading.Lock()
        # A supervisor must be able to latch failure even while the writer
        # thread is blocked in disk I/O under _lock.
        self._abort_requested = threading.Event()

    @property
    def state(self) -> str:
        with self._lock:
            if self._writer is None:
                return "waiting"
            if self._finalize_deadline_ns is not None:
                return "post_roll"
            return "recording"

    @property
    def current_episode_index(self) -> Optional[int]:
        with self._lock:
            return None if self._writer is None else self._writer.episode_index

    @property
    def finalize_deadline_ns(self) -> Optional[int]:
        """Post-roll deadline used by independent ingress watermarks."""

        with self._lock:
            return self._finalize_deadline_ns

    def _find_next_episode_index(self) -> int:
        indices = []
        for child in self.output_root.iterdir():
            match = _EPISODE_RE.match(child.name)
            if match:
                indices.append(int(match.group(1)))
        return 0 if not indices else max(indices) + 1

    def _evict_pre_roll(self, now_ns: int) -> None:
        cutoff_ns = int(now_ns) - self.pre_roll_ns
        # Receive order is close enough to timestamp order for cheap ongoing
        # eviction.  _start_episode performs the authoritative full
        # filter/sort, so a slightly out-of-order event cannot enter an episode.
        while self._pre_roll and event_monotonic_ns(self._pre_roll[0]) < cutoff_ns:
            self._pre_roll.popleft()

    def _evict_camera_pre_roll(self, now_ns: int) -> None:
        cutoff_ns = int(now_ns) - self.pre_roll_ns
        while (
            self._camera_pre_roll
            and event_monotonic_ns(self._camera_pre_roll[0]) < cutoff_ns
        ):
            self._camera_pre_roll.popleft()

    @staticmethod
    def _button_state(event: Dict[str, Any], name: str) -> bool:
        buttons = event.get("controller_buttons")
        return bool(buttons.get(name, False)) if isinstance(buttons, dict) else False

    def _start_episode(self, trigger: Dict[str, Any], now_ns: int) -> None:
        start_wall_time_ns = int(
            trigger.get("recorder_recv_wall_time_ns")
            or trigger.get("bridge_recv_wall_time_ns")
            or time.time_ns()
        )
        trigger_drop_snapshot = trigger.get("recorder_ingress_drop_snapshot")
        if isinstance(trigger_drop_snapshot, dict):
            self._drop_baseline = {
                str(name): max(0, int(count))
                for name, count in trigger_drop_snapshot.items()
            }
        else:
            self._drop_baseline = self.drop_counts()
        new_writer = RawEpisodeWriter(
            output_root=self.output_root,
            episode_index=self._next_episode_index,
            task=self.task,
            start_monotonic_ns=now_ns,
            start_wall_time_ns=start_wall_time_ns,
            source_config=self.source_config,
        )
        self._next_episode_index += 1
        self._termination = None
        cutoff_ns = now_ns - self.pre_roll_ns
        buffered_events = sorted(
            (
                event
                for event in self._pre_roll
                if cutoff_ns <= event_monotonic_ns(event)
            ),
            key=event_monotonic_ns,
        )
        # The independent camera dispatcher can route a frame received after
        # this A edge before the state dispatcher reaches the edge.  Such a
        # frame is already buffered with timestamp > now_ns and belongs to the
        # new episode; retaining it avoids an A-boundary camera hole.
        with self._camera_route_lock:
            buffered_camera_events = sorted(
                (
                    event
                    for event in self._camera_pre_roll
                    if cutoff_ns <= event_monotonic_ns(event)
                ),
                key=event_monotonic_ns,
            )
            self._camera_pre_roll.clear()
            self._writer = new_writer
            for buffered_camera_event in buffered_camera_events:
                new_writer.write_event(buffered_camera_event)
        for buffered_event in buffered_events:
            new_writer.write_event(buffered_event)
        self._pre_roll.clear()
        print(f"[recorder] episode {new_writer.episode_index:06d} started | task={self.task!r}")

    def _camera_start_rejection(self, now_ns: int) -> Optional[str]:
        """Return a human-readable reason when a required camera is not fresh."""

        if not self._camera_guard_enabled:
            return None
        with self._camera_route_lock:
            last_camera_ns = self._last_camera_recv_monotonic_ns
        if last_camera_ns is None:
            return "no camera frame has reached the recorder"
        age_ns = max(0, int(now_ns) - int(last_camera_ns))
        if age_ns <= self._camera_start_max_age_ns:
            return None
        return (
            f"latest camera frame is {age_ns / 1e9:.3f}s old "
            f"(limit={self._camera_start_max_age_ns / 1e9:.3f}s)"
        )

    def _camera_stall_evidence(self, now_ns: int) -> Optional[Dict[str, Any]]:
        """Describe an active-recording camera stall, if one has occurred.

        The caller owns ``_lock``.  It only takes ``_camera_route_lock`` in the
        existing manager order; the camera ingress path never takes ``_lock``.
        """

        if (
            not self._camera_guard_enabled
            or self._writer is None
            or self._finalize_deadline_ns is not None
        ):
            return None
        with self._camera_route_lock:
            last_camera_ns = self._last_camera_recv_monotonic_ns
        if last_camera_ns is None:
            return None
        age_ns = max(0, int(now_ns) - int(last_camera_ns))
        if age_ns <= self._camera_stall_timeout_ns:
            return None
        return {
            "reason": "camera_stall",
            "detected_monotonic_ns": int(now_ns),
            "last_camera_recv_monotonic_ns": int(last_camera_ns),
            "camera_age_s": age_ns / 1e9,
            "stall_timeout_s": self._camera_stall_timeout_ns / 1e9,
        }

    def _finish_camera_stall_if_needed(self, now_ns: int) -> Optional[Path]:
        evidence = self._camera_stall_evidence(now_ns)
        if evidence is None:
            return None
        assert self._writer is not None
        self._stop_trigger_ns = int(now_ns)
        self._termination = evidence
        self._writer.mark_stop_trigger(int(now_ns))
        print(
            "[recorder] camera stalled for "
            f"{evidence['camera_age_s']:.3f}s; saving episode as invalid"
        )
        return self._finish_episode(status="invalid", success=None)

    def _finish_episode(self, status: str, success: Optional[bool]) -> Optional[Path]:
        if self._writer is None:
            return None
        if self._abort_requested.is_set():
            status = "interrupted"
            success = None
        current_drops = self.drop_counts()
        episode_drops = {
            stream: max(0, int(count) - int(self._drop_baseline.get(stream, 0)))
            for stream, count in current_drops.items()
            if int(count) - int(self._drop_baseline.get(stream, 0)) > 0
        }
        writer = self._writer
        with self._camera_route_lock:
            try:
                output = writer.finalize(
                    status=status,
                    stop_trigger_monotonic_ns=self._stop_trigger_ns,
                    success=success,
                    ingress_drops=episode_drops,
                    termination=self._termination,
                    abort_requested=self._abort_requested.is_set,
                )
            except BaseException:
                # An asynchronous camera failure is reported only after
                # finalize safely exposes an interrupted episode. Clear that
                # completed writer before propagating the fatal error.
                if writer.closed:
                    self._writer = None
                    self._stop_trigger_ns = None
                    self._finalize_deadline_ns = None
                    self._termination = None
                    self._drop_baseline = {}
                    self._pre_roll.clear()
                raise
            self._writer = None
        manifest_status = status
        manifest_validation: Dict[str, Any] = {}
        try:
            with (output / "manifest.json").open("r", encoding="utf-8") as handle:
                saved_manifest = json.load(handle)
                manifest_status = str(saved_manifest.get("status", status))
                manifest_validation = saved_manifest.get("validation", {})
        except Exception:
            pass
        print(f"[recorder] episode saved: {output} | status={manifest_status}")
        if manifest_status == "invalid":
            print(
                "[recorder] validation failed: "
                f"missing_streams={manifest_validation.get('missing_streams', [])}, "
                f"stale_streams={manifest_validation.get('stale_streams', [])}, "
                f"missing_joint_names={manifest_validation.get('missing_joint_names', [])}, "
                f"missing_joint_topics={manifest_validation.get('missing_joint_topics', [])}, "
                f"stale_joint_topics={manifest_validation.get('stale_joint_topics', [])}, "
                "failed_matching_event_requirements="
                f"{manifest_validation.get('failed_matching_event_requirements', [])}"
            )
        self._stop_trigger_ns = None
        self._finalize_deadline_ns = None
        self._termination = None
        self._drop_baseline = {}
        self._pre_roll.clear()
        return output

    def handle_event(self, event: Dict[str, Any]) -> None:
        if str(event.get("stream", "")) == "camera_head":
            self.handle_camera_event(event)
            return
        now_ns = event_monotonic_ns(event)
        with self._lock:
            is_controller = event.get("stream") == "controller"
            start_pressed = (
                self._button_state(event, "right_key_one") if is_controller else self._prev_start
            )
            stop_pressed = (
                self._button_state(event, "left_key_one") if is_controller else self._prev_stop
            )
            start_edge = is_controller and start_pressed and not self._prev_start
            stop_edge = is_controller and stop_pressed and not self._prev_stop

            if self._writer is None:
                self._evict_pre_roll(now_ns)
                if start_edge:
                    rejection = self._camera_start_rejection(now_ns)
                    if rejection is None:
                        self._start_episode(event, now_ns)
                        self._writer.write_event(event)
                    else:
                        print(
                            "[recorder] start rejected: required camera is not "
                            f"fresh: {rejection}; release and press again"
                        )
                        self._pre_roll.append(event)
                else:
                    self._pre_roll.append(event)
            else:
                self._writer.write_event(event)
                if stop_edge and self._finalize_deadline_ns is None:
                    self._stop_trigger_ns = now_ns
                    self._finalize_deadline_ns = now_ns + self.post_roll_ns
                    self._writer.mark_stop_trigger(now_ns)
                    print(
                        "[recorder] stop received; collecting "
                        f"{self.post_roll_ns / 1e9:.2f}s post-roll"
                    )

            if is_controller:
                self._prev_start = start_pressed
                self._prev_stop = stop_pressed
            self._finish_camera_stall_if_needed(now_ns)

    def handle_camera_event(self, event: Dict[str, Any]) -> None:
        """Route one camera frame without involving the state dispatcher.

        Camera transport has its own ingress worker.  That worker calls this
        method, which only performs episode/pre-roll routing while holding the
        manager lock; :meth:`RawEpisodeWriter.write_event` then hands the JPEG
        to its bounded asynchronous camera writer.  Keeping this entry point
        separate makes it impossible for high-rate state JSON traffic in the
        public dispatcher to coalesce camera frames before they reach the
        episode writer.
        """

        if str(event.get("stream", "")) != "camera_head":
            raise ValueError("handle_camera_event requires stream='camera_head'")
        now_ns = event_monotonic_ns(event)
        with self._camera_route_lock:
            if (
                self._last_camera_recv_monotonic_ns is None
                or now_ns > self._last_camera_recv_monotonic_ns
            ):
                self._last_camera_recv_monotonic_ns = int(now_ns)
            if self._writer is None:
                self._evict_camera_pre_roll(now_ns)
                self._camera_pre_roll.append(event)
            else:
                # This is a non-blocking enqueue into RawEpisodeWriter's
                # dedicated camera thread; no image or JSON file I/O occurs
                # while the manager lock is held here.
                self._writer.write_event(event)

    def tick(self, now_ns: Optional[int] = None) -> None:
        with self._lock:
            current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
            if self._finish_camera_stall_if_needed(current_ns) is not None:
                return
            if self._finalize_deadline_ns is not None and current_ns >= self._finalize_deadline_ns:
                self._finish_episode(status="complete", success=None)

    def close(self) -> Optional[Path]:
        with self._lock:
            status = "complete" if self._stop_trigger_ns is not None else "interrupted"
            return self._finish_episode(status=status, success=None)

    def request_abort(self) -> None:
        """Latch interrupted status without waiting for the writer lock."""

        self._abort_requested.set()

    def abort(self) -> Optional[Path]:
        """Finalize any active episode as interrupted after a writer failure.

        A stop trigger alone must not promote an episode to ``complete`` once
        the dispatcher has observed an I/O or serialization failure.  If the
        final flush itself also fails, the hidden ``.partial`` directory is
        intentionally left in place for manual recovery.
        """

        self.request_abort()
        with self._lock:
            return self._finish_episode(status="interrupted", success=None)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
