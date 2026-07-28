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
                "pre_roll_s": source_config.get("pre_roll_s"),
                "post_roll_s": source_config.get("post_roll_s"),
            },
            "source_config": source_config,
            "stream_counts": {},
            "ingress_drops": {},
        }
        _write_json(self.partial_dir / "manifest.json", self._manifest)

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
        if timestamp_ns < self._start_monotonic_ns or self._validation_stop_ns is not None:
            return
        self._update_timing_stats(self._active_stream_stats, stream, timestamp_ns)
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
        if self._validation_stop_ns is None:
            self._validation_stop_ns = int(timestamp_ns)

    def write_event(self, event: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed episode")
        stream = str(event.get("stream", "unknown"))
        serializable = dict(event)
        timestamp_ns = event_monotonic_ns(serializable)
        self._track_active_event(stream, serializable, timestamp_ns)

        if stream == "camera_head":
            image_data = serializable.pop("data", None)
            if not isinstance(image_data, (bytes, bytearray, memoryview)):
                return
            image_format = str(serializable.get("format", "jpeg")).lower()
            suffix = ".png" if "png" in image_format else ".jpg"
            filename = f"{self._camera_index:08d}_{timestamp_ns}{suffix}"
            image_path = self.image_dir / filename
            with image_path.open("wb") as handle:
                handle.write(bytes(image_data))
            serializable["image_path"] = str(image_path.relative_to(self.partial_dir))
            serializable["encoded_size_bytes"] = len(image_data)
            self._camera_index += 1

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
    ) -> Path:
        if self._closed:
            return self.final_dir

        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._handles.clear()

        self._manifest["status"] = str(status)
        self._manifest["success"] = success
        self._manifest["stream_counts"] = dict(sorted(self._counts.items()))
        self._manifest["ingress_drops"] = dict(sorted(ingress_drops.items()))
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

        validation_valid = not any(
            (
                missing_streams,
                stale_streams,
                missing_joint_names,
                missing_joint_topics,
                stale_joint_topics,
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

        self.partial_dir.replace(self.final_dir)
        self._closed = True
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

        self._pre_roll: Deque[Dict[str, Any]] = deque()
        self._writer: Optional[RawEpisodeWriter] = None
        self._next_episode_index = self._find_next_episode_index()
        self._prev_start = False
        self._prev_stop = False
        self._stop_trigger_ns: Optional[int] = None
        self._finalize_deadline_ns: Optional[int] = None
        self._drop_baseline: Dict[str, int] = {}
        self._lock = threading.Lock()

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
        self._drop_baseline = self.drop_counts()
        self._writer = RawEpisodeWriter(
            output_root=self.output_root,
            episode_index=self._next_episode_index,
            task=self.task,
            start_monotonic_ns=now_ns,
            start_wall_time_ns=start_wall_time_ns,
            source_config=self.source_config,
        )
        self._next_episode_index += 1
        cutoff_ns = now_ns - self.pre_roll_ns
        buffered_events = sorted(
            (
                event
                for event in self._pre_roll
                if cutoff_ns <= event_monotonic_ns(event) <= now_ns
            ),
            key=event_monotonic_ns,
        )
        for buffered_event in buffered_events:
            self._writer.write_event(buffered_event)
        self._pre_roll.clear()
        print(f"[recorder] episode {self._writer.episode_index:06d} started | task={self.task!r}")

    def _finish_episode(self, status: str, success: Optional[bool]) -> Optional[Path]:
        if self._writer is None:
            return None
        current_drops = self.drop_counts()
        episode_drops = {
            stream: max(0, int(count) - int(self._drop_baseline.get(stream, 0)))
            for stream, count in current_drops.items()
            if int(count) - int(self._drop_baseline.get(stream, 0)) > 0
        }
        output = self._writer.finalize(
            status=status,
            stop_trigger_monotonic_ns=self._stop_trigger_ns,
            success=success,
            ingress_drops=episode_drops,
        )
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
                f"stale_joint_topics={manifest_validation.get('stale_joint_topics', [])}"
            )
        self._writer = None
        self._stop_trigger_ns = None
        self._finalize_deadline_ns = None
        self._drop_baseline = {}
        self._pre_roll.clear()
        return output

    def handle_event(self, event: Dict[str, Any]) -> None:
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
                    self._start_episode(event, now_ns)
                    self._writer.write_event(event)
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

    def tick(self, now_ns: Optional[int] = None) -> None:
        with self._lock:
            current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
            if self._finalize_deadline_ns is not None and current_ns >= self._finalize_deadline_ns:
                self._finish_episode(status="complete", success=None)

    def close(self) -> Optional[Path]:
        with self._lock:
            status = "complete" if self._stop_trigger_ns is not None else "interrupted"
            return self._finish_episode(status=status, success=None)


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
