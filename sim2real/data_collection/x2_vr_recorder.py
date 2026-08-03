#!/usr/bin/env python3
"""Record X2 VR demonstrations without touching the real-time control path.

The process subscribes to the teleop bridge's independent PUB tap and, when
ROS is enabled, to the robot's compressed head camera, split joint states and
IMUs.  Right ``key_one`` starts an episode; left ``key_one`` stops it.

This writes a loss-preserving raw format.  Conversion/resampling into a
LeRobotDataset happens offline in ``convert_to_lerobot.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .raw_episode_writer import RawEpisodeManager, ros_stamp_to_ns
    from .schema import TELEOP_TAP_SCHEMA_VERSION, X2_TRACKING_JOINT_NAMES
except ImportError:  # Direct execution from this directory.
    from raw_episode_writer import RawEpisodeManager, ros_stamp_to_ns
    from schema import TELEOP_TAP_SCHEMA_VERSION, X2_TRACKING_JOINT_NAMES


DEFAULT_CAMERA_TOPIC = "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed"
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


class IngressQueue:
    """Bounded callback queue with per-stream receive/drop accounting."""

    def __init__(self, maxsize: int) -> None:
        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max(1, int(maxsize)))
        self._received: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._peak_size = 0
        self._lock = threading.Lock()

    def put(self, event: Dict[str, Any]) -> None:
        stream = str(event.get("stream", "unknown"))
        with self._lock:
            self._received[stream] += 1
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            if stream == "controller":
                # Preserve the newest button sample (and therefore start/stop
                # edges) by sacrificing one older queued data event.
                try:
                    evicted = self.queue.get_nowait()
                    self.queue.task_done()
                    with self._lock:
                        self._dropped[str(evicted.get("stream", "unknown"))] += 1
                    self.queue.put_nowait(event)
                except (queue.Empty, queue.Full):
                    with self._lock:
                        self._dropped[stream] += 1
            else:
                with self._lock:
                    self._dropped[stream] += 1
        with self._lock:
            self._peak_size = max(self._peak_size, self.queue.qsize())

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
        return self.queue.qsize()

    def peak_size(self) -> int:
        with self._lock:
            return self._peak_size


class EventDispatcher:
    def __init__(self, ingress: IngressQueue, manager: RawEpisodeManager) -> None:
        self.ingress = ingress
        self.manager = manager
        self.stop_event = threading.Event()
        self.fatal_exception: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, name="x2-raw-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.ingress.queue.empty():
            try:
                event = self.ingress.queue.get(timeout=0.05)
            except queue.Empty:
                try:
                    self.manager.tick()
                except BaseException as exc:
                    self.fatal_exception = exc
                    self.stop_event.set()
                    print(f"[recorder] fatal writer error: {exc}")
                    return
                continue
            try:
                self.manager.handle_event(event)
            except BaseException as exc:
                self.fatal_exception = exc
                self.stop_event.set()
                print(f"[recorder] fatal writer error: {exc}")
                return
            finally:
                self.ingress.queue.task_done()
            # Do not finalize post-roll while already-received events remain in
            # the writer backlog.  Raw may contain a little extra tail; the
            # converter crops exactly at the stop trigger.
            if self.ingress.queue.empty():
                try:
                    self.manager.tick()
                except BaseException as exc:
                    self.fatal_exception = exc
                    self.stop_event.set()
                    print(f"[recorder] fatal writer error: {exc}")
                    return

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            print("[recorder] writer did not stop within 10 s; preserving partial episode")
        else:
            try:
                self.manager.close()
            except BaseException as exc:
                if self.fatal_exception is None:
                    self.fatal_exception = exc
                print(f"[recorder] failed to finalize active episode: {exc}")


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


def _build_ros_node(ingress: IngressQueue, args: argparse.Namespace) -> Any:
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
    if args.sensor_profile == "aimdk":
        try:
            from aimdk_msgs.msg import JointStateArray
        except ImportError as exc:
            raise ImportError(
                "AimDK ROS messages are unavailable. Source the built "
                "x1_digit_mc/install/setup.bash after /opt/ros/humble/setup.bash, "
                "or use --sensor_profile compat when compatibility topics exist."
            ) from exc
        joint_message_type = JointStateArray

    class X2SensorRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("x2_vr_data_recorder")
            # ``Node`` itself owns an internal ``_subscriptions`` collection.
            # Keep our Python references under a distinct name; shadowing that
            # attribute duplicates entries and makes ``destroy_node()`` fail.
            self._owned_subscriptions = []
            self._callback_group = ReentrantCallbackGroup()

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

            if args.camera_topic:
                self._owned_subscriptions.append(
                    self.create_subscription(
                        CompressedImage,
                        args.camera_topic,
                        self._on_camera,
                        camera_qos,
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

            self.get_logger().info(
                f"camera={args.camera_topic or '<disabled>'}, "
                f"joint_topics={args.joint_topics}, imu_topics={args.imu_topics}"
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
            event = self._base_event(
                "camera_head", ros_stamp_to_ns(message.header.stamp), message.header.frame_id
            )
            event.update(
                {
                    "topic": args.camera_topic,
                    "format": str(message.format),
                    "data": bytes(message.data),
                }
            )
            ingress.put(event)

        def _on_joint_state(self, topic: str, message: Any) -> None:
            event = self._base_event(
                "joint_states", ros_stamp_to_ns(message.header.stamp), message.header.frame_id
            )
            event.update({"topic": topic, **_joint_state_payload(message)})
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
    profile = SENSOR_PROFILES[args.sensor_profile]
    if args.joint_topics is None:
        args.joint_topics = list(profile["joint_topics"])
    if args.imu_topics is None:
        args.imu_topics = list(profile["imu_topics"])
    return args


def main() -> None:
    args = parse_args()
    if not args.task.strip():
        raise ValueError("--task must not be empty")
    if args.pre_roll_s < 0 or args.post_roll_s < 0:
        raise ValueError("pre/post-roll durations must be non-negative")

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq is required by the X2 VR recorder") from exc

    output_root = Path(args.output_root).expanduser().resolve()
    ingress = IngressQueue(args.queue_size)
    source_config = {
        "hostname": socket.gethostname(),
        "tap_addr": args.tap_addr,
        "camera_topic": None if args.disable_ros else args.camera_topic,
        "camera_qos_reliability": args.camera_qos_reliability,
        "sensor_profile": args.sensor_profile,
        "joint_message_type": SENSOR_PROFILES[args.sensor_profile]["joint_message_type"],
        "joint_topics": [] if args.disable_ros else list(args.joint_topics),
        "imu_topics": [] if args.disable_ros else list(args.imu_topics),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", "0"),
        "required_streams": (
            ["controller", "reference"]
            if args.disable_ros
            else ["controller", "reference", "camera_head", "joint_states", "imu_torso"]
        ),
        "minimum_stream_counts": (
            {"controller": 2, "reference": 2}
            if args.disable_ros
            else {
                "controller": 2,
                "reference": 2,
                "camera_head": 2,
                "joint_states": 3,
                "imu_torso": 2,
            }
        ),
        "max_stream_gap_s": (
            {"controller": 1.0, "reference": 0.5}
            if args.disable_ros
            else {
                "controller": 1.0,
                "reference": 0.5,
                "camera_head": 0.5,
                "imu_torso": 0.5,
            }
        ),
        "required_joint_names": [] if args.disable_ros else list(X2_TRACKING_JOINT_NAMES),
        "required_joint_topics": (
            [] if args.disable_ros else _required_joint_topics(args.joint_topics)
        ),
        "max_joint_topic_gap_s": 0.0 if args.disable_ros else 0.5,
    }
    manager = RawEpisodeManager(
        output_root=output_root,
        task=args.task,
        pre_roll_s=args.pre_roll_s,
        post_roll_s=args.post_roll_s,
        source_config=source_config,
        drop_counts=ingress.drops,
    )
    dispatcher = EventDispatcher(ingress, manager)

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
            _, ros_node = _build_ros_node(ingress, args)
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
    tap_socket.setsockopt(zmq.SUBSCRIBE, b"")
    tap_socket.connect(args.tap_addr)
    poller = zmq.Poller()
    poller.register(tap_socket, zmq.POLLIN)

    dispatcher.start()
    if ros_thread is not None:
        ros_thread.start()

    print(f"[recorder] output: {output_root}")
    print(f"[recorder] teleop tap: {args.tap_addr}")
    print("[recorder] waiting: right key_one starts; left key_one stops and saves")
    if args.disable_ros:
        print("[recorder] ROS disabled: this run records XR/reference/controller only")

    next_status_time = time.monotonic() + max(0.1, args.status_interval_s)
    last_tap_seq: Optional[int] = None
    writer_failed = False
    shutdown_requested = threading.Event()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: shutdown_requested.set())
    try:
        while not shutdown_requested.is_set():
            if dispatcher.fatal_exception is not None:
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
                    try:
                        tap_seq = int(event["tap_seq"])
                    except (KeyError, TypeError, ValueError):
                        tap_seq = None
                    if tap_seq is not None:
                        gap = (
                            max(0, tap_seq - last_tap_seq - 1)
                            if last_tap_seq is not None and tap_seq > last_tap_seq
                            else 0
                        )
                        if gap:
                            event["tap_gap_before"] = gap
                            ingress.note_drop("teleop_tap_transport", gap)
                        last_tap_seq = (
                            tap_seq if last_tap_seq is None else max(last_tap_seq, tap_seq)
                        )
                    ingress.put(event)

            now = time.monotonic()
            if now >= next_status_time:
                print(
                    f"[recorder] state={manager.state}, episode={manager.current_episode_index}, "
                    f"queue={ingress.size()}/{args.queue_size} (peak={ingress.peak_size()}), "
                    f"received={ingress.counts()}, dropped={ingress.drops()}"
                )
                next_status_time = now + max(0.1, args.status_interval_s)
    except KeyboardInterrupt:
        print("\n[recorder] stopping")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        poller.unregister(tap_socket)
        tap_socket.close(0)

        if ros_executor is not None:
            ros_executor.shutdown(timeout_sec=2.0)
        if ros_thread is not None:
            ros_thread.join(timeout=3.0)
        if ros_node is not None:
            ros_node.destroy_node()
        if ros_api is not None and ros_api.ok():
            ros_api.shutdown()

        dispatcher.close()
        print(f"[recorder] final received={ingress.counts()}, dropped={ingress.drops()}")

    if writer_failed or dispatcher.fatal_exception is not None:
        raise RuntimeError(
            "Raw recorder writer failed; inspect the partial episode and error above"
        ) from dispatcher.fatal_exception


if __name__ == "__main__":
    main()
