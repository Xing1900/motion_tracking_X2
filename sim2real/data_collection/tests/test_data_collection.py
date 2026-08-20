from __future__ import annotations

import base64
import json
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_COLLECTION_DIR))

from convert_to_lerobot import (  # noqa: E402
    CONVERSION_REPORT_SCHEMA_VERSION,
    ConvertedSample,
    EpisodeSamples,
    HandCommandSeries,
    ReferenceSeries,
    _conversion_report_payload,
    _decode_rgb,
    _load_hand_command_series,
    _write_conversion_report,
    build_episode_samples,
    split_contiguous_samples,
)
from camera_tap_client import (  # noqa: E402
    CAMERA_FRAME_HEADER,
    CAMERA_FRAME_MAGIC,
    CAMERA_FRAME_VERSION,
    CameraTapClient,
    parse_camera_tap_addr,
    read_camera_tap_frame,
)
from raw_episode_writer import RawEpisodeManager, RawEpisodeWriter  # noqa: E402
from schema import ACTION_NAMES, HAND_ACTION_NAMES, X2_TRACKING_JOINT_NAMES  # noqa: E402
from synchronization import (  # noqa: E402
    TIME_BASIS_RECEIVER,
    TIME_BASIS_SOURCE,
    source_time_point,
    timed_stream,
)
from x2_vr_recorder import (  # noqa: E402
    DEFAULT_AIMDK_IMU_TOPICS,
    DEFAULT_AIMDK_JOINT_TOPICS,
    DEFAULT_CAMERA_TOPIC,
    DEFAULT_HAND_STATUS_TOPIC,
    EventDispatcher,
    IngressQueue,
    SENSOR_PROFILES,
    _imu_stream_name,
    _hand_status_payload,
    _joint_state_payload,
    _required_joint_topics,
)


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def timed_event(stream: str, timestamp_ns: int, **payload):
    return {
        "stream": stream,
        "recorder_recv_monotonic_ns": timestamp_ns,
        "recorder_recv_wall_time_ns": timestamp_ns + 10_000,
        **payload,
    }


def source_timed_event(
    stream: str,
    source_monotonic_ns: int,
    recorder_receive_monotonic_ns: int,
    **payload,
):
    wall_epoch_ns = 1_700_000_000_000_000_000
    event = {
        "stream": stream,
        "source_timestamp_ns": wall_epoch_ns + source_monotonic_ns,
        "recorder_recv_monotonic_ns": recorder_receive_monotonic_ns,
        "recorder_recv_wall_time_ns": wall_epoch_ns + recorder_receive_monotonic_ns,
        **payload,
    }
    if stream == "reference":
        event.update(
            {
                "bridge_recv_monotonic_ns": source_monotonic_ns,
                "bridge_enqueue_monotonic_ns": source_monotonic_ns + 1_000_000,
                "bridge_enqueue_wall_time_ns": (
                    wall_epoch_ns + source_monotonic_ns + 1_000_000
                ),
            }
        )
    return event


class RawEpisodeManagerTest(unittest.TestCase):
    def test_controller_enqueue_never_evicts_queued_controller_edge(self):
        ingress = IngressQueue(maxsize=3)
        release = timed_event("controller", 100, edge="release")
        state = timed_event("joint_states", 150)
        press = timed_event("controller", 200, edge="press")
        newest = timed_event("controller", 250, edge="newest")
        ingress.put(release)
        ingress.put(state)
        ingress.put(press)

        ingress.put(newest)

        queued = list(ingress.queue.queue)
        self.assertEqual(
            [event.get("edge") for event in queued if event["stream"] == "controller"],
            ["release", "press", "newest"],
        )
        self.assertEqual(ingress.drops(), {"joint_states": 1})

    def test_dispatcher_join_timeout_is_fatal_and_preserves_active_writer(self):
        class BlockingManager:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.close_called = False
                self.abort_called = False

            def handle_event(self, _event):
                self.entered.set()
                self.release.wait(timeout=2.0)

            def tick(self):
                pass

            def close(self):
                self.close_called = True

            def abort(self):
                self.abort_called = True

            def request_abort(self):
                self.abort_called = True

        manager = BlockingManager()
        ingress = IngressQueue(maxsize=2)
        dispatcher = EventDispatcher(ingress, manager, join_timeout_s=0.01)
        dispatcher.start()
        ingress.put(timed_event("reference", 100))
        self.assertTrue(manager.entered.wait(timeout=1.0))

        dispatcher.close()

        self.assertIsInstance(dispatcher.fatal_exception, TimeoutError)
        self.assertFalse(manager.close_called)
        self.assertTrue(manager.abort_called)
        manager.release.set()
        dispatcher.thread.join(timeout=1.0)
        self.assertFalse(dispatcher.thread.is_alive())

    def test_dispatcher_timeout_after_stop_can_never_complete_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RawEpisodeManager(
                output_root=root,
                task="timeout after stop regression",
                pre_roll_s=0.0,
                post_roll_s=0.0,
                source_config={},
                drop_counts=lambda: {},
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_000_000_000,
                    controller_buttons={"right_key_one": True, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_100_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_200_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": True},
                )
            )
            original_handle_event = manager.handle_event
            entered = threading.Event()
            release = threading.Event()

            def blocked_handle_event(event):
                entered.set()
                release.wait(timeout=2.0)
                original_handle_event(event)

            manager.handle_event = blocked_handle_event
            ingress = IngressQueue(maxsize=2)
            dispatcher = EventDispatcher(ingress, manager, join_timeout_s=0.01)
            dispatcher.start()
            ingress.put(timed_event("reference", 1_300_000_000))
            self.assertTrue(entered.wait(timeout=1.0))

            dispatcher.close()
            self.assertIsInstance(dispatcher.fatal_exception, TimeoutError)
            release.set()
            dispatcher.thread.join(timeout=1.0)
            self.assertFalse(dispatcher.thread.is_alive())

            episode = root / "episode_000000"
            self.assertTrue(episode.is_dir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")

    def test_dispatcher_timeout_during_finalize_fsync_is_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RawEpisodeManager(
                output_root=root,
                task="timeout during finalize regression",
                pre_roll_s=0.0,
                post_roll_s=0.0,
                source_config={},
                drop_counts=lambda: {},
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_000_000_000,
                    controller_buttons={"right_key_one": True, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_100_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_200_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": True},
                )
            )
            entered_fsync = threading.Event()
            release_fsync = threading.Event()

            def blocked_fsync(_fileno):
                entered_fsync.set()
                release_fsync.wait(timeout=2.0)

            ingress = IngressQueue(maxsize=2)
            dispatcher = EventDispatcher(ingress, manager, join_timeout_s=0.01)
            with mock.patch("raw_episode_writer.os.fsync", side_effect=blocked_fsync):
                dispatcher.start()
                ingress.put(timed_event("reference", 1_300_000_000))
                self.assertTrue(entered_fsync.wait(timeout=1.0))

                dispatcher.close()
                self.assertIsInstance(dispatcher.fatal_exception, TimeoutError)
                release_fsync.set()
                dispatcher.thread.join(timeout=1.0)
                self.assertFalse(dispatcher.thread.is_alive())

            episode = root / "episode_000000"
            self.assertTrue(episode.is_dir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")

    def test_dispatcher_writer_failure_after_stop_cannot_finalize_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RawEpisodeManager(
                output_root=root,
                task="writer failure regression",
                pre_roll_s=0.0,
                post_roll_s=10.0,
                source_config={},
                drop_counts=lambda: {},
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_000_000_000,
                    controller_buttons={"right_key_one": True, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_100_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_200_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": True},
                )
            )

            ingress = IngressQueue(maxsize=8)
            dispatcher = EventDispatcher(ingress, manager)
            manager.handle_event = mock.Mock(side_effect=OSError("simulated disk failure"))
            dispatcher.start()
            ingress.put(timed_event("reference", 1_300_000_000))
            dispatcher.thread.join(timeout=2.0)
            self.assertIsInstance(dispatcher.fatal_exception, OSError)

            dispatcher.close()
            episode = root / "episode_000000"
            self.assertTrue(episode.is_dir())
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertFalse((root / ".episode_000000.partial").exists())

    def test_button_edges_create_atomic_episode_with_preroll(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drops = {"camera_head": 2}
            manager = RawEpisodeManager(
                output_root=root,
                task="touch the red button",
                pre_roll_s=1.0,
                post_roll_s=0.0,
                source_config={},
                drop_counts=lambda: dict(drops),
            )
            manager.handle_event(
                timed_event("camera_head", 1_000_000_000, format="png", data=VALID_PNG)
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_100_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_200_000_000,
                    controller_buttons={"right_key_one": True, "left_key_one": False},
                )
            )
            drops["camera_head"] = 4
            manager.handle_event(
                timed_event(
                    "reference",
                    1_300_000_000,
                    frames_qpos_root_xyz_quat_wxyz_dof=[[0.0] * 36],
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_400_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": False},
                )
            )
            manager.handle_event(
                timed_event(
                    "controller",
                    1_500_000_000,
                    controller_buttons={"right_key_one": False, "left_key_one": True},
                )
            )
            manager.tick(now_ns=1_500_000_000)

            episode = root / "episode_000000"
            self.assertTrue(episode.is_dir())
            self.assertFalse((root / ".episode_000000.partial").exists())
            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["task"], "touch the red button")
            self.assertEqual(manifest["stream_counts"]["camera_head"], 1)
            self.assertEqual(manifest["ingress_drops"]["camera_head"], 2)
            self.assertEqual(len(list((episode / "images" / "head_rgb").glob("*.png"))), 1)

    def test_controller_displaces_data_when_ingress_is_full(self):
        ingress = IngressQueue(maxsize=1)
        ingress.put({"stream": "camera_head"})
        ingress.put({"stream": "controller", "controller_buttons": {}})
        self.assertEqual(ingress.queue.get_nowait()["stream"], "controller")
        self.assertEqual(ingress.drops()["camera_head"], 1)

    def test_direct_camera_pressure_coalesces_outside_public_fifo(self):
        ingress = IngressQueue(maxsize=4)

        for sequence in range(100):
            ingress.put_latest_camera(
                timed_event("camera_head", 1_000 + sequence, sequence=sequence)
            )

        self.assertTrue(ingress.queue.empty())
        self.assertEqual(ingress.size(), 1)
        self.assertEqual(ingress.peak_size(), 1)
        self.assertEqual(ingress.counts()["camera_head"], 100)
        self.assertEqual(ingress.drops()["camera_coalesced"], 99)
        item = ingress.get_next(timeout=0.0)
        self.assertIsNotNone(item)
        event, came_from_fifo = item
        self.assertFalse(came_from_fifo)
        self.assertEqual(event["sequence"], 99)
        self.assertTrue(ingress.empty())

    def test_direct_camera_slot_is_thread_safe_under_concurrent_callbacks(self):
        ingress = IngressQueue(maxsize=4)
        start = threading.Barrier(5)

        def publish(worker: int) -> None:
            start.wait()
            for sequence in range(100):
                ingress.put_latest_camera(
                    timed_event(
                        "camera_head",
                        10_000 + worker * 100 + sequence,
                        sequence=worker * 100 + sequence,
                    )
                )

        workers = [threading.Thread(target=publish, args=(index,)) for index in range(4)]
        for worker in workers:
            worker.start()
        start.wait()
        for worker in workers:
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())

        self.assertTrue(ingress.queue.empty())
        self.assertEqual(ingress.size(), 1)
        self.assertEqual(ingress.counts()["camera_head"], 400)
        self.assertEqual(ingress.drops()["camera_coalesced"], 399)

    def test_coalesced_camera_does_not_overtake_older_fifo_events(self):
        ingress = IngressQueue(maxsize=8)
        ingress.put(timed_event("joint_states", 100, tag="joint-old"))
        ingress.put_latest_camera(timed_event("camera_head", 200, tag="camera"))
        ingress.put(timed_event("controller", 150, tag="start-edge"))
        ingress.put(timed_event("imu_torso", 250, tag="imu-new"))

        dispatched = []
        fifo_items = 0
        while not ingress.empty():
            item = ingress.get_next(timeout=0.0)
            self.assertIsNotNone(item)
            event, came_from_fifo = item
            dispatched.append(event["tag"])
            if came_from_fifo:
                fifo_items += 1
                ingress.queue.task_done()

        self.assertEqual(
            dispatched,
            ["joint-old", "start-edge", "camera", "imu-new"],
        )
        self.assertEqual(fifo_items, 3)

    def test_dispatcher_close_flushes_latest_camera_and_fifo(self):
        class RecordingManager:
            def __init__(self):
                self.events = []
                self.closed = False

            def handle_event(self, event):
                self.events.append(event)

            def tick(self):
                pass

            def close(self):
                self.closed = True

        ingress = IngressQueue(maxsize=8)
        manager = RecordingManager()
        ingress.put(timed_event("controller", 100, tag="controller"))
        for sequence in range(10):
            ingress.put_latest_camera(
                timed_event(
                    "camera_head",
                    101 + sequence,
                    tag=f"camera-{sequence}",
                    sequence=sequence,
                )
            )

        dispatcher = EventDispatcher(ingress, manager)
        dispatcher.start()
        dispatcher.close()
        ingress.queue.join()

        self.assertFalse(dispatcher.thread.is_alive())
        self.assertTrue(manager.closed)
        self.assertEqual(
            [event["tag"] for event in manager.events],
            ["controller", "camera-9"],
        )
        self.assertEqual(ingress.drops()["camera_coalesced"], 9)
        self.assertTrue(ingress.empty())

    def test_missing_required_stream_marks_episode_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = RawEpisodeWriter(
                output_root=Path(temporary),
                episode_index=0,
                task="test task",
                start_monotonic_ns=1,
                start_wall_time_ns=2,
                source_config={"required_streams": ["controller", "camera_head"]},
            )
            writer.write_event(timed_event("controller", 2, controller_buttons={}))
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=3,
                success=None,
                ingress_drops={},
            )
            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "invalid")
            self.assertEqual(manifest["validation"]["missing_streams"], ["camera_head"])

    def test_matching_event_requirement_uses_only_active_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = RawEpisodeWriter(
                output_root=Path(temporary),
                episode_index=0,
                task="test hand validation",
                start_monotonic_ns=100,
                start_wall_time_ns=2,
                source_config={
                    "minimum_matching_event_counts": [
                        {
                            "name": "hand_command_active",
                            "stream": "hand_command",
                            "field": "active",
                            "equals": True,
                            "minimum_count": 2,
                        }
                    ]
                },
            )
            writer.write_event(timed_event("hand_command", 99, active=True))
            writer.write_event(timed_event("hand_command", 100, active=False))
            writer.write_event(timed_event("hand_command", 101, active=True))
            writer.mark_stop_trigger(102)
            writer.write_event(timed_event("hand_command", 103, active=True))
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=102,
                success=None,
                ingress_drops={},
            )

            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            validation = manifest["validation"]
            self.assertEqual(manifest["status"], "invalid")
            self.assertEqual(
                validation["failed_matching_event_requirements"],
                ["hand_command_active"],
            )
            self.assertEqual(
                validation["matching_event_requirements"],
                [
                    {
                        "name": "hand_command_active",
                        "stream": "hand_command",
                        "field": "active",
                        "equals": True,
                        "minimum_count": 2,
                        "actual_count": 1,
                        "passed": False,
                    }
                ],
            )

    def test_matching_event_requirement_passes_at_minimum_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = RawEpisodeWriter(
                output_root=Path(temporary),
                episode_index=0,
                task="test hand validation",
                start_monotonic_ns=100,
                start_wall_time_ns=2,
                source_config={
                    "minimum_matching_event_counts": [
                        {
                            "name": "hand_command_active",
                            "stream": "hand_command",
                            "field": "active",
                            "equals": True,
                            "minimum_count": 2,
                        }
                    ]
                },
            )
            writer.write_event(timed_event("hand_command", 100, active=True))
            writer.write_event(timed_event("hand_command", 101, active=True))
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=101,
                success=None,
                ingress_drops={},
            )

            manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                manifest["validation"]["failed_matching_event_requirements"], []
            )


class RecorderSensorAdapterTest(unittest.TestCase):
    def test_default_camera_topic_matches_current_x2_aimdk_stream(self):
        self.assertEqual(
            DEFAULT_CAMERA_TOPIC,
            "/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed",
        )

    def test_aimdk_profile_uses_real_x2_hal_topics(self):
        self.assertEqual(
            SENSOR_PROFILES["aimdk"]["joint_message_type"],
            "aimdk_msgs/msg/JointStateArray",
        )
        self.assertEqual(
            DEFAULT_AIMDK_JOINT_TOPICS[:3],
            [
                "/aima/hal/joint/leg/state",
                "/aima/hal/joint/waist/state",
                "/aima/hal/joint/arm/state",
            ],
        )
        self.assertEqual(
            DEFAULT_AIMDK_IMU_TOPICS,
            [
                "/aima/hal/imu/torso/state",
                "/aima/hal/imu/chest/state",
            ],
        )

    def test_aimdk_joint_state_array_is_normalized(self):
        message = SimpleNamespace(
            state=SimpleNamespace(value=0),
            joints=[
                SimpleNamespace(
                    name="left_hip_pitch_joint",
                    position=-0.3,
                    velocity=0.02,
                    effort=1.5,
                    error_code=7,
                ),
                SimpleNamespace(
                    name="left_hip_roll_joint",
                    position=0.1,
                    velocity=-0.01,
                    effort=0.5,
                    error_code=0,
                ),
            ],
        )

        payload = _joint_state_payload(message)

        self.assertEqual(
            payload["name"], ["left_hip_pitch_joint", "left_hip_roll_joint"]
        )
        self.assertEqual(payload["position"], [-0.3, 0.1])
        self.assertEqual(payload["velocity"], [0.02, -0.01])
        self.assertEqual(payload["effort"], [1.5, 0.5])
        self.assertEqual(payload["error_code"], [7, 0])
        self.assertEqual(payload["domain_state"], 0)

    def test_sensor_msgs_joint_state_remains_supported(self):
        message = SimpleNamespace(
            name=["waist_yaw_joint"],
            position=[0.2],
            velocity=[-0.1],
            effort=[0.0],
        )

        payload = _joint_state_payload(message)

        self.assertEqual(payload["name"], ["waist_yaw_joint"])
        self.assertEqual(payload["position"], [0.2])
        self.assertEqual(payload["velocity"], [-0.1])
        self.assertEqual(payload["source_message_type"], "sensor_msgs/msg/JointState")
        self.assertNotIn("error_code", payload)

    def test_real_profile_validation_requires_leg_waist_and_arm(self):
        self.assertEqual(
            _required_joint_topics(DEFAULT_AIMDK_JOINT_TOPICS),
            DEFAULT_AIMDK_JOINT_TOPICS[:3],
        )
        self.assertEqual(_imu_stream_name(DEFAULT_AIMDK_IMU_TOPICS[0], 0), "imu_torso")
        self.assertEqual(_imu_stream_name(DEFAULT_AIMDK_IMU_TOPICS[1], 1), "imu_chest")

    def test_authoritative_hand_status_is_normalized(self):
        message = SimpleNamespace(
            sequence=42,
            active=True,
            mode=2,
            left_grasp=0.25,
            right_grasp=0.75,
        )

        self.assertEqual(DEFAULT_HAND_STATUS_TOPIC, "/vr_hand_controller/status")
        self.assertEqual(
            _hand_status_payload(message),
            {
                "sequence": 42,
                "active": True,
                "mode": 2,
                "left_grasp": 0.25,
                "right_grasp": 0.75,
                "source_message_type": "x1_protocol/msg/VrHandControlStatus",
            },
        )


class CameraTapClientTest(unittest.TestCase):
    class _FragmentedSocket:
        def __init__(self, payload, chunk_size=3):
            self._payload = bytearray(payload)
            self._chunk_size = chunk_size

        def recv(self, length):
            if not self._payload:
                return b""
            count = min(length, self._chunk_size, len(self._payload))
            result = bytes(self._payload[:count])
            del self._payload[:count]
            return result

        def setsockopt(self, *_args):
            pass

        def settimeout(self, _timeout):
            pass

        def shutdown(self, _how):
            pass

        def close(self):
            pass

    @staticmethod
    def _wire_frame(sequence, payload=b"compressed-image"):
        metadata = json.dumps(
            {
                "schema_version": 1,
                "sequence": sequence,
                "source_timestamp_ns": 123,
                "bridge_recv_monotonic_ns": 456,
                "bridge_recv_wall_time_ns": 789,
                "frame_id": "head_camera",
                "format": "jpeg",
                "topic": "/head/compressed",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return (
            CAMERA_FRAME_HEADER.pack(
                CAMERA_FRAME_MAGIC,
                CAMERA_FRAME_VERSION,
                len(metadata),
                len(payload),
            )
            + metadata
            + payload
        )

    def test_address_requires_standard_tcp_uri(self):
        self.assertEqual(parse_camera_tap_addr("tcp://172.66.88.241:28706"), (
            "172.66.88.241",
            28706,
        ))
        with self.assertRaises(ValueError):
            parse_camera_tap_addr("172.66.88.241:28706")

    def test_reads_fragmented_complete_frame(self):
        wire = self._wire_frame(4, b"jpeg-bytes")
        metadata, payload = read_camera_tap_frame(self._FragmentedSocket(wire))
        self.assertEqual(metadata["sequence"], 4)
        self.assertEqual(payload, b"jpeg-bytes")

    def test_reads_frame_over_socketpair(self):
        try:
            reader, writer = socket.socketpair()
            writer.sendall(self._wire_frame(5, b"socket-image"))
        except PermissionError as exc:
            for current_socket in (locals().get("reader"), locals().get("writer")):
                if current_socket is not None:
                    current_socket.close()
            self.skipTest(f"socket I/O denied by test sandbox: {exc}")
        try:
            metadata, payload = read_camera_tap_frame(reader)
        finally:
            reader.close()
            writer.close()
        self.assertEqual(metadata["sequence"], 5)
        self.assertEqual(payload, b"socket-image")

    def test_reconnects_and_records_transport_sequence_gap(self):
        events = []
        drops = {}

        def note_drop(stream, count):
            drops[stream] = drops.get(stream, 0) + count

        client = CameraTapClient(
            "tcp://127.0.0.1:28706",
            on_event=lambda event: (
                events.append(event),
                client._stop_event.set() if len(events) == 2 else None,
            ),
            note_drop=note_drop,
            log_info=lambda _message: None,
            log_warning=lambda _message: None,
            connect_timeout_s=0.1,
            read_timeout_s=0.05,
            reconnect_initial_s=0.01,
            reconnect_max_s=0.02,
        )
        connections = [
            self._FragmentedSocket(self._wire_frame(10), chunk_size=7),
            self._FragmentedSocket(self._wire_frame(12), chunk_size=11),
        ]
        with mock.patch(
            "camera_tap_client.socket.create_connection", side_effect=connections
        ) as create_connection:
            client.start()
            client._thread.join(timeout=2.0)
        client.close()

        self.assertEqual([event["sequence"] for event in events], [10, 12])
        self.assertEqual(events[1]["camera_tap_gap_before"], 1)
        self.assertEqual(drops["camera_tap_transport"], 1)
        self.assertEqual(client.status().transport_gaps, 1)
        self.assertEqual(create_connection.call_count, 2)

    def test_tcp_camera_gap_and_local_coalescing_are_accounted_separately(self):
        ingress = IngressQueue(maxsize=1)

        def on_event(event):
            ingress.put_latest_camera(event)
            if ingress.counts().get("camera_head") == 2:
                client._stop_event.set()

        client = CameraTapClient(
            "tcp://127.0.0.1:28706",
            on_event=on_event,
            note_drop=ingress.note_drop,
            log_info=lambda _message: None,
            log_warning=lambda _message: None,
            connect_timeout_s=0.1,
            read_timeout_s=0.05,
            reconnect_initial_s=0.01,
            reconnect_max_s=0.02,
        )
        connections = [
            self._FragmentedSocket(
                self._wire_frame(10) + self._wire_frame(12), chunk_size=11
            )
        ]
        with mock.patch(
            "camera_tap_client.socket.create_connection", side_effect=connections
        ):
            client.start()
            client._thread.join(timeout=2.0)
        client.close()

        self.assertEqual(ingress.counts()["camera_head"], 2)
        self.assertEqual(ingress.drops()["camera_tap_transport"], 1)
        self.assertEqual(ingress.drops()["camera_coalesced"], 1)
        self.assertTrue(ingress.queue.empty())
        event, came_from_fifo = ingress.get_next(timeout=0.0)
        self.assertFalse(came_from_fifo)
        self.assertEqual(event["sequence"], 12)
        self.assertEqual(event["camera_tap_gap_before"], 1)


class ConversionTest(unittest.TestCase):
    def test_source_timeline_distinguishes_small_reordering_from_clock_reset(self):
        events = [
            source_timed_event("camera_head", 1_000_000_000, 1_010_000_000),
            source_timed_event("camera_head", 950_000_000, 1_020_000_000),
            source_timed_event("camera_head", 700_000_000, 1_030_000_000),
        ]

        stream = timed_stream(events, "camera_head", time_basis=TIME_BASIS_SOURCE)

        self.assertEqual(stream.source_regression_count, 2)
        self.assertEqual(stream.large_source_regression_count, 1)
        self.assertAlmostEqual(stream.max_source_regression_ms, 300.0)
        self.assertEqual(
            [event["_sync_time_ns"] for event in stream.events],
            sorted(event["_sync_time_ns"] for event in stream.events),
        )

    def test_strict_source_conversion_rejects_large_clock_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="clock regression",
                start_monotonic_ns=1_000_000_000,
                start_wall_time_ns=1_700_000_001_000_000_000,
                source_config={},
            )
            q = [0.0] * 29
            qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *q]
            writer.write_event(
                source_timed_event(
                    "reference",
                    1_100_000_000,
                    1_110_000_000,
                    retarget_age_ms=0.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[qpos],
                )
            )
            writer.write_event(
                source_timed_event(
                    "hand_command",
                    1_100_000_000,
                    1_110_000_000,
                    sequence=1,
                    active=True,
                    left_grasp=0.0,
                    right_grasp=0.0,
                )
            )
            writer.write_event(
                source_timed_event(
                    "joint_states",
                    1_100_000_000,
                    1_110_000_000,
                    name=X2_TRACKING_JOINT_NAMES,
                    position=q,
                    velocity=q,
                )
            )
            writer.write_event(
                source_timed_event(
                    "imu_torso",
                    1_100_000_000,
                    1_110_000_000,
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    angular_velocity_xyz=[0.0, 0.0, 0.0],
                    linear_acceleration_xyz=[0.0, 0.0, 9.81],
                )
            )
            writer.write_event(
                source_timed_event(
                    "camera_head",
                    1_400_000_000,
                    1_410_000_000,
                    format="png",
                    data=VALID_PNG,
                )
            )
            writer.write_event(
                source_timed_event(
                    "camera_head",
                    1_200_000_000,
                    1_420_000_000,
                    format="png",
                    data=VALID_PNG,
                )
            )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=1_500_000_000,
                success=True,
                ingress_drops={},
            )

            with self.assertRaisesRegex(
                ValueError, "Large source-time regression.*camera_head"
            ):
                build_episode_samples(
                    episode,
                    fps=25,
                    max_camera_age_ms=100.0,
                    max_reference_age_ms=100.0,
                    max_reference_gap_ms=120.0,
                    max_retarget_age_ms=100.0,
                    max_hand_command_age_ms=100.0,
                    max_joint_age_ms=100.0,
                    max_imu_age_ms=100.0,
                    time_basis=TIME_BASIS_SOURCE,
                )

    def test_source_clock_maps_header_time_into_recorder_monotonic(self):
        event = source_timed_event(
            "camera_head",
            source_monotonic_ns=1_000_000_000,
            recorder_receive_monotonic_ns=1_100_000_000,
        )

        point = source_time_point(event, "camera_head")

        self.assertIsNotNone(point)
        self.assertEqual(point.time_ns, 1_000_000_000)
        self.assertEqual(point.apparent_latency_ms, 100.0)

    def test_reference_uses_command_time_not_gmr_sample_target(self):
        event = source_timed_event(
            "reference",
            source_monotonic_ns=1_000_000_000,
            recorder_receive_monotonic_ns=1_020_000_000,
            sample_target_monotonic_ns=975_000_000,
        )

        stream = timed_stream([event], "reference", time_basis=TIME_BASIS_SOURCE)

        self.assertEqual(len(stream.events), 1)
        self.assertEqual(stream.events[0]["_sync_time_ns"], 1_000_000_000)

    def test_source_camera_is_unique_while_receiver_keeps_legacy_target_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="source time alignment",
                start_monotonic_ns=1_000_000_000,
                start_wall_time_ns=1_700_000_001_000_000_000,
                source_config={},
            )
            q_before = [0.1] * 29
            q_after = [0.9] * 29
            dq = [0.0] * 29
            qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *q_before]

            writer.write_event(
                source_timed_event(
                    "reference",
                    980_000_000,
                    990_000_000,
                    frame_dt_ns=20_000_000,
                    retarget_age_ms=0.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[qpos],
                )
            )
            writer.write_event(
                source_timed_event(
                    "hand_command",
                    990_000_000,
                    1_000_000_000,
                    sequence=1,
                    active=True,
                    left_grasp=0.2,
                    right_grasp=0.4,
                )
            )
            writer.write_event(
                source_timed_event(
                    "joint_states",
                    990_000_000,
                    1_000_000_000,
                    name=X2_TRACKING_JOINT_NAMES,
                    position=q_before,
                    velocity=dq,
                    effort=[0.0] * 29,
                )
            )
            writer.write_event(
                source_timed_event(
                    "imu_torso",
                    990_000_000,
                    1_000_000_000,
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    angular_velocity_xyz=[0.0, 0.0, 0.0],
                    linear_acceleration_xyz=[0.0, 0.0, 9.81],
                )
            )
            writer.write_event(
                source_timed_event(
                    "camera_head",
                    1_000_000_000,
                    1_100_000_000,
                    format="png",
                    data=VALID_PNG,
                )
            )
            writer.write_event(
                source_timed_event(
                    "joint_states",
                    1_050_000_000,
                    1_060_000_000,
                    name=X2_TRACKING_JOINT_NAMES,
                    position=q_after,
                    velocity=dq,
                    effort=[0.0] * 29,
                )
            )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=1_110_000_000,
                success=True,
                ingress_drops={},
            )

            common = dict(
                fps=10,
                max_camera_age_ms=110.0,
                max_reference_age_ms=200.0,
                max_reference_gap_ms=200.0,
                max_retarget_age_ms=200.0,
                max_hand_command_age_ms=200.0,
                max_joint_age_ms=200.0,
                max_imu_age_ms=200.0,
            )
            source_aligned = build_episode_samples(
                episode, time_basis=TIME_BASIS_SOURCE, **common
            )
            receiver_aligned = build_episode_samples(
                episode, time_basis=TIME_BASIS_RECEIVER, **common
            )

            self.assertEqual(len(source_aligned.samples), 1)
            self.assertEqual(len(receiver_aligned.samples), 2)
            np.testing.assert_allclose(source_aligned.samples[0].state[:29], q_before)
            np.testing.assert_allclose(receiver_aligned.samples[0].state[:29], q_before)
            np.testing.assert_allclose(receiver_aligned.samples[1].state[:29], q_after)
            self.assertEqual(source_aligned.skip_counts["camera_reused"], 1)
            self.assertNotIn("camera_reused", receiver_aligned.skip_counts)
            self.assertEqual(receiver_aligned.samples[0].timing_ms[0], 100.0)
            self.assertEqual(
                source_aligned.samples[0].observation_timestamp_ns,
                1_000_000_000,
            )

    def test_reference_receiver_interpolates_but_source_sampling_is_causal(self):
        first = np.zeros(36, dtype=float)
        first[3] = 1.0
        second = first.copy()
        second[7] = 2.0
        series = ReferenceSeries(
            [
                timed_event(
                    "reference",
                    100_000_000,
                    retarget_age_ms=0.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[first.tolist()],
                ),
                timed_event(
                    "reference",
                    200_000_000,
                    retarget_age_ms=0.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[second.tolist()],
                ),
            ]
        )

        causal, _, _, _ = series.sample_previous(150_000_000)
        interpolated, _, _, _ = series.interpolate(150_000_000)

        self.assertEqual(causal[7], 0.0)
        self.assertEqual(interpolated[7], 1.0)

    def test_reference_batch_retarget_age_includes_scheduled_frame_offset(self):
        frame = np.zeros(36, dtype=float)
        frame[3] = 1.0
        series = ReferenceSeries(
            [
                timed_event(
                    "reference",
                    100_000_000,
                    frame_dt_ns=20_000_000,
                    retarget_age_ms=5.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[
                        frame.tolist(),
                        frame.tolist(),
                    ],
                )
            ]
        )

        _, command_age_ms, retarget_age_ms, _ = series.sample_previous(120_000_000)

        self.assertEqual(command_age_ms, 0.0)
        self.assertEqual(retarget_age_ms, 25.0)

    def test_receiver_reference_batch_preserves_legacy_packet_retarget_age(self):
        frame = np.zeros(36, dtype=float)
        frame[3] = 1.0
        series = ReferenceSeries(
            [
                timed_event(
                    "reference",
                    100_000_000,
                    frame_dt_ns=20_000_000,
                    retarget_age_ms=5.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[
                        frame.tolist(),
                        frame.tolist(),
                    ],
                )
            ],
            account_for_scheduled_frame_age=False,
        )

        _, reference_age_ms, retarget_age_ms, _ = series.interpolate(120_000_000)

        self.assertEqual(reference_age_ms, 0.0)
        self.assertEqual(retarget_age_ms, 5.0)

    def test_receiver_converter_wires_legacy_packet_retarget_age(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="receiver compatibility",
                start_monotonic_ns=120_000_000,
                start_wall_time_ns=1_700_000_000_120_000_000,
                source_config={},
            )
            q = [0.0] * 29
            qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *q]
            writer.write_event(
                timed_event(
                    "reference",
                    100_000_000,
                    frame_dt_ns=20_000_000,
                    retarget_age_ms=5.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[qpos, qpos],
                )
            )
            writer.write_event(
                timed_event(
                    "hand_command",
                    120_000_000,
                    sequence=1,
                    active=True,
                    left_grasp=0.0,
                    right_grasp=0.0,
                )
            )
            writer.write_event(
                timed_event(
                    "joint_states",
                    120_000_000,
                    name=X2_TRACKING_JOINT_NAMES,
                    position=q,
                    velocity=q,
                )
            )
            writer.write_event(
                timed_event(
                    "imu_torso",
                    120_000_000,
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    angular_velocity_xyz=[0.0, 0.0, 0.0],
                    linear_acceleration_xyz=[0.0, 0.0, 9.81],
                )
            )
            writer.write_event(
                timed_event(
                    "camera_head",
                    120_000_000,
                    format="png",
                    data=VALID_PNG,
                )
            )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=121_000_000,
                success=True,
                ingress_drops={},
            )

            converted = build_episode_samples(
                episode,
                fps=25,
                max_camera_age_ms=30.0,
                max_reference_age_ms=30.0,
                max_reference_gap_ms=30.0,
                max_retarget_age_ms=10.0,
                max_hand_command_age_ms=30.0,
                max_joint_age_ms=30.0,
                max_imu_age_ms=30.0,
                time_basis=TIME_BASIS_RECEIVER,
            )

        self.assertEqual(len(converted.samples), 1)
        self.assertNotIn("retarget_stale", converted.skip_counts)

    def test_source_mode_rejects_missing_time_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="missing source timestamp",
                start_monotonic_ns=1_000_000_000,
                start_wall_time_ns=1_700_000_001_000_000_000,
                source_config={},
            )
            q = [0.0] * 29
            qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *q]
            writer.write_event(
                source_timed_event(
                    "reference",
                    990_000_000,
                    995_000_000,
                    retarget_age_ms=0.0,
                    frames_qpos_root_xyz_quat_wxyz_dof=[qpos],
                )
            )
            writer.write_event(
                source_timed_event(
                    "hand_command",
                    990_000_000,
                    995_000_000,
                    sequence=1,
                    active=True,
                    left_grasp=0.0,
                    right_grasp=0.0,
                )
            )
            writer.write_event(
                source_timed_event(
                    "joint_states",
                    990_000_000,
                    995_000_000,
                    name=X2_TRACKING_JOINT_NAMES,
                    position=q,
                    velocity=q,
                )
            )
            writer.write_event(
                source_timed_event(
                    "imu_torso",
                    990_000_000,
                    995_000_000,
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    angular_velocity_xyz=[0.0, 0.0, 0.0],
                    linear_acceleration_xyz=[0.0, 0.0, 9.81],
                )
            )
            # Deliberately omit source_timestamp_ns from the camera event.
            writer.write_event(
                timed_event("camera_head", 1_000_000_000, format="png", data=VALID_PNG)
            )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=1_010_000_000,
                success=True,
                ingress_drops={},
            )
            common = dict(
                fps=25,
                max_camera_age_ms=100.0,
                max_reference_age_ms=100.0,
                max_reference_gap_ms=120.0,
                max_retarget_age_ms=100.0,
                max_hand_command_age_ms=100.0,
                max_joint_age_ms=100.0,
                max_imu_age_ms=100.0,
                time_basis=TIME_BASIS_SOURCE,
            )

            with self.assertRaisesRegex(ValueError, "camera_head=1/1"):
                build_episode_samples(episode, **common)
            partial = build_episode_samples(
                episode, allow_partial_source_time=True, **common
            )

            self.assertEqual(partial.samples, [])
            self.assertEqual(
                partial.synchronization_diagnostics["camera_head"]["missing_time_events"],
                1,
            )

    def test_source_camera_anchor_is_limited_to_active_a_x_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="active camera interval",
                start_monotonic_ns=1_000_000_000,
                start_wall_time_ns=1_700_000_001_000_000_000,
                source_config={},
            )
            q = [0.0] * 29
            qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *q]
            for stream, payload in (
                (
                    "reference",
                    {
                        "retarget_age_ms": 0.0,
                        "frames_qpos_root_xyz_quat_wxyz_dof": [qpos],
                    },
                ),
                (
                    "hand_command",
                    {
                        "sequence": 1,
                        "active": True,
                        "left_grasp": 0.0,
                        "right_grasp": 0.0,
                    },
                ),
                (
                    "joint_states",
                    {"name": X2_TRACKING_JOINT_NAMES, "position": q, "velocity": q},
                ),
                (
                    "imu_torso",
                    {
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "angular_velocity_xyz": [0.0, 0.0, 0.0],
                        "linear_acceleration_xyz": [0.0, 0.0, 9.81],
                    },
                ),
            ):
                writer.write_event(
                    source_timed_event(stream, 995_000_000, 999_000_000, **payload)
                )
            for source_ns, receive_ns in (
                (990_000_000, 1_000_000_000),
                (1_020_000_000, 1_030_000_000),
                (1_090_000_000, 1_100_000_000),
            ):
                writer.write_event(
                    source_timed_event(
                        "camera_head",
                        source_ns,
                        receive_ns,
                        format="png",
                        data=VALID_PNG,
                    )
                )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=1_080_000_000,
                success=True,
                ingress_drops={},
            )

            converted = build_episode_samples(
                episode,
                fps=25,
                max_camera_age_ms=100.0,
                max_reference_age_ms=100.0,
                max_reference_gap_ms=120.0,
                max_retarget_age_ms=100.0,
                max_hand_command_age_ms=100.0,
                max_joint_age_ms=100.0,
                max_imu_age_ms=100.0,
                time_basis=TIME_BASIS_SOURCE,
            )

            self.assertEqual(len(converted.samples), 1)
            self.assertEqual(converted.samples[0].observation_timestamp_ns, 1_020_000_000)
            self.assertEqual(converted.samples[0].timing_ms[0], 20.0)

    def test_hand_command_sampling_is_causal(self):
        series = HandCommandSeries(
            [
                timed_event(
                    "hand_command",
                    90_000_000,
                    sequence=10,
                    active=True,
                    left_grasp=0.2,
                    right_grasp=0.4,
                ),
                timed_event(
                    "hand_command",
                    101_000_000,
                    sequence=11,
                    active=True,
                    left_grasp=0.8,
                    right_grasp=1.0,
                ),
            ],
            legacy_controller=False,
        )

        action, age_ms, status = series.sample(100_000_000)

        self.assertEqual(status, "ok")
        self.assertEqual(age_ms, 10.0)
        np.testing.assert_allclose(action, [0.2, 0.4])

    def test_hand_command_sequence_gap_rejects_only_interval_ticks(self):
        series = HandCommandSeries(
            [
                timed_event(
                    "hand_command",
                    90_000_000,
                    sequence=10,
                    active=True,
                    left_grasp=0.2,
                    right_grasp=0.4,
                ),
                timed_event(
                    "hand_command",
                    110_000_000,
                    sequence=12,
                    active=True,
                    left_grasp=0.8,
                    right_grasp=1.0,
                ),
            ],
            legacy_controller=False,
        )

        first_action, _, first_status = series.sample(90_000_000)
        gap_action, _, gap_status = series.sample(100_000_000)
        next_action, _, next_status = series.sample(110_000_000)

        self.assertEqual(first_status, "ok")
        np.testing.assert_allclose(first_action, [0.2, 0.4])
        self.assertIsNone(gap_action)
        self.assertEqual(gap_status, "sequence_gap")
        self.assertEqual(next_status, "ok")
        np.testing.assert_allclose(next_action, [0.8, 1.0])

    def test_hand_command_sequence_wraparound_is_contiguous(self):
        series = HandCommandSeries(
            [
                timed_event(
                    "hand_command",
                    90_000_000,
                    sequence=(1 << 32) - 1,
                    active=True,
                    left_grasp=0.2,
                    right_grasp=0.4,
                ),
                timed_event(
                    "hand_command",
                    110_000_000,
                    sequence=0,
                    active=True,
                    left_grasp=0.8,
                    right_grasp=1.0,
                ),
            ],
            legacy_controller=False,
        )

        action, _, status = series.sample(100_000_000)

        self.assertEqual(status, "ok")
        np.testing.assert_allclose(action, [0.2, 0.4])

    def test_legacy_controller_grips_use_live_deadzone_mapping(self):
        series = HandCommandSeries(
            [
                timed_event(
                    "controller",
                    100_000_000,
                    controller_age_ms=5.0,
                    controller_buttons={
                        "left_grip_value": 0.10,
                        "right_grip_value": 0.50,
                    },
                )
            ],
            legacy_controller=True,
        )

        action, age_ms, status = series.sample(110_000_000)

        self.assertEqual(status, "ok")
        self.assertEqual(age_ms, 15.0)
        np.testing.assert_allclose(action, [0.0, 0.5])

    def test_authoritative_hand_stream_precedes_warned_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary) / "episode_000000"
            streams = episode / "streams"
            streams.mkdir(parents=True)
            controller = timed_event(
                "controller",
                100_000_000,
                controller_age_ms=0.0,
                controller_buttons={
                    "left_grip_value": 1.0,
                    "right_grip_value": 0.0,
                },
            )
            hand_command = timed_event(
                "hand_command",
                100_000_000,
                sequence=1,
                active=True,
                left_grasp=0.2,
                right_grasp=0.4,
            )
            (streams / "controller.jsonl").write_text(
                json.dumps(controller) + "\n", encoding="utf-8"
            )
            hand_path = streams / "hand_command.jsonl"
            hand_path.write_text(json.dumps(hand_command) + "\n", encoding="utf-8")

            with mock.patch("warnings.warn") as warn:
                authoritative = _load_hand_command_series(episode)
            warn.assert_not_called()
            action, _, status = authoritative.sample(100_000_000)
            self.assertEqual(status, "ok")
            np.testing.assert_allclose(action, [0.2, 0.4])

            hand_path.unlink()
            with self.assertWarnsRegex(RuntimeWarning, "legacy controller grip"):
                legacy = _load_hand_command_series(episode)
            action, _, status = legacy.sample(100_000_000)
            self.assertEqual(status, "ok")
            np.testing.assert_allclose(action, [1.0, 0.0])

    def test_decoded_camera_can_be_rotated_180_degrees(self):
        try:
            import cv2
        except ImportError as exc:
            self.skipTest(str(exc))

        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        )
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "corners.png"
            image_path.write_bytes(encoded.tobytes())
            rotated = _decode_rgb(image_path, rotation_deg=180)

        np.testing.assert_array_equal(rotated, np.rot90(rgb, 2))

    def test_synchronizes_reference_joint_imu_and_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = RawEpisodeWriter(
                output_root=root,
                episode_index=0,
                task="touch the red button",
                start_monotonic_ns=1_000_000_000,
                start_wall_time_ns=2_000_000_000,
                source_config={},
            )
            all_q = [float(index) * 0.01 for index in range(29)]
            all_dq = [float(index) * 0.001 for index in range(29)]
            for sequence, timestamp_ns in enumerate(
                (1_000_000_000, 1_100_000_000, 1_200_000_000), start=1
            ):
                camera_timestamp_ns = (
                    timestamp_ns - 10_000_000 if sequence == 2 else timestamp_ns
                )
                writer.write_event(
                    timed_event(
                        "camera_head", camera_timestamp_ns, format="png", data=VALID_PNG
                    )
                )
                qpos = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *all_q]
                writer.write_event(
                    timed_event(
                        "reference",
                        timestamp_ns,
                        frame_dt_ns=20_000_000,
                        retarget_age_ms=0.0,
                        frames_qpos_root_xyz_quat_wxyz_dof=[qpos],
                    )
                )
                writer.write_event(
                    timed_event(
                        "hand_command",
                        timestamp_ns,
                        sequence=sequence,
                        active=True,
                        left_grasp=0.25,
                        right_grasp=0.75,
                    )
                )
                writer.write_event(
                    timed_event(
                        "joint_states",
                        timestamp_ns,
                        name=X2_TRACKING_JOINT_NAMES,
                        position=all_q,
                        velocity=all_dq,
                        effort=[0.0] * 29,
                    )
                )
                writer.write_event(
                    timed_event(
                        "imu_torso",
                        timestamp_ns,
                        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                        angular_velocity_xyz=[0.1, 0.2, 0.3],
                        linear_acceleration_xyz=[0.0, 0.0, 9.81],
                    )
                )
            episode = writer.finalize(
                status="complete",
                stop_trigger_monotonic_ns=1_200_000_000,
                success=True,
                ingress_drops={},
            )

            converted = build_episode_samples(
                episode,
                fps=10,
                max_camera_age_ms=60.0,
                max_reference_age_ms=60.0,
                max_reference_gap_ms=120.0,
                max_retarget_age_ms=60.0,
                max_hand_command_age_ms=60.0,
                max_joint_age_ms=60.0,
                max_imu_age_ms=60.0,
                time_basis=TIME_BASIS_RECEIVER,
            )
            self.assertEqual(converted.candidate_count, 3)
            self.assertEqual(len(converted.samples), 3)
            self.assertEqual(converted.samples[0].state.shape, (68,))
            self.assertEqual(converted.samples[0].action.shape, (38,))
            np.testing.assert_allclose(converted.samples[0].action[7:36], all_q)
            np.testing.assert_allclose(converted.samples[0].action[-2:], [0.25, 0.75])
            self.assertEqual(converted.samples[1].timing_ms[0], -10.0)
            self.assertEqual(_decode_rgb(converted.samples[0].image_path).shape, (1, 1, 3))

    def test_missing_ticks_are_split_instead_of_time_compressed(self):
        dummy = lambda timestamp_ns: ConvertedSample(  # noqa: E731
            timestamp_ns=timestamp_ns,
            image_path=Path("unused"),
            state=np.zeros(68, dtype=np.float32),
            action=np.zeros(38, dtype=np.float32),
            timing_ms=np.zeros(5, dtype=np.float32),
        )
        segments, dropped = split_contiguous_samples(
            [dummy(0), dummy(40_000_000), dummy(120_000_000), dummy(160_000_000)],
            fps=25,
            min_frames=2,
        )
        self.assertEqual([len(segment) for segment in segments], [2, 2])
        self.assertEqual(dropped, 0)

    def test_conversion_report_preserves_episode_segment_and_timestamp_provenance(self):
        first = ConvertedSample(
            timestamp_ns=1_000_000_000,
            observation_timestamp_ns=1_005_000_000,
            image_path=Path("first.jpg"),
            state=np.zeros(68, dtype=np.float32),
            action=np.zeros(38, dtype=np.float32),
            timing_ms=np.zeros(5, dtype=np.float32),
        )
        last = ConvertedSample(
            timestamp_ns=1_040_000_000,
            observation_timestamp_ns=1_038_000_000,
            image_path=Path("last.jpg"),
            state=np.zeros(68, dtype=np.float32),
            action=np.zeros(38, dtype=np.float32),
            timing_ms=np.zeros(5, dtype=np.float32),
        )
        episode = EpisodeSamples(
            episode_dir=Path("/raw/episode_000007"),
            task="reach the red marker",
            samples=[first, last],
            candidate_count=3,
            skip_counts={"camera_reused": 1},
            time_basis=TIME_BASIS_SOURCE,
            synchronization_diagnostics={
                "camera_head": {
                    "input_events": 3,
                    "timed_events": 3,
                    "missing_time_events": 0,
                    "source_regressions": 0,
                }
            },
        )
        payload = _conversion_report_payload(
            raw_root=Path("/raw"),
            output_root=Path("/converted"),
            repo_id="local/x2_vr",
            fps=25,
            time_basis=TIME_BASIS_SOURCE,
            allow_partial_source_time=False,
            max_camera_age_ms=100.0,
            max_reference_age_ms=60.0,
            max_reference_gap_ms=120.0,
            max_retarget_age_ms=100.0,
            max_hand_command_age_ms=100.0,
            max_joint_age_ms=100.0,
            max_imu_age_ms=100.0,
            camera_rotation_deg=180,
            min_segment_frames=2,
            require_success=True,
            converted=[episode],
            output_segments=[(episode, [first, last])],
            short_fragment_frames_dropped=1,
        )

        self.assertEqual(payload["schema_version"], CONVERSION_REPORT_SCHEMA_VERSION)
        self.assertEqual(payload["conversion"]["thresholds_ms"]["reference"], 60.0)
        self.assertEqual(payload["raw_episodes"][0]["output_segment_indices"], [0])
        segment = payload["output_segments"][0]
        self.assertEqual(segment["frame_count"], 2)
        self.assertEqual(segment["first_target_timestamp_ns"], 1_000_000_000)
        self.assertEqual(segment["last_camera_timestamp_ns"], 1_038_000_000)

        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "conversion_report.json"
            _write_conversion_report(report_path, payload)
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted, payload)

    def test_schema_uses_tracking_axis_order(self):
        self.assertEqual(X2_TRACKING_JOINT_NAMES[12:15], [
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ])
        self.assertEqual(X2_TRACKING_JOINT_NAMES[19:22], [
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
        ])
        self.assertEqual(
            HAND_ACTION_NAMES,
            ["hand.left_grasp_fraction", "hand.right_grasp_fraction"],
        )
        self.assertEqual(len(ACTION_NAMES), 38)


if __name__ == "__main__":
    unittest.main()
