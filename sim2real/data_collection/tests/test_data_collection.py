from __future__ import annotations

import base64
import json
import socket
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_COLLECTION_DIR))

from convert_to_lerobot import (  # noqa: E402
    ConvertedSample,
    _decode_rgb,
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
from schema import ACTION_NAMES, X2_TRACKING_JOINT_NAMES  # noqa: E402
from x2_vr_recorder import (  # noqa: E402
    DEFAULT_AIMDK_IMU_TOPICS,
    DEFAULT_AIMDK_JOINT_TOPICS,
    IngressQueue,
    SENSOR_PROFILES,
    _imu_stream_name,
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


class RawEpisodeManagerTest(unittest.TestCase):
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


class RecorderSensorAdapterTest(unittest.TestCase):
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


class ConversionTest(unittest.TestCase):
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
            for timestamp_ns in (1_000_000_000, 1_100_000_000, 1_200_000_000):
                writer.write_event(
                    timed_event("camera_head", timestamp_ns, format="png", data=VALID_PNG)
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
                max_joint_age_ms=60.0,
                max_imu_age_ms=60.0,
            )
            self.assertEqual(converted.candidate_count, 3)
            self.assertEqual(len(converted.samples), 3)
            self.assertEqual(converted.samples[0].state.shape, (68,))
            self.assertEqual(converted.samples[0].action.shape, (36,))
            np.testing.assert_allclose(converted.samples[0].action[7:], all_q)
            self.assertEqual(_decode_rgb(converted.samples[0].image_path).shape, (1, 1, 3))

    def test_missing_ticks_are_split_instead_of_time_compressed(self):
        dummy = lambda timestamp_ns: ConvertedSample(  # noqa: E731
            timestamp_ns=timestamp_ns,
            image_path=Path("unused"),
            state=np.zeros(68, dtype=np.float32),
            action=np.zeros(36, dtype=np.float32),
            timing_ms=np.zeros(4, dtype=np.float32),
        )
        segments, dropped = split_contiguous_samples(
            [dummy(0), dummy(40_000_000), dummy(120_000_000), dummy(160_000_000)],
            fps=25,
            min_frames=2,
        )
        self.assertEqual([len(segment) for segment in segments], [2, 2])
        self.assertEqual(dropped, 0)

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
        self.assertEqual(len(ACTION_NAMES), 36)


if __name__ == "__main__":
    unittest.main()
