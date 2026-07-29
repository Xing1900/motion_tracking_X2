from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_COLLECTION_DIR))

from convert_to_lerobot import (  # noqa: E402
    ConvertedSample,
    _decode_rgb,
    build_episode_samples,
    split_contiguous_samples,
)
from raw_episode_writer import RawEpisodeManager, RawEpisodeWriter  # noqa: E402
from schema import ACTION_NAMES, X2_TRACKING_JOINT_NAMES  # noqa: E402
from x2_vr_recorder import IngressQueue  # noqa: E402


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
