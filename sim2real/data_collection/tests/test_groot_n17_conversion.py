import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


DATA_COLLECTION_DIR = Path(__file__).resolve().parents[1]
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from convert_to_groot_n17 import (  # noqa: E402
    GrootSample,
    SequenceAwarePreviousSeries,
    _capture_contract,
    _quat_wxyz_to_rot6d,
    _reference_age_components,
    _reference_pipeline_diagnostics,
    _telemetry_payload,
    _validate_modality_json,
    _validate_manifest,
    build_episode_samples,
    localize_segment_root_references,
    split_contiguous_samples,
)
from schema import (  # noqa: E402
    GROOT_N17_ACTION_NAMES,
    GROOT_N17_STATE_NAMES,
    GROOT_N17_TIMING_NAMES,
    REFERENCE_DIAGNOSTIC_FIELD_NAMES,
    REFERENCE_DIAGNOSTICS_SCHEMA_VERSION,
    X2_TRACKING_JOINT_NAMES,
)


class GrootN17ConversionTest(unittest.TestCase):
    @staticmethod
    def _provenance():
        names = (
            "recorder",
            "controller_binary",
            "controller_config",
            "controller_policy",
            "controller_policy_data",
            "teleop_bridge",
            "gmr_config",
            "gmr_runtime",
            "hand_config",
        )
        return {
            name: {
                "path": f"/runtime/{name}",
                "size_bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, name in enumerate(names)
        }

    @staticmethod
    def _bridge_runtime_effective_params():
        return {
            "actual_human_height": 1.6,
            "gmr_max_iter": 5,
            "lookback_ms": 15.0,
            "min_link_height": 0.0,
            "min_link_height_align_strategy": "startup_fixed",
            "min_link_height_bootstrap_frames": 10,
        }

    def _telemetry(self):
        return {
            "schema_version": 1,
            "joint_count": 29,
            "joint_names": list(X2_TRACKING_JOINT_NAMES),
            "reference_root_position": [0.1, -0.2, 0.8],
            "reference_source_time_exact": True,
            "reference_root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "reference_joint_position": [0.01 * index for index in range(29)],
            "measured_joint_position": [0.0] * 29,
            "measured_joint_velocity": [0.0] * 29,
            "root_angular_velocity": [0.1, 0.2, 0.3],
            "projected_gravity": [0.0, 0.0, -1.0],
            "policy_action": [0.0] * 29,
            "command_joint_position": [0.0] * 29,
            "reference_diagnostics_schema_version": (
                REFERENCE_DIAGNOSTICS_SCHEMA_VERSION
            ),
            "reference_sample_mode": "interpolate",
            "latest_raw_motion_age_at_bridge_ms": 8.0,
            "latest_retarget_age_at_bridge_ms": 12.0,
            "bridge_request_to_reply_us": 300,
            "latest_raw_motion_sequence": 103,
            "latest_retarget_raw_motion_sequence": 102,
            "latest_retarget_worker_queue_us": 400,
            "latest_retarget_worker_compute_us": 1200,
            "latest_retarget_dropped_before_process": 0,
            "reference_support_retarget_raw_motion_sequence": 101,
            "reference_support_worker_queue_us": 450,
            "reference_support_worker_compute_us": 1250,
            "reference_support_dropped_before_process": 1,
        }

    def test_schema_dimensions(self):
        self.assertEqual(len(GROOT_N17_STATE_NAMES), 104)
        self.assertEqual(len(GROOT_N17_ACTION_NAMES), 40)
        self.assertEqual(len(GROOT_N17_TIMING_NAMES), 6)
        self.assertEqual(
            GROOT_N17_TIMING_NAMES[:4],
            [
                "camera_grid_offset_signed_ms",
                "tracking_telemetry_previous_age_ms",
                "hand_command_previous_age_ms",
                "consumed_reference_source_age_ms",
            ],
        )

    def test_manifest_requires_latest_hand_state_and_fixed_head(self):
        manifest = {
            "schema_version": "x2-vr-raw-v1",
            "robot_type": "agibot_x2",
            "task": "test",
            "joint_order": [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
                "left_wrist_pitch_joint", "left_wrist_roll_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
                "right_wrist_pitch_joint", "right_wrist_roll_joint",
            ],
            "source_config": {
                "record_profile": "groot_n17",
                "hand_status_delivery_semantics": "latest_state",
                "head_joint_assumption": "fixed_not_recorded",
                "tracking_telemetry_schema_version": 1,
                "tracking_telemetry_delivery_semantics": (
                    "bounded_nonblocking_sequence_checked"
                ),
                "tracking_telemetry_reference_diagnostics_schema_version": 1,
                "tracking_telemetry_reference_diagnostics_semantics": (
                    "bridge_gmr_root_cause_v1"
                ),
                "bridge_runtime_effective_params": (
                    self._bridge_runtime_effective_params()
                ),
                "capture_provenance": self._provenance(),
            },
        }
        _validate_manifest(manifest, Path("episode_000000"))

        bad_hand = json.loads(json.dumps(manifest))
        bad_hand["source_config"]["hand_status_delivery_semantics"] = "history"
        with self.assertRaisesRegex(ValueError, "latest-state"):
            _validate_manifest(bad_hand, Path("episode_000000"))

        bad_head = json.loads(json.dumps(manifest))
        bad_head["source_config"].pop("head_joint_assumption")
        with self.assertRaisesRegex(ValueError, "fixed-head"):
            _validate_manifest(bad_head, Path("episode_000000"))

        missing_recorder = json.loads(json.dumps(manifest))
        missing_recorder["source_config"]["capture_provenance"].pop("recorder")
        with self.assertRaisesRegex(ValueError, "missing capture provenance: recorder"):
            _validate_manifest(missing_recorder, Path("episode_000000"))

        changed_runtime = json.loads(json.dumps(manifest))
        changed_runtime["source_config"]["bridge_runtime_effective_params"][
            "lookback_ms"
        ] = 20.0
        self.assertNotEqual(
            _capture_contract(manifest, Path("episode_000000")),
            _capture_contract(changed_runtime, Path("episode_000001")),
        )

        missing_runtime_param = json.loads(json.dumps(manifest))
        missing_runtime_param["source_config"]["bridge_runtime_effective_params"].pop(
            "gmr_max_iter"
        )
        with self.assertRaisesRegex(ValueError, "bridge runtime effective params"):
            _validate_manifest(missing_runtime_param, Path("episode_000000"))

        bad_diagnostics = json.loads(json.dumps(manifest))
        bad_diagnostics["source_config"][
            "tracking_telemetry_reference_diagnostics_semantics"
        ] = "unknown"
        with self.assertRaisesRegex(ValueError, "reference diagnostics"):
            _validate_manifest(bad_diagnostics, Path("episode_000000"))

    def test_quaternion_to_gr00t_row_major_rot6d(self):
        angle = math.pi / 2.0
        quaternion = np.asarray([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
        actual = _quat_wxyz_to_rot6d(quaternion)
        np.testing.assert_allclose(actual, [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(
            _quat_wxyz_to_rot6d(-quaternion), actual, atol=1e-7
        )

    def test_telemetry_parser_is_strict_and_finite(self):
        parsed = _telemetry_payload(self._telemetry())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["reference_joint"].shape, (29,))
        self.assertEqual(parsed["root_rot6d"].shape, (6,))

        bad = self._telemetry()
        bad["measured_joint_position"] = [0.0] * 28
        self.assertIsNone(_telemetry_payload(bad))
        bad = self._telemetry()
        bad["projected_gravity"][1] = float("nan")
        self.assertIsNone(_telemetry_payload(bad))

    def test_reference_age_split_uses_upstream_not_total_for_freshness(self):
        event = {
            "reference_source_age_ms": 120.0,
            "reference_total_age_ms": 120.0,
            "reference_upstream_age_at_bridge_ms": 20.0,
            "reference_bridge_to_policy_age_ms": 100.0,
        }
        ages, status = _reference_age_components(event)
        self.assertEqual(status, "ok")
        self.assertEqual(
            ages,
            {"upstream_ms": 20.0, "total_ms": 120.0, "bridge_to_policy_ms": 100.0},
        )

        legacy = {"reference_source_age_ms": 120.0}
        self.assertEqual(
            _reference_age_components(legacy)[1],
            "reference_upstream_age_missing_legacy",
        )

        alias_mismatch = dict(event, reference_total_age_ms=121.0)
        self.assertEqual(
            _reference_age_components(alias_mismatch)[1],
            "reference_total_age_alias_mismatch",
        )

        split_mismatch = dict(event, reference_bridge_to_policy_age_ms=99.0)
        self.assertEqual(
            _reference_age_components(split_mismatch)[1],
            "reference_age_split_inconsistent",
        )

        tolerance_boundary = dict(
            event,
            reference_total_age_ms=120.5,
            reference_bridge_to_policy_age_ms=100.5,
        )
        self.assertEqual(_reference_age_components(tolerance_boundary)[1], "ok")

        missing_bridge_component = dict(event)
        del missing_bridge_component["reference_bridge_to_policy_age_ms"]
        self.assertEqual(
            _reference_age_components(missing_bridge_component)[1],
            "reference_age_split_missing",
        )

    def test_reference_pipeline_diagnostics_classify_without_changing_gate(self):
        base = self._telemetry()
        base.update(
            {
                "vr_session_active": True,
                "reference_source_time_exact": True,
                "reference_is_transition": False,
                "reference_is_padded": False,
                "reference_is_fallback": True,
                "reference_upstream_age_at_bridge_ms": 90.0,
            }
        )
        xr_stale = dict(
            base,
            latest_raw_motion_age_at_bridge_ms=120.0,
            latest_retarget_age_at_bridge_ms=130.0,
        )
        gmr_stale = dict(
            base,
            latest_raw_motion_age_at_bridge_ms=10.0,
            latest_retarget_age_at_bridge_ms=120.0,
            reference_sample_mode="fallback_latest",
        )
        selected_old = dict(
            base,
            latest_raw_motion_age_at_bridge_ms=10.0,
            latest_retarget_age_at_bridge_ms=20.0,
            reference_sample_mode="fallback_oldest",
        )
        unknown = dict(
            base,
            reference_diagnostics_schema_version=None,
            latest_raw_motion_age_at_bridge_ms=None,
        )
        fresh = dict(base, reference_upstream_age_at_bridge_ms=20.0)

        report = _reference_pipeline_diagnostics(
            [xr_stale, gmr_stale, selected_old, unknown, fresh],
            max_upstream_age_ms=80.0,
        )

        upstream = report["upstream_freshness"]
        self.assertEqual(upstream["eligible_operator_reference_events"], 5)
        self.assertEqual(upstream["fresh_events"], 1)
        self.assertEqual(upstream["stale_events"], 4)
        self.assertEqual(
            upstream["stale_root_cause_counts"],
            {
                "diagnostics_unknown": 1,
                "gmr_output_stale_with_fresh_raw": 1,
                "selected_reference_stale_with_fresh_latest": 1,
                "xr_body_input_stale": 1,
            },
        )
        self.assertEqual(upstream["stale_fallback_events"], 4)
        self.assertEqual(
            report["derived_distributions"][
                "latest_raw_minus_retarget_sequence"
            ]["p50"],
            1.0,
        )
        self.assertEqual(
            report["numeric_distributions"]["bridge_request_to_reply_us"][
                "count"
            ],
            5,
        )

    def test_reference_pipeline_diagnostics_honor_explicit_legacy_marker(self):
        legacy = self._telemetry()
        legacy["reference_diagnostics_fields_present"] = False
        for key in REFERENCE_DIAGNOSTIC_FIELD_NAMES:
            legacy[key] = None

        report = _reference_pipeline_diagnostics(
            [legacy], max_upstream_age_ms=80.0
        )

        contract = report["contract"]
        self.assertEqual(contract["all_fields_present_events"], 0)
        self.assertEqual(contract["diagnostics_schema_v1_events"], 0)
        self.assertEqual(
            set(contract["structurally_missing_by_field"]),
            set(REFERENCE_DIAGNOSTIC_FIELD_NAMES),
        )
        self.assertTrue(
            all(
                count == 1
                for count in contract["structurally_missing_by_field"].values()
            )
        )

    def test_segment_requires_true_25hz_continuity(self):
        def sample(timestamp_ns):
            return GrootSample(
                timestamp_ns=timestamp_ns,
                observation_timestamp_ns=timestamp_ns,
                image_path=Path("image.jpg"),
                state=np.zeros(104, dtype=np.float32),
                action=np.zeros(40, dtype=np.float32),
                timing_ms=np.zeros(6, dtype=np.float32),
                telemetry_sequence=timestamp_ns,
                tracking_error_sq=0.0,
                command_error_sq=0.0,
                global_root_reference_xyz_rot6d=np.zeros(9, dtype=np.float32),
            )

        samples = [sample(index * 40_000_000) for index in range(40)]
        samples.extend(sample((index + 42) * 40_000_000) for index in range(40))
        segments, dropped = split_contiguous_samples(samples, fps=25, min_frames=40)
        self.assertEqual([len(segment) for segment in segments], [40, 40])
        self.assertEqual(dropped, 0)

    def test_sequence_aware_previous_rejects_gap_and_reset_open_intervals(self):
        def event(timestamp_ns, sequence, **extra):
            return {
                "recorder_recv_monotonic_ns": timestamp_ns,
                "sequence": sequence,
                **extra,
            }

        gap = SequenceAwarePreviousSeries(
            [event(0, 10), event(20_000_000, 12)],
            sequence_modulus=1 << 64,
        )
        self.assertEqual(gap.previous(0)[2], "ok")
        self.assertEqual(gap.previous(10_000_000)[2], "sequence_gap")
        self.assertEqual(gap.previous(20_000_000)[2], "ok")

        reset = SequenceAwarePreviousSeries(
            [
                event(0, 100),
                event(
                    20_000_000,
                    0,
                    tracking_telemetry_sequence_reset=True,
                ),
            ],
            sequence_modulus=1 << 64,
            reset_field="tracking_telemetry_sequence_reset",
        )
        self.assertEqual(reset.previous(10_000_000)[2], "sequence_reset")
        self.assertEqual(reset.previous(20_000_000)[2], "ok")

    def test_sequence_aware_previous_accepts_uint32_and_uint64_wrap(self):
        for modulus in (1 << 32, 1 << 64):
            with self.subTest(modulus=modulus):
                series = SequenceAwarePreviousSeries(
                    [
                        {
                            "recorder_recv_monotonic_ns": 0,
                            "sequence": modulus - 1,
                        },
                        {
                            "recorder_recv_monotonic_ns": 20_000_000,
                            "sequence": 0,
                            # The telemetry receiver marks every numeric
                            # decrease as reset; a true wrap remains contiguous.
                            "tracking_telemetry_sequence_reset": True,
                        },
                    ],
                    sequence_modulus=modulus,
                    reset_field="tracking_telemetry_sequence_reset",
                )
                self.assertEqual(series.previous(10_000_000)[2], "ok")

    def test_segment_root_is_expressed_in_first_reference_frame(self):
        def make_sample(position, quaternion):
            rot6d = _quat_wxyz_to_rot6d(np.asarray(quaternion, dtype=np.float64))
            pose = np.concatenate([position, rot6d]).astype(np.float32)
            state = np.zeros(104, dtype=np.float32)
            action = np.zeros(40, dtype=np.float32)
            state[64:73] = pose
            action[:9] = pose
            return GrootSample(
                timestamp_ns=0,
                observation_timestamp_ns=0,
                image_path=Path("image.jpg"),
                state=state,
                action=action,
                timing_ms=np.zeros(6, dtype=np.float32),
                telemetry_sequence=0,
                tracking_error_sq=0.0,
                command_error_sq=0.0,
                global_root_reference_xyz_rot6d=pose.copy(),
            )

        z90 = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        z180 = [0.0, 0.0, 0.0, 1.0]
        samples = [
            make_sample([1.0, 2.0, 0.8], z90),
            make_sample([1.0, 3.0, 0.8], z180),
        ]
        localize_segment_root_references(samples)
        np.testing.assert_allclose(
            samples[0].action[:9],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(samples[1].action[:3], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(samples[1].state[64:73], samples[1].action[:9])

    def test_modality_ranges_match_vectors(self):
        modality_path = (
            Path(__file__).resolve().parents[4]
            / "Isaac-GR00T"
            / "examples"
            / "X2"
            / "modality.json"
        )
        # The repositories are normally siblings.  Keep this test portable to
        # a checkout that only contains motion_tracking.
        if not modality_path.is_file():
            self.skipTest("Isaac-GR00T sibling checkout is not available")
        modality = json.loads(modality_path.read_text(encoding="utf-8"))
        self.assertEqual(modality["state"]["current_grasp"]["end"], 104)
        self.assertEqual(modality["action"]["grasp"]["end"], 40)
        self.assertEqual(len(_validate_modality_json(modality_path)), 64)

        with tempfile.TemporaryDirectory() as temp_dir:
            wrong_path = Path(temp_dir) / "modality.json"
            modality["action"]["grasp"]["end"] = 39
            wrong_path.write_text(json.dumps(modality), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact X2 N1.7"):
                _validate_modality_json(wrong_path)

    def test_source_time_build_uses_atomic_controller_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = Path(temp_dir) / "episode_000000"
            streams = episode / "streams"
            images = episode / "images" / "head_rgb"
            streams.mkdir(parents=True)
            images.mkdir(parents=True)
            base_mono = 1_000_000_000_000
            base_wall = 2_000_000_000_000_000_000
            frame_count = 50
            step_ns = 40_000_000
            manifest = {
                "schema_version": "x2-vr-raw-v1",
                "status": "complete",
                "robot_type": "agibot_x2",
                "task": "walk to and touch the marker",
                "joint_order": [
                    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
                    "left_wrist_pitch_joint", "left_wrist_roll_joint",
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
                    "right_wrist_pitch_joint", "right_wrist_roll_joint",
                ],
                "source_config": {
                    "record_profile": "groot_n17",
                    "hand_status_delivery_semantics": "latest_state",
                    "head_joint_assumption": "fixed_not_recorded",
                    "tracking_telemetry_schema_version": 1,
                    "tracking_telemetry_delivery_semantics": (
                        "bounded_nonblocking_sequence_checked"
                    ),
                    "tracking_telemetry_reference_diagnostics_schema_version": 1,
                    "tracking_telemetry_reference_diagnostics_semantics": (
                        "bridge_gmr_root_cause_v1"
                    ),
                    "bridge_runtime_effective_params": (
                        self._bridge_runtime_effective_params()
                    ),
                    "capture_provenance": self._provenance(),
                },
                "recording": {
                    "start_monotonic_ns": base_mono,
                    "stop_trigger_monotonic_ns": base_mono + (frame_count - 1) * step_ns,
                },
            }
            (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            camera_events = []
            telemetry_events = []
            hand_events = []
            for index in range(frame_count):
                camera_mono = base_mono + index * step_ns
                camera_wall = base_wall + index * step_ns
                image_path = images / f"{index:06d}.jpg"
                image_path.write_bytes(b"jpeg-placeholder")
                camera_events.append(
                    {
                        "source_timestamp_ns": camera_wall,
                        "recorder_recv_monotonic_ns": camera_mono + 5_000_000,
                        "recorder_recv_wall_time_ns": camera_wall + 5_000_000,
                        "image_path": str(image_path.relative_to(episode)),
                    }
                )

                # The controller and hand samples precede exposure by 10 ms;
                # their receive latency must not alter physical alignment.
                sample_mono = camera_mono - 10_000_000
                sample_wall = camera_wall - 10_000_000
                telemetry = self._telemetry()
                telemetry.update(
                    {
                        "sequence": index,
                        "sample_monotonic_ns": sample_mono,
                        "sample_wall_time_ns": sample_wall,
                        "recorder_recv_monotonic_ns": camera_mono + 9_000_000,
                        "recorder_recv_wall_time_ns": camera_wall + 9_000_000,
                        "vr_user_enabled": True,
                        "vr_session_active": True,
                        # The intentional future queue makes total age exceed
                        # 80 ms, while the upstream/GMR selection is fresh.
                        "reference_source_age_ms": 120.0,
                        "reference_total_age_ms": 120.0,
                        "reference_upstream_age_at_bridge_ms": 20.0,
                        "reference_bridge_to_policy_age_ms": 100.0,
                        "reference_is_transition": False,
                        "reference_is_padded": False,
                        "reference_is_fallback": False,
                    }
                )
                telemetry_events.append(telemetry)
                hand_events.append(
                    {
                        "source_timestamp_ns": sample_wall,
                        "recorder_recv_monotonic_ns": camera_mono + 8_000_000,
                        "recorder_recv_wall_time_ns": camera_wall + 8_000_000,
                        "sequence": index,
                        "active": True,
                        "left_grasp": 0.25,
                        "right_grasp": 0.75,
                    }
                )

            def write_stream(name, events):
                with (streams / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for event in events:
                        handle.write(json.dumps(event) + "\n")

            for name, events in (
                ("camera_head", camera_events),
                ("tracking_telemetry", telemetry_events),
                ("hand_command", hand_events),
            ):
                write_stream(name, events)

            converted = build_episode_samples(episode)
            self.assertEqual(converted.candidate_count, frame_count)
            self.assertEqual(len(converted.samples), frame_count)
            self.assertEqual(converted.skip_counts, {})
            first = converted.samples[0]
            self.assertEqual(first.state.shape, (104,))
            self.assertEqual(first.action.shape, (40,))
            self.assertEqual(first.timing_ms.shape, (6,))
            np.testing.assert_allclose(first.timing_ms[3:], [120.0, 20.0, 100.0])
            self.assertEqual(
                converted.reference_pipeline_diagnostics["contract"][
                    "diagnostics_schema_v1_events"
                ],
                frame_count - 1,
            )
            self.assertEqual(
                converted.reference_pipeline_diagnostics[
                    "numeric_distributions"
                ]["reference_support_worker_compute_us"]["p95"],
                1250.0,
            )
            np.testing.assert_allclose(first.state[64:73], first.action[:9])
            np.testing.assert_allclose(first.state[73:102], first.action[9:38])
            np.testing.assert_allclose(first.action[38:40], [0.25, 0.75])

            fresh_fallback = [
                dict(event, reference_is_fallback=True)
                for event in telemetry_events
            ]
            write_stream("tracking_telemetry", fresh_fallback)
            fallback_accepted = build_episode_samples(episode)
            self.assertEqual(len(fallback_accepted.samples), frame_count)

            transition = [
                dict(event, reference_is_transition=True)
                for event in telemetry_events
            ]
            write_stream("tracking_telemetry", transition)
            transition_rejected = build_episode_samples(episode)
            self.assertEqual(len(transition_rejected.samples), 0)
            self.assertEqual(
                transition_rejected.skip_counts["reference_transition"],
                frame_count,
            )

            padded = [
                dict(event, reference_is_padded=True)
                for event in telemetry_events
            ]
            write_stream("tracking_telemetry", padded)
            padded_rejected = build_episode_samples(episode)
            self.assertEqual(len(padded_rejected.samples), 0)
            self.assertEqual(
                padded_rejected.skip_counts["reference_padded"],
                frame_count,
            )

            stale_upstream = [
                dict(
                    event,
                    reference_is_fallback=True,
                    reference_upstream_age_at_bridge_ms=90.0,
                    reference_bridge_to_policy_age_ms=30.0,
                )
                for event in telemetry_events
            ]
            write_stream("tracking_telemetry", stale_upstream)
            upstream_rejected = build_episode_samples(episode)
            self.assertEqual(len(upstream_rejected.samples), 0)
            self.assertEqual(
                upstream_rejected.skip_counts["reference_upstream_stale"],
                frame_count,
            )

            legacy_only = []
            for event in telemetry_events:
                legacy_event = dict(event)
                legacy_event.pop("reference_total_age_ms")
                legacy_event.pop("reference_upstream_age_at_bridge_ms")
                legacy_event.pop("reference_bridge_to_policy_age_ms")
                legacy_only.append(legacy_event)
            write_stream("tracking_telemetry", legacy_only)
            legacy_rejected = build_episode_samples(episode)
            self.assertEqual(len(legacy_rejected.samples), 0)
            self.assertEqual(
                legacy_rejected.skip_counts[
                    "reference_upstream_age_missing_legacy"
                ],
                frame_count,
            )

            # A single missing 25 Hz telemetry publication would otherwise be
            # accepted at exactly the 50 ms age threshold. Reject only the
            # proven-loss open interval; the absent grid tick then cuts the
            # output into two truly contiguous segments.
            write_stream(
                "tracking_telemetry",
                [event for event in telemetry_events if event["sequence"] != 20],
            )
            telemetry_gap = build_episode_samples(episode)
            self.assertEqual(len(telemetry_gap.samples), frame_count - 2)
            self.assertEqual(telemetry_gap.skip_counts["tracking_sequence_gap"], 2)
            self.assertEqual(telemetry_gap.telemetry_sequence_gaps, 1)
            segments, dropped = split_contiguous_samples(
                telemetry_gap.samples, fps=25, min_frames=1
            )
            self.assertEqual([len(segment) for segment in segments], [19, 29])
            self.assertEqual(dropped, 0)

            # Apply the same rule to latest-state hand status. Restore telemetry
            # first so the hand-specific skip reason is isolated.
            write_stream("tracking_telemetry", telemetry_events)
            write_stream(
                "hand_command",
                [event for event in hand_events if event["sequence"] != 20],
            )
            hand_gap = build_episode_samples(episode)
            self.assertEqual(len(hand_gap.samples), frame_count - 2)
            self.assertEqual(hand_gap.skip_counts["hand_sequence_gap"], 2)
            segments, dropped = split_contiguous_samples(
                hand_gap.samples, fps=25, min_frames=1
            )
            self.assertEqual([len(segment) for segment in segments], [19, 29])
            self.assertEqual(dropped, 0)


if __name__ == "__main__":
    unittest.main()
