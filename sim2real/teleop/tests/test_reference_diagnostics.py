from __future__ import annotations

import json
import sys
import threading
import unittest
from collections import deque
from pathlib import Path

import numpy as np


TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

import xrobot_teleop_to_pose_zmq_server as bridge_module  # noqa: E402
from xrobot_teleop_to_pose_zmq_server import (  # noqa: E402
    LowLatencyTeleopPoseZMQServer,
    RetargetedFrame,
    _nonnegative_age_ms,
    _nonnegative_duration_us,
)


EXPECTED_REFERENCE_DIAGNOSTIC_KEYS = {
    "bridge_request_to_reply_us",
    "latest_raw_motion_age_at_bridge_ms",
    "latest_raw_motion_sequence",
    "latest_retarget_age_at_bridge_ms",
    "latest_retarget_dropped_before_process",
    "latest_retarget_raw_motion_sequence",
    "latest_retarget_worker_compute_us",
    "latest_retarget_worker_queue_us",
    "reference_diagnostics_schema_version",
    "reference_sample_mode",
    "reference_support_dropped_before_process",
    "reference_support_retarget_raw_motion_sequence",
    "reference_support_worker_compute_us",
    "reference_support_worker_queue_us",
}


def _qpos(value: float) -> np.ndarray:
    qpos = np.full(36, value, dtype=np.float32)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    return qpos


def _frame(
    recv_ns: int,
    sequence: int,
    *,
    queue_us: int,
    compute_us: int,
    dropped: int,
) -> RetargetedFrame:
    return RetargetedFrame(
        recv_ns=recv_ns,
        qpos=_qpos(float(sequence)),
        raw_motion_sequence=sequence,
        worker_queue_us=queue_us,
        worker_compute_us=compute_us,
        worker_dropped_before_process=dropped,
    )


class ReferenceDiagnosticsTest(unittest.TestCase):
    def test_monotonic_helpers_reject_negative_order(self) -> None:
        self.assertEqual(_nonnegative_duration_us(1_000, 2_999), 1)
        self.assertEqual(_nonnegative_age_ms(2_500_000, 1_000_000), 1.5)
        self.assertIsNone(_nonnegative_duration_us(2_000, 1_999))
        self.assertIsNone(_nonnegative_duration_us(None, 1_999))
        self.assertIsNone(_nonnegative_age_ms(1_000, 1_001))

    def test_interpolation_uses_newer_support_frame(self) -> None:
        server = object.__new__(LowLatencyTeleopPoseZMQServer)
        server.default_qpos = _qpos(0.0)
        frames = [
            _frame(1_100_000_000, 10, queue_us=100, compute_us=10_000, dropped=0),
            _frame(1_140_000_000, 11, queue_us=200, compute_us=11_000, dropped=1),
            _frame(1_160_000_000, 12, queue_us=300, compute_us=12_000, dropped=2),
        ]

        _sample, fallback, info = server._sample_target_qpos(frames, 1_120_000_000)

        self.assertFalse(fallback)
        self.assertEqual(info["mode"], "interpolate")
        self.assertEqual(info["support_raw_motion_sequence"], 11)
        self.assertEqual(info["support_worker_queue_us"], 200)
        self.assertEqual(info["support_worker_compute_us"], 11_000)
        self.assertEqual(info["support_worker_dropped_before_process"], 1)

    def test_reply_diagnostics_match_downstream_wire_contract(self) -> None:
        server = object.__new__(LowLatencyTeleopPoseZMQServer)
        server.default_qpos = _qpos(0.0)
        server.latest_vr_lock = threading.Lock()
        server.latest_vr_recv_ns = 1_180_000_000
        server.latest_vr_seq = 13
        server.retarget_buffer_lock = threading.Lock()
        server.retarget_buffer = deque(
            [
                _frame(1_100_000_000, 10, queue_us=100, compute_us=10_000, dropped=0),
                _frame(1_140_000_000, 11, queue_us=200, compute_us=11_000, dropped=1),
                _frame(1_160_000_000, 12, queue_us=300, compute_us=12_000, dropped=2),
            ]
        )
        _sample, _fallback, sample_info = server._sample_target_qpos(
            list(server.retarget_buffer), 1_120_000_000
        )

        diagnostics = server._build_reference_diagnostics(
            sample_info=sample_info,
            req_recv_ns=1_190_000_000,
            reply_now_ns=1_200_000_000,
        )

        self.assertEqual(set(diagnostics), EXPECTED_REFERENCE_DIAGNOSTIC_KEYS)
        self.assertEqual(diagnostics["reference_diagnostics_schema_version"], 1)
        self.assertEqual(diagnostics["reference_sample_mode"], "interpolate")
        self.assertEqual(diagnostics["latest_raw_motion_age_at_bridge_ms"], 20.0)
        self.assertEqual(diagnostics["latest_retarget_age_at_bridge_ms"], 40.0)
        self.assertEqual(diagnostics["bridge_request_to_reply_us"], 10_000)
        self.assertEqual(diagnostics["latest_raw_motion_sequence"], 13)
        self.assertEqual(diagnostics["latest_retarget_raw_motion_sequence"], 12)
        self.assertEqual(diagnostics["latest_retarget_worker_queue_us"], 300)
        self.assertEqual(diagnostics["latest_retarget_dropped_before_process"], 2)
        self.assertEqual(
            diagnostics["reference_support_retarget_raw_motion_sequence"], 11
        )
        self.assertEqual(diagnostics["reference_support_worker_compute_us"], 11_000)
        self.assertEqual(diagnostics["reference_support_dropped_before_process"], 1)
        json.dumps(diagnostics, allow_nan=False)

    def test_worker_attaches_queue_compute_and_drop_provenance(self) -> None:
        class FakeRuntime:
            def __init__(self, _config):
                pass

            def process_packet(self, packet):
                return {
                    "type": "retarget_result",
                    "seq": packet["seq"],
                    "recv_ns": packet["recv_ns"],
                    "qpos": _qpos(1.0),
                }

        class FakeRawConnection:
            def __init__(self, packet):
                self.packet = packet
                self.poll_count = 0

            def poll(self, _timeout=None):
                self.poll_count += 1
                return self.poll_count in (1, 3)

            def recv(self):
                if self.poll_count == 1:
                    return self.packet
                return {"type": "shutdown"}

        class FakeResultConnection:
            def __init__(self):
                self.messages = []

            def send(self, payload):
                self.messages.append(payload)

        raw = FakeRawConnection(
            {"seq": 7, "recv_ns": bridge_module.time.monotonic_ns(), "poses": []}
        )
        result = FakeResultConnection()
        original_runtime = bridge_module._RetargetWorkerRuntime
        bridge_module._RetargetWorkerRuntime = FakeRuntime
        try:
            bridge_module._retarget_worker_main(raw, result, {})
        finally:
            bridge_module._RetargetWorkerRuntime = original_runtime

        self.assertEqual(result.messages[0]["type"], "worker_ready")
        worker_result = result.messages[1]
        self.assertEqual(worker_result["seq"], 7)
        self.assertGreaterEqual(worker_result["worker_queue_us"], 0)
        self.assertGreaterEqual(worker_result["worker_compute_us"], 0)
        self.assertEqual(worker_result["dropped_before_process"], 0)

    def test_no_data_and_invalid_order_are_null(self) -> None:
        server = object.__new__(LowLatencyTeleopPoseZMQServer)
        server.default_qpos = _qpos(0.0)
        server.latest_vr_lock = threading.Lock()
        server.latest_vr_recv_ns = 2_000
        server.latest_vr_seq = 0
        server.retarget_buffer_lock = threading.Lock()
        server.retarget_buffer = deque()
        _sample, _fallback, sample_info = server._sample_target_qpos([], 1_000)

        diagnostics = server._build_reference_diagnostics(
            sample_info=sample_info,
            req_recv_ns=2_000,
            reply_now_ns=1_999,
        )

        self.assertEqual(diagnostics["reference_sample_mode"], "default")
        self.assertIsNone(diagnostics["bridge_request_to_reply_us"])
        self.assertIsNone(diagnostics["latest_raw_motion_age_at_bridge_ms"])
        self.assertIsNone(diagnostics["latest_raw_motion_sequence"])
        for key, value in diagnostics.items():
            if key.startswith("reference_support_"):
                self.assertIsNone(value, key)


if __name__ == "__main__":
    unittest.main()
