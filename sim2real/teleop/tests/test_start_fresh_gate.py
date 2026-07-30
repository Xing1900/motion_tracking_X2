from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path

import numpy as np


TELEOP_DIR = Path(__file__).resolve().parents[1]
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from xrobot_teleop_to_pose_zmq_server import (  # noqa: E402
    LowLatencyTeleopPoseZMQServer,
    RetargetedFrame,
)


def _qpos(marker: float) -> np.ndarray:
    return np.full(36, marker, dtype=np.float32)


def _make_server(*, timeout_s: float = 0.03) -> LowLatencyTeleopPoseZMQServer:
    server = object.__new__(LowLatencyTeleopPoseZMQServer)
    server.start_fresh_frames = 3
    server.start_max_retarget_age_ns = int(80e6)
    server.start_fresh_wait_timeout_s = timeout_s
    server.retarget_buffer_window_ns = int(0.5e9)
    server.retarget_buffer_lock = threading.Lock()
    server.retarget_buffer_condition = threading.Condition(server.retarget_buffer_lock)
    server.retarget_buffer = deque()
    server.latest_vr_lock = threading.Lock()
    server.controller_control_epoch = 1
    server.controller_start_active = True
    server.controller_start_edge_recv_ns = 0
    server.stop_event = threading.Event()
    return server


def _controller_snapshot(*, right: bool = False, left: bool = False) -> dict:
    return {
        "timestamp_ns": time.time_ns(),
        "controllers": {
            "left": {"primary_button": left},
            "right": {"primary_button": right},
        },
        "body": {"available": False},
    }


class StartFreshFrameGateTest(unittest.TestCase):
    def test_request_loop_replies_with_newest_validated_start_frame(self) -> None:
        server = _make_server(timeout_s=0.02)
        now_ns = time.monotonic_ns()
        request_recv_ns = now_ns - int(50e6)
        server.controller_start_edge_recv_ns = now_ns - int(60e6)
        for offset_ms, marker in ((40, 1.0), (25, 2.0), (10, 3.0)):
            server._append_retarget_frame(now_ns - int(offset_ms * 1e6), _qpos(marker))

        server.req_count = 0
        server.req_merged_total = 0
        server.latest_merged_reqs = 0
        server.last_req_monotonic = None
        server.start_gate_wait_count = 0
        server.start_gate_ready_count = 0
        server.fallback_count = 0
        server.reply_count = 0
        server.reply_drop_count = 0
        server.frame_seq = 0
        server.stats_lock = threading.Lock()
        server.latest_debug_info = {}
        server.latest_vr_recv_ns = 0
        server.tap_accepting = False
        server._drain_requests_blocking = lambda: (
            {"start": True},
            request_recv_ns,
            1,
        )

        payloads: list[dict] = []

        class _FakePushSocket:
            def send_string(self, payload: str, flags: int = 0) -> None:
                del flags
                payloads.append(json.loads(payload))
                server.stop_event.set()

        server.rep_sock = _FakePushSocket()
        server._request_loop()

        self.assertEqual(len(payloads), 1)
        self.assertTrue(payloads[0]["start"])
        self.assertFalse(payloads[0]["no_interp_applied"])
        self.assertEqual(payloads[0]["frames"][0]["root_pos"], [3.0, 3.0, 3.0])
        self.assertLessEqual(payloads[0]["retarget_age_ms"], 80)

    def test_controller_edges_advance_epoch_and_stop_wins(self) -> None:
        server = _make_server()
        server.last_controller_buttons = {}
        server.latest_vr_poses = None
        server.latest_vr_recv_ns = 0
        server.latest_vr_seq = 0
        server.latest_vr_motion_timestamp_ns = None
        server.latest_controller_source_timestamp_ns = None
        server.latest_controller_recv_monotonic_ns = 0
        server.latest_controller_recv_wall_time_ns = 0
        server.callback_count = 0
        server.vr_frame_event = threading.Event()
        server.tap_accepting = False

        server._on_vr_frame(_controller_snapshot(right=True))
        self.assertEqual(server.controller_control_epoch, 2)
        self.assertTrue(server.controller_start_active)

        server._on_vr_frame(_controller_snapshot(right=True))
        self.assertEqual(server.controller_control_epoch, 2)

        server._on_vr_frame(_controller_snapshot())
        server._on_vr_frame(_controller_snapshot(right=True, left=True))
        self.assertEqual(server.controller_control_epoch, 3)
        self.assertFalse(server.controller_start_active)

        server._on_vr_frame(_controller_snapshot())
        server._on_vr_frame(_controller_snapshot(right=True))
        self.assertEqual(server.controller_control_epoch, 4)
        self.assertTrue(server.controller_start_active)

    def test_stale_tail_is_not_accepted(self) -> None:
        server = _make_server(timeout_s=0.01)
        now_ns = time.monotonic_ns()
        with server.retarget_buffer_condition:
            server.retarget_buffer.append(
                RetargetedFrame(recv_ns=now_ns - int(2e9), qpos=_qpos(1.0))
            )

        selected, info = server._wait_for_fresh_start_qpos(cutoff_ns=now_ns)

        self.assertIsNone(selected)
        self.assertEqual(info["mode"], "start_wait_fresh")
        self.assertEqual(info["fresh_frame_count"], 0)
        self.assertGreater(info["latest_age_ms"], 1000.0)

    def test_inflight_old_result_is_ignored_until_three_new_frames(self) -> None:
        server = _make_server(timeout_s=0.2)
        cutoff_ns = time.monotonic_ns()
        result: list[tuple[np.ndarray | None, dict]] = []

        waiter = threading.Thread(
            target=lambda: result.append(
                server._wait_for_fresh_start_qpos(cutoff_ns=cutoff_ns)
            )
        )
        waiter.start()

        # This simulates a GMR result that completed after start but originated
        # from a raw XR packet received before the start cutoff.
        server._append_retarget_frame(cutoff_ns - 1, _qpos(9.0))
        for marker in (1.0, 2.0, 3.0):
            time.sleep(0.005)
            server._append_retarget_frame(time.monotonic_ns(), _qpos(marker))

        waiter.join(timeout=1.0)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(result), 1)
        selected, info = result[0]
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected, _qpos(3.0))
        self.assertEqual(info["mode"], "start_fresh")
        self.assertEqual(info["fresh_frame_count"], 3)

    def test_post_start_frames_must_also_be_recent(self) -> None:
        server = _make_server(timeout_s=0.01)
        now_ns = time.monotonic_ns()
        cutoff_ns = now_ns - int(300e6)
        with server.retarget_buffer_condition:
            for offset_ms, marker in ((180, 1.0), (160, 2.0), (140, 3.0)):
                server.retarget_buffer.append(
                    RetargetedFrame(
                        recv_ns=now_ns - int(offset_ms * 1e6),
                        qpos=_qpos(marker),
                    )
                )

        selected, info = server._wait_for_fresh_start_qpos(cutoff_ns=cutoff_ns)

        self.assertIsNone(selected)
        self.assertEqual(info["fresh_frame_count"], 0)
        self.assertGreater(info["latest_age_ms"], 80.0)

    def test_stop_cancels_a_waiting_start_epoch(self) -> None:
        server = _make_server(timeout_s=0.5)
        cutoff_ns = time.monotonic_ns()
        result: list[tuple[np.ndarray | None, dict]] = []

        waiter = threading.Thread(
            target=lambda: result.append(
                server._wait_for_fresh_start_qpos(
                    cutoff_ns=cutoff_ns,
                    expected_epoch=1,
                )
            )
        )
        waiter.start()
        time.sleep(0.01)

        with server.latest_vr_lock:
            server.controller_control_epoch = 2
            server.controller_start_active = False
        with server.retarget_buffer_condition:
            server.retarget_buffer_condition.notify_all()

        waiter.join(timeout=0.2)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(result), 1)
        selected, info = result[0]
        self.assertIsNone(selected)
        self.assertEqual(info["mode"], "start_cancelled")

    def test_new_start_epoch_invalidates_cached_selection(self) -> None:
        server = _make_server()
        now_ns = time.monotonic_ns()
        selected_recv_ns = now_ns - int(10e6)

        self.assertTrue(
            server._is_start_gate_current(
                expected_epoch=1,
                selected_recv_ns=selected_recv_ns,
                now_ns=now_ns,
            )
        )
        with server.latest_vr_lock:
            server.controller_control_epoch = 2
            server.controller_start_active = True

        self.assertFalse(
            server._is_start_gate_current(
                expected_epoch=1,
                selected_recv_ns=selected_recv_ns,
                now_ns=now_ns,
            )
        )

    def test_cached_start_selection_expires(self) -> None:
        server = _make_server()
        now_ns = time.monotonic_ns()

        self.assertFalse(
            server._is_start_gate_current(
                expected_epoch=1,
                selected_recv_ns=now_ns - int(81e6),
                now_ns=now_ns,
            )
        )


if __name__ == "__main__":
    unittest.main()
