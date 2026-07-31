#!/usr/bin/env python3
"""
Low-latency PICO/XRobot teleop bridge for sim2real.

Architecture:
1. XR callback thread stores the latest VR snapshot with a monotonic timestamp.
2. A retarget thread waits for new VR data and only retargets the latest snapshot.
3. A request thread serves the newest ZMQ request using time-based interpolation over a
   short retarget history buffer.
4. A control thread publishes controller buttons at a fixed rate.
"""

import argparse
import json
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from default_mimic_obs import DEFAULT_MIMIC_OBS

GMR = None
RobotMotionViewer = None
quat_mul_np = None
xrt = None

XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]


def _load_runtime_dependencies(*, visualize: bool = False) -> None:
    global GMR, RobotMotionViewer, quat_mul_np, xrt

    try:
        from general_motion_retargeting import GeneralMotionRetargeting as _GMR
        from general_motion_retargeting.rot_utils import quat_mul_np as _quat_mul_np

        if visualize:
            from general_motion_retargeting import RobotMotionViewer as _RobotMotionViewer
        else:
            _RobotMotionViewer = None
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'general_motion_retargeting'. Install GMR in the active Python environment."
        ) from exc

    try:
        import xrobotoolkit_sdk as _xrt
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'xrobotoolkit_sdk'. Install the patched SDK in the active Python environment."
        ) from exc

    for name in (
        "register_frame_callback",
        "clear_frame_callback",
        "has_frame_callback",
    ):
        if not hasattr(_xrt, name):
            raise ImportError(
                "Installed xrobotoolkit_sdk does not expose callback APIs. "
                "Reinstall the patched XRoboToolkit-PC-Service-Pybind build."
            )

    GMR = _GMR
    RobotMotionViewer = _RobotMotionViewer
    quat_mul_np = _quat_mul_np
    xrt = _xrt


@dataclass
class RetargetedFrame:
    recv_ns: int
    qpos: np.ndarray


class _RetargetWorkerRuntime:
    ROBOT_GROUND_REFERENCE_BODY_NAMES = ("left_toe_link", "right_toe_link")

    def __init__(self, worker_config: Dict[str, Any]):
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.rot_utils import quat_mul_np as worker_quat_mul_np

        self._quat_mul_np = worker_quat_mul_np
        self.retarget = GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot=str(worker_config.get("robot", "unitree_g1")),
            actual_human_height=float(worker_config["actual_human_height"]),
        )
        self.retarget.max_iter = int(worker_config["gmr_max_iter"])
        self.send_human_motion = bool(worker_config["send_human_motion"])
        self.min_link_height = float(worker_config["min_link_height"])
        self.min_link_height_align_strategy = str(worker_config["min_link_height_align_strategy"])
        self.min_link_height_bootstrap_frames = max(1, int(worker_config["min_link_height_bootstrap_frames"]))
        self.fixed_min_link_height_offset: Optional[float] = None
        self.min_link_height_offset_samples: list[float] = []
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.rotation_quat = R.from_matrix(self.rotation_matrix).as_quat(scalar_first=True)

    def _body_poses_to_pose_dict(self, poses: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(poses, (list, tuple)) or len(poses) < len(XR_BODY_JOINT_NAMES):
            return None

        body_pose_dict: Dict[str, Any] = {}
        for i, joint_name in enumerate(XR_BODY_JOINT_NAMES):
            pose = poses[i]
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                return None
            x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
            pos = np.array([x, y, z], dtype=np.float64) @ self.rotation_matrix.T
            rot = self._quat_mul_np(
                self.rotation_quat.reshape(1, 4),
                np.array([[qw, qx, qy, qz]], dtype=np.float64),
                scalar_first=True,
            )[0]
            body_pose_dict[joint_name] = [pos.tolist(), rot.tolist()]
        return body_pose_dict

    def _get_current_min_body_z(self) -> Optional[float]:
        body_z = self.retarget.configuration.data.xpos[1:, 2]
        if body_z.size == 0:
            return None
        min_body_z = float(np.min(body_z))
        if not np.isfinite(min_body_z):
            return None
        return min_body_z

    def _get_current_ground_reference_z(self) -> Optional[float]:
        toe_z_values: list[float] = []
        body_name_map = getattr(self.retarget, "robot_body_names", {})
        data = self.retarget.configuration.data

        for body_name in self.ROBOT_GROUND_REFERENCE_BODY_NAMES:
            body_id = body_name_map.get(body_name)
            if body_id is None:
                continue
            if body_id < 0 or body_id >= data.xpos.shape[0]:
                continue
            z = float(data.xpos[body_id, 2])
            if np.isfinite(z):
                toe_z_values.append(z)

        if toe_z_values:
            return float(min(toe_z_values))
        return self._get_current_min_body_z()

    def _apply_min_link_height_offset(self, qpos: np.ndarray) -> np.ndarray:
        qpos_adj = np.asarray(qpos, dtype=np.float32).copy()
        ground_ref_z = self._get_current_ground_reference_z()
        if ground_ref_z is None:
            return qpos_adj

        if self.min_link_height_align_strategy == "per_frame":
            qpos_adj[2] += self.min_link_height - ground_ref_z
            return qpos_adj

        if self.fixed_min_link_height_offset is None:
            offset = self.min_link_height - ground_ref_z
            self.min_link_height_offset_samples.append(offset)
            if len(self.min_link_height_offset_samples) >= self.min_link_height_bootstrap_frames:
                self.fixed_min_link_height_offset = float(np.mean(self.min_link_height_offset_samples))
                print(
                    "[Info] worker startup_fixed ground calibration: "
                    f"{self.fixed_min_link_height_offset:.6f} m from "
                    f"{len(self.min_link_height_offset_samples)} frames"
                )
                self.min_link_height_offset_samples.clear()

        applied_offset = (
            self.fixed_min_link_height_offset
            if self.fixed_min_link_height_offset is not None
            else float(np.mean(self.min_link_height_offset_samples))
            if self.min_link_height_offset_samples
            else 0.0
        )
        qpos_adj[2] += applied_offset
        return qpos_adj

    @staticmethod
    def _copy_human_motion_data(human_motion_data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(human_motion_data, dict):
            return None
        copied: Dict[str, Any] = {}
        for key, value in human_motion_data.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            pos = np.asarray(value[0], dtype=np.float32).copy()
            rot = np.asarray(value[1], dtype=np.float32).copy()
            copied[key] = (pos, rot)
        return copied

    def process_packet(self, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body_pose_dict = self._body_poses_to_pose_dict(packet.get("poses"))
        if body_pose_dict is None:
            return None

        qpos_curr = self.retarget.retarget(body_pose_dict, offset_to_ground=False)
        if qpos_curr is None:
            return None

        qpos_curr = np.asarray(qpos_curr, dtype=np.float32).reshape(-1)
        if qpos_curr.shape[0] < 36:
            raise ValueError(f"retarget qpos too short: {qpos_curr.shape[0]}")
        qpos_curr = self._apply_min_link_height_offset(qpos_curr[:36])

        return {
            "type": "retarget_result",
            "seq": int(packet["seq"]),
            "recv_ns": int(packet["recv_ns"]),
            "source_timestamp_ns": packet.get("source_timestamp_ns"),
            "qpos": qpos_curr.astype(np.float32, copy=True),
            "human_motion_data": self._copy_human_motion_data(self.retarget.scaled_human_data)
            if self.send_human_motion
            else None,
        }


def _retarget_worker_main(
    raw_recv_conn: Any,
    result_send_conn: Any,
    worker_config: Dict[str, Any],
) -> None:
    try:
        runtime = _RetargetWorkerRuntime(worker_config)
    except Exception as exc:
        try:
            result_send_conn.send({"type": "worker_init_error", "error": str(exc)})
        except Exception:
            pass
        return

    try:
        result_send_conn.send({"type": "worker_ready"})
    except Exception:
        return

    last_processed_seq = 0
    while True:
        try:
            if not raw_recv_conn.poll(0.1):
                continue
            packet = raw_recv_conn.recv()
        except EOFError:
            break
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if isinstance(packet, dict) and packet.get("type") == "shutdown":
            break

        dropped_before_process = 0
        while raw_recv_conn.poll():
            try:
                newer_packet = raw_recv_conn.recv()
            except EOFError:
                newer_packet = None
            if newer_packet is None:
                break
            if isinstance(newer_packet, dict) and newer_packet.get("type") == "shutdown":
                return
            dropped_before_process += 1
            packet = newer_packet

        prev_processed_seq = last_processed_seq
        try:
            result = runtime.process_packet(packet)
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if result is None:
            continue

        result["dropped_before_process"] = int(dropped_before_process)
        result["prev_processed_seq"] = int(prev_processed_seq)
        last_processed_seq = int(result["seq"])

        try:
            result_send_conn.send(result)
        except (BrokenPipeError, EOFError, OSError):
            break


class LowLatencyTeleopPoseZMQServer:
    BODY_JOINT_NAMES = XR_BODY_JOINT_NAMES

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = args.robot
        self.vis_fps = int(args.vis_fps)
        self.ctrl_fps = int(args.ctrl_fps)
        self.lookback_ns = int(float(args.lookback_ms) * 1e6)
        self.retarget_buffer_window_ns = int(float(args.retarget_buffer_window_s) * 1e9)
        self.log_interval_s = float(args.log_interval_s)
        self.start_fresh_frames = int(args.start_fresh_frames)
        self.start_max_retarget_age_ns = int(float(args.start_max_retarget_age_ms) * 1e6)
        self.start_fresh_wait_timeout_s = float(args.start_fresh_wait_timeout_ms) / 1000.0
        self.hand_ctrl_bind_addr = str(args.hand_ctrl_bind_addr).strip()
        self.hand_ctrl_source_timeout_ns = int(float(args.hand_ctrl_source_timeout_ms) * 1e6)

        if self.vis_fps <= 0:
            raise ValueError("vis_fps must be > 0")
        if self.ctrl_fps <= 0:
            raise ValueError("ctrl_fps must be > 0")
        if self.lookback_ns < 0:
            raise ValueError("lookback_ms must be >= 0")
        if self.retarget_buffer_window_ns <= 0:
            raise ValueError("retarget_buffer_window_s must be > 0")
        if self.log_interval_s < 0:
            raise ValueError("log_interval_s must be >= 0")
        if self.start_fresh_frames <= 0:
            raise ValueError("start_fresh_frames must be > 0")
        if self.start_max_retarget_age_ns <= 0:
            raise ValueError("start_max_retarget_age_ms must be > 0")
        if self.start_fresh_wait_timeout_s <= 0:
            raise ValueError("start_fresh_wait_timeout_ms must be > 0")
        if self.hand_ctrl_source_timeout_ns <= 0:
            raise ValueError("hand_ctrl_source_timeout_ms must be > 0")

        self.retarget = None
        self.viewer = None
        self.gmr_max_iter = 5

        self.zmq_context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None

        # A recorder must not attach another PULL socket to rep/ctrl: ZeroMQ
        # PUSH/PULL load-balances messages and the recorder would steal frames
        # from the controller.  The tap is an independent PUB channel.  All
        # producers enqueue lightweight events; one dedicated thread owns the
        # PUB socket so XR/retarget/control callbacks are never blocked by I/O.
        self.tap_bind_addr = str(args.tap_bind_addr).strip()
        self.tap_accepting = bool(self.tap_bind_addr)
        self.tap_queue: queue.Queue[tuple[str, Dict[str, Any]]] = queue.Queue(
            maxsize=max(1, int(args.tap_queue_size))
        )
        self.tap_stats_lock = threading.Lock()
        self.tap_next_seq = 0
        self.tap_enqueued_count = 0
        self.tap_sent_count = 0
        self.tap_queue_drop_count = 0
        self.tap_send_drop_count = 0
        self.tap_prepare_drop_count = 0

        self.default_qpos = self._build_default_qpos()
        self.last_controller_buttons: Dict[str, Any] = self._default_controller_buttons()

        self.min_link_height = float(args.min_link_height)
        self.min_link_height_align_strategy = str(args.min_link_height_align_strategy)
        self.min_link_height_bootstrap_frames = max(1, int(args.min_link_height_bootstrap_frames))
        self.fixed_min_link_height_offset: Optional[float] = None
        self.min_link_height_offset_samples: list[float] = []

        self.rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.rotation_quat = R.from_matrix(self.rotation_matrix).as_quat(scalar_first=True)

        self.latest_vr_lock = threading.Lock()
        self.latest_vr_poses: Optional[Any] = None
        self.latest_vr_recv_ns: int = 0
        self.latest_vr_seq: int = 0
        self.latest_vr_motion_timestamp_ns: Optional[int] = None
        self.latest_controller_source_timestamp_ns: Optional[int] = None
        self.latest_controller_recv_monotonic_ns: int = 0
        self.latest_controller_recv_wall_time_ns: int = 0
        self.latest_left_controller_update_sequence: Optional[int] = None
        self.latest_right_controller_update_sequence: Optional[int] = None
        self.latest_left_controller_recv_monotonic_ns: int = 0
        self.latest_right_controller_recv_monotonic_ns: int = 0
        self.latest_left_controller_source_valid: bool = False
        self.latest_right_controller_source_valid: bool = False
        self.controller_control_epoch: int = 0
        self.controller_start_active: bool = False
        self.controller_start_edge_recv_ns: int = 0

        self.retarget_buffer_lock = threading.Lock()
        self.retarget_buffer_condition = threading.Condition(self.retarget_buffer_lock)
        self.retarget_buffer: deque[RetargetedFrame] = deque()
        self.vis_lock = threading.Lock()
        self.latest_vis_qpos: Optional[np.ndarray] = None
        self.latest_vis_human_motion: Optional[Dict[str, Any]] = None

        self.vr_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()

        self.frame_seq = 0
        self.last_vis_monotonic = 0.0
        self.last_req_monotonic: Optional[float] = None
        self.req_count = 0
        self.reply_count = 0
        self.reply_drop_count = 0
        self.req_merged_total = 0
        self.fallback_count = 0
        self.start_gate_wait_count = 0
        self.start_gate_ready_count = 0
        self.raw_motion_drop_count = 0
        self.latest_req_dt_ms: Optional[float] = None
        self.latest_merged_reqs = 0

        self.callback_count = 0
        self.retarget_count = 0
        self.latest_debug_info: Dict[str, Any] = {
            "mode": "no_data",
            "target_age_ms": None,
            "older_age_ms": None,
            "newer_age_ms": None,
            "span_ms": None,
            "buffer_len": 0,
            "retarget_age_ms": None,
            "raw_motion_age_ms": None,
        }

        self.retarget_thread = None
        self.raw_sender_thread = None
        self.worker_result_thread = None
        self.request_thread = None
        self.control_thread = None
        self.tap_thread = None
        self.stats_thread = None
        self.visualization_thread = None

        self.mp_ctx = mp.get_context("spawn")
        self.raw_send_conn = None
        self.raw_recv_conn = None
        self.result_send_conn = None
        self.result_recv_conn = None
        self.retarget_process = None

    def _build_default_qpos(self) -> np.ndarray:
        mimic = np.asarray(DEFAULT_MIMIC_OBS[self.robot], dtype=np.float32).reshape(-1)
        if mimic.shape[0] < 35:
            raise ValueError(f"DEFAULT_MIMIC_OBS[{self.robot}] must be at least 35 dims")
        dof_pos = mimic[6:35]
        root_z = float(mimic[2])
        root_pos = np.array([0.0, 0.0, root_z], dtype=np.float32)
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([root_pos, root_quat, dof_pos], axis=0).astype(np.float32)

    @staticmethod
    def _default_controller_buttons() -> Dict[str, Any]:
        return {
            "left_key_one": False,
            "left_key_two": False,
            "left_axis_click": False,
            "left_index_trig": False,
            "left_index_trig_value": 0.0,
            "left_grip": False,
            "left_grip_value": 0.0,
            "left_axis": [0.0, 0.0],
            "right_key_one": False,
            "right_key_two": False,
            "right_axis_click": False,
            "right_index_trig": False,
            "right_index_trig_value": 0.0,
            "right_grip": False,
            "right_grip_value": 0.0,
            "right_axis": [0.0, 0.0],
        }

    @staticmethod
    def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    def _slerp_quat_wxyz(self, quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
        q0 = self._normalize_quat_wxyz(quat0).astype(np.float64)
        q1 = self._normalize_quat_wxyz(quat1).astype(np.float64)
        t = float(np.clip(alpha, 0.0, 1.0))

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            out = q0 + t * (q1 - q0)
            return self._normalize_quat_wxyz(out)

        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = float(np.sin(theta_0))
        if abs(sin_theta_0) < 1e-8:
            return self._normalize_quat_wxyz(q0)

        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        out = s0 * q0 + s1 * q1
        return self._normalize_quat_wxyz(out)

    def _interpolate_qpos(self, prev_qpos: np.ndarray, next_qpos: np.ndarray, alpha: float) -> np.ndarray:
        t = float(np.clip(alpha, 0.0, 1.0))
        frame = prev_qpos * (1.0 - t) + next_qpos * t
        frame[3:7] = self._slerp_quat_wxyz(prev_qpos[3:7], next_qpos[3:7], t)
        return frame.astype(np.float32)

    def _extract_controller_buttons_from_snapshot(self, snapshot: Optional[dict]) -> Dict[str, Any]:
        if snapshot is None:
            return self.last_controller_buttons

        controllers = snapshot.get("controllers", {}) if isinstance(snapshot, dict) else {}
        left = controllers.get("left", {}) if isinstance(controllers, dict) else {}
        right = controllers.get("right", {}) if isinstance(controllers, dict) else {}

        def _axis(values: Any) -> list[float]:
            if isinstance(values, (list, tuple)) and len(values) >= 2:
                return [float(values[0]), float(values[1])]
            return [0.0, 0.0]

        def _unit(value: Any) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(numeric):
                return 0.0
            return float(np.clip(numeric, 0.0, 1.0))

        left_trigger = _unit(left.get("trigger", 0.0))
        left_grip = _unit(left.get("grip", 0.0))
        right_trigger = _unit(right.get("trigger", 0.0))
        right_grip = _unit(right.get("grip", 0.0))

        return {
            "left_key_one": bool(left.get("primary_button", False)),
            "left_key_two": bool(left.get("secondary_button", False)),
            "left_axis_click": bool(left.get("axis_click", False)),
            "left_index_trig": left_trigger > 1e-4,
            "left_index_trig_value": left_trigger,
            "left_grip": left_grip > 1e-4,
            "left_grip_value": left_grip,
            "left_axis": _axis(left.get("axis", [0.0, 0.0])),
            "right_key_one": bool(right.get("primary_button", False)),
            "right_key_two": bool(right.get("secondary_button", False)),
            "right_axis_click": bool(right.get("axis_click", False)),
            "right_index_trig": right_trigger > 1e-4,
            "right_index_trig_value": right_trigger,
            "right_grip": right_grip > 1e-4,
            "right_grip_value": right_grip,
            "right_axis": _axis(right.get("axis", [0.0, 0.0])),
        }

    @staticmethod
    def _extract_controller_freshness(
        snapshot: Optional[dict],
    ) -> tuple[bool, Optional[int], bool, Optional[int]]:
        """Read per-side freshness metadata supplied by the SDK binding.

        Missing metadata deliberately fails closed.  A whole XR snapshot may
        keep arriving from the headset/body while one controller has dropped;
        treating the snapshot timestamp as controller freshness would then
        replay a frozen grip value indefinitely.
        """

        controllers = snapshot.get("controllers", {}) if isinstance(snapshot, dict) else {}
        left = controllers.get("left", {}) if isinstance(controllers, dict) else {}
        right = controllers.get("right", {}) if isinstance(controllers, dict) else {}

        def _side(controller: Any) -> tuple[bool, Optional[int]]:
            if not isinstance(controller, dict) or not bool(controller.get("source_valid", False)):
                return False, None
            for key in ("trigger", "grip"):
                try:
                    value = float(controller.get(key, 0.0))
                except (TypeError, ValueError):
                    return False, None
                # Tolerate only floating-point noise around the SDK's [0, 1]
                # contract. A bad analog value must disarm, never become an
                # implicit release/open command.
                if not np.isfinite(value) or value < -1e-3 or value > 1.001:
                    return False, None
            try:
                sequence = int(controller.get("update_sequence", 0))
            except (TypeError, ValueError):
                return False, None
            if sequence <= 0:
                return False, None
            return True, sequence

        left_valid, left_sequence = _side(left)
        right_valid, right_sequence = _side(right)
        return left_valid, left_sequence, right_valid, right_sequence

    @staticmethod
    def _build_hand_control_payload(
        *,
        buttons: Dict[str, Any],
        tracking_active: bool,
        sample_monotonic_ns: int,
        sample_wall_time_ns: int,
        controller_source_timestamp_ns: Optional[int],
        left_controller_source_valid: bool,
        right_controller_source_valid: bool,
        left_controller_update_sequence: Optional[int],
        right_controller_update_sequence: Optional[int],
        left_controller_last_update_ns: int,
        right_controller_last_update_ns: int,
        source_timeout_ns: int,
    ) -> Dict[str, Any]:
        def _freshness(
            _source_valid: bool, last_update_ns: int
        ) -> tuple[bool, Optional[float]]:
            # A single SDK snapshot may omit one controller while the last
            # valid controller sample is still recent. Apply the configured
            # grace period to that last valid sample instead of invalidating
            # the hand stream immediately.
            if last_update_ns <= 0:
                return False, None
            age_ns = max(0, int(sample_monotonic_ns) - int(last_update_ns))
            return age_ns <= int(source_timeout_ns), age_ns / 1e6

        left_valid, left_age_ms = _freshness(
            left_controller_source_valid, left_controller_last_update_ns
        )
        right_valid, right_age_ms = _freshness(
            right_controller_source_valid, right_controller_last_update_ns
        )
        source_valid = left_valid and right_valid
        controller_age_ms = (
            max(left_age_ms, right_age_ms)
            if left_age_ms is not None and right_age_ms is not None
            else None
        )

        def _float(name: str) -> float:
            try:
                value = float(buttons.get(name, 0.0))
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value):
                return 0.0
            return float(np.clip(value, 0.0, 1.0))

        return {
            "schema_version": 1,
            "bridge_sample_monotonic_ns": int(sample_monotonic_ns),
            "bridge_sample_wall_time_ns": int(sample_wall_time_ns),
            "controller_source_timestamp_ns": controller_source_timestamp_ns,
            "controller_age_ms": controller_age_ms,
            "source_valid": bool(source_valid),
            "tracking_active": bool(tracking_active),
            "left": {
                "grip": _float("left_grip_value"),
                "trigger": _float("left_index_trig_value"),
                "secondary_button": bool(buttons.get("left_key_two", False)),
                "source_valid": bool(left_valid),
                "age_ms": left_age_ms,
                "update_sequence": left_controller_update_sequence,
            },
            "right": {
                "grip": _float("right_grip_value"),
                "trigger": _float("right_index_trig_value"),
                "secondary_button": bool(buttons.get("right_key_two", False)),
                "source_valid": bool(right_valid),
                "age_ms": right_age_ms,
                "update_sequence": right_controller_update_sequence,
            },
        }

    @staticmethod
    def _copy_pose_list(poses: Any) -> Optional[list[list[float]]]:
        if not isinstance(poses, (list, tuple)):
            return None
        copied: list[list[float]] = []
        for pose in poses:
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                return None
            copied.append([float(value) for value in pose[:7]])
        return copied

    @staticmethod
    def _copy_numeric_list(values: Any) -> Optional[list[float]]:
        if not isinstance(values, (list, tuple)):
            return None
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return None

    def _enqueue_tap(self, topic: str, payload: Dict[str, Any]) -> None:
        if not self.tap_accepting:
            return

        event = {
            "tap_schema_version": 1,
            "type": str(topic),
            "bridge_enqueue_monotonic_ns": time.monotonic_ns(),
            "bridge_enqueue_wall_time_ns": time.time_ns(),
            **payload,
        }
        # Assign sequence and enqueue under one short lock so events from
        # different producer threads cannot enter the queue out of seq order.
        with self.tap_stats_lock:
            event["tap_seq"] = self.tap_next_seq
            self.tap_next_seq += 1
            try:
                self.tap_queue.put_nowait((str(topic), event))
                self.tap_enqueued_count += 1
            except queue.Full:
                self.tap_queue_drop_count += 1

    def _count_tap_prepare_drop(self) -> None:
        """Account for a tap payload that could not be prepared.

        This intentionally does not log from XR/control callbacks: recorder
        diagnostics must never add blocking I/O to the realtime path.
        """

        with self.tap_stats_lock:
            self.tap_prepare_drop_count += 1

    @staticmethod
    def _serialize_qpos_frame(qpos: np.ndarray) -> Dict[str, Any]:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        return {
            "root_pos": q[0:3].tolist(),
            "root_quat": q[3:7].tolist(),
            "dof_pos": q[7:36].tolist(),
        }

    def _on_vr_frame(self, snapshot: dict) -> None:
        recv_ns = time.monotonic_ns()
        recv_wall_time_ns = time.time_ns()
        incoming_controller_buttons = self._extract_controller_buttons_from_snapshot(snapshot)
        (
            left_controller_valid,
            left_controller_sequence,
            right_controller_valid,
            right_controller_sequence,
        ) = self._extract_controller_freshness(snapshot)
        top_timestamp_ns = None
        try:
            top_timestamp_ns = int(snapshot.get("timestamp_ns", 0)) if isinstance(snapshot, dict) else None
        except Exception:
            top_timestamp_ns = None
        body = snapshot.get("body", {}) if isinstance(snapshot, dict) else {}
        body_available = bool(body.get("available", False)) if isinstance(body, dict) else False
        body_timestamp_ns = None
        if body_available:
            try:
                body_timestamp_ns = int(body.get("timestamp_ns", 0))
            except Exception:
                body_timestamp_ns = None
        motion_timestamp_ns = body_timestamp_ns if body_timestamp_ns not in (None, 0) else top_timestamp_ns

        should_wake_retarget = False
        should_wake_start_gate = False
        vr_seq = 0
        with self.latest_vr_lock:
            controller_buttons = dict(self.last_controller_buttons)
            left_is_new = (
                left_controller_valid
                and left_controller_sequence != self.latest_left_controller_update_sequence
            )
            right_is_new = (
                right_controller_valid
                and right_controller_sequence != self.latest_right_controller_update_sequence
            )
            for side, side_valid, side_is_new in (
                ("left", left_controller_valid, left_is_new),
                ("right", right_controller_valid, right_is_new),
            ):
                if side_is_new:
                    for key, value in incoming_controller_buttons.items():
                        if key.startswith(f"{side}_"):
                            controller_buttons[key] = value
                # Keep the last valid values through brief incomplete SDK
                # snapshots. Freshness is governed by the per-side timestamp.

            prev_right_key_one = bool(self.last_controller_buttons.get("right_key_one", False))
            prev_left_key_one = bool(self.last_controller_buttons.get("left_key_one", False))
            right_key_one = bool(controller_buttons.get("right_key_one", False))
            left_key_one = bool(controller_buttons.get("left_key_one", False))
            start_rise = right_key_one and not prev_right_key_one
            stop_rise = left_key_one and not prev_left_key_one
            # Match the C++ source: stop wins if both buttons rise together.
            if stop_rise:
                self.controller_control_epoch += 1
                self.controller_start_active = False
                should_wake_start_gate = True
            elif start_rise:
                self.controller_control_epoch += 1
                self.controller_start_active = True
                self.controller_start_edge_recv_ns = recv_ns
                should_wake_start_gate = True
            self.last_controller_buttons = controller_buttons
            self.latest_controller_source_timestamp_ns = top_timestamp_ns
            self.latest_controller_recv_monotonic_ns = recv_ns
            self.latest_controller_recv_wall_time_ns = recv_wall_time_ns
            self.latest_left_controller_source_valid = left_controller_valid
            self.latest_right_controller_source_valid = right_controller_valid
            if left_is_new:
                self.latest_left_controller_update_sequence = left_controller_sequence
                self.latest_left_controller_recv_monotonic_ns = recv_ns
            if right_is_new:
                self.latest_right_controller_update_sequence = right_controller_sequence
                self.latest_right_controller_recv_monotonic_ns = recv_ns
            self.callback_count += 1
            if body_available and motion_timestamp_ns is not None:
                if self.latest_vr_motion_timestamp_ns != motion_timestamp_ns:
                    self.latest_vr_poses = body.get("poses", None)
                    self.latest_vr_recv_ns = recv_ns
                    self.latest_vr_seq += 1
                    self.latest_vr_motion_timestamp_ns = motion_timestamp_ns
                    should_wake_retarget = True
            vr_seq = self.latest_vr_seq
        if should_wake_start_gate:
            with self.retarget_buffer_condition:
                self.retarget_buffer_condition.notify_all()
        if should_wake_retarget:
            # Wake the control path before doing any recorder-only copying.
            self.vr_frame_event.set()
            if self.tap_accepting:
                try:
                    body_poses = self._copy_pose_list(body.get("poses", None))
                    headset_pose = self._copy_numeric_list(
                        snapshot.get("headset_pose", None) if isinstance(snapshot, dict) else None
                    )
                    self._enqueue_tap(
                        "xr",
                        {
                            "seq": int(vr_seq),
                            "sdk_timestamp_ns": top_timestamp_ns,
                            "body_source_timestamp_ns": motion_timestamp_ns,
                            "bridge_recv_monotonic_ns": int(recv_ns),
                            "bridge_recv_wall_time_ns": int(recv_wall_time_ns),
                            "body_poses_xyz_xyzw": body_poses,
                            "headset_pose": headset_pose,
                            "controller_buttons": controller_buttons,
                        },
                    )
                except Exception:
                    self._count_tap_prepare_drop()

    def _append_retarget_frame(self, recv_ns: int, qpos: np.ndarray) -> None:
        cutoff_ns = recv_ns - self.retarget_buffer_window_ns
        with self.retarget_buffer_condition:
            self.retarget_buffer.append(RetargetedFrame(recv_ns=recv_ns, qpos=qpos.astype(np.float32, copy=True)))
            while self.retarget_buffer and self.retarget_buffer[0].recv_ns < cutoff_ns:
                self.retarget_buffer.popleft()
            self.retarget_buffer_condition.notify_all()

    @staticmethod
    def _copy_human_motion_data(human_motion_data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(human_motion_data, dict):
            return None

        copied: Dict[str, Any] = {}
        for key, value in human_motion_data.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            pos = np.asarray(value[0], dtype=np.float32).copy()
            rot = np.asarray(value[1], dtype=np.float32).copy()
            copied[key] = (pos, rot)
        return copied

    def _get_retarget_frames_snapshot(self) -> list[RetargetedFrame]:
        with self.retarget_buffer_lock:
            return list(self.retarget_buffer)

    def _get_controller_start_epoch(self) -> tuple[int, bool, int]:
        with self.latest_vr_lock:
            return (
                self.controller_control_epoch,
                self.controller_start_active,
                self.controller_start_edge_recv_ns,
            )

    def _is_start_gate_current(
        self,
        expected_epoch: int,
        selected_recv_ns: Optional[int] = None,
        now_ns: Optional[int] = None,
    ) -> bool:
        with self.latest_vr_lock:
            epoch_matches = self.controller_control_epoch == expected_epoch
            start_active = self.controller_start_active
        if not epoch_matches or not start_active:
            return False
        if selected_recv_ns is None:
            return True
        check_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        return max(0, check_ns - int(selected_recv_ns)) <= self.start_max_retarget_age_ns

    def _wait_for_fresh_start_qpos(
        self,
        cutoff_ns: int,
        expected_epoch: Optional[int] = None,
    ) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Wait for a small post-start retarget window and return its newest frame.

        A start acknowledgement must never be built from the normal
        fallback_latest/default path.  In particular, the retarget deque only
        expires old entries when a new result is appended, so its tail may be
        arbitrarily old while XR body tracking is frozen even though controller
        button callbacks continue to arrive.
        """

        deadline_ns = time.monotonic_ns() + int(self.start_fresh_wait_timeout_s * 1e9)
        last_info: Dict[str, Any] = {
            "mode": "start_wait_fresh",
            "target_ns": int(cutoff_ns),
            "older_ns": None,
            "newer_ns": None,
            "alpha": None,
            "buffer_len": 0,
            "fresh_frame_count": 0,
            "latest_age_ms": None,
        }

        with self.retarget_buffer_condition:
            while not self.stop_event.is_set():
                now_ns = time.monotonic_ns()
                if expected_epoch is not None and not self._is_start_gate_current(
                    expected_epoch=expected_epoch,
                    now_ns=now_ns,
                ):
                    last_info["mode"] = "start_cancelled"
                    return None, last_info

                # All frames in the validation window must belong to this
                # start epoch and still be recent, not only the selected tail.
                fresh_cutoff_ns = max(cutoff_ns, now_ns - self.start_max_retarget_age_ns)
                fresh = [
                    frame for frame in self.retarget_buffer if frame.recv_ns >= fresh_cutoff_ns
                ]
                newest = self.retarget_buffer[-1] if self.retarget_buffer else None
                newest_age_ns = None if newest is None else max(0, now_ns - newest.recv_ns)
                last_info = {
                    "mode": "start_wait_fresh",
                    "target_ns": int(cutoff_ns),
                    "older_ns": None if not fresh else int(fresh[0].recv_ns),
                    "newer_ns": None if newest is None else int(newest.recv_ns),
                    "alpha": None,
                    "buffer_len": len(self.retarget_buffer),
                    "fresh_frame_count": len(fresh),
                    "latest_age_ms": (
                        None if newest_age_ns is None else round(newest_age_ns / 1e6, 3)
                    ),
                }

                if (
                    len(fresh) >= self.start_fresh_frames
                    and newest_age_ns is not None
                    and newest_age_ns <= self.start_max_retarget_age_ns
                ):
                    selected = fresh[-1]
                    ready_info = dict(last_info)
                    ready_info.update(
                        {
                            "mode": "start_fresh",
                            "older_ns": int(fresh[0].recv_ns),
                            "newer_ns": int(selected.recv_ns),
                            "selected_recv_ns": int(selected.recv_ns),
                            "selected_age_ms": round(max(0, now_ns - selected.recv_ns) / 1e6, 3),
                        }
                    )
                    return selected.qpos.astype(np.float32, copy=True), ready_info

                remaining_ns = deadline_ns - now_ns
                if remaining_ns <= 0:
                    break
                self.retarget_buffer_condition.wait(timeout=min(0.05, remaining_ns / 1e9))

        return None, last_info

    def _sample_target_qpos(self, frames: list[RetargetedFrame], target_ns: int) -> tuple[np.ndarray, bool, Dict[str, Any]]:
        if not frames:
            return self.default_qpos.copy(), True, {
                "mode": "default",
                "target_ns": target_ns,
                "older_ns": None,
                "newer_ns": None,
                "alpha": None,
                "buffer_len": 0,
        }
        if len(frames) == 1:
            only_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "single_frame",
                "target_ns": target_ns,
                "older_ns": only_ns,
                "newer_ns": only_ns,
                "alpha": None,
                "buffer_len": 1,
            }
        if target_ns <= frames[0].recv_ns:
            oldest_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_oldest",
                "target_ns": target_ns,
                "older_ns": oldest_ns,
                "newer_ns": oldest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }
        if target_ns >= frames[-1].recv_ns:
            latest_ns = frames[-1].recv_ns
            return frames[-1].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_latest",
                "target_ns": target_ns,
                "older_ns": latest_ns,
                "newer_ns": latest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }

        for idx in range(1, len(frames)):
            prev_frame = frames[idx - 1]
            next_frame = frames[idx]
            if target_ns <= next_frame.recv_ns:
                dt = next_frame.recv_ns - prev_frame.recv_ns
                if dt <= 0:
                    same_ns = next_frame.recv_ns
                    return next_frame.qpos.astype(np.float32, copy=True), True, {
                        "mode": "degenerate_dt",
                        "target_ns": target_ns,
                        "older_ns": same_ns,
                        "newer_ns": same_ns,
                        "alpha": None,
                        "buffer_len": len(frames),
                    }
                alpha = float(target_ns - prev_frame.recv_ns) / float(dt)
                return self._interpolate_qpos(prev_frame.qpos, next_frame.qpos, alpha), False, {
                    "mode": "interpolate",
                    "target_ns": target_ns,
                    "older_ns": prev_frame.recv_ns,
                    "newer_ns": next_frame.recv_ns,
                    "alpha": alpha,
                    "buffer_len": len(frames),
                }

        latest_ns = frames[-1].recv_ns
        return frames[-1].qpos.astype(np.float32, copy=True), True, {
            "mode": "fallback_latest",
            "target_ns": target_ns,
            "older_ns": latest_ns,
            "newer_ns": latest_ns,
            "alpha": None,
            "buffer_len": len(frames),
        }

    def _build_reply_frames(self, req_recv_ns: int) -> tuple[list[np.ndarray], bool, Dict[str, Any]]:
        frames = self._get_retarget_frames_snapshot()
        target_base_ns = req_recv_ns - self.lookback_ns
        qpos, used_fallback, sample_info = self._sample_target_qpos(frames, target_base_ns)
        return [qpos], used_fallback, sample_info

    def _get_latest_frame_ages_ms(self, now_ns: Optional[int] = None) -> tuple[Optional[float], Optional[float]]:
        if now_ns is None:
            now_ns = time.monotonic_ns()

        with self.latest_vr_lock:
            latest_raw_recv_ns = int(self.latest_vr_recv_ns) if self.latest_vr_recv_ns > 0 else None

        with self.retarget_buffer_lock:
            latest_retarget_recv_ns = self.retarget_buffer[-1].recv_ns if self.retarget_buffer else None

        raw_motion_age_ms = None
        if latest_raw_recv_ns is not None:
            raw_motion_age_ms = round((now_ns - latest_raw_recv_ns) / 1e6, 3)

        retarget_age_ms = None
        if latest_retarget_recv_ns is not None:
            retarget_age_ms = round((now_ns - latest_retarget_recv_ns) / 1e6, 3)

        return retarget_age_ms, raw_motion_age_ms

    def _update_debug_info(self, sample_info: Dict[str, Any], req_recv_ns: int) -> None:
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms()

        info = {
            "mode": sample_info.get("mode"),
            "target_age_ms": round((req_recv_ns - int(sample_info["target_ns"])) / 1e6, 3),
            "older_age_ms": None if older_ns is None else round((req_recv_ns - int(older_ns)) / 1e6, 3),
            "newer_age_ms": None if newer_ns is None else round((req_recv_ns - int(newer_ns)) / 1e6, 3),
            "span_ms": None
            if older_ns is None or newer_ns is None
            else round((int(newer_ns) - int(older_ns)) / 1e6, 3),
            "alpha": sample_info.get("alpha"),
            "buffer_len": int(sample_info.get("buffer_len", 0)),
            "retarget_age_ms": retarget_age_ms,
            "raw_motion_age_ms": raw_motion_age_ms,
        }
        with self.stats_lock:
            self.latest_debug_info = info

    def _warn_on_fallback(self, sample_info: Dict[str, Any]) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        target_age_ms = round((now_ns - int(sample_info["target_ns"])) / 1e6, 3)
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        older_age_ms = None if older_ns is None else round((now_ns - int(older_ns)) / 1e6, 3)
        newer_age_ms = None if newer_ns is None else round((now_ns - int(newer_ns)) / 1e6, 3)
        print(
            "[Warning] interpolation fallback "
            f"mode={sample_info.get('mode')}, "
            f"target_age_ms={target_age_ms}, "
            f"older_age_ms={older_age_ms}, "
            f"newer_age_ms={newer_age_ms}, "
            f"buffer={int(sample_info.get('buffer_len', 0))}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}"
        )

    def _warn_on_raw_motion_drop(self, dropped_count: int, latest_seq: int, last_processed_seq: int) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        print(
            "[Warning] retarget lag dropped raw motion frames "
            f"dropped={int(dropped_count)}, "
            f"last_processed_seq={int(last_processed_seq)}, "
            f"latest_seq={int(latest_seq)}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}"
        )

    def _raw_sender_loop(self) -> None:
        last_sent_seq = 0

        while not self.stop_event.is_set():
            if not self.vr_frame_event.wait(timeout=0.1):
                continue

            while not self.stop_event.is_set():
                with self.latest_vr_lock:
                    poses = self.latest_vr_poses
                    recv_ns = self.latest_vr_recv_ns
                    seq = self.latest_vr_seq
                    source_timestamp_ns = self.latest_vr_motion_timestamp_ns

                if poses is None or seq == last_sent_seq:
                    with self.latest_vr_lock:
                        if self.latest_vr_seq == last_sent_seq:
                            self.vr_frame_event.clear()
                            break
                    continue

                if last_sent_seq != 0 and seq > last_sent_seq + 1:
                    dropped_count = seq - last_sent_seq - 1
                    self.raw_motion_drop_count += int(dropped_count)
                    self._warn_on_raw_motion_drop(
                        dropped_count=dropped_count,
                        latest_seq=seq,
                        last_processed_seq=last_sent_seq,
                    )

                try:
                    self.raw_send_conn.send(
                        {
                            "seq": int(seq),
                            "recv_ns": int(recv_ns),
                            "source_timestamp_ns": source_timestamp_ns,
                            "poses": poses,
                        }
                    )
                except (BrokenPipeError, EOFError, OSError) as exc:
                    print(f"[Warning] raw->worker pipe failed: {exc}")
                    self.stop_event.set()
                    self.vr_frame_event.set()
                    break

                last_sent_seq = seq

    def _worker_result_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.result_recv_conn.poll(0.1):
                    continue
                payload = self.result_recv_conn.recv()
            except EOFError:
                print("[Warning] worker->main pipe closed")
                self.stop_event.set()
                break
            except Exception as exc:
                print(f"[Warning] worker result recv failed: {exc}")
                self.stop_event.set()
                break

            if not isinstance(payload, dict):
                continue

            payload_type = payload.get("type")
            if payload_type == "worker_ready":
                continue
            if payload_type in ("worker_init_error", "worker_runtime_error"):
                print(f"[Warning] retarget worker error: {payload.get('error')}")
                if payload_type == "worker_init_error":
                    self.stop_event.set()
                continue
            if payload_type != "retarget_result":
                continue

            dropped_before_process = int(payload.get("dropped_before_process", 0))
            if dropped_before_process > 0:
                self.raw_motion_drop_count += dropped_before_process
                self._warn_on_raw_motion_drop(
                    dropped_count=dropped_before_process,
                    latest_seq=int(payload.get("seq", 0)),
                    last_processed_seq=int(payload.get("prev_processed_seq", 0)),
                )

            qpos_curr = np.asarray(payload.get("qpos"), dtype=np.float32).reshape(-1)
            recv_ns = int(payload["recv_ns"])
            self._append_retarget_frame(recv_ns=recv_ns, qpos=qpos_curr)
            self.retarget_count += 1
            if self.tap_accepting:
                try:
                    self._enqueue_tap(
                        "retarget",
                        {
                            "seq": int(payload.get("seq", 0)),
                            "body_source_timestamp_ns": payload.get("source_timestamp_ns"),
                            "bridge_recv_monotonic_ns": int(recv_ns),
                            "qpos_root_xyz_quat_wxyz_dof": qpos_curr[:36].tolist(),
                            "dropped_before_process": dropped_before_process,
                        },
                    )
                except Exception:
                    self._count_tap_prepare_drop()

            if self.viewer is not None:
                with self.vis_lock:
                    self.latest_vis_qpos = qpos_curr.astype(np.float32, copy=True)
                    self.latest_vis_human_motion = payload.get("human_motion_data")

    def _drain_requests_blocking(self) -> tuple[Optional[Dict[str, Any]], Optional[int], int]:
        import zmq

        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self.req_sock not in events:
                continue

            latest_req: Optional[Dict[str, Any]] = None
            req_recv_ns: Optional[int] = None
            merged_reqs = 0
            any_start = False

            while True:
                try:
                    raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                    req_recv_ns = time.monotonic_ns()
                except zmq.Again:
                    break
                except Exception as exc:
                    print(f"[Warning] request recv failed: {exc}")
                    break

                try:
                    req = json.loads(raw)
                except Exception:
                    print("[Warning] bad request JSON")
                    continue
                if not isinstance(req, dict):
                    continue

                merged_reqs += 1
                any_start = any_start or bool(req.get("start", False))
                latest_req = req

            if latest_req is None:
                continue

            latest_req["start"] = any_start
            return latest_req, req_recv_ns, merged_reqs

        return None, None, 0

    def _request_loop(self) -> None:
        import zmq

        active_start_epoch: Optional[int] = None
        start_cutoff_ns: Optional[int] = None
        start_ack_qpos: Optional[np.ndarray] = None
        start_ack_info: Optional[Dict[str, Any]] = None
        start_ack_sent = False
        last_start_wait_log_ns = 0

        while not self.stop_event.is_set():
            req, req_recv_ns, merged_reqs = self._drain_requests_blocking()
            if req is None or req_recv_ns is None:
                continue

            now = time.monotonic()
            self.req_count += 1
            self.req_merged_total += int(merged_reqs)
            self.latest_merged_reqs = int(merged_reqs)
            if self.last_req_monotonic is None:
                self.latest_req_dt_ms = None
            else:
                self.latest_req_dt_ms = (now - self.last_req_monotonic) * 1000.0
            self.last_req_monotonic = now

            request_start = bool(req.get("start", False))
            if request_start:
                controller_epoch, start_active, controller_edge_ns = (
                    self._get_controller_start_epoch()
                )
                if not start_active:
                    # This is a queued request from a session that has already
                    # been stopped. Never turn it into a start acknowledgement.
                    active_start_epoch = None
                    start_cutoff_ns = None
                    start_ack_qpos = None
                    start_ack_info = None
                    start_ack_sent = False
                    continue
                if active_start_epoch != controller_epoch or start_cutoff_ns is None:
                    active_start_epoch = controller_epoch
                    # Require results originating after both the button edge and
                    # the first corresponding start request. Results already in
                    # flight carry an older recv_ns and are filtered out.
                    start_cutoff_ns = max(int(req_recv_ns), int(controller_edge_ns))
                    start_ack_qpos = None
                    start_ack_info = None
                    start_ack_sent = False

                if start_ack_info is not None:
                    selected_recv_ns = start_ack_info.get("selected_recv_ns")
                    if selected_recv_ns is None or not self._is_start_gate_current(
                        expected_epoch=active_start_epoch,
                        selected_recv_ns=int(selected_recv_ns),
                    ):
                        # PUSH enqueue success is not an acknowledgement from
                        # the C++ consumer. Refresh an expired cached start pose
                        # instead of replaying it indefinitely.
                        start_ack_qpos = None
                        start_ack_info = None
                        start_ack_sent = False

                if start_ack_qpos is None or start_ack_info is None:
                    start_ack_qpos, start_ack_info = self._wait_for_fresh_start_qpos(
                        cutoff_ns=start_cutoff_ns,
                        expected_epoch=active_start_epoch,
                    )
                    if start_ack_qpos is None:
                        self._update_debug_info(
                            sample_info=start_ack_info,
                            req_recv_ns=req_recv_ns,
                        )
                        if start_ack_info.get("mode") == "start_cancelled":
                            start_ack_info = None
                            start_ack_qpos = None
                            continue
                        self.start_gate_wait_count += 1
                        now_ns = time.monotonic_ns()
                        if now_ns - last_start_wait_log_ns >= int(1e9):
                            print(
                                "[Warning] start fresh-frame gate waiting "
                                f"epoch={active_start_epoch}, "
                                f"fresh={start_ack_info.get('fresh_frame_count', 0)}/"
                                f"{self.start_fresh_frames}, "
                                f"latest_age_ms={start_ack_info.get('latest_age_ms')}, "
                                f"cutoff_ns={start_cutoff_ns}"
                            )
                            last_start_wait_log_ns = now_ns
                        # Do not acknowledge start with fallback_latest/default.
                        # The C++ source keeps neutral A active and retries.
                        continue

                    self.start_gate_ready_count += 1
                    right_arm = np.asarray(start_ack_qpos, dtype=np.float32).reshape(-1)[29:36]
                    print(
                        "[Info] start fresh-frame gate ready "
                        f"epoch={active_start_epoch}, "
                        f"fresh={start_ack_info.get('fresh_frame_count')}, "
                        f"selected_age_ms={start_ack_info.get('selected_age_ms')}, "
                        f"right_arm={np.round(right_arm, 4).tolist()}"
                    )

                out_frames = [start_ack_qpos.astype(np.float32, copy=True)]
                used_fallback = False
                sample_info = dict(start_ack_info)
                if start_ack_sent:
                    sample_info["mode"] = "start_fresh_cached"
            else:
                # A regular request proves the controller accepted the start
                # acknowledgement. Retire its cached reply so the next button
                # epoch must pass a new freshness gate.
                active_start_epoch = None
                start_cutoff_ns = None
                start_ack_qpos = None
                start_ack_info = None
                start_ack_sent = False
                out_frames, used_fallback, sample_info = self._build_reply_frames(
                    req_recv_ns=req_recv_ns
                )

            self._update_debug_info(sample_info=sample_info, req_recv_ns=req_recv_ns)
            if used_fallback:
                self.fallback_count += 1
                self._warn_on_fallback(sample_info=sample_info)
            seq_start = int(self.frame_seq)
            self.frame_seq += len(out_frames)

            retarget_age_ms = None
            if request_start:
                selected_recv_ns = sample_info.get("selected_recv_ns")
                if selected_recv_ns is not None:
                    retarget_age_ms = int(
                        max(0, time.monotonic_ns() - int(selected_recv_ns)) / 1e6
                    )
            else:
                retarget_frames = self._get_retarget_frames_snapshot()
                if retarget_frames:
                    retarget_age_ms = int(
                        max(0, time.monotonic_ns() - retarget_frames[-1].recv_ns) / 1e6
                    )

            payload = {
                "start": request_start,
                "no_interp_applied": bool(used_fallback),
                "chunk_size": len(out_frames),
                "frame_seq_start": seq_start,
                "retarget_age_ms": retarget_age_ms,
                "t_rep_ms": int(time.time() * 1000),
                "frames": [self._serialize_qpos_frame(x) for x in out_frames],
            }
            payload_json = json.dumps(payload)

            try:
                if request_start:
                    # Make the epoch/age check and PUSH enqueue atomic relative
                    # to controller edge updates in the XR callback.
                    with self.latest_vr_lock:
                        selected_recv_ns = int(sample_info["selected_recv_ns"])
                        now_ns = time.monotonic_ns()
                        send_is_current = (
                            self.controller_control_epoch == active_start_epoch
                            and self.controller_start_active
                            and max(0, now_ns - selected_recv_ns)
                            <= self.start_max_retarget_age_ns
                        )
                        if not send_is_current:
                            start_ack_qpos = None
                            start_ack_info = None
                            start_ack_sent = False
                            continue
                        self.rep_sock.send_string(payload_json, flags=zmq.NOBLOCK)
                else:
                    self.rep_sock.send_string(payload_json, flags=zmq.NOBLOCK)
                self.reply_count += 1
                if request_start:
                    start_ack_sent = True
            except zmq.Again:
                self.reply_drop_count += 1
                print("[Warning] reply queue full, drop one reply")
            except Exception as exc:
                print(f"[Warning] reply send failed: {exc}")
            else:
                if self.tap_accepting:
                    try:
                        self._enqueue_tap(
                            "reference",
                            {
                                "bridge_recv_monotonic_ns": int(req_recv_ns),
                                "request_start": request_start,
                                "start_control_epoch": (
                                    active_start_epoch if request_start else None
                                ),
                                "start_fresh_frame_count": sample_info.get(
                                    "fresh_frame_count"
                                ),
                                "start_selected_recv_ns": sample_info.get(
                                    "selected_recv_ns"
                                ),
                                "frame_seq_start": seq_start,
                                "frame_dt_ns": int(round(1e9 / float(self.ctrl_fps))),
                                "no_interp_applied": bool(used_fallback),
                                "sample_mode": sample_info.get("mode"),
                                "sample_target_monotonic_ns": sample_info.get("target_ns"),
                                "retarget_age_ms": retarget_age_ms,
                                "frames_qpos_root_xyz_quat_wxyz_dof": [
                                    np.asarray(frame, dtype=np.float32).reshape(-1)[:36].tolist()
                                    for frame in out_frames
                                ],
                            },
                        )
                    except Exception:
                        self._count_tap_prepare_drop()

    def _stats_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(timeout=self.log_interval_s):
                break

            with self.stats_lock:
                info = dict(self.latest_debug_info)
            with self.latest_vr_lock:
                callback_count = int(self.callback_count)
            retarget_count = int(self.retarget_count)
            req_count = int(self.req_count)
            reply_count = int(self.reply_count)
            reply_drop_count = int(self.reply_drop_count)
            req_merged_total = int(self.req_merged_total)
            fallback_count = int(self.fallback_count)
            start_gate_wait_count = int(self.start_gate_wait_count)
            start_gate_ready_count = int(self.start_gate_ready_count)
            raw_motion_drop_count = int(self.raw_motion_drop_count)
            latest_merged_reqs = int(self.latest_merged_reqs)
            latest_req_dt_ms = self.latest_req_dt_ms
            with self.tap_stats_lock:
                tap_sent_count = self.tap_sent_count
                tap_queue_drop_count = self.tap_queue_drop_count
                tap_send_drop_count = self.tap_send_drop_count
                tap_prepare_drop_count = self.tap_prepare_drop_count

            alpha = info.get("alpha")
            alpha_str = "None" if alpha is None else f"{float(alpha):.3f}"
            req_dt_str = "None" if latest_req_dt_ms is None else f"{float(latest_req_dt_ms):.2f}"
            print(
                "[Stats] "
                f"req={req_count}, rep={reply_count}, rep_drop={reply_drop_count}, "
                f"req_merged_total={req_merged_total}, latest_merged={latest_merged_reqs}, "
                f"fallback={fallback_count}, raw_drop={raw_motion_drop_count}, "
                f"start_wait={start_gate_wait_count}, start_ready={start_gate_ready_count}, "
                f"tap_sent={tap_sent_count}, tap_queue_drop={tap_queue_drop_count}, "
                f"tap_send_drop={tap_send_drop_count}, tap_prepare_drop={tap_prepare_drop_count}, "
                f"cb={callback_count}, retarget={retarget_count}, "
                f"mode={info.get('mode')}, buffer={info.get('buffer_len')}, "
                f"latest_req_dt_ms={req_dt_str}, "
                f"target_age_ms={info.get('target_age_ms')}, "
                f"older_age_ms={info.get('older_age_ms')}, "
                f"newer_age_ms={info.get('newer_age_ms')}, "
                f"span_ms={info.get('span_ms')}, alpha={alpha_str}, "
                f"retarget_age_ms={info.get('retarget_age_ms')}, "
                f"raw_motion_age_ms={info.get('raw_motion_age_ms')}"
            )

    def _control_loop(self) -> None:
        import zmq

        period_s = 1.0 / float(self.ctrl_fps)
        hand_ctrl_sock = None
        if self.hand_ctrl_bind_addr:
            try:
                hand_ctrl_sock = self.zmq_context.socket(zmq.PUB)
                hand_ctrl_sock.setsockopt(zmq.LINGER, 0)
                hand_ctrl_sock.setsockopt(zmq.SNDHWM, 1)
                hand_ctrl_sock.setsockopt(zmq.CONFLATE, 1)
                hand_ctrl_sock.bind(self.hand_ctrl_bind_addr)
                print(f"[hand_ctrl] publishing fresh controller state at {self.hand_ctrl_bind_addr}")
            except Exception as exc:
                if hand_ctrl_sock is not None:
                    hand_ctrl_sock.close(0)
                    hand_ctrl_sock = None
                print(f"[Warning] hand control publisher disabled: {exc}")

        try:
            while not self.stop_event.is_set():
                with self.latest_vr_lock:
                    buttons = dict(self.last_controller_buttons)
                    controller_source_timestamp_ns = self.latest_controller_source_timestamp_ns
                    controller_last_update_ns = self.latest_controller_recv_monotonic_ns
                    controller_last_update_wall_time_ns = self.latest_controller_recv_wall_time_ns
                    left_controller_source_valid = self.latest_left_controller_source_valid
                    right_controller_source_valid = self.latest_right_controller_source_valid
                    left_controller_update_sequence = self.latest_left_controller_update_sequence
                    right_controller_update_sequence = self.latest_right_controller_update_sequence
                    left_controller_last_update_ns = self.latest_left_controller_recv_monotonic_ns
                    right_controller_last_update_ns = self.latest_right_controller_recv_monotonic_ns
                    tracking_active = bool(self.controller_start_active)

                payload = {
                    "t_ms": int(time.time() * 1000),
                    "controller_buttons": buttons,
                }
                try:
                    self.ctrl_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as exc:
                    print(f"[Warning] control send failed: {exc}")

                sample_ns = time.monotonic_ns()
                sample_wall_time_ns = time.time_ns()
                if hand_ctrl_sock is not None:
                    hand_payload = self._build_hand_control_payload(
                        buttons=buttons,
                        tracking_active=tracking_active,
                        sample_monotonic_ns=sample_ns,
                        sample_wall_time_ns=sample_wall_time_ns,
                        controller_source_timestamp_ns=controller_source_timestamp_ns,
                        left_controller_source_valid=left_controller_source_valid,
                        right_controller_source_valid=right_controller_source_valid,
                        left_controller_update_sequence=left_controller_update_sequence,
                        right_controller_update_sequence=right_controller_update_sequence,
                        left_controller_last_update_ns=left_controller_last_update_ns,
                        right_controller_last_update_ns=right_controller_last_update_ns,
                        source_timeout_ns=self.hand_ctrl_source_timeout_ns,
                    )
                    try:
                        hand_ctrl_sock.send_string(
                            json.dumps(hand_payload, separators=(",", ":")), flags=zmq.NOBLOCK
                        )
                    except zmq.Again:
                        pass
                    except Exception as exc:
                        print(f"[Warning] hand control send failed: {exc}")

                if self.tap_accepting:
                    try:
                        self._enqueue_tap(
                            "controller",
                            {
                                "bridge_sample_monotonic_ns": sample_ns,
                                "bridge_sample_wall_time_ns": sample_wall_time_ns,
                                "controller_source_timestamp_ns": controller_source_timestamp_ns,
                                "controller_last_update_monotonic_ns": controller_last_update_ns or None,
                                "controller_last_update_wall_time_ns": controller_last_update_wall_time_ns or None,
                                "controller_age_ms": (
                                    None
                                    if controller_last_update_ns <= 0
                                    else (sample_ns - controller_last_update_ns) / 1e6
                                ),
                                "controller_buttons": buttons,
                            },
                        )
                    except Exception:
                        self._count_tap_prepare_drop()

                self.stop_event.wait(timeout=period_s)
        finally:
            if hand_ctrl_sock is not None:
                hand_ctrl_sock.close(0)

    def _tap_publisher_loop(self) -> None:
        if not self.tap_bind_addr:
            return

        import zmq

        tap_sock = self.zmq_context.socket(zmq.PUB)
        tap_sock.setsockopt(zmq.LINGER, 0)
        tap_sock.setsockopt(zmq.SNDHWM, max(100, self.tap_queue.maxsize))
        try:
            tap_sock.bind(self.tap_bind_addr)
            print(f"[teleop_tap] publishing recorder events at {self.tap_bind_addr}")
            while not self.stop_event.is_set() or not self.tap_queue.empty():
                try:
                    topic, event = self.tap_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    tap_sock.send_multipart(
                        [
                            topic.encode("utf-8"),
                            json.dumps(event, separators=(",", ":")).encode("utf-8"),
                        ],
                        flags=zmq.NOBLOCK,
                    )
                    with self.tap_stats_lock:
                        self.tap_sent_count += 1
                except zmq.Again:
                    with self.tap_stats_lock:
                        self.tap_send_drop_count += 1
                except Exception as exc:
                    with self.tap_stats_lock:
                        self.tap_send_drop_count += 1
                    print(f"[Warning] recorder tap send failed: {exc}")
                finally:
                    self.tap_queue.task_done()
        except Exception as exc:
            self.tap_accepting = False
            while True:
                try:
                    self.tap_queue.get_nowait()
                    self.tap_queue.task_done()
                except queue.Empty:
                    break
            print(f"[Warning] recorder tap disabled: failed to bind {self.tap_bind_addr}: {exc}")
        finally:
            self.tap_accepting = False
            tap_sock.close(0)

    def _visualization_loop(self) -> None:
        if self.viewer is None:
            return

        period_s = 1.0 / float(self.vis_fps)
        while not self.stop_event.is_set():
            with self.vis_lock:
                qpos = None if self.latest_vis_qpos is None else self.latest_vis_qpos.copy()
                human_motion_data = self.latest_vis_human_motion

            if qpos is not None:
                try:
                    self.viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:36],
                        human_motion_data=human_motion_data,
                        rate_limit=False,
                        follow_camera=True,
                    )
                except Exception as exc:
                    print(f"[Warning] visualization failed, disabling viewer: {exc}")
                    self.viewer.close()
                    self.viewer = None
                    return

            self.stop_event.wait(timeout=period_s)

    def setup(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for the teleop ZMQ server.") from exc

        if self.args.visualize:
            self.viewer = RobotMotionViewer(
                robot_type=self.robot,
                motion_fps=self.vis_fps,
                transparent_robot=1,
            )

        self.raw_recv_conn, self.raw_send_conn = self.mp_ctx.Pipe(duplex=False)
        self.result_recv_conn, self.result_send_conn = self.mp_ctx.Pipe(duplex=False)
        worker_config = {
            "robot": str(self.args.robot),
            "actual_human_height": float(self.args.actual_human_height),
            "gmr_max_iter": int(self.gmr_max_iter),
            "send_human_motion": bool(self.args.visualize),
            "min_link_height": self.min_link_height,
            "min_link_height_align_strategy": self.min_link_height_align_strategy,
            "min_link_height_bootstrap_frames": self.min_link_height_bootstrap_frames,
        }
        self.retarget_process = self.mp_ctx.Process(
            target=_retarget_worker_main,
            args=(self.raw_recv_conn, self.result_send_conn, worker_config),
            name="teleop-retarget-worker",
            daemon=True,
        )
        self.retarget_process.start()
        self.raw_recv_conn.close()
        self.raw_recv_conn = None
        self.result_send_conn.close()
        self.result_send_conn = None

        if not self.result_recv_conn.poll(10.0):
            raise RuntimeError("Retarget worker did not become ready within 10 seconds.")
        worker_msg = self.result_recv_conn.recv()
        if not isinstance(worker_msg, dict) or worker_msg.get("type") != "worker_ready":
            raise RuntimeError(f"Retarget worker failed to start: {worker_msg}")

        xrt.init()
        xrt.register_frame_callback(self._on_vr_frame)

        self.zmq_context = zmq.Context.instance()

        self.req_sock = self.zmq_context.socket(zmq.PULL)
        self.req_sock.setsockopt(zmq.LINGER, 0)
        self.req_sock.setsockopt(zmq.RCVHWM, 500)
        self.req_sock.bind(self.args.req_bind_addr)

        self.rep_sock = self.zmq_context.socket(zmq.PUSH)
        self.rep_sock.setsockopt(zmq.LINGER, 0)
        self.rep_sock.setsockopt(zmq.SNDHWM, 500)
        self.rep_sock.bind(self.args.rep_bind_addr)

        self.ctrl_sock = self.zmq_context.socket(zmq.PUSH)
        self.ctrl_sock.setsockopt(zmq.LINGER, 0)
        self.ctrl_sock.setsockopt(zmq.SNDHWM, 500)
        self.ctrl_sock.bind(self.args.ctrl_bind_addr)

        print("Low-latency teleop ZMQ pose server initialized")
        print(f"  req_bind_addr: {self.args.req_bind_addr}")
        print(f"  rep_bind_addr: {self.args.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.args.ctrl_bind_addr}")
        print(f"  hand_ctrl_bind_addr: {self.hand_ctrl_bind_addr or '<disabled>'}")
        print(f"  hand_ctrl_source_timeout_ms: {self.hand_ctrl_source_timeout_ns / 1e6:.3f}")
        print(f"  tap_bind_addr: {self.tap_bind_addr or '<disabled>'}")
        print(f"  ctrl_fps: {self.ctrl_fps}")
        print(f"  gmr_max_iter: {self.gmr_max_iter}")
        print("  chunk_size: fixed to 1 frame per reply")
        print(f"  lookback_ms: {self.lookback_ns / 1e6:.3f}")
        print(f"  retarget_buffer_window_s: {self.retarget_buffer_window_ns / 1e9:.3f}")
        print(f"  start_fresh_frames: {self.start_fresh_frames}")
        print(f"  start_max_retarget_age_ms: {self.start_max_retarget_age_ns / 1e6:.3f}")
        print(f"  start_fresh_wait_timeout_ms: {self.start_fresh_wait_timeout_s * 1e3:.3f}")
        print(f"  log_interval_s: {self.log_interval_s:.3f}")
        print(f"  visualize: {self.args.visualize}")
        print(f"  retarget_worker_pid: {self.retarget_process.pid if self.retarget_process else None}")

    def run(self) -> None:
        self.setup()

        self.raw_sender_thread = threading.Thread(
            target=self._raw_sender_loop,
            name="teleop-raw-sender",
            daemon=True,
        )
        self.worker_result_thread = threading.Thread(
            target=self._worker_result_loop,
            name="teleop-worker-result",
            daemon=True,
        )
        self.request_thread = threading.Thread(
            target=self._request_loop,
            name="teleop-request",
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="teleop-control",
            daemon=True,
        )
        if self.tap_bind_addr:
            self.tap_thread = threading.Thread(
                target=self._tap_publisher_loop,
                name="teleop-tap-publisher",
                daemon=True,
            )
        if self.viewer is not None:
            self.visualization_thread = threading.Thread(
                target=self._visualization_loop,
                name="teleop-visualization",
                daemon=True,
            )
        if self.log_interval_s > 0.0:
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                name="teleop-stats",
                daemon=True,
            )

        if self.tap_thread is not None:
            self.tap_thread.start()
        self.raw_sender_thread.start()
        self.worker_result_thread.start()
        self.request_thread.start()
        self.control_thread.start()
        if self.visualization_thread is not None:
            self.visualization_thread.start()
        if self.stats_thread is not None:
            self.stats_thread.start()

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting low-latency teleop ZMQ pose server.")
        finally:
            self.stop_event.set()
            self.vr_frame_event.set()
            try:
                xrt.clear_frame_callback()
            except Exception:
                pass

            for thread in (
                self.raw_sender_thread,
                self.worker_result_thread,
                self.request_thread,
                self.control_thread,
                self.tap_thread,
                self.visualization_thread,
                self.stats_thread,
            ):
                if thread is not None:
                    thread.join(timeout=1.0)

            if self.raw_send_conn is not None:
                try:
                    self.raw_send_conn.send({"type": "shutdown"})
                except Exception:
                    pass
            if self.raw_send_conn is not None:
                self.raw_send_conn.close()
            if self.raw_recv_conn is not None:
                self.raw_recv_conn.close()
            if self.result_send_conn is not None:
                self.result_send_conn.close()
            if self.result_recv_conn is not None:
                self.result_recv_conn.close()
            if self.retarget_process is not None:
                self.retarget_process.join(timeout=2.0)
                if self.retarget_process.is_alive():
                    self.retarget_process.terminate()
                    self.retarget_process.join(timeout=1.0)

            if self.viewer is not None:
                self.viewer.close()
            if self.req_sock is not None:
                self.req_sock.close(0)
            if self.rep_sock is not None:
                self.rep_sock.close(0)
            if self.ctrl_sock is not None:
                self.ctrl_sock.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency ZMQ teleop pose server")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "agibot_x2"],
        default="unitree_g1",
        help="Robot key for defaults",
    )
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--vis_fps", type=int, default=10, help="Viewer update frequency")
    parser.add_argument("--ctrl_fps", type=int, default=50, help="Controller button publish frequency")
    parser.add_argument(
        "--lookback_ms",
        type=float,
        default=15.0,
        help="Sample reply frames at request_time - lookback_ms",
    )
    parser.add_argument(
        "--retarget_buffer_window_s",
        type=float,
        default=0.5,
        help="How much retarget history to keep for timestamp interpolation",
    )
    parser.add_argument(
        "--start_fresh_frames",
        type=int,
        default=3,
        help="Post-start retarget frames required before acknowledging a VR session",
    )
    parser.add_argument(
        "--start_max_retarget_age_ms",
        type=float,
        default=80.0,
        help="Maximum age of the retarget frame used to acknowledge VR start",
    )
    parser.add_argument(
        "--start_fresh_wait_timeout_ms",
        type=float,
        default=250.0,
        help="Wait per start request before deferring acknowledgement for a retry",
    )
    parser.add_argument(
        "--log_interval_s",
        type=float,
        default=1.0,
        help="Periodic debug log interval. Set to 0 to disable.",
    )
    parser.add_argument("--req_bind_addr", type=str, default="tcp://*:28701")
    parser.add_argument("--rep_bind_addr", type=str, default="tcp://*:28702")
    parser.add_argument("--ctrl_bind_addr", type=str, default="tcp://*:28703")
    parser.add_argument(
        "--hand_ctrl_bind_addr",
        type=str,
        default="tcp://*:28705",
        help=(
            "Independent PUB endpoint for fresh analog hand-controller state. "
            "Use a dedicated endpoint because rep/ctrl are PUSH/PULL and cannot be shared."
        ),
    )
    parser.add_argument(
        "--hand_ctrl_source_timeout_ms",
        type=float,
        default=500.0,
        help="Mark hand controller samples invalid when the XR callback is older than this",
    )
    parser.add_argument(
        "--tap_bind_addr",
        type=str,
        default="",
        help=(
            "Independent PUB endpoint for recorder events (disabled by default). "
            "Recorders must use this tap instead of attaching to rep/ctrl PUSH sockets."
        ),
    )
    parser.add_argument(
        "--tap_queue_size",
        type=int,
        default=2048,
        help="Bounded recorder-tap queue; full queues drop events instead of blocking teleop",
    )
    parser.add_argument("--min_link_height", type=float, default=0.0)
    parser.add_argument(
        "--min_link_height_align_strategy",
        type=str,
        choices=["startup_fixed", "per_frame"],
        default="startup_fixed",
    )
    parser.add_argument("--min_link_height_bootstrap_frames", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime_dependencies(visualize=args.visualize)
    server = LowLatencyTeleopPoseZMQServer(args)
    server.run()


if __name__ == "__main__":
    main()
