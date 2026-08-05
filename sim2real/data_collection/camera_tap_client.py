"""Reconnectable client for the X2 vision bridge's compressed-camera tap."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit


CAMERA_FRAME_MAGIC = b"X2CF"
CAMERA_FRAME_VERSION = 1
CAMERA_FRAME_HEADER = struct.Struct("!4sBII")
MAX_METADATA_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

EventFn = Callable[[Dict[str, Any]], None]
DropFn = Callable[[str, int], None]
LogFn = Callable[[str], None]


class CameraTapProtocolError(ValueError):
    """Raised when the peer sends a malformed or unsupported frame."""


class _StopRequested(Exception):
    pass


@dataclass(frozen=True)
class CameraTapStatus:
    connected: bool
    connections: int
    received_frames: int
    received_bytes: int
    transport_gaps: int
    protocol_errors: int
    connection_errors: int
    last_error: str


def parse_camera_tap_addr(address: str) -> Tuple[str, int]:
    """Parse the CLI's ``tcp://host:port`` address."""

    parsed = urlsplit(str(address))
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid camera tap address {address!r}: {exc}") from exc
    if (
        parsed.scheme != "tcp"
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"invalid camera tap address {address!r}; expected tcp://host:port"
        )
    return parsed.hostname, int(port)


def _recv_exact(
    current_socket: socket.socket,
    length: int,
    stop_event: Optional[threading.Event] = None,
) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        if stop_event is not None and stop_event.is_set():
            raise _StopRequested
        try:
            chunk = current_socket.recv(length - len(chunks))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("camera tap peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def read_camera_tap_frame(
    current_socket: socket.socket,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[Dict[str, Any], bytes]:
    """Read and validate one complete ``X2CF`` frame from a TCP socket."""

    header = _recv_exact(current_socket, CAMERA_FRAME_HEADER.size, stop_event)
    magic, version, metadata_length, payload_length = CAMERA_FRAME_HEADER.unpack(header)
    if magic != CAMERA_FRAME_MAGIC:
        raise CameraTapProtocolError(f"invalid camera frame magic: {magic!r}")
    if version != CAMERA_FRAME_VERSION:
        raise CameraTapProtocolError(f"unsupported camera frame version: {version}")
    if metadata_length <= 0 or metadata_length > MAX_METADATA_BYTES:
        raise CameraTapProtocolError(
            f"invalid camera metadata length: {metadata_length}"
        )
    if payload_length <= 0 or payload_length > MAX_PAYLOAD_BYTES:
        raise CameraTapProtocolError(f"invalid camera payload length: {payload_length}")

    metadata_bytes = _recv_exact(current_socket, metadata_length, stop_event)
    payload = _recv_exact(current_socket, payload_length, stop_event)
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CameraTapProtocolError("invalid camera frame JSON metadata") from exc
    if not isinstance(metadata, dict):
        raise CameraTapProtocolError("camera frame metadata must be a JSON object")
    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CameraTapProtocolError("invalid camera metadata schema_version")
    if schema_version != CAMERA_FRAME_VERSION:
        raise CameraTapProtocolError(
            f"camera metadata schema mismatch: expected {CAMERA_FRAME_VERSION}, "
            f"got {metadata.get('schema_version')!r}"
        )
    for field in (
        "sequence",
        "source_timestamp_ns",
        "bridge_recv_monotonic_ns",
        "bridge_recv_wall_time_ns",
    ):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CameraTapProtocolError(
                f"camera metadata {field} must be a non-negative integer"
            )
    for field in ("frame_id", "format", "topic"):
        if not isinstance(metadata.get(field), str):
            raise CameraTapProtocolError(f"camera metadata {field} must be a string")
    return metadata, payload


class CameraTapClient:
    """Receive camera frames on a background reconnect loop.

    The reader never blocks the recorder's ZMQ or ROS callback threads.  A
    disconnect discards only the incomplete frame and retries the same address.
    """

    def __init__(
        self,
        address: str,
        on_event: EventFn,
        note_drop: DropFn,
        log_info: LogFn = print,
        log_warning: LogFn = print,
        connect_timeout_s: float = 1.0,
        read_timeout_s: float = 0.5,
        reconnect_initial_s: float = 0.2,
        reconnect_max_s: float = 2.0,
    ) -> None:
        self.address = str(address)
        self._host, self._port = parse_camera_tap_addr(self.address)
        self._on_event = on_event
        self._note_drop = note_drop
        self._log_info = log_info
        self._log_warning = log_warning
        self._connect_timeout_s = max(0.05, float(connect_timeout_s))
        self._read_timeout_s = max(0.05, float(read_timeout_s))
        self._reconnect_initial_s = max(0.01, float(reconnect_initial_s))
        self._reconnect_max_s = max(
            self._reconnect_initial_s, float(reconnect_max_s)
        )

        self._stop_event = threading.Event()
        self._socket_lock = threading.Lock()
        self._current_socket: Optional[socket.socket] = None
        self._status_lock = threading.Lock()
        self._connected = False
        self._connections = 0
        self._received_frames = 0
        self._received_bytes = 0
        self._transport_gaps = 0
        self._protocol_errors = 0
        self._connection_errors = 0
        self._last_error = ""
        self._last_sequence: Optional[int] = None
        self._thread = threading.Thread(
            target=self._run, name="x2-camera-tap-client", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._socket_lock:
            current_socket = self._current_socket
        self._close_socket(current_socket)
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def status(self) -> CameraTapStatus:
        with self._status_lock:
            return CameraTapStatus(
                connected=self._connected,
                connections=self._connections,
                received_frames=self._received_frames,
                received_bytes=self._received_bytes,
                transport_gaps=self._transport_gaps,
                protocol_errors=self._protocol_errors,
                connection_errors=self._connection_errors,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        retry_delay = self._reconnect_initial_s
        last_reported_error = ""
        while not self._stop_event.is_set():
            try:
                current_socket = socket.create_connection(
                    (self._host, self._port), timeout=self._connect_timeout_s
                )
                current_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                current_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                current_socket.settimeout(self._read_timeout_s)
                with self._socket_lock:
                    self._current_socket = current_socket
                with self._status_lock:
                    self._connected = True
                    self._connections += 1
                    self._last_error = ""
                self._log_info(f"[camera_tap] connected to {self.address}")
                last_reported_error = ""
                retry_delay = self._reconnect_initial_s
                try:
                    self._receive_loop(current_socket)
                finally:
                    self._close_socket(current_socket)
                    with self._socket_lock:
                        if self._current_socket is current_socket:
                            self._current_socket = None
                    with self._status_lock:
                        self._connected = False
                if not self._stop_event.is_set():
                    self._log_warning(
                        f"[camera_tap] disconnected from {self.address}; reconnecting"
                    )
            except _StopRequested:
                break
            except CameraTapProtocolError as exc:
                error = str(exc)
                with self._status_lock:
                    self._protocol_errors += 1
                    self._last_error = error
                self._log_warning(
                    f"[camera_tap] protocol error from {self.address}: {error}; reconnecting"
                )
            except (ConnectionError, OSError) as exc:
                error = str(exc)
                with self._status_lock:
                    self._connection_errors += 1
                    self._last_error = error
                # Avoid printing an identical connection-refused line at every
                # retry while still making the initial problem visible.
                if error != last_reported_error:
                    self._log_warning(
                        f"[camera_tap] cannot receive from {self.address}: "
                        f"{error}; reconnecting"
                    )
                    last_reported_error = error

            if not self._stop_event.wait(retry_delay):
                retry_delay = min(self._reconnect_max_s, retry_delay * 2.0)

    def _receive_loop(self, current_socket: socket.socket) -> None:
        while not self._stop_event.is_set():
            metadata, payload = read_camera_tap_frame(
                current_socket, self._stop_event
            )
            recv_monotonic_ns = time.monotonic_ns()
            recv_wall_time_ns = time.time_ns()
            try:
                sequence = int(metadata["sequence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CameraTapProtocolError(
                    "camera metadata is missing an integer sequence"
                ) from exc
            if sequence < 0:
                raise CameraTapProtocolError(
                    f"camera metadata sequence must be non-negative: {sequence}"
                )

            gap = 0
            sequence_reset = False
            if self._last_sequence is not None:
                if sequence > self._last_sequence:
                    gap = max(0, sequence - self._last_sequence - 1)
                else:
                    # The robot-side vision bridge may have restarted.  TCP is
                    # ordered, so a non-increasing sequence marks a new epoch.
                    sequence_reset = True
            self._last_sequence = sequence

            event = dict(metadata)
            event.update(
                {
                    "stream": "camera_head",
                    "data": payload,
                    "recorder_recv_monotonic_ns": recv_monotonic_ns,
                    "recorder_recv_wall_time_ns": recv_wall_time_ns,
                    "camera_tap_addr": self.address,
                }
            )
            if gap:
                event["camera_tap_gap_before"] = gap
                self._note_drop("camera_tap_transport", gap)
            if sequence_reset:
                event["camera_tap_sequence_reset"] = True
            self._on_event(event)
            with self._status_lock:
                self._received_frames += 1
                self._received_bytes += len(payload)
                self._transport_gaps += gap

    @staticmethod
    def _close_socket(current_socket: Optional[socket.socket]) -> None:
        if current_socket is None:
            return
        try:
            current_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            current_socket.close()
        except OSError:
            pass
