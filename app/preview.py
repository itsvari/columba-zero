"""Browser preview and demonstration recording of the latest processed frame.

The detection loop only swaps one latest `RenderJob` under a short lock. A daemon
encoder thread annotates, letterboxes and JPEG-encodes that job at a capped rate,
then shares the resulting bytes with HTTP clients and the recorder. Unannotated
evidence recording uses a separate JPEG at the same capped rate. Neither slow
browsers nor slow storage can reach back into detection or control.
"""

from __future__ import annotations

import ipaddress
import json
import math
import select
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast, get_args
from urllib.parse import urlsplit

from .config import AppConfig
from .demo_frames import EncodedFrame, FrameTimebase
from .json_data import JsonValue
from .models import Frame
from .output import annotate_frame
from .pan_tilt_control import ControlAction, ControllerStatus
from .pipeline import RenderJob
from .preview_page import PREVIEW_PAGE

if TYPE_CHECKING:
    from .demo_recording import FrameRecorder

MILLISECONDS_PER_SECOND = 1000.0
MAX_HTTP_CLIENTS = 8
CLIENT_SOCKET_TIMEOUT_S = 5.0
REJECTED_CLIENT_TIMEOUT_S = 0.5
MAX_CONTROL_BODY_BYTES = 256
STREAM_WAIT_S = 0.5
SERVER_POLL_INTERVAL_S = 0.1
ENCODER_SHUTDOWN_TIMEOUT_S = 3.0
SERVER_SHUTDOWN_TIMEOUT_S = 2.0
STREAM_BOUNDARY = "columbaframe"
BUSY_RESPONSE = (
    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
)
CONTROL_HEADER = "X-Columba-Control"
CONTROL_HEADER_VALUE = "1"
JSON_CONTENT_TYPE = "application/json"
CONTROL_ACTIONS: frozenset[str] = frozenset(get_args(ControlAction))
TIMEBASES: dict[str, FrameTimebase] = {
    "replay_source": "replay_source",
    "delivery_monotonic": "delivery_monotonic",
    "processing_monotonic": "processing_monotonic",
}
ClientAddress: TypeAlias = tuple[str, int] | tuple[str, int, int, int]


class PreviewEventSink(Protocol):
    def event(self, event: str, **fields: JsonValue) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishedState:
    """The light-weight parts of the newest job, readable without the frame."""

    target: JsonValue
    controller: ControllerStatus | None
    processing_fps: float | None
    capture_fps: float | None


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def letterbox(image: Frame, width_px: int, height_px: int) -> Frame:
    """Fit the whole frame inside a fixed canvas so overlays keep their geometry."""
    import cv2
    import numpy

    source_height_px, source_width_px = image.shape[:2]
    scale = min(width_px / source_width_px, height_px / source_height_px)
    fitted_width_px = min(width_px, max(1, round(source_width_px * scale)))
    fitted_height_px = min(height_px, max(1, round(source_height_px * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (fitted_width_px, fitted_height_px), interpolation=interpolation)
    canvas = numpy.zeros((height_px, width_px, 3), dtype=numpy.uint8)
    left_px = (width_px - fitted_width_px) // 2
    top_px = (height_px - fitted_height_px) // 2
    canvas[top_px : top_px + fitted_height_px, left_px : left_px + fitted_width_px] = resized
    return canvas


def encode_jpeg(image: Frame, quality: int) -> bytes:
    import cv2

    succeeded, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not succeeded:
        raise RuntimeError("JPEG encoding failed")
    return bytes(encoded.tobytes())


class LatestFrame:
    """The newest encoded frame, shared by every stream and snapshot reader."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: EncodedFrame | None = None
        self._generation = 0
        self._published_at_s: float | None = None
        self._closed = False

    def publish(self, frame: EncodedFrame, delivered_at_s: float | None = None) -> None:
        with self._condition:
            self._frame = frame
            self._generation += 1
            self._published_at_s = (
                delivered_at_s if delivered_at_s is not None else time.monotonic()
            )
            self._condition.notify_all()

    def latest(self) -> EncodedFrame | None:
        with self._condition:
            return self._frame

    def age_ms(self) -> float | None:
        with self._condition:
            published_at_s = self._published_at_s
        if published_at_s is None:
            return None
        return (time.monotonic() - published_at_s) * MILLISECONDS_PER_SECOND

    def wait_for_newer(
        self, seen_generation: int, timeout_s: float
    ) -> tuple[int, EncodedFrame | None]:
        """Block until a frame newer than `seen_generation` exists, or time out."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._generation != seen_generation, timeout=timeout_s
            )
            if self._closed or self._generation == seen_generation:
                return seen_generation, None
            return self._generation, self._frame

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class PreviewHttpServer(ThreadingHTTPServer):
    """A bounded HTTP server: at most MAX_HTTP_CLIENTS concurrent connections."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        frames: LatestFrame,
        state: Callable[[], dict[str, JsonValue]],
        control: Callable[[ControlAction], None] | None,
    ) -> None:
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.frames = frames
        self.state = state
        self.control = control
        self.allowed_hostnames = frozenset({"localhost", address[0].casefold().rstrip(".")})
        self._client_slots = threading.BoundedSemaphore(MAX_HTTP_CLIENTS)
        super().__init__(address, handler)

    def verify_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: ClientAddress
    ) -> bool:
        if self._client_slots.acquire(blocking=False):
            return True
        self._reject_busy(cast(socket.socket, request))
        return False

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: ClientAddress
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._client_slots.release()

    def _reject_busy(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(REJECTED_CLIENT_TIMEOUT_S)
            connection.sendall(BUSY_RESPONSE)
        except OSError:
            pass

    def handle_error(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: ClientAddress
    ) -> None:
        # disconnecting browsers are routine; they must not spam the console
        return


class PreviewRequestHandler(BaseHTTPRequestHandler):
    timeout = CLIENT_SOCKET_TIMEOUT_S
    server_version = "ColumbaPreview"
    sys_version = ""

    @property
    def preview_server(self) -> PreviewHttpServer:
        return cast(PreviewHttpServer, self.server)

    def log_message(self, format: str, *arguments: str | int | float) -> None:
        return

    def do_GET(self) -> None:
        if not self._accept_host():
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", PREVIEW_PAGE.encode("utf-8"))
        elif path == "/api/state":
            self._send_state()
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._accept_host():
            return
        if urlsplit(self.path).path != "/api/control":
            self._send_json(404, {"error": "not found"})
            return
        self._handle_control()

    def _accept_host(self) -> bool:
        authority = self.headers.get("Host")
        if self._allowed_host(authority):
            return True
        self._send_json(
            403, {"error": "host must be a literal IP, localhost, or the configured host"}
        )
        return False

    def _allowed_host(self, authority: str | None) -> bool:
        if authority is None:
            return False
        try:
            parsed = urlsplit(f"http://{authority}")
            hostname = parsed.hostname
            port = parsed.port
            if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                return False
            if hostname is None or (port is not None and port != self.preview_server.server_port):
                return False
            if hostname.casefold().rstrip(".") in self.preview_server.allowed_hostnames:
                return True
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False

    def _send_headers(self, status: int, content_type: str, length: int | None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_headers(status, content_type, len(body))
        for name, value in () if extra_headers is None else extra_headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Mapping[str, JsonValue]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(status, JSON_CONTENT_TYPE, body)

    def _send_state(self) -> None:
        try:
            self._send_json(200, self.preview_server.state())
        except ValueError:
            self._send_json(500, {"error": "state is not representable as JSON"})

    def _send_snapshot(self) -> None:
        frame = self.preview_server.frames.latest()
        if frame is None:
            self._send_json(503, {"error": "no frame yet"})
            return
        age_ms = self.preview_server.frames.age_ms()
        headers = {} if age_ms is None else {"X-Frame-Age-Ms": f"{age_ms:.0f}"}
        self._send_bytes(200, "image/jpeg", frame.jpeg, headers)

    def _stream(self) -> None:
        self._send_headers(200, f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}", None)
        self.end_headers()
        frames = self.preview_server.frames
        seen_generation = 0
        while not frames.closed:
            seen_generation, frame = frames.wait_for_newer(seen_generation, STREAM_WAIT_S)
            if frame is None:
                if self._peer_closed():
                    return
                continue
            part_header = (
                f"--{STREAM_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame.jpeg)}\r\n\r\n"
            ).encode("ascii")
            try:
                self.wfile.write(part_header + frame.jpeg + b"\r\n")
            except OSError:
                return

    def _peer_closed(self) -> bool:
        readable, _, _ = select.select([self.connection], [], [], 0)
        if not readable:
            return False
        try:
            return len(self.connection.recv(1, socket.MSG_PEEK)) == 0
        except OSError:
            return True

    def _foreign_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return False
        return urlsplit(origin).netloc != self.headers.get("Host")

    def _read_action(self) -> ControlAction | None:
        """Return the validated action, or send the error response and return None."""
        if self._foreign_origin():
            self._send_json(403, {"error": "foreign origin"})
            return None
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != JSON_CONTENT_TYPE:
            self._send_json(415, {"error": f"content type must be {JSON_CONTENT_TYPE}"})
            return None
        if self.headers.get(CONTROL_HEADER) != CONTROL_HEADER_VALUE:
            self._send_json(403, {"error": f"missing {CONTROL_HEADER} header"})
            return None
        length_text = self.headers.get("Content-Length")
        if length_text is None or not length_text.isdigit():
            self._send_json(411, {"error": "content length required"})
            return None
        length = int(length_text)
        if length > MAX_CONTROL_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return None
        try:
            body = self.rfile.read(length)
        except OSError:
            self._send_json(408, {"error": "request body timed out"})
            return None
        return self._parse_action(body)

    def _parse_action(self, body: bytes) -> ControlAction | None:
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            self._send_json(400, {"error": "body is not valid JSON"})
            return None
        action = payload.get("action") if isinstance(payload, dict) else None
        if not isinstance(action, str) or action not in CONTROL_ACTIONS:
            self._send_json(400, {"error": "action must be pause, resume or home"})
            return None
        return cast(ControlAction, action)

    def _handle_control(self) -> None:
        action = self._read_action()
        if action is None:
            return
        control = self.preview_server.control
        if control is None:
            self._send_json(409, {"error": "controls are not available"})
            return
        try:
            control(action)
        except Exception as error:  # the callback is foreign code
            self._send_json(500, {"error": f"control failed: {error}"})
            return
        self._send_json(200, {"ok": True})


class PreviewService:
    def __init__(
        self,
        config: AppConfig,
        events: PreviewEventSink,
        recorder: FrameRecorder | None = None,
        control: Callable[[ControlAction], None] | None = None,
        controller_state: Callable[[], ControllerStatus] | None = None,
    ) -> None:
        self._config = config
        self._events = events
        self._recorder = recorder
        self._control = control
        self._controller_state = controller_state
        self._active = config.preview.enabled or config.preview.record
        self._min_interval_s = 1.0 / config.preview.max_fps if config.preview.max_fps > 0 else 0.0
        self._job_condition = threading.Condition()
        self._pending: RenderJob | None = None
        self._published: PublishedState | None = None
        self._closing = False
        self._started = False
        self._closed = False
        self._overwritten_frames = 0
        self._encoded_frames = 0
        self._last_encode_ms: float | None = None
        self._failure: str | None = None
        self._encoder_failed = False
        self._recorder_usable = recorder is not None
        self._frames = LatestFrame()
        self._encoder: threading.Thread | None = None
        self._server: PreviewHttpServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def failure(self) -> str | None:
        with self._job_condition:
            return self._failure

    @property
    def address(self) -> tuple[str, int] | None:
        server = self._server
        if server is None:
            return None
        host, port = server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if not self._active or self._started:
            return
        self._started = True
        if self._config.preview.enabled:
            self._start_server()
        self._start_recorder()
        self._encoder = threading.Thread(
            target=self._encoder_loop, name="preview-encoder", daemon=True
        )
        self._encoder.start()
        address = self.address
        self._events.event(
            "preview_started",
            host=address[0] if address is not None else None,
            port=address[1] if address is not None else None,
            record=self._recorder is not None,
        )

    def _start_server(self) -> None:
        preview = self._config.preview
        try:
            server = PreviewHttpServer(
                (preview.host, preview.port),
                PreviewRequestHandler,
                self._frames,
                self.state,
                self._control,
            )
        except OSError as error:
            self._record_failure(
                f"preview server cannot bind {preview.host}:{preview.port}: {error}"
            )
            raise
        self._server = server
        self._server_thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": SERVER_POLL_INTERVAL_S},
            name="preview-http",
            daemon=True,
        )
        self._server_thread.start()

    def _start_recorder(self) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.start()
        except Exception as error:  # recording must never stop the preview or detection
            self._recorder_usable = False
            self._record_failure(f"recorder failed to start: {error}")

    def publish(self, job: RenderJob) -> None:
        if not self._active:
            return
        with self._job_condition:
            if self._closing or self._encoder_failed:
                return
            if self._pending is not None:
                self._overwritten_frames += 1
            self._pending = job
            self._published = PublishedState(
                target=job.target.as_dict() if job.target is not None else None,
                controller=job.controller,
                processing_fps=finite_or_none(job.processing_fps),
                capture_fps=finite_or_none(job.capture_fps),
            )
            self._job_condition.notify()

    def _encoder_alive(self) -> bool:
        encoder = self._encoder
        return encoder is not None and encoder.is_alive()

    def state(self) -> dict[str, JsonValue]:
        with self._job_condition:
            published = self._published
            overwritten_frames = self._overwritten_frames
            encoded_frames = self._encoded_frames
            last_encode_ms = self._last_encode_ms
            failure = self._failure
            running = (
                self._started
                and not self._closing
                and not self._encoder_failed
                and self._encoder_alive()
            )
        controller = self._live_controller(published.controller if published else None)
        performance: dict[str, JsonValue] = {
            "processing_fps": published.processing_fps if published else None,
            "capture_fps": published.capture_fps if published else None,
            "encoded_frames": encoded_frames,
            "overwritten_frames": overwritten_frames,
            "last_encode_ms": last_encode_ms,
        }
        preview: dict[str, JsonValue] = {
            "running": running,
            "frame_age_ms": finite_or_none(self._frames.age_ms()),
            "failure": failure,
        }
        return {
            "target": published.target if published else None,
            "controller": controller.as_dict() if controller is not None else None,
            "performance": performance,
            "preview": preview,
            "recording": self._recording_snapshot(),
            "controls_enabled": self._control is not None,
        }

    def _live_controller(self, fallback: ControllerStatus | None) -> ControllerStatus | None:
        if self._controller_state is None:
            return fallback
        try:
            return self._controller_state()
        except Exception:  # a failing callback falls back to the job's own snapshot
            return fallback

    def _recording_snapshot(self) -> JsonValue:
        if self._recorder is None:
            return None
        try:
            return self._recorder.snapshot().as_dict()
        except Exception:  # state must stay available when storage misbehaves
            return None

    def _record_failure(self, message: str) -> None:
        with self._job_condition:
            self._failure = message if self._failure is None else f"{self._failure}; {message}"
        self._events.event("preview_failure", error=message)

    def _take_job(self, not_before_s: float) -> RenderJob | None:
        """Wait for the newest job, no sooner than the frame-rate cap allows."""
        with self._job_condition:
            while True:
                if self._pending is not None:
                    remaining_s = not_before_s - time.monotonic()
                    if self._closing or remaining_s <= 0:
                        job, self._pending = self._pending, None
                        return job
                    self._job_condition.wait(timeout=remaining_s)
                elif self._closing:
                    return None
                else:
                    self._job_condition.wait()

    def _encoder_loop(self) -> None:
        not_before_s = 0.0
        try:
            while True:
                job = self._take_job(not_before_s)
                if job is None:
                    return
                picked_at_s = time.monotonic()
                not_before_s = picked_at_s + self._min_interval_s
                self._encode_and_share(job, picked_at_s)
        except Exception as error:  # an encoder fault is reported, never propagated
            with self._job_condition:
                self._encoder_failed = True
                self._pending = None
            self._record_failure(f"preview encoder failed: {error}")

    def _encode_and_share(self, job: RenderJob, picked_at_s: float) -> None:
        preview = self._config.preview
        controller = job.controller
        started_at_s = time.monotonic()
        annotated = annotate_frame(
            job.frame,
            job.detections,
            job.processing_fps,
            job.capture_fps,
            self._config.camera.fps,
            job.confirmed_track_ids,
            job.target,
            controller,
            acquisition_confidence=(
                self._config.model.confidence if self._config.follow.enabled else None
            ),
        )
        fitted = letterbox(annotated, preview.width_px, preview.height_px)
        jpeg = encode_jpeg(fitted, preview.jpeg_quality)
        frame = self._build_frame(job, picked_at_s, jpeg, controller)
        recording_frame: EncodedFrame | None = frame
        if preview.record_unannotated and self._recorder_usable:
            recording_frame = self._unannotated_frame(job, frame)
        encode_ms = (time.monotonic() - started_at_s) * MILLISECONDS_PER_SECOND
        self._frames.publish(
            frame,
            delivered_at_s=(
                job.delivered_monotonic_s if job.delivered_monotonic_s is not None else picked_at_s
            ),
        )
        with self._job_condition:
            self._encoded_frames += 1
            self._last_encode_ms = encode_ms
        if recording_frame is not None:
            self._offer_to_recorder(recording_frame)

    def _unannotated_frame(self, job: RenderJob, frame: EncodedFrame) -> EncodedFrame | None:
        preview = self._config.preview
        try:
            image = letterbox(job.frame, preview.width_px, preview.height_px)
            return replace(
                frame,
                jpeg=encode_jpeg(image, preview.jpeg_quality),
                metadata=MappingProxyType({**frame.metadata, "annotations_included": False}),
            )
        except Exception as error:
            self._recorder_usable = False
            self._record_failure(f"unannotated recording encode failed: {error}")
            return None

    def _build_frame(
        self, job: RenderJob, picked_at_s: float, jpeg: bytes, controller: ControllerStatus | None
    ) -> EncodedFrame:
        if job.presentation_time_s is not None:
            monotonic_s, timebase = job.presentation_time_s, TIMEBASES["replay_source"]
        elif job.delivered_monotonic_s is not None:
            monotonic_s, timebase = job.delivered_monotonic_s, TIMEBASES["delivery_monotonic"]
        else:
            monotonic_s, timebase = picked_at_s, TIMEBASES["processing_monotonic"]
        metadata: dict[str, JsonValue] = {
            "annotations_included": True,
            "source_width_px": job.frame.shape[1],
            "source_height_px": job.frame.shape[0],
            "recording_width_px": self._config.preview.width_px,
            "recording_height_px": self._config.preview.height_px,
            "detections": cast(
                list[JsonValue], [detection.as_dict() for detection in job.detections]
            ),
            "confirmed_track_ids": cast(list[JsonValue], sorted(job.confirmed_track_ids)),
            "target": job.target.as_dict() if job.target is not None else None,
            "controller": controller.as_dict() if controller is not None else None,
            "processing_fps": finite_or_none(job.processing_fps),
            "capture_fps": finite_or_none(job.capture_fps),
        }
        return EncodedFrame(
            sequence=job.sequence,
            captured_at=job.captured_at,
            monotonic_s=monotonic_s,
            timebase=timebase,
            jpeg=jpeg,
            metadata=MappingProxyType(metadata),
        )

    def _offer_to_recorder(self, frame: EncodedFrame) -> None:
        recorder = self._recorder
        if recorder is None or not self._recorder_usable:
            return
        try:
            recorder.offer(frame)
        except Exception as error:  # storage faults must not stop the live preview
            self._recorder_usable = False
            self._record_failure(f"recorder rejected a frame: {error}")

    def close(self) -> None:
        """Drain the newest accepted frame, then stop; every wait is bounded."""
        with self._job_condition:
            if self._closed:
                return
            self._closed = True
            self._closing = True
            self._job_condition.notify_all()
        self._stop_encoder()
        self._frames.close()
        self._stop_server()
        self._events.event(
            "preview_stopped",
            encoded_frames=self._encoded_frames,
            overwritten_frames=self._overwritten_frames,
            failure=self.failure,
        )

    def _stop_encoder(self) -> None:
        encoder = self._encoder
        if encoder is None:
            return
        encoder.join(timeout=ENCODER_SHUTDOWN_TIMEOUT_S)
        if encoder.is_alive():
            self._record_failure(
                f"preview encoder did not stop within {ENCODER_SHUTDOWN_TIMEOUT_S:g} s"
            )

    def _stop_server(self) -> None:
        server, thread = self._server, self._server_thread
        if server is None or thread is None:
            return
        server.shutdown()
        thread.join(timeout=SERVER_SHUTDOWN_TIMEOUT_S)
        server.server_close()
        if thread.is_alive():
            self._record_failure(
                f"preview server did not stop within {SERVER_SHUTDOWN_TIMEOUT_S:g} s"
            )
