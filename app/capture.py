from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import cv2
    from picamera2 import Picamera2

from .models import Frame, FramePacket

CAMERA_BUFFER_COUNT = 3
CAPTURE_SHUTDOWN_TIMEOUT_S = 3.0


class LiveCapture(Protocol):
    def start(self) -> None: ...
    def read(self) -> Frame: ...
    def close(self) -> None: ...


class CaptureError(RuntimeError):
    pass


class EndOfStream(CaptureError):
    pass


class Picamera2Capture:
    """Picamera2 RGB888 is byte-ordered BGR, as required by OpenCV/Ultralytics."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        self.width, self.height, self.fps = width, height, fps
        self._camera: Picamera2 | None = None

    def start(self) -> None:
        try:
            from libcamera import controls
            from picamera2 import Picamera2

            self._camera = Picamera2()
            requested_controls = {"FrameRate": self.fps}
            if "AfMode" in self._camera.camera_controls:
                requested_controls["AfMode"] = controls.AfModeEnum.Continuous
            config = self._camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls=requested_controls,
                buffer_count=CAMERA_BUFFER_COUNT,
                queue=False,
            )
            self._camera.configure(config)
            self._camera.start()
        except Exception as error:
            raise CaptureError(
                f"cannot start CSI camera: {error}; install OS Picamera2 packages"
            ) from error

    def read(self) -> Frame:
        try:
            if self._camera is None:
                raise CaptureError("CSI camera has not been started")
            frame = self._camera.capture_array("main")
            if frame is None:
                raise ValueError("empty frame")
            return cast(Frame, frame)
        except Exception as error:
            raise CaptureError(f"CSI capture failed: {error}") from error

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.stop()
            finally:
                camera.close()


class VideoFileCapture:
    """Read every frame sequentially; no replay pacing and no frame dropping."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._capture: cv2.VideoCapture | None = None
        self.sequence = 0
        self.fps = 0.0
        self.frame_count = 0

    def start(self) -> None:
        import cv2

        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise CaptureError(f"cannot open video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = int(count) if math.isfinite(count) and count > 0 else 0

    def get(self, timeout_s: float) -> FramePacket:
        if self._capture is None:
            raise CaptureError("video capture has not been started")
        decoded, frame = self._capture.read()
        if not decoded:
            if self.sequence == 0 or (self.frame_count and self.sequence < self.frame_count):
                raise CaptureError(
                    f"video decoder failed at frame {self.sequence} "
                    f"(declared frame count: {self.frame_count})"
                )
            raise EndOfStream("video completed")
        packet = FramePacket(self.sequence, time.time(), cast(Frame, frame))
        self.sequence += 1
        return packet

    dropped_frames = 0

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class LatestFrameStream:
    """Single-slot live buffer; sequence gaps identify discarded captured frames."""

    def __init__(self, capture: LiveCapture) -> None:
        self.capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._packet: FramePacket | None = None
        self._error: Exception | None = None
        self.dropped_frames = 0
        self.captured_frames = 0
        self.started_at_monotonic_s = 0.0

    def start(self) -> None:
        self.capture.start()
        self.started_at_monotonic_s = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.capture.read()
                packet = FramePacket(self.captured_frames, time.time(), frame)
                with self._condition:
                    self.captured_frames += 1
                    if self._packet is not None:
                        self.dropped_frames += 1
                    self._packet = packet
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def get(self, timeout_s: float) -> FramePacket:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._packet is not None or self._error is not None, timeout=timeout_s
            )
            if self._error is not None:
                raise CaptureError(f"live capture failed: {self._error}") from self._error
            if not ready:
                raise CaptureError(f"no camera frame within {timeout_s:.2f}s")
            packet, self._packet = self._packet, None
            assert packet is not None
            return packet

    @property
    def capture_fps(self) -> float:
        elapsed_s = time.monotonic() - self.started_at_monotonic_s
        return (
            self.captured_frames / elapsed_s
            if elapsed_s > 0 and self.started_at_monotonic_s
            else 0.0
        )

    def close(self) -> None:
        self._stop.set()
        # stopping the camera unblocks an outstanding capture_array call.
        try:
            self.capture.close()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=CAPTURE_SHUTDOWN_TIMEOUT_S)
                if self._thread.is_alive():
                    raise CaptureError(
                        f"capture thread did not stop within {CAPTURE_SHUTDOWN_TIMEOUT_S:g} seconds"
                    )
