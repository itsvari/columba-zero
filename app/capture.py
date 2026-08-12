from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .models import FramePacket


class CaptureError(RuntimeError):
    pass


class EndOfStream(CaptureError):
    pass


class FrameCapture(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


class Picamera2Capture(FrameCapture):
    def __init__(self, width: int, height: int, fps: float, rotation: int = 0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.rotation = rotation
        self._camera: Any = None

    def start(self) -> None:
        try:
            from libcamera import Transform
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CaptureError(
                "Picamera2 is unavailable; install it from Raspberry Pi OS packages"
            ) from exc
        transform = Transform(
            hflip=self.rotation in (180, 270),
            vflip=self.rotation in (180, 90),
        )
        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
            transform=transform,
            buffer_count=3,
        )
        self._camera.configure(config)
        self._camera.start()

    def read(self) -> Any:
        if self._camera is None:
            raise CaptureError("camera has not been started")
        frame = self._camera.capture_array("main")
        if frame is None:
            raise CaptureError("Picamera2 returned an empty frame")
        return frame

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None


class VideoFileCapture(FrameCapture):
    def __init__(self, path: str, realtime: bool = False) -> None:
        self.path = path
        self.realtime = realtime
        self._capture: Any = None
        self._frame_delay = 0.0
        self._last_read = 0.0

    def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CaptureError("OpenCV is required for video replay") from exc
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise CaptureError(f"cannot open video: {self.path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._frame_delay = 1.0 / fps if fps > 0 else 0.0

    def read(self) -> Any:
        if self._capture is None:
            raise CaptureError("video has not been started")
        if self.realtime and self._last_read and self._frame_delay:
            remaining = self._frame_delay - (time.monotonic() - self._last_read)
            if remaining > 0:
                time.sleep(remaining)
        ok, frame = self._capture.read()
        self._last_read = time.monotonic()
        if not ok:
            raise EndOfStream(f"video completed: {self.path}")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class LatestFrameStream:
    """Acquire continuously while retaining only the newest captured frame."""

    def __init__(
        self,
        capture: FrameCapture,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capture = capture
        self.clock = clock
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.dropped_frames = 0

    def start(self) -> None:
        self.capture.start()
        self._thread = threading.Thread(target=self._run, name="frame-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        sequence = 0
        try:
            while not self._stop.is_set():
                frame = self.capture.read()
                packet = FramePacket(sequence=sequence, captured_at=self.clock(), frame=frame)
                sequence += 1
                try:
                    self._queue.put_nowait(packet)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.dropped_frames += 1
                    self._queue.put_nowait(packet)
        except BaseException as exc:  # surfaced on the consumer thread
            self._error = exc
        finally:
            self._stop.set()

    def get(self, timeout: float) -> FramePacket:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            if self._error is not None:
                if isinstance(self._error, EndOfStream):
                    raise self._error from None
                raise CaptureError(str(self._error)) from self._error
            raise CaptureError(f"no camera frame received within {timeout:.2f}s") from exc

    def close(self) -> None:
        self._stop.set()
        self.capture.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> LatestFrameStream:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
