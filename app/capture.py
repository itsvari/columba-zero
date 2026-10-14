from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import cv2
    from picamera2 import Picamera2

from .models import Frame, FramePacket

CAMERA_BUFFER_COUNT = 3
CAPTURE_SHUTDOWN_TIMEOUT_S = 3.0
NANOSECONDS_PER_MILLISECOND = 1_000_000
MAXIMUM_PLAUSIBLE_DELIVERY_MS = 10_000.0


class LiveCapture(Protocol):
    def start(self) -> None: ...
    def read(self) -> Frame: ...
    def close(self) -> None: ...


class CaptureError(RuntimeError):
    pass


class EndOfStream(CaptureError):
    pass


@dataclass(frozen=True, slots=True)
class CaptureMetadata:
    lux: float | None = None
    delivery_ms: float | None = None
    exposure_us: int | None = None


def sensor_clock_ns() -> int:
    """libcamera's SensorTimestamp clock (CLOCK_BOOTTIME on Linux)."""
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    return time.clock_gettime_ns(boottime) if boottime is not None else time.monotonic_ns()


def read_capture_metadata(
    metadata: object, now_ns: Callable[[], int] | None = None
) -> CaptureMetadata:
    if not isinstance(metadata, dict):
        return CaptureMetadata()
    lux = metadata.get("Lux")
    exposure_us = metadata.get("ExposureTime")
    sensor_timestamp_ns = metadata.get("SensorTimestamp")
    delivery_ms: float | None = None
    if isinstance(sensor_timestamp_ns, int) and not isinstance(sensor_timestamp_ns, bool):
        clock_ns = (now_ns or sensor_clock_ns)()
        delivery_ms = (clock_ns - sensor_timestamp_ns) / NANOSECONDS_PER_MILLISECOND
        if not 0 <= delivery_ms <= MAXIMUM_PLAUSIBLE_DELIVERY_MS:
            delivery_ms = None
    return CaptureMetadata(
        float(lux) if isinstance(lux, (int, float)) and math.isfinite(lux) else None,
        delivery_ms,
        exposure_us if isinstance(exposure_us, int) else None,
    )


def cap_exposure(tuning: object, max_exposure_us: int) -> bool:
    """Clamp every AGC exposure mode's exposure steps; False if the tuning has none.

    This is the Picamera2 manual's way to bound automatic exposure. Older tuning
    files call the steps `shutter`, newer ones `exposure`; version 2 files nest the
    AGC settings in `channels`.
    """
    if not isinstance(tuning, dict):
        return False
    algorithms = tuning.get("algorithms")
    candidates = (
        [algorithm for algorithm in algorithms if isinstance(algorithm, dict)]
        if isinstance(algorithms, list)
        else [tuning]
    )
    capped = False
    for algorithm in candidates:
        agc = algorithm.get("rpi.agc")
        if not isinstance(agc, dict):
            continue
        channels = agc.get("channels")
        for channel in channels if isinstance(channels, list) else [agc]:
            modes = channel.get("exposure_modes") if isinstance(channel, dict) else None
            for mode in modes.values() if isinstance(modes, dict) else ():
                for key in ("shutter", "exposure"):
                    steps = mode.get(key) if isinstance(mode, dict) else None
                    if isinstance(steps, list):
                        mode[key] = [min(step, max_exposure_us) for step in steps]
                        capped = True
    return capped


def even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


class Picamera2Capture:
    """Picamera2 RGB888 is byte-ordered BGR, as required by OpenCV/Ultralytics.

    `stream: lores` lets the camera hardware scale the full field of view down to
    `width` x `height`. Unsupported options fall back to the previous behaviour
    with a warning in `warnings` (printed to stderr at start).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        *,
        stream: str = "main",
        main_size: tuple[int, int] | None = None,
        lens_position: float | None = None,
        max_exposure_us: int | None = None,
        tuning_file: str | None = None,
        camera_index: int = 0,
        name: str | None = None,
    ) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.stream = stream
        self.main_size = main_size
        self.lens_position = lens_position
        self.max_exposure_us = max_exposure_us
        self.tuning_file = tuning_file
        self.camera_index = camera_index
        self.name = name or f"camera{camera_index}"
        self.warnings: list[str] = []
        self.last_metadata = CaptureMetadata()
        self._convert_yuv = False
        self._camera: Picamera2 | None = None
        self._stopped = False

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"camera {self.name}: {message}", file=sys.stderr)

    def _tuning(self, picamera2: type[Picamera2]) -> dict[str, object] | None:
        if self.tuning_file is None and self.max_exposure_us is None:
            return None
        try:
            if self.tuning_file is not None and Path(self.tuning_file).is_file():
                tuning = json.loads(Path(self.tuning_file).read_text(encoding="utf-8"))
            elif self.tuning_file is not None:
                tuning = picamera2.load_tuning_file(self.tuning_file)
            else:
                model = picamera2.global_camera_info()[self.camera_index]["Model"]
                tuning = picamera2.load_tuning_file(f"{model}.json")
        except Exception as error:
            self.warn(f"cannot load tuning file ({error}); using the default tuning")
            return None
        if not isinstance(tuning, dict):
            self.warn("tuning file is not a JSON mapping; using the default tuning")
            return None
        if self.max_exposure_us is not None and not cap_exposure(tuning, self.max_exposure_us):
            self.warn("tuning has no AGC exposure modes; max_exposure_us not applied")
        return cast(dict[str, object], tuning)

    def _controls(self, camera: Picamera2, controls: object) -> dict[str, object]:
        available = camera.camera_controls
        requested: dict[str, object] = {"FrameRate": self.fps}
        autofocus = cast(object, getattr(controls, "AfModeEnum", None))
        if self.lens_position is not None:
            if "LensPosition" in available and "AfMode" in available:
                position = self.lens_position
                limits = available["LensPosition"]
                if isinstance(limits, tuple) and len(limits) >= 2:
                    low, high = limits[0], limits[1]
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                        position = min(max(position, low), high)
                requested["AfMode"] = getattr(autofocus, "Manual", None)
                requested["LensPosition"] = position
                return requested
            self.warn("lens_position is not supported; using the default focus")
        if "AfMode" in available:
            requested["AfMode"] = getattr(autofocus, "Continuous", None)
        return requested

    def _main_size(self, camera: Picamera2) -> tuple[int, int]:
        if self.main_size is not None:
            return self.main_size
        sensor = getattr(camera, "sensor_resolution", None)
        if isinstance(sensor, tuple) and len(sensor) == 2:
            # a 2x2 binned mode keeps the full field of view at video frame rates.
            scale = min(sensor[0] / 2 / self.width, sensor[1] / 2 / self.height)
            if scale >= 1:
                return even(self.width * scale), even(self.height * scale)
        return self.width, self.height

    def _configure(self, camera: Picamera2, requested_controls: dict[str, object]) -> None:
        size = (self.width, self.height)
        if self.stream == "main":
            camera.configure(
                camera.create_video_configuration(
                    main={"size": size, "format": "RGB888"},
                    controls=requested_controls,
                    buffer_count=CAMERA_BUFFER_COUNT,
                    queue=False,
                )
            )
            return
        main = {"size": self._main_size(camera), "format": "YUV420"}
        try:
            camera.configure(
                camera.create_video_configuration(
                    main=main,
                    lores={"size": size, "format": "RGB888"},
                    controls=requested_controls,
                    buffer_count=CAMERA_BUFFER_COUNT,
                    queue=False,
                )
            )
        except Exception as error:
            # Pi 4 and earlier only offer YUV420 on the low-resolution stream.
            self.warn(f"RGB lores stream unavailable ({error}); converting YUV420")
            camera.configure(
                camera.create_video_configuration(
                    main=main,
                    lores={"size": size, "format": "YUV420"},
                    controls=requested_controls,
                    buffer_count=CAMERA_BUFFER_COUNT,
                    queue=False,
                )
            )
            self._convert_yuv = True

    def start(self) -> None:
        try:
            from libcamera import controls
            from picamera2 import Picamera2

            tuning = self._tuning(Picamera2)
            self._stopped = False
            self._camera = (
                Picamera2(self.camera_index)
                if tuning is None
                else Picamera2(self.camera_index, tuning=tuning)
            )
            self._configure(self._camera, self._controls(self._camera, controls))
            self._camera.start()
        except Exception as error:
            raise CaptureError(
                f"cannot start CSI camera: {error}; install OS Picamera2 packages"
            ) from error

    def read(self) -> Frame:
        try:
            if self._camera is None:
                raise CaptureError("CSI camera has not been started")
            request = self._camera.capture_request()
            try:
                frame = request.make_array(self.stream)
                metadata = request.get_metadata()
            finally:
                request.release()
            if frame is None:
                raise ValueError("empty frame")
            self.last_metadata = read_capture_metadata(metadata)
            if self._convert_yuv:
                import cv2

                frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2BGR)
            return cast(Frame, frame)
        except Exception as error:
            raise CaptureError(f"CSI capture failed: {error}") from error

    def close(self) -> None:
        """Stop and release the camera. The camera is kept until its release succeeds,
        so a later close retries it; a stop that succeeded is not repeated."""
        camera = self._camera
        if camera is None:
            return
        stop_error: Exception | None = None
        if not self._stopped:
            try:
                camera.stop()
                self._stopped = True
            except Exception as error:
                stop_error = error
        camera.close()
        self._camera, self._stopped = None, False
        if stop_error is not None:
            raise stop_error


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
        packet = FramePacket(
            self.sequence, time.time(), cast(Frame, frame), delivered_monotonic_s=time.monotonic()
        )
        self.sequence += 1
        return packet

    dropped_frames = 0

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class LatestFrameStream:
    """Single-slot live buffer; sequence gaps identify discarded captured frames."""

    def __init__(self, capture: LiveCapture, name: str | None = None) -> None:
        self.capture = capture
        self.name = name or (capture.name if isinstance(capture, Picamera2Capture) else None)
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
                metadata = (
                    self.capture.last_metadata
                    if isinstance(self.capture, Picamera2Capture)
                    else CaptureMetadata()
                )
                packet = FramePacket(
                    self.captured_frames,
                    time.time(),
                    frame,
                    delivered_monotonic_s=time.monotonic(),
                    camera=self.name,
                    lux=metadata.lux,
                    delivery_ms=metadata.delivery_ms,
                )
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
