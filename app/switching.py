"""Day/night camera switching: one active camera, chosen by light level with hysteresis."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import replace

from .capture import CaptureError, LatestFrameStream
from .config import CameraConfig, CameraSwitchConfig
from .json_data import JsonValue
from .models import FramePacket

MILLISECONDS_PER_SECOND = 1000
# after a failed switch, stay on the working camera this long before trying again.
SWITCH_RETRY_S = 60.0


class LuxHysteresis:
    """Decide when to change camera from the active camera's lux estimate.

    Day → night when lux stays below `night_below_lux` for `hold_s`; night → day
    when it stays above `day_above_lux` for `hold_s`. Missing readings reset the
    timer. Each camera reports its own lux, and the NoIR camera under an infrared
    illuminator reads brighter than the scene looks, so the gap between the two
    thresholds must be calibrated on site (WS10).
    """

    def __init__(self, config: CameraSwitchConfig) -> None:
        self.config = config
        self._since_s: float | None = None

    def target(self, role: str, lux: float | None) -> str | None:
        if lux is None:
            return None
        if role == "day" and lux < self.config.night_below_lux:
            return "night"
        if role == "night" and lux > self.config.day_above_lux:
            return "day"
        return None

    def update(self, role: str, lux: float | None, now_s: float) -> str | None:
        target = self.target(role, lux)
        if target is None:
            self._since_s = None
            return None
        if self._since_s is None:
            self._since_s = now_s
        if now_s - self._since_s < self.config.hold_s:
            return None
        self._since_s = None
        return target


class SwitchingCameraStream:
    """A live frame source over a day and a night camera; only one is open at a time.

    Frame numbers continue across switches. Switching happens inside `get`, on
    the processing thread; `take_events` returns `camera_switch` records.
    """

    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        config: CameraSwitchConfig,
        open_stream: Callable[[CameraConfig], LatestFrameStream],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cameras = cameras
        self.hysteresis = LuxHysteresis(config)
        self.open_stream = open_stream
        self.clock = clock
        self.role = "day"
        self._stream: LatestFrameStream | None = None
        self._sequence_offset = 0
        self._previous_dropped_frames = 0
        self._events: list[dict[str, JsonValue]] = []
        self._retry_at_s = float("-inf")

    @property
    def camera(self) -> CameraConfig:
        return self.cameras[self.role]

    def start(self) -> None:
        self._stream = self._open(self.role)

    def _open(self, role: str) -> LatestFrameStream:
        stream = self.open_stream(self.cameras[role])
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception as error:
                print(f"camera cleanup error: {error}", file=sys.stderr)
            raise
        return stream

    def _close_active(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            self._sequence_offset += stream.captured_frames
            self._previous_dropped_frames += stream.dropped_frames
            stream.close()

    def get(self, timeout_s: float) -> FramePacket:
        if self._stream is None:
            raise CaptureError("camera stream has not been started")
        packet = self._stream.get(timeout_s)
        packet = replace(packet, sequence=packet.sequence + self._sequence_offset)
        now_s = self.clock()
        if now_s < self._retry_at_s:
            return packet
        target = self.hysteresis.update(self.role, packet.lux, now_s)
        if target is not None:
            self.switch(target, packet.lux)
        return packet

    def switch(self, target: str, lux: float | None) -> None:
        previous = self.role
        started_s = self.clock()
        reason = "dark" if target == "night" else "bright"
        self._close_active()
        error: str | None = None
        try:
            self._stream = self._open(target)
            self.role = target
        except Exception as failure:
            # keep running on the camera that worked; a second failure stops the run.
            error = str(failure)
            self._retry_at_s = self.clock() + SWITCH_RETRY_S
            print(f"camera switch to {self.cameras[target].label} failed: {error}", file=sys.stderr)
            self._stream = self._open(previous)
        self._events.append(
            {
                "from": self.cameras[previous].label,
                "to": self.cameras[target].label,
                "lux": lux,
                "reason": reason,
                "active": self.camera.label,
                "switch_ms": (self.clock() - started_s) * MILLISECONDS_PER_SECOND,
                "error": error,
            }
        )

    def take_events(self) -> list[dict[str, JsonValue]]:
        events, self._events = self._events, []
        return events

    @property
    def dropped_frames(self) -> int:
        active = self._stream.dropped_frames if self._stream is not None else 0
        return self._previous_dropped_frames + active

    @property
    def capture_fps(self) -> float:
        return self._stream.capture_fps if self._stream is not None else 0.0

    def close(self) -> None:
        self._close_active()
