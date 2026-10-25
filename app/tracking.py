"""Target association and bounded camera motion, independent of vision/I2C libraries."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

from .config import TrackingConfig
from .models import Detection
from .pantilt import PCA9685, Calibration, SimulatedMount


class TrackingController:
    def __init__(self, config: TrackingConfig, calibration: Calibration) -> None:
        self.config, self.calibration = config, calibration
        self.pan = calibration.pan.home_angle
        self.tilt = calibration.tilt.home_angle
        self.target: Detection | None = None
        self.center: tuple[float, float] | None = None
        self.last_seen: float | None = None
        self.observed_at: float | None = None
        self.error: tuple[float, float] | None = None
        self.state = "home"

    def _expire(self, now: float) -> None:
        if self.last_seen is not None and now - self.last_seen >= self.config.loss_timeout_s:
            self.target = self.center = self.error = None
            self.last_seen = None
            self.state = "returning"

    def observe(self, detections: list[Detection], width: int, height: int,
                observed_at: float, now: float) -> None:
        self._expire(now)
        # A late inference must not renew a target or command from an old image.
        if now - observed_at >= self.config.observation_timeout_s:
            self.error = None
            if self.target is not None:
                self.state = "stale"
            return
        candidates = []
        for d in detections:
            values = (d.confidence, d.x1, d.y1, d.x2, d.y2)
            if (d.label != "pigeon" or not all(math.isfinite(v) for v in values)
                    or not 0 < d.confidence <= 1 or width <= 0 or height <= 0
                    or not 0 <= d.x1 < d.x2 <= width or not 0 <= d.y1 < d.y2 <= height):
                continue
            center = ((d.x1 + d.x2) / (2 * width), (d.y1 + d.y2) / (2 * height))
            candidates.append((d, center))
        selected = None
        if self.center is None:
            if candidates:
                selected = max(candidates, key=lambda item: (
                    (item[0].x2 - item[0].x1) * (item[0].y2 - item[0].y1),
                    item[0].confidence))
        else:
            matches = [(math.dist(center, self.center), -d.confidence, d, center)
                       for d, center in candidates]
            matches = [item for item in matches if item[0] <= self.config.match_distance]
            if matches:
                match = min(matches, key=lambda item: (round(item[0], 12), item[1]))
                selected = match[2], match[3]
        self.observed_at = observed_at
        self.error = None
        if selected is not None:
            self.target, self.center = selected
            self.last_seen = observed_at
            self.error = tuple(2 * (value - 0.5) for value in self.center)
            self.state = "tracking"
        elif self.target is not None:
            self.state = "waiting"

    def step(self, now: float, dt: float) -> tuple[float, float]:
        self._expire(now)
        # Do not make a large catch-up move after a scheduler stall.
        dt = max(0.0, min(dt, 0.1))
        fresh = (self.observed_at is not None
                 and now - self.observed_at < self.config.observation_timeout_s)
        if self.error is not None and fresh:
            values = []
            for angle, error, axis in ((self.pan, self.error[0], self.calibration.pan),
                                       (self.tilt, self.error[1], self.calibration.tilt)):
                speed = 0.0 if abs(error) <= 2 * self.config.deadband else (
                    max(-self.config.max_speed,
                        min(self.config.max_speed, self.config.gain * error)) * axis.direction)
                values.append(axis.clamp(angle + speed * dt))
            self.pan, self.tilt = values
        elif self.state in ("returning", "home"):
            values = []
            for angle, axis in ((self.pan, self.calibration.pan),
                                (self.tilt, self.calibration.tilt)):
                delta = axis.home_angle - angle
                step = self.config.max_speed * dt
                values.append(axis.clamp(angle + max(-step, min(step, delta))))
            self.pan, self.tilt = values
            self.state = ("home" if self.pan == self.calibration.pan.home_angle
                          and self.tilt == self.calibration.tilt.home_angle else "returning")
        elif not fresh:
            self.state = "stale"
        return self.pan, self.tilt

    def snapshot(self) -> dict:
        return {"state": self.state,
                "selected_target": self.target.as_dict() if self.target else None,
                "commanded_pan_deg": self.pan, "commanded_tilt_deg": self.tilt}


class TrackingSession:
    """One worker owns physical writes; replay simulation uses source frame time."""

    def __init__(self, config: TrackingConfig, mode: str, replay: bool = False) -> None:
        if mode not in ("hardware", "simulate") or (mode == "hardware" and replay):
            raise ValueError("tracking session requires hardware live mode or simulation")
        self.mode, self.replay = mode, replay
        calibration = (Calibration.load(config.calibration)
                       if mode == "hardware" or Path(config.calibration).exists()
                       else Calibration())
        self.controller = TrackingController(config, calibration)
        self.mount = PCA9685(calibration) if mode == "hardware" else SimulatedMount()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._last_replay_time: float | None = None
        self._closed = False

    def start(self) -> None:
        self.mount.move(self.controller.pan, self.controller.tilt)
        if not self.replay:
            self._thread = threading.Thread(target=self._run, name="pan-tilt", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        last = time.monotonic()
        try:
            while not self._stop.wait(0.05):
                now = time.monotonic()
                with self._lock:
                    angles = self.controller.step(now, now - last)
                self.mount.move(*angles)
                last = now
        except Exception as exc:
            self._error = exc
        finally:
            try:
                self.mount.close()
            except Exception as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._closed = True

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Pan-Tilt tracking failed: {self._error}") from self._error

    def submit(self, detections: list[Detection], width: int, height: int,
               observed_at: float, replay_time: float | None = None) -> dict:
        self.check()
        now = time.monotonic()
        with self._lock:
            if self.replay:
                now = replay_time if replay_time is not None else 0.0
                previous = self._last_replay_time
                if previous is not None:
                    # Advance the previous command at virtual 20 Hz before observing this frame.
                    cursor = previous
                    while cursor < now:
                        dt = min(0.05, now - cursor)
                        cursor += dt
                        self.controller.step(cursor, dt)
                self._last_replay_time = now
                observed_at = now
            self.controller.observe(detections, width, height, observed_at, now)
            snapshot = self.controller.snapshot()
        return {"mode": self.mode, **snapshot}

    def request_stop(self) -> None:
        """Signal the worker without doing I2C in the runtime's signal handler."""
        self._stop.set()

    def close(self) -> None:
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("Pan-Tilt worker did not stop within 3 seconds")
        if not self._closed:
            try:
                self.mount.close()
            finally:
                self._closed = True
        self.check()
