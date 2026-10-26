"""Bounded proportional control and a single hardware-owning worker."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from .config import PanTiltConfig
from .follow_models import TargetSnapshot
from .json_data import JsonValue
from .pantilt import PCA9685, AxisCalibration, Calibration, PanTiltDriver, SimulatedMount

ObservationTiming = Literal["sensor", "delivery", "replay", "unavailable"]
ControlAction = Literal["pause", "resume", "home"]
ControlState = Literal[
    "acquiring", "tracking", "waiting", "stale", "uncertain", "paused", "returning", "home", "fault"
]
ControlMode = Literal["simulate", "hardware"]
SECONDS_TO_MILLISECONDS = 1000.0
NORMALISED_ERROR_SCALE = 2.0
IMAGE_CENTRE_FRACTION = 0.5
SHUTDOWN_TIMEOUT_S = 3.0
STOP_POLL_INTERVAL_S = 0.05


class ControlEventSink(Protocol):
    def event(self, event: str, **fields: JsonValue) -> None: ...


@dataclass(frozen=True, slots=True)
class ControllerStatus:
    state: ControlState = "acquiring"
    commanded_pan_deg: float = 90.0
    commanded_tilt_deg: float = 90.0
    target_track_id: int | None = None
    pan_limited: bool = False
    tilt_limited: bool = False
    deadband: float = 0.05
    observation_age_ms: float | None = None
    timing_source: ObservationTiming = "unavailable"
    fault: str | None = None
    mode: ControlMode = "simulate"
    error_x_fraction: float | None = None
    error_y_fraction: float | None = None
    close_fault: str | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "state": self.state,
            "commanded_pan_deg": self.commanded_pan_deg,
            "commanded_tilt_deg": self.commanded_tilt_deg,
            "target_track_id": self.target_track_id,
            "pan_limited": self.pan_limited,
            "tilt_limited": self.tilt_limited,
            "deadband": self.deadband,
            "observation_age_ms": self.observation_age_ms,
            "timing_source": self.timing_source,
            "fault": self.fault,
            "mode": self.mode,
            "error_x_fraction": self.error_x_fraction,
            "error_y_fraction": self.error_y_fraction,
            "close_fault": self.close_fault,
        }


@dataclass(frozen=True, slots=True)
class ControlObservation:
    target: TargetSnapshot
    width_px: int
    height_px: int
    submitted_at_s: float
    timing_source: ObservationTiming


@dataclass(frozen=True, slots=True)
class _AxisMotion:
    angle_deg: float
    speed_deg_s: float = 0.0
    limited: bool = False


def _approach(current: float, desired: float, maximum_change: float) -> float:
    return current + max(-maximum_change, min(maximum_change, desired - current))


def _advance_axis(
    previous: _AxisMotion,
    desired_speed_deg_s: float,
    axis: AxisCalibration,
    elapsed_s: float,
    acceleration_deg_s2: float,
    destination_deg: float | None = None,
) -> _AxisMotion:
    speed_deg_s = _approach(
        previous.speed_deg_s, desired_speed_deg_s, acceleration_deg_s2 * elapsed_s
    )
    requested_deg = previous.angle_deg + speed_deg_s * elapsed_s
    if destination_deg is not None:
        if (destination_deg - previous.angle_deg) * (destination_deg - requested_deg) <= 0:
            requested_deg = destination_deg
            speed_deg_s = 0.0
    angle_deg = axis.clamp(requested_deg)
    limited = angle_deg != requested_deg or (
        angle_deg in (axis.min_angle, axis.max_angle)
        and (angle_deg - axis.home_angle) * desired_speed_deg_s > 0
    )
    return _AxisMotion(angle_deg, 0.0 if limited else speed_deg_s, limited)


class ProportionalController:
    """No I/O or wall clock: callers supply observations and elapsed source time."""

    def __init__(self, config: PanTiltConfig, calibration: Calibration, retention_s: float) -> None:
        self.config = config
        self.calibration = calibration
        self.retention_s = retention_s
        self._pan = _AxisMotion(calibration.pan.home_angle)
        self._tilt = _AxisMotion(calibration.tilt.home_angle)
        self._observation: ControlObservation | None = None
        self._last_seen_s: float | None = None
        self._track_id: int | None = None
        self._paused = False
        self._homing = False
        self._status = ControllerStatus(
            commanded_pan_deg=self._pan.angle_deg,
            commanded_tilt_deg=self._tilt.angle_deg,
            deadband=config.deadband,
            mode=config.mode,
        )

    def observe(self, observation: ControlObservation) -> None:
        self._observation = observation

    def request(self, action: ControlAction) -> None:
        self._paused = action != "resume"
        self._homing = action == "home"
        self._stop_motion()

    def snapshot(self) -> ControllerStatus:
        return self._status

    def step(self, now_s: float, elapsed_s: float) -> ControllerStatus:
        # a stalled scheduler must not turn missed ticks into one large servo jump.
        elapsed_s = max(0.0, min(elapsed_s, 1.0 / self.config.control_hz))
        observation = self._observation
        age_s = self._age_s(observation, now_s)
        error = self._error(observation)
        eligible = (
            observation is not None
            and observation.target.state == "tracking"
            and error is not None
            and age_s is not None
            and 0 <= age_s < self.config.observation_timeout_s
        )
        if eligible and observation is not None and observation.target.track is not None:
            self._track_id = observation.target.track.track_id
            self._last_seen_s = observation.target.track.observed_at_s
        expired = self._last_seen_s is not None and now_s - self._last_seen_s > self.retention_s
        if expired:
            self._track_id = None
        state = self._advance(eligible, error, age_s, elapsed_s, expired)
        self._status = ControllerStatus(
            state=state,
            commanded_pan_deg=self._pan.angle_deg,
            commanded_tilt_deg=self._tilt.angle_deg,
            target_track_id=self._track_id,
            pan_limited=self._pan.limited,
            tilt_limited=self._tilt.limited,
            deadband=self.config.deadband,
            observation_age_ms=age_s * SECONDS_TO_MILLISECONDS if age_s is not None else None,
            timing_source=observation.timing_source if observation is not None else "unavailable",
            mode=self.config.mode,
            error_x_fraction=error[0] if error is not None else None,
            error_y_fraction=error[1] if error is not None else None,
        )
        return self._status

    def _age_s(self, observation: ControlObservation | None, now_s: float) -> float | None:
        if observation is None or observation.timing_source == "unavailable":
            return None
        track = observation.target.track
        if track is None or not math.isfinite(track.observed_at_s):
            return None
        age_s = now_s - track.observed_at_s
        return age_s if math.isfinite(age_s) else None

    def _error(self, observation: ControlObservation | None) -> tuple[float, float] | None:
        if observation is None or observation.width_px <= 0 or observation.height_px <= 0:
            return None
        track = observation.target.track
        if track is None or not track.confirmed or not track.matched:
            return None
        detection = track.detection
        if detection.label != "pigeon" or detection.track_id != track.track_id:
            return None
        values = (
            detection.confidence,
            detection.left_px,
            detection.top_px,
            detection.right_px,
            detection.bottom_px,
        )
        if not all(math.isfinite(value) for value in values) or not 0 < detection.confidence <= 1:
            return None
        if not (
            0 <= detection.left_px < detection.right_px <= observation.width_px
            and 0 <= detection.top_px < detection.bottom_px <= observation.height_px
        ):
            return None
        return (
            (detection.left_px + detection.right_px)
            / (NORMALISED_ERROR_SCALE * observation.width_px)
            - IMAGE_CENTRE_FRACTION,
            (detection.top_px + detection.bottom_px)
            / (NORMALISED_ERROR_SCALE * observation.height_px)
            - IMAGE_CENTRE_FRACTION,
        )

    def _advance(
        self,
        eligible: bool,
        error: tuple[float, float] | None,
        age_s: float | None,
        elapsed_s: float,
        expired: bool,
    ) -> ControlState:
        unknown_timing = (
            self._observation is not None
            and self._observation.timing_source == "unavailable"
            and self._observation.target.state != "lost"
        )
        if self._homing or (expired and not self._paused and not unknown_timing):
            return self._return_home(elapsed_s)
        if self._paused:
            self._stop_motion()
            return "paused"
        if eligible and error is not None:
            self._pan = self._follow_axis(self._pan, error[0], self.calibration.pan, elapsed_s)
            self._tilt = self._follow_axis(self._tilt, error[1], self.calibration.tilt, elapsed_s)
            return "tracking"
        self._stop_motion()
        if age_s is not None and age_s >= self.config.observation_timeout_s:
            return "stale"
        if unknown_timing:
            return "uncertain"
        if self._observation is None or self._observation.target.state == "acquiring":
            return "acquiring"
        if self._observation.target.state == "waiting":
            return "waiting"
        return "uncertain"

    def _follow_axis(
        self, previous: _AxisMotion, error_fraction: float, axis: AxisCalibration, elapsed_s: float
    ) -> _AxisMotion:
        if abs(error_fraction) <= self.config.deadband + math.ulp(IMAGE_CENTRE_FRACTION):
            return _AxisMotion(previous.angle_deg)
        desired_speed_deg_s = (
            max(
                -self.config.max_speed_deg_s,
                min(
                    self.config.max_speed_deg_s,
                    self.config.gain_deg_s * NORMALISED_ERROR_SCALE * error_fraction,
                ),
            )
            * axis.direction
        )
        return _advance_axis(
            previous, desired_speed_deg_s, axis, elapsed_s, self.config.max_acceleration_deg_s2
        )

    def _return_home(self, elapsed_s: float) -> ControlState:
        self._pan = self._home_axis(self._pan, self.calibration.pan, elapsed_s)
        self._tilt = self._home_axis(self._tilt, self.calibration.tilt, elapsed_s)
        if (
            self._pan.angle_deg == self.calibration.pan.home_angle
            and self._tilt.angle_deg == self.calibration.tilt.home_angle
        ):
            self._homing = False
            return "home"
        return "returning"

    def _home_axis(
        self, previous: _AxisMotion, axis: AxisCalibration, elapsed_s: float
    ) -> _AxisMotion:
        distance_deg = axis.home_angle - previous.angle_deg
        maximum_speed_deg_s = min(self.config.home_speed_deg_s, self.config.max_speed_deg_s)
        braking_speed_deg_s = math.sqrt(
            2.0 * self.config.max_acceleration_deg_s2 * abs(distance_deg)
        )
        desired_speed_deg_s = math.copysign(
            min(maximum_speed_deg_s, braking_speed_deg_s), distance_deg
        )
        return _advance_axis(
            previous,
            desired_speed_deg_s,
            axis,
            elapsed_s,
            self.config.max_acceleration_deg_s2,
            axis.home_angle,
        )

    def _stop_motion(self) -> None:
        # uncertainty and the deadband require an immediate hold, even while accelerating.
        self._pan = _AxisMotion(self._pan.angle_deg)
        self._tilt = _AxisMotion(self._tilt.angle_deg)


class FollowSession:
    """A single-slot mailbox decouples capture and controls from driver latency."""

    def __init__(
        self,
        config: PanTiltConfig,
        retention_s: float,
        replay: bool = False,
        stop: threading.Event | None = None,
        events: ControlEventSink | None = None,
        driver_factory: Callable[[Calibration], PanTiltDriver] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = replace(config, mode="simulate") if replay else config
        self._retention_s = retention_s
        self._replay = replay
        self._shared_stop = stop
        self._stop = threading.Event()
        self._events = events
        self._driver_factory = driver_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._observation: ControlObservation | None = None
        self._action: ControlAction | None = None
        self._status = ControllerStatus(deadband=config.deadband, mode=self._config.mode)
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._worker_failure: Exception | None = None
        self._worker_failure_observed = False
        self._close_failure: Exception | None = None
        self._close_failure_observed = False
        self._controller: ProportionalController | None = None
        self._replay_cursor_s: float | None = None
        self._replay_now_s: float | None = None

    def _cancelled(self) -> bool:
        return self._stop.is_set() or (self._shared_stop is not None and self._shared_stop.is_set())

    def start(self) -> None:
        with self._lock:
            if self._started or self._closed or self._cancelled():
                return
            self._started = True
        if self._replay:
            self._controller = self._make_controller(Calibration())
            return
        thread = threading.Thread(target=self._run, name="pan-tilt", daemon=True)
        with self._lock:
            self._thread = thread
            thread.start()

    def _make_controller(self, calibration: Calibration) -> ProportionalController:
        controller = ProportionalController(self._config, calibration, self._retention_s)
        self._publish(controller.snapshot())
        self._emit(
            "controller_start",
            mode=self._config.mode,
            calibration=calibration.as_dict(),
            image_orientation="raw_unflipped",
            control_hz=self._config.control_hz,
            observation_timeout_s=self._config.observation_timeout_s,
            sensor_timing_available=None,
        )
        return controller

    def submit(
        self,
        target: TargetSnapshot,
        width_px: int,
        height_px: int,
        now_s: float,
        timing_source: ObservationTiming,
    ) -> ControllerStatus:
        observation = ControlObservation(target, width_px, height_px, now_s, timing_source)
        if self._replay:
            if not self._cancelled() and self._started and not self._closed:
                self._advance_replay(observation)
            return self.snapshot()
        with self._lock:
            self._observation = observation
            return self._status

    def snapshot(self) -> ControllerStatus:
        with self._lock:
            return self._status

    def request(self, action: ControlAction) -> None:
        if action not in ("pause", "resume", "home"):
            raise ValueError(f"unknown control action: {action}")
        with self._lock:
            self._action = action

    def request_stop(self) -> None:
        self._stop.set()

    def _take_mailbox(self) -> tuple[ControlObservation | None, ControlAction | None]:
        with self._lock:
            observation, action = self._observation, self._action
            self._action = None
        return observation, action

    def _run(self) -> None:
        driver: PanTiltDriver | None = None
        try:
            if self._cancelled():
                return
            calibration = (
                Calibration.load(self._config.calibration)
                if self._config.mode == "hardware"
                else Calibration()
            )
            controller = self._make_controller(calibration)
            if self._cancelled():
                return
            factory = self._driver_factory
            if factory is None:
                factory = PCA9685 if self._config.mode == "hardware" else create_simulated_mount
            driver = factory(calibration)
            if self._cancelled():
                return
            driver.move(calibration.pan.home_angle, calibration.tilt.home_angle)
            self._run_ticks(controller, driver)
        except Exception as exception:
            self._record_failure(exception, closing=False)
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception as exception:
                    self._record_failure(exception, closing=True)

    def _run_ticks(self, controller: ProportionalController, driver: PanTiltDriver) -> None:
        period_s = 1.0 / self._config.control_hz
        previous_s = self._clock()
        deadline_s = previous_s + period_s
        while not self._cancelled():
            if self._stop.wait(min(max(0.0, deadline_s - self._clock()), STOP_POLL_INTERVAL_S)):
                return
            now_s = self._clock()
            if now_s < deadline_s or self._cancelled():
                continue
            observation, action = self._take_mailbox()
            if observation is not None:
                controller.observe(observation)
            if action is not None:
                controller.request(action)
            status = controller.step(now_s, now_s - previous_s)
            previous_s = now_s
            deadline_s = max(deadline_s + period_s, now_s + period_s)
            if self._cancelled():
                return
            previous = self.snapshot()
            if (status.commanded_pan_deg, status.commanded_tilt_deg) != (
                previous.commanded_pan_deg,
                previous.commanded_tilt_deg,
            ):
                driver.move(status.commanded_pan_deg, status.commanded_tilt_deg)
            else:
                driver.hold()
            self._publish(status)

    def _advance_replay(self, observation: ControlObservation) -> None:
        controller = self._controller
        if controller is None:
            return
        now_s = observation.submitted_at_s
        if not math.isfinite(now_s) or (
            self._replay_now_s is not None and now_s < self._replay_now_s
        ):
            raise ValueError("replay observation time must be finite and non-decreasing")
        _, action = self._take_mailbox()
        if action is not None:
            controller.request(action)
        period_s = 1.0 / self._config.control_hz
        cursor_s = self._replay_cursor_s
        if cursor_s is None:
            cursor_s = now_s
        while cursor_s + period_s <= now_s + math.ulp(now_s):
            cursor_s += period_s
            self._publish(controller.step(cursor_s, period_s))
        self._replay_cursor_s = cursor_s
        self._replay_now_s = now_s
        controller.observe(observation)
        self._publish(controller.step(now_s, 0.0))

    def _publish(self, status: ControllerStatus) -> None:
        with self._lock:
            self._status = status
        self._emit(
            "controller_state",
            **status.as_dict(),
            sensor_timing_available=status.timing_source == "sensor",
        )

    def _emit(self, event: str, **fields: JsonValue) -> None:
        if self._events is not None:
            self._events.event(event, **fields)

    def _record_failure(self, exception: Exception, closing: bool) -> None:
        with self._lock:
            if closing:
                self._close_failure = self._close_failure or exception
            else:
                self._worker_failure = self._worker_failure or exception
            first_failure = self._worker_failure or self._close_failure
            self._status = replace(
                self._status,
                state="fault",
                fault=str(first_failure),
                close_fault=str(self._close_failure) if self._close_failure is not None else None,
            )

    def check(self) -> None:
        with self._lock:
            failure = self._worker_failure or self._close_failure
            if self._worker_failure is not None:
                self._worker_failure_observed = True
            elif self._close_failure is not None:
                self._close_failure_observed = True
        if failure is not None:
            raise RuntimeError(f"Pan-tilt worker failed: {failure}") from failure

    def close(self) -> None:
        self.request_stop()
        with self._lock:
            thread = self._thread
            self._closed = True
        if thread is not None:
            thread.join(timeout=SHUTDOWN_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError(f"Pan-tilt worker did not stop within {SHUTDOWN_TIMEOUT_S:g}s")
        with self._lock:
            failure = None if self._worker_failure_observed else self._worker_failure
            close_failure = None if self._close_failure_observed else self._close_failure
            if failure is not None:
                self._worker_failure_observed = True
            if close_failure is not None:
                self._close_failure_observed = True
        if failure is not None:
            if close_failure is not None:
                failure.add_note(f"driver close also failed: {close_failure}")
            raise RuntimeError(f"Pan-tilt worker failed: {failure}") from failure
        if close_failure is not None:
            raise RuntimeError(f"Pan-tilt driver close failed: {close_failure}") from close_failure


def create_simulated_mount(calibration: Calibration) -> PanTiltDriver:
    return SimulatedMount()
