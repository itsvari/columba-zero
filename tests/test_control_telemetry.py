from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

import pytest

from app.config import PanTiltConfig
from app.follow_models import TargetSnapshot, TrackSnapshot
from app.json_data import JsonValue
from app.models import Detection
from app.pan_tilt_control import ControlAction, ControlMode, FollowSession, ObservationTiming
from app.pantilt import Calibration, PanTiltDriver

START_S = 16.0
CONTROL_HZ = 16.0
PERIOD_S = 1.0 / CONTROL_HZ
DRIVER_DELAY_S = 0.125
SHORT_DRIVER_DELAY_S = PERIOD_S / 2
RETENTION_S = 0.75
FRAME_SIZE_PX = 100
TRACK_ID = 7
MILLISECONDS_PER_SECOND = 1000.0
TELEMETRY_FIELDS = (
    "tick_started_at_monotonic_s",
    "driver_action",
    "driver_action_completed_at_monotonic_s",
    "tracking_observation_age_at_command_ms",
)


class DeterministicClock:
    def __init__(self, audit: list[str]) -> None:
        self.now_s = START_S
        self.audit = audit

    def __call__(self) -> float:
        self.audit.append("clock")
        return self.now_s

    def wait(self, timeout_s: float) -> bool:
        self.now_s += timeout_s
        return False


class DelayedDriver:
    def __init__(self, clock: DeterministicClock, delay_s: float = DRIVER_DELAY_S) -> None:
        self.clock = clock
        self.delay_s = delay_s
        self.actions: list[str] = []
        self.closed = False
        self.fail_tracking_move = False
        self.fail_close = False

    def move(self, pan_deg: float, tilt_deg: float) -> None:
        self.actions.append("move")
        if len(self.actions) == 1:
            return
        self.clock.now_s += self.delay_s
        if self.fail_tracking_move:
            raise OSError("tracking write failed")
        self.clock.audit.append("move returned")

    def hold(self) -> None:
        self.actions.append("hold")
        self.clock.now_s += self.delay_s
        self.clock.audit.append("hold returned")

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise OSError("close failed")


class RecordingEvents:
    def __init__(self, audit: list[str]) -> None:
        self.audit = audit
        self.records: list[tuple[str, dict[str, JsonValue]]] = []
        self.after_tick: Callable[[dict[str, JsonValue]], None] | None = None

    def event(self, event: str, **fields: JsonValue) -> None:
        self.records.append((event, fields))
        if event == "controller_state" and fields["tick_started_at_monotonic_s"] is not None:
            self.audit.append("tick published")
            if self.after_tick is not None:
                self.after_tick(fields)

    def states(self) -> list[dict[str, JsonValue]]:
        return [fields for event, fields in self.records if event == "controller_state"]


def tracking_target(observed_at_s: float, *, centred: bool = False) -> TargetSnapshot:
    centre_px = FRAME_SIZE_PX * (0.5 if centred else 0.8)
    half_size_px = 2.0
    detection = Detection(
        "pigeon",
        0.9,
        centre_px - half_size_px,
        centre_px - half_size_px,
        centre_px + half_size_px,
        centre_px + half_size_px,
        track_id=TRACK_ID,
    )
    track = TrackSnapshot(TRACK_ID, detection, True, True, observed_at_s, 16.0)
    return TargetSnapshot(track, "tracking", observed_at_s)


def run_fake_worker(session: FollowSession) -> None:
    clock = session._clock
    assert isinstance(clock, DeterministicClock)
    with (
        patch.object(Calibration, "load", return_value=Calibration()),
        patch.object(session._stop, "wait", side_effect=clock.wait),
    ):
        session._run()


@pytest.mark.parametrize("mode", ["hardware", "simulate"])
@pytest.mark.parametrize("timing_source", ["sensor", "delivery"])
def test_driver_return_extends_command_age_without_changing_status(
    mode: ControlMode, timing_source: ObservationTiming
) -> None:
    audit: list[str] = []
    clock = DeterministicClock(audit)
    driver = DelayedDriver(clock)
    events = RecordingEvents(audit)

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        return driver

    session = FollowSession(
        replace(PanTiltConfig(), mode=mode, control_hz=CONTROL_HZ),
        RETENTION_S,
        events=events,
        driver_factory=create_driver,
        clock=clock,
    )
    events.after_tick = lambda fields: session.request_stop()
    session.submit(tracking_target(START_S), FRAME_SIZE_PX, FRAME_SIZE_PX, START_S, timing_source)
    run_fake_worker(session)
    session.close()

    initial, moved = events.states()
    assert all(initial[field] is None for field in TELEMETRY_FIELDS)
    assert moved["state"] == "tracking"
    assert moved["mode"] == mode
    assert moved["driver_action"] == "move"
    assert moved["tick_started_at_monotonic_s"] == START_S + PERIOD_S
    assert moved["driver_action_completed_at_monotonic_s"] == START_S + PERIOD_S + DRIVER_DELAY_S
    assert moved["observation_age_ms"] == PERIOD_S * MILLISECONDS_PER_SECOND
    assert (
        moved["tracking_observation_age_at_command_ms"]
        == (PERIOD_S + DRIVER_DELAY_S) * MILLISECONDS_PER_SECOND
    )
    returned_index = audit.index("move returned")
    assert audit[returned_index:] == ["move returned", "clock", "tick published"]
    assert session.snapshot().as_dict() == {
        name: value for name, value in moved.items() if name in session.snapshot().as_dict()
    }
    assert not set(TELEMETRY_FIELDS).intersection(session.snapshot().as_dict())
    assert driver.actions == ["move", "move"]
    assert driver.closed


@pytest.mark.parametrize(
    ("transition", "expected_state", "expected_action"),
    [
        ("centred", "tracking", "hold"),
        ("no_target", "acquiring", "hold"),
        ("pause", "paused", "hold"),
        ("home", "returning", "move"),
        ("unknown", "uncertain", "hold"),
        ("future", "uncertain", "hold"),
        ("replay_timing", "tracking", "move"),
    ],
)
def test_only_tracking_position_moves_with_known_timing_have_command_age(
    transition: str, expected_state: str, expected_action: str
) -> None:
    audit: list[str] = []
    clock = DeterministicClock(audit)
    driver = DelayedDriver(clock, SHORT_DRIVER_DELAY_S)
    events = RecordingEvents(audit)

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        return driver

    session = FollowSession(
        replace(PanTiltConfig(), control_hz=CONTROL_HZ),
        RETENTION_S,
        events=events,
        driver_factory=create_driver,
        clock=clock,
    )
    ticks = 0

    def after_tick(fields: dict[str, JsonValue]) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            session.request_stop()
            return
        if ticks != 2:
            return
        timing_source: ObservationTiming = "sensor"
        selected = tracking_target(clock.now_s, centred=transition == "centred")
        if transition == "no_target":
            selected = TargetSnapshot()
        elif transition == "unknown":
            timing_source = "unavailable"
        elif transition == "future":
            selected = tracking_target(clock.now_s + RETENTION_S)
        elif transition == "replay_timing":
            timing_source = "replay"
        action: ControlAction | None = (
            "pause" if transition == "pause" else "home" if transition == "home" else None
        )
        if action is not None:
            session.request(action)
        session.submit(selected, FRAME_SIZE_PX, FRAME_SIZE_PX, clock.now_s, timing_source)

    events.after_tick = after_tick
    session.submit(tracking_target(START_S), FRAME_SIZE_PX, FRAME_SIZE_PX, START_S, "sensor")
    run_fake_worker(session)
    session.close()

    initial, first_move, second_move, transitioned = events.states()
    assert initial["driver_action"] is None
    assert first_move["tracking_observation_age_at_command_ms"] is not None
    assert second_move["tracking_observation_age_at_command_ms"] is not None
    assert transitioned["state"] == expected_state
    assert transitioned["driver_action"] == expected_action
    assert transitioned["tracking_observation_age_at_command_ms"] is None
    assert transitioned["driver_action_completed_at_monotonic_s"] == clock.now_s
    assert driver.actions == ["move", "move", "move", expected_action]
    assert driver.closed


def test_replay_has_no_driver_clock_or_live_measurements_even_with_sensor_observations() -> None:
    events = RecordingEvents([])

    def forbidden_factory(calibration: Calibration) -> PanTiltDriver:
        raise AssertionError("replay constructed a driver")

    def forbidden_clock() -> float:
        raise AssertionError("replay sampled real-time clock")

    session = FollowSession(
        PanTiltConfig(mode="hardware"),
        RETENTION_S,
        replay=True,
        events=events,
        driver_factory=forbidden_factory,
        clock=forbidden_clock,
        image_rotation_deg=180,
    )
    with patch.object(Calibration, "load", side_effect=AssertionError("loaded hardware profile")):
        session.start()
        session.submit(tracking_target(START_S), FRAME_SIZE_PX, FRAME_SIZE_PX, START_S, "sensor")
        status = session.submit(
            tracking_target(START_S + PERIOD_S),
            FRAME_SIZE_PX,
            FRAME_SIZE_PX,
            START_S + PERIOD_S,
            "sensor",
        )
        session.close()
    assert status.commanded_pan_deg > Calibration().pan.home_angle
    assert status.mode == "simulate"
    assert all(fields[field] is None for fields in events.states() for field in TELEMETRY_FIELDS)
    startup = next(fields for event, fields in events.records if event == "controller_start")
    assert startup["image_orientation"] == "as_stored"
    assert startup["image_rotation_deg"] == 0


def test_failed_tick_never_publishes_completion_and_preserves_first_error_and_cleanup() -> None:
    audit: list[str] = []
    clock = DeterministicClock(audit)
    driver = DelayedDriver(clock)
    driver.fail_tracking_move = True
    driver.fail_close = True
    events = RecordingEvents(audit)

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        return driver

    session = FollowSession(
        replace(PanTiltConfig(), control_hz=CONTROL_HZ),
        RETENTION_S,
        events=events,
        driver_factory=create_driver,
        clock=clock,
    )
    session.submit(tracking_target(START_S), FRAME_SIZE_PX, FRAME_SIZE_PX, START_S, "sensor")
    run_fake_worker(session)
    assert len(events.states()) == 1
    assert all(events.states()[0][field] is None for field in TELEMETRY_FIELDS)
    assert session.snapshot().fault == "tracking write failed"
    assert session.snapshot().close_fault == "close failed"
    assert driver.closed
    with pytest.raises(RuntimeError, match="tracking write failed"):
        session.check()
    with pytest.raises(RuntimeError, match="close failed"):
        session.close()
    session.close()
