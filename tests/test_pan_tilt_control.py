from __future__ import annotations

import threading
import time
from dataclasses import replace
from unittest.mock import patch

import pytest

from app.config import PanTiltConfig
from app.follow_models import TargetSnapshot, TrackSnapshot
from app.json_data import JsonValue
from app.models import Detection
from app.pan_tilt_control import (
    ControlObservation,
    FollowSession,
    ObservationTiming,
    ProportionalController,
)
from app.pantilt import AxisCalibration, Calibration, PanTiltDriver

PERIOD_S = 0.05
RETENTION_S = 0.75
FRAME_WIDTH_PX = 100
FRAME_HEIGHT_PX = 100
TEST_TIMEOUT_S = 2.0
SHORT_SHUTDOWN_S = 0.03


def target(
    observed_at_s: float = 0.0,
    horizontal_fraction: float = 0.8,
    vertical_fraction: float = 0.5,
    track_id: int = 7,
    matched: bool = True,
    confirmed: bool = True,
    label: str = "pigeon",
) -> TargetSnapshot:
    half_size_px = 2.0
    centre_x_px = FRAME_WIDTH_PX * horizontal_fraction
    centre_y_px = FRAME_HEIGHT_PX * vertical_fraction
    detection = Detection(
        label,
        0.9,
        centre_x_px - half_size_px,
        centre_y_px - half_size_px,
        centre_x_px + half_size_px,
        centre_y_px + half_size_px,
        track_id=track_id,
    )
    track = TrackSnapshot(track_id, detection, confirmed, matched, observed_at_s, 16.0)
    return TargetSnapshot(track, "tracking" if matched else "waiting", observed_at_s)


def observe(
    controller: ProportionalController,
    selected: TargetSnapshot,
    now_s: float = 0.0,
    timing_source: ObservationTiming = "sensor",
) -> None:
    controller.observe(
        ControlObservation(selected, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, now_s, timing_source)
    )


def controller(config: PanTiltConfig | None = None) -> ProportionalController:
    return ProportionalController(config or PanTiltConfig(), Calibration(), RETENTION_S)


class RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float, float]] = []
        self.moved = threading.Event()
        self.held = threading.Event()
        self.closed = threading.Event()
        self.release = threading.Event()
        self.block_moves = False
        self.block_holds = False
        self.block_close = False
        self.move_failure: Exception | None = None
        self.close_failure: Exception | None = None

    def move(self, pan_deg: float, tilt_deg: float) -> None:
        self.calls.append(("move", threading.get_ident(), pan_deg, tilt_deg))
        self.moved.set()
        if self.block_moves:
            assert self.release.wait(TEST_TIMEOUT_S)
        if self.move_failure is not None:
            raise self.move_failure

    def hold(self) -> None:
        self.calls.append(("hold", threading.get_ident(), 0.0, 0.0))
        self.held.set()
        if self.block_holds:
            assert self.release.wait(TEST_TIMEOUT_S)

    def close(self) -> None:
        self.calls.append(("close", threading.get_ident(), 0.0, 0.0))
        self.closed.set()
        if self.block_close:
            assert self.release.wait(TEST_TIMEOUT_S)
        if self.close_failure is not None:
            raise self.close_failure


class RecordingEvents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, JsonValue]]] = []

    def event(self, event: str, **fields: JsonValue) -> None:
        self.calls.append((event, fields))


def factory(driver: RecordingDriver) -> FollowSession:
    def create_driver(calibration: Calibration) -> PanTiltDriver:
        return driver

    return FollowSession(PanTiltConfig(), RETENTION_S, driver_factory=create_driver)


def wait_until_finished(session: FollowSession) -> None:
    thread = session._thread
    assert thread is not None
    thread.join(TEST_TIMEOUT_S)
    assert not thread.is_alive()


@pytest.mark.parametrize("horizontal_fraction", [0.45, 0.5, 0.55])
def test_deadband_is_five_percent_of_full_image(horizontal_fraction: float) -> None:
    control = controller()
    observe(control, target(horizontal_fraction=horizontal_fraction, vertical_fraction=0.55))
    status = control.step(PERIOD_S, PERIOD_S)
    assert status.state == "tracking"
    assert status.commanded_pan_deg == 90.0
    assert status.commanded_tilt_deg == 90.0


def test_proportional_gain_uses_normalised_error_and_calibrated_direction() -> None:
    calibration = replace(Calibration(), tilt=AxisCalibration(1, direction=-1))
    control = ProportionalController(
        replace(PanTiltConfig(), max_acceleration_deg_s2=10000), calibration, RETENTION_S
    )
    observe(control, target(horizontal_fraction=0.6, vertical_fraction=0.4))
    status = control.step(PERIOD_S, PERIOD_S)
    assert status.error_x_fraction == pytest.approx(0.1)
    assert status.error_y_fraction == pytest.approx(-0.1)
    assert status.commanded_pan_deg == pytest.approx(90.6)
    assert status.commanded_tilt_deg == pytest.approx(90.6)


def test_acceleration_speed_travel_limits_and_stall_cap() -> None:
    control = controller()
    previous_pan_deg = 90.0
    previous_speed_deg_s = 0.0
    for tick in range(1, 80):
        now_s = tick * PERIOD_S
        observe(control, target(now_s, vertical_fraction=0.1), now_s)
        status = control.step(now_s, PERIOD_S)
        speed_deg_s = (status.commanded_pan_deg - previous_pan_deg) / PERIOD_S
        assert 0 <= speed_deg_s <= 30.0 + 1e-10
        if not status.pan_limited:
            assert speed_deg_s - previous_speed_deg_s <= 60.0 * PERIOD_S + 1e-10
        assert 60 <= status.commanded_tilt_deg <= 120
        assert 60 <= status.commanded_pan_deg <= 120
        previous_pan_deg = status.commanded_pan_deg
        previous_speed_deg_s = speed_deg_s
    assert status.commanded_pan_deg == 120
    assert status.commanded_tilt_deg == 60
    assert status.pan_limited and status.tilt_limited
    stalled = controller()
    observe(stalled, target())
    assert stalled.step(PERIOD_S, 100.0).commanded_pan_deg == pytest.approx(90.15)


def test_direction_reversal_is_acceleration_limited() -> None:
    control = controller()
    for tick in range(1, 5):
        now_s = tick * PERIOD_S
        observe(control, target(now_s), now_s)
        status = control.step(now_s, PERIOD_S)
    previous_pan_deg = status.commanded_pan_deg
    previous_speed_deg_s = 12.0
    for tick in range(5, 15):
        now_s = tick * PERIOD_S
        observe(control, target(now_s, horizontal_fraction=0.2), now_s)
        status = control.step(now_s, PERIOD_S)
        speed_deg_s = (status.commanded_pan_deg - previous_pan_deg) / PERIOD_S
        assert abs(speed_deg_s - previous_speed_deg_s) <= 60 * PERIOD_S + 1e-10
        previous_pan_deg, previous_speed_deg_s = status.commanded_pan_deg, speed_deg_s
    assert previous_speed_deg_s < 0


def test_entering_deadband_immediately_holds_instead_of_coasting() -> None:
    control = controller()
    observe(control, target())
    moved = control.step(PERIOD_S, PERIOD_S)
    observe(control, target(PERIOD_S, horizontal_fraction=0.5), PERIOD_S)
    held = control.step(2 * PERIOD_S, PERIOD_S)
    assert held.commanded_pan_deg == moved.commanded_pan_deg


def test_freshness_includes_sensor_age_and_expires_at_250_ms() -> None:
    control = controller()
    observe(control, target(observed_at_s=10.0), 10.2)
    fresh = control.step(10.2, PERIOD_S)
    assert fresh.commanded_pan_deg > 90
    assert fresh.observation_age_ms == pytest.approx(200)
    stale = control.step(10.25, PERIOD_S)
    assert stale.commanded_pan_deg == fresh.commanded_pan_deg
    assert stale.state == "stale"
    assert stale.target_track_id == 7


def test_late_inference_never_acquires_or_renews_identity() -> None:
    control = controller()
    observe(control, target(), now_s=0.3)
    status = control.step(0.3, PERIOD_S)
    assert status.commanded_pan_deg == 90
    assert status.state == "stale"
    assert status.target_track_id is None


@pytest.mark.parametrize(
    "uncertain_target",
    [
        target(0.1, matched=False),
        target(0.1, confirmed=False),
        target(0.1, label="person"),
        target(observed_at_s=10.0),
        target(observed_at_s=float("nan")),
    ],
    ids=["unmatched", "unconfirmed", "wrong_class", "future", "nan_timestamp"],
)
def test_uncertain_observations_hold_without_renewing_identity(
    uncertain_target: TargetSnapshot,
) -> None:
    control = controller()
    observe(control, target())
    moving = control.step(PERIOD_S, PERIOD_S)
    observe(control, uncertain_target, 0.1)
    held = control.step(0.1, PERIOD_S)
    assert held.commanded_pan_deg == moving.commanded_pan_deg
    assert held.target_track_id == 7
    expired = control.step(0.8, PERIOD_S)
    assert expired.target_track_id is None
    assert expired.state in ("returning", "home")


def test_identity_retention_and_slow_automatic_home_are_separate_from_freshness() -> None:
    control = controller()
    for tick in range(1, 10):
        now_s = tick * PERIOD_S
        observe(control, target(now_s), now_s)
        moving = control.step(now_s, PERIOD_S)
    last_seen_s = 9 * PERIOD_S
    observe(control, target(last_seen_s, matched=False), last_seen_s + PERIOD_S)
    held = control.step(last_seen_s + RETENTION_S, PERIOD_S)
    assert held.target_track_id == 7
    assert held.commanded_pan_deg == moving.commanded_pan_deg
    homing = control.step(last_seen_s + RETENTION_S + PERIOD_S, PERIOD_S)
    assert homing.target_track_id is None
    assert homing.state == "returning"
    assert 0 < held.commanded_pan_deg - homing.commanded_pan_deg <= 10.0 * PERIOD_S


def test_unavailable_timing_holds_even_after_retention_until_known_loss_or_home() -> None:
    control = controller()
    observe(control, target())
    moved = control.step(PERIOD_S, PERIOD_S)
    observe(control, target(2.0), 2.0, "unavailable")
    uncertain = control.step(2.0, PERIOD_S)
    assert uncertain.commanded_pan_deg == moved.commanded_pan_deg
    assert uncertain.state == "uncertain"
    assert uncertain.observation_age_ms is None
    assert uncertain.target_track_id is None
    observe(control, TargetSnapshot(state="lost"), 2.1, "unavailable")
    known_lost = control.step(2.1, PERIOD_S)
    assert known_lost.commanded_pan_deg < uncertain.commanded_pan_deg


def test_initial_unavailable_timing_never_commands_a_target() -> None:
    control = controller()
    observe(control, target(), timing_source="unavailable")
    status = control.step(PERIOD_S, PERIOD_S)
    assert status.state == "uncertain"
    assert status.commanded_pan_deg == 90
    assert status.target_track_id is None


def test_explicit_home_is_bounded_and_stays_paused_until_resume() -> None:
    control = controller()
    for tick in range(1, 15):
        now_s = tick * PERIOD_S
        observe(control, target(now_s), now_s)
        status = control.step(now_s, PERIOD_S)
    control.request("home")
    previous_pan_deg = status.commanded_pan_deg
    for tick in range(15, 70):
        now_s = tick * PERIOD_S
        observe(control, target(now_s), now_s)
        status = control.step(now_s, PERIOD_S)
        assert 0 <= previous_pan_deg - status.commanded_pan_deg <= 10.0 * PERIOD_S + 1e-10
        previous_pan_deg = status.commanded_pan_deg
    assert status.state == "paused"
    assert status.commanded_pan_deg == 90
    control.request("resume")
    resumed = control.step(69 * PERIOD_S, PERIOD_S)
    assert resumed.state == "tracking"
    assert resumed.commanded_pan_deg > 90
    control.request("pause")
    paused = control.step(70 * PERIOD_S, PERIOD_S)
    assert paused.state == "paused"
    assert paused.commanded_pan_deg == resumed.commanded_pan_deg


def test_replay_ignores_hardware_calibration_factory_and_wall_clock() -> None:
    def forbidden_factory(calibration: Calibration) -> PanTiltDriver:
        pytest.fail("replay constructed a driver")

    def forbidden_clock() -> float:
        pytest.fail("replay used wall time")

    with (
        patch("app.pan_tilt_control.PCA9685", side_effect=AssertionError("physical driver")),
        patch.object(Calibration, "load", side_effect=AssertionError("calibration file")),
    ):
        session = FollowSession(
            PanTiltConfig(mode="hardware", calibration="does-not-exist.yaml"),
            RETENTION_S,
            replay=True,
            driver_factory=forbidden_factory,
            clock=forbidden_clock,
        )
        session.start()
        first = session.submit(target(), 100, 100, 0.0, "replay")
        second = session.submit(target(0.1), 100, 100, 0.1, "replay")
        session.close()
    assert first.mode == second.mode == "simulate"
    assert first.commanded_pan_deg == 90
    assert second.commanded_pan_deg == pytest.approx(90.45)
    assert session._thread is None


def test_replay_uses_fixed_virtual_ticks_independent_of_frame_frequency() -> None:
    session = FollowSession(PanTiltConfig(), RETENTION_S, replay=True)
    session.start()
    session.submit(target(), 100, 100, 0.0, "replay")
    status = session.submit(target(0.04), 100, 100, 0.04, "replay")
    assert status.commanded_pan_deg == 90
    status = session.submit(target(0.08), 100, 100, 0.08, "replay")
    assert status.commanded_pan_deg == pytest.approx(90.15)
    status = session.submit(target(0.12), 100, 100, 0.12, "replay")
    assert status.commanded_pan_deg == pytest.approx(90.45)
    with pytest.raises(ValueError, match="non-decreasing"):
        session.submit(target(), 100, 100, 0.1, "replay")
    session.close()


def test_replay_gap_expires_observation_before_next_frame() -> None:
    session = FollowSession(PanTiltConfig(), RETENTION_S, replay=True)
    session.start()
    session.submit(target(), 100, 100, 0.0, "replay")
    status = session.submit(target(0.5), 100, 100, 0.5, "replay")
    # only ticks before the 250 ms freshness boundary can use the previous observation.
    assert status.commanded_pan_deg == pytest.approx(91.5)
    assert status.state == "tracking"
    session.close()


def test_replay_controls_and_stop_use_the_same_public_interface() -> None:
    session = FollowSession(PanTiltConfig(), RETENTION_S, replay=True)
    session.start()
    session.submit(target(), 100, 100, 0.0, "replay")
    session.request("pause")
    paused = session.submit(target(0.1), 100, 100, 0.1, "replay")
    assert paused.state == "paused"
    assert paused.commanded_pan_deg == 90
    session.request("resume")
    moving = session.submit(target(0.2), 100, 100, 0.2, "replay")
    assert moving.commanded_pan_deg > 90
    session.request_stop()
    stopped = session.submit(target(0.3), 100, 100, 0.3, "replay")
    assert stopped == moving
    session.close()


def test_shared_cancellation_prevents_factory_and_first_move() -> None:
    stopped = threading.Event()
    calls: list[str] = []

    def forbidden_factory(calibration: Calibration) -> PanTiltDriver:
        calls.append("factory")
        raise AssertionError("cancelled session constructed driver")

    session = FollowSession(
        PanTiltConfig(mode="hardware"), RETENTION_S, stop=stopped, driver_factory=forbidden_factory
    )
    stopped.set()
    with patch.object(Calibration, "load") as load:
        session.start()
        session.close()
    assert calls == []
    load.assert_not_called()


def test_stop_before_factory_after_calibration_resolution() -> None:
    stopped = threading.Event()
    driver = RecordingDriver()
    calibration_entered = threading.Event()
    release_calibration = threading.Event()
    factory_calls: list[int] = []

    def load_calibration(path: str) -> Calibration:
        calibration_entered.set()
        assert release_calibration.wait(TEST_TIMEOUT_S)
        return Calibration()

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        factory_calls.append(threading.get_ident())
        return driver

    session = FollowSession(
        PanTiltConfig(mode="hardware"), RETENTION_S, stop=stopped, driver_factory=create_driver
    )
    with patch.object(Calibration, "load", side_effect=load_calibration):
        session.start()
        assert calibration_entered.wait(TEST_TIMEOUT_S)
        stopped.set()
        release_calibration.set()
        session.close()
    assert factory_calls == []
    assert driver.calls == []


def test_cancellation_during_factory_closes_on_owner_without_first_move() -> None:
    driver = RecordingDriver()
    entered = threading.Event()
    release = threading.Event()
    owner_ids: list[int] = []

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        owner_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(TEST_TIMEOUT_S)
        return driver

    session = FollowSession(PanTiltConfig(), RETENTION_S, driver_factory=create_driver)
    session.start()
    assert entered.wait(TEST_TIMEOUT_S)
    session.request_stop()
    release.set()
    session.close()
    assert [call[0] for call in driver.calls] == ["close"]
    assert [call[1] for call in driver.calls] == owner_ids
    assert owner_ids[0] != threading.get_ident()


@pytest.mark.parametrize("blocked_operation", ["factory", "move", "hold", "close"])
def test_blocked_driver_never_blocks_mailbox_and_never_creates_second_owner(
    blocked_operation: str,
) -> None:
    driver = RecordingDriver()
    factory_entered = threading.Event()
    owner_ids: list[int] = []
    driver.block_moves = blocked_operation == "move"
    driver.block_holds = blocked_operation == "hold"
    driver.block_close = blocked_operation == "close"

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        owner_ids.append(threading.get_ident())
        factory_entered.set()
        if blocked_operation == "factory":
            assert driver.release.wait(TEST_TIMEOUT_S)
        return driver

    session = FollowSession(PanTiltConfig(), RETENTION_S, driver_factory=create_driver)
    session.start()
    entered = {
        "factory": factory_entered,
        "move": driver.moved,
        "hold": driver.held,
        "close": driver.closed,
    }[blocked_operation]
    if blocked_operation == "close":
        assert driver.moved.wait(TEST_TIMEOUT_S)
        session.request_stop()
    assert entered.wait(TEST_TIMEOUT_S)
    submitted = threading.Event()

    def interact() -> None:
        session.submit(target(time.monotonic()), 100, 100, time.monotonic(), "delivery")
        session.request("home")
        session.snapshot()
        session.check()
        submitted.set()

    interaction = threading.Thread(target=interact, daemon=True)
    interaction.start()
    try:
        assert submitted.wait(SHORT_SHUTDOWN_S * 10)
        with patch("app.pan_tilt_control.SHUTDOWN_TIMEOUT_S", SHORT_SHUTDOWN_S):
            with pytest.raises(RuntimeError, match="did not stop"):
                session.close()
        if blocked_operation != "close":
            assert not driver.closed.is_set()
    finally:
        driver.release.set()
        interaction.join(TEST_TIMEOUT_S)
        session.close()
    assert owner_ids == [driver.calls[0][1]]
    assert {call[1] for call in driver.calls} == set(owner_ids)
    assert [call[0] for call in driver.calls].count("close") == 1


def test_live_worker_uses_latest_observation_on_independent_ticks() -> None:
    driver = RecordingDriver()
    session = factory(driver)
    session.start()
    try:
        assert driver.moved.wait(TEST_TIMEOUT_S)
        now_s = time.monotonic()
        session.submit(target(now_s, horizontal_fraction=0.1), 100, 100, now_s, "sensor")
        session.submit(target(now_s, horizontal_fraction=0.9), 100, 100, now_s, "delivery")
        deadline_s = time.monotonic() + TEST_TIMEOUT_S
        while session.snapshot().commanded_pan_deg == 90 and time.monotonic() < deadline_s:
            time.sleep(PERIOD_S)
        status = session.snapshot()
        assert status.commanded_pan_deg > 90
        assert status.timing_source == "delivery"
    finally:
        session.close()


def test_live_worker_runs_at_twenty_hz_without_observations() -> None:
    timestamps_s: list[float] = []
    enough_ticks = threading.Event()

    class TimedDriver(RecordingDriver):
        def hold(self) -> None:
            timestamps_s.append(time.monotonic())
            super().hold()
            if len(timestamps_s) >= 4:
                enough_ticks.set()

    driver = TimedDriver()
    session = factory(driver)
    session.start()
    try:
        assert enough_ticks.wait(TEST_TIMEOUT_S)
    finally:
        session.close()
    intervals_s = [
        later - earlier for earlier, later in zip(timestamps_s, timestamps_s[1:], strict=False)
    ]
    assert all(PERIOD_S * 0.8 <= interval_s < PERIOD_S * 3 for interval_s in intervals_s)


def test_check_acknowledges_worker_failure_without_repeating_it_on_close() -> None:
    driver = RecordingDriver()
    failure = OSError("first write failed")
    driver.move_failure = failure
    session = factory(driver)
    session.start()
    wait_until_finished(session)
    with pytest.raises(RuntimeError, match="first write failed") as caught:
        session.check()
    assert caught.value.__cause__ is failure
    assert session.snapshot().fault == "first write failed"
    assert session.snapshot().state == "fault"
    session.close()
    session.close()
    assert [call[0] for call in driver.calls].count("close") == 1


def test_close_reports_unacknowledged_worker_failure_once() -> None:
    driver = RecordingDriver()
    driver.move_failure = OSError("write failed")
    session = factory(driver)
    session.start()
    wait_until_finished(session)
    with pytest.raises(RuntimeError, match="write failed"):
        session.close()
    session.close()


def test_distinct_close_error_survives_acknowledged_worker_error() -> None:
    driver = RecordingDriver()
    driver.move_failure = OSError("first write failed")
    driver.close_failure = OSError("separate close failed")
    session = factory(driver)
    session.start()
    wait_until_finished(session)
    with pytest.raises(RuntimeError, match="first write failed"):
        session.check()
    with pytest.raises(RuntimeError, match="separate close failed"):
        session.close()
    assert session.snapshot().fault == "first write failed"
    assert session.snapshot().close_fault == "separate close failed"
    session.close()


def test_close_failure_alone_is_reported() -> None:
    driver = RecordingDriver()
    driver.close_failure = OSError("close failed")
    session = factory(driver)
    session.start()
    assert driver.moved.wait(TEST_TIMEOUT_S)
    with pytest.raises(RuntimeError, match="close failed"):
        session.close()
    assert session.snapshot().fault == "close failed"
    session.close()


def test_hardware_mode_passes_loaded_calibration_to_driver_and_records_it() -> None:
    calibration = Calibration(pan=AxisCalibration(3, direction=-1), tilt=AxisCalibration(4))
    events = RecordingEvents()
    driver = RecordingDriver()
    resolved_calibrations: list[Calibration] = []

    def create_driver(resolved: Calibration) -> PanTiltDriver:
        resolved_calibrations.append(resolved)
        return driver

    session = FollowSession(
        PanTiltConfig(mode="hardware"), RETENTION_S, events=events, driver_factory=create_driver
    )
    with patch.object(Calibration, "load", return_value=calibration):
        session.start()
        assert driver.moved.wait(TEST_TIMEOUT_S)
        session.close()
    assert resolved_calibrations == [calibration]
    startup = next(fields for name, fields in events.calls if name == "controller_start")
    assert startup["calibration"] == calibration.as_dict()


def test_repeated_start_and_close_never_construct_another_driver() -> None:
    driver = RecordingDriver()
    constructed: list[Calibration] = []

    def create_driver(calibration: Calibration) -> PanTiltDriver:
        constructed.append(calibration)
        return driver

    session = FollowSession(PanTiltConfig(), RETENTION_S, driver_factory=create_driver)
    session.start()
    session.start()
    assert driver.moved.wait(TEST_TIMEOUT_S)
    session.close()
    session.start()
    session.close()
    assert len(constructed) == 1
