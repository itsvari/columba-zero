from __future__ import annotations

import pytest
from conftest import make_config

from app.actuators import ActuatorError, DeterrentActuator, MemoryOutput


def test_pulse_is_bounded_and_always_turns_off() -> None:
    config = make_config()
    output = MemoryOutput()
    shutter = MemoryOutput()
    actuator = DeterrentActuator(config.actuator, output, shutter, sleeper=lambda _: None)
    actuator.pulse()
    assert output.transitions[-2:] == [True, False]
    assert True not in shutter.transitions
    assert not output.value
    assert not shutter.value
    with pytest.raises(ActuatorError, match="outside"):
        actuator.pulse(1.0)


def test_laser_refuses_open_interlock() -> None:
    config = make_config(actuator={"mode": "laser", "bench_mode": True, "laser_enabled": True})
    actuator = DeterrentActuator(
        config.actuator,
        MemoryOutput(),
        MemoryOutput(),
        interlock_closed=lambda: False,
        sleeper=lambda _: None,
    )
    with pytest.raises(ActuatorError, match="interlock"):
        actuator.pulse()


def test_laser_stops_if_interlock_opens_during_pulse() -> None:
    config = make_config(actuator={"mode": "laser", "bench_mode": True, "laser_enabled": True})
    checks = iter([True, True, False])
    output = MemoryOutput()
    shutter = MemoryOutput()
    actuator = DeterrentActuator(
        config.actuator,
        output,
        shutter,
        interlock_closed=lambda: next(checks),
        sleeper=lambda _: None,
    )
    with pytest.raises(ActuatorError, match="during pulse"):
        actuator.pulse()
    assert not output.value
    assert not shutter.value
