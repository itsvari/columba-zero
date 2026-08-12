from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from .config import ActuatorConfig


class ActuatorError(RuntimeError):
    pass


class OutputPort(Protocol):
    def on(self) -> None: ...

    def off(self) -> None: ...

    def close(self) -> None: ...


class MemoryOutput:
    def __init__(self) -> None:
        self.value = False
        self.transitions: list[bool] = []

    def on(self) -> None:
        self.value = True
        self.transitions.append(True)

    def off(self) -> None:
        self.value = False
        self.transitions.append(False)

    def close(self) -> None:
        self.off()


class DeterrentActuator:
    """Owns both the output driver and normally-closed shutter output."""

    def __init__(
        self,
        config: ActuatorConfig,
        output: OutputPort,
        shutter: OutputPort,
        interlock_closed: Callable[[], bool] = lambda: True,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.output = output
        self.shutter = shutter
        self.interlock_closed = interlock_closed
        self.sleeper = sleeper
        self.all_off()

    def all_off(self) -> None:
        self.output.off()
        self.shutter.off()

    def pulse(self, duration_s: float | None = None) -> None:
        duration = self.config.pulse_s if duration_s is None else duration_s
        if duration <= 0 or duration > self.config.max_pulse_s:
            raise ActuatorError("pulse duration is outside the configured bound")
        if self.config.mode == "laser":
            if not self.config.bench_mode or not self.config.laser_enabled:
                raise ActuatorError(
                    "laser output is not explicitly enabled for enclosed bench mode"
                )
            if not self.interlock_closed():
                raise ActuatorError("hardware interlock is open")
        try:
            self.output.on()
            if self.config.mode == "laser":
                self.shutter.on()
                remaining = duration
                while remaining > 0:
                    if not self.interlock_closed():
                        raise ActuatorError("hardware interlock opened during pulse")
                    interval = min(0.01, remaining)
                    self.sleeper(interval)
                    remaining -= interval
            else:
                self.sleeper(duration)
        finally:
            self.shutter.off()
            self.output.off()

    def close(self) -> None:
        self.all_off()
        self.shutter.close()
        self.output.close()


def build_actuator(
    config: ActuatorConfig,
    hardware_outputs: bool,
    interlock_closed: Callable[[], bool],
) -> DeterrentActuator:
    if hardware_outputs:
        try:
            from gpiozero import DigitalOutputDevice
        except ImportError as exc:
            raise ActuatorError("gpiozero is required for hardware actuator outputs") from exc
        output: OutputPort = DigitalOutputDevice(
            config.output_pin, active_high=config.active_high, initial_value=False
        )
        shutter: OutputPort = DigitalOutputDevice(
            config.shutter_pin, active_high=config.active_high, initial_value=False
        )
    else:
        output = MemoryOutput()
        shutter = MemoryOutput()
    return DeterrentActuator(config, output, shutter, interlock_closed)
