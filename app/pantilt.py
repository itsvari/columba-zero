"""Calibrated camera mount and PCA9685 I2C driver (no GPIO library required)."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config import ConfigError


@dataclass(frozen=True)
class AxisCalibration:
    channel: int
    pulse_min_us: float = 1000
    pulse_max_us: float = 2000
    min_angle: float = 60
    home_angle: float = 90
    max_angle: float = 120
    direction: int = 1

    def validate(self) -> None:
        if type(self.channel) is not int or not 0 <= self.channel <= 15:
            raise ConfigError("servo channel must be an integer in [0, 15]")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ConfigError("servo direction must be -1 or 1")
        for name in ("pulse_min_us", "pulse_max_us", "min_angle", "home_angle", "max_angle"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigError(f"servo {name} must be finite")
        if not 500 <= self.pulse_min_us < self.pulse_max_us <= 2500:
            raise ConfigError("servo pulse bounds must satisfy 500 <= min < max <= 2500 us")
        if not 0 <= self.min_angle < self.home_angle < self.max_angle <= 180:
            raise ConfigError("servo angles must satisfy 0 <= min < home < max <= 180")

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, angle))

    def pulse(self, angle: float) -> float:
        # Logical servo degrees map linearly across the configured pulse envelope.
        return self.pulse_min_us + self.clamp(angle) / 180 * (
            self.pulse_max_us - self.pulse_min_us)


@dataclass(frozen=True)
class Calibration:
    pan: AxisCalibration = AxisCalibration(channel=0)
    tilt: AxisCalibration = AxisCalibration(channel=1)
    bus: int = 1
    address: int = 0x40

    def validate(self) -> None:
        self.pan.validate()
        self.tilt.validate()
        if self.pan.channel == self.tilt.channel:
            raise ConfigError("pan and tilt must use different channels")
        if type(self.bus) is not int or self.bus < 0:
            raise ConfigError("I2C bus must be a non-negative integer")
        if type(self.address) is not int or not 0x08 <= self.address <= 0x77:
            raise ConfigError("I2C address must be a usable 7-bit integer")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not {"pan", "tilt"} <= raw.keys():
                raise ConfigError("calibration must contain pan and tilt mappings")
            # Require the measured fields; defaults are for simulation only.
            required = {"channel", "pulse_min_us", "pulse_max_us", "min_angle",
                        "home_angle", "max_angle", "direction"}
            for axis in ("pan", "tilt"):
                if not isinstance(raw[axis], dict) or not required <= raw[axis].keys():
                    raise ConfigError(f"calibration {axis} is incomplete")
            raw = dict(raw)
            raw["pan"] = AxisCalibration(**raw["pan"])
            raw["tilt"] = AxisCalibration(**raw["tilt"])
            result = cls(**raw)
            result.validate()
            return result
        except (OSError, TypeError, yaml.YAMLError, ConfigError) as exc:
            raise ConfigError(f"cannot load calibration {path}: {exc}; "
                              "run python -m app.calibrate first") from exc

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


class PCA9685:
    frequency = 50

    def __init__(self, calibration: Calibration, bus_factory=None) -> None:
        calibration.validate()
        self.calibration = calibration
        self._bus = None
        try:
            if bus_factory is None:
                from smbus2 import SMBus
                bus_factory = SMBus
            self._bus = bus_factory(calibration.bus)
            # Disable our channels before changing the shared PWM frequency.
            self.disable()
            old = self._bus.read_byte_data(calibration.address, 0x00)
            mode = (old & ~0x90) | 0x20  # auto-increment for four-register block writes
            self._write(0x00, mode | 0x10)  # sleep
            self._write(0xFE, round(25_000_000 / (4096 * self.frequency)) - 1)
            self._write(0x00, mode)
            time.sleep(0.005)
            self._write(0x00, mode | 0x80)  # restart
            self._write(0x01, 0x04)  # totem-pole output
        except Exception as exc:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(f"cannot initialize Pan-Tilt HAT: {exc}; enable I2C, check "
                               "bus/address and install the hardware extra (smbus2)") from exc

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.calibration.address, register, value)

    def pulse(self, channel: int, pulse_us: float) -> None:
        if (channel not in (self.calibration.pan.channel, self.calibration.tilt.channel)
                or not math.isfinite(pulse_us) or not 500 <= pulse_us <= 2500):
            raise ValueError("invalid servo pulse or unconfigured channel")
        count = round(pulse_us * self.frequency * 4096 / 1_000_000)
        self._bus.write_i2c_block_data(self.calibration.address, 0x06 + 4 * channel,
                                      [0, 0, count & 0xFF, (count >> 8) & 0x0F])

    def move(self, pan: float, tilt: float) -> None:
        for axis, angle in ((self.calibration.pan, pan), (self.calibration.tilt, tilt)):
            self.pulse(axis.channel, axis.pulse(angle))

    def disable(self) -> None:
        error = None
        for axis in (self.calibration.pan, self.calibration.tilt):
            try:
                self._write(0x09 + 4 * axis.channel, 0x10)  # FULL_OFF
            except Exception as exc:
                error = exc
        if error is not None:
            raise error

    def close(self) -> None:
        if self._bus is not None:
            try:
                self.disable()
            finally:
                bus, self._bus = self._bus, None
                bus.close()


class SimulatedMount:
    def move(self, pan: float, tilt: float) -> None:
        pass

    def close(self) -> None:
        pass
