"""Calibrated camera mount and PCA9685 I2C driver with optional hardware imports."""

from __future__ import annotations

import errno
import importlib
import math
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml

from .config import CameraRotation, ConfigError, finite_number, read_mapping, rotation_degrees
from .json_data import JsonValue

SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2500
SERVO_TRAVEL_DEG = 180
SERVO_CHANNEL_COUNT = 16
PWM_FREQUENCY_HZ = 50
PWM_RESOLUTION = 4096
OSCILLATOR_FREQUENCY_HZ = 25_000_000
MICROSECONDS_PER_SECOND = 1_000_000
OSCILLATOR_SETTLE_S = 0.005
MODE_REGISTER = 0x00
OUTPUT_REGISTER = 0x01
PRESCALE_REGISTER = 0xFE
CHANNEL_REGISTER = 0x06
CHANNEL_REGISTER_COUNT = 4
FULL_OFF_OFFSET = 3
FULL_OFF = 0x10
SLEEP = 0x10
RESTART = 0x80
AUTO_INCREMENT = 0x20
TOTEM_POLE = 0x04
BYTE_MASK = 0xFF
HIGH_COUNT_MASK = 0x0F
BITS_PER_BYTE = 8
MIN_I2C_ADDRESS = 0x08
MAX_I2C_ADDRESS = 0x77
AXIS_FIELDS = frozenset(
    {"channel", "pulse_min_us", "pulse_max_us", "min_angle", "home_angle", "max_angle", "direction"}
)
CALIBRATION_FIELDS = frozenset(
    {
        "pan",
        "tilt",
        "bus",
        "address",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "image_rotation_deg",
    }
)


class PanTiltDriver(Protocol):
    def move(self, pan_deg: float, tilt_deg: float) -> None: ...

    def hold(self) -> None:
        """Keep the last commanded position without disabling servo output."""
        ...

    def close(self) -> None: ...


class I2CBus(Protocol):
    def read_byte_data(self, address: int, register: int, /) -> int: ...

    def write_byte_data(self, address: int, register: int, value: int, /) -> None: ...

    def write_i2c_block_data(self, address: int, register: int, values: list[int], /) -> None: ...

    def close(self, /) -> None: ...


def calibration_integer(value: JsonValue, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    return value


def optional_field_of_view(value: JsonValue, name: str) -> float | None:
    return None if value is None else finite_number(value, name)


@dataclass(frozen=True)
class AxisCalibration:
    channel: int
    pulse_min_us: float = 1000
    pulse_max_us: float = 2000
    min_angle: float = 60
    home_angle: float = 90
    max_angle: float = 120
    direction: int = 1

    def validate(self, name: str = "servo") -> None:
        if type(self.channel) is not int or not 0 <= self.channel < SERVO_CHANNEL_COUNT:
            raise ConfigError(f"{name} channel must be an integer in [0, 15]")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ConfigError(f"{name} direction must be -1 or 1")
        for field, value in (
            ("pulse_min_us", self.pulse_min_us),
            ("pulse_max_us", self.pulse_max_us),
            ("min_angle", self.min_angle),
            ("home_angle", self.home_angle),
            ("max_angle", self.max_angle),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigError(f"{name} {field} must be finite")
        if not SERVO_MIN_PULSE_US <= self.pulse_min_us < self.pulse_max_us <= SERVO_MAX_PULSE_US:
            raise ConfigError(f"{name} pulse bounds must satisfy 500 <= min < max <= 2500 us")
        if not 0 <= self.min_angle < self.home_angle < self.max_angle <= SERVO_TRAVEL_DEG:
            raise ConfigError(f"{name} angles must satisfy 0 <= min < home < max <= 180")
        if not (
            self.angle_from_pulse(self.pulse_min_us) <= self.min_angle
            and self.max_angle <= self.angle_from_pulse(self.pulse_max_us)
        ):
            raise ConfigError(f"{name} angle limits must fit within the pulse envelope")

    @classmethod
    def from_mapping(cls, value: JsonValue, name: str) -> AxisCalibration:
        fields = read_mapping(value, f"calibration {name}")
        if not AXIS_FIELDS <= fields.keys():
            raise ConfigError(f"calibration {name} is incomplete")
        if fields.keys() - AXIS_FIELDS:
            raise ConfigError(f"unknown calibration {name} settings: {fields.keys() - AXIS_FIELDS}")
        return cls(
            channel=calibration_integer(fields["channel"], f"{name} channel"),
            pulse_min_us=finite_number(fields["pulse_min_us"], f"{name} pulse_min_us"),
            pulse_max_us=finite_number(fields["pulse_max_us"], f"{name} pulse_max_us"),
            min_angle=finite_number(fields["min_angle"], f"{name} min_angle"),
            home_angle=finite_number(fields["home_angle"], f"{name} home_angle"),
            max_angle=finite_number(fields["max_angle"], f"{name} max_angle"),
            direction=calibration_integer(fields["direction"], f"{name} direction"),
        )

    def clamp(self, angle_deg: float) -> float:
        if not math.isfinite(angle_deg):
            raise ValueError("servo angle must be finite")
        return max(self.min_angle, min(self.max_angle, angle_deg))

    def pulse(self, angle_deg: float) -> float:
        """Map nominal servo degrees to pulses; physical angles need hardware measurement."""
        pulse_us = SERVO_MIN_PULSE_US + self.clamp(angle_deg) / SERVO_TRAVEL_DEG * (
            SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US
        )
        return max(self.pulse_min_us, min(self.pulse_max_us, pulse_us))

    def angle_from_pulse(self, pulse_us: float) -> float:
        if not math.isfinite(pulse_us) or not self.pulse_min_us <= pulse_us <= self.pulse_max_us:
            raise ValueError("servo pulse must be within its calibrated envelope")
        # one nominal scale keeps speed independent of the chosen safety envelope.
        return (
            (pulse_us - SERVO_MIN_PULSE_US)
            / (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)
            * SERVO_TRAVEL_DEG
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "channel": self.channel,
            "pulse_min_us": self.pulse_min_us,
            "pulse_max_us": self.pulse_max_us,
            "min_angle": self.min_angle,
            "home_angle": self.home_angle,
            "max_angle": self.max_angle,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class Calibration:
    pan: AxisCalibration = AxisCalibration(channel=0)
    tilt: AxisCalibration = AxisCalibration(channel=1)
    bus: int = 1
    address: int = 0x40
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None
    image_rotation_deg: CameraRotation = 0

    def validate_image_rotation(self, rotation_deg: CameraRotation) -> None:
        if self.image_rotation_deg != rotation_deg:
            raise ConfigError(
                "calibration image_rotation_deg does not match camera.rotation_deg; "
                "verify both axis directions with the configured capture orientation"
            )

    def validate(self) -> None:
        self.pan.validate("pan")
        self.tilt.validate("tilt")
        if self.pan.channel == self.tilt.channel:
            raise ConfigError("pan and tilt must use different channels")
        if type(self.bus) is not int or self.bus < 0:
            raise ConfigError("I2C bus must be a non-negative integer")
        if type(self.address) is not int or not MIN_I2C_ADDRESS <= self.address <= MAX_I2C_ADDRESS:
            raise ConfigError("I2C address must be a usable 7-bit integer")
        rotation_degrees(self.image_rotation_deg, "calibration image_rotation_deg")

        for name, value in (
            ("horizontal_fov_deg", self.horizontal_fov_deg),
            ("vertical_fov_deg", self.vertical_fov_deg),
        ):
            if value is not None and not 0 < finite_number(value, name) < SERVO_TRAVEL_DEG:
                raise ConfigError(f"{name} must be between 0 and 180 degrees, exclusively")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        try:
            fields = read_mapping(
                cast(JsonValue, yaml.safe_load(Path(path).read_text(encoding="utf-8"))),
                "calibration",
            )
            if not {"pan", "tilt"} <= fields.keys():
                raise ConfigError("calibration must contain pan and tilt mappings")
            if fields.keys() - CALIBRATION_FIELDS:
                raise ConfigError(
                    f"unknown calibration settings: {fields.keys() - CALIBRATION_FIELDS}"
                )
            result = cls(
                pan=AxisCalibration.from_mapping(fields["pan"], "pan"),
                tilt=AxisCalibration.from_mapping(fields["tilt"], "tilt"),
                bus=calibration_integer(fields.get("bus", 1), "I2C bus"),
                address=calibration_integer(fields.get("address", 0x40), "I2C address"),
                horizontal_fov_deg=optional_field_of_view(
                    fields.get("horizontal_fov_deg"), "horizontal_fov_deg"
                ),
                vertical_fov_deg=optional_field_of_view(
                    fields.get("vertical_fov_deg"), "vertical_fov_deg"
                ),
                image_rotation_deg=rotation_degrees(
                    fields.get("image_rotation_deg", 0), "calibration image_rotation_deg"
                ),
            )
            result.validate()
            return result
        except (OSError, UnicodeError, TypeError, yaml.YAMLError, ConfigError) as exception:
            raise ConfigError(
                f"cannot load calibration {path}: {exception}; run python -m app.calibrate first"
            ) from exception

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "pan": self.pan.as_dict(),
            "tilt": self.tilt.as_dict(),
            "bus": self.bus,
            "address": self.address,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "image_rotation_deg": self.image_rotation_deg,
        }

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                if destination.exists():
                    os.fchmod(temporary.fileno(), stat.S_IMODE(destination.stat().st_mode))
                yaml.safe_dump(self.as_dict(), temporary, sort_keys=False)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            sync_directory(destination.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as exception:
        if exception.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise
    finally:
        os.close(descriptor)


class PCA9685:
    frequency = PWM_FREQUENCY_HZ

    def __init__(
        self, calibration: Calibration, bus_factory: Callable[[int], I2CBus] | None = None
    ) -> None:
        calibration.validate()
        self.calibration = calibration
        self._bus: I2CBus | None = None
        try:
            if bus_factory is None:
                bus_factory = cast(Callable[[int], I2CBus], importlib.import_module("smbus2").SMBus)
            self._bus = bus_factory(calibration.bus)
            # disable our channels before changing the shared PWM frequency.
            self.disable()
            previous_mode = self._require_bus().read_byte_data(calibration.address, MODE_REGISTER)
            mode = (previous_mode & ~(RESTART | SLEEP)) | AUTO_INCREMENT
            self._write(MODE_REGISTER, mode | SLEEP)
            self._write(
                PRESCALE_REGISTER,
                round(OSCILLATOR_FREQUENCY_HZ / (PWM_RESOLUTION * self.frequency)) - 1,
            )
            self._write(MODE_REGISTER, mode)
            time.sleep(OSCILLATOR_SETTLE_S)
            self._write(MODE_REGISTER, mode | RESTART)
            self._write(OUTPUT_REGISTER, TOTEM_POLE)
        except BaseException as exception:
            try:
                self.close()
            except BaseException as cleanup_exception:
                exception.add_note(f"Pan-Tilt HAT cleanup also failed: {cleanup_exception}")
            if not isinstance(exception, Exception):
                raise
            raise RuntimeError(
                f"cannot initialise Pan-Tilt HAT: {exception}; enable I2C, check "
                "bus/address and install the hardware extra (smbus2)"
            ) from exception

    def _require_bus(self) -> I2CBus:
        if self._bus is None:
            raise RuntimeError("Pan-Tilt HAT is closed")
        return self._bus

    def _write(self, register: int, value: int) -> None:
        self._require_bus().write_byte_data(self.calibration.address, register, value)

    def pulse(self, channel: int, pulse_us: float) -> None:
        if (
            type(channel) is not int
            or channel not in (self.calibration.pan.channel, self.calibration.tilt.channel)
            or not math.isfinite(pulse_us)
            or not SERVO_MIN_PULSE_US <= pulse_us <= SERVO_MAX_PULSE_US
        ):
            raise ValueError("invalid servo pulse or unconfigured channel")
        count = round(pulse_us * self.frequency * PWM_RESOLUTION / MICROSECONDS_PER_SECOND)
        self._require_bus().write_i2c_block_data(
            self.calibration.address,
            CHANNEL_REGISTER + CHANNEL_REGISTER_COUNT * channel,
            [0, 0, count & BYTE_MASK, (count >> BITS_PER_BYTE) & HIGH_COUNT_MASK],
        )

    def move(self, pan_deg: float, tilt_deg: float) -> None:
        pan_pulse_us = self.calibration.pan.pulse(pan_deg)
        tilt_pulse_us = self.calibration.tilt.pulse(tilt_deg)
        self.pulse(self.calibration.pan.channel, pan_pulse_us)
        self.pulse(self.calibration.tilt.channel, tilt_pulse_us)

    def hold(self) -> None:
        self._require_bus()

    def disable(self) -> None:
        first_error: Exception | None = None
        for axis in (self.calibration.pan, self.calibration.tilt):
            try:
                self._write(
                    CHANNEL_REGISTER + FULL_OFF_OFFSET + CHANNEL_REGISTER_COUNT * axis.channel,
                    FULL_OFF,
                )
            except Exception as exception:
                if first_error is None:
                    first_error = exception
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        bus = self._bus
        if bus is None:
            return
        failure: BaseException | None = None
        try:
            self.disable()
        except BaseException as exception:
            failure = exception
        self._bus = None
        try:
            bus.close()
        except BaseException as exception:
            if failure is None:
                failure = exception
            else:
                failure.add_note(f"I2C bus close also failed: {exception}")
        if failure is not None:
            raise failure


class SimulatedMount:
    def move(self, pan_deg: float, tilt_deg: float) -> None:
        pass

    def hold(self) -> None:
        pass

    def close(self) -> None:
        pass
