"""Interactive, manually stepped calibration: python -m app.calibrate."""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Protocol, TextIO, cast

from .config import CameraRotation, ConfigError
from .pantilt import PCA9685, AxisCalibration, Calibration

INITIAL_PULSE_US = 1500.0
PULSE_STEP_US = 10
ESCAPE_TIMEOUT_S = 0.05
MAX_ESCAPE_LENGTH = 8
ARROW_COMMANDS = {"LEFT": "pan -", "RIGHT": "pan +", "UP": "tilt -", "DOWN": "tilt +"}
IGNORE_KEY = "IGNORE"
ReadCommand = Callable[[str], str]
WriteMessage = Callable[[str], None]


class PulseMount(Protocol):
    def pulse(self, channel: int, pulse_us: float) -> None: ...


class WindowsConsole(Protocol):
    def getwch(self) -> str: ...


class CalibrationInterrupted(KeyboardInterrupt):
    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def interrupt_calibration(signal_number: int, frame: FrameType | None) -> None:
    raise CalibrationInterrupted(signal_number)


@contextmanager
def cancellation_signals() -> Iterator[None]:
    signals = (signal.SIGTERM,) if os.name == "nt" else (signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {}
    try:
        for signal_number in signals:
            previous_handlers[signal_number] = signal.signal(signal_number, interrupt_calibration)
        yield
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


class KeyboardCommands:
    """Immediate arrows plus editable command lines, sharing the calibration command path."""

    def __init__(self, read_key: Callable[[], str], output: TextIO) -> None:
        self.read_key = read_key
        self.output = output

    def __call__(self, prompt: str) -> str:
        self.output.write(prompt)
        self.output.flush()
        buffer = ""
        while True:
            key = self.read_key()
            if key in ("\x04", "\x1a", ""):
                self.output.write("\n")
                raise EOFError
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ARROW_COMMANDS:
                if buffer:
                    continue
                command = ARROW_COMMANDS[key]
                self.output.write(command + "\n")
                self.output.flush()
                return command
            if key in ("\r", "\n"):
                self.output.write("\n")
                self.output.flush()
                return buffer
            if key in ("\x08", "\x7f"):
                if buffer:
                    buffer = buffer[:-1]
                    self.output.write("\b \b")
            elif len(key) == 1 and key.isascii() and key.isprintable():
                buffer += key
                self.output.write(key)
            self.output.flush()


def read_ascii_key(descriptor: int) -> str:
    value = os.read(descriptor, 1)
    if not value:
        return ""
    try:
        return value.decode("ascii")
    except UnicodeDecodeError:
        return IGNORE_KEY


def posix_key(descriptor: int) -> str:
    import select

    value = read_ascii_key(descriptor)
    if value != "\x1b":
        return value
    sequence = ""
    while (
        len(sequence) < MAX_ESCAPE_LENGTH
        and select.select([descriptor], [], [], ESCAPE_TIMEOUT_S)[0]
    ):
        character = read_ascii_key(descriptor)
        if not character or character == IGNORE_KEY:
            return IGNORE_KEY
        sequence += character
        if len(sequence) >= 2 and (character.isalpha() or character == "~"):
            break
    return {
        "[A": "UP",
        "[B": "DOWN",
        "[C": "RIGHT",
        "[D": "LEFT",
        "OA": "UP",
        "OB": "DOWN",
        "OC": "RIGHT",
        "OD": "LEFT",
    }.get(sequence, IGNORE_KEY)


def windows_key() -> str:
    console = cast(WindowsConsole, importlib.import_module("msvcrt"))
    key = console.getwch()
    if key in ("\x00", "\xe0"):
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(console.getwch(), IGNORE_KEY)
    return key


@contextmanager
def command_input(line_input: bool = False) -> Iterator[ReadCommand]:
    with cancellation_signals():
        if line_input or not sys.stdin.isatty() or not sys.stdout.isatty():
            yield input
            return
        if os.name == "nt":
            yield KeyboardCommands(windows_key, sys.stdout)
            return
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        previous = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            yield KeyboardCommands(lambda: posix_key(descriptor), sys.stdout)
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


@dataclass(frozen=True)
class AxisMarks:
    min_pulse_us: float | None = None
    home_pulse_us: float | None = None
    max_pulse_us: float | None = None
    direction: int | None = None

    def mark(self, command: str, pulse_us: float) -> AxisMarks:
        if command == "min":
            return replace(self, min_pulse_us=pulse_us)
        if command == "home":
            return replace(self, home_pulse_us=pulse_us)
        return replace(self, max_pulse_us=pulse_us)

    def apply(self, axis: AxisCalibration, name: str) -> AxisCalibration:
        if (
            self.min_pulse_us is None
            or self.home_pulse_us is None
            or self.max_pulse_us is None
            or self.direction is None
        ):
            raise ConfigError(f"Record min, home, max, and direction for {name} first.")
        result = replace(
            axis,
            min_angle=axis.angle_from_pulse(self.min_pulse_us),
            home_angle=axis.angle_from_pulse(self.home_pulse_us),
            max_angle=axis.angle_from_pulse(self.max_pulse_us),
            direction=self.direction,
        )
        result.validate(name)
        return result


def calibrate(
    profile: Calibration,
    mount: PulseMount,
    path: str | Path,
    read: ReadCommand = input,
    write: WriteMessage = print,
) -> bool:
    profile.validate()
    axes = {"pan": profile.pan, "tilt": profile.tilt}
    pulses = {name: INITIAL_PULSE_US for name in axes}
    marks = {name: AxisMarks() for name in axes}
    for name, axis in axes.items():
        if not axis.pulse_min_us <= INITIAL_PULSE_US <= axis.pulse_max_us:
            raise ConfigError(f"{name} pulse envelope must contain the initial 1500 us")
    for axis in axes.values():
        mount.pulse(axis.channel, INITIAL_PULSE_US)
    write(
        "Both axes commanded to 1500 us. No automatic sweep.\n"
        "Nominal servo degrees use 500-2500 us = 0-180 degrees; "
        "measure physical travel on hardware.\n"
        "Arrow keys: left/right = pan -/+; up/down = tilt -/+ (10 us per step).\n"
        "Commands: pan + | pan - | tilt + | tilt - (10 us per step);\n"
        "pan min|home|max (mark current position); pan direction 1|-1;\n"
        "use tilt for the other axis; status; save; quit.\n"
        "Direction uses the configured capture image: x right, y down.\n"
        "Direction 1 means increasing pulse aims the camera towards a target right/down;\n"
        "a stationary target then moves left/up in that image.\n"
        "Match --image-rotation-deg to camera.rotation_deg and recheck both directions.\n"
        "A separate preview flip must not change this direction convention.\n"
        "Verify each axis direction and mark both travel limits and home before saving."
    )
    write(f"Calibration settings: {profile}; capture rotation: {profile.image_rotation_deg} deg.")
    while True:
        try:
            parts = read("calibrate> ").strip().lower().split()
        except EOFError:
            return False
        if not parts:
            continue
        if parts == ["quit"]:
            return False
        if parts == ["status"]:
            write(str({"pulses_us": pulses, "marks": marks}))
            continue
        if parts == ["save"]:
            try:
                result = replace(
                    profile,
                    pan=marks["pan"].apply(profile.pan, "pan"),
                    tilt=marks["tilt"].apply(profile.tilt, "tilt"),
                )
                result.save(path)
            except (ConfigError, OSError) as exception:
                write(str(exception))
                continue
            write(f"Saved {path}: {result}; capture rotation: {result.image_rotation_deg} deg.")
            return True
        if len(parts) not in (2, 3) or parts[0] not in axes:
            write("Unknown command; use pan/tilt +, -, min, home, max, direction 1|-1.")
            continue
        name, command = parts[:2]
        axis = axes[name]
        if command in ("+", "-") and len(parts) == 2:
            pulse_us = pulses[name] + (PULSE_STEP_US if command == "+" else -PULSE_STEP_US)
            if not axis.pulse_min_us <= pulse_us <= axis.pulse_max_us:
                write(f"{name} pulse envelope reached; movement refused.")
                continue
            mount.pulse(axis.channel, pulse_us)
            pulses[name] = pulse_us
            write(f"{name}: {pulse_us:g} us")
        elif command in ("min", "home", "max") and len(parts) == 2:
            marks[name] = marks[name].mark(command, pulses[name])
            write(f"Marked {name} {command}: {pulses[name]:g} us")
        elif command == "direction" and len(parts) == 3 and parts[2] in ("1", "-1"):
            marks[name] = replace(marks[name], direction=int(parts[2]))
        else:
            write("Unknown command.")


@dataclass
class CalibrationArguments:
    output: str = "calibration.local.yaml"
    bus: int = 1
    address: int = 0x40
    line_input: bool = False
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None
    image_rotation_deg: CameraRotation = 0
    pan_channel: int = 0
    pan_pulse_min_us: float = 1000
    pan_pulse_max_us: float = 2000
    tilt_channel: int = 1
    tilt_pulse_min_us: float = 1000
    tilt_pulse_max_us: float = 2000

    def profile(self) -> Calibration:
        return Calibration(
            pan=calibration_axis(
                "pan", self.pan_channel, self.pan_pulse_min_us, self.pan_pulse_max_us
            ),
            tilt=calibration_axis(
                "tilt", self.tilt_channel, self.tilt_pulse_min_us, self.tilt_pulse_max_us
            ),
            bus=self.bus,
            address=self.address,
            horizontal_fov_deg=self.horizontal_fov_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            image_rotation_deg=self.image_rotation_deg,
        )


def calibration_axis(
    name: str, channel: int, pulse_min_us: float, pulse_max_us: float
) -> AxisCalibration:
    axis = AxisCalibration(channel, pulse_min_us, pulse_max_us)
    if not pulse_min_us < INITIAL_PULSE_US < pulse_max_us:
        raise ConfigError(f"{name} pulse envelope must contain 1500 us strictly between its limits")
    axis = replace(
        axis,
        min_angle=axis.angle_from_pulse(pulse_min_us),
        max_angle=axis.angle_from_pulse(pulse_max_us),
    )
    axis.validate(name)
    return axis


def cli() -> None:
    parser = argparse.ArgumentParser(description="Manually calibrate the Waveshare camera mount")
    parser.add_argument("--output", default="calibration.local.yaml")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument(
        "--image-rotation-deg",
        type=int,
        choices=(0, 180),
        default=0,
        help="camera.rotation_deg used when verifying axis directions; no camera is opened",
    )
    parser.add_argument(
        "--line-input",
        action="store_true",
        help="use typed commands only, without immediate arrow keys",
    )
    for name, axis in (("pan", Calibration().pan), ("tilt", Calibration().tilt)):
        parser.add_argument(f"--{name}-channel", type=int, default=axis.channel)
        parser.add_argument(f"--{name}-pulse-min-us", type=float, default=axis.pulse_min_us)
        parser.add_argument(f"--{name}-pulse-max-us", type=float, default=axis.pulse_max_us)
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=None,
        help="measured horizontal field of view in degrees, if known",
    )
    parser.add_argument(
        "--vertical-fov-deg",
        type=float,
        default=None,
        help="measured vertical field of view in degrees, if known",
    )
    arguments = CalibrationArguments()
    parser.parse_args(namespace=arguments)
    try:
        with cancellation_signals():
            run_calibration(arguments)
    except (ConfigError, OSError, RuntimeError) as exception:
        parser.exit(1, f"calibration failed: {exception}\n")
    except CalibrationInterrupted as exception:
        parser.exit(128 + exception.signal_number, "\nCalibration cancelled.\n")
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")


def run_calibration(arguments: CalibrationArguments) -> None:
    profile = arguments.profile()
    profile.validate()
    for name, axis in (("pan", profile.pan), ("tilt", profile.tilt)):
        if not axis.pulse_min_us <= INITIAL_PULSE_US <= axis.pulse_max_us:
            raise ConfigError(f"{name} pulse envelope must contain 1500 us")
    print("Prepare for both servos to move to 1500 us. Follow the assembly guide first.")
    mount: PCA9685 | None = None
    failure: BaseException | None = None
    try:
        mount = PCA9685(profile)
        with command_input(arguments.line_input) as read:
            calibrate(profile, mount, arguments.output, read=read)
    except BaseException as exception:
        failure = exception
        raise
    finally:
        close_calibration_mount(mount, failure)


def close_calibration_mount(mount: PCA9685 | None, failure: BaseException | None) -> None:
    if mount is None:
        return
    try:
        mount.close()
    except Exception as exception:
        if failure is None:
            raise
        failure.add_note(f"calibration cleanup also failed: {exception}")


if __name__ == "__main__":
    cli()
