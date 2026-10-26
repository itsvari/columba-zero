from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from app.calibrate import (
    CalibrationArguments,
    CalibrationInterrupted,
    KeyboardCommands,
    calibrate,
    cancellation_signals,
    cli,
    command_input,
    posix_key,
    run_calibration,
    windows_key,
)
from app.config import ConfigError
from app.pantilt import AxisCalibration, Calibration

PROCESS_TIMEOUT_S = 5
POLL_INTERVAL_S = 0.05


def commands_from(keys: Sequence[str]) -> KeyboardCommands:
    iterator = iter(keys)
    return KeyboardCommands(lambda: next(iterator), io.StringIO())


def measured_commands() -> list[str]:
    return [
        "save",
        "pan home",
        "pan -",
        "pan min",
        "pan +",
        "pan +",
        "pan max",
        "pan direction -1",
        "tilt home",
        "tilt -",
        "tilt min",
        "tilt +",
        "tilt +",
        "tilt max",
        "tilt direction 1",
        "save",
    ]


class CalibrationTests(unittest.TestCase):
    def test_requires_measured_marks_and_saves_profile(self) -> None:
        commands = iter(measured_commands())
        output = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            self.assertTrue(
                calibrate(
                    replace(Calibration(), image_rotation_deg=180),
                    Mock(),
                    path,
                    read=lambda _: next(commands),
                    write=output,
                )
            )
            result = Calibration.load(path)
            self.assertEqual(result.image_rotation_deg, 180)
            self.assertEqual(result.pan.direction, -1)
            self.assertEqual(result.pan.home_angle, 90)
            self.assertAlmostEqual(result.pan.min_angle, 89.1)
        self.assertTrue(any("Record min" in call.args[0] for call in output.call_args_list))
        saved = next(
            str(call.args[0])
            for call in output.call_args_list
            if str(call.args[0]).startswith("Saved ")
        )
        self.assertIn("capture rotation: 180 deg.", saved)

    def test_safety_envelope_endpoints_save_in_nominal_degrees(self) -> None:
        commands = iter(
            [
                "pan home",
                *(["pan -"] * 50),
                "pan min",
                *(["pan +"] * 100),
                "pan max",
                "pan direction 1",
                "tilt home",
                "tilt -",
                "tilt min",
                "tilt +",
                "tilt +",
                "tilt max",
                "tilt direction -1",
                "save",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            self.assertTrue(
                calibrate(Calibration(), Mock(), path, read=lambda _: next(commands), write=Mock())
            )
            result = Calibration.load(path)
            self.assertEqual(result.pan.min_angle, 45)
            self.assertEqual(result.pan.max_angle, 135)

    def test_invalid_mark_order_reports_axis_and_allows_correction(self) -> None:
        commands = iter(
            [
                "pan min",
                "pan home",
                "pan max",
                "pan direction 1",
                "save",
                *measured_commands()[1:],
            ]
        )
        output = Mock()
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(
                calibrate(
                    Calibration(),
                    Mock(),
                    Path(directory) / "profile.yaml",
                    read=lambda _: next(commands),
                    write=output,
                )
            )
        self.assertTrue(any("pan angles must" in call.args[0] for call in output.call_args_list))

    def test_refuses_envelope_overrun_and_quit_does_not_save(self) -> None:
        commands = iter([*(["pan +"] * 51), "quit"])
        mount, output = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-saved.yaml"
            self.assertFalse(
                calibrate(Calibration(), mount, path, read=lambda _: next(commands), write=output)
            )
            self.assertFalse(path.exists())
            self.assertLessEqual(max(call.args[1] for call in mount.pulse.call_args_list), 2000)
        self.assertIn("pan pulse envelope", output.call_args.args[0])

    def test_invalid_initial_envelope_fails_before_any_movement(self) -> None:
        mount = Mock()
        profile = replace(
            Calibration(), tilt=AxisCalibration(1, pulse_min_us=1600, min_angle=100, home_angle=110)
        )
        with self.assertRaisesRegex(ConfigError, "tilt pulse envelope"):
            calibrate(profile, mount, "unused.yaml")
        mount.pulse.assert_not_called()

    def test_save_failure_preserves_marks_for_retry(self) -> None:
        commands = iter([*measured_commands(), "save"])
        output = Mock()
        save = Mock(side_effect=[OSError("disk full"), None])
        with patch.object(Calibration, "save", save):
            self.assertTrue(
                calibrate(
                    Calibration(),
                    Mock(),
                    "unused.yaml",
                    read=lambda _: next(commands),
                    write=output,
                )
            )
        self.assertEqual(save.call_count, 2)
        self.assertTrue(any("disk full" in call.args[0] for call in output.call_args_list))

    def test_undecodable_keys_preserve_unsaved_marks_until_save(self) -> None:
        keys = "\r".join(measured_commands()) + "\r"
        encoded = keys.replace("pan max\r", "pan max\r\u00e9").encode("utf-8")
        raw_reads = [bytes([value]) for value in encoded]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            with patch("app.calibrate.os.read", side_effect=raw_reads):
                reader = KeyboardCommands(lambda: posix_key(7), io.StringIO())
                self.assertTrue(calibrate(Calibration(), Mock(), path, read=reader, write=Mock()))
            self.assertEqual(Calibration.load(path).pan.direction, -1)


class KeyboardTests(unittest.TestCase):
    def test_editing_and_arrow_during_partial_command(self) -> None:
        read = commands_from([*"pan homx", "\x7f", "LEFT", "e", "\r"])
        self.assertEqual(read("> "), "pan home")

    def test_eof_and_keyboard_interrupt(self) -> None:
        for key in ("", "\x04", "\x1a"):
            with self.subTest(key=key), self.assertRaises(EOFError):
                commands_from([key])("> ")
        with self.assertRaises(KeyboardInterrupt):
            commands_from(["\x03"])("> ")
        with patch("app.calibrate.os.read", return_value=b""):
            self.assertEqual(posix_key(7), "")

    def test_arrow_motion_uses_configured_channels_and_envelope(self) -> None:
        read = commands_from(["LEFT", "UP", *(["RIGHT"] * 52), *"quit", "\r"])
        mount = Mock()
        profile = Calibration(pan=AxisCalibration(1), tilt=AxisCalibration(0))
        self.assertFalse(calibrate(profile, mount, "unused.yaml", read=read, write=Mock()))
        calls = mount.pulse.call_args_list
        self.assertEqual(
            [call.args for call in calls[:4]],
            [
                (1, 1500),
                (0, 1500),
                (1, 1490),
                (0, 1490),
            ],
        )
        self.assertLessEqual(max(call.args[1] for call in calls), 2000)

    def test_posix_escape_sequence_decoding(self) -> None:
        for sequence, expected in (
            (b"\x1b[A", "UP"),
            (b"\x1bOC", "RIGHT"),
            (b"\x1b[3~", "IGNORE"),
            (b"\x1b\xff", "IGNORE"),
        ):
            with (
                patch("app.calibrate.os.read", side_effect=[bytes([value]) for value in sequence]),
                patch("select.select", return_value=([7], [], [])),
            ):
                self.assertEqual(posix_key(7), expected)

    def test_windows_extended_keys(self) -> None:
        for code, expected in (("H", "UP"), ("M", "RIGHT"), ("\x86", "IGNORE")):
            console = Mock()
            console.getwch.side_effect = ["\xe0", code]
            with patch.dict("sys.modules", {"msvcrt": console}):
                self.assertEqual(windows_key(), expected)

    def test_line_input_fallback(self) -> None:
        with command_input(line_input=True) as read:
            self.assertIs(read, input)
        with patch("app.calibrate.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            with command_input() as read:
                self.assertIs(read, input)


class LifecycleTests(unittest.TestCase):
    def test_cli_flags_reach_the_calibration_profile(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "calibrate",
                    "--line-input",
                    "--pan-channel",
                    "1",
                    "--tilt-channel",
                    "0",
                    "--pan-pulse-max-us",
                    "1800",
                    "--horizontal-fov-deg",
                    "62.2",
                    "--vertical-fov-deg",
                    "48.8",
                ],
            ),
            patch("app.calibrate.PCA9685") as driver,
            patch("app.calibrate.calibrate"),
            patch("builtins.print"),
        ):
            cli()
        profile = driver.call_args.args[0]
        self.assertEqual((profile.pan.channel, profile.tilt.channel), (1, 0))
        self.assertEqual(profile.pan.pulse_max_us, 1800)
        self.assertEqual((profile.horizontal_fov_deg, profile.vertical_fov_deg), (62.2, 48.8))

    def test_initialisation_failure_and_interrupted_calibration_cleanup(self) -> None:
        with (
            patch("app.calibrate.PCA9685", side_effect=RuntimeError("initialisation failed")),
            patch("builtins.print"),
            self.assertRaisesRegex(RuntimeError, "initialisation failed"),
        ):
            run_calibration(CalibrationArguments())
        with (
            patch("app.calibrate.PCA9685") as driver,
            patch("app.calibrate.calibrate", side_effect=KeyboardInterrupt),
            patch("builtins.print"),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_calibration(CalibrationArguments(line_input=True))
        driver.return_value.close.assert_called_once()

    def test_cleanup_failure_preserves_original_exception(self) -> None:
        original = OSError("command failed")
        with (
            patch("app.calibrate.PCA9685") as driver,
            patch("app.calibrate.calibrate", side_effect=original),
            patch("builtins.print"),
        ):
            driver.return_value.close.side_effect = OSError("close failed")
            with self.assertRaises(OSError) as caught:
                run_calibration(CalibrationArguments(line_input=True))
        self.assertIs(caught.exception, original)
        self.assertIn("close failed", original.__notes__[0])

    def test_cleanup_failure_without_active_exception_propagates(self) -> None:
        with (
            patch("app.calibrate.PCA9685") as driver,
            patch("app.calibrate.calibrate"),
            patch("builtins.print"),
        ):
            driver.return_value.close.side_effect = OSError("close failed")
            with self.assertRaisesRegex(OSError, "close failed"):
                run_calibration(CalibrationArguments(line_input=True))

    @unittest.skipIf(os.name == "nt", "POSIX signals required")
    def test_prior_signal_handlers_restored(self) -> None:
        previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
        for signal_number in previous:
            with self.subTest(signal_number=signal_number):
                with self.assertRaises(CalibrationInterrupted), cancellation_signals():
                    signal.raise_signal(signal_number)
                self.assertEqual(
                    {number: signal.getsignal(number) for number in previous}, previous
                )

    @unittest.skipIf(os.name == "nt", "POSIX pseudo-terminal and signals required")
    def test_subprocess_signals_restore_terminal_close_mount_and_restore_handlers(self) -> None:
        import pty
        import select
        import termios

        program = """
import signal
import sys
import termios
from pathlib import Path
from app import calibrate

marker = Path(sys.argv[1])
closed = False

def previous_handler(number, frame):
    pass

for number in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(number, previous_handler)

class Mount:
    def __init__(self, profile):
        pass
    def pulse(self, channel, pulse_us):
        pass
    def close(self):
        global closed
        closed = True

calibrate.PCA9685 = Mount
Path(str(marker) + ".terminal").write_text(repr(termios.tcgetattr(0)))
sys.argv = ["calibrate"]
try:
    calibrate.cli()
finally:
    restored = all(signal.getsignal(number) is previous_handler
                   for number in (signal.SIGTERM, signal.SIGHUP))
    marker.write_text(f"closed={closed} restored={restored}")
"""
        for signal_number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with (
                self.subTest(signal_number=signal_number),
                tempfile.TemporaryDirectory() as directory,
            ):
                marker = Path(directory) / "cleanup.txt"
                master, slave = pty.openpty()
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", program, str(marker)],
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                )
                try:
                    output = b""
                    deadline_s = time.monotonic() + PROCESS_TIMEOUT_S
                    while b"calibrate> " not in output and time.monotonic() < deadline_s:
                        if select.select([master], [], [], POLL_INTERVAL_S)[0]:
                            output += os.read(master, 4096)
                    self.assertIn(b"calibrate> ", output)
                    self.assertFalse(termios.tcgetattr(slave)[3] & termios.ICANON)
                    process.send_signal(signal_number)
                    process.wait(timeout=PROCESS_TIMEOUT_S)
                    # macOS clears its automatic PENDIN flag when the next input arrives.
                    os.write(master, b"\n")
                    self.assertTrue(select.select([slave], [], [], PROCESS_TIMEOUT_S)[0])
                    self.assertEqual(os.read(slave, 1), b"\n")
                    baseline = Path(str(marker) + ".terminal").read_text()
                    self.assertEqual(repr(termios.tcgetattr(slave)), baseline)
                    self.assertEqual(marker.read_text(), "closed=True restored=True")
                    expected_status = 0 if signal_number == signal.SIGINT else 128 + signal_number
                    self.assertEqual(process.returncode, expected_status)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=PROCESS_TIMEOUT_S)
                    os.close(master)
                    os.close(slave)
