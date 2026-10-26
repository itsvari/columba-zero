from __future__ import annotations

import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import yaml

from app.config import AppConfig, ConfirmationConfig, FollowConfig, PanTiltConfig, PreviewConfig
from app.pantilt import Calibration
from tools.pi_preflight import (
    IMPORT_PROBE,
    PreflightCheck,
    mount_checks,
    platform_check,
    preflight,
    probe_module,
    probe_python,
)


class PiPreflightTests(unittest.TestCase):
    def test_platform_requires_pi5_and_a_64_bit_arm_interpreter(self) -> None:
        with (
            patch("tools.pi_preflight.Path.read_text", return_value="Raspberry Pi 5 Model B\x00"),
            patch("tools.pi_preflight.platform.system", return_value="Linux"),
            patch("tools.pi_preflight.platform.machine", return_value="aarch64"),
            patch("tools.pi_preflight.struct.calcsize", return_value=8),
        ):
            self.assertTrue(platform_check().passed)
            with patch("tools.pi_preflight.struct.calcsize", return_value=4):
                self.assertFalse(platform_check().passed)
            with patch("tools.pi_preflight.platform.machine", return_value="x86_64"):
                self.assertFalse(platform_check().passed)
            with patch("tools.pi_preflight.Path.read_text", return_value="Raspberry Pi 4 Model B"):
                self.assertFalse(platform_check().passed)

    def test_missing_pi_model_is_reported(self) -> None:
        with patch("tools.pi_preflight.Path.read_text", side_effect=FileNotFoundError("no Pi")):
            self.assertFalse(platform_check().passed)

    def test_import_uses_a_bounded_child_and_reports_a_broken_binary_dependency(self) -> None:
        process = subprocess.CompletedProcess[str]([], 1, "", "ImportError: missing library")
        with patch("tools.pi_preflight.subprocess.run", return_value=process) as run:
            check = probe_python("cv2", "import cv2")
        self.assertFalse(check.passed)
        self.assertIn("missing library", check.detail)
        self.assertIn("timeout", run.call_args.kwargs)
        self.assertEqual(run.call_args.args[0][1:], ("-c", "import cv2"))

    def test_stuck_import_is_reported_without_stopping_other_checks(self) -> None:
        with patch(
            "tools.pi_preflight.subprocess.run",
            side_effect=subprocess.TimeoutExpired("import", 20),
        ):
            self.assertFalse(probe_python("cv2", "import cv2").passed)

    def test_hardware_library_checks_do_not_execute_package_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("picamera2", "libcamera", "smbus2"):
                Path(directory, f"{name}.py").write_text(
                    "raise RuntimeError('package code was executed')\n", encoding="utf-8"
                )
            with patch.dict("os.environ", {"PYTHONPATH": directory}):
                for name in ("picamera2", "libcamera", "smbus2"):
                    with self.subTest(module=name):
                        availability = probe_module(name)
                        self.assertTrue(availability.passed, availability.detail)
                        self.assertIn("runtime importability unverified", availability.detail)
                        imported = probe_python(name, IMPORT_PROBE, name)
                        self.assertFalse(imported.passed)
                        self.assertIn("package code was executed", imported.detail)

    def test_disabled_following_and_simulation_do_not_require_calibration_or_i2c(self) -> None:
        for configuration in (
            AppConfig(pan_tilt=PanTiltConfig(mode="hardware")),
            AppConfig(
                confirmation=ConfirmationConfig(enabled=True), follow=FollowConfig(enabled=True)
            ),
        ):
            with (
                self.subTest(configuration=configuration),
                tempfile.TemporaryDirectory() as directory,
                patch(
                    "tools.pi_preflight.platform_check", return_value=PreflightCheck("pi", True, "")
                ),
                patch(
                    "tools.pi_preflight.probe_python",
                    return_value=PreflightCheck("import", True, ""),
                ),
                patch(
                    "tools.pi_preflight.executable_check",
                    return_value=PreflightCheck("binary", True, ""),
                ),
                patch(
                    "tools.pi_preflight.Calibration.load",
                    side_effect=AssertionError("hardware access"),
                ),
            ):
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(asdict(configuration)), encoding="utf-8")
                checks = preflight(path)
                self.assertTrue(all(check.passed for check in checks))

    def test_recording_requires_both_export_programmes(self) -> None:
        configuration = replace(AppConfig(), preview=PreviewConfig(record=True))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("tools.pi_preflight.platform_check", return_value=PreflightCheck("pi", True, "")),
            patch(
                "tools.pi_preflight.probe_python", return_value=PreflightCheck("import", True, "")
            ),
            patch(
                "tools.pi_preflight.executable_check",
                return_value=PreflightCheck("binary", False, ""),
            ) as executable,
        ):
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(asdict(configuration)), encoding="utf-8")
            checks = preflight(path)
        self.assertFalse(all(check.passed for check in checks))
        self.assertEqual(
            [call.args[0] for call in executable.call_args_list],
            ["rpicam-hello", "ffmpeg", "ffprobe"],
        )

    def test_invalid_configuration_becomes_a_failed_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("follow: {enabled: not-a-boolean}\n", encoding="utf-8")
            checks = preflight(path)
        self.assertEqual(checks[-1].name, "configuration")
        self.assertFalse(checks[-1].passed)

    def test_missing_calibration_is_reported_without_bus_access(self) -> None:
        configuration = replace(AppConfig(), pan_tilt=PanTiltConfig(calibration="/missing/profile"))
        with patch("tools.pi_preflight.os.access", side_effect=AssertionError("bus check")):
            checks = mount_checks(configuration)
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0].passed)

    def test_bus_access_requires_a_character_device_and_permission(self) -> None:
        for character_device, permission, expected in (
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ):
            with (
                self.subTest(character_device=character_device, permission=permission),
                patch("tools.pi_preflight.Calibration.load", return_value=Calibration(bus=3)),
                patch("tools.pi_preflight.Path.is_char_device", return_value=character_device),
                patch("tools.pi_preflight.os.access", return_value=permission),
            ):
                checks = mount_checks(AppConfig())
            self.assertEqual(checks[-1].passed, expected)
            self.assertIn("/dev/i2c-3", checks[-1].detail)

    def test_mismatched_capture_orientation_fails_before_the_bus_check(self) -> None:
        base = AppConfig()
        configuration = replace(base, camera=replace(base.camera, rotation_deg=180))
        with (
            patch("tools.pi_preflight.Calibration.load", return_value=Calibration()),
            patch("tools.pi_preflight.os.access", side_effect=AssertionError("bus check")),
        ):
            checks = mount_checks(configuration)
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0].passed)
        self.assertIn("camera.rotation_deg", checks[0].detail)
