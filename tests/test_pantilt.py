from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from app.config import ConfigError
from app.json_data import JsonValue
from app.pantilt import PCA9685, AxisCalibration, Calibration, PanTiltDriver


class HardwareTests(unittest.TestCase):
    def driver(self, bus: Mock) -> PCA9685:
        bus.read_byte_data.return_value = 0
        with patch("app.pantilt.time.sleep"):
            return PCA9685(Calibration(), bus_factory=Mock(return_value=bus))

    def test_initialisation_pulse_conversion_clamping_and_disable(self) -> None:
        bus = Mock()
        driver = self.driver(bus)
        writes = bus.write_byte_data.call_args_list
        self.assertTrue(any(call.args == (0x40, 0xFE, 121) for call in writes))
        self.assertTrue(any(call.args == (0x40, 0x00, 0xA0) for call in writes))
        driver.pulse(1, 1500)
        bus.write_i2c_block_data.assert_called_with(0x40, 0x0A, [0, 0, 51, 1])
        driver.move(pan_deg=1000, tilt_deg=-1000)
        blocks = bus.write_i2c_block_data.call_args_list[-2:]
        self.assertEqual(blocks[0].args[2], [0, 0, 119, 1])
        self.assertEqual(blocks[1].args[2], [0, 0, 239, 0])
        driver.close()
        driver.close()
        bus.close.assert_called_once()
        self.assertEqual(
            [call.args[1] for call in bus.write_byte_data.call_args_list[-2:]], [0x09, 0x0D]
        )
        self.assertFalse(
            any(0xFA <= call.args[1] <= 0xFD for call in bus.write_byte_data.call_args_list)
        )

    def test_hold_preserves_output_without_bus_writes(self) -> None:
        bus = Mock()
        driver: PanTiltDriver = self.driver(bus)
        driver.move(pan_deg=90, tilt_deg=90)
        bus.reset_mock()
        driver.hold()
        self.assertEqual(bus.mock_calls, [])
        driver.close()

    def test_startup_failure_attempts_both_channels_and_closes_bus(self) -> None:
        bus = Mock()
        bus.write_byte_data.side_effect = OSError("I2C failed")
        with self.assertRaisesRegex(RuntimeError, "enable I2C"):
            self.driver(bus)
        bus.close.assert_called_once()
        self.assertEqual(bus.write_byte_data.call_count, 4)

    def test_interrupted_initialisation_closes_bus(self) -> None:
        bus = Mock()
        bus.read_byte_data.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.driver(bus)
        bus.close.assert_called_once()
        self.assertEqual(bus.write_byte_data.call_count, 4)

    def test_shutdown_write_failure_still_closes_bus_and_attempts_both_axes(self) -> None:
        bus = Mock()
        driver = self.driver(bus)
        bus.reset_mock()
        first_error = OSError("pan disconnected")
        bus.write_byte_data.side_effect = [first_error, OSError("tilt disconnected")]
        with self.assertRaises(OSError) as failure:
            driver.close()
        self.assertIs(failure.exception, first_error)
        bus.close.assert_called_once()
        self.assertEqual(bus.write_byte_data.call_count, 2)
        driver.close()

    def test_invalid_commands_never_write_bus(self) -> None:
        bus = Mock()
        driver = self.driver(bus)
        bus.reset_mock()
        for pulse_us in (float("nan"), float("inf"), 499, 2501):
            with self.subTest(pulse_us=pulse_us), self.assertRaises(ValueError):
                driver.pulse(0, pulse_us)
        with self.assertRaises(ValueError):
            driver.pulse(15, 1500)
        for angle_deg in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(angle_deg=angle_deg), self.assertRaises(ValueError):
                driver.move(90, angle_deg)
        self.assertEqual(bus.mock_calls, [])
        driver.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            driver.move(90, 90)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            driver.hold()

    def test_invalid_profile_does_not_construct_bus(self) -> None:
        factory = Mock()
        with self.assertRaisesRegex(ConfigError, "tilt"):
            PCA9685(replace(Calibration(), tilt=AxisCalibration(16)), factory)
        factory.assert_not_called()


class ProfileTests(unittest.TestCase):
    def test_profile_roundtrip_and_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            Calibration().save(path)
            self.assertEqual(Calibration.load(path), Calibration())
            invalid_profiles: tuple[JsonValue, ...] = (
                {},
                {"pan": {}, "tilt": {}},
                [],
                {"pan": None, "tilt": {}},
            )
            for value in invalid_profiles:
                path.write_text(yaml.safe_dump(value), encoding="utf-8")
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ConfigError, "app.calibrate"),
                ):
                    Calibration.load(path)

    def test_axis_validation_labels_errors(self) -> None:
        invalid_axes = (
            AxisCalibration(True),
            AxisCalibration(16),
            AxisCalibration(1, direction=0),
            AxisCalibration(1, direction=True),
            AxisCalibration(1, min_angle=100),
            AxisCalibration(1, home_angle=float("nan")),
            AxisCalibration(1, pulse_min_us=400),
            AxisCalibration(1, min_angle=True),
            AxisCalibration(1, max_angle=181),
        )
        for axis in invalid_axes:
            for name in ("pan", "tilt"):
                profile = (
                    replace(Calibration(), pan=axis)
                    if name == "pan"
                    else replace(Calibration(), tilt=axis)
                )
                with self.subTest(axis=axis, name=name), self.assertRaisesRegex(ConfigError, name):
                    profile.validate()
        with self.assertRaises(ConfigError):
            replace(Calibration(), tilt=AxisCalibration(0)).validate()

    def test_yaml_scalar_types_and_unknown_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            for field, value in (("channel", True), ("direction", 1.0), ("min_angle", "60")):
                Calibration().save(path)
                contents = path.read_text(encoding="utf-8")
                original = "0" if field == "channel" else "1" if field == "direction" else "60"
                replacement = yaml.safe_dump(value).splitlines()[0]
                path.write_text(
                    contents.replace(f"{field}: {original}", f"{field}: {replacement}", 1),
                    encoding="utf-8",
                )
                with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "pan"):
                    Calibration.load(path)
            Calibration().save(path)
            with path.open("a", encoding="utf-8") as output:
                output.write("unknown: 1\n")
            with self.assertRaisesRegex(ConfigError, "unknown calibration"):
                Calibration.load(path)

    def test_nominal_pulse_mapping_is_independent_of_safety_envelope(self) -> None:
        for minimum_us, maximum_us in ((1000, 2000), (603.1, 2309.7), (500, 2500)):
            axis = AxisCalibration(0, pulse_min_us=minimum_us, pulse_max_us=maximum_us)
            minimum_deg = axis.angle_from_pulse(minimum_us)
            maximum_deg = axis.angle_from_pulse(maximum_us)
            axis = replace(axis, min_angle=minimum_deg, max_angle=maximum_deg)
            axis.validate("pan")
            self.assertAlmostEqual(minimum_deg, (minimum_us - 500) / 2000 * 180)
            self.assertAlmostEqual(maximum_deg, (maximum_us - 500) / 2000 * 180)
            for angle_deg in (minimum_deg, 60, 90, 120, maximum_deg):
                self.assertAlmostEqual(axis.angle_from_pulse(axis.pulse(angle_deg)), angle_deg)
            self.assertEqual(axis.pulse(minimum_deg), minimum_us)
            self.assertEqual(axis.pulse(maximum_deg), maximum_us)
            self.assertEqual(axis.pulse(90), 1500)
        full_travel = AxisCalibration(0, 500, 2500, 0, 90, 180)
        self.assertEqual(full_travel.angle_from_pulse(500), 0)
        self.assertEqual(full_travel.angle_from_pulse(2500), 180)
        for pulse_us in (999, 2001, float("nan")):
            with self.assertRaises(ValueError):
                AxisCalibration(0).angle_from_pulse(pulse_us)

    def test_angle_limits_outside_safety_envelope_are_rejected(self) -> None:
        for axis in (AxisCalibration(0, min_angle=0), AxisCalibration(0, max_angle=180)):
            with self.assertRaisesRegex(ConfigError, "pan angle limits.*pulse envelope"):
                axis.validate("pan")

    def test_interrupted_atomic_save_preserves_previous_profile_and_leaves_no_temporary_file(
        self,
    ) -> None:
        interruptions: tuple[tuple[str, BaseException], ...] = (
            ("app.pantilt.os.replace", OSError("replace failed")),
            ("app.pantilt.yaml.safe_dump", KeyboardInterrupt()),
        )
        for target, interruption in interruptions:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profile.yaml"
                Calibration().save(path)
                original = path.read_bytes()
                with (
                    self.subTest(target=target),
                    patch(target, side_effect=interruption),
                    self.assertRaises(type(interruption)),
                ):
                    replace(Calibration(), bus=2).save(path)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_save_leaves_existing_file_intact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            Calibration().save(path)
            with self.assertRaises(ConfigError):
                replace(Calibration(), tilt=AxisCalibration(0)).save(path)
            self.assertEqual(Calibration.load(path), Calibration())


class FieldOfViewTests(unittest.TestCase):
    def test_unknown_and_measured_field_of_view_roundtrip(self) -> None:
        for profile in (
            Calibration(),
            replace(Calibration(), horizontal_fov_deg=62.2, vertical_fov_deg=48.8),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profile.yaml"
                profile.save(path)
                self.assertEqual(Calibration.load(path), profile)

    def test_profiles_without_field_of_view_remain_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            Calibration().save(path)
            path.write_text(
                "\n".join(line for line in path.read_text().splitlines() if "fov_deg" not in line),
                encoding="utf-8",
            )
            profile = Calibration.load(path)
            self.assertIsNone(profile.horizontal_fov_deg)
            self.assertIsNone(profile.vertical_fov_deg)

    def test_invalid_measured_field_of_view_is_rejected(self) -> None:
        for value in (0, -1, 180, 181, float("nan"), float("inf"), True):
            for profile, name in (
                (replace(Calibration(), horizontal_fov_deg=value), "horizontal_fov_deg"),
                (replace(Calibration(), vertical_fov_deg=value), "vertical_fov_deg"),
            ):
                with (
                    self.subTest(value=value, name=name),
                    self.assertRaisesRegex(ConfigError, name),
                ):
                    profile.validate()


class PersistenceTests(unittest.TestCase):
    def test_atomic_save_preserves_existing_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            Calibration().save(path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o640)
            replace(Calibration(), bus=2).save(path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(Calibration.load(path).bus, 2)

    def test_close_preserves_disable_failure_when_bus_close_also_fails(self) -> None:
        bus = Mock()
        bus.read_byte_data.return_value = 0
        with patch("app.pantilt.time.sleep"):
            driver = PCA9685(Calibration(), lambda _: bus)
        first_failure = OSError("disable failed")
        bus.write_byte_data.side_effect = first_failure
        bus.close.side_effect = OSError("bus close failed")
        with self.assertRaises(OSError) as caught:
            driver.close()
        self.assertIs(caught.exception, first_failure)
        self.assertIn("bus close failed", caught.exception.__notes__[0])
        bus.close.assert_called_once()
        driver.close()

    def test_initialisation_failure_retains_cleanup_failure_note(self) -> None:
        bus = Mock()
        bus.read_byte_data.side_effect = OSError("initialisation failed")
        bus.close.side_effect = OSError("bus close failed")
        with self.assertRaises(RuntimeError) as caught:
            PCA9685(Calibration(), lambda _: bus)
        cause = caught.exception.__cause__
        self.assertIsNotNone(cause)
        assert cause is not None
        self.assertIn("initialisation failed", str(cause))
        self.assertIn("bus close failed", cause.__notes__[0])
