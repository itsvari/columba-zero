from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from app.calibrate import calibrate
from app.capture import EndOfStream
from app.config import AppConfig, ConfigError, TrackingConfig
from app.main import build_parser, run
from app.models import Detection, FramePacket
from app.output import RunOutput
from app.pantilt import PCA9685, AxisCalibration, Calibration
from app.tracking import TrackingController, TrackingSession

Path("outputs/test-tmp").mkdir(parents=True, exist_ok=True)


def bird(x=80, y=50, size=10, confidence=0.8):
    return Detection("pigeon", confidence, x - size / 2, y - size / 2,
                     x + size / 2, y + size / 2)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = TrackingController(TrackingConfig(), Calibration())

    def observe(self, detections, now=0):
        self.controller.observe(detections, 100, 100, now, now)

    def test_acquire_largest_and_keep_match_despite_larger_competitor(self):
        self.observe([bird(30, size=10), bird(80, size=20)])
        self.assertEqual(self.controller.target, bird(80, size=20))
        self.observe([bird(30, size=40), bird(78)], 0.1)
        self.assertEqual(self.controller.target, bird(78))

    def test_distance_gate_and_confidence_tiebreak(self):
        self.observe([bird(50)])
        self.observe([bird(45, confidence=0.7), bird(55, confidence=0.9)], 0.1)
        self.assertEqual(self.controller.target.confidence, 0.9)
        self.observe([bird(10)], 0.2)
        self.assertEqual(self.controller.state, "waiting")
        self.assertIsNone(self.controller.error)

    def test_invalid_detections_do_not_acquire(self):
        invalid = [bird(x=-1), bird(size=0), bird(size=float("nan")),
                   replace(bird(), label="person"), bird(confidence=float("inf")),
                   bird(confidence=0), bird(x=101)]
        self.observe(invalid)
        self.assertIsNone(self.controller.target)

    def test_deadband_gain_and_direction(self):
        self.observe([bird(54, 46)])
        self.assertEqual(self.controller.step(0.05, 0.05), (90, 90))
        self.observe([bird(75, 25)], 0.1)
        # Previous target is outside gate: no movement toward a different bird.
        self.assertEqual(self.controller.step(0.15, 0.05), (90, 90))
        self.controller = TrackingController(
            TrackingConfig(), replace(Calibration(), tilt=AxisCalibration(0, direction=-1)))
        self.observe([bird(75, 25)])
        self.assertEqual(self.controller.step(0.05, 0.05), (90.75, 90.75))

    def test_speed_and_travel_limits_and_scheduler_stall(self):
        self.controller = TrackingController(replace(TrackingConfig(), gain=1000), Calibration())
        self.observe([bird(95)])
        self.assertEqual(self.controller.step(0.05, 0.05), (91.5, 90))
        self.assertEqual(self.controller.step(0.1, 100), (94.5, 90))
        for i in range(200):
            self.observe([bird(95)], i * 0.05)
            self.controller.step(i * 0.05, 0.05)
        self.assertEqual(self.controller.pan, 120)

    def test_gap_stops_movement_then_return_at_two_seconds(self):
        self.observe([bird()])
        self.controller.step(0.05, 0.05)
        angle = self.controller.pan
        self.observe([], 0.1)
        self.assertEqual(self.controller.step(1.9, 0.05)[0], angle)
        self.controller.step(2, 0.05)
        self.assertIsNone(self.controller.target)
        self.assertEqual(self.controller.pan, 90)
        self.assertEqual(self.controller.state, "home")

    def test_reappearance_before_timeout_and_new_target_after_timeout(self):
        self.observe([bird(80)])
        self.observe([], 0.1)
        self.observe([bird(79)], 1.9)
        self.assertEqual(self.controller.state, "tracking")
        self.controller.pan = 110
        self.controller.step(3.9, 0.05)
        self.assertEqual(self.controller.state, "returning")
        self.observe([bird(10)], 4)
        self.assertEqual(self.controller.target, bird(10))
        self.assertEqual(self.controller.state, "tracking")

    def test_stale_observation_and_inference_stall(self):
        self.observe([bird()])
        self.controller.step(0.05, 0.05)
        angle = self.controller.pan
        self.assertEqual(self.controller.step(1, 0.05)[0], angle)
        self.assertEqual(self.controller.state, "stale")
        self.controller.observe([bird()], 100, 100, 0.1, 1.2)
        self.assertEqual(self.controller.last_seen, 0)
        self.controller.step(2, 0.05)
        self.assertIsNone(self.controller.target)
        self.assertEqual(self.controller.pan, 90)


class HardwareTests(unittest.TestCase):
    def test_initialization_pulse_conversion_clamp_and_disable(self):
        bus = Mock()
        bus.read_byte_data.return_value = 0
        driver = PCA9685(Calibration(), bus_factory=Mock(return_value=bus))
        writes = bus.write_byte_data.call_args_list
        self.assertTrue(any(c.args == (0x40, 0xFE, 121) for c in writes))
        self.assertTrue(any(c.args == (0x40, 0x00, 0xA0) for c in writes))
        driver.pulse(1, 1500)
        bus.write_i2c_block_data.assert_called_with(0x40, 0x0A, [0, 0, 51, 1])
        driver.move(1000, -1000)
        blocks = bus.write_i2c_block_data.call_args_list[-2:]
        self.assertEqual(blocks[0].args[2], [0, 0, 85, 1])  # pan clamped to 120
        self.assertEqual(blocks[1].args[2], [0, 0, 17, 1])  # tilt clamped to 60
        driver.close()
        driver.close()
        bus.close.assert_called_once()
        self.assertEqual([c.args[1] for c in bus.write_byte_data.call_args_list[-2:]],
                         [0x0D, 0x09])
        self.assertFalse(any(0xFA <= c.args[1] <= 0xFD
                             for c in bus.write_byte_data.call_args_list))

    def test_startup_failure_attempts_both_channels_and_closes_bus(self):
        bus = Mock()
        bus.write_byte_data.side_effect = OSError("I2C failed")
        with self.assertRaisesRegex(RuntimeError, "enable I2C"):
            PCA9685(Calibration(), bus_factory=Mock(return_value=bus))
        bus.close.assert_called_once()
        self.assertEqual(bus.write_byte_data.call_count, 4)

    def test_shutdown_write_failure_still_closes_bus(self):
        bus = Mock()
        bus.read_byte_data.return_value = 0
        driver = PCA9685(Calibration(), bus_factory=Mock(return_value=bus))
        bus.write_byte_data.side_effect = OSError("disconnected")
        with self.assertRaises(OSError):
            driver.close()
        bus.close.assert_called_once()
        with self.assertRaises(ValueError):
            driver.pulse(15, 1500)

    def test_profile_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            path = Path(directory) / "profile.yaml"
            Calibration().save(path)
            self.assertEqual(Calibration.load(path), Calibration())
            for value in ({}, {"pan": {}, "tilt": {}}, [], {"pan": None, "tilt": {}}):
                path.write_text(yaml.safe_dump(value))
                with self.assertRaisesRegex(ConfigError, "app.calibrate"):
                    Calibration.load(path)
            for axis in (AxisCalibration(True), AxisCalibration(16),
                         AxisCalibration(1, direction=0), AxisCalibration(1, direction=True),
                         AxisCalibration(1, min_angle=100),
                         AxisCalibration(1, home_angle=float("nan")),
                         AxisCalibration(1, pulse_min_us=400)):
                with self.subTest(axis=axis), self.assertRaises(ConfigError):
                    axis.validate()
            with self.assertRaises(ConfigError):
                replace(Calibration(), tilt=AxisCalibration(1)).validate()

    def test_calibration_requires_measured_marks_and_saves(self):
        commands = iter(["save", "pan home", "pan -", "pan min", "pan +", "pan +",
                         "pan max", "pan direction -1", "tilt home", "tilt -", "tilt min",
                         "tilt +", "tilt +", "tilt max", "tilt direction 1", "save"])
        mount, output = Mock(), Mock()
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            path = Path(directory) / "calibration.yaml"
            self.assertTrue(calibrate(Calibration(), mount, str(path),
                                      read=lambda _: next(commands), write=output))
            result = Calibration.load(path)
            self.assertEqual(result.pan.direction, -1)
            self.assertEqual(result.pan.home_angle, 90)
            self.assertAlmostEqual(result.pan.min_angle, 88.2)
            self.assertEqual(mount.pulse.call_count, 8)
        self.assertIn("Record min", output.call_args_list[1].args[0])

    def test_calibration_refuses_envelope_overrun_and_quit_does_not_save(self):
        commands = iter([*(["pan +"] * 51), "quit"])
        mount, output = Mock(), Mock()
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            path = Path(directory) / "not-saved.yaml"
            self.assertFalse(calibrate(Calibration(), mount, str(path),
                                       read=lambda _: next(commands), write=output))
            self.assertFalse(path.exists())
            self.assertLessEqual(max(c.args[1] for c in mount.pulse.call_args_list), 2000)
        self.assertIn("movement refused", output.call_args.args[0])


class SessionTests(unittest.TestCase):
    def config(self):
        return TrackingConfig(calibration="outputs/not-a-real-calibration.yaml")

    def test_simulation_never_imports_smbus_or_opens_driver(self):
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "smbus2":
                raise AssertionError("simulation attempted a hardware import")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard), patch("app.tracking.PCA9685") as hw:
            session = TrackingSession(self.config(), "simulate", replay=True)
            session.start()
            first = session.submit([bird()], 100, 100, 0, 0)
            second = session.submit([bird()], 100, 100, 0, 0.5)
            self.assertEqual(first["commanded_pan_deg"], 90)
            self.assertGreater(second["commanded_pan_deg"], 90)
            session.close()
            hw.assert_not_called()

    def test_worker_error_disables_mount_and_propagates(self):
        session = TrackingSession(self.config(), "simulate")
        session.mount = Mock()
        session.mount.move.side_effect = OSError("I2C write")
        session._stop = Mock()
        session._stop.wait.return_value = False
        with patch("app.tracking.time.monotonic", side_effect=[0, 0.05]):
            session._run()
        session.mount.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "I2C write"):
            session.check()
        with self.assertRaisesRegex(RuntimeError, "I2C write"):
            session.close()

    def test_worker_uses_latest_observation_and_fixed_interval(self):
        session = TrackingSession(self.config(), "simulate")
        session.mount = Mock()
        with patch("app.tracking.time.monotonic", return_value=0):
            session.submit([bird(80)], 100, 100, 0)
            session.submit([bird(60)], 100, 100, 0)
        session._stop = Mock()
        session._stop.wait.side_effect = [False, True]
        with patch("app.tracking.time.monotonic", side_effect=[0, 0.05]):
            session._run()
        session.mount.move.assert_called_once()
        self.assertAlmostEqual(session.mount.move.call_args.args[0], 90.3)
        session._stop.wait.assert_called_with(0.05)

    def test_missing_hardware_profile_fails_before_driver(self):
        with patch("app.tracking.PCA9685") as hardware:
            with self.assertRaises(ConfigError):
                TrackingSession(self.config(), "hardware")
            hardware.assert_not_called()

    def test_real_worker_shutdown_disables_mount_and_joins(self):
        session = TrackingSession(self.config(), "simulate")
        session.mount = Mock()
        session.start()
        session.request_stop()
        session.close()
        self.assertFalse(session._thread.is_alive())
        session.mount.close.assert_called_once()

    def test_shutdown_failure_is_reported(self):
        session = TrackingSession(self.config(), "simulate")
        session.mount = Mock()
        session.mount.close.side_effect = OSError("PWM disable failed")
        session._stop = Mock()
        session._stop.wait.return_value = True
        session._run()
        with self.assertRaisesRegex(RuntimeError, "PWM disable failed"):
            session.close()


class IntegrationTests(unittest.TestCase):
    def test_replay_hardware_rejected_before_startup(self):
        with patch("app.main.UltralyticsDetector") as detector:
            with self.assertRaisesRegex(ConfigError, "recorded video"):
                run(build_parser().parse_args(["--video", "fake.avi", "--tracking", "hardware"]))
            detector.assert_not_called()

    def test_tracking_config_invalid_values(self):
        for key, value in (("gain", 0), ("max_speed", True), ("deadband", 0.5),
                           ("deadband", float("nan")), ("match_distance", 2),
                           ("loss_timeout_s", -1), ("observation_timeout_s", "1"),
                           ("calibration", "")):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                replace(AppConfig(), tracking=replace(TrackingConfig(), **{key: value})).validate()

    def test_runtime_modes_and_hardware_startup_failure_cleanup(self):
        for flags, mode in (([], "hardware"), (["--video", "fake.avi"], "simulate"),
                            (["--tracking", "off"], None)):
            source, detector, output, session = Mock(), Mock(), Mock(), Mock()
            source.get.side_effect = EndOfStream()
            with (patch("app.main.UltralyticsDetector", return_value=detector),
                  patch("app.main.RunOutput", return_value=output),
                  patch("app.main.VideoFileCapture", return_value=source),
                  patch("app.main.LatestFrameStream", return_value=source),
                  patch("app.main.TrackingSession", return_value=session) as factory):
                self.assertEqual(run(build_parser().parse_args(flags)), 0)
                if mode:
                    self.assertEqual(factory.call_args.args[1], mode)
                    session.close.assert_called_once()
                else:
                    factory.assert_not_called()
            source.close.assert_called_once()
        source, output, session = Mock(), Mock(), Mock()
        session.start.side_effect = OSError("servo disconnected")
        with (patch("app.main.UltralyticsDetector"),
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.LatestFrameStream", return_value=source),
              patch("app.main.TrackingSession", return_value=session)):
            self.assertEqual(run(build_parser().parse_args([])), 1)
        session.close.assert_called_once()
        source.close.assert_called_once()
        output.close.assert_called_once()

    def test_overlay_and_tracking_jsonl(self):
        state = {"mode": "simulate", "state": "tracking", "selected_target": bird().as_dict(),
                 "commanded_pan_deg": 90, "commanded_tilt_deg": 90}
        import cv2
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            config = replace(AppConfig(), output=replace(AppConfig().output, directory=directory))
            output = RunOutput(config)
            with (patch.object(cv2, "imshow") as show,
                  patch.object(cv2, "getWindowProperty", return_value=1),
                  patch.object(cv2, "waitKey", return_value=0)):
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                self.assertTrue(output.render(frame, [bird()], 10, 15, state))
                self.assertTrue(show.call_args.args[1].any())
                self.assertFalse(frame.any())
            output.event("frame", tracking=state)
            output.close()
            data = json.loads((output.directory / "detections.jsonl").read_text())
            self.assertEqual(data["tracking"], state)

    def test_replay_frame_record_includes_tracking(self):
        source, output = Mock(), Mock()
        source.get.side_effect = [FramePacket(0, 0, np.zeros((100, 100, 3))), EndOfStream()]
        source.fps, source.dropped_frames = 15, 0
        with (patch("app.main.UltralyticsDetector") as detector,
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.VideoFileCapture", return_value=source)):
            detector.return_value.detect.return_value = [bird()]
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 0)
        frames = [c.kwargs for c in output.event.call_args_list if c.args[0] == "frame"]
        self.assertEqual(frames[0]["tracking"]["state"], "tracking")

    def test_runtime_uses_monotonic_frame_time_and_cleans_up_tracking_failure(self):
        source, output, session = Mock(), Mock(), Mock()
        source.get.return_value = FramePacket(0, -1000, np.zeros((100, 100, 3)), 123)
        source.dropped_frames = 0
        session.submit.side_effect = RuntimeError("I2C worker failed")
        # Use the real class for isinstance while replacing just its operations.
        with (patch("app.main.UltralyticsDetector") as detector,
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.Picamera2Capture"),
              patch("app.main.LatestFrameStream.start"),
              patch("app.main.LatestFrameStream.get", side_effect=source.get),
              patch("app.main.LatestFrameStream.close") as close,
              patch("app.main.TrackingSession", return_value=session)):
            detector.return_value.detect.return_value = [bird()]
            self.assertEqual(run(build_parser().parse_args([])), 1)
        self.assertEqual(session.submit.call_args.args[3], 123)
        session.close.assert_called_once()
        close.assert_called_once()
        output.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
