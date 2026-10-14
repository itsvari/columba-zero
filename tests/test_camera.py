from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy
import yaml

from app.capture import (
    CaptureError,
    LatestFrameStream,
    Picamera2Capture,
    cap_exposure,
    read_capture_metadata,
)
from app.config import AppConfig, CameraConfig, CameraSwitchConfig, ConfigError
from app.json_data import JsonValue
from app.main import create_camera, create_source
from app.models import Frame, FramePacket
from app.switching import LuxHysteresis, SwitchingCameraStream

THREAD_TEST_TIMEOUT_S = 2.0
TUNING: dict[str, object] = {
    "version": 2.0,
    "algorithms": [
        {"rpi.black_level": {"black_level": 4096}},
        {
            "rpi.agc": {
                "channels": [
                    {
                        "exposure_modes": {
                            "normal": {"shutter": [100, 10000, 30000, 60000], "gain": [1, 2, 4, 6]},
                            "short": {"shutter": [100, 5000, 10000, 20000], "gain": [1, 2, 4, 6]},
                        }
                    },
                    {"exposure_modes": {"normal": {"exposure": [100, 66666], "gain": [1, 8]}}},
                ]
            }
        },
    ],
}


def fake_camera(controls: Mapping[str, object] | None = None) -> Mock:
    camera = Mock()
    camera.camera_controls = controls if controls is not None else {}
    camera.sensor_resolution = (4608, 2592)
    return camera


@contextmanager
def picamera(camera: Mock, tuning: object = None) -> Iterator[tuple[Mock, Mock]]:
    picamera2 = Mock(return_value=camera)
    picamera2.global_camera_info.return_value = [
        {"Model": "imx708", "Num": 0},
        {"Model": "imx708_wide", "Num": 1},
    ]
    picamera2.load_tuning_file.return_value = copy.deepcopy(tuning)
    controls = Mock()
    with (
        patch.dict(
            "sys.modules",
            {"picamera2": Mock(Picamera2=picamera2), "libcamera": Mock(controls=controls)},
        ),
        redirect_stderr(StringIO()),
    ):
        yield picamera2, controls


class CameraConfigTests(unittest.TestCase):
    def load(self, value: JsonValue) -> AppConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(value))
            return AppConfig.load(path)

    def test_defaults_keep_previous_capture(self) -> None:
        camera = AppConfig.load("app/config.yaml").camera
        self.assertEqual((camera.width, camera.height, camera.fps), (640, 480, 15))
        self.assertEqual((camera.stream, camera.camera_index, camera.label), ("main", 0, "camera0"))
        self.assertIsNone(camera.lens_position)
        self.assertIsNone(camera.max_exposure_us)
        self.assertIsNone(camera.tuning_file)

    def test_camera_options(self) -> None:
        camera = self.load(
            {
                "camera": {
                    "width": 640,
                    "height": 360,
                    "stream": "lores",
                    "main_size": [2304, 1296],
                    "lens_position": 0.5,
                    "max_exposure_us": 16666,
                    "tuning_file": "imx708_noir.json",
                    "camera_index": 1,
                    "name": "noir",
                }
            }
        ).camera
        self.assertEqual(camera.main_size, (2304, 1296))
        self.assertEqual((camera.stream, camera.label, camera.camera_index), ("lores", "noir", 1))
        invalid_settings: tuple[dict[str, JsonValue], ...] = (
            {"stream": "raw"},
            {"main_size": [320, 180]},
            {"main_size": [1920]},
            {"lens_position": -1},
            {"max_exposure_us": 0},
            {"camera_index": -1},
            {"tuning_file": ""},
        )
        for invalid in invalid_settings:
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                self.load({"camera": {"width": 640, "height": 360, **invalid}})


class Picamera2CaptureTests(unittest.TestCase):
    def test_lores_stream_scales_full_field_of_view(self) -> None:
        camera = fake_camera()
        with picamera(camera):
            capture = create_camera(CameraConfig(width=640, height=360, stream="lores"))
            capture.start()
        arguments = camera.create_video_configuration.call_args.kwargs
        self.assertEqual(arguments["lores"], {"size": (640, 360), "format": "RGB888"})
        self.assertEqual(arguments["main"], {"size": (2304, 1296), "format": "YUV420"})
        self.assertEqual(capture.warnings, [])

    def test_lores_falls_back_to_yuv_and_converts(self) -> None:
        import cv2

        camera = fake_camera()
        camera.configure.side_effect = [RuntimeError("lores must be YUV420"), None]
        request = Mock()
        request.make_array.return_value = numpy.zeros((540, 640), numpy.uint8)
        request.get_metadata.return_value = {}
        camera.capture_request.return_value = request
        converted = numpy.zeros((360, 640, 3), numpy.uint8)
        with picamera(camera), patch.object(cv2, "cvtColor", return_value=converted) as convert:
            capture = Picamera2Capture(640, 360, 30, stream="lores", main_size=(1280, 720))
            capture.start()
            self.assertIs(capture.read(), converted)
        self.assertEqual(
            camera.create_video_configuration.call_args.kwargs["lores"]["format"], "YUV420"
        )
        self.assertEqual(convert.call_args.args[1], cv2.COLOR_YUV420p2BGR)
        request.make_array.assert_called_once_with("lores")
        self.assertEqual(len(capture.warnings), 1)

    def test_manual_focus_and_fallback(self) -> None:
        camera = fake_camera({"AfMode": (), "LensPosition": (0.0, 15.0, 1.0)})
        with picamera(camera) as (_, controls):
            Picamera2Capture(640, 480, 15, lens_position=32.0).start()
        requested = camera.create_video_configuration.call_args.kwargs["controls"]
        self.assertEqual(requested["AfMode"], controls.AfModeEnum.Manual)
        self.assertEqual(requested["LensPosition"], 15.0)
        for available, expected in (({"AfMode": ()}, "Continuous"), ({}, None)):
            camera = fake_camera(available)
            with self.subTest(available=available), picamera(camera) as (_, controls):
                capture = Picamera2Capture(640, 480, 15, lens_position=2.0)
                capture.start()
                requested = camera.create_video_configuration.call_args.kwargs["controls"]
                self.assertNotIn("LensPosition", requested)
                self.assertEqual(
                    requested.get("AfMode"),
                    None if expected is None else controls.AfModeEnum.Continuous,
                )
                self.assertIn("lens_position", capture.warnings[0])

    def test_exposure_cap_uses_the_sensor_tuning(self) -> None:
        camera = fake_camera()
        with picamera(camera, TUNING) as (picamera2, _):
            capture = Picamera2Capture(640, 480, 30, max_exposure_us=16666, camera_index=1)
            capture.start()
        picamera2.load_tuning_file.assert_called_once_with("imx708_wide.json")
        self.assertEqual(picamera2.call_args.args, (1,))
        tuning = picamera2.call_args.kwargs["tuning"]
        channels = tuning["algorithms"][1]["rpi.agc"]["channels"]
        self.assertEqual(
            channels[0]["exposure_modes"]["normal"]["shutter"], [100, 10000, 16666, 16666]
        )
        self.assertEqual(channels[1]["exposure_modes"]["normal"]["exposure"], [100, 16666])
        self.assertEqual(capture.warnings, [])

    def test_tuning_file_by_name_path_and_failure(self) -> None:
        camera = fake_camera()
        with picamera(camera, TUNING) as (picamera2, _):
            Picamera2Capture(640, 480, 15, tuning_file="imx708_noir.json").start()
        picamera2.load_tuning_file.assert_called_once_with("imx708_noir.json")
        self.assertEqual(picamera2.call_args.kwargs["tuning"], TUNING)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            path.write_text(json.dumps({"rpi.agc": {"exposure_modes": {"normal": {}}}}))
            with picamera(camera) as (picamera2, _):
                capture = Picamera2Capture(640, 480, 15, tuning_file=str(path), max_exposure_us=1)
                capture.start()
            picamera2.load_tuning_file.assert_not_called()
            self.assertIn("max_exposure_us not applied", capture.warnings[0])
        with picamera(camera) as (picamera2, _):
            picamera2.load_tuning_file.side_effect = FileNotFoundError("imx708_noir.json")
            capture = Picamera2Capture(640, 480, 15, tuning_file="imx708_noir.json")
            capture.start()
        self.assertEqual(picamera2.call_args.kwargs, {})
        self.assertIn("default tuning", capture.warnings[0])

    def test_cap_exposure_on_version_one_tuning(self) -> None:
        tuning: dict[str, object] = {"rpi.agc": {"exposure_modes": {"normal": {"shutter": [9e4]}}}}
        self.assertTrue(cap_exposure(tuning, 1000))
        self.assertEqual(tuning, {"rpi.agc": {"exposure_modes": {"normal": {"shutter": [1000]}}}})
        self.assertFalse(cap_exposure({"algorithms": [{"rpi.awb": {}}]}, 1000))
        self.assertFalse(cap_exposure([], 1000))

    def test_read_releases_request_and_records_metadata(self) -> None:
        camera = fake_camera()
        request = Mock()
        frame = numpy.zeros((480, 640, 3), numpy.uint8)
        request.make_array.return_value = frame
        request.get_metadata.return_value = {
            "Lux": 3.5,
            "ExposureTime": 16000,
            "SensorTimestamp": 1_000_000_000,
        }
        camera.capture_request.return_value = request
        with (
            picamera(camera),
            patch("app.capture.sensor_clock_ns", return_value=1_045_000_000),
        ):
            capture = Picamera2Capture(640, 480, 15)
            capture.start()
            self.assertIs(capture.read(), frame)
        request.release.assert_called_once()
        metadata = capture.last_metadata
        self.assertEqual(
            (metadata.lux, metadata.delivery_ms, metadata.exposure_us), (3.5, 45, 16000)
        )
        request.make_array.side_effect = RuntimeError("dequeue failed")
        with picamera(camera):
            capture = Picamera2Capture(640, 480, 15)
            capture.start()
            with self.assertRaisesRegex(Exception, "dequeue failed"):
                capture.read()
        self.assertEqual(request.release.call_count, 2)
        for metadata_value in (None, {"SensorTimestamp": 2_000_000_000}, {"Lux": float("nan")}):
            parsed = read_capture_metadata(metadata_value, lambda: 1_000_000_000)
            self.assertIsNone(parsed.delivery_ms)
            self.assertIsNone(parsed.lux)

    def test_stream_packets_carry_camera_metadata(self) -> None:
        delivered = threading.Event()
        capture = Picamera2Capture(640, 480, 15, name="wide")

        def read() -> Frame:
            capture.last_metadata = read_capture_metadata(
                {"Lux": 120.0, "SensorTimestamp": 0}, lambda: 30_000_000
            )
            delivered.set()
            return numpy.zeros((4, 4, 3), numpy.uint8)

        with (
            patch.object(capture, "start"),
            patch.object(capture, "read", side_effect=read),
            patch.object(capture, "close"),
        ):
            source = LatestFrameStream(capture)
            source.start()
            try:
                self.assertTrue(delivered.wait(THREAD_TEST_TIMEOUT_S))
                packet = source.get(THREAD_TEST_TIMEOUT_S)
            finally:
                source.close()
        self.assertEqual((packet.camera, packet.lux, packet.delivery_ms), ("wide", 120.0, 30.0))
        self.assertIsNotNone(packet.delivered_monotonic_s)


def lux_packet(sequence: int, lux: float | None, camera: str = "wide") -> FramePacket:
    return FramePacket(
        sequence, float(sequence), numpy.zeros((4, 4, 3), numpy.uint8), 0.0, camera, lux
    )


def fake_stream(camera: str, lux: list[float | None]) -> Mock:
    stream = Mock(spec=LatestFrameStream)
    stream.get.side_effect = [lux_packet(index, value, camera) for index, value in enumerate(lux)]
    stream.captured_frames = len(lux)
    stream.dropped_frames = 1
    stream.capture_fps = 30.0
    return stream


class DayNightTests(unittest.TestCase):
    def test_lux_hysteresis_needs_a_sustained_change(self) -> None:
        hysteresis = LuxHysteresis(CameraSwitchConfig(10, 40, 5))
        self.assertEqual(
            [hysteresis.update("day", lux, now) for now, lux in ((0, 5), (4, 5), (5, 5))],
            [None, None, "night"],
        )
        # a reading between the thresholds, or no reading, restarts the hold time.
        steps = ((0, 5), (1, 20), (2, 5), (3, None), (4, 5), (8, 5), (9, 5))
        self.assertEqual(
            [hysteresis.update("day", lux, now) for now, lux in steps],
            [None, None, None, None, None, None, "night"],
        )
        self.assertIsNone(hysteresis.update("night", 30, 0))
        self.assertIsNone(hysteresis.update("night", 50, 10))
        self.assertEqual(hysteresis.update("night", 50, 15), "day")

    def test_switches_to_night_camera_and_back(self) -> None:
        now = [0.0]
        streams = {
            "wide": [fake_stream("wide", [100, 3, 3]), fake_stream("wide", [100])],
            "noir": [fake_stream("noir", [20, 60, 60])],
        }
        opened: list[str] = []

        def open_stream(camera: CameraConfig) -> LatestFrameStream:
            opened.append(camera.label)
            return cast(LatestFrameStream, streams[camera.label].pop(0))

        cameras = {
            "day": CameraConfig(name="wide", camera_index=0),
            "night": CameraConfig(name="noir", camera_index=1),
        }
        source = SwitchingCameraStream(
            cameras, CameraSwitchConfig(10, 40, 1), open_stream, lambda: now[0]
        )
        source.start()
        packets = []
        for _ in range(7):
            packets.append(source.get(1))
            now[0] += 1
        self.assertEqual(opened, ["wide", "noir", "wide"])
        self.assertEqual([packet.sequence for packet in packets], list(range(7)))
        self.assertEqual(
            [packet.camera for packet in packets], ["wide"] * 3 + ["noir"] * 3 + ["wide"]
        )
        events = source.take_events()
        self.assertEqual(
            [(event["from"], event["to"], event["lux"], event["reason"]) for event in events],
            [("wide", "noir", 3, "dark"), ("noir", "wide", 60, "bright")],
        )
        self.assertEqual(source.take_events(), [])
        self.assertEqual(source.dropped_frames, 3)
        source.close()

    def test_failed_switch_keeps_previous_camera(self) -> None:
        streams = [fake_stream("wide", [1, 1]), fake_stream("wide", [1])]
        broken = fake_stream("noir", [])
        broken.start.side_effect = CaptureError("noir unavailable")

        def open_stream(camera: CameraConfig) -> LatestFrameStream:
            return cast(LatestFrameStream, broken if camera.label == "noir" else streams.pop(0))

        cameras = {"day": CameraConfig(name="wide"), "night": CameraConfig(name="noir")}
        source = SwitchingCameraStream(
            cameras, CameraSwitchConfig(10, 40, 0), open_stream, lambda: 0
        )
        source.start()
        with redirect_stderr(StringIO()):
            source.get(1)
        broken.close.assert_called_once()
        (event,) = source.take_events()
        self.assertEqual((event["active"], event["error"]), ("wide", "noir unavailable"))
        self.assertEqual(source.role, "day")
        self.assertEqual(source.get(1).camera, "wide")

    def test_cameras_config(self) -> None:
        config = CameraConfigTests.load(
            CameraConfigTests(),
            {
                "camera": {
                    "width": 640,
                    "height": 360,
                    "stream": "lores",
                    "max_exposure_us": 30000,
                },
                "cameras": [
                    {"name": "wide", "role": "day", "camera_index": 0},
                    {
                        "name": "noir",
                        "role": "night",
                        "camera_index": 1,
                        "tuning_file": "imx708_noir.json",
                        "max_exposure_us": 16666,
                    },
                ],
                "camera_switch": {"night_below_lux": 5, "day_above_lux": 50, "hold_s": 10},
            },
        )
        night = config.camera_for(config.cameras[1])
        day = config.camera_for(config.cameras[0])
        self.assertEqual(
            (night.label, night.camera_index, night.max_exposure_us), ("noir", 1, 16666)
        )
        self.assertEqual((day.max_exposure_us, day.tuning_file, day.stream), (30000, None, "lores"))
        self.assertEqual(config.camera_switch, CameraSwitchConfig(5, 50, 10))
        self.assertIsInstance(create_source(config, None), SwitchingCameraStream)
        self.assertIsInstance(create_source(AppConfig(), None), LatestFrameStream)
        self.assertEqual(AppConfig.load("app/config.yaml").cameras, ())
        day_entry: dict[str, JsonValue] = {"name": "wide", "role": "day"}
        night_entry: dict[str, JsonValue] = {"name": "noir", "role": "night"}
        invalid_configurations: tuple[dict[str, JsonValue], ...] = (
            {"cameras": [day_entry]},
            {"cameras": [day_entry, {"name": "noir", "role": "day"}]},
            {"cameras": [day_entry, {"name": "wide", "role": "night"}]},
            {"cameras": [day_entry, {**night_entry, "fps": 10}]},
            {"cameras": {"day": day_entry}},
            {"camera_switch": {"hold_s": 1}},
            {
                "cameras": [day_entry, night_entry],
                "camera_switch": {"night_below_lux": 50, "day_above_lux": 40},
            },
            {"cameras": [day_entry, night_entry], "camera_switch": {"hold_s": -1}},
        )
        for invalid in invalid_configurations:
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                CameraConfigTests.load(CameraConfigTests(), invalid)


if __name__ == "__main__":
    unittest.main()
