from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass, replace
from io import StringIO
from itertools import product
from pathlib import Path
from typing import cast
from unittest.mock import Mock, call, patch

import numpy
import yaml

from app.capture import CAMERA_BUFFER_COUNT, CaptureError, Picamera2Capture
from app.config import (
    AppConfig,
    CameraConfig,
    CameraEntry,
    CameraRotation,
    ConfigError,
    rotation_degrees,
)
from app.json_data import JsonValue

CAMERA_WIDTH_PX = 640
CAMERA_HEIGHT_PX = 480
CAMERA_FPS = 15.0
MAIN_SIZE_PX = (1280, 960)
INVALID_ROTATIONS: tuple[JsonValue, ...] = (
    True,
    False,
    0.0,
    180.0,
    90,
    -180,
    360,
    "180",
    None,
    [],
    {},
)


@dataclass(frozen=True, slots=True)
class CameraTransform:
    hflip: bool = False
    vflip: bool = False


def load_configuration(configuration: dict[str, JsonValue]) -> AppConfig:
    with tempfile.TemporaryDirectory() as directory:
        configuration_path = Path(directory) / "config.yaml"
        configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
        return AppConfig.load(configuration_path)


@contextmanager
def camera_modules(camera: Mock, transform: Mock | None = None) -> Iterator[Mock]:
    camera_factory = Mock(return_value=camera)
    controls = Mock()
    libcamera = (
        Mock(spec=["controls"], controls=controls)
        if transform is None
        else Mock(spec=["controls", "Transform"], controls=controls, Transform=transform)
    )
    with (
        patch.dict(
            "sys.modules",
            {"picamera2": Mock(Picamera2=camera_factory), "libcamera": libcamera},
        ),
        redirect_stderr(StringIO()),
    ):
        yield camera_factory


def camera_mock() -> Mock:
    camera = Mock()
    camera.camera_controls = {}
    return camera


class CameraRotationConfigTests(unittest.TestCase):
    def test_defaults_and_shared_rotation_preserve_camera_overrides(self) -> None:
        baseline = load_configuration({})
        self.assertEqual(baseline.camera.rotation_deg, 0)
        configuration = load_configuration(
            {
                "camera": {
                    "rotation_deg": 180,
                    "stream": "lores",
                    "max_exposure_us": 30000,
                },
                "cameras": [
                    {"name": "wide", "role": "day", "camera_index": 0},
                    {
                        "name": "noir",
                        "role": "night",
                        "camera_index": 1,
                        "rotation_deg": 0,
                        "max_exposure_us": 16000,
                    },
                ],
            }
        )
        day_entry, night_entry = configuration.cameras
        self.assertIsNone(day_entry.rotation_deg)
        self.assertEqual(
            configuration.camera_for(day_entry),
            replace(configuration.camera, name="wide", camera_index=0),
        )
        self.assertEqual(
            configuration.camera_for(night_entry),
            replace(
                configuration.camera,
                name="noir",
                camera_index=1,
                rotation_deg=0,
                max_exposure_us=16000,
            ),
        )
        self.assertEqual(
            baseline.camera_for(replace(night_entry, rotation_deg=180)).rotation_deg, 180
        )
        self.assertEqual(configuration.camera.rotation_deg, 180)
        for rotation_deg in (0, 180):
            validated_rotation_deg = rotation_degrees(rotation_deg, "camera.rotation_deg")
            direct = AppConfig(camera=CameraConfig(rotation_deg=validated_rotation_deg))
            direct.validate()
            self.assertEqual(
                load_configuration({"camera": {"rotation_deg": rotation_deg}}).camera,
                direct.camera,
            )

    def test_invalid_shared_rotations_fail_in_yaml_and_direct_configuration(self) -> None:
        for invalid_rotation in INVALID_ROTATIONS:
            with self.subTest(rotation_deg=invalid_rotation):
                with self.assertRaisesRegex(ConfigError, r"camera\.rotation_deg"):
                    rotation_degrees(invalid_rotation, "camera.rotation_deg")
                with self.assertRaisesRegex(ConfigError, r"camera\.rotation_deg"):
                    load_configuration({"camera": {"rotation_deg": invalid_rotation}})
                direct = AppConfig(
                    camera=CameraConfig(rotation_deg=cast(CameraRotation, invalid_rotation))
                )
                with self.assertRaisesRegex(ConfigError, r"camera\.rotation_deg"):
                    direct.validate()

    def test_entry_rotation_validation_and_null_inheritance_match_optional_overrides(self) -> None:
        configuration = load_configuration(
            {
                "camera": {"rotation_deg": 180},
                "cameras": [
                    {"name": "wide", "role": "day", "rotation_deg": None},
                    {"name": "noir", "role": "night"},
                ],
            }
        )
        self.assertEqual(
            tuple(configuration.camera_for(entry).rotation_deg for entry in configuration.cameras),
            (180, 180),
        )
        for invalid_rotation in INVALID_ROTATIONS:
            if invalid_rotation is None:
                continue
            with self.subTest(rotation_deg=invalid_rotation):
                with self.assertRaisesRegex(ConfigError, r"cameras\[1\]\.rotation_deg"):
                    load_configuration(
                        {
                            "cameras": [
                                {"name": "wide", "role": "day"},
                                {"name": "noir", "role": "night", "rotation_deg": invalid_rotation},
                            ]
                        }
                    )
                direct = replace(
                    configuration,
                    cameras=(
                        CameraEntry("wide", "day"),
                        CameraEntry(
                            "noir", "night", rotation_deg=cast(CameraRotation, invalid_rotation)
                        ),
                    ),
                )
                with self.assertRaisesRegex(ConfigError, r"cameras\[1\]\.rotation_deg"):
                    direct.validate()


class CameraRotationCaptureTests(unittest.TestCase):
    def test_default_capture_needs_no_transform_and_keeps_configuration_arguments(self) -> None:
        for stream, fallback in (("main", False), ("lores", False), ("lores", True)):
            with self.subTest(stream=stream, fallback=fallback):
                camera = camera_mock()
                if fallback:
                    camera.configure.side_effect = [RuntimeError("RGB lores unavailable"), None]
                with camera_modules(camera) as camera_factory:
                    capture = Picamera2Capture(
                        CAMERA_WIDTH_PX,
                        CAMERA_HEIGHT_PX,
                        CAMERA_FPS,
                        stream=stream,
                        main_size=MAIN_SIZE_PX,
                    )
                    capture.start()
                    capture.close()
                camera_factory.assert_called_once_with(0)
                for index, configuration_call in enumerate(
                    camera.create_video_configuration.call_args_list
                ):
                    lores_options = (
                        {
                            "lores": {
                                "size": (CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX),
                                "format": "YUV420" if index else "RGB888",
                            }
                        }
                        if stream == "lores"
                        else {}
                    )
                    self.assertEqual(
                        configuration_call.kwargs,
                        {
                            "main": {
                                "size": MAIN_SIZE_PX
                                if stream == "lores"
                                else (CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX),
                                "format": "YUV420" if stream == "lores" else "RGB888",
                            },
                            "controls": {"FrameRate": CAMERA_FPS},
                            "buffer_count": CAMERA_BUFFER_COUNT,
                            "queue": False,
                            **lores_options,
                        },
                    )
                self.assertEqual(camera.create_video_configuration.call_count, 2 if fallback else 1)
                camera.camera_configuration.assert_not_called()
                camera.start.assert_called_once()

    def test_rotation_reaches_main_and_rgb_lores_without_changing_delivered_frames(self) -> None:
        frame = numpy.zeros((CAMERA_HEIGHT_PX, CAMERA_WIDTH_PX, 3), dtype=numpy.uint8)
        for stream in ("main", "lores"):
            with self.subTest(stream=stream):
                camera = camera_mock()
                request = camera.capture_request.return_value
                request.make_array.return_value = frame
                request.get_metadata.return_value = {"Lux": 50.0}
                transform = Mock(side_effect=CameraTransform)
                camera.camera_configuration.return_value = {
                    "transform": CameraTransform(hflip=True, vflip=True)
                }
                with camera_modules(camera, transform):
                    capture = Picamera2Capture(
                        CAMERA_WIDTH_PX,
                        CAMERA_HEIGHT_PX,
                        CAMERA_FPS,
                        stream=stream,
                        main_size=MAIN_SIZE_PX,
                        rotation_deg=180,
                    )
                    capture.start()
                    self.assertIs(capture.read(), frame)
                    capture.close()
                self.assertEqual(
                    transform.call_args_list,
                    [call(hflip=True, vflip=True), call(hflip=True, vflip=True)],
                )
                arguments = camera.create_video_configuration.call_args.kwargs
                self.assertEqual(arguments["transform"], CameraTransform(hflip=True, vflip=True))
                self.assertEqual(arguments[stream]["format"], "RGB888")
                camera.camera_configuration.assert_called_once_with()
                self.assertLess(
                    camera.mock_calls.index(
                        call.configure(camera.create_video_configuration.return_value)
                    ),
                    camera.mock_calls.index(call.camera_configuration()),
                )
                self.assertLess(
                    camera.mock_calls.index(call.camera_configuration()),
                    camera.mock_calls.index(call.start()),
                )
                camera.start.assert_called_once_with()
                request.make_array.assert_called_once_with(stream)
                request.release.assert_called_once()
                self.assertEqual(capture.last_metadata.lux, 50.0)
                self.assertEqual(capture.warnings, [])

    def test_yuv_fallback_retains_rotation_before_colour_conversion(self) -> None:
        import cv2

        camera = camera_mock()
        camera.configure.side_effect = [RuntimeError("RGB lores unavailable"), None]
        request = camera.capture_request.return_value
        raw_frame = numpy.zeros((CAMERA_HEIGHT_PX * 3 // 2, CAMERA_WIDTH_PX), dtype=numpy.uint8)
        converted_frame = numpy.zeros((CAMERA_HEIGHT_PX, CAMERA_WIDTH_PX, 3), dtype=numpy.uint8)
        request.make_array.return_value = raw_frame
        request.get_metadata.return_value = {}
        transform = Mock(side_effect=CameraTransform)
        camera.camera_configuration.return_value = {
            "transform": CameraTransform(hflip=True, vflip=True)
        }
        with (
            camera_modules(camera, transform),
            patch.object(cv2, "cvtColor", return_value=converted_frame) as convert_colour,
        ):
            capture = Picamera2Capture(
                CAMERA_WIDTH_PX,
                CAMERA_HEIGHT_PX,
                CAMERA_FPS,
                stream="lores",
                main_size=MAIN_SIZE_PX,
                rotation_deg=180,
            )
            capture.start()
            self.assertIs(capture.read(), converted_frame)
            capture.close()
        self.assertEqual(
            transform.call_args_list,
            [call(hflip=True, vflip=True), call(hflip=True, vflip=True)],
        )
        configuration_calls = camera.create_video_configuration.call_args_list
        self.assertEqual(len(configuration_calls), 2)
        for configuration_call, pixel_format in zip(
            configuration_calls, ("RGB888", "YUV420"), strict=True
        ):
            self.assertEqual(
                configuration_call.kwargs["transform"], CameraTransform(hflip=True, vflip=True)
            )
            self.assertEqual(configuration_call.kwargs["lores"]["format"], pixel_format)
        convert_colour.assert_called_once_with(raw_frame, cv2.COLOR_YUV420p2BGR)
        request.make_array.assert_called_once_with("lores")
        request.release.assert_called_once()
        self.assertEqual(len(capture.warnings), 1)

    def test_adjusted_transform_is_rejected_after_configuration_before_start(self) -> None:
        for stream, applied_transform in product(
            ("main", "lores"),
            (
                CameraTransform(),
                CameraTransform(hflip=True),
                CameraTransform(vflip=True),
            ),
        ):
            with self.subTest(stream=stream, applied_transform=applied_transform):
                camera = camera_mock()
                camera.camera_configuration.return_value = {"transform": applied_transform}
                transform = Mock(side_effect=CameraTransform)
                with camera_modules(camera, transform):
                    capture = Picamera2Capture(
                        CAMERA_WIDTH_PX,
                        CAMERA_HEIGHT_PX,
                        CAMERA_FPS,
                        stream=stream,
                        main_size=MAIN_SIZE_PX,
                        rotation_deg=180,
                    )
                    with self.assertRaisesRegex(CaptureError, "rotation_deg=180 transform"):
                        capture.start()
                    capture.close()
                camera.configure.assert_called_once()
                camera.camera_configuration.assert_called_once_with()
                camera.start.assert_not_called()
                self.assertEqual(capture.warnings, [])

    def test_missing_or_unreadable_applied_transform_never_starts_camera(self) -> None:
        for readback_failure in ("missing", "invalid", "unreadable"):
            with self.subTest(readback_failure=readback_failure):
                camera = camera_mock()
                camera.camera_configuration.return_value = (
                    None if readback_failure == "invalid" else {}
                )
                if readback_failure == "unreadable":
                    camera.camera_configuration.side_effect = RuntimeError("cannot read transform")
                transform = Mock(side_effect=CameraTransform)
                with camera_modules(camera, transform):
                    capture = Picamera2Capture(
                        CAMERA_WIDTH_PX,
                        CAMERA_HEIGHT_PX,
                        CAMERA_FPS,
                        rotation_deg=180,
                    )
                    with self.assertRaises(CaptureError):
                        capture.start()
                    capture.close()
                camera.configure.assert_called_once()
                camera.camera_configuration.assert_called_once_with()
                camera.start.assert_not_called()

    def test_missing_or_rejected_transform_never_starts_an_unrotated_camera(self) -> None:
        for stream, transform_available in (("main", False), ("main", True), ("lores", True)):
            with self.subTest(stream=stream, transform_available=transform_available):
                camera = camera_mock()
                camera.configure.side_effect = RuntimeError("transform rejected")
                transform = Mock() if transform_available else None
                with camera_modules(camera, transform):
                    capture = Picamera2Capture(
                        CAMERA_WIDTH_PX,
                        CAMERA_HEIGHT_PX,
                        CAMERA_FPS,
                        stream=stream,
                        main_size=MAIN_SIZE_PX,
                        rotation_deg=180,
                    )
                    with self.assertRaises(CaptureError):
                        capture.start()
                    capture.close()
                camera.start.assert_not_called()
                if transform is None:
                    camera.create_video_configuration.assert_not_called()
                    continue
                transform.assert_called_once_with(hflip=True, vflip=True)
                for configuration_call in camera.create_video_configuration.call_args_list:
                    self.assertIs(configuration_call.kwargs["transform"], transform.return_value)


if __name__ == "__main__":
    unittest.main()
