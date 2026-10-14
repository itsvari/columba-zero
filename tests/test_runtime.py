from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import FrameType, ModuleType
from typing import cast
from unittest.mock import Mock, patch

import numpy as numpy
import yaml

from app.capture import (
    CaptureError,
    EndOfStream,
    LatestFrameStream,
    Picamera2Capture,
    VideoFileCapture,
)
from app.config import AppConfig, ConfigError, DisplayConfig, ModelConfig, OutputConfig
from app.detector import (
    DetectorError,
    UltralyticsDetector,
    check_model,
    load_model_info,
    validate_names,
)
from app.json_data import JsonValue
from app.main import build_parser, create_detector, run
from app.models import Detection, Frame, FramePacket
from app.output import RunOutput

TEST_FRAME_SIZE_PX = 32
COLOUR_CHANNEL_COUNT = 3
THREAD_TEST_TIMEOUT_S = 2.0


def create_test_frame() -> Frame:
    return numpy.zeros((TEST_FRAME_SIZE_PX, TEST_FRAME_SIZE_PX, COLOUR_CHANNEL_COUNT), numpy.uint8)


def write_package(
    path: Path,
    names: dict[int, str] | None = None,
    image_size: list[int] | None = None,
    columba: dict[str, JsonValue] | None = None,
) -> Path:
    for name in ("model.ncnn.bin", "model.ncnn.param"):
        (path / name).write_bytes(b"test")
    metadata: dict[str, JsonValue] = {
        "task": "detect",
        "names": cast(JsonValue, names or {0: "pigeon"}),
        "imgsz": cast(JsonValue, image_size or [640, 640]),
    }
    if columba is not None:
        metadata["columba"] = columba
    (path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    return path


class ConfigTests(unittest.TestCase):
    def test_default_config(self) -> None:
        config = AppConfig.load("app/config.yaml")
        self.assertEqual(config.model.image_size, 640)
        self.assertEqual(config.model.confidence, 0.5)
        self.assertTrue(config.display.enabled)

    def test_obsolete_and_malformed_config_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            invalid_configurations: tuple[JsonValue, ...] = (
                {"actuator": {}},
                {"unknown": {}},
                {"camera": {"oops": 1}},
                {"display": {"enabled": "false"}},
                {"model": {"confidence": float("nan")}},
                {"camera": {"fps": 0}},
                {"model": {"image_size": 641}},
                {"model": {"image_size": [384, 650]}},
                {"model": {"image_size": [384]}},
                {"model": {"backend": "tensorrt"}},
                {"model": {"threads": 0}},
                {"tracking": {"enabled": True}},
                [],
            )
            for value in invalid_configurations:
                path.write_text(yaml.safe_dump(value))
                with self.subTest(value=value), self.assertRaises(ConfigError):
                    AppConfig.load(path)

    def test_rectangular_input_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"model": {"image_size": [384, 640], "backend": "native", "threads": 2}}
                )
            )
            config = AppConfig.load(path)
        self.assertEqual(config.model.image_size, (384, 640))
        self.assertEqual((config.model.backend, config.model.threads), ("native", 2))
        defaults = AppConfig.load("app/config.yaml").model
        self.assertEqual((defaults.backend, defaults.threads), ("ultralytics", 4))

    def test_old_cli_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--hardware"])


class DetectorTests(unittest.TestCase):
    def test_missing_model_checked_before_import(self) -> None:
        with self.assertRaisesRegex(DetectorError, "does not exist"):
            UltralyticsDetector("not-present.pt", 640, 0.5)

    def test_export_size_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(DetectorError, "incomplete"):
                check_model(str(path), 640)
            for name in ("model.ncnn.bin", "model.ncnn.param"):
                (path / name).write_bytes(b"test")
            (path / "metadata.yaml").write_text(
                yaml.safe_dump({"task": "detect", "names": {0: "pigeon"}, "imgsz": [640, 640]})
            )
            self.assertEqual(check_model(str(path), 640), path)
            with self.assertRaisesRegex(DetectorError, "export size"):
                check_model(str(path), 416)

    def test_rectangular_export_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_package(Path(directory), image_size=[384, 640])
            self.assertEqual(check_model(str(path), (384, 640)), path)
            for size in (640, (640, 384)):
                with self.subTest(size=size), self.assertRaisesRegex(DetectorError, "export size"):
                    check_model(str(path), size)

    def test_class_mapping(self) -> None:
        validate_names({0: "pigeon"})
        validate_names({0: "pigeon", 1: "spotted_dove", 2: "other_bird"})
        validate_names(["pigeon", "zebra_dove"])
        for names in (
            {0: "bird"},
            {0: "person", 1: "pigeon"},
            {1: "pigeon"},
            {0: "pigeon", 2: "other_bird"},
            {0: "pigeon", 1: "pigeon"},
            None,
        ):
            with self.subTest(names=names), self.assertRaises(DetectorError):
                validate_names(names)

    def test_package_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            legacy = load_model_info(str(write_package(path)), 640)
            self.assertEqual(
                (legacy.names, legacy.trigger_classes, legacy.decode),
                ({0: "pigeon"}, ("pigeon",), "yolo11"),
            )
            species = {0: "pigeon", 1: "spotted_dove", 2: "other_bird"}
            write_package(
                path,
                species,
                [384, 640],
                {"decode": "yolo26", "input_hw": [384, 640], "trigger_classes": ["pigeon"]},
            )
            info = load_model_info(str(path), (384, 640))
            self.assertEqual(
                (info.names, info.decode, info.input_hw), (species, "yolo26", (384, 640))
            )
            invalid_metadata: tuple[tuple[dict[str, JsonValue], str], ...] = (
                ({"decode": "magic"}, "decode"),
                ({"trigger_classes": ["hawk"]}, "trigger classes"),
                ({"trigger_classes": []}, "trigger_classes"),
                ({"input_hw": [352, 608]}, "input_hw"),
            )
            for columba, message in invalid_metadata:
                write_package(path, species, [384, 640], columba)
                with self.subTest(columba=columba), self.assertRaisesRegex(DetectorError, message):
                    load_model_info(str(path), (384, 640))

    def test_multiclass_detections_and_stage_timings(self) -> None:
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.image_size, detector.confidence = (384, 640), 0.5
        prediction_model = Mock()
        detector._model = prediction_model

        def prediction_box(class_id: int, confidence: float) -> Mock:
            box = Mock()
            box.cls = Mock(item=lambda: class_id)
            box.conf = Mock(item=lambda: confidence)
            box.xyxy = [Mock(tolist=lambda: [1, 2, 3, 4])]
            return box

        prediction_model.predict.return_value = [
            Mock(
                names={0: "Pigeon", 1: "spotted_dove"},
                boxes=[prediction_box(1, 0.9), prediction_box(0, 0.7), prediction_box(1, 0.2)],
                speed={"preprocess": 1.5, "inference": 20.0, "postprocess": 0.5},
            )
        ]
        result = detector.detect(FramePacket(0, 0, create_test_frame()))
        self.assertEqual(
            [(item.label, item.class_id) for item in result], [("spotted_dove", 1), ("pigeon", 0)]
        )
        self.assertEqual(prediction_model.predict.call_args.kwargs["imgsz"], [384, 640])
        self.assertEqual(
            detector.last_stage_ms, {"preprocess": 1.5, "inference": 20.0, "postprocess": 0.5}
        )
        self.assertEqual(
            Detection("pigeon", 0.8, 1, 2, 3, 4, class_id=0, track_id=7).as_dict(),
            {
                "label": "pigeon",
                "confidence": 0.8,
                "bbox": [1, 2, 3, 4],
                "class_id": 0,
                "track_id": 7,
            },
        )

    def test_backend_factory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_package(
                Path(directory),
                {0: "pigeon", 1: "zebra_dove"},
                [384, 640],
                {"decode": "yolo11", "trigger_classes": ["pigeon"]},
            )
            model = ModelConfig(path=str(path), image_size=(384, 640), backend="native", threads=2)
            native_module = ModuleType("app.native_detector")
            native_class = Mock()
            vars(native_module)["NativeDetector"] = native_class
            with patch.dict("sys.modules", {"app.native_detector": native_module}):
                detector, trigger_classes = create_detector(model)
            native_class.assert_called_once_with(str(path), (384, 640), 0.5, threads=2)
            self.assertIs(detector, native_class.return_value)
            self.assertEqual(trigger_classes, ("pigeon",))
            with (
                patch.dict("sys.modules", {"app.native_detector": None}),
                self.assertRaisesRegex(DetectorError, "native is unavailable"),
            ):
                create_detector(model)
            with patch("app.main.UltralyticsDetector") as ultralytics:
                ultralytics.return_value.trigger_classes = ("pigeon",)
                create_detector(replace(model, backend="ultralytics"))
            ultralytics.assert_called_once_with(str(path), (384, 640), 0.5)

    def test_filter_and_warmup(self) -> None:
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.image_size, detector.confidence = 640, 0.5
        prediction_model = Mock()
        detector._model = prediction_model

        def prediction_box(confidence: float) -> Mock:
            result = Mock()
            result.cls = Mock(item=lambda: 0)
            result.conf = Mock(item=lambda: confidence)
            result.xyxy = [Mock(tolist=lambda: [1, 2, 3, 4])]
            return result

        prediction_model.predict.return_value = [
            Mock(names={0: "pigeon"}, boxes=[prediction_box(0.4), prediction_box(0.8)])
        ]
        result = detector.detect(FramePacket(0, 0, numpy.zeros((480, 640, 3), numpy.uint8)))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.8)
        detector.warmup(640, 480)
        prediction_arguments = prediction_model.predict.call_args.kwargs
        self.assertEqual(prediction_arguments["source"].shape, (480, 640, 3))
        self.assertFalse(prediction_arguments["rect"])
        self.assertEqual(prediction_arguments["device"], "cpu")
        prediction_model.predict.side_effect = RuntimeError("bad inference")
        with self.assertRaisesRegex(DetectorError, "bad inference"):
            detector.warmup(640, 480)


class CaptureTests(unittest.TestCase):
    def test_video_sequential_completion_and_release(self) -> None:
        import cv2

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 2
        frame = create_test_frame()
        capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
        source = VideoFileCapture("example.avi")
        with patch.object(cv2, "VideoCapture", return_value=capture):
            source.start()
            self.assertEqual([source.get(1).sequence, source.get(1).sequence], [0, 1])
            with self.assertRaises(EndOfStream):
                source.get(1)
            source.close()
        capture.release.assert_called_once()
        self.assertEqual(source.dropped_frames, 0)

    def test_video_decode_failure_is_not_completion(self) -> None:
        import cv2

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 10
        capture.read.return_value = (False, None)
        with patch.object(cv2, "VideoCapture", return_value=capture):
            source = VideoFileCapture("broken.avi")
            source.start()
            with self.assertRaisesRegex(CaptureError, "decoder failed"):
                source.get(1)
            source.close()

    def test_video_open_failure_cleanup(self) -> None:
        import cv2

        capture = Mock()
        capture.isOpened.return_value = False
        source = VideoFileCapture("missing.avi")
        with patch.object(cv2, "VideoCapture", return_value=capture):
            with self.assertRaises(CaptureError):
                source.start()
            source.close()
        capture.release.assert_called_once()

    def test_latest_buffer_and_shutdown(self) -> None:
        reached = threading.Event()
        released = threading.Event()

        class Camera:
            count = 0

            def start(self) -> None:
                pass

            def read(self) -> Frame:
                self.count += 1
                if self.count == 4:
                    reached.set()
                    released.wait(THREAD_TEST_TIMEOUT_S)
                return create_test_frame()

            def close(self) -> None:
                released.set()

        source = LatestFrameStream(Camera())
        source.start()
        try:
            self.assertTrue(reached.wait(THREAD_TEST_TIMEOUT_S))
            packet = source.get(1)
            self.assertEqual(packet.sequence, 2)
            self.assertEqual(source.dropped_frames, 2)
        finally:
            source.close()
        assert source._thread is not None
        self.assertFalse(source._thread.is_alive())

    def test_capture_failure_and_timeout(self) -> None:
        source = LatestFrameStream(Mock())
        with self.assertRaisesRegex(CaptureError, "no camera frame"):
            source.get(0.001)
        camera = Mock()
        camera.read.side_effect = RuntimeError("disconnected")
        source = LatestFrameStream(camera)
        source.start()
        try:
            with self.assertRaisesRegex(CaptureError, "disconnected"):
                source.get(1)
        finally:
            source.close()
        camera.close.assert_called_once()

    def test_picamera_bgr_format_and_cleanup(self) -> None:
        camera = Mock()
        camera.camera_controls = {"AfMode": ()}
        fake_module = Mock(Picamera2=Mock(return_value=camera))
        fake_controls = Mock()
        with patch.dict(
            "sys.modules", {"picamera2": fake_module, "libcamera": Mock(controls=fake_controls)}
        ):
            source = Picamera2Capture(640, 480, 15)
            source.start()
            self.assertEqual(
                camera.create_video_configuration.call_args.kwargs["main"]["format"], "RGB888"
            )
            self.assertEqual(
                camera.create_video_configuration.call_args.kwargs["controls"]["AfMode"],
                fake_controls.AfModeEnum.Continuous,
            )
            source.close()
        camera.stop.assert_called_once()
        camera.close.assert_called_once()

    def test_picamera_without_autofocus_omits_autofocus_control(self) -> None:
        camera = Mock()
        camera.camera_controls = {"FrameDurationLimits": ()}
        with patch.dict(
            "sys.modules",
            {"picamera2": Mock(Picamera2=Mock(return_value=camera)), "libcamera": Mock()},
        ):
            source = Picamera2Capture(640, 480, 15)
            try:
                source.start()
                requested_controls = camera.create_video_configuration.call_args.kwargs["controls"]
                self.assertEqual(requested_controls, {"FrameRate": 15})
            finally:
                source.close()


class OutputTests(unittest.TestCase):
    def test_logs_and_unique_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                AppConfig(), display=DisplayConfig(False), output=OutputConfig(directory=directory)
            )
            first, second = RunOutput(config), RunOutput(config)
            self.assertNotEqual(first.directory, second.directory)
            first.event(
                "frame",
                frame_number=7,
                captured_at=12.5,
                inference_s=0.02,
                detections=[Detection("pigeon", 0.8, 1, 2, 3, 4).as_dict()],
                dropped_frames=2,
            )
            first.close()
            second.close()
            record = json.loads((first.directory / "detections.jsonl").read_text())
            self.assertEqual(record["detections"][0]["bbox"], [1, 2, 3, 4])
            self.assertEqual(record["dropped_frames"], 2)

    def test_recording_failure_releases_writer(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                AppConfig(),
                display=DisplayConfig(False),
                output=OutputConfig(directory=directory, record=True),
            )
            output = RunOutput(config)
            writer = Mock()
            writer.isOpened.return_value = False
            with patch.object(cv2, "VideoWriter", return_value=writer):
                with self.assertRaisesRegex(RuntimeError, "video writer"):
                    output.render(create_test_frame(), [], 1, None)
                output.close()
            writer.release.assert_called_once()

    def test_q_and_window_close(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = RunOutput(replace(AppConfig(), output=OutputConfig(directory=directory)))
            with (
                patch.object(cv2, "namedWindow"),
                patch.object(cv2, "imshow"),
                patch.object(cv2, "destroyAllWindows") as destroy,
                patch.object(cv2, "waitKey", return_value=ord("q")),
                patch.object(cv2, "getWindowProperty", return_value=1),
            ):
                output.start()
                self.assertFalse(output.render(create_test_frame(), [], 1, 15))
                with (
                    patch.object(cv2, "waitKey", return_value=0),
                    patch.object(cv2, "getWindowProperty", return_value=0),
                ):
                    self.assertFalse(output.render(create_test_frame(), [], 1, 15))
                output.close()
                destroy.assert_called_once()


class RuntimeTests(unittest.TestCase):
    def exercise(
        self,
        packets: Sequence[FramePacket | Exception],
        failure: Exception | None = None,
        output_failure: bool = False,
        max_frames: int = 0,
    ) -> tuple[int, Mock, Mock, Mock]:
        config = replace(AppConfig(), display=DisplayConfig(False))
        arguments = build_parser().parse_args(
            ["--video", "fake.avi", "--max-frames", str(max_frames)]
        )
        source, detector, output = Mock(), Mock(), Mock()
        source.get.side_effect = packets
        source.dropped_frames = 0
        detector.detect.side_effect = failure
        detector.detect.return_value = []
        output.render.return_value = True
        if output_failure:
            output.start.side_effect = RuntimeError("display failed")
        with (
            patch("app.main.AppConfig.load", return_value=config),
            patch("app.main.UltralyticsDetector", return_value=detector),
            patch("app.main.VideoFileCapture", return_value=source),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            result = run(arguments)
        return result, source, detector, output

    def test_replay_no_dropped_frames_and_clean_completion(self) -> None:
        packets = [
            FramePacket(frame_index, 10 + frame_index, create_test_frame())
            for frame_index in range(3)
        ]
        result, source, detector, output = self.exercise([*packets, EndOfStream()])
        self.assertEqual(result, 0)
        self.assertEqual(detector.detect.call_count, 3)
        source.close.assert_called_once()
        output.close.assert_called_once()
        events = output.event.call_args_list
        frames = [event_call.kwargs for event_call in events if event_call.args[0] == "frame"]
        self.assertEqual([record["frame_number"] for record in frames], [0, 1, 2])
        self.assertEqual(events[-1].kwargs["reason"], "video_complete")

    def test_inference_and_capture_failure_cleanup(self) -> None:
        packet = FramePacket(0, 0, create_test_frame())
        for packets, error in (([packet], DetectorError("bad")), ([CaptureError("camera")], None)):
            result, source, _, output = self.exercise(packets, error)
            self.assertEqual(result, 1)
            source.close.assert_called_once()
            output.close.assert_called_once()

    def test_startup_output_failure_cleanup(self) -> None:
        result, source, _, output = self.exercise([], output_failure=True)
        self.assertEqual(result, 1)
        source.start.assert_not_called()
        output.close.assert_called_once()

    def test_max_frames(self) -> None:
        result, source, _, output = self.exercise(
            [FramePacket(0, 0, create_test_frame())], max_frames=1
        )
        self.assertEqual(result, 0)
        self.assertEqual(source.get.call_count, 1)
        self.assertEqual(output.event.call_args.kwargs["reason"], "max_frames")

    def test_warmup_failure_before_capture_and_outputs(self) -> None:
        arguments = build_parser().parse_args([])
        with (
            patch("app.main.UltralyticsDetector") as detector,
            patch("app.main.Picamera2Capture") as camera,
            patch("app.main.RunOutput") as output,
        ):
            detector.return_value.warmup.side_effect = DetectorError("warmup")
            self.assertEqual(run(arguments), 1)
        camera.assert_not_called()
        output.assert_not_called()

    def test_sigterm_restores_handlers_and_closes_resources(self) -> None:
        import signal

        installed: dict[int, Callable[[int, FrameType | None], None] | Mock] = {}
        original = Mock()

        def install(
            signal_number: int,
            handler: Callable[[int, FrameType | None], None] | Mock,
        ) -> Callable[[int, FrameType | None], None] | Mock:
            previous = installed.get(signal_number, original)
            installed[signal_number] = handler
            return previous

        source, detector, output = Mock(), Mock(), Mock()
        source.get.return_value = FramePacket(0, 0, create_test_frame())
        source.dropped_frames = 0
        output.render.return_value = True

        def detect(_packet: FramePacket) -> list[Detection]:
            installed[signal.SIGTERM](signal.SIGTERM, None)
            return []

        detector.detect.side_effect = detect
        with (
            patch("app.main.signal.signal", side_effect=install),
            patch("app.main.UltralyticsDetector", return_value=detector),
            patch("app.main.VideoFileCapture", return_value=source),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 0)
        source.close.assert_called_once()
        output.close.assert_called_once()
        self.assertEqual(source.get.call_count, 1)
        self.assertIs(installed[signal.SIGTERM], original)
        self.assertIs(installed[signal.SIGINT], original)

    def test_cli_overrides_and_invalid_frame_limit(self) -> None:
        arguments = build_parser().parse_args(
            ["--model", "override.pt", "--confidence", "0.7", "--record", "--max-frames", "1"]
        )
        source, output = Mock(), Mock()
        source.get.return_value = FramePacket(0, 0, create_test_frame())
        source.dropped_frames, source.capture_fps = 0, 15.0
        with (
            patch("app.main.UltralyticsDetector") as detector,
            patch("app.main.RunOutput", return_value=output) as factory,
            patch("app.main.health_snapshot", return_value={}),
        ):
            detector.return_value.detect.return_value = []
            # patching video construction leaves the live stream class available for type checks.
            arguments.video = "fake.avi"
            with patch("app.main.VideoFileCapture", return_value=source):
                self.assertEqual(run(arguments), 0)
            detector.assert_called_once_with("override.pt", 640, 0.7)
            self.assertTrue(factory.call_args.args[0].output.record)
        arguments.max_frames = -1
        with self.assertRaises(ConfigError):
            run(arguments)

    def test_source_cleanup_failure_still_closes_outputs(self) -> None:
        source, output = Mock(), Mock()
        source.get.side_effect = EndOfStream()
        source.close.side_effect = RuntimeError("release failed")
        with (
            patch("app.main.UltralyticsDetector"),
            patch("app.main.VideoFileCapture", return_value=source),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 1)
        output.close.assert_called_once()
        self.assertEqual(output.event.call_args.kwargs["reason"], "cleanup_error")

    def test_no_hardware_imports(self) -> None:
        import ast

        forbidden = {"gpiozero", "RPi", "pigpio", "lgpio"}
        for source_path in Path("app").glob("*.py"):
            tree = ast.parse(source_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        {imported_name.name.split(".")[0] for imported_name in node.names}
                        & forbidden
                    )
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden)


if __name__ == "__main__":
    unittest.main()
