from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from app.capture import (
    CaptureError,
    EndOfStream,
    LatestFrameStream,
    Picamera2Capture,
    VideoFileCapture,
)
from app.config import AppConfig, ConfigError, DisplayConfig, OutputConfig
from app.detector import DetectorError, UltralyticsDetector, check_model, validate_names
from app.main import build_parser, run
from app.models import Detection, FramePacket
from app.output import RunOutput

Path("outputs/test-tmp").mkdir(parents=True, exist_ok=True)


class ConfigTests(unittest.TestCase):
    def test_default_config(self):
        config = AppConfig.load("app/config.yaml")
        self.assertEqual(config.model.image_size, 640)
        self.assertEqual(config.model.confidence, 0.5)
        self.assertTrue(config.display.enabled)

    def test_obsolete_and_malformed_config_rejected(self):
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            path = Path(directory) / "config.yaml"
            for value in ({"actuator": {}}, {"unknown": {}}, {"camera": {"oops": 1}},
                          {"display": {"enabled": "false"}},
                          {"model": {"confidence": float("nan")}},
                          {"camera": {"fps": 0}}, {"model": {"image_size": 641}}, []):
                path.write_text(yaml.safe_dump(value))
                with self.subTest(value=value), self.assertRaises(ConfigError):
                    AppConfig.load(path)

    def test_old_cli_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--hardware"])


class DetectorTests(unittest.TestCase):
    def test_missing_model_checked_before_import(self):
        with self.assertRaisesRegex(DetectorError, "does not exist"):
            UltralyticsDetector("not-present.pt", 640, 0.5)

    def test_export_size_and_files(self):
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            path = Path(directory)
            with self.assertRaisesRegex(DetectorError, "incomplete"):
                check_model(str(path), 640)
            for name in ("model.ncnn.bin", "model.ncnn.param"):
                (path / name).write_bytes(b"test")
            (path / "metadata.yaml").write_text(yaml.safe_dump(
                {"task": "detect", "names": {0: "pigeon"}, "imgsz": [640, 640]}))
            self.assertEqual(check_model(str(path), 640), path)
            with self.assertRaisesRegex(DetectorError, "export size"):
                check_model(str(path), 416)

    def test_class_mapping(self):
        validate_names({0: "pigeon"})
        for names in ({0: "bird"}, {0: "pigeon", 1: "person"}, {1: "pigeon"}, None):
            with self.subTest(names=names), self.assertRaises(DetectorError):
                validate_names(names)

    def test_filter_and_warmup(self):
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.image_size, detector.confidence = 640, 0.5
        detector._model = Mock()
        def box(conf):
            result = Mock()
            result.cls = Mock(item=lambda: 0)
            result.conf = Mock(item=lambda: conf)
            result.xyxy = [Mock(tolist=lambda: [1, 2, 3, 4])]
            return result
        detector._model.predict.return_value = [
            Mock(names={0: "pigeon"}, boxes=[box(0.4), box(0.8)])]
        result = detector.detect(FramePacket(0, 0, np.zeros((480, 640, 3), np.uint8)))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.8)
        detector.warmup(640, 480)
        kwargs = detector._model.predict.call_args.kwargs
        self.assertEqual(kwargs["source"].shape, (480, 640, 3))
        self.assertFalse(kwargs["rect"])
        self.assertEqual(kwargs["device"], "cpu")
        detector._model.predict.side_effect = RuntimeError("bad inference")
        with self.assertRaisesRegex(DetectorError, "bad inference"):
            detector.warmup(640, 480)


class CaptureTests(unittest.TestCase):
    def test_video_sequential_completion_and_release(self):
        import cv2
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 2
        frame = np.zeros((32, 32, 3), np.uint8)
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

    def test_video_decode_failure_is_not_completion(self):
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

    def test_video_open_failure_cleanup(self):
        import cv2
        capture = Mock()
        capture.isOpened.return_value = False
        source = VideoFileCapture("missing.avi")
        with patch.object(cv2, "VideoCapture", return_value=capture):
            with self.assertRaises(CaptureError):
                source.start()
            source.close()
        capture.release.assert_called_once()

    def test_latest_buffer_and_shutdown(self):
        reached = threading.Event()
        released = threading.Event()
        class Camera:
            count = 0
            def start(self):
                pass
            def read(self):
                self.count += 1
                if self.count == 4:
                    reached.set()
                    released.wait(2)
                return self.count
            def close(self):
                released.set()
        source = LatestFrameStream(Camera())
        source.start()
        try:
            self.assertTrue(reached.wait(2))
            packet = source.get(1)
            self.assertEqual(packet.sequence, 2)
            self.assertEqual(source.dropped_frames, 2)
        finally:
            source.close()
        self.assertFalse(source._thread.is_alive())

    def test_capture_failure_and_timeout(self):
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

    def test_picamera_bgr_format_and_cleanup(self):
        camera = Mock()
        fake_module = Mock(Picamera2=Mock(return_value=camera))
        fake_controls = Mock()
        with patch.dict("sys.modules", {"picamera2": fake_module,
                                        "libcamera": Mock(controls=fake_controls)}):
            source = Picamera2Capture(640, 480, 15)
            source.start()
            self.assertEqual(camera.create_video_configuration.call_args.kwargs["main"]["format"],
                             "RGB888")
            self.assertEqual(
                camera.create_video_configuration.call_args.kwargs["controls"]["AfMode"],
                fake_controls.AfModeEnum.Continuous)
            source.close()
        camera.stop.assert_called_once()
        camera.close.assert_called_once()


class OutputTests(unittest.TestCase):
    def test_logs_and_unique_directories(self):
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            config = replace(AppConfig(), display=DisplayConfig(False),
                             output=OutputConfig(directory=directory))
            first, second = RunOutput(config), RunOutput(config)
            self.assertNotEqual(first.directory, second.directory)
            first.event("frame", frame_number=7, captured_at=12.5, inference_s=0.02,
                        detections=[Detection("pigeon", 0.8, 1, 2, 3, 4).as_dict()],
                        dropped_frames=2)
            first.close()
            second.close()
            record = json.loads((first.directory / "detections.jsonl").read_text())
            self.assertEqual(record["detections"][0]["bbox"], [1, 2, 3, 4])
            self.assertEqual(record["dropped_frames"], 2)

    def test_recording_failure_releases_writer(self):
        import cv2
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            config = replace(AppConfig(), display=DisplayConfig(False),
                             output=OutputConfig(directory=directory, record=True))
            output = RunOutput(config)
            writer = Mock()
            writer.isOpened.return_value = False
            with patch.object(cv2, "VideoWriter", return_value=writer):
                with self.assertRaisesRegex(RuntimeError, "video writer"):
                    output.render(np.zeros((32, 32, 3), np.uint8), [], 1, None)
                output.close()
            writer.release.assert_called_once()

    def test_q_and_window_close(self):
        import cv2
        with tempfile.TemporaryDirectory(dir="outputs/test-tmp") as directory:
            output = RunOutput(replace(AppConfig(), output=OutputConfig(directory=directory)))
            with (patch.object(cv2, "namedWindow"), patch.object(cv2, "imshow"),
                  patch.object(cv2, "destroyAllWindows") as destroy,
                  patch.object(cv2, "waitKey", return_value=ord("q")),
                  patch.object(cv2, "getWindowProperty", return_value=1)):
                output.start()
                self.assertFalse(output.render(np.zeros((32, 32, 3), np.uint8), [], 1, 15))
                with (patch.object(cv2, "waitKey", return_value=0),
                      patch.object(cv2, "getWindowProperty", return_value=0)):
                    self.assertFalse(output.render(np.zeros((32, 32, 3), np.uint8), [], 1, 15))
                output.close()
                destroy.assert_called_once()


class RuntimeTests(unittest.TestCase):
    def exercise(self, packets, failure=None, output_failure=False, max_frames=0):
        config = replace(AppConfig(), display=DisplayConfig(False))
        args = build_parser().parse_args(["--video", "fake.avi", "--max-frames", str(max_frames)])
        source, detector, output = Mock(), Mock(), Mock()
        source.get.side_effect = packets
        source.dropped_frames = 0
        detector.detect.side_effect = failure
        detector.detect.return_value = []
        output.render.return_value = True
        if output_failure:
            output.start.side_effect = RuntimeError("display failed")
        with (patch("app.main.AppConfig.load", return_value=config),
              patch("app.main.UltralyticsDetector", return_value=detector),
              patch("app.main.VideoFileCapture", return_value=source),
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.health_snapshot", return_value={})):
            result = run(args)
        return result, source, detector, output

    def test_replay_no_dropped_frames_and_clean_completion(self):
        packets = [FramePacket(i, 10+i, np.zeros((32,32,3), np.uint8)) for i in range(3)]
        result, source, detector, output = self.exercise([*packets, EndOfStream()])
        self.assertEqual(result, 0)
        self.assertEqual(detector.detect.call_count, 3)
        source.close.assert_called_once()
        output.close.assert_called_once()
        events = output.event.call_args_list
        frames = [c.kwargs for c in events if c.args[0] == "frame"]
        self.assertEqual([r["frame_number"] for r in frames], [0,1,2])
        self.assertEqual(events[-1].kwargs["reason"], "video_complete")

    def test_inference_and_capture_failure_cleanup(self):
        packet = FramePacket(0, 0, None)
        for packets, error in (([packet], DetectorError("bad")), ([CaptureError("camera")], None)):
            result, source, _, output = self.exercise(packets, error)
            self.assertEqual(result, 1)
            source.close.assert_called_once()
            output.close.assert_called_once()

    def test_startup_output_failure_cleanup(self):
        result, source, _, output = self.exercise([], output_failure=True)
        self.assertEqual(result, 1)
        source.start.assert_not_called()
        output.close.assert_called_once()

    def test_max_frames(self):
        result, source, _, output = self.exercise([FramePacket(0, 0, None)], max_frames=1)
        self.assertEqual(result, 0)
        self.assertEqual(source.get.call_count, 1)
        self.assertEqual(output.event.call_args.kwargs["reason"], "max_frames")

    def test_warmup_failure_before_capture_and_outputs(self):
        args = build_parser().parse_args([])
        with (patch("app.main.UltralyticsDetector") as detector,
              patch("app.main.Picamera2Capture") as camera,
              patch("app.main.RunOutput") as output):
            detector.return_value.warmup.side_effect = DetectorError("warmup")
            self.assertEqual(run(args), 1)
        camera.assert_not_called()
        output.assert_not_called()

    def test_sigterm_restores_handlers_and_closes_resources(self):
        import signal
        installed = {}
        original = object()
        def install(sig, handler):
            previous = installed.get(sig, original)
            installed[sig] = handler
            return previous
        source, detector, output = Mock(), Mock(), Mock()
        source.get.return_value = FramePacket(0, 0, None)
        source.dropped_frames = 0
        output.render.return_value = True
        def detect(_packet):
            installed[signal.SIGTERM](signal.SIGTERM, None)
            return []
        detector.detect.side_effect = detect
        with (patch("app.main.signal.signal", side_effect=install),
              patch("app.main.UltralyticsDetector", return_value=detector),
              patch("app.main.VideoFileCapture", return_value=source),
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.health_snapshot", return_value={})):
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 0)
        source.close.assert_called_once()
        output.close.assert_called_once()
        self.assertEqual(source.get.call_count, 1)
        self.assertIs(installed[signal.SIGTERM], original)
        self.assertIs(installed[signal.SIGINT], original)

    def test_cli_overrides_and_invalid_frame_limit(self):
        args = build_parser().parse_args(["--model", "override.pt", "--confidence", "0.7",
                                          "--record", "--max-frames", "1"])
        source, output = Mock(), Mock()
        source.get.return_value = FramePacket(0, 0, None)
        source.dropped_frames, source.capture_fps = 0, 15.0
        with (patch("app.main.UltralyticsDetector") as detector,
              patch("app.main.RunOutput", return_value=output) as factory,
              patch("app.main.health_snapshot", return_value={})):
            detector.return_value.detect.return_value = []
            # Use video source so isinstance checks still refer to a real type.
            args.video = "fake.avi"
            with patch("app.main.VideoFileCapture", return_value=source):
                self.assertEqual(run(args), 0)
            detector.assert_called_once_with("override.pt", 640, 0.7)
            self.assertTrue(factory.call_args.args[0].output.record)
        args.max_frames = -1
        with self.assertRaises(ConfigError):
            run(args)

    def test_source_cleanup_failure_still_closes_outputs(self):
        source, output = Mock(), Mock()
        source.get.side_effect = EndOfStream()
        source.close.side_effect = RuntimeError("release failed")
        with (patch("app.main.UltralyticsDetector"),
              patch("app.main.VideoFileCapture", return_value=source),
              patch("app.main.RunOutput", return_value=output),
              patch("app.main.health_snapshot", return_value={})):
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 1)
        output.close.assert_called_once()
        self.assertEqual(output.event.call_args.kwargs["reason"], "cleanup_error")

    def test_no_hardware_imports(self):
        import ast
        forbidden = {"gpiozero", "RPi", "pigpio", "lgpio"}
        for file in Path("app").glob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse({n.name.split(".")[0] for n in node.names} & forbidden)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden)


if __name__ == "__main__":
    unittest.main()
