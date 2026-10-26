from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import FrameType
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
from app.config import (
    AppConfig,
    ConfigError,
    ConfirmationConfig,
    DisplayConfig,
    ModelConfig,
    OutputConfig,
)
from app.detector import (
    DetectorError,
    FrameDetector,
    UltralyticsDetector,
    load_model_info,
    validate_names,
)
from app.json_data import JsonValue
from app.main import CommandOptions, build_parser, create_detector, load_configuration, run
from app.models import Detection, Frame, FramePacket
from app.output import RunOutput
from app.pipeline import FrameProcessor, OutputQueue, RenderJob
from app.reaction import ReactionAnalysis, ReactionPlan, ReactionTest, StimulusSchedule
from app.tracking import ConfirmationTracker
from tools.summarize_run import summarize_log

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
    def test_deployment_config_matches_the_selected_tracked_package(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        manifest = cast(
            dict[str, JsonValue],
            json.loads((repository / "deploy/model_manifest.json").read_text()),
        )
        config = AppConfig.load(repository / "app/config.yaml")
        threshold = cast(dict[str, JsonValue], manifest["operating_threshold"])
        self.assertEqual(config.model.path, manifest["runtime_directory"])
        self.assertEqual(config.model.image_size, tuple(cast(list[int], manifest["input_hw"])))
        self.assertEqual(config.model.confidence, threshold["value"])
        source = repository / cast(str, manifest["source"])
        self.assertEqual(load_model_info(str(source), config.model.image_size).path, source)

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
                {"confirmation": {"k": 4, "n": 3}},
                {"confirmation": {"iou": 1.5}},
                {"confirmation": {"max_age_frames": -1}},
                {"confirmation": {"window": 3}},
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


class DetectorTests(unittest.TestCase):
    def test_missing_model_checked_before_import(self) -> None:
        with self.assertRaisesRegex(DetectorError, "does not exist"):
            UltralyticsDetector("not-present.pt", 640, 0.5)

    def test_export_size_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(DetectorError, "incomplete"):
                load_model_info(str(path), 640)
            for name in ("model.ncnn.bin", "model.ncnn.param"):
                (path / name).write_bytes(b"test")
            (path / "metadata.yaml").write_text(
                yaml.safe_dump({"task": "detect", "names": {0: "pigeon"}, "imgsz": [640, 640]})
            )
            self.assertEqual(load_model_info(str(path), 640).path, path)
            with self.assertRaisesRegex(DetectorError, "export size"):
                load_model_info(str(path), 416)

    def test_rectangular_export_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_package(Path(directory), image_size=[384, 640])
            self.assertEqual(load_model_info(str(path), (384, 640)).path, path)
            for size in (640, (640, 384)):
                with self.subTest(size=size), self.assertRaisesRegex(DetectorError, "export size"):
                    load_model_info(str(path), size)

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

    def test_trigger_classes_limited_to_pigeon(self) -> None:
        species = {0: "pigeon", 1: "spotted_dove", 2: "other_bird"}
        with tempfile.TemporaryDirectory() as directory:
            trigger_sets: tuple[list[JsonValue], ...] = (
                ["pigeon", "spotted_dove"],
                ["spotted_dove"],
            )
            for trigger_classes in trigger_sets:
                path = write_package(
                    Path(directory), species, [384, 640], {"trigger_classes": trigger_classes}
                )
                model = ModelConfig(path=str(path), image_size=(384, 640), backend="native")
                with self.subTest(trigger_classes=trigger_classes):
                    with self.assertRaisesRegex(DetectorError, "may only trigger on"):
                        load_model_info(str(path), (384, 640))
                    with self.assertRaisesRegex(DetectorError, "may only trigger on"):
                        create_detector(model)

    def test_ultralytics_backend_rejects_nms_free_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_package(Path(directory), columba={"decode": "yolo26"})
            with (
                patch.dict("sys.modules", {"ultralytics": Mock()}),
                self.assertRaisesRegex(DetectorError, "model.backend: native"),
            ):
                UltralyticsDetector(str(path), 640, 0.5)

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
                ({"trigger_classes": ["hawk"]}, "may only trigger on"),
                ({"trigger_classes": []}, "trigger_classes"),
                ({"input_hw": [352, 608]}, "input_hw"),
            )
            for columba, message in invalid_metadata:
                write_package(path, species, [384, 640], columba)
                with self.subTest(columba=columba), self.assertRaisesRegex(DetectorError, message):
                    load_model_info(str(path), (384, 640))

    def test_package_directories_are_named_for_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, linked in (("yolo26n-v2s", True), ("best_ncnn_model", False)):
                package = Path(directory) / name
                package.mkdir()
                detector = UltralyticsDetector.__new__(UltralyticsDetector)
                detector.model_info = load_model_info(str(write_package(package)), 640)
                path = detector.ultralytics_path()
                with self.subTest(name=name):
                    self.assertTrue(path.name.endswith("_ncnn_model"))
                    self.assertEqual(path.is_symlink(), linked)
                    self.assertEqual(path.resolve(), package.resolve())

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

    def test_native_backend_unavailable_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_package(Path(directory), image_size=[384, 640])
            model = ModelConfig(path=str(path), image_size=(384, 640), backend="native")
            with (
                patch.dict("sys.modules", {"app.native_detector": None}),
                self.assertRaisesRegex(DetectorError, "native is unavailable"),
            ):
                create_detector(model)

    def test_warmup_failure_is_reported_as_detector_error(self) -> None:
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.image_size, detector.confidence = 640, 0.5
        detector._model = Mock()
        detector._model.predict.side_effect = RuntimeError("bad inference")
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
                self.assertFalse(output.poll())
                self.assertFalse(output.render(create_test_frame(), [], 1, 15))
                with (
                    patch.object(cv2, "waitKey", return_value=0),
                    patch.object(cv2, "getWindowProperty", return_value=0),
                ):
                    self.assertFalse(output.render(create_test_frame(), [], 1, 15))
                output.close()
                destroy.assert_called_once()


def render_job(index: int) -> RenderJob:
    return RenderJob(create_test_frame(), [], float(index), None)


def rendered_fps(job: RenderJob | None) -> float | None:
    return None if job is None else job.processing_fps


class FastCamera:
    def start(self) -> None:
        pass

    def read(self) -> Frame:
        time.sleep(0.001)
        return create_test_frame()

    def close(self) -> None:
        pass


class OutputQueueTests(unittest.TestCase):
    def test_controller_samples_cannot_exhaust_lifecycle_event_capacity(self) -> None:
        stop = threading.Event()
        queue = OutputQueue(log_capacity=4, event_allowance=2, on_overflow=stop.set)
        queue.event("target_acquired", track_id=1)
        for sequence in range(10_000):
            queue.event("controller_state", commanded_pan_deg=sequence)
            queue.event("frame", frame_number=sequence)
        self.assertFalse(stop.is_set())
        self.assertIsNone(queue.overflow_error)
        records = queue.take_logs()
        self.assertLessEqual(len(records), queue.log_capacity)
        self.assertEqual(records[0].event, "target_acquired")
        self.assertGreater(queue.dropped_records, 0)

    def test_log_queue_is_bounded_but_keeps_events(self) -> None:
        queue = OutputQueue(log_capacity=4)
        for index in range(10):
            queue.event("frame", frame_number=index)
            if index == 2:
                queue.event("trigger", track_id=1)
            if index == 5:
                queue.event("camera_switch", to="noir")
        records = queue.take_logs()
        self.assertEqual(
            [(record.event, record.fields) for record in records],
            [
                ("trigger", {"track_id": 1}),
                ("camera_switch", {"to": "noir"}),
                ("frame", {"frame_number": 8}),
                ("frame", {"frame_number": 9}),
            ],
        )
        self.assertEqual(queue.dropped_records, 8)

    def test_event_only_backlog_drops_frames_and_fails_past_allowance(self) -> None:
        overflows: list[str] = []
        queue = OutputQueue(
            log_capacity=4, event_allowance=3, on_overflow=lambda: overflows.append("stop")
        )
        for track_id in range(5):
            queue.event("trigger", track_id=track_id)
        # no queued frame can make room, so the incoming frame is dropped.
        queue.event("frame", frame_number=0)
        self.assertEqual(queue.dropped_records, 1)
        self.assertIsNone(queue.overflow_error)
        for track_id in range(5, 100):
            queue.event("trigger", track_id=track_id)
        queue.event("frame", frame_number=1)
        self.assertEqual(overflows, ["stop"])
        assert queue.overflow_error is not None
        self.assertIn("log writer", queue.overflow_error)
        records = queue.take_logs()
        # events accepted before the overflow are kept, in order.
        self.assertEqual([record.fields["track_id"] for record in records], list(range(7)))
        self.assertEqual(queue.dropped_records, 2)

    def test_log_enqueue_cost_does_not_grow_with_backlog(self) -> None:
        def enqueue_seconds(backlog: int) -> float:
            queue = OutputQueue(log_capacity=4096, event_allowance=8192)
            for track_id in range(backlog):
                queue.event("trigger", track_id=track_id)
            started_s = time.perf_counter()
            for frame_number in range(2000):
                queue.event("frame", frame_number=frame_number)
            return time.perf_counter() - started_s

        # best of a few attempts; the Mac is shared with other jobs.
        empty = min(enqueue_seconds(0) for _ in range(3))
        saturated = min(enqueue_seconds(4096) for _ in range(3))
        self.assertLess(saturated, empty * 5)

    def test_live_queue_drops_oldest_frames_but_never_logs(self) -> None:
        queue = OutputQueue(render_capacity=2)
        for index in range(5):
            queue.event("frame", frame_number=index)
            queue.render(render_job(index))
        self.assertEqual(queue.dropped_renders, 3)
        records = queue.take_logs()
        self.assertEqual([record.fields["frame_number"] for record in records], list(range(5)))
        self.assertEqual(rendered_fps(queue.take_render()), 3.0)
        self.assertEqual(rendered_fps(queue.take_render()), 4.0)
        self.assertEqual((queue.take_logs(), queue.take_render()), ([], None))

    def test_replay_queue_waits_for_space_until_closed(self) -> None:
        queue = OutputQueue(render_capacity=1, drop_oldest=False)
        queue.render(render_job(0))
        submitted = threading.Event()

        def produce() -> None:
            queue.render(render_job(1))
            submitted.set()

        producer = threading.Thread(target=produce)
        producer.start()
        self.assertFalse(submitted.wait(0.05))
        self.assertEqual(rendered_fps(queue.take_render()), 0.0)
        self.assertTrue(submitted.wait(THREAD_TEST_TIMEOUT_S))
        producer.join()
        self.assertEqual(queue.dropped_renders, 0)
        self.assertEqual(rendered_fps(queue.take_render()), 1.0)
        queue.render(render_job(2))
        blocked = threading.Thread(target=queue.render, args=(render_job(3),))
        blocked.start()
        queue.close()
        blocked.join(THREAD_TEST_TIMEOUT_S)
        self.assertFalse(blocked.is_alive())
        self.assertIsNone(queue.take_render())


class ProcessorTests(unittest.TestCase):
    def test_camera_switch_events_and_tracker_reset(self) -> None:
        source, detector, tracker = Mock(), Mock(), Mock()
        packets = [
            FramePacket(0, 0, create_test_frame(), camera="wide"),
            FramePacket(1, 1, create_test_frame(), camera="wide"),
            FramePacket(2, 2, create_test_frame(), camera="noir"),
        ]
        source.get.side_effect = [*packets, EndOfStream()]
        source.dropped_frames = 0
        detector.detect.return_value = []
        tracker.update.return_value = Mock(detections=[], triggers=[])
        tracker.confirmed_track_ids = frozenset()
        switches: list[list[dict[str, JsonValue]]] = [
            [],
            [{"from": "wide", "to": "noir", "lux": 3.0, "reason": "dark"}],
            [],
        ]
        queue = OutputQueue()
        processor = FrameProcessor(
            replace(AppConfig(), display=DisplayConfig(False)),
            source,
            detector,
            tracker,
            queue,
            threading.Event(),
            time.monotonic(),
            camera_events=lambda: switches.pop(0),
        )
        processor.run()
        self.assertEqual(processor.result.reason, "video_complete")
        tracker.reset.assert_called_once()
        records = queue.take_logs()
        self.assertEqual(
            [record.event for record in records],
            ["frame", "frame", "camera_switch", "frame"],
        )


class InstrumentationTests(unittest.TestCase):
    def frame_records(
        self, detector: object, tracker: ConfirmationTracker | None
    ) -> list[dict[str, JsonValue]]:
        source = Mock()
        delivered_s = time.monotonic() - 0.25
        source.get.side_effect = [
            FramePacket(
                0, 10.0, create_test_frame(), delivered_s, "wide", lux=12.0, delivery_ms=40.0
            ),
            EndOfStream(),
        ]
        source.dropped_frames = 0
        queue = OutputQueue()
        processor = FrameProcessor(
            replace(AppConfig(), display=DisplayConfig(False)),
            source,
            cast(FrameDetector, detector),
            tracker,
            queue,
            threading.Event(),
            time.monotonic(),
        )
        processor.run()
        return [record.fields for record in queue.take_logs() if record.event == "frame"]

    def test_stage_timings_from_ultralytics_speed(self) -> None:
        class Detector:
            names = {0: "pigeon"}
            last_stage_ms = {"preprocess": 1.0, "inference": 30.0, "postprocess": 2.0}

            def warmup(self, width: int, height: int) -> None:
                pass

            def detect(self, packet: FramePacket) -> list[Detection]:
                return [Detection("pigeon", 0.9, 1, 2, 30, 40)]

        tracker = ConfirmationTracker(ConfirmationConfig(enabled=True), ["pigeon"])
        (record,) = self.frame_records(Detector(), tracker)
        stages = cast(dict[str, float], record["stage_ms"])
        self.assertEqual(
            [stages["preprocess"], stages["inference"], stages["postprocess"]], [1.0, 30.0, 2.0]
        )
        self.assertGreaterEqual(stages["wait"], 250)
        self.assertGreaterEqual(stages["track"], 0)
        self.assertGreaterEqual(cast(float, record["decision_ms"]), stages["wait"])
        self.assertEqual(
            (record["camera"], record["lux"], record["delivery_ms"]), ("wide", 12.0, 40.0)
        )
        self.assertAlmostEqual(cast(float, record["decided_at"]), time.time(), delta=5)

    def test_backend_without_stage_timings(self) -> None:
        detector = Mock(spec=["names", "warmup", "detect"])
        detector.detect.return_value = []
        (record,) = self.frame_records(detector, None)
        stages = cast(dict[str, JsonValue], record["stage_ms"])
        self.assertEqual(
            (stages["preprocess"], stages["postprocess"], stages["track"]), (None, None, None)
        )
        self.assertAlmostEqual(
            cast(float, stages["inference"]), cast(float, record["inference_s"]) * 1000
        )

    def test_summary_reports_values_beyond_the_histogram(self) -> None:
        def summary_of(waits: list[float]) -> dict[str, JsonValue]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "detections.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(
                            {"event": "frame", "inference_s": 0.01, "stage_ms": {"wait": wait}}
                        )
                        + "\n"
                        for wait in waits
                    )
                )
                distributions = cast(dict[str, JsonValue], summarize_log(path)["distributions_ms"])
                return cast(dict[str, JsonValue], distributions["wait"])

        wait = summary_of([5.0] * 18 + [60_000.0, 30_000.0])
        # in-range percentiles are bin upper edges; values beyond 10 s stay exact.
        self.assertEqual(
            (wait["p50"], wait["p95"], wait["p99"], wait["max"]),
            (5.1, 30_000.0, 60_000.0, 60_000.0),
        )
        self.assertNotIn("lower_bound", wait)
        with patch("tools.summarize_run.OVERFLOW_EXACT_LIMIT", 2):
            wait = summary_of([5.0] * 7 + [20_000.0, 60_000.0, 40_000.0])
        self.assertEqual((wait["p95"], wait["max"]), (10_000.0, 60_000.0))
        self.assertEqual(wait["lower_bound"], ["p95", "p99"])

    def test_summary_reports_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            records: list[dict[str, JsonValue]] = [
                {
                    "event": "frame",
                    "inference_s": index / 1000,
                    "stage_ms": {
                        "wait": float(index),
                        "preprocess": 1.0,
                        "inference": 20.0,
                        "postprocess": None,
                        "track": 0.05,
                    },
                    "delivery_ms": 45.0,
                    "decision_ms": 50.0 + index,
                    "camera": "wide" if index < 90 else "noir",
                }
                for index in range(1, 101)
            ]
            records += [
                {"event": "frame", "inference_s": 0.02},
                {"event": "trigger", "first_seen_captured_at": 100.0, "confirmed_at": 100.3},
                {"event": "camera_switch", "from": "wide", "to": "noir"},
                {"event": "run_end", "error": None},
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            summary = summarize_log(path)
        distributions = cast(dict[str, dict[str, float] | None], summary["distributions_ms"])
        wait = distributions["wait"]
        assert wait is not None
        # percentiles are the upper edge of their 0.1 ms bin.
        self.assertEqual(
            (wait["count"], wait["p50"], wait["p95"], wait["max"]), (100, 50.1, 95.1, 100)
        )
        self.assertAlmostEqual(wait["mean"], 50.5)
        self.assertIsNone(distributions["postprocess"])
        self.assertEqual(cast(dict[str, float], distributions["track"])["p99"], 0.05)
        self.assertEqual(cast(dict[str, float], distributions["inference_total"])["count"], 101)
        self.assertAlmostEqual(cast(dict[str, float], distributions["reaction"])["max"], 300)
        self.assertEqual((summary["triggers"], summary["camera_switches"]), (1, 1))
        self.assertEqual(summary["frames_by_camera"], {"wide": 89, "noir": 11})


class ReactionTests(unittest.TestCase):
    def test_schedule_alternates_and_waits_for_late_triggers(self) -> None:
        schedule = StimulusSchedule(
            ReactionPlan(trials=2, show_s=3, gap_min_s=4, gap_max_s=4, settle_s=5, grace_s=1)
        )
        schedule.begin(0)
        steps = [(4.9, None), (5, "on"), (7.9, None), (8, "off"), (11.9, None), (12, "on")]
        steps += [(15, "off"), (15.5, None)]
        for now_s, expected in steps:
            with self.subTest(now_s=now_s):
                self.assertEqual(schedule.advance(now_s), expected)
        self.assertFalse(schedule.done)
        self.assertIsNone(schedule.advance(16))
        self.assertTrue(schedule.done)
        self.assertEqual(schedule.trial, 2)

    def test_analysis_is_independent_of_arrival_order(self) -> None:
        pigeon: list[JsonValue] = [{"label": "pigeon"}]
        records: list[tuple[str, dict[str, JsonValue]]] = [
            ("frame", {"captured_at": 100.1, "decided_at": 100.15, "detections": pigeon}),
            ("trigger", {"confirmed_at": 100.3, "track_id": 1}),
        ]
        orders = (
            # the log writer saw the trigger before the stimulus was registered.
            ["records", "on", "off"],
            ["on", "records", "off"],
            ["on", "off", "records"],
        )
        for order in orders:
            analysis = ReactionAnalysis(ReactionPlan(grace_s=1))
            for step in order:
                if step == "records":
                    for event, fields in records:
                        analysis.observe(event, fields)
                elif step == "on":
                    analysis.stimulus(1, "on", 100.0)
                else:
                    analysis.stimulus(1, "off", 103.0)
            report = analysis.report()
            with self.subTest(order=order):
                self.assertEqual(
                    (report["triggered"], report["missed"], report["false_triggers"]), (1, 0, 0)
                )
                trial = cast(list[dict[str, JsonValue]], report["per_trial"])[0]
                self.assertAlmostEqual(cast(float, trial["reaction_s"]), 0.3)
                self.assertAlmostEqual(cast(float, trial["first_detection_s"]), 0.15)

    def test_analysis_matches_triggers_to_trials(self) -> None:
        analysis = ReactionAnalysis(ReactionPlan(grace_s=1))
        pigeon: list[JsonValue] = [{"label": "pigeon"}]
        analysis.stimulus(1, "on", 100.0)
        analysis.observe("frame", {"captured_at": 99.9, "decided_at": 100.1, "detections": pigeon})
        analysis.observe(
            "frame", {"captured_at": 100.2, "decided_at": 100.25, "detections": pigeon}
        )
        analysis.observe("trigger", {"confirmed_at": 100.4, "track_id": 3})
        analysis.observe("trigger", {"confirmed_at": 100.9, "track_id": 4})
        analysis.stimulus(1, "off", 103.0)
        analysis.observe("trigger", {"confirmed_at": 103.5, "track_id": 5})
        analysis.observe("trigger", {"confirmed_at": 106.0, "track_id": 6})
        analysis.stimulus(2, "on", 108.0)
        analysis.stimulus(2, "off", 111.0)
        report = analysis.report()
        self.assertEqual(
            (report["trials"], report["triggered"], report["missed"], report["false_triggers"]),
            (2, 1, 1, 1),
        )
        trials = cast(list[dict[str, JsonValue]], report["per_trial"])
        self.assertAlmostEqual(cast(float, trials[0]["reaction_s"]), 0.4)
        self.assertAlmostEqual(cast(float, trials[0]["first_detection_s"]), 0.25)
        self.assertEqual(trials[0]["track_id"], 3)
        self.assertIsNone(trials[1]["reaction_s"])
        self.assertEqual(cast(dict[str, JsonValue], report["reaction_s"])["count"], 1)

    def test_stimulus_window(self) -> None:
        import cv2

        clock = iter([0.0, 4.0, 5.0, 5.01, 8.0, 8.02])
        with (
            patch.object(cv2, "imread", return_value=numpy.full((4, 4, 3), 255, numpy.uint8)),
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "setWindowProperty"),
            patch.object(cv2, "imshow") as show,
            patch.object(cv2, "waitKey", return_value=-1),
            patch.object(cv2, "destroyWindow") as destroy,
        ):
            test = ReactionTest(
                "pigeon.jpg",
                ReactionPlan(trials=1, show_s=3, settle_s=5),
                clock=lambda: next(clock),
            )
            test.start()
            self.assertIsNone(test.tick())
            self.assertEqual(test.tick(), {"trial": 1, "state": "on", "displayed_at": 5.01})
            self.assertEqual(test.tick(), {"trial": 1, "state": "off", "displayed_at": 8.02})
            test.close()
            test.close()
        self.assertEqual([call.args[1].max() for call in show.call_args_list], [0, 255, 0])
        destroy.assert_called_once()
        with patch.object(cv2, "imread", return_value=None), self.assertRaises(ValueError):
            ReactionTest("missing.jpg", ReactionPlan())

    def test_reaction_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "pigeon.jpg"
            image.write_bytes(b"jpeg")
            options = CommandOptions(reaction_test=str(image))
            config = load_configuration(options)
            self.assertFalse(config.display.enabled)
            self.assertTrue(config.confirmation.enabled)
            for invalid in (
                replace(options, video="clip.avi"),
                replace(options, reaction_test=str(Path(directory) / "missing.jpg")),
                replace(options, reaction_trials=0),
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                    load_configuration(invalid)

    def test_run_reports_reaction_times(self) -> None:
        reaction = Mock()
        reaction.finished = False

        def tick() -> dict[str, JsonValue]:
            reaction.finished = True
            return {"trial": 1, "state": "off", "displayed_at": 1.0}

        reaction.tick.side_effect = tick
        reaction.analysis.report.return_value = {
            key: None for key in ("trials", "triggered", "missed", "false_triggers")
        } | {"reaction_s": None, "first_detection_s": None}
        detector, output = Mock(), Mock()
        detector.detect.return_value = []
        detector.trigger_classes = ("pigeon",)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "pigeon.jpg"
            image.write_bytes(b"jpeg")
            with (
                patch("app.main.create_detector", return_value=(detector, ("pigeon",))),
                patch("app.main.Picamera2Capture", return_value=FastCamera()),
                patch("app.main.RunOutput", return_value=output),
                patch("app.main.ReactionTest", return_value=reaction) as factory,
                patch("app.main.health_snapshot", return_value={}),
                redirect_stdout(StringIO()),
            ):
                status = run(build_parser().parse_args(["--reaction-test", str(image)]))
        self.assertEqual(status, 0)
        self.assertEqual(factory.call_args.args[2], ("pigeon",))
        reaction.start.assert_called_once()
        reaction.close.assert_called_once()
        names = [call.args[0] for call in output.event.call_args_list]
        self.assertIn("stimulus", names)
        self.assertEqual(names[-2:], ["reaction_test", "run_end"])
        self.assertEqual(output.event.call_args.kwargs["reason"], "reaction_test_complete")
        output.write_json.assert_called_once()


class RuntimeTests(unittest.TestCase):
    def exercise(
        self,
        packets: Sequence[FramePacket | Exception],
        failure: BaseException | None = None,
        output_failure: bool = False,
        max_frames: int = 0,
        config: AppConfig | None = None,
        detections: Sequence[Sequence[Detection]] = (),
    ) -> tuple[int, Mock, Mock, Mock]:
        config = config or replace(AppConfig(), display=DisplayConfig(False))
        arguments = build_parser().parse_args(
            ["--video", "fake.avi", "--max-frames", str(max_frames)]
        )
        source, detector, output = Mock(), Mock(), Mock()
        source.get.side_effect = packets
        source.dropped_frames = 0
        detector.detect.side_effect = failure or (
            [list(items) for items in detections] if detections else None
        )
        detector.detect.return_value = []
        detector.trigger_classes = ("pigeon",)
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

    def test_confirmation_is_opt_in(self) -> None:
        packets = [FramePacket(index, 10 + index, create_test_frame()) for index in range(3)]
        pigeon = Detection("pigeon", 0.9, 1, 2, 30, 40)
        frames = [[pigeon], [], [pigeon]]
        _, _, _, output = self.exercise([*packets, EndOfStream()], detections=frames)
        names = [event_call.args[0] for event_call in output.event.call_args_list]
        self.assertNotIn("trigger", names)
        records = [
            event_call.kwargs
            for event_call in output.event.call_args_list
            if event_call.args[0] == "frame"
        ]
        self.assertIsNone(records[0]["detections"][0]["track_id"])
        config = replace(
            AppConfig(),
            display=DisplayConfig(False),
            confirmation=ConfirmationConfig(enabled=True),
        )
        _, _, _, output = self.exercise([*packets, EndOfStream()], config=config, detections=frames)
        events = output.event.call_args_list
        triggers = [event_call.kwargs for event_call in events if event_call.args[0] == "trigger"]
        self.assertEqual(len(triggers), 1)
        self.assertEqual(
            (triggers[0]["track_id"], triggers[0]["class"], triggers[0]["first_seen_frame"]),
            (1, "pigeon", 0),
        )
        self.assertEqual(triggers[0]["first_seen_captured_at"], 10)
        names = [event_call.args[0] for event_call in events]
        # the trigger follows the frame record that confirmed it.
        self.assertEqual(names[names.index("trigger") - 1], "frame")

    def test_render_failure_keeps_frame_and_trigger_records(self) -> None:
        pigeon = Detection("pigeon", 0.9, 1, 2, 30, 40)
        output = Mock()
        output.poll.return_value = True
        processed = threading.Event()

        def failing_render(*_arguments: object, **_keywords: object) -> bool:
            self.assertTrue(processed.wait(THREAD_TEST_TIMEOUT_S))
            raise RuntimeError("display failed")

        output.render.side_effect = failing_render
        # the render waits for the whole run, so every record is still queued when it fails.
        status = self.live_run(output, [[], [], [pigeon], [pigeon]], processed)
        self.assertEqual(status, 1)
        events = output.event.call_args_list
        frames = [call for call in events if call.args[0] == "frame"]
        triggers = [call for call in events if call.args[0] == "trigger"]
        run_end = events[-1]
        self.assertEqual(run_end.args[0], "run_end")
        self.assertEqual(
            (run_end.kwargs["reason"], run_end.kwargs["processed_frames"]), ("error", 4)
        )
        self.assertEqual(len(frames), run_end.kwargs["processed_frames"])
        self.assertEqual([call.kwargs["frame_number"] for call in triggers], [3])

    def live_run(
        self,
        output: Mock,
        detections: Sequence[Sequence[Detection]],
        processed: threading.Event,
        config: AppConfig | None = None,
        camera: object | None = None,
    ) -> int:
        """Run live with a fast fake camera, confirmation and display on; `processed`
        is set once the last frame has been detected."""
        calls = 0

        def detect(_packet: FramePacket) -> list[Detection]:
            nonlocal calls
            calls += 1
            if calls == len(detections):
                processed.set()
            return list(detections[min(calls, len(detections)) - 1])

        detector = Mock()
        detector.detect.side_effect = detect
        detector.trigger_classes = ("pigeon",)
        config = config or replace(
            AppConfig(),
            display=DisplayConfig(True),
            confirmation=ConfirmationConfig(enabled=True),
        )
        with (
            patch("app.main.AppConfig.load", return_value=config),
            patch("app.main.UltralyticsDetector", return_value=detector),
            patch("app.main.Picamera2Capture", return_value=camera or FastCamera()),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            return run(build_parser().parse_args(["--max-frames", str(len(detections))]))

    def test_worker_base_exceptions_are_failures(self) -> None:
        for error in (SystemExit("worker exit"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                packet = FramePacket(0, 0, create_test_frame())
                with redirect_stderr(StringIO()):
                    result, source, _, output = self.exercise([packet], error)
                self.assertEqual(result, 1)
                self.assertEqual(output.event.call_args.kwargs["reason"], "error")
                self.assertIsNotNone(output.event.call_args.kwargs["error"])
                source.close.assert_called_once()

    def test_display_stall_does_not_block_logging(self) -> None:
        frame_limit = 10
        output = Mock()
        output.poll.return_value = True
        processed = threading.Event()
        logged_during_stall: list[int] = []

        def frames_logged() -> int:
            return sum(call.args[0] == "frame" for call in output.event.call_args_list)

        def stalled_render(*_arguments: object, **_keywords: object) -> bool:
            self.assertTrue(processed.wait(THREAD_TEST_TIMEOUT_S))
            deadline = time.monotonic() + THREAD_TEST_TIMEOUT_S
            while frames_logged() < frame_limit and time.monotonic() < deadline:
                time.sleep(0.005)
            logged_during_stall.append(frames_logged())
            return True

        output.render.side_effect = stalled_render
        status = self.live_run(output, [[]] * frame_limit, processed)
        self.assertEqual(status, 0)
        self.assertEqual(logged_during_stall[0], frame_limit)

    def test_permanently_blocked_log_writer_cannot_block_exit(self) -> None:
        never = threading.Event()
        self.addCleanup(never.set)
        output = Mock()
        output.poll.return_value = True
        output.render.return_value = True

        def blocked_event(event: str, **_fields: JsonValue) -> None:
            if event == "frame":
                never.wait()

        output.event.side_effect = blocked_event
        camera = Mock(wraps=FastCamera())
        stderr = StringIO()
        started_s = time.monotonic()
        with patch("app.main.LOG_DRAIN_TIMEOUT_S", 0.2), redirect_stderr(stderr):
            status = self.live_run(output, [[]] * 5, threading.Event(), camera=camera)
        self.assertLess(time.monotonic() - started_s, THREAD_TEST_TIMEOUT_S)
        self.assertEqual(status, 1)
        camera.close.assert_called_once()
        self.assertIn("log writer stalled; 5 records unwritten", stderr.getvalue())
        names = [call.args[0] for call in output.event.call_args_list]
        # nothing more is written to the blocked sink, and it is not closed under the writer.
        self.assertNotIn("run_end", names)
        output.close.assert_not_called()
        output.close_recording_and_display.assert_called_once()

    def test_trigger_backlog_beyond_allowance_stops_the_run(self) -> None:
        frame_limit = 200
        output = Mock()
        output.poll.return_value = True
        output.render.return_value = True
        processed = threading.Event()
        released_at: list[float] = []

        def stalled_event(event: str, **_fields: JsonValue) -> None:
            # the log writer stalls for a second on its first record.
            if event in ("frame", "trigger"):
                if not released_at:
                    released_at.append(time.monotonic() + 1.0)
                while time.monotonic() < released_at[0]:
                    time.sleep(0.01)

        output.event.side_effect = stalled_event
        # a new bird far from the last one every frame, each confirming at once.
        detections = [
            [Detection("pigeon", 0.9, index * 50, 0, index * 50 + 20, 20)]
            for index in range(frame_limit)
        ]
        config = replace(
            AppConfig(),
            display=DisplayConfig(True),
            confirmation=ConfirmationConfig(enabled=True, k=1, n=1),
        )
        with (
            patch("app.main.LOG_QUEUE_CAPACITY", 4),
            patch("app.main.LOG_EVENT_ALLOWANCE", 4),
        ):
            status = self.live_run(output, detections, processed, config)
        self.assertEqual(status, 1)
        run_end = output.event.call_args.kwargs
        self.assertEqual(run_end["reason"], "error")
        self.assertIn("log writer", run_end["error"])
        self.assertLess(run_end["processed_frames"], frame_limit)
        triggers = [
            call.kwargs["track_id"]
            for call in output.event.call_args_list
            if call.args[0] == "trigger"
        ]
        self.assertEqual(triggers, sorted(triggers))
        self.assertGreaterEqual(len(triggers), 4)

    def test_log_writer_stall_drops_frames_but_keeps_triggers(self) -> None:
        frame_limit = 40
        pigeon = Detection("pigeon", 0.9, 1, 2, 30, 40)
        output = Mock()
        output.poll.return_value = True
        output.render.return_value = True
        processed = threading.Event()

        def stalled_event(event: str, **_fields: JsonValue) -> None:
            if event == "frame":
                processed.wait(THREAD_TEST_TIMEOUT_S)

        output.event.side_effect = stalled_event
        detections: list[list[Detection]] = [[]] * (frame_limit - 2) + [[pigeon], [pigeon]]
        with patch("app.main.LOG_QUEUE_CAPACITY", 8):
            status = self.live_run(output, detections, processed)
        self.assertEqual(status, 0)
        events = output.event.call_args_list
        run_end = events[-1].kwargs
        frames = sum(call.args[0] == "frame" for call in events)
        self.assertEqual(run_end["processed_frames"], frame_limit)
        self.assertGreater(run_end["log_dropped_records"], 0)
        self.assertEqual(frames + run_end["log_dropped_records"], frame_limit)
        self.assertEqual(
            [call.kwargs["frame_number"] for call in events if call.args[0] == "trigger"],
            [frame_limit - 1],
        )

    def test_live_processing_never_waits_for_display(self) -> None:
        frame_limit = 20
        processed = threading.Event()
        detect_calls = 0

        def detect(_packet: FramePacket) -> list[Detection]:
            nonlocal detect_calls
            detect_calls += 1
            if detect_calls == frame_limit:
                processed.set()
            return []

        def slow_render(*_arguments: object, **_keywords: object) -> bool:
            # the display is stuck until every frame has been processed.
            self.assertTrue(processed.wait(THREAD_TEST_TIMEOUT_S))
            return True

        detector, output = Mock(), Mock()
        detector.detect.side_effect = detect
        output.render.side_effect = slow_render
        output.poll.return_value = True
        config = replace(AppConfig(), display=DisplayConfig(True))
        with (
            patch("app.main.AppConfig.load", return_value=config),
            patch("app.main.UltralyticsDetector", return_value=detector),
            patch("app.main.Picamera2Capture", return_value=FastCamera()),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            self.assertEqual(run(build_parser().parse_args(["--max-frames", str(frame_limit)])), 0)
        run_end = output.event.call_args.kwargs
        self.assertEqual((run_end["reason"], run_end["processed_frames"]), ("max_frames", 20))
        self.assertGreater(run_end["render_dropped_frames"], 0)
        self.assertLess(output.render.call_count, frame_limit)
        frames = [call for call in output.event.call_args_list if call.args[0] == "frame"]
        self.assertEqual(len(frames), frame_limit)

    def test_window_close_and_render_failure_stop_processing(self) -> None:
        packets = [FramePacket(index, index, create_test_frame()) for index in range(50)]
        config = replace(AppConfig(), display=DisplayConfig(True))
        for render_result, expected_status, expected_reason in (
            (False, 0, "window_closed_or_q"),
            (RuntimeError("writer failed"), 1, "error"),
        ):
            with self.subTest(expected_reason=expected_reason):
                source, detector, output = Mock(), Mock(), Mock()
                source.get.side_effect = packets
                source.dropped_frames = 0
                detector.detect.return_value = []
                if isinstance(render_result, Exception):
                    output.render.side_effect = render_result
                else:
                    output.render.return_value = render_result
                with (
                    patch("app.main.AppConfig.load", return_value=config),
                    patch("app.main.UltralyticsDetector", return_value=detector),
                    patch("app.main.VideoFileCapture", return_value=source),
                    patch("app.main.RunOutput", return_value=output),
                    patch("app.main.health_snapshot", return_value={}),
                ):
                    status = run(build_parser().parse_args(["--video", "fake.avi"]))
                self.assertEqual(status, expected_status)
                self.assertEqual(output.event.call_args.args[0], "run_end")
                self.assertEqual(output.event.call_args.kwargs["reason"], expected_reason)
                self.assertLess(source.get.call_count, len(packets))
                source.close.assert_called_once()
                output.close.assert_called_once()

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
        detector = Mock()
        detector.warmup.side_effect = DetectorError("warmup")
        with (
            patch("app.main.create_detector", return_value=(detector, ("pigeon",))),
            patch("app.main.Picamera2Capture") as camera,
            patch("app.main.RunOutput") as output,
        ):
            self.assertEqual(run(arguments), 1)
        detector.warmup.assert_called_once()
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
            patch("app.main.create_detector", return_value=(detector, ("pigeon",))),
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

    def test_command_line_overrides_configuration_and_rejects_negative_frame_limit(self) -> None:
        overridden = load_configuration(
            CommandOptions(model="override.pt", confidence=0.7, record=True)
        )
        self.assertEqual(overridden.model.path, "override.pt")
        self.assertEqual(overridden.model.confidence, 0.7)
        self.assertTrue(overridden.output.record)
        with self.assertRaises(ConfigError):
            load_configuration(CommandOptions(max_frames=-1))

    def test_source_cleanup_failure_still_closes_outputs(self) -> None:
        source, output = Mock(), Mock()
        source.get.side_effect = EndOfStream()
        source.close.side_effect = RuntimeError("release failed")
        with (
            patch("app.main.create_detector", return_value=(Mock(), ("pigeon",))),
            patch("app.main.VideoFileCapture", return_value=source),
            patch("app.main.RunOutput", return_value=output),
            patch("app.main.health_snapshot", return_value={}),
        ):
            self.assertEqual(run(build_parser().parse_args(["--video", "fake.avi"])), 1)
        output.close.assert_called_once()
        self.assertEqual(output.event.call_args.kwargs["reason"], "cleanup_error")


if __name__ == "__main__":
    unittest.main()
