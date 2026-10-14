from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock, patch

import numpy
import yaml

from app.detector import DetectorError
from app.models import FramePacket
from app.native_detector import MetadataValue, NativeDetector
from native.verify_native import match_clipped_boxes
from tools.compare_exports import ExportPrediction

try:
    import columba_native
except ImportError:
    columba_native = None


class NativeDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)
        for name in ("model.ncnn.param", "model.ncnn.bin"):
            (self.model / name).write_bytes(b"test")
        self.metadata: dict[str | int, MetadataValue] = {
            "task": "detect",
            "names": {0: "pigeon"},
            "imgsz": [640, 640],
        }
        self.write_metadata()
        self.frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        self.extension = ModuleType("columba_native")
        self.native_model = Mock(names={0: "pigeon"}, input_hw=(640, 640), decode="yolo11")
        self.native_model.detect.return_value = numpy.empty((0, 6), dtype=numpy.float32)
        self.native_model.detect_with_timings.side_effect = lambda frame, confidence: (
            self.native_model.detect(frame, confidence),
            {"preprocess": 1.0, "extract": 2.0, "postprocess": 3.0, "total": 6.0},
        )
        self.constructor = Mock(return_value=self.native_model)
        self.extension.__dict__["Detector"] = self.constructor

    def write_metadata(self) -> None:
        (self.model / "metadata.yaml").write_text(yaml.safe_dump(self.metadata), encoding="utf-8")

    def detector(self, image_size: int | tuple[int, int] = 640) -> NativeDetector:
        with patch.dict(sys.modules, {"columba_native": self.extension}):
            return NativeDetector(str(self.model), image_size, 0.5, threads=2)

    def test_missing_and_incomplete_model_before_extension_import(self) -> None:
        with patch("app.native_detector.importlib.import_module") as importer:
            with self.assertRaisesRegex(DetectorError, "model does not exist"):
                NativeDetector(str(self.model / "missing"), 640, 0.5)
            (self.model / "model.ncnn.bin").write_bytes(b"")
            with self.assertRaisesRegex(DetectorError, "incomplete NCNN export"):
                self.detector()
            importer.assert_not_called()

    def test_rejects_checkpoint_and_invalid_metadata(self) -> None:
        checkpoint = self.model / "model.pt"
        checkpoint.write_bytes(b"test")
        with self.assertRaisesRegex(DetectorError, "complete NCNN directory"):
            NativeDetector(str(checkpoint), 640, 0.5)
        self.metadata["task"] = "classify"
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "invalid NCNN metadata"):
            self.detector()
        self.metadata["task"] = "detect"
        self.metadata["names"] = {0: "person"}
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "single-class"):
            self.detector()

    def test_legacy_package_and_lazy_loading(self) -> None:
        detector = self.detector()
        self.constructor.assert_called_once_with(str(self.model), threads=2)
        self.assertEqual(detector.names, {0: "pigeon"})
        self.assertEqual(detector._last_class_ids, ())
        with patch(
            "app.native_detector.importlib.import_module", side_effect=ImportError("missing")
        ):
            with self.assertRaisesRegex(DetectorError, "build columba_native"):
                NativeDetector(str(self.model), 640, 0.5)

    def test_rectangular_c8_package_and_size_validation(self) -> None:
        self.metadata["imgsz"] = [384, 640]
        self.metadata["columba"] = {"input_hw": [384, 640], "decode": "yolo26"}
        self.write_metadata()
        self.detector((384, 640))
        with self.assertRaisesRegex(DetectorError, "export size"):
            self.detector(640)
        self.metadata["columba"] = {"input_hw": [640, 384], "decode": "yolo26"}
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "input_hw"):
            self.detector((384, 640))
        self.metadata["columba"] = {"input_hw": [384, 640], "decode": "nanodet"}
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "unsupported decode"):
            self.detector((384, 640))
        del self.metadata["columba"]
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "legacy NCNN"):
            self.detector((384, 640))

    def test_filter_preserves_frame_and_class_id(self) -> None:
        self.native_model.detect.return_value = numpy.asarray(
            [[1, 2, 3, 4, 0.8, 0], [5, 6, 7, 8, 0.4, 0], [9, 10, 11, 12, 0.5, 0]],
            dtype=numpy.float32,
        )
        detector = self.detector()
        result = detector.detect(FramePacket(0, 0.0, self.frame))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "pigeon")
        self.assertEqual(result[0].left_px, 1)
        self.assertAlmostEqual(result[0].confidence, 0.8)
        self.assertIs(self.native_model.detect.call_args.args[0], self.frame)
        self.assertEqual(detector._last_class_ids, (0, 0))
        self.assertEqual(
            detector.last_stage_ms, {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
        )
        self.native_model.detect.assert_called_once_with(self.frame, 0.5)

    def test_warmup_and_inference_errors(self) -> None:
        detector = self.detector()
        detector.warmup(640, 480)
        frame = self.native_model.detect.call_args.args[0]
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, numpy.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        self.native_model.detect.side_effect = RuntimeError("bad inference")
        with self.assertRaisesRegex(DetectorError, "model inference failed: bad inference"):
            detector.detect(FramePacket(0, 0.0, self.frame))

    def test_invalid_confidence_and_threads(self) -> None:
        for confidence in (float("nan"), -0.1, 1.1):
            with self.subTest(confidence=confidence), self.assertRaises(DetectorError):
                NativeDetector(str(self.model), 640, confidence)
        with self.assertRaisesRegex(DetectorError, "threads must be positive"):
            NativeDetector(str(self.model), 640, 0.5, threads=0)

    def test_parity_handles_identical_zero_area_clipped_boxes(self) -> None:
        expected: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.49,
            "box": [0.0, 0.0, 637.0, 0.0],
        }
        self.assertEqual(len(match_clipped_boxes([expected], [expected])), 1)
        shifted: ExportPrediction = {**expected, "box": [0.0, 0.0, 638.0, 0.0]}
        self.assertEqual(match_clipped_boxes([expected], [shifted]), [])


@unittest.skipIf(columba_native is None, "columba_native is not built")
class NativeExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)
        (self.model / "model.ncnn.param").write_text(
            "7767517\n2 2\nInput input 0 1 in0\nMemoryData constant 0 1 out0 0=4 1=5\n",
            encoding="utf-8",
        )
        output = numpy.asarray(
            [
                [320, 320, 100, 100],
                [320, 320, 100, 100],
                [200, 200, 50, 50],
                [100, 100, 50, 50],
                [0.9, 0.8, 0.4, 0.7],
            ],
            dtype=numpy.float32,
        )
        (self.model / "model.ncnn.bin").write_bytes(output.tobytes())
        self.metadata: dict[str | int, MetadataValue] = {
            "task": "detect",
            "names": {0: "pigeon"},
            "imgsz": [640, 640],
        }
        self.write_metadata()

    def write_metadata(self) -> None:
        (self.model / "metadata.yaml").write_text(yaml.safe_dump(self.metadata), encoding="utf-8")

    def test_detector_returns_owned_float32_boxes_and_empty_array(self) -> None:
        extension = cast(ModuleType, columba_native)
        detector = extension.Detector(str(self.model), threads=2)
        self.assertEqual(detector.input_hw, (640, 640))
        self.assertEqual(detector.names, {0: "pigeon"})
        self.assertEqual(detector.decode, "yolo11")
        frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        actual = detector.detect(frame, 0.5)
        numpy.testing.assert_allclose(
            actual, [[220, 190, 420, 290, 0.9, 0], [75, 0, 125, 45, 0.7, 0]]
        )
        self.assertEqual(actual.dtype, numpy.float32)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(detector.detect(frame, 1.0).shape, (0, 6))
        del detector
        self.assertEqual(actual[0, 0], 220)
        self.assertFalse(frame.any())

    def test_one_to_one_topk_and_extract_timing(self) -> None:
        extension = cast(ModuleType, columba_native)
        self.metadata["columba"] = {"input_hw": [640, 640], "decode": "yolo26", "max_det": 2}
        self.write_metadata()
        detector = extension.Detector(str(self.model))
        frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        detections, timings = detector.detect_with_timings(frame, 0.5)
        self.assertEqual(detector.decode, "yolo26")
        numpy.testing.assert_allclose(detections[:, 4], [0.9, 0.8])
        numpy.testing.assert_array_equal(detections[0, :4], detections[1, :4])
        self.assertEqual(set(timings), {"preprocess", "extract", "postprocess", "total"})
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertGreaterEqual(timings["total"], timings["extract"])

    def test_extension_validates_metadata_threads_and_confidence(self) -> None:
        extension = cast(ModuleType, columba_native)
        with self.assertRaisesRegex(ValueError, "threads"):
            extension.Detector(str(self.model), threads=0)
        detector = extension.Detector(str(self.model))
        with self.assertRaisesRegex(ValueError, "confidence"):
            detector.detect(numpy.zeros((10, 20, 3), dtype=numpy.uint8), float("nan"))
        self.metadata["columba"] = {"input_hw": [640, 640], "decode": "yolo11", "letterbox": False}
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "letterbox"):
            extension.Detector(str(self.model))

    def test_letterbox_matches_installed_ultralytics(self) -> None:
        if importlib.util.find_spec("ultralytics") is None:
            self.skipTest("Ultralytics is not installed")
        from ultralytics.data.augment import LetterBox

        extension = cast(ModuleType, columba_native)
        random = numpy.random.default_rng(8)
        for height_px, width_px in ((480, 640), (321, 640), (333, 517), (720, 1280), (77, 51)):
            frame = random.integers(0, 256, (height_px, width_px, 3), dtype=numpy.uint8)
            for input_hw in ((640, 640), (384, 640)):
                with self.subTest(frame=frame.shape, input_hw=input_hw):
                    expected = LetterBox(input_hw, auto=False)(image=frame)
                    actual = extension._letterbox(frame, *input_hw)
                    numpy.testing.assert_array_equal(actual, expected)

    def test_rejects_array_conversion_and_empty_frames(self) -> None:
        extension = cast(ModuleType, columba_native)
        frame = numpy.zeros((20, 30, 3), dtype=numpy.uint8)
        for invalid in (frame[:, ::2], frame.astype(numpy.float32), frame[:, :, :2]):
            with self.subTest(shape=invalid.shape), self.assertRaises((TypeError, ValueError)):
                extension._letterbox(invalid, 640, 640)
        with self.assertRaises(ValueError):
            extension._letterbox(numpy.zeros((0, 30, 3), dtype=numpy.uint8), 640, 640)


if __name__ == "__main__":
    unittest.main()
