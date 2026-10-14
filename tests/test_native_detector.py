from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock, patch

import numpy
import yaml

from app.config import ModelConfig
from app.detector import DetectorError
from app.main import create_detector
from app.models import FramePacket
from app.native_detector import MetadataValue, NativeDetector
from native.rescore_parity import rescore_report
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
        with self.assertRaisesRegex(DetectorError, "pigeon must have class index 0"):
            self.detector()

    def test_legacy_package_and_lazy_loading(self) -> None:
        detector = self.detector()
        self.constructor.assert_called_once_with(str(self.model), threads=2, compute=None)
        self.assertEqual(detector.names, {0: "pigeon"})
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
        self.assertEqual([detection.class_id for detection in result], [0, 0])
        self.assertEqual(
            detector.last_stage_ms, {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
        )
        self.native_model.detect.assert_called_once_with(self.frame, 0.5)

    def test_species_names_class_ids_and_pigeon_only_triggers(self) -> None:
        self.metadata["names"] = {0: "PIGEON", 1: "Spotted_Dove"}
        self.metadata["columba"] = {
            "input_hw": [640, 640],
            "decode": "yolo11",
            "trigger_classes": ["pigeon"],
        }
        self.write_metadata()
        self.native_model.names = {0: "PIGEON", 1: "Spotted_Dove"}
        self.native_model.detect.return_value = numpy.asarray(
            [[1, 2, 3, 4, 0.8, 1], [5, 6, 7, 8, 0.7, 0]], dtype=numpy.float32
        )
        detector = self.detector()
        detections = detector.detect(FramePacket(0, 0.0, self.frame))
        self.assertEqual([item.label for item in detections], ["spotted_dove", "pigeon"])
        self.assertEqual([item.class_id for item in detections], [1, 0])
        self.assertEqual(detector.trigger_classes, ("pigeon",))
        self.metadata["columba"] = {
            "input_hw": [640, 640],
            "decode": "yolo11",
            "trigger_classes": ["spotted_dove"],
        }
        self.write_metadata()
        with self.assertRaisesRegex(DetectorError, "may only trigger on pigeon"):
            self.detector()

    def test_runtime_factory_uses_native_wrapper_and_stage_timings(self) -> None:
        self.metadata["imgsz"] = [384, 640]
        self.metadata["columba"] = {
            "input_hw": [384, 640],
            "decode": "yolo26",
            "trigger_classes": ["pigeon"],
        }
        self.write_metadata()
        configuration = ModelConfig(
            path=str(self.model), image_size=(384, 640), backend="native", threads=2
        )
        with patch.dict(sys.modules, {"columba_native": self.extension}):
            detector, triggers = create_detector(configuration)
        self.assertIsInstance(detector, NativeDetector)
        self.assertEqual(triggers, ("pigeon",))
        detector.detect(FramePacket(0, 0.0, self.frame))
        self.assertEqual(
            cast(NativeDetector, detector).last_stage_ms,
            {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
        )
        self.constructor.assert_called_once_with(str(self.model), threads=2, compute=None)

    def test_compute_override_forwarding_and_validation(self) -> None:
        for compute in (None, "fp16", "fp32"):
            with self.subTest(compute=compute):
                self.constructor.reset_mock()
                with patch.dict(sys.modules, {"columba_native": self.extension}):
                    NativeDetector(str(self.model), 640, 0.5, threads=2, compute=compute)
                self.constructor.assert_called_once_with(
                    str(self.model), threads=2, compute=compute
                )
        for compute in ("", "auto", "FP16", "int8"):
            with self.subTest(compute=compute):
                with patch("app.native_detector.importlib.import_module") as importer:
                    with self.assertRaisesRegex(DetectorError, "compute must be"):
                        NativeDetector(str(self.model), 640, 0.5, compute=compute)
                    importer.assert_not_called()

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
        one_ulp = float(numpy.nextafter(numpy.float32(637.0), numpy.float32(numpy.inf)))
        rounded: ExportPrediction = {**expected, "box": [0.0, 0.0, one_ulp, 0.0]}
        self.assertEqual(len(match_clipped_boxes([expected], [rounded])), 1)
        outside_tolerance: ExportPrediction = {**expected, "box": [0.0, 0.0, 637.001, 0.0]}
        self.assertEqual(match_clipped_boxes([expected], [outside_tolerance]), [])
        shifted: ExportPrediction = {**expected, "box": [0.0, 0.0, 638.0, 0.0]}
        self.assertEqual(match_clipped_boxes([expected], [shifted]), [])

    def test_degenerate_box_rounding_uses_box_scale_before_cancellation(self) -> None:
        expected: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.8,
            "box": [427.0, 2.0595295429229736, 427.0, 638.6720581054688],
        }
        rounded: ExportPrediction = {
            **expected,
            "box": [427.0, 2.059478759765625, 427.0, 638.6719970703125],
        }
        self.assertEqual(len(match_clipped_boxes([expected], [rounded])), 1)
        shifted: ExportPrediction = {**rounded, "box": [427.0, 2.06, 427.0, 638.6719970703125]}
        self.assertEqual(len(match_clipped_boxes([expected], [shifted])), 1)
        outside_tolerance: ExportPrediction = {**rounded, "box": [427.0, 2.07, 427.0, 638.67]}
        self.assertEqual(match_clipped_boxes([expected], [outside_tolerance]), [])
        low_score: ExportPrediction = {**rounded, "confidence": 0.7}
        self.assertEqual(match_clipped_boxes([expected], [low_score]), [])

    def test_rescore_preserves_linux_predictions_and_uses_relative_rounding_bound(self) -> None:
        expected: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.2612849473953247,
            "box": [0.0, 7.390136241912842, 0.0, 1276.990966796875],
        }
        actual: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.2612834572792053,
            "box": [0.0, 7.391153812408447, 0.0, 1276.990966796875],
        }
        self.assertEqual(len(match_clipped_boxes([expected], [actual])), 1)
        shifted: ExportPrediction = {**actual, "box": [0.0, 7.4, 0.0, 1276.990966796875]}
        self.assertEqual(match_clipped_boxes([expected], [shifted]), [])
        source = self.model / "original-parity.json"
        destination = self.model / "rescored-parity.json"
        source.write_text(
            json.dumps(
                {
                    "min_iou": 0.8,
                    "max_confidence_delta": 0.05,
                    "informational": False,
                    "records": [
                        {"image": "image.jpg", "reference": [expected], "native": [actual]}
                    ],
                }
            )
        )
        original = source.read_bytes()
        rescored = rescore_report(source, destination)
        self.assertEqual(rescored["failures"], 0)
        self.assertEqual(source.read_bytes(), original)
        output = json.loads(destination.read_text())
        self.assertEqual(output["records"][0]["reference"], [expected])
        self.assertEqual(output["records"][0]["native"], [actual])
        self.assertEqual(len(output["rescore"]["prediction_sha256"]), 64)


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

    def test_compute_defaults_and_overrides_set_loaded_network_options(self) -> None:
        extension = cast(ModuleType, columba_native)
        for quantisation in (None, "fp32", "fp16", "int8"):
            if quantisation is None:
                self.metadata.pop("columba", None)
            else:
                self.metadata["columba"] = {
                    "input_hw": [640, 640],
                    "decode": "yolo11",
                    "quantization": quantisation,
                }
            self.write_metadata()
            for compute in (None, "fp16", "fp32"):
                with self.subTest(quantisation=quantisation, compute=compute):
                    detector = extension.Detector(str(self.model), threads=2, compute=compute)
                    expected = compute or ("fp16" if quantisation in ("fp16", "int8") else "fp32")
                    self.assertEqual(detector.compute, expected)
                    self.assertEqual(
                        detector.compute_flags,
                        {
                            name: expected == "fp16"
                            for name in (
                                "use_fp16_packed",
                                "use_fp16_storage",
                                "use_fp16_arithmetic",
                            )
                        },
                    )

    def test_extension_rejects_invalid_compute(self) -> None:
        extension = cast(ModuleType, columba_native)
        for compute in ("", "auto", "FP16", "int8"):
            with self.subTest(compute=compute):
                with self.assertRaisesRegex(ValueError, "compute must be"):
                    extension.Detector(str(self.model), compute=compute)
        with self.assertRaises(TypeError):
            extension.Detector(str(self.model), compute=16)

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
