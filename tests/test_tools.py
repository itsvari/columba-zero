from __future__ import annotations

import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from unittest.mock import Mock, patch

import numpy

from app.main import finalize_output
from app.models import Frame
from tools.benchmark_detector import benchmark_video
from tools.compare_exports import ExportPrediction, intersection_over_union, match_predictions
from tools.summarize_run import summarize_log
from tools.train_model import main as train_main

TEST_FRAME_SIZE_PX = 32
COLOUR_CHANNEL_COUNT = 3
SUMMARY_FRAME_COUNT = 1005


class ToolTests(unittest.TestCase):
    def test_training_evaluates_the_checkpoint_from_the_actual_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.yaml"
            dataset.write_text("names: [pigeon]\ntest: test/images\n", encoding="utf-8")
            checkpoint = (
                Path(directory) / "runs/detect/pigeon_detector/pigeon_experiment2/weights/best.pt"
            )
            training_model = Mock()
            training_model.trainer.best = checkpoint
            evaluation_model = Mock()
            factory = Mock(side_effect=[training_model, evaluation_model])
            with (
                patch("sys.argv", ["train_model.py", str(dataset)]),
                patch.dict("sys.modules", {"ultralytics": Mock(YOLO=factory)}),
            ):
                train_main()
            self.assertEqual(factory.call_args.args, (str(checkpoint),))
            evaluation_model.val.assert_called_once_with(
                data=str(dataset), split="test", imgsz=640, plots=True
            )

    def test_export_matching_is_one_to_one_and_class_aware(self) -> None:
        expected: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.8,
            "box": [0, 0, 10, 10],
        }
        actual: ExportPrediction = {
            "label": "pigeon",
            "confidence": 0.78,
            "box": [0, 0, 10, 10],
        }
        matches = match_predictions([expected, expected], [actual])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["iou"], 1)
        self.assertAlmostEqual(matches[0]["confidence_delta"], 0.02)
        self.assertEqual(matches[0]["box_delta_px"], [0, 0, 0, 0])
        self.assertEqual(match_predictions([expected], [{**actual, "label": "bird"}]), [])
        self.assertEqual(intersection_over_union([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_export_matching_finds_a_valid_assignment_in_every_order(self) -> None:
        expected: list[ExportPrediction] = [
            {"label": "pigeon", "confidence": 0.8, "box": [20, 0, 120, 100]},
            {"label": "pigeon", "confidence": 0.8, "box": [38, 0, 138, 100]},
        ]
        actual: list[ExportPrediction] = [
            {"label": "pigeon", "confidence": 0.8, "box": [28, 0, 128, 100]},
            {"label": "pigeon", "confidence": 0.8, "box": [10, 0, 110, 100]},
        ]
        for expected_order in permutations(expected):
            for actual_order in permutations(actual):
                matches = match_predictions(expected_order, actual_order)
                self.assertEqual(len(matches), 2)
                self.assertTrue(all(match["iou"] >= 0.8 for match in matches))

    def test_export_matching_uses_confidence_and_custom_tolerances(self) -> None:
        expected: list[ExportPrediction] = [
            {"label": "pigeon", "confidence": 0.9, "box": [0, 0, 100, 100]},
            {"label": "pigeon", "confidence": 0.8, "box": [0, 0, 100, 100]},
        ]
        matches = match_predictions(expected, expected)
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(match["confidence_delta"] <= 0.05 for match in matches))
        shifted: ExportPrediction = {"label": "pigeon", "confidence": 0.7, "box": [10, 0, 110, 100]}
        self.assertEqual(match_predictions(expected[:1], [shifted]), [])
        matches = match_predictions(
            expected[:1],
            [shifted],
            minimum_intersection_over_union=0.8,
            maximum_confidence_delta=0.25,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(
            match_predictions(
                expected[:1],
                [shifted],
                minimum_intersection_over_union=0.9,
                maximum_confidence_delta=0.25,
            ),
            [],
        )

    def test_summary_preserves_fields_over_recent_frame_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            with path.open("w", encoding="utf-8") as log:
                log.write(
                    json.dumps({"event": "health", "memory_rss_bytes": 0, "temperature_c": 40})
                    + "\n"
                )
                for frame_index in range(SUMMARY_FRAME_COUNT):
                    log.write(
                        json.dumps({"event": "frame", "inference_s": frame_index / 1000}) + "\n"
                    )
                log.write(
                    json.dumps({"event": "health", "memory_rss_bytes": 1024, "temperature_c": 35})
                    + "\n"
                )
                log.write(json.dumps({"event": "run_end", "error": None}) + "\n")
            report = summarize_log(path)
        self.assertEqual(report["frames"], SUMMARY_FRAME_COUNT)
        self.assertEqual(report["inference_mean_ms"], 502)
        self.assertEqual(report["inference_max_ms"], 1004)
        self.assertEqual(report["inference_p95_last_1000_ms"], 954)
        self.assertEqual(report["memory_first_bytes"], 0)
        self.assertEqual(report["memory_last_bytes"], 1024)
        self.assertEqual(report["memory_max_bytes"], 1024)
        self.assertEqual(report["temperature_max_c"], 40)
        self.assertTrue(report["completed_cleanly"])

    def test_summary_requires_finite_inference_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            for inference in (True, "slow", float("nan")):
                with self.subTest(inference=inference):
                    path.write_text(json.dumps({"event": "frame", "inference_s": inference}))
                    with self.assertRaisesRegex(ValueError, "finite number"):
                        summarize_log(path)
            path.write_text(json.dumps({"event": "health"}))
            self.assertFalse(summarize_log(path)["completed_cleanly"])

    def test_benchmark_releases_video_on_inference_and_open_failure(self) -> None:
        import cv2

        frame: Frame = numpy.zeros(
            (TEST_FRAME_SIZE_PX, TEST_FRAME_SIZE_PX, COLOUR_CHANNEL_COUNT),
            numpy.uint8,
        )
        for opened in (False, True):
            with self.subTest(opened=opened):
                capture = Mock()
                capture.isOpened.return_value = opened
                capture.read.return_value = (True, frame)
                model = Mock()
                model.predict.side_effect = RuntimeError("inference failed")
                with patch.object(cv2, "VideoCapture", return_value=capture):
                    with self.assertRaises((SystemExit, RuntimeError)):
                        benchmark_video(model, "example.avi", 640, 1, 0.5)
                capture.release.assert_called_once()

    def test_final_log_failure_still_closes_output(self) -> None:
        output = Mock()
        output.event.side_effect = OSError("disk full")
        with patch("app.main.health_snapshot", return_value={}):
            self.assertFalse(finalize_output(output, "max_frames", None, 1, 0, None))
        output.close.assert_called_once()
