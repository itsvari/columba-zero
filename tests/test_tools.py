from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy

from app.main import finalize_output
from app.models import Frame
from tools.benchmark_detector import benchmark_video
from tools.compare_exports import ExportPrediction, intersection_over_union, match_predictions
from tools.summarize_run import summarize_log

TEST_FRAME_SIZE_PX = 32
COLOUR_CHANNEL_COUNT = 3
SUMMARY_FRAME_COUNT = 1005


class ToolTests(unittest.TestCase):
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
