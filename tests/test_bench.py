from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy

from app.json_data import JsonValue
from tools.bench_ncnn import (
    NCNN_COMMIT,
    apply_calibration,
    benchmark,
    calibration_factor,
    decode,
    letterbox,
    make_calibration,
    summary,
    write_report,
)
from tools.reaction_budget import ReactionSettings, processing_samples, simulate_reaction


class BenchmarkTests(unittest.TestCase):
    def test_documented_script_entry_points_work_outside_the_repository(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script in ("bench_ncnn.py", "reaction_budget.py"):
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(repository / "tools" / script), "--help"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("usage:", result.stdout)

    def test_benchmark_rejects_short_runs_and_invalid_export_metadata_without_ncnn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory)
            with self.assertRaisesRegex(ValueError, ">=200"):
                benchmark(location, location, (352, 608), "xcng", "model", iterations=199)
            (location / "build-evidence.txt").write_text(f"profile=xcng commit={NCNN_COMMIT}")
            (location / "metadata.yaml").write_text("[]")
            with self.assertRaisesRegex(ValueError, "mapping"):
                benchmark(location, location, (352, 608), "xcng", "model")
            (location / "metadata.yaml").write_text("imgsz: [640, 640]\nnames: [pigeon]\n")
            with self.assertRaisesRegex(ValueError, "export shape"):
                benchmark(location, location, (352, 608), "xcng", "model")

    def test_percentiles_use_linear_interpolation_and_population_std(self) -> None:
        result = summary([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(result["p50"], 25)
        self.assertEqual(result["p95"], 38.5)
        self.assertAlmostEqual(float(str(result["std"])), numpy.sqrt(125))
        for values in ([], [-1.0], [float("nan")]):
            with self.assertRaises(ValueError):
                summary(values)

    def test_c7_writer_sanitises_ids_and_preserves_samples(self) -> None:
        report: dict[str, JsonValue] = {
            "host": "xcng0",
            "model_id": "recipe/seed0",
            "format": "ncnn-fp16",
            "input_hw": [352, 608],
            "samples_ms": [1.0, 2.0],
            "label": "measured",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = write_report(report, Path(directory))
            self.assertEqual(output.name, "xcng0-recipe_seed0-ncnn-fp16-352x608.json")
            self.assertEqual(json.loads(output.read_text()), report)

    def test_calibration_is_host_matched_and_preserves_measurements(self) -> None:
        report: dict[str, JsonValue] = {
            "host_kind": "xcng-n1",
            "end_to_end_ms": {"p95": 40},
            "label": "measured",
        }
        calibration: dict[str, JsonValue] = {"id": "cal-test", "factor": 2}
        result = apply_calibration(report, calibration)
        self.assertEqual(
            result["estimated_pi5"],
            {
                "end_to_end_p95_ms": 80,
                "calibration_id": "cal-test",
                "factor": 2,
                "label": "estimated",
            },
        )
        self.assertNotIn("estimated_pi5", report)
        with self.assertRaisesRegex(ValueError, "no factor"):
            calibration_factor(calibration, "mac-a76compat")
        with self.assertRaises(ValueError):
            calibration_factor({"factor": 0}, "xcng-n1")

    def test_calibration_anchors_mean_and_checks_model_precision_and_size(self) -> None:
        reference: dict[str, JsonValue] = {
            "host_kind": "xcng-n1",
            "host": "xcng0",
            "model_id": "yolo11n-coco",
            "format": "ncnn-fp32",
            "input_hw": [640, 640],
            "inference_ms": {"mean": 47.015, "p95": 70},
        }
        calibrated = make_calibration(reference, "https://official.example/guide")
        self.assertAlmostEqual(float(str(calibrated["factor"])), 2)
        invalid_fields: list[tuple[str, JsonValue]] = [
            ("format", "ncnn-fp16"),
            ("input_hw", [352, 608]),
        ]
        for key, value in invalid_fields:
            changed: dict[str, JsonValue] = {**reference, key: value}
            with self.assertRaisesRegex(ValueError, "FP32 at 640x640"):
                make_calibration(changed, "source")

    def test_processing_samples_scale_and_summary_fallback(self) -> None:
        values, _ = processing_samples({"samples_ms": [10, 20]}, 2)
        self.assertEqual(values, [20, 40])
        fallback, _ = processing_samples({"end_to_end_ms": {"p50": 10, "p95": 10}}, 2)
        self.assertTrue(all(abs(value - 20) < 1e-10 for value in fallback))

    def test_letterbox_rgb_normalisation_and_rectangular_shape(self) -> None:
        frame = numpy.zeros((32, 64, 3), dtype=numpy.uint8)
        frame[:, :, 2] = 255
        image = letterbox(frame, (64, 64))
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertEqual(image[0, 32, 32], 1)
        self.assertEqual(image[2, 32, 32], 0)
        self.assertAlmostEqual(float(image[0, 0, 0]), 114 / 255)

    def test_one_to_one_topk_can_select_multiple_classes_for_one_anchor(self) -> None:
        prediction = numpy.asarray([[50], [50], [20], [20], [0.9], [0.8]], dtype=numpy.float32)
        detections = decode(prediction, nms_free=True)
        self.assertEqual(len(detections), 2)
        self.assertEqual(list(detections[:, -1]), [0, 1])
        self.assertEqual(len(decode(prediction)), 1)

    def test_decode_nms_is_class_aware_and_scores_are_already_sigmoid(self) -> None:
        prediction = numpy.asarray(
            [
                [50, 50, 50],
                [50, 50, 50],
                [20, 20, 20],
                [20, 20, 20],
                [0.9, 0.8, 0.1],
                [0.1, 0.1, 0.9],
            ],
            dtype=numpy.float32,
        )
        self.assertEqual(len(decode(prediction)), 2)
        self.assertEqual(len(decode(prediction, nms_free=True)), 3)
        self.assertEqual(len(decode(prediction, confidence=0.95)), 0)


class ReactionTests(unittest.TestCase):
    settings = ReactionSettings(trials=2000, seed=17)

    def test_single_perfect_hit_equals_sensor_plus_wait_plus_processing(self) -> None:
        result = simulate_reaction(
            [20],
            replace(self.settings, recall=1, k=1, n=1, sensor_min_periods=1, sensor_max_periods=1),
        )
        timings = result["reaction_ms"]
        assert isinstance(timings, dict)
        self.assertAlmostEqual(float(str(timings["p50"])), 1000 / 30 + 20)
        self.assertEqual(result["mean_processed_visible_frames"], 1)
        self.assertEqual(result["mean_dropped_visible_frames"], 0)

    def test_seed_is_deterministic_and_lower_recall_is_slower(self) -> None:
        first = simulate_reaction([40, 60, 80], self.settings)
        self.assertEqual(first, simulate_reaction([40, 60, 80], self.settings))
        lower_recall = simulate_reaction([40, 60, 80], replace(self.settings, recall=0.5))
        first_timings = first["reaction_ms"]
        lower_timings = lower_recall["reaction_ms"]
        assert isinstance(first_timings, dict) and isinstance(lower_timings, dict)
        self.assertGreater(float(str(lower_timings["p95"])), float(str(first_timings["p95"])))

    def test_latest_buffer_drops_frames_and_fifo_builds_a_backlog(self) -> None:
        latest = simulate_reaction([100], self.settings)
        fifo = simulate_reaction([100], replace(self.settings, frame_policy="fifo"))
        self.assertGreater(float(str(latest["mean_dropped_visible_frames"])), 0)
        self.assertEqual(fifo["mean_dropped_visible_frames"], 0)

    def test_correlation_extends_the_tail(self) -> None:
        independent = simulate_reaction([50], self.settings)
        correlated = simulate_reaction([50], replace(self.settings, correlation=0.8))
        first_timings, second_timings = independent["reaction_ms"], correlated["reaction_ms"]
        assert isinstance(first_timings, dict) and isinstance(second_timings, dict)
        self.assertGreater(float(str(second_timings["p95"])), float(str(first_timings["p95"])))

    def test_invalid_settings_and_processing_are_rejected(self) -> None:
        for settings in (
            replace(self.settings, camera_fps=0),
            replace(self.settings, recall=0),
            replace(self.settings, k=4, n=3),
            replace(self.settings, correlation=1),
        ):
            with self.assertRaises(ValueError):
                simulate_reaction([20], settings)
        for values in ([], [0.0], [float("inf")]):
            with self.assertRaises(ValueError):
                simulate_reaction(values, self.settings)
