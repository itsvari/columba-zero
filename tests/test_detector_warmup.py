from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.detector import DetectorError, UltralyticsDetector
from app.main import build_parser, run


class DetectorWarmupTests(unittest.TestCase):
    def make_detector(self) -> UltralyticsDetector:
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector._model = Mock()
        detector._model.predict.return_value = []
        detector.image_size = 416
        detector.confidence = 0.35
        return detector

    def test_warmup_runs_prediction_with_camera_shape(self) -> None:
        detector = self.make_detector()
        detector.warmup(640, 480)
        kwargs = detector._model.predict.call_args.kwargs
        self.assertEqual(kwargs["source"].shape, (480, 640, 3))
        self.assertEqual(kwargs["source"].dtype, np.uint8)
        self.assertEqual(kwargs["imgsz"], 416)
        self.assertEqual(kwargs["conf"], 0.35)

    def test_warmup_failure_leaves_camera_and_gpio_unopened(self) -> None:
        detector = self.make_detector()
        detector._model.predict.side_effect = RuntimeError("backend failed")
        args = build_parser().parse_args(["--hardware"])
        with (
            patch("app.main.UltralyticsDetector", type(detector)),
            patch.object(UltralyticsDetector, "__new__", return_value=detector),
            patch.object(UltralyticsDetector, "__init__", return_value=None),
            patch("app.main.EventLogger"),
            patch("app.main.Picamera2Capture") as camera,
            patch("app.main.GPIOInputs") as gpio,
        ):
            with self.assertRaisesRegex(DetectorError, "backend failed"):
                run(args)
            camera.assert_not_called()
            gpio.assert_not_called()
