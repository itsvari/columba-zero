from __future__ import annotations

import unittest

try:
    import numpy
except ImportError:
    numpy = None  # type: ignore[assignment]


@unittest.skipIf(numpy is None, "optional NumPy dependency missing")
class ChallengerTests(unittest.TestCase):
    def test_detr_scores_are_sigmoid_and_topk_keeps_class_pairs_without_nms(self) -> None:
        from training.adapters import dfine, rfdetr

        prediction = numpy.asarray(
            [[0.5, 0.5, 0.25, 0.5, 3, 2], [0.5, 0.5, 0.25, 0.5, 1, -10]], dtype=numpy.float32
        )
        expected_box = [240, 120, 400, 360]
        for decoder in (dfine.decode, rfdetr.decode):
            detections = decoder(prediction, (480, 640), confidence=0.5)
            self.assertEqual(detections.shape, (3, 6))
            numpy.testing.assert_allclose(detections[:, :4], [expected_box] * 3)
            numpy.testing.assert_allclose(
                detections[:, 4], [0.952574, 0.880797, 0.731059], rtol=1e-5
            )
            self.assertEqual(detections[:, 5].tolist(), [0, 1, 0])

    def test_detr_empty_and_invalid_layout(self) -> None:
        from training.adapters.dfine import decode

        prediction = numpy.asarray([[0.5, 0.5, 0.1, 0.1, -20]], dtype=numpy.float32)
        self.assertEqual(decode(prediction, (480, 640), 0.5).shape, (0, 6))
        with self.assertRaisesRegex(ValueError, "Q×"):
            decode(numpy.zeros((4, 10, 2), dtype=numpy.float32), (480, 640))

    def test_sigmoid_handles_extreme_logits_without_overflow(self) -> None:
        from training.adapters.postprocess import sigmoid

        with numpy.errstate(over="raise"):
            actual = sigmoid(numpy.asarray([-1000, 0, 1000], dtype=numpy.float32))
        numpy.testing.assert_array_equal(actual, [0, 0.5, 1])

    def test_nanodet_priors_include_ceil_stride64_and_no_half_cell_offset(self) -> None:
        from training.adapters.nanodet import centre_priors

        priors = centre_priors((352, 608))
        self.assertEqual(priors.shape, (4449, 3))
        numpy.testing.assert_array_equal(priors[0], [0, 0, 8])
        numpy.testing.assert_array_equal(priors[-1], [576, 320, 64])

    def test_nanodet_distribution_decode_clips_and_maps_original_frame(self) -> None:
        from training.adapters import nanodet

        input_hw = (32, 64)
        priors = nanodet.centre_priors(input_hw)
        prediction = numpy.full((len(priors), 112), -20, dtype=numpy.float32)
        distributions = prediction[:, 80:].reshape(-1, 4, 8)
        distributions[:, :, 1] = 20
        prediction[10, 3] = 20
        detections = nanodet.decode(prediction, (64, 128), input_hw)
        self.assertEqual(detections.shape, (1, 6))
        numpy.testing.assert_allclose(detections[0], [16, 0, 48, 32, 1, 3], atol=1e-5)
        with self.assertRaisesRegex(ValueError, "raw NanoDet"):
            nanodet.decode(prediction[:-1], (64, 128), input_hw)

    def test_nanodet_nms_retains_overlapping_different_classes(self) -> None:
        from training.adapters.postprocess import multiclass_nms

        boxes = numpy.asarray([[0, 0, 10, 10]] * 2, dtype=numpy.float32)
        scores = numpy.asarray([[0.9, 0.8], [0.7, 0.1]], dtype=numpy.float32)
        detections = multiclass_nms(boxes, scores, 0.05, 0.6, 100)
        self.assertEqual(detections.shape, (2, 6))
        numpy.testing.assert_allclose(detections[:, 4], [0.9, 0.8])
        self.assertEqual(detections[:, 5].tolist(), [0, 1])

    def test_framework_normalisation_preserves_own_colour_order(self) -> None:
        try:
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("optional OpenCV/Pillow dependency missing")
        from training.adapters import dfine, nanodet, rfdetr

        frame = numpy.full((2, 2, 3), [0, 128, 255], dtype=numpy.uint8)
        numpy.testing.assert_allclose(dfine.preprocess(frame, (2, 2))[:, 0, 0], [1, 128 / 255, 0])
        expected_rgb = (numpy.asarray([1, 128 / 255, 0]) - [0.485, 0.456, 0.406]) / [
            0.229,
            0.224,
            0.225,
        ]
        numpy.testing.assert_allclose(
            rfdetr.preprocess(frame, (2, 2))[:, 0, 0], expected_rgb, rtol=1e-6
        )
        expected_bgr = (
            numpy.asarray([0, 128, 255]) - nanodet.NORMALISE_MEAN
        ) / nanodet.NORMALISE_STD
        numpy.testing.assert_allclose(
            nanodet.preprocess(frame, (2, 2))[:, 0, 0], expected_bgr, rtol=1e-6
        )

    def test_parity_matches_detections_by_class_and_geometry_and_rejects_missing(self) -> None:
        from training.adapters.verify_challenger import compare_detections

        expected = numpy.asarray(
            [[1, 2, 3, 4, 0.8, 0], [10, 20, 30, 40, 0.7, 1]], dtype=numpy.float32
        )
        self.assertEqual(compare_detections(expected[::-1], expected), (0, 0))
        with self.assertRaisesRegex(ValueError, "count"):
            compare_detections(expected[:1], expected)
        altered = expected.copy()
        altered[0, 5] = 2
        with self.assertRaisesRegex(ValueError, "classes"):
            compare_detections(altered, expected)


if __name__ == "__main__":
    unittest.main()
