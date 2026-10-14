"""NanoDet-Plus-m BGR warp preprocessing and GFL distribution decoder."""

from __future__ import annotations

import math

import numpy

from training.adapters.postprocess import Frame, Values, multiclass_nms, sigmoid
from training.adapters.postprocess import Values as InferenceValues

CLASS_COUNT = 80
REGRESSION_BINS = 8
BOX_SIDES = 4
STRIDES = (8, 16, 32, 64)
NMS_IOU = 0.6
MAXIMUM_DETECTIONS = 100
PIXEL_MAXIMUM = 255.0
NORMALISE_MEAN = numpy.asarray([103.53, 116.28, 123.675], dtype=numpy.float32)
NORMALISE_STD = numpy.asarray([57.375, 57.12, 58.395], dtype=numpy.float32)


def preprocess(frame: Frame, input_hw: tuple[int, int]) -> InferenceValues:
    import cv2

    transform = numpy.diag([input_hw[1] / frame.shape[1], input_hw[0] / frame.shape[0], 1])
    resized = cv2.warpPerspective(frame, transform, (input_hw[1], input_hw[0]))
    normalised = resized.astype(numpy.float32) / PIXEL_MAXIMUM - NORMALISE_MEAN / PIXEL_MAXIMUM
    normalised /= NORMALISE_STD / PIXEL_MAXIMUM
    return numpy.ascontiguousarray(normalised.transpose(2, 0, 1))


def centre_priors(input_hw: tuple[int, int]) -> Values:
    levels: list[Values] = []
    for stride in STRIDES:
        height_cells, width_cells = (math.ceil(size / stride) for size in input_hw)
        horizontal, vertical = numpy.meshgrid(numpy.arange(width_cells), numpy.arange(height_cells))
        levels.append(
            numpy.asarray(
                numpy.column_stack(
                    (
                        horizontal.reshape(-1) * stride,
                        vertical.reshape(-1) * stride,
                        numpy.full(horizontal.size, stride),
                    )
                ),
                dtype=numpy.float32,
            )
        )
    return numpy.concatenate(levels)


def decode(
    prediction: Values,
    camera_hw: tuple[int, int],
    input_hw: tuple[int, int],
    confidence: float = 0.05,
) -> Values:
    priors = centre_priors(input_hw)
    expected_shape = (len(priors), CLASS_COUNT + BOX_SIDES * REGRESSION_BINS)
    if prediction.shape != expected_shape:
        raise ValueError(f"expected raw NanoDet logits {expected_shape}, got {prediction.shape}")
    distributions = prediction[:, CLASS_COUNT:].reshape(-1, BOX_SIDES, REGRESSION_BINS)
    exponentials = numpy.exp(distributions - distributions.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    distances_px = (probabilities * numpy.arange(REGRESSION_BINS)).sum(axis=-1)
    distances_px *= priors[:, 2, None]
    boxes = numpy.concatenate(
        (priors[:, :2] - distances_px[:, :2], priors[:, :2] + distances_px[:, 2:]), axis=1
    ).astype(numpy.float32)
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, input_hw[1])
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, input_hw[0])
    detections = multiclass_nms(
        boxes, sigmoid(prediction[:, :CLASS_COUNT]), confidence, NMS_IOU, MAXIMUM_DETECTIONS
    )
    detections[:, :BOX_SIDES] *= numpy.asarray(
        [camera_hw[1] / input_hw[1], camera_hw[0] / input_hw[0]] * 2, dtype=numpy.float32
    )
    return detections
