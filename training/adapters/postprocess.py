"""NumPy postprocessing shared by the challenger export adapters."""

from __future__ import annotations

import numpy
from numpy.typing import NDArray

Values = NDArray[numpy.float32]
Frame = NDArray[numpy.uint8]
BOX_COORDINATES = 4
DETR_MAXIMUM_DETECTIONS = 300


def sigmoid(logits: Values) -> Values:
    exponentials = numpy.exp(-numpy.abs(logits))
    return numpy.asarray(
        numpy.where(logits >= 0, 1 / (1 + exponentials), exponentials / (1 + exponentials)),
        dtype=numpy.float32,
    )


def decode_queries(
    prediction: Values, camera_hw: tuple[int, int], confidence: float = 0.05
) -> Values:
    """Decode packed Q×(4+C) normalised cxcywh and class logits, without NMS."""
    if prediction.ndim != 2 or prediction.shape[1] <= BOX_COORDINATES:
        raise ValueError("expected Q×(4+C) normalised cxcywh and class logits")
    probabilities = sigmoid(prediction[:, BOX_COORDINATES:])
    flattened = probabilities.reshape(-1)
    ranked = numpy.argsort(-flattened, kind="stable")[:DETR_MAXIMUM_DETECTIONS]
    ranked = ranked[flattened[ranked] >= confidence]
    query_indices = ranked // probabilities.shape[1]
    centres = prediction[query_indices, :2]
    sizes = prediction[query_indices, 2:BOX_COORDINATES]
    boxes = numpy.concatenate((centres - sizes / 2, centres + sizes / 2), axis=1)
    scale_xy = numpy.asarray([camera_hw[1], camera_hw[0]] * 2, dtype=numpy.float32)
    return numpy.asarray(
        numpy.column_stack((boxes * scale_xy, flattened[ranked], ranked % probabilities.shape[1])),
        dtype=numpy.float32,
    )


def multiclass_nms(
    boxes: Values, scores: Values, confidence: float, overlap_threshold: float, maximum: int
) -> Values:
    point_indices, class_indices = numpy.nonzero(scores > confidence)
    selected_scores = scores[point_indices, class_indices]
    selected_boxes = boxes[point_indices]
    order = numpy.argsort(-selected_scores, kind="stable")
    areas = numpy.maximum(0, selected_boxes[:, 2:] - selected_boxes[:, :2]).prod(axis=1)
    kept: list[int] = []
    while len(order) and len(kept) < maximum:
        current = int(order[0])
        kept.append(current)
        others = order[1:]
        intersection_hw = numpy.maximum(
            0,
            numpy.minimum(selected_boxes[current, 2:], selected_boxes[others, 2:])
            - numpy.maximum(selected_boxes[current, :2], selected_boxes[others, :2]),
        )
        intersections = intersection_hw.prod(axis=1)
        unions = areas[current] + areas[others] - intersections
        overlaps = intersections / numpy.maximum(unions, numpy.finfo(numpy.float32).eps)
        order = others[
            (overlaps <= overlap_threshold) | (class_indices[others] != class_indices[current])
        ]
    return numpy.asarray(
        numpy.column_stack((selected_boxes[kept], selected_scores[kept], class_indices[kept])),
        dtype=numpy.float32,
    )
