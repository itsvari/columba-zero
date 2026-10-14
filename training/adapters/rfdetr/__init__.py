"""RF-DETR RGB ImageNet normalisation and fixed-query decoder."""

from __future__ import annotations

import numpy

from training.adapters.dfine import preprocess as resize_rgb
from training.adapters.postprocess import Frame
from training.adapters.postprocess import Values as InferenceValues
from training.adapters.postprocess import decode_queries as decode

__all__ = ["decode", "preprocess"]
IMAGENET_MEAN = numpy.asarray([0.485, 0.456, 0.406], dtype=numpy.float32)[:, None, None]
IMAGENET_STD = numpy.asarray([0.229, 0.224, 0.225], dtype=numpy.float32)[:, None, None]


def preprocess(frame: Frame, input_hw: tuple[int, int]) -> InferenceValues:
    return numpy.ascontiguousarray((resize_rgb(frame, input_hw) - IMAGENET_MEAN) / IMAGENET_STD)
