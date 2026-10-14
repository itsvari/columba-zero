"""D-FINE deploy preprocessing and fixed-query decoder."""

from __future__ import annotations

import numpy
from PIL import Image

from training.adapters.postprocess import Frame
from training.adapters.postprocess import Values as InferenceValues
from training.adapters.postprocess import decode_queries as decode

__all__ = ["decode", "preprocess"]
PIXEL_MAXIMUM = 255.0


def preprocess(frame: Frame, input_hw: tuple[int, int]) -> InferenceValues:
    image = Image.fromarray(frame[:, :, ::-1]).resize(
        (input_hw[1], input_hw[0]), resample=Image.Resampling.BILINEAR
    )
    return (
        numpy.ascontiguousarray(numpy.asarray(image).transpose(2, 0, 1), dtype=numpy.float32)
        / PIXEL_MAXIMUM
    )
