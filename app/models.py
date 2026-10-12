from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy
from numpy.typing import NDArray

from .json_data import JsonValue

Frame: TypeAlias = NDArray[numpy.uint8]


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: Frame


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    left_px: float
    top_px: float
    right_px: float
    bottom_px: float

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": [self.left_px, self.top_px, self.right_px, self.bottom_px],
        }
