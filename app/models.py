from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy
from numpy.typing import NDArray

from .json_data import JsonValue

Frame: TypeAlias = NDArray[numpy.uint8]


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A delivered frame. `captured_at` is Unix time at delivery to the application.

    `delivered_monotonic_s` is the same moment on the monotonic clock, `delivery_ms`
    the time from the start of sensor exposure to delivery when the camera reports
    it, and `lux` the camera's light estimate.
    """

    sequence: int
    captured_at: float
    frame: Frame
    delivered_monotonic_s: float | None = None
    camera: str | None = None
    lux: float | None = None
    delivery_ms: float | None = None


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    left_px: float
    top_px: float
    right_px: float
    bottom_px: float
    class_id: int = 0
    track_id: int | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": [self.left_px, self.top_px, self.right_px, self.bottom_px],
            "class_id": self.class_id,
            "track_id": self.track_id,
        }
