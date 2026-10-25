from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    captured_at: float  # Unix wall-clock seconds
    frame: Any  # OpenCV BGR array
    delivered_monotonic: float | None = None  # freshness, immune to wall-clock changes


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence,
                "bbox": [self.x1, self.y1, self.x2, self.y2]}
