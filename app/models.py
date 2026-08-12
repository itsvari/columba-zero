from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RuntimeState(StrEnum):
    SAFE = "SAFE"
    IDLE = "IDLE"
    VERIFYING = "VERIFYING"
    AIMING = "AIMING"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: Any


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class Target:
    pixel_x: float
    pixel_y: float
    member_count: int
    total_area: float


@dataclass(frozen=True, slots=True)
class AimCommand:
    yaw: float
    pitch: float
    issued_at: float


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    reason: str = "ok"
    critical: bool = False
