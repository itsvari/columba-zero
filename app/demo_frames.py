"""An encoded processed frame shared by browser delivery and recording."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .json_data import JsonValue

FrameTimebase = Literal["delivery_monotonic", "replay_source", "processing_monotonic"]


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    sequence: int
    captured_at: float
    monotonic_s: float
    timebase: FrameTimebase
    jpeg: bytes
    metadata: Mapping[str, JsonValue]
