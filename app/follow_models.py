"""Immutable observations shared by tracking, selection, control, and output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .json_data import JsonValue
from .models import Detection
from .tracking import TriggerEvent

TargetState = Literal["acquiring", "tracking", "waiting", "lost"]
TargetEventName = Literal["target_acquired", "target_switched", "target_lost"]


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    track_id: int
    detection: Detection
    confirmed: bool
    matched: bool
    observed_at_s: float
    smoothed_area_px2: float

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "track_id": self.track_id,
            "detection": self.detection.as_dict(),
            "confirmed": self.confirmed,
            "matched": self.matched,
            "observed_at_s": self.observed_at_s,
            "smoothed_area_px2": self.smoothed_area_px2,
        }


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    detections: tuple[Detection, ...]
    triggers: tuple[TriggerEvent, ...]
    tracks: tuple[TrackSnapshot, ...]


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    track: TrackSnapshot | None = None
    state: TargetState = "acquiring"
    selected_at_s: float | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "track": self.track.as_dict() if self.track is not None else None,
            "state": self.state,
            "selected_at_s": self.selected_at_s,
        }


@dataclass(frozen=True, slots=True)
class TargetEvent:
    name: TargetEventName
    track_id: int
    at_s: float
    previous_track_id: int | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "track_id": self.track_id,
            "at_s": self.at_s,
            "previous_track_id": self.previous_track_id,
        }


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    target: TargetSnapshot
    events: tuple[TargetEvent, ...] = ()
