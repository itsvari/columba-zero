"""Greedy IoU tracking with k-of-n trigger confirmation per track (PLAN.md section 4)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy
from numpy.typing import NDArray

from .config import ConfirmationConfig
from .json_data import JsonValue
from .models import Detection

BoxArray = NDArray[numpy.float64]


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    track_id: int
    label: str
    first_seen_captured_at: float
    first_seen_frame: int
    confirmed_at: float
    frame_number: int
    frames_seen: int
    k: int
    n: int

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "track_id": self.track_id,
            "class": self.label,
            "first_seen_captured_at": self.first_seen_captured_at,
            "first_seen_frame": self.first_seen_frame,
            "confirmed_at": self.confirmed_at,
            "frame_number": self.frame_number,
            "frames_seen": self.frames_seen,
            "k": self.k,
            "n": self.n,
        }


@dataclass(slots=True)
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    first_seen_captured_at: float
    first_seen_frame: int
    hits: deque[bool]
    frames_seen: int = 1
    missed_frames: int = 0
    confirmed: bool = False


@dataclass(slots=True)
class TrackUpdate:
    detections: list[Detection]
    triggers: list[TriggerEvent] = field(default_factory=list)


def box_of(detection: Detection) -> tuple[float, float, float, float]:
    return detection.left_px, detection.top_px, detection.right_px, detection.bottom_px


def iou_matrix(first: BoxArray, second: BoxArray) -> BoxArray:
    """Intersection over union for every pair of xyxy boxes, shape (len(first), len(second))."""
    left = numpy.maximum(first[:, None, 0], second[None, :, 0])
    top = numpy.maximum(first[:, None, 1], second[None, :, 1])
    right = numpy.minimum(first[:, None, 2], second[None, :, 2])
    bottom = numpy.minimum(first[:, None, 3], second[None, :, 3])
    intersection = numpy.clip(right - left, 0, None) * numpy.clip(bottom - top, 0, None)
    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = first_area[:, None] + second_area[None, :] - intersection
    overlaps: BoxArray = numpy.divide(
        intersection, union, out=numpy.zeros_like(intersection), where=union > 0
    )
    return overlaps


def greedy_matches(
    track_boxes: BoxArray, detection_boxes: BoxArray, threshold: float
) -> list[tuple[int, int]]:
    """Pair tracks and detections by descending IoU, each used at most once."""
    if not len(track_boxes) or not len(detection_boxes):
        return []
    overlaps = iou_matrix(track_boxes, detection_boxes)
    candidates = numpy.argwhere(overlaps >= threshold)
    order = numpy.argsort(-overlaps[candidates[:, 0], candidates[:, 1]], kind="stable")
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for track_index, detection_index in candidates[order].tolist():
        if track_index not in used_tracks and detection_index not in used_detections:
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))
    return matches


class ConfirmationTracker:
    """Associates detections across frames and confirms trigger-class tracks.

    Association ignores class, so a bird whose class flickers keeps one track. A frame
    counts as a hit for confirmation only when the track's matched detection has a
    trigger class; a track confirms once, when it has k hits among its last n frames.
    Tracks missing for more than `max_age_frames` consecutive frames are dropped.
    """

    def __init__(
        self,
        config: ConfirmationConfig,
        trigger_classes: Iterable[str],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.trigger_classes = frozenset(trigger_classes)
        self.clock = clock
        self.tracks: list[Track] = []
        self._next_track_id = 1

    def update(
        self, detections: Sequence[Detection], frame_number: int, captured_at: float
    ) -> TrackUpdate:
        config = self.config
        track_boxes = numpy.array([track.box for track in self.tracks], dtype=numpy.float64)
        detection_boxes = numpy.array(
            [box_of(detection) for detection in detections], dtype=numpy.float64
        )
        matches = greedy_matches(
            track_boxes.reshape(-1, 4), detection_boxes.reshape(-1, 4), config.iou
        )
        matched_tracks = {track_index: detection_index for track_index, detection_index in matches}
        track_ids: list[int | None] = [None] * len(detections)
        confirming: list[tuple[Track, Detection]] = []
        surviving: list[Track] = []
        for track_index, track in enumerate(self.tracks):
            detection_index = matched_tracks.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                track.hits.append(False)
                if track.missed_frames <= config.max_age_frames:
                    surviving.append(track)
                continue
            detection = detections[detection_index]
            track.box = box_of(detection)
            track.missed_frames = 0
            track.frames_seen += 1
            track.hits.append(detection.label in self.trigger_classes)
            track_ids[detection_index] = track.track_id
            surviving.append(track)
            if not track.confirmed and sum(track.hits) >= config.k:
                confirming.append((track, detection))
        for detection_index, detection in enumerate(detections):
            if track_ids[detection_index] is not None:
                continue
            track = Track(
                self._next_track_id,
                box_of(detection),
                captured_at,
                frame_number,
                deque([detection.label in self.trigger_classes], maxlen=config.n),
            )
            self._next_track_id += 1
            track_ids[detection_index] = track.track_id
            surviving.append(track)
            if sum(track.hits) >= config.k:
                confirming.append((track, detection))
        self.tracks = surviving
        triggers: list[TriggerEvent] = []
        if confirming:
            confirmed_at = self.clock()
            for track, detection in confirming:
                track.confirmed = True
                triggers.append(
                    TriggerEvent(
                        track.track_id,
                        detection.label,
                        track.first_seen_captured_at,
                        track.first_seen_frame,
                        confirmed_at,
                        frame_number,
                        track.frames_seen,
                        config.k,
                        config.n,
                    )
                )
        return TrackUpdate(
            [
                replace(detection, track_id=track_id)
                for detection, track_id in zip(detections, track_ids, strict=True)
            ],
            triggers,
        )

    @property
    def confirmed_track_ids(self) -> frozenset[int]:
        return frozenset(track.track_id for track in self.tracks if track.confirmed)
