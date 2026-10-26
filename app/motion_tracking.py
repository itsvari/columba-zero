"""Shared motion identities with confidence-gated confirmation and background flow."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy
from numpy.typing import NDArray

from .config import ConfirmationConfig
from .follow_models import TrackingUpdate, TrackSnapshot
from .models import Detection, Frame, FramePacket
from .tracking import TriggerEvent, box_of, iou_matrix

Coordinates = NDArray[numpy.float64]
STATE_POSITION_COUNT = 4
STATE_SIZE = STATE_POSITION_COUNT * 2
MINIMUM_BOX_SIZE_PX = 1.0
INITIAL_POSITION_VARIANCE_PX2 = 10.0
INITIAL_VELOCITY_VARIANCE_PX2_S2 = 10_000.0
MEASUREMENT_VARIANCE_PX2 = 4.0
# walking can start between frames; a stationary history must not lock velocity near zero.
ACCELERATION_VARIANCE_PX2_S4 = 40_000.0
AREA_OBSERVATION_WEIGHT = 0.25
UNAVAILABLE_ASSOCIATION_COST = 1_000_000.0
FLOW_MAXIMUM_WIDTH_PX = 640
FLOW_MAXIMUM_CORNERS = 200
FLOW_CORNER_QUALITY = 0.01
FLOW_CORNER_DISTANCE_PX = 8.0
FLOW_WINDOW_SIZE_PX = 21
FLOW_PYRAMID_LEVEL = 3
FLOW_MINIMUM_POINTS = 6
FLOW_FORWARD_BACKWARD_ERROR_PX = 1.5
FLOW_INLIER_DISTANCE_PX = 2.0
FLOW_MASK_MARGIN_PX = 5


def _measurement(detection: Detection) -> Coordinates:
    return numpy.array(
        [
            (detection.left_px + detection.right_px) / 2,
            (detection.top_px + detection.bottom_px) / 2,
            max(MINIMUM_BOX_SIZE_PX, detection.right_px - detection.left_px),
            max(MINIMUM_BOX_SIZE_PX, detection.bottom_px - detection.top_px),
        ],
        dtype=numpy.float64,
    )


def _area(detection: Detection) -> float:
    return max(0.0, detection.right_px - detection.left_px) * max(
        0.0, detection.bottom_px - detection.top_px
    )


@dataclass(slots=True)
class _MotionTrack:
    track_id: int
    state: Coordinates
    covariance: Coordinates
    predicted_at_s: float
    detection: Detection
    observed_at_s: float
    smoothed_area_px2: float
    first_seen_captured_at: float
    first_seen_frame: int
    hits: deque[bool]
    frames_seen: int = 1
    confirmed: bool = False
    matched: bool = True

    @property
    def predicted_box(self) -> tuple[float, float, float, float]:
        centre_x_px, centre_y_px, width_px, height_px = self.state[:STATE_POSITION_COUNT]
        return (
            float(centre_x_px - width_px / 2),
            float(centre_y_px - height_px / 2),
            float(centre_x_px + width_px / 2),
            float(centre_y_px + height_px / 2),
        )

    def predict(self, observed_at_s: float, shift_px: tuple[float, float]) -> None:
        elapsed_s = max(0.0, observed_at_s - self.predicted_at_s)
        transition = numpy.eye(STATE_SIZE)
        transition[:STATE_POSITION_COUNT, STATE_POSITION_COUNT:] = (
            numpy.eye(STATE_POSITION_COUNT) * elapsed_s
        )
        acceleration = numpy.concatenate(
            (
                numpy.eye(STATE_POSITION_COUNT) * elapsed_s**2 / 2,
                numpy.eye(STATE_POSITION_COUNT) * elapsed_s,
            )
        )
        self.state = transition @ self.state
        self.state[:2] += shift_px
        self.state[2:STATE_POSITION_COUNT] = numpy.maximum(
            self.state[2:STATE_POSITION_COUNT], MINIMUM_BOX_SIZE_PX
        )
        self.covariance = (
            transition @ self.covariance @ transition.T
            + acceleration @ acceleration.T * ACCELERATION_VARIANCE_PX2_S4
        )
        self.predicted_at_s = observed_at_s
        self.matched = False

    def observe(self, detection: Detection, observed_at_s: float, hit: bool) -> None:
        residual = _measurement(detection) - self.state[:STATE_POSITION_COUNT]
        measurement_noise = numpy.eye(STATE_POSITION_COUNT) * MEASUREMENT_VARIANCE_PX2
        innovation = (
            self.covariance[:STATE_POSITION_COUNT, :STATE_POSITION_COUNT] + measurement_noise
        )
        gain = numpy.linalg.solve(innovation, self.covariance[:STATE_POSITION_COUNT, :]).T
        self.state = self.state + gain @ residual
        correction = numpy.eye(STATE_SIZE)
        correction[:, :STATE_POSITION_COUNT] -= gain
        self.covariance = (
            correction @ self.covariance @ correction.T + gain @ measurement_noise @ gain.T
        )
        self.detection = replace(detection, track_id=self.track_id)
        self.observed_at_s = observed_at_s
        self.smoothed_area_px2 += AREA_OBSERVATION_WEIGHT * (
            _area(detection) - self.smoothed_area_px2
        )
        self.frames_seen += 1
        self.hits.append(hit)
        self.matched = True

    def snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            self.track_id,
            self.detection,
            self.confirmed,
            self.matched,
            self.observed_at_s,
            self.smoothed_area_px2,
        )


class _CameraMotion:
    def __init__(self) -> None:
        self.previous_frame: Frame | None = None
        self.previous_mask: Frame | None = None

    def update(
        self, frame: Frame, boxes: Sequence[tuple[float, float, float, float]]
    ) -> tuple[float, float]:
        import cv2

        height_px, width_px = frame.shape[:2]
        scale = min(1.0, FLOW_MAXIMUM_WIDTH_PX / width_px)
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cast(Frame, cv2.resize(grey, (round(width_px * scale), round(height_px * scale))))
        mask = numpy.full(grey.shape, 255, dtype=numpy.uint8)
        for left_px, top_px, right_px, bottom_px in boxes:
            left = max(0, math.floor(left_px * scale) - FLOW_MASK_MARGIN_PX)
            top = max(0, math.floor(top_px * scale) - FLOW_MASK_MARGIN_PX)
            right = max(0, math.ceil(right_px * scale) + FLOW_MASK_MARGIN_PX)
            bottom = max(0, math.ceil(bottom_px * scale) + FLOW_MASK_MARGIN_PX)
            mask[top:bottom, left:right] = 0
        shift_px = (0.0, 0.0)
        if self.previous_frame is not None and self.previous_frame.shape == grey.shape:
            shift_px = self._estimate(grey, mask, scale)
        self.previous_frame = grey
        self.previous_mask = mask
        return shift_px

    def _estimate(self, grey: Frame, mask: Frame, scale: float) -> tuple[float, float]:
        import cv2

        assert self.previous_frame is not None
        points = cv2.goodFeaturesToTrack(
            self.previous_frame,
            maxCorners=FLOW_MAXIMUM_CORNERS,
            qualityLevel=FLOW_CORNER_QUALITY,
            minDistance=FLOW_CORNER_DISTANCE_PX,
            mask=self.previous_mask,
        )
        if points is None or len(points) < FLOW_MINIMUM_POINTS:
            return 0.0, 0.0
        window_size = (FLOW_WINDOW_SIZE_PX, FLOW_WINDOW_SIZE_PX)
        following, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            grey,
            points,
            numpy.empty_like(points),
            winSize=window_size,
            maxLevel=FLOW_PYRAMID_LEVEL,
        )
        if following is None or status is None:
            return 0.0, 0.0
        preceding, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            grey,
            self.previous_frame,
            following,
            numpy.empty_like(following),
            winSize=window_size,
            maxLevel=FLOW_PYRAMID_LEVEL,
        )
        if preceding is None or reverse_status is None:
            return 0.0, 0.0
        original = points.reshape(-1, 2)
        moved = following.reshape(-1, 2)
        valid = (
            (status.ravel() == 1)
            & (reverse_status.ravel() == 1)
            & numpy.isfinite(moved).all(axis=1)
            & (
                numpy.linalg.norm(preceding.reshape(-1, 2) - original, axis=1)
                <= FLOW_FORWARD_BACKWARD_ERROR_PX
            )
        )
        height_px, width_px = mask.shape
        destinations = numpy.rint(numpy.nan_to_num(moved)).astype(numpy.int64)
        inside = (
            (destinations[:, 0] >= 0)
            & (destinations[:, 0] < width_px)
            & (destinations[:, 1] >= 0)
            & (destinations[:, 1] < height_px)
        )
        destinations[:, 0] = numpy.clip(destinations[:, 0], 0, width_px - 1)
        destinations[:, 1] = numpy.clip(destinations[:, 1], 0, height_px - 1)
        valid &= inside & (mask[destinations[:, 1], destinations[:, 0]] != 0)
        displacements = (moved - original)[valid]
        if len(displacements) < FLOW_MINIMUM_POINTS:
            return 0.0, 0.0
        median = numpy.median(displacements, axis=0)
        inliers = displacements[
            numpy.linalg.norm(displacements - median, axis=1) <= FLOW_INLIER_DISTANCE_PX
        ]
        if len(inliers) < FLOW_MINIMUM_POINTS:
            return 0.0, 0.0
        shift = numpy.median(inliers, axis=0) / scale
        return float(shift[0]), float(shift[1])


def _associate(
    tracks: Sequence[_MotionTrack],
    detections: Sequence[Detection],
    indices: Sequence[int],
    minimum_overlap: float,
) -> list[tuple[_MotionTrack, int]]:
    if not tracks or not indices:
        return []
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    predictions = numpy.array([track.predicted_box for track in tracks], dtype=numpy.float64)
    observations = numpy.array(
        [box_of(detections[index]) for index in indices], dtype=numpy.float64
    )
    overlaps = iou_matrix(predictions, observations)
    # dummy columns allow every track to remain unmatched without blocking a valid pair.
    unmatched_cost = float(len(tracks) + 1)
    costs = numpy.full((len(tracks), len(indices) + len(tracks)), unmatched_cost)
    costs[:, : len(indices)] = numpy.where(
        overlaps >= minimum_overlap, 1.0 - overlaps, UNAVAILABLE_ASSOCIATION_COST
    )
    assign = cast(
        Callable[[Coordinates], tuple[NDArray[numpy.intp], NDArray[numpy.intp]]],
        linear_sum_assignment,
    )
    rows, columns = assign(costs)
    return [
        (tracks[int(row)], indices[int(column)])
        for row, column in zip(rows, columns, strict=True)
        if column < len(indices) and overlaps[row, column] >= minimum_overlap
    ]


class MotionTracker:
    """Predict identities at source time; emit triggers only from acquisition hits.

    `now_s` and `observed_at_s` share the live monotonic or replay source clock.
    Predictions are internal. Unmatched snapshots retain the last real observation.
    """

    def __init__(
        self,
        confirmation: ConfirmationConfig,
        trigger_classes: Iterable[str],
        acquisition_confidence: float,
        maintenance_confidence: float = 0.15,
        retention_s: float = 0.75,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0 <= maintenance_confidence <= acquisition_confidence <= 1:
            raise ValueError(
                "confidence thresholds must satisfy 0 <= maintenance <= acquisition <= 1"
            )
        if not math.isfinite(retention_s) or retention_s <= 0:
            raise ValueError("retention_s must be finite and positive")
        self.confirmation = confirmation
        self.trigger_classes = frozenset(trigger_classes)
        self.acquisition_confidence = acquisition_confidence
        self.maintenance_confidence = maintenance_confidence
        self.retention_s = retention_s
        self.clock = clock
        self._tracks: list[_MotionTrack] = []
        self._next_track_id = 1
        self._camera_motion = _CameraMotion()

    def update(
        self,
        packet: FramePacket,
        detections: Sequence[Detection],
        now_s: float,
        observed_at_s: float,
    ) -> TrackingUpdate:
        self._tracks = [
            track for track in self._tracks if now_s - track.observed_at_s <= self.retention_s
        ]
        masks = [box_of(detection) for detection in detections]
        masks.extend(track.predicted_box for track in self._tracks)
        shift_px = self._camera_motion.update(packet.frame, masks)
        for track in self._tracks:
            track.predict(observed_at_s, shift_px)
        high = [
            index
            for index, detection in enumerate(detections)
            if detection.confidence >= self.acquisition_confidence
        ]
        low = [
            index
            for index, detection in enumerate(detections)
            if self.maintenance_confidence <= detection.confidence < self.acquisition_confidence
        ]
        matches = _associate(self._tracks, detections, high, self.confirmation.iou)
        matched_ids = {track.track_id for track, _ in matches}
        remaining = [track for track in self._tracks if track.track_id not in matched_ids]
        matches.extend(_associate(remaining, detections, low, self.confirmation.iou))
        matched_indices = {index for _, index in matches}
        assigned = [replace(detection, track_id=None) for detection in detections]
        for track, index in matches:
            detection = detections[index]
            hit = detection.confidence >= self.acquisition_confidence and (
                detection.label in self.trigger_classes
            )
            track.observe(detection, observed_at_s, hit)
            assigned[index] = track.detection
        for track in self._tracks:
            if not track.matched:
                track.hits.append(False)
        for index in high:
            if index not in matched_indices:
                track = self._acquire(packet, detections[index], observed_at_s)
                self._tracks.append(track)
                assigned[index] = track.detection
        triggers = self._confirm(packet)
        return TrackingUpdate(
            tuple(assigned), tuple(triggers), tuple(track.snapshot() for track in self._tracks)
        )

    def _acquire(
        self, packet: FramePacket, detection: Detection, observed_at_s: float
    ) -> _MotionTrack:
        state = numpy.zeros(STATE_SIZE)
        state[:STATE_POSITION_COUNT] = _measurement(detection)
        covariance = numpy.diag(
            [INITIAL_POSITION_VARIANCE_PX2] * STATE_POSITION_COUNT
            + [INITIAL_VELOCITY_VARIANCE_PX2_S2] * STATE_POSITION_COUNT
        )
        track = _MotionTrack(
            self._next_track_id,
            state,
            covariance,
            observed_at_s,
            replace(detection, track_id=self._next_track_id),
            observed_at_s,
            _area(detection),
            packet.captured_at,
            packet.sequence,
            deque([detection.label in self.trigger_classes], maxlen=self.confirmation.n),
        )
        self._next_track_id += 1
        return track

    def _confirm(self, packet: FramePacket) -> list[TriggerEvent]:
        confirming = [
            track
            for track in self._tracks
            if not track.confirmed
            and track.matched
            and track.hits[-1]
            and sum(track.hits) >= self.confirmation.k
        ]
        if not confirming:
            return []
        confirmed_at = self.clock()
        triggers = []
        for track in confirming:
            track.confirmed = True
            triggers.append(
                TriggerEvent(
                    track.track_id,
                    track.detection.label,
                    track.first_seen_captured_at,
                    track.first_seen_frame,
                    confirmed_at,
                    packet.sequence,
                    track.frames_seen,
                    self.confirmation.k,
                    self.confirmation.n,
                )
            )
        return triggers

    def reset(self) -> None:
        """Clear view-specific state while preserving monotonically increasing IDs."""
        self._tracks = []
        self._camera_motion = _CameraMotion()

    @property
    def confirmed_track_ids(self) -> frozenset[int]:
        return frozenset(track.track_id for track in self._tracks if track.confirmed)
