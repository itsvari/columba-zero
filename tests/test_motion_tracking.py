from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy
import pytest

from app.config import ConfirmationConfig
from app.follow_models import TrackingUpdate
from app.models import Detection, Frame, FramePacket
from app.motion_tracking import MotionTracker

FRAME_WIDTH_PX = 320
FRAME_HEIGHT_PX = 240
CAPTURED_AT_S = 1_800_000_000.0
CONFIRMED_AT_S = CAPTURED_AT_S + 100


class OcclusionDetection(TypedDict):
    confidence: float
    bounds_px: list[float]


class OcclusionObservation(TypedDict):
    sequence: int
    observed_at_s: float
    processed_at_s: float
    detections: list[OcclusionDetection]


class OcclusionReplay(TypedDict):
    observations: list[OcclusionObservation]


def detection(
    left_px: float = 40,
    top_px: float = 60,
    width_px: float = 40,
    height_px: float = 40,
    confidence: float = 0.8,
    label: str = "pigeon",
    class_id: int = 0,
) -> Detection:
    return Detection(
        label,
        confidence,
        left_px,
        top_px,
        left_px + width_px,
        top_px + height_px,
        class_id=class_id,
    )


def tracker(k: int = 2, n: int = 3, retention_s: float = 0.75) -> MotionTracker:
    return MotionTracker(
        ConfirmationConfig(enabled=True, k=k, n=n),
        ["pigeon"],
        0.5,
        retention_s=retention_s,
        clock=lambda: CONFIRMED_AT_S,
    )


def update(
    motion_tracker: MotionTracker,
    detections: list[Detection],
    at_s: float,
    sequence: int = 0,
    frame: Frame | None = None,
    now_s: float | None = None,
) -> TrackingUpdate:
    if frame is None:
        frame = numpy.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=numpy.uint8)
    packet = FramePacket(sequence, CAPTURED_AT_S + sequence, frame)
    return motion_tracker.update(packet, detections, at_s if now_s is None else now_s, at_s)


def identities(result: TrackingUpdate) -> list[int | None]:
    return [item.track_id for item in result.detections]


def test_partial_occlusion_replay_preserves_identity_when_bird_reappears() -> None:
    fixture = Path(__file__).parent / "fixtures" / "monitor_partial_occlusion.json"
    replay = cast(OcclusionReplay, json.loads(fixture.read_text()))
    motion_tracker = MotionTracker(
        ConfirmationConfig(enabled=True, k=2, n=3, iou=0.3),
        ["pigeon"],
        acquisition_confidence=0.62646484375,
    )
    frame = numpy.zeros((450, 800, 3), dtype=numpy.uint8)
    trigger_count = 0
    for observation in replay["observations"]:
        detections = []
        for item in observation["detections"]:
            left_px, top_px, right_px, bottom_px = item["bounds_px"]
            detections.append(
                Detection("pigeon", item["confidence"], left_px, top_px, right_px, bottom_px)
            )
        result = update(
            motion_tracker,
            detections,
            observation["observed_at_s"],
            observation["sequence"],
            frame,
            now_s=observation["processed_at_s"],
        )
        trigger_count += len(result.triggers)
        assert identities(result) == [1] * len(detections), observation["sequence"]
    assert trigger_count == 1


@pytest.mark.parametrize(
    ("frame_rate_fps", "walking_speed_px_s", "box_size_px"),
    ((15.0, 120.0, 40.0), (15.0, 200.0, 80.0), (30.0, 120.0, 40.0), (30.0, 200.0, 80.0)),
)
@pytest.mark.parametrize("walking_confidence", (0.8, 0.3))
def test_walking_from_rest_keeps_the_confirmed_identity(
    frame_rate_fps: float,
    walking_speed_px_s: float,
    box_size_px: float,
    walking_confidence: float,
) -> None:
    motion_tracker = tracker()
    frame = numpy.zeros((450, 800, 3), dtype=numpy.uint8)
    stationary_duration_s = 1.0
    walking_duration_s = 2.0
    stationary_frames = round(frame_rate_fps * stationary_duration_s)
    frame_count = round(frame_rate_fps * (stationary_duration_s + walking_duration_s))
    observed_identities: set[int | None] = set()
    for sequence in range(frame_count):
        walking_frames = max(0, sequence - stationary_frames + 1)
        left_px = 100.0 + walking_frames * walking_speed_px_s / frame_rate_fps
        result = update(
            motion_tracker,
            [
                detection(
                    left_px=left_px,
                    width_px=box_size_px,
                    height_px=box_size_px,
                    confidence=walking_confidence if walking_frames else 0.8,
                )
            ],
            sequence / frame_rate_fps,
            sequence=sequence,
            frame=frame,
        )
        observed_identities.add(result.detections[0].track_id)
    assert observed_identities == {1}


def test_confirmation_reuses_event_shape_and_shared_identity() -> None:
    motion_tracker = tracker()
    first = update(motion_tracker, [detection()], 0.0, sequence=10)
    gap = update(motion_tracker, [], 0.1, sequence=11)
    confirmed = update(motion_tracker, [detection(left_px=42)], 0.2, sequence=12)
    repeated = update(motion_tracker, [detection(left_px=44)], 0.3, sequence=13)
    assert not first.triggers and not gap.triggers and not repeated.triggers
    assert confirmed.triggers[0].as_dict() == {
        "track_id": 1,
        "class": "pigeon",
        "first_seen_captured_at": CAPTURED_AT_S + 10,
        "first_seen_frame": 10,
        "confirmed_at": CONFIRMED_AT_S,
        "frame_number": 12,
        "frames_seen": 2,
        "k": 2,
        "n": 3,
    }
    assert confirmed.tracks[0].track_id == confirmed.detections[0].track_id == 1
    assert confirmed.tracks[0].detection == confirmed.detections[0]
    assert motion_tracker.confirmed_track_ids == {1}


def test_low_confidence_cannot_acquire_or_reuse_incoming_identity() -> None:
    motion_tracker = tracker(k=1, n=1)
    low = replace(detection(confidence=0.3), track_id=99)
    for sequence in range(5):
        result = update(motion_tracker, [low], sequence / 10)
        assert identities(result) == [None]
        assert not result.tracks and not result.triggers
    assert identities(update(motion_tracker, [detection()], 0.5)) == [1]


def test_low_confidence_maintains_but_cannot_confirm_or_trigger() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 0.0)
    for sequence in range(1, 10):
        result = update(motion_tracker, [detection(confidence=0.15)], sequence / 10)
        assert identities(result) == [1]
        assert result.tracks[0].matched and not result.tracks[0].confirmed
        assert not result.triggers
    high = update(motion_tracker, [detection()], 1.0)
    assert not high.triggers
    confirmed = update(motion_tracker, [detection()], 1.1)
    assert len(confirmed.triggers) == 1
    maintained = update(motion_tracker, [detection(confidence=0.49)], 1.2)
    assert maintained.tracks[0].confirmed and not maintained.triggers


def test_low_hit_is_a_miss_in_confirmation_window() -> None:
    motion_tracker = tracker(k=2, n=2)
    update(motion_tracker, [detection()], 0.0)
    update(motion_tracker, [detection(confidence=0.3)], 0.1)
    assert not update(motion_tracker, [detection()], 0.2).triggers
    assert len(update(motion_tracker, [detection()], 0.3).triggers) == 1


def test_detection_below_maintenance_floor_does_not_refresh_observation() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 0.0)
    result = update(motion_tracker, [detection(confidence=0.149)], 0.5)
    assert identities(result) == [None]
    assert not result.tracks[0].matched
    assert result.tracks[0].observed_at_s == 0.0
    assert not update(motion_tracker, [], 0.751).tracks


def test_high_confidence_association_has_priority_over_low_confidence() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 0.0)
    result = update(motion_tracker, [detection(left_px=45), detection(confidence=0.3)], 0.1)
    assert identities(result) == [1, None]
    assert len(result.tracks) == 1


def test_hungarian_assignment_preserves_both_eligible_matches() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection(left_px=0), detection(left_px=20)], 0.0)
    result = update(motion_tracker, [detection(left_px=8), detection(left_px=-10)], 0.1)
    assert identities(result) == [2, 1]
    assert len(result.tracks) == 2


def test_unmatched_rows_cannot_block_a_valid_hungarian_match() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection(left_px=0), detection(left_px=200)], 0.0)
    result = update(motion_tracker, [detection(left_px=201), detection(left_px=100)], 0.1)
    assert identities(result) == [2, 3]
    assert not result.tracks[0].matched


def test_source_time_prediction_bridges_gap_beyond_last_box_overlap() -> None:
    motion_tracker = tracker()
    for sequence, left_px in enumerate([0, 10, 20]):
        assert identities(update(motion_tracker, [detection(left_px=left_px)], sequence / 10)) == [
            1
        ]
    missing = update(motion_tracker, [], 0.3)
    assert not missing.tracks[0].matched
    assert missing.tracks[0].detection.left_px == 20
    assert missing.tracks[0].observed_at_s == 0.2
    result = update(motion_tracker, [detection(left_px=60)], 0.6)
    assert identities(result) == [1]
    assert result.tracks[0].detection.left_px == 60
    assert result.tracks[0].observed_at_s == 0.6


def test_prediction_keeps_crossing_identities_when_detection_order_reverses() -> None:
    motion_tracker = tracker()
    for sequence in range(4):
        result = update(
            motion_tracker,
            [detection(left_px=sequence * 10), detection(left_px=100 - sequence * 10)],
            sequence / 10,
        )
        assert identities(result) == [1, 2]
    result = update(motion_tracker, [detection(left_px=40), detection(left_px=60)], 0.6)
    assert identities(result) == [2, 1]


def test_retention_expires_by_seconds_even_with_adjacent_frame_numbers() -> None:
    motion_tracker = tracker(k=1, n=1)
    update(motion_tracker, [detection()], 5.0, sequence=0)
    assert update(motion_tracker, [], 5.75, sequence=1).tracks
    assert not update(motion_tracker, [], 5.751, sequence=2).tracks
    assert not motion_tracker.confirmed_track_ids
    assert identities(update(motion_tracker, [detection()], 5.8, sequence=3)) == [2]


def test_retention_uses_observation_age_including_delivery_delay() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 10.0, now_s=10.2)
    result = update(motion_tracker, [], 10.7, now_s=10.751)
    assert not result.tracks


def test_sequence_and_capture_wall_time_do_not_drive_prediction_or_retention() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 0.0, sequence=1)
    result = update(motion_tracker, [detection()], 0.1, sequence=10_000)
    assert identities(result) == [1]
    assert result.tracks[0].confirmed


def test_class_flicker_retains_identity_and_counts_trigger_class_only() -> None:
    motion_tracker = tracker()
    dove = detection(label="zebra_dove", class_id=2)
    frames = [[dove], [detection()], [dove], [detection()]]
    results = [
        update(motion_tracker, frame, sequence / 10) for sequence, frame in enumerate(frames)
    ]
    assert [identities(result) for result in results] == [[1]] * 4
    assert [len(result.triggers) for result in results] == [0, 0, 0, 1]
    assert results[2].detections[0].class_id == 2
    assert results[3].triggers[0].label == "pigeon"


def test_area_is_smoothed_from_real_matches_only_and_snapshots_are_stable() -> None:
    motion_tracker = tracker()
    initial = update(motion_tracker, [detection()], 0.0)
    larger = update(motion_tracker, [detection(width_px=60)], 0.1)
    assert initial.tracks[0].smoothed_area_px2 == 1600
    assert 1600 < larger.tracks[0].smoothed_area_px2 < 2400
    missing = update(motion_tracker, [], 0.2)
    assert missing.tracks[0].smoothed_area_px2 == larger.tracks[0].smoothed_area_px2
    assert missing.tracks[0].detection == larger.detections[0]
    assert initial.tracks[0].detection.right_px == 80


def test_reset_preserves_id_sequence_and_clears_confirmation_and_motion() -> None:
    motion_tracker = tracker(k=1, n=1)
    initial = update(motion_tracker, [detection()], 100.0)
    motion_tracker.reset()
    assert not motion_tracker.confirmed_track_ids
    following = update(motion_tracker, [detection()], 0.0)
    assert identities(initial) == [1] and identities(following) == [2]
    assert following.tracks[0].confirmed


def textured_frame() -> Frame:
    generator = numpy.random.default_rng(42)
    grey = generator.integers(0, 256, (FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=numpy.uint8)
    return numpy.repeat(grey[:, :, None], 3, axis=2)


def translate(frame: Frame, right_px: float, down_px: float) -> Frame:
    transform = numpy.array([[1, 0, right_px], [0, 1, down_px]], dtype=numpy.float64)
    return cast(Frame, cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0])))


def test_real_sparse_optical_flow_compensates_camera_shift_without_box_overlap() -> None:
    motion_tracker = tracker()
    first_frame = textured_frame()
    first_bird = detection(left_px=100, top_px=100, width_px=20, height_px=20)
    update(motion_tracker, [first_bird], 0.0, frame=first_frame)
    next_frame = translate(first_frame, 24, -8)
    moved_bird = detection(left_px=124, top_px=92, width_px=20, height_px=20)
    result = update(motion_tracker, [moved_bird], 0.1, frame=next_frame)
    assert identities(result) == [1]
    assert result.tracks[0].confirmed
    assert result.tracks[0].detection.left_px == 124


def test_flow_masks_foreground_bird_motion() -> None:
    motion_tracker = tracker()
    generator = numpy.random.default_rng(13)
    background = generator.integers(0, 64, (480, 640, 3), dtype=numpy.uint8)
    foreground = generator.integers(0, 256, (140, 220, 3), dtype=numpy.uint8)
    first = background.copy()
    first[180:320, 200:420] = foreground
    bird = detection(left_px=200, top_px=180, width_px=220, height_px=140)
    distant_bird = detection(left_px=50, top_px=50, width_px=12, height_px=12)
    update(motion_tracker, [bird, distant_bird], 0.0, frame=first)
    following = translate(background, 24, 0)
    following[180:320, 170:390] = foreground
    moved_bird = detection(left_px=170, top_px=180, width_px=220, height_px=140)
    shifted_distant_bird = detection(left_px=74, top_px=50, width_px=12, height_px=12)
    result = update(motion_tracker, [moved_bird, shifted_distant_bird], 0.1, frame=following)
    assert identities(result) == [1, 2]


def test_blank_frames_and_changed_resolution_fall_back_without_flow() -> None:
    motion_tracker = tracker()
    update(motion_tracker, [detection()], 0.0)
    smaller = numpy.zeros((120, 160, 3), dtype=numpy.uint8)
    assert identities(update(motion_tracker, [detection()], 0.1, frame=smaller)) == [1]


@pytest.mark.parametrize(
    ("acquisition", "maintenance", "retention_s"),
    [
        (0.1, 0.15, 0.75),
        (1.1, 0.15, 0.75),
        (0.5, -0.1, 0.75),
        (0.5, 0.15, 0.0),
        (0.5, 0.15, float("nan")),
    ],
)
def test_invalid_tracker_policy_is_rejected(
    acquisition: float, maintenance: float, retention_s: float
) -> None:
    with pytest.raises(ValueError):
        MotionTracker(ConfirmationConfig(), ["pigeon"], acquisition, maintenance, retention_s)


def test_import_does_not_require_optional_follow_dependencies() -> None:
    script = """
import importlib.abc
import sys
class RejectOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'cv2'}:
            raise ImportError('optional dependency unavailable')
sys.meta_path.insert(0, RejectOptionalImports())
from app.motion_tracking import MotionTracker
from app.config import ConfirmationConfig
MotionTracker(ConfirmationConfig(), ['pigeon'], 0.5)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
