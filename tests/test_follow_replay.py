from __future__ import annotations

import unittest

import numpy

from app.config import ConfirmationConfig
from app.follow_models import SelectionUpdate, TrackingUpdate
from app.models import Detection, FramePacket
from app.motion_tracking import MotionTracker
from app.target_selection import TargetSelector

FRAME_WIDTH_PX = 320
FRAME_HEIGHT_PX = 180
FRAME_INTERVAL_S = 0.05
ACQUISITION_CONFIDENCE = 0.6
HIGH_CONFIDENCE = 0.9


class FollowReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = MotionTracker(
            ConfirmationConfig(enabled=True), ("pigeon",), ACQUISITION_CONFIDENCE
        )
        self.selector = TargetSelector()
        self.frame = numpy.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=numpy.uint8)

    def advance(
        self, sequence: int, detections: list[Detection]
    ) -> tuple[TrackingUpdate, SelectionUpdate]:
        source_time_s = sequence * FRAME_INTERVAL_S
        packet = FramePacket(sequence, 1000.0 + source_time_s, self.frame)
        tracking = self.tracker.update(packet, detections, source_time_s, source_time_s)
        return tracking, self.selector.update(tracking.tracks, source_time_s)

    def detection(self, confidence: float = HIGH_CONFIDENCE, left_px: float = 50) -> Detection:
        return Detection("pigeon", confidence, left_px, 50, left_px + 40, 90)

    def test_confirmation_selection_and_observation_share_one_identity(self) -> None:
        first, first_selection = self.advance(0, [self.detection()])
        self.assertIsNone(first_selection.target.track)
        second, second_selection = self.advance(1, [self.detection(left_px=52)])
        selected = second_selection.target.track
        assert selected is not None
        self.assertEqual(len(second.triggers), 1)
        self.assertEqual(
            {
                selected.track_id,
                selected.detection.track_id,
                first.detections[0].track_id,
                second.triggers[0].track_id,
                second_selection.events[0].track_id,
            },
            {selected.track_id},
        )

    def test_occlusion_retains_observation_time_and_expires_in_source_time(self) -> None:
        self.advance(0, [self.detection()])
        _, confirmed = self.advance(1, [self.detection()])
        assert confirmed.target.track is not None
        _, waiting = self.advance(2, [])
        self.assertEqual(waiting.target.state, "waiting")
        assert waiting.target.track is not None
        self.assertFalse(waiting.target.track.matched)
        self.assertEqual(waiting.target.track.observed_at_s, FRAME_INTERVAL_S)
        _, expired = self.advance(17, [])
        self.assertIsNone(expired.target.track)
        self.assertEqual(expired.events[0].name, "target_lost")
        self.assertEqual(expired.events[0].track_id, confirmed.target.track.track_id)


if __name__ == "__main__":
    unittest.main()
