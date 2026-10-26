from __future__ import annotations

import unittest

import numpy

from app.config import ConfirmationConfig
from app.models import Detection
from app.tracking import ConfirmationTracker, greedy_matches, iou_matrix


def detection(label: str = "pigeon", left_px: float = 10, class_id: int = 0) -> Detection:
    return Detection(label, 0.8, left_px, 10, left_px + 40, 50, class_id=class_id)


def run_frames(
    tracker: ConfirmationTracker, frames: list[list[Detection]]
) -> list[tuple[int, list[int | None]]]:
    """Return (trigger count, track ids) per frame."""
    results: list[tuple[int, list[int | None]]] = []
    for frame_number, detections in enumerate(frames):
        update = tracker.update(detections, frame_number, 100.0 + frame_number)
        results.append((len(update.triggers), [item.track_id for item in update.detections]))
    return results


class TrackingTests(unittest.TestCase):
    def tracker(self, k: int = 2, n: int = 3, max_age_frames: int = 2) -> ConfirmationTracker:
        config = ConfirmationConfig(enabled=True, k=k, n=n, max_age_frames=max_age_frames)
        return ConfirmationTracker(config, ["pigeon"], clock=lambda: 500.0)

    def test_iou_and_greedy_order(self) -> None:
        boxes = numpy.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=numpy.float64)
        self.assertAlmostEqual(iou_matrix(boxes, boxes)[0, 1], 50 / 150)
        tracks = numpy.array([[0, 0, 10, 10], [4, 0, 14, 10]], dtype=numpy.float64)
        detections = numpy.array([[1, 0, 11, 10], [6, 0, 16, 10]], dtype=numpy.float64)
        self.assertEqual(greedy_matches(tracks, detections, 0.3), [(0, 0), (1, 1)])
        # both tracks overlap the only detection; the higher IoU wins.
        self.assertEqual(greedy_matches(tracks, detections[:1], 0.3), [(0, 0)])
        self.assertEqual(greedy_matches(tracks[:0], detections, 0.3), [])

    def test_k_of_n_confirms_once_with_first_seen_time(self) -> None:
        tracker = self.tracker()
        triggers = []
        frames = [[detection()], [], [detection(left_px=12)], [detection(left_px=14)]]
        for frame_number, detections in enumerate(frames):
            triggers.extend(tracker.update(detections, frame_number, 100.0 + frame_number).triggers)
        self.assertEqual(len(triggers), 1)
        event = triggers[0].as_dict()
        self.assertEqual(
            event,
            {
                "track_id": 1,
                "class": "pigeon",
                "first_seen_captured_at": 100.0,
                "first_seen_frame": 0,
                "confirmed_at": 500.0,
                "frame_number": 2,
                "frames_seen": 2,
                "k": 2,
                "n": 3,
            },
        )
        self.assertEqual(tracker.confirmed_track_ids, {1})

    def test_k_equals_one_confirms_on_first_frame(self) -> None:
        self.assertEqual(run_frames(self.tracker(k=1, n=1), [[detection()]])[0][0], 1)

    def test_only_trigger_classes_confirm(self) -> None:
        dove = detection("spotted_dove", class_id=1)
        self.assertEqual(
            sum(count for count, _ in run_frames(self.tracker(), [[dove]] * 6)),
            0,
        )
        # class flicker keeps one track; two pigeon hits within three frames confirm.
        flicker = [[dove], [detection()], [dove], [detection()]]
        results = run_frames(self.tracker(), flicker)
        self.assertEqual([count for count, _ in results], [0, 0, 0, 1])
        self.assertEqual({ids[0] for _, ids in results}, {1})
        # pigeon hits further apart than n frames never confirm.
        spread = [[detection()], [dove], [dove], [detection()], [dove], [dove]]
        self.assertEqual(sum(count for count, _ in run_frames(self.tracker(), spread)), 0)

    def test_tracks_expire_after_max_age(self) -> None:
        frames = [[detection()], [], [], [], [detection()]]
        results = run_frames(self.tracker(k=3, n=5, max_age_frames=2), frames)
        self.assertEqual(results[0][1], [1])
        self.assertEqual(results[4][1], [2])
        frames = [[detection()], [], [], [detection()]]
        results = run_frames(self.tracker(k=3, n=5, max_age_frames=2), frames)
        self.assertEqual(results[3][1], [1])

    def test_separate_birds_keep_their_tracks(self) -> None:
        frames = [
            [detection(left_px=0), detection(left_px=200)],
            [detection(left_px=205), detection(left_px=4)],
        ]
        results = run_frames(self.tracker(k=3), frames)
        self.assertEqual(results[0][1], [1, 2])
        self.assertEqual(results[1][1], [2, 1])


if __name__ == "__main__":
    unittest.main()
