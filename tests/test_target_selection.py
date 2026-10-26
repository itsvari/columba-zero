from __future__ import annotations

import unittest

from app.follow_models import SelectionUpdate, TrackSnapshot
from app.models import Detection
from app.target_selection import TargetSelector

BASE_AREA_PX2 = 8000.0
LARGER_AREA_PX2 = 10000.0
FRAME_INTERVAL_S = 0.1
RETENTION_S = 0.75
SWITCH_HOLD_S = 0.5


def snapshot(
    track_id: int,
    area_px2: float = BASE_AREA_PX2,
    at_s: float = 0.0,
    *,
    label: str = "pigeon",
    confidence: float = 0.8,
    confirmed: bool = True,
    matched: bool = True,
) -> TrackSnapshot:
    detection = Detection(label, confidence, 10, 10, 50, 50, track_id=track_id)
    return TrackSnapshot(track_id, detection, confirmed, matched, at_s, area_px2)


def event_names(update: SelectionUpdate) -> list[str]:
    return [event.name for event in update.events]


class TargetSelectionTests(unittest.TestCase):
    def selector(self) -> TargetSelector:
        return TargetSelector(RETENTION_S, 1.25, SWITCH_HOLD_S)

    def acquire(self, selector: TargetSelector, track_id: int = 1) -> SelectionUpdate:
        return selector.update([snapshot(track_id)], 0.0)

    def test_acquires_largest_smoothed_area_not_raw_box_or_confidence(self) -> None:
        small_raw_large_smoothed = snapshot(1, area_px2=9000.0, at_s=1.0, confidence=0.3)
        large_raw_small_smoothed = Detection("pigeon", 0.95, 0, 0, 500, 500, track_id=2)
        decoy = TrackSnapshot(2, large_raw_small_smoothed, True, True, 1.0, 4000.0)
        update = self.selector().update([decoy, small_raw_large_smoothed], 1.0)
        assert update.target.track is not None
        self.assertEqual(update.target.track.track_id, 1)
        self.assertEqual(update.target.state, "tracking")
        self.assertEqual(update.target.selected_at_s, 1.0)
        event = update.events[0]
        self.assertEqual((event.name, event.track_id, event.at_s), ("target_acquired", 1, 1.0))
        self.assertIsNone(event.previous_track_id)

    def test_ties_choose_lowest_track_id_regardless_of_input_order(self) -> None:
        tracks = [snapshot(7), snapshot(3), snapshot(5)]
        for ordering in (tracks, tracks[::-1], [tracks[1], tracks[0], tracks[2]]):
            update = self.selector().update(ordering, 0.0)
            assert update.target.track is not None
            self.assertEqual(update.target.track.track_id, 3)

    def test_only_confirmed_matched_pigeons_are_selectable(self) -> None:
        ineligible = [
            snapshot(1, 20000.0, confirmed=False),
            snapshot(2, 20000.0, matched=False),
            snapshot(3, 20000.0, label="crow"),
            snapshot(4, 0.0),
            snapshot(5, float("nan")),
        ]
        selector = self.selector()
        update = selector.update(ineligible, 0.0)
        self.assertIsNone(update.target.track)
        self.assertEqual(update.events, ())
        update = selector.update([*ineligible, snapshot(6, 100.0)], 0.1)
        assert update.target.track is not None
        self.assertEqual(update.target.track.track_id, 6)

    def test_stale_snapshots_are_ignored_even_when_passed_in(self) -> None:
        selector = self.selector()
        update = selector.update([snapshot(1, at_s=0.0)], RETENTION_S + 0.01)
        self.assertIsNone(update.target.track)
        update = selector.update([snapshot(1, at_s=0.0)], RETENTION_S)
        assert update.target.track is not None
        self.assertEqual(update.target.track.track_id, 1)

    def test_brief_occlusion_waits_then_reacquires_without_events(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        missing = selector.update([], 0.4)
        assert missing.target.track is not None
        self.assertEqual(missing.target.state, "waiting")
        self.assertFalse(missing.target.track.matched)
        self.assertEqual(missing.target.track.track_id, 1)
        self.assertEqual(missing.target.track.observed_at_s, 0.0)
        self.assertEqual(missing.target.selected_at_s, 0.0)
        self.assertEqual(missing.events, ())
        back = selector.update([snapshot(1, at_s=0.7)], 0.7)
        self.assertEqual(back.target.state, "tracking")
        self.assertTrue(back.target.track is not None and back.target.track.matched)
        self.assertEqual(back.target.selected_at_s, 0.0)
        self.assertEqual(back.events, ())

    def test_occlusion_does_not_hand_over_to_a_visible_larger_bird(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        for step in range(1, 7):
            now_s = step * FRAME_INTERVAL_S
            update = selector.update([snapshot(2, 4 * BASE_AREA_PX2, now_s)], now_s)
            self.assertEqual(update.target.state, "waiting")
            assert update.target.track is not None
            self.assertEqual(update.target.track.track_id, 1)
            self.assertEqual(update.events, ())

    def test_expiry_emits_lost_once_then_stays_lost(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        held = selector.update([], RETENTION_S)
        self.assertEqual(held.target.state, "waiting")
        expired = selector.update([], RETENTION_S + 0.01)
        self.assertEqual(expired.target.state, "lost")
        self.assertIsNone(expired.target.track)
        self.assertIsNone(expired.target.selected_at_s)
        self.assertEqual([(e.name, e.track_id) for e in expired.events], [("target_lost", 1)])
        self.assertEqual(expired.events[0].at_s, RETENTION_S + 0.01)
        again = selector.update([], 2.0)
        self.assertEqual(again.target.state, "lost")
        self.assertEqual(again.events, ())

    def test_expiry_counts_from_last_pigeon_observation_not_last_update(self) -> None:
        # a stream of unmatched or wrong-class reports must not keep the incumbent alive
        for label, matched in (("pigeon", False), ("crow", True)):
            selector = self.selector()
            self.acquire(selector)
            names: list[str] = []
            for step in range(1, 12):
                now_s = step * FRAME_INTERVAL_S
                report = snapshot(1, at_s=now_s, label=label, matched=matched)
                names.extend(event_names(selector.update([report], now_s)))
            self.assertEqual(names, ["target_lost"], (label, matched))

    def test_expiry_then_new_bird_emits_lost_and_acquired_with_distinct_ids(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        update = selector.update([snapshot(2, at_s=1.0)], 1.0)
        self.assertEqual(event_names(update), ["target_lost", "target_acquired"])
        self.assertEqual([event.track_id for event in update.events], [1, 2])
        self.assertEqual(update.target.state, "tracking")

    def test_lost_target_recovers_on_acquisition(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        selector.update([], 1.0)
        update = selector.update([snapshot(1, at_s=1.1)], 1.1)
        self.assertEqual(event_names(update), ["target_acquired"])
        self.assertEqual(update.target.state, "tracking")

    def test_class_uncertainty_holds_incumbent_without_correcting(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        flicker = snapshot(1, at_s=0.2, label="crow")
        update = selector.update([flicker], 0.2)
        assert update.target.track is not None
        self.assertEqual(update.target.state, "waiting")
        self.assertEqual(update.target.track.detection.label, "pigeon")
        self.assertFalse(update.target.track.matched)
        self.assertEqual(update.events, ())
        recovered = selector.update([snapshot(1, at_s=0.3)], 0.3)
        self.assertEqual(recovered.target.state, "tracking")
        self.assertEqual(recovered.events, ())

    def test_switch_threshold_is_inclusive_at_the_ratio(self) -> None:
        for rival_area_px2, switches in ((LARGER_AREA_PX2, True), (LARGER_AREA_PX2 - 1, False)):
            with self.subTest(rival_area_px2=rival_area_px2):
                selector = self.selector()
                self.acquire(selector)
                names: list[str] = []
                for step in range(0, 8):
                    now_s = step * FRAME_INTERVAL_S
                    tracks = [snapshot(1, at_s=now_s), snapshot(2, rival_area_px2, now_s)]
                    names.extend(event_names(selector.update(tracks, now_s)))
                self.assertEqual(names, ["target_switched"] if switches else [])

    def test_switch_waits_for_continuous_hold_then_emits_once_with_shared_ids(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        updates = []
        for step in range(0, 9):
            now_s = step * FRAME_INTERVAL_S
            tracks = [snapshot(1, at_s=now_s), snapshot(2, LARGER_AREA_PX2, now_s)]
            updates.append(selector.update(tracks, now_s))
        before_hold = updates[: round(SWITCH_HOLD_S / FRAME_INTERVAL_S)]
        for update in before_hold:
            assert update.target.track is not None
            self.assertEqual(update.target.track.track_id, 1)
            self.assertEqual(update.events, ())
        switched = updates[round(SWITCH_HOLD_S / FRAME_INTERVAL_S)]
        event = switched.events[0]
        self.assertEqual(
            (event.name, event.track_id, event.previous_track_id), ("target_switched", 2, 1)
        )
        self.assertEqual(event.at_s, switched.target.selected_at_s)
        assert switched.target.track is not None
        self.assertEqual(switched.target.track.track_id, 2)
        self.assertEqual(switched.target.state, "tracking")
        self.assertEqual(sum(len(update.events) for update in updates), 1)

    def test_challenge_resets_when_challenger_shrinks_or_disappears(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        big = [snapshot(1, at_s=0.1), snapshot(2, LARGER_AREA_PX2, 0.1)]
        selector.update(big, 0.1)
        selector.update([snapshot(1, at_s=0.4), snapshot(2, LARGER_AREA_PX2 - 500, 0.4)], 0.4)
        for now_s in (0.5, 0.8, 0.9):
            update = selector.update(
                [snapshot(1, at_s=now_s), snapshot(2, LARGER_AREA_PX2, now_s)], now_s
            )
            self.assertEqual(update.events, ())
        selector.update([snapshot(1, at_s=1.0)], 1.0)
        update = selector.update([snapshot(1, at_s=1.1), snapshot(2, LARGER_AREA_PX2, 1.1)], 1.1)
        self.assertEqual(update.events, ())
        update = selector.update([snapshot(1, at_s=1.6), snapshot(2, LARGER_AREA_PX2, 1.6)], 1.6)
        self.assertEqual(event_names(update), ["target_switched"])

    def test_challenge_resets_when_a_different_bird_becomes_the_largest(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        selector.update(
            [snapshot(1, at_s=0.1), snapshot(2, LARGER_AREA_PX2, 0.1), snapshot(3, 9000, 0.1)],
            0.1,
        )
        update = selector.update(
            [snapshot(1, at_s=0.5), snapshot(2, LARGER_AREA_PX2, 0.5), snapshot(3, 12000, 0.5)],
            0.5,
        )
        self.assertEqual(update.events, ())
        update = selector.update(
            [snapshot(1, at_s=1.0), snapshot(2, LARGER_AREA_PX2, 1.0), snapshot(3, 12000, 1.0)],
            1.0,
        )
        self.assertEqual([(e.name, e.track_id) for e in update.events], [("target_switched", 3)])

    def test_unconfirmed_or_wrong_class_rival_never_challenges(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        for step in range(1, 12):
            now_s = step * FRAME_INTERVAL_S
            tracks = [
                snapshot(1, at_s=now_s),
                snapshot(2, 5 * BASE_AREA_PX2, now_s, confirmed=False),
                snapshot(3, 5 * BASE_AREA_PX2, now_s, label="crow"),
                snapshot(4, 5 * BASE_AREA_PX2, now_s, matched=False),
            ]
            self.assertEqual(selector.update(tracks, now_s).events, ())

    def test_no_switching_jitter_between_similar_birds(self) -> None:
        selector = self.selector()
        selector.update([snapshot(1), snapshot(2, BASE_AREA_PX2 - 1)], 0.0)
        selected: set[int] = set()
        events = 0
        for step in range(1, 200):
            now_s = step * FRAME_INTERVAL_S
            wobble = 400.0 if step % 2 else -400.0
            tracks = [
                snapshot(1, BASE_AREA_PX2 - wobble, now_s),
                snapshot(2, BASE_AREA_PX2 + wobble, now_s),
            ]
            update = selector.update(tracks, now_s)
            events += len(update.events)
            assert update.target.track is not None
            selected.add(update.target.track.track_id)
        self.assertEqual(selected, {1})
        self.assertEqual(events, 0)

    def test_reset_clears_target_and_reports_loss_once(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        events = selector.reset()
        self.assertEqual([(e.name, e.track_id, e.at_s) for e in events], [("target_lost", 1, 0.0)])
        self.assertEqual(selector.reset(), ())
        update = selector.update([], 0.1)
        self.assertEqual(update.target.state, "acquiring")
        self.assertEqual(update.events, ())

    def test_reset_after_loss_returns_to_acquiring(self) -> None:
        selector = self.selector()
        self.acquire(selector)
        selector.update([], 2.0)
        self.assertEqual(selector.reset(), ())
        self.assertEqual(selector.update([], 2.1).target.state, "acquiring")

    def test_constructor_rejects_invalid_policy(self) -> None:
        with self.assertRaises(ValueError):
            TargetSelector(retention_s=0.0)
        with self.assertRaises(ValueError):
            TargetSelector(switch_area_ratio=0.9)
        with self.assertRaises(ValueError):
            TargetSelector(switch_hold_s=-0.1)


if __name__ == "__main__":
    unittest.main()
