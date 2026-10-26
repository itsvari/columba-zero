"""Choose the pigeon the camera follows from tracker snapshots, with switching hysteresis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .follow_models import SelectionUpdate, TargetEvent, TargetSnapshot, TrackSnapshot

PIGEON_LABEL = "pigeon"
DEFAULT_RETENTION_S = 0.75
DEFAULT_SWITCH_AREA_RATIO = 1.25
DEFAULT_SWITCH_HOLD_S = 0.5


@dataclass(frozen=True, slots=True)
class _Incumbent:
    last_observation: TrackSnapshot
    selected_at_s: float
    observing: bool


@dataclass(frozen=True, slots=True)
class _Challenge:
    track_id: int
    since_s: float


class TargetSelector:
    """Pick one confirmed pigeon by smoothed area and keep it until clearly outgrown.

    Identities come from the tracker; the selector never assigns or merges them. Only a
    currently matched pigeon observation can become, or be, the commanded target. A missing
    or ambiguous incumbent stays selected as `waiting` so a brief occlusion does not hand
    the camera to another bird.
    """

    def __init__(
        self,
        retention_s: float = DEFAULT_RETENTION_S,
        switch_area_ratio: float = DEFAULT_SWITCH_AREA_RATIO,
        switch_hold_s: float = DEFAULT_SWITCH_HOLD_S,
    ) -> None:
        if retention_s <= 0:
            raise ValueError("retention_s must be positive")
        if switch_area_ratio < 1:
            raise ValueError("switch_area_ratio must be at least 1")
        if switch_hold_s < 0:
            raise ValueError("switch_hold_s must not be negative")
        self._retention_s = retention_s
        self._switch_area_ratio = switch_area_ratio
        self._switch_hold_s = switch_hold_s
        self._incumbent: _Incumbent | None = None
        self._challenge: _Challenge | None = None
        self._lost = False
        self._last_update_s: float | None = None

    def update(self, tracks: Sequence[TrackSnapshot], now_s: float) -> SelectionUpdate:
        self._last_update_s = now_s
        candidates = self._eligible_candidates(tracks, now_s)
        events: list[TargetEvent] = []
        if self._incumbent is not None:
            events.extend(self._advance_incumbent(candidates, now_s))
        if self._incumbent is None:
            events.extend(self._acquire(candidates, now_s))
        return SelectionUpdate(self._snapshot(), tuple(events))

    def reset(self) -> tuple[TargetEvent, ...]:
        """Forget the target when tracking restarts, e.g. after a camera switch.

        Returns the `target_lost` event for a target that was still selected, so the caller
        can log it; the return value may be ignored.
        """
        events = self._lose(self._last_update_s) if self._incumbent is not None else []
        self._incumbent = None
        self._challenge = None
        self._lost = False
        self._last_update_s = None
        return tuple(events)

    def _eligible_candidates(
        self, tracks: Sequence[TrackSnapshot], now_s: float
    ) -> dict[int, TrackSnapshot]:
        candidates: dict[int, TrackSnapshot] = {}
        for track in tracks:
            if self._is_eligible(track, now_s):
                candidates.setdefault(track.track_id, track)
        return candidates

    def _is_eligible(self, track: TrackSnapshot, now_s: float) -> bool:
        if not (track.confirmed and track.matched):
            return False
        if track.detection.label != PIGEON_LABEL:
            return False
        if not (math.isfinite(track.smoothed_area_px2) and track.smoothed_area_px2 > 0):
            return False
        return now_s - track.observed_at_s <= self._retention_s

    def _advance_incumbent(
        self, candidates: dict[int, TrackSnapshot], now_s: float
    ) -> list[TargetEvent]:
        incumbent = self._incumbent
        if incumbent is None:
            return []
        current = candidates.get(incumbent.last_observation.track_id)
        if current is None:
            return self._hold_or_lose(incumbent, now_s)
        self._incumbent = replace(incumbent, last_observation=current, observing=True)
        return self._evaluate_challenge(current, candidates, now_s)

    def _hold_or_lose(self, incumbent: _Incumbent, now_s: float) -> list[TargetEvent]:
        # a challenger must outgrow an observed bird, so occlusion pauses the challenge
        self._challenge = None
        if now_s - incumbent.last_observation.observed_at_s > self._retention_s:
            return self._lose(now_s)
        self._incumbent = replace(incumbent, observing=False)
        return []

    def _evaluate_challenge(
        self, incumbent: TrackSnapshot, candidates: dict[int, TrackSnapshot], now_s: float
    ) -> list[TargetEvent]:
        required_area_px2 = incumbent.smoothed_area_px2 * self._switch_area_ratio
        rivals = [
            candidate
            for candidate in candidates.values()
            if candidate.track_id != incumbent.track_id
            and candidate.smoothed_area_px2 >= required_area_px2
        ]
        if not rivals:
            self._challenge = None
            return []
        challenger = _largest(rivals)
        if self._challenge is None or self._challenge.track_id != challenger.track_id:
            self._challenge = _Challenge(challenger.track_id, now_s)
        if now_s - self._challenge.since_s < self._switch_hold_s:
            return []
        self._incumbent = _Incumbent(challenger, now_s, observing=True)
        self._challenge = None
        return [TargetEvent("target_switched", challenger.track_id, now_s, incumbent.track_id)]

    def _acquire(self, candidates: dict[int, TrackSnapshot], now_s: float) -> list[TargetEvent]:
        if not candidates:
            return []
        chosen = _largest(list(candidates.values()))
        self._incumbent = _Incumbent(chosen, now_s, observing=True)
        self._challenge = None
        self._lost = False
        return [TargetEvent("target_acquired", chosen.track_id, now_s)]

    def _lose(self, at_s: float | None) -> list[TargetEvent]:
        incumbent = self._incumbent
        if incumbent is None or at_s is None:
            return []
        self._incumbent = None
        self._challenge = None
        self._lost = True
        return [TargetEvent("target_lost", incumbent.last_observation.track_id, at_s)]

    def _snapshot(self) -> TargetSnapshot:
        incumbent = self._incumbent
        if incumbent is None:
            return TargetSnapshot(state="lost" if self._lost else "acquiring")
        if incumbent.observing:
            return TargetSnapshot(incumbent.last_observation, "tracking", incumbent.selected_at_s)
        waiting_track = replace(incumbent.last_observation, matched=False)
        return TargetSnapshot(waiting_track, "waiting", incumbent.selected_at_s)


def _largest(tracks: Sequence[TrackSnapshot]) -> TrackSnapshot:
    # lowest identifier breaks area ties so input order never changes the choice
    return min(tracks, key=lambda track: (-track.smoothed_area_px2, track.track_id))
