"""Reaction-time self-test: show a pigeon on the monitor and time the trigger.

Point the camera at the monitor. The test alternates a blank screen and a
pigeon image at randomised times, logs each change as a `stimulus` event, and
matches `trigger` events to the trials. `displayed_at` is taken when OpenCV
returns from drawing; the monitor shows the image somewhat later (compositor,
refresh and panel latency), so the reported reaction times overestimate the
true ones by that display latency.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from operator import itemgetter
from typing import cast

import numpy

from .json_data import JsonValue
from .models import Frame

WINDOW_EVENT_DELAY_MS = 1


@dataclass(frozen=True)
class ReactionPlan:
    trials: int = 10
    show_s: float = 3.0
    gap_min_s: float = 3.0
    gap_max_s: float = 6.0
    settle_s: float = 5.0
    # a trigger this long after the image disappears still belongs to the trial.
    grace_s: float = 1.0


@dataclass
class Trial:
    number: int
    on_at: float | None = None
    off_at: float | None = None
    trigger_at: float | None = None
    track_id: int | None = None
    first_detection_at: float | None = None

    def contains(self, moment: float, grace_s: float) -> bool:
        if self.on_at is None or moment < self.on_at:
            return False
        return self.off_at is None or moment <= self.off_at + grace_s


@dataclass
class StimulusSchedule:
    """When to show and hide the pigeon; pure timing logic."""

    plan: ReactionPlan
    rng: random.Random = field(default_factory=random.Random)
    trial: int = 0
    showing: bool = False
    next_change_s: float = 0.0
    finishing: bool = False
    done: bool = False

    def begin(self, now_s: float) -> None:
        self.next_change_s = now_s + self.plan.settle_s

    def advance(self, now_s: float) -> str | None:
        """Return "on" or "off" when the screen should change, else None.

        After the last trial the test waits `grace_s` for late triggers, then is done.
        """
        if self.done or now_s < self.next_change_s:
            return None
        if self.finishing:
            self.done = True
            return None
        if self.showing:
            self.showing = False
            if self.trial >= self.plan.trials:
                self.finishing = True
                self.next_change_s = now_s + self.plan.grace_s
            else:
                self.next_change_s = now_s + self.rng.uniform(
                    self.plan.gap_min_s, self.plan.gap_max_s
                )
            return "off"
        self.trial += 1
        self.showing = True
        self.next_change_s = now_s + self.plan.show_s
        return "on"


def percentiles(values: list[float]) -> dict[str, JsonValue] | None:
    if not values:
        return None
    ordered = sorted(values)

    def rank(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": rank(0.5),
        "p95": rank(0.95),
        "max": ordered[-1],
    }


class ReactionAnalysis:
    """Match stimulus, frame and trigger records into per-trial reaction times.

    Records are only collected while the test runs; `report` matches them by their
    timestamps afterwards, so the order in which threads deliver them (a trigger
    may reach the log writer before its stimulus is registered) does not matter.
    """

    def __init__(self, plan: ReactionPlan, trigger_classes: tuple[str, ...] = ("pigeon",)):
        self.plan = plan
        self.trigger_classes = trigger_classes
        self._stimuli: list[tuple[float, int, str]] = []
        self._triggers: list[tuple[float, int | None]] = []
        self._detections: list[tuple[float, float]] = []
        # stimuli arrive from the main thread, frames and triggers from the log writer.
        self._lock = threading.Lock()

    def stimulus(self, number: int, state: str, displayed_at: float) -> None:
        with self._lock:
            self._stimuli.append((displayed_at, number, state))

    def observe(self, event: str, fields: dict[str, JsonValue]) -> None:
        with self._lock:
            self._observe(event, fields)

    def _observe(self, event: str, fields: dict[str, JsonValue]) -> None:
        if event == "trigger":
            confirmed_at = fields.get("confirmed_at")
            track_id = fields.get("track_id")
            if isinstance(confirmed_at, (int, float)):
                self._triggers.append(
                    (float(confirmed_at), track_id if isinstance(track_id, int) else None)
                )
        elif event == "frame":
            decided_at = fields.get("decided_at")
            captured_at = fields.get("captured_at")
            detections = fields.get("detections")
            if not isinstance(decided_at, (int, float)) or not isinstance(
                captured_at, (int, float)
            ):
                return
            if isinstance(detections, list) and any(
                isinstance(item, dict) and item.get("label") in self.trigger_classes
                for item in detections
            ):
                self._detections.append((float(captured_at), float(decided_at)))

    def trial_at(self, trials: list[Trial], moment: float) -> Trial | None:
        for trial in reversed(trials):
            if trial.contains(moment, self.plan.grace_s):
                return trial
        return None

    def matched_trials(self) -> tuple[list[Trial], int]:
        """Trials with their first trigger and detection, and the count of triggers
        outside every trial."""
        trials: list[Trial] = []
        for displayed_at, number, state in sorted(self._stimuli):
            if state == "on":
                trials.append(Trial(number, on_at=displayed_at))
            elif trials and trials[-1].off_at is None:
                trials[-1].off_at = displayed_at
        false_triggers = 0
        for confirmed_at, track_id in sorted(self._triggers, key=itemgetter(0)):
            trial = self.trial_at(trials, confirmed_at)
            if trial is None:
                false_triggers += 1
            elif trial.trigger_at is None:
                trial.trigger_at, trial.track_id = confirmed_at, track_id
        for captured_at, decided_at in sorted(self._detections):
            trial = self.trial_at(trials, captured_at)
            if trial is not None and trial.first_detection_at is None:
                trial.first_detection_at = decided_at
        return trials, false_triggers

    def report(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._report()

    def _report(self) -> dict[str, JsonValue]:
        trials, false_triggers = self.matched_trials()
        completed = [trial for trial in trials if trial.on_at is not None]
        reactions = [
            trial.trigger_at - trial.on_at
            for trial in completed
            if trial.trigger_at is not None and trial.on_at is not None
        ]
        detections = [
            trial.first_detection_at - trial.on_at
            for trial in completed
            if trial.first_detection_at is not None and trial.on_at is not None
        ]
        return {
            "trials": len(completed),
            "triggered": len(reactions),
            "missed": len(completed) - len(reactions),
            "false_triggers": false_triggers,
            "reaction_s": percentiles(reactions),
            "first_detection_s": percentiles(detections),
            "per_trial": [
                {
                    "trial": trial.number,
                    "on_at": trial.on_at,
                    "off_at": trial.off_at,
                    "reaction_s": (
                        trial.trigger_at - trial.on_at
                        if trial.trigger_at is not None and trial.on_at is not None
                        else None
                    ),
                    "first_detection_s": (
                        trial.first_detection_at - trial.on_at
                        if trial.first_detection_at is not None and trial.on_at is not None
                        else None
                    ),
                    "track_id": trial.track_id,
                }
                for trial in completed
            ],
            "timing_note": (
                "measured from when the image was drawn; display latency makes these "
                "overestimates of the camera-to-trigger reaction time"
            ),
        }


class ReactionTest:
    """Fullscreen stimulus window, driven from the main thread."""

    window = "Columba Zero reaction test"

    def __init__(
        self,
        image_path: str,
        plan: ReactionPlan,
        trigger_classes: tuple[str, ...] = ("pigeon",),
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"cannot read reaction-test image: {image_path}")
        self.image = cast(Frame, image)
        self.blank: Frame = numpy.zeros_like(self.image)
        self.schedule = StimulusSchedule(plan, rng or random.Random())
        self.analysis = ReactionAnalysis(plan, trigger_classes)
        self.clock = clock
        self._open = False

    @property
    def finished(self) -> bool:
        return self.schedule.done

    def start(self) -> None:
        import cv2

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self._open = True
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window, self.blank)
        cv2.waitKey(WINDOW_EVENT_DELAY_MS)
        self.schedule.begin(self.clock())

    def tick(self) -> dict[str, JsonValue] | None:
        """Update the screen if due; return the stimulus record to log."""
        import cv2

        change = self.schedule.advance(self.clock())
        if change is None:
            cv2.waitKey(WINDOW_EVENT_DELAY_MS)
            return None
        cv2.imshow(self.window, self.image if change == "on" else self.blank)
        cv2.waitKey(WINDOW_EVENT_DELAY_MS)
        displayed_at = self.clock()
        self.analysis.stimulus(self.schedule.trial, change, displayed_at)
        return {"trial": self.schedule.trial, "state": change, "displayed_at": displayed_at}

    def close(self) -> None:
        if self._open:
            import cv2

            self._open = False
            cv2.destroyWindow(self.window)
