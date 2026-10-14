"""Frame processing on its own thread, handing logs and frames to the main thread.

The main thread owns the OpenCV window (macOS and Qt require GUI calls there), the
JSONL log and the recording. The processing thread never waits for them while
live: log records queue without limit (they are small and written quickly), and
frames to render go to a bounded queue that drops the oldest waiting frame.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .capture import EndOfStream
from .config import AppConfig
from .detector import FrameDetector, stage_timings
from .json_data import JsonValue
from .models import Detection, Frame, FramePacket
from .output import health_snapshot
from .tracking import ConfirmationTracker, TriggerEvent

RENDER_QUEUE_CAPACITY = 2
MILLISECONDS_PER_SECOND = 1000
CONSUMER_POLL_S = 0.01


def milliseconds(seconds: float) -> float:
    return seconds * MILLISECONDS_PER_SECOND


class FrameSource(Protocol):
    @property
    def dropped_frames(self) -> int: ...
    def get(self, timeout_s: float) -> FramePacket: ...


@dataclass(frozen=True, slots=True)
class RenderJob:
    frame: Frame
    detections: list[Detection]
    processing_fps: float
    capture_fps: float | None
    confirmed_track_ids: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class LogRecord:
    event: str
    fields: dict[str, JsonValue]


class OutputQueue:
    """Log records (never dropped, in order) and a bounded queue of frames to render.

    With `drop_oldest` the producer never waits: a full render queue loses its
    oldest frame. Without it (replay, which has no real-time pacing) the producer
    waits for space so that recordings keep every processed frame.
    """

    def __init__(self, render_capacity: int = RENDER_QUEUE_CAPACITY, drop_oldest: bool = True):
        self.render_capacity = render_capacity
        self.drop_oldest = drop_oldest
        self.dropped_renders = 0
        self._records: deque[LogRecord] = deque()
        self._renders: deque[RenderJob] = deque()
        self._condition = threading.Condition()
        self._closed = False

    def event(self, event: str, **fields: JsonValue) -> None:
        with self._condition:
            self._records.append(LogRecord(event, fields))
            self._condition.notify_all()

    def render(self, job: RenderJob) -> None:
        with self._condition:
            if not self.drop_oldest:
                self._condition.wait_for(
                    lambda: self._closed or len(self._renders) < self.render_capacity
                )
            if self._closed:
                return
            if len(self._renders) >= self.render_capacity:
                self._renders.popleft()
                self.dropped_renders += 1
            self._renders.append(job)
            self._condition.notify_all()

    def take(self, timeout_s: float = 0.0) -> tuple[list[LogRecord], RenderJob | None]:
        """Return all waiting log records and the oldest waiting frame."""
        with self._condition:
            if timeout_s > 0:
                self._condition.wait_for(lambda: self._records or self._renders, timeout_s)
            records = list(self._records)
            self._records.clear()
            job = self._renders.popleft() if self._renders else None
            self._condition.notify_all()
            return records, job

    def close(self) -> None:
        """Stop accepting frames and release a waiting producer."""
        with self._condition:
            self._closed = True
            self._renders.clear()
            self._condition.notify_all()


@dataclass
class ProcessingResult:
    processed_frames: int = 0
    dropped_frames: int = 0
    reason: str = "stopped"
    error: Exception | None = None
    triggers: list[TriggerEvent] = field(default_factory=list)


class FrameProcessor:
    """Capture → detect → confirm loop; run it on a thread with `run`."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        detector: FrameDetector,
        tracker: ConfirmationTracker | None,
        sink: OutputQueue,
        stop: threading.Event,
        started_at_monotonic_s: float,
        max_frames: int = 0,
        capture_fps: Callable[[], float | None] = lambda: None,
        camera_events: Callable[[], list[dict[str, JsonValue]]] = list,
    ) -> None:
        self.config = config
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.sink = sink
        self.stop = stop
        self.started_at_monotonic_s = started_at_monotonic_s
        self.max_frames = max_frames
        self.capture_fps = capture_fps
        self.camera_events = camera_events
        self._camera: str | None = None
        self.render_frames = config.display.enabled or config.output.record
        self.result = ProcessingResult()
        self._last_health_at_monotonic_s = started_at_monotonic_s

    def stage_ms(
        self, wait_s: float | None, inference_s: float, track_ms: float | None
    ) -> dict[str, JsonValue]:
        """C9 stage timings. Without per-stage figures from the detector, the
        whole detect call counts as inference."""
        stages = stage_timings(getattr(self.detector, "last_stage_ms", None))
        if all(value is None for value in stages.values()):
            stages["inference"] = milliseconds(inference_s)
        return {
            "wait": None if wait_s is None else milliseconds(wait_s),
            **stages,
            "track": track_ms,
        }

    def run(self) -> None:
        result = self.result
        try:
            while not self.stop.is_set():
                self.process(self.source.get(self.config.camera.frame_timeout_s))
                if self.max_frames and result.processed_frames >= self.max_frames:
                    result.reason = "max_frames"
                    break
        except EndOfStream:
            result.reason = "video_complete"
        except Exception as error:
            result.reason, result.error = "error", error

    def process(self, packet: FramePacket) -> None:
        result = self.result
        picked_up_monotonic_s = time.monotonic()
        if packet.camera != self._camera:
            if self.tracker is not None and self._camera is not None:
                self.tracker.reset()
            self._camera = packet.camera
        inference_started_at_monotonic_s = time.monotonic()
        detections = self.detector.detect(packet)
        inference_finished_at_monotonic_s = time.monotonic()
        inference_s = inference_finished_at_monotonic_s - inference_started_at_monotonic_s
        triggers: list[TriggerEvent] = []
        confirmed_track_ids: frozenset[int] = frozenset()
        track_ms: float | None = None
        if self.tracker is not None:
            update = self.tracker.update(detections, packet.sequence, packet.captured_at)
            detections, triggers = update.detections, update.triggers
            confirmed_track_ids = self.tracker.confirmed_track_ids
            track_ms = milliseconds(time.monotonic() - inference_finished_at_monotonic_s)
        decided_at = time.time()
        decided_at_monotonic_s = time.monotonic()
        delivered_monotonic_s = packet.delivered_monotonic_s
        result.dropped_frames = self.source.dropped_frames
        result.processed_frames += 1
        elapsed_s = time.monotonic() - self.started_at_monotonic_s
        processing_fps = result.processed_frames / elapsed_s if elapsed_s else 0.0
        capture_fps = self.capture_fps()
        if self.render_frames:
            self.sink.render(
                RenderJob(
                    packet.frame, detections, processing_fps, capture_fps, confirmed_track_ids
                )
            )
        self.sink.event(
            "frame",
            frame_number=packet.sequence,
            captured_at=packet.captured_at,
            inference_s=inference_s,
            pigeon_count=sum(detection.label == "pigeon" for detection in detections),
            detections=[detection.as_dict() for detection in detections],
            dropped_frames=result.dropped_frames,
            processing_fps=processing_fps,
            capture_fps=capture_fps,
            stage_ms=self.stage_ms(
                picked_up_monotonic_s - delivered_monotonic_s
                if delivered_monotonic_s is not None
                else None,
                inference_s,
                track_ms,
            ),
            camera=packet.camera,
            lux=packet.lux,
            delivery_ms=packet.delivery_ms,
            decided_at=decided_at,
            decision_ms=(
                milliseconds(decided_at_monotonic_s - delivered_monotonic_s)
                if delivered_monotonic_s is not None
                else None
            ),
        )
        for trigger in triggers:
            self.sink.event("trigger", **trigger.as_dict())
        for camera_switch in self.camera_events():
            self.sink.event("camera_switch", **camera_switch)
        result.triggers.extend(triggers)
        now_monotonic_s = time.monotonic()
        if (
            now_monotonic_s - self._last_health_at_monotonic_s
            >= self.config.output.health_interval_s
        ):
            self.sink.event(
                "health",
                processed_frames=result.processed_frames,
                dropped_frames=result.dropped_frames,
                render_dropped_frames=self.sink.dropped_renders,
                **health_snapshot(),
            )
            self._last_health_at_monotonic_s = now_monotonic_s
