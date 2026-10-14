"""Frame processing on its own thread, handing logs and frames to other threads.

The main thread owns the OpenCV window (macOS and Qt require GUI calls there) and
the recording; a log-writer thread owns the JSONL log, so a stalled display never
blocks logging. The processing thread never waits for either while live: log
records go to a bounded queue that drops the oldest frame records first, and
frames to render go to a bounded queue that drops the oldest waiting frame.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .capture import EndOfStream
from .config import AppConfig
from .detector import FrameDetector, stage_timings
from .json_data import JsonValue
from .models import Detection, Frame, FramePacket
from .output import health_snapshot
from .tracking import ConfirmationTracker, TriggerEvent

RENDER_QUEUE_CAPACITY = 2
LOG_QUEUE_CAPACITY = 4096
# only per-frame records may be dropped; trigger, camera_switch, health, stimulus and
# error/run records are always kept.
DROPPABLE_EVENTS = frozenset({"frame"})
MILLISECONDS_PER_SECOND = 1000
CONSUMER_POLL_S = 0.01


def milliseconds(seconds: float) -> float:
    return seconds * MILLISECONDS_PER_SECOND


def describe(error: BaseException) -> str:
    message = str(error)
    if isinstance(error, Exception) and message:
        return message
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


class FrameSource(Protocol):
    @property
    def dropped_frames(self) -> int: ...
    def get(self, timeout_s: float) -> FramePacket: ...


class EventWriter(Protocol):
    def event(self, event: str, **fields: JsonValue) -> None: ...


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
    """A bounded log queue and a bounded queue of frames to render.

    Logs: past `log_capacity` the oldest frame record is dropped (counted in
    `dropped_records`); other records are never dropped. Frames: with
    `drop_oldest` the producer never waits and a full queue loses its oldest
    frame. Without it (replay, which has no real-time pacing) the producer waits
    for space so that recordings keep every processed frame.
    """

    def __init__(
        self,
        render_capacity: int = RENDER_QUEUE_CAPACITY,
        drop_oldest: bool = True,
        log_capacity: int = LOG_QUEUE_CAPACITY,
    ):
        self.render_capacity = render_capacity
        self.drop_oldest = drop_oldest
        self.log_capacity = log_capacity
        self.dropped_renders = 0
        self.dropped_records = 0
        self._records: deque[LogRecord] = deque()
        self._renders: deque[RenderJob] = deque()
        self._log_condition = threading.Condition()
        self._render_condition = threading.Condition()
        self._closed = False

    def event(self, event: str, **fields: JsonValue) -> None:
        with self._log_condition:
            if len(self._records) >= self.log_capacity:
                for index, record in enumerate(self._records):
                    if record.event in DROPPABLE_EVENTS:
                        del self._records[index]
                        self.dropped_records += 1
                        break
            self._records.append(LogRecord(event, fields))
            self._log_condition.notify_all()

    def take_logs(self, timeout_s: float = 0.0) -> list[LogRecord]:
        with self._log_condition:
            if timeout_s > 0:
                self._log_condition.wait_for(lambda: self._records, timeout_s)
            records = list(self._records)
            self._records.clear()
            return records

    def render(self, job: RenderJob) -> None:
        with self._render_condition:
            if not self.drop_oldest:
                self._render_condition.wait_for(
                    lambda: self._closed or len(self._renders) < self.render_capacity
                )
            if self._closed:
                return
            if len(self._renders) >= self.render_capacity:
                self._renders.popleft()
                self.dropped_renders += 1
            self._renders.append(job)
            self._render_condition.notify_all()

    def take_render(self, timeout_s: float = 0.0) -> RenderJob | None:
        """Return the oldest waiting frame, if any."""
        with self._render_condition:
            if timeout_s > 0:
                self._render_condition.wait_for(lambda: self._renders, timeout_s)
            job = self._renders.popleft() if self._renders else None
            self._render_condition.notify_all()
            return job

    def close(self) -> None:
        """Stop accepting frames to render and release a waiting producer."""
        with self._render_condition:
            self._closed = True
            self._renders.clear()
            self._render_condition.notify_all()


class LogWriter:
    """Writes queued log records on its own thread until `finish` drains the queue."""

    def __init__(
        self,
        output: EventWriter,
        queue: OutputQueue,
        on_failure: Callable[[], None],
        observe: Callable[[LogRecord], None] | None = None,
    ) -> None:
        self.output = output
        self.queue = queue
        self.on_failure = on_failure
        self.observe = observe
        self.error: BaseException | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self.run, name="log-writer")

    def start(self) -> None:
        self._thread.start()

    def run(self) -> None:
        try:
            while True:
                done = self._done.is_set()
                records = self.queue.take_logs(0.0 if done else CONSUMER_POLL_S)
                for record in records:
                    self.output.event(record.event, **record.fields)
                    if self.observe is not None:
                        self.observe(record)
                if done and not records:
                    return
        except BaseException as error:
            self.error = error
            print(f"log writing failed: {describe(error)}", file=sys.stderr)
            self.on_failure()

    def finish(self) -> None:
        """Write everything still queued, then stop."""
        self._done.set()
        if self._thread.ident is not None:
            self._thread.join()


@dataclass
class ProcessingResult:
    processed_frames: int = 0
    dropped_frames: int = 0
    reason: str = "stopped"
    error: BaseException | None = None
    # set only when the loop ends normally; anything else is a failure.
    completed: bool = False


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
            result.completed = True
        except EndOfStream:
            result.reason, result.completed = "video_complete", True
        except BaseException as error:
            # also SystemExit and KeyboardInterrupt, which would otherwise end the
            # thread silently and look like a clean stop.
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
                log_dropped_records=self.sink.dropped_records,
                **health_snapshot(),
            )
            self._last_health_at_monotonic_s = now_monotonic_s
