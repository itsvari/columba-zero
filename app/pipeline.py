"""Frame processing on its own thread, handing logs and frames to other threads.

The main thread owns the OpenCV window (macOS and Qt require GUI calls there) and
the recording; a log-writer thread owns the JSONL log, so a stalled display never
blocks logging. The processing thread never waits for either while live: log
records go to a bounded queue that drops the oldest frame records first, and
frames to render go to a bounded queue that drops the oldest waiting frame.
"""

from __future__ import annotations

import heapq
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter
from typing import Protocol

from .capture import EndOfStream
from .config import AppConfig
from .detector import FrameDetector, stage_timings
from .follow_models import TargetEvent, TargetSnapshot
from .json_data import JsonValue
from .models import Detection, Frame, FramePacket
from .motion_tracking import MotionTracker
from .output import health_snapshot
from .pan_tilt_control import ControllerStatus, FollowSession, ObservationTiming
from .target_selection import TargetSelector
from .tracking import ConfirmationTracker, TriggerEvent

RENDER_QUEUE_CAPACITY = 2
LOG_QUEUE_CAPACITY = 4096
LOG_EVENT_ALLOWANCE = 1024
# how long shutdown waits for the log writer to drain before giving up on it.
LOG_DRAIN_TIMEOUT_S = 2.0
# sampled frame and controller records may be dropped; lifecycle events are always kept.
DROPPABLE_EVENTS = frozenset({"frame", "controller_state"})
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
    target: TargetSnapshot | None = None
    controller: ControllerStatus | None = None
    sequence: int = 0
    captured_at: float = 0.0
    delivered_monotonic_s: float | None = None
    presentation_time_s: float | None = None


class FramePublisher(Protocol):
    def publish(self, job: RenderJob) -> None: ...


@dataclass(frozen=True, slots=True)
class LogRecord:
    event: str
    fields: dict[str, JsonValue]


class OutputQueue:
    """A bounded log queue and a bounded queue of frames to render.

    Logs: sampled records and lifecycle events wait in separate deques, merged back into
    order by sequence number, so enqueueing is O(1). Past `log_capacity` records
    the oldest queued sample is dropped, or the incoming one when none is
    queued (both counted in `dropped_records`). Other events are never dropped:
    they may exceed the capacity by `event_allowance`. Past that the log writer is
    too far behind to record triggers, so `overflow_error` is set, `on_overflow`
    is called once (it stops the run), and further events are refused while
    those already accepted are kept.

    Frames to render: with `drop_oldest` the producer never waits and a full
    queue loses its oldest frame. Without it (replay, which has no real-time
    pacing) the producer waits for space so that recordings keep every
    processed frame.
    """

    def __init__(
        self,
        render_capacity: int = RENDER_QUEUE_CAPACITY,
        drop_oldest: bool = True,
        log_capacity: int = LOG_QUEUE_CAPACITY,
        event_allowance: int = LOG_EVENT_ALLOWANCE,
        on_overflow: Callable[[], None] = lambda: None,
    ):
        self.render_capacity = render_capacity
        self.drop_oldest = drop_oldest
        self.log_capacity = log_capacity
        self.event_allowance = event_allowance
        self.on_overflow = on_overflow
        self.dropped_renders = 0
        self.dropped_records = 0
        self.overflow_error: str | None = None
        self._sequence = 0
        self._frame_records: deque[tuple[int, LogRecord]] = deque()
        self._event_records: deque[tuple[int, LogRecord]] = deque()
        self._renders: deque[RenderJob] = deque()
        self._log_condition = threading.Condition()
        self._render_condition = threading.Condition()
        self._closed = False

    def event(self, event: str, **fields: JsonValue) -> None:
        overflowed = False
        with self._log_condition:
            queued = len(self._frame_records) + len(self._event_records)
            if event in DROPPABLE_EVENTS:
                if queued >= self.log_capacity:
                    self.dropped_records += 1
                    if not self._frame_records:
                        return
                    self._frame_records.popleft()
                self._frame_records.append((self._sequence, LogRecord(event, fields)))
            else:
                if self.overflow_error is not None:
                    return
                if queued >= self.log_capacity and self._frame_records:
                    self._frame_records.popleft()
                    self.dropped_records += 1
                if len(self._event_records) >= self.log_capacity + self.event_allowance:
                    self.overflow_error = (
                        f"log writer fell more than {self.log_capacity + self.event_allowance} "
                        "events behind; stopping because triggers can no longer be recorded"
                    )
                    overflowed = True
                else:
                    self._event_records.append((self._sequence, LogRecord(event, fields)))
            self._sequence += 1
            self._log_condition.notify_all()
        if overflowed:
            self.on_overflow()

    @property
    def pending_logs(self) -> int:
        with self._log_condition:
            return len(self._frame_records) + len(self._event_records)

    def take_logs(self, timeout_s: float = 0.0) -> list[LogRecord]:
        with self._log_condition:
            if timeout_s > 0:
                self._log_condition.wait_for(
                    lambda: self._frame_records or self._event_records, timeout_s
                )
            merged = heapq.merge(self._frame_records, self._event_records, key=itemgetter(0))
            records = [record for _, record in merged]
            self._frame_records.clear()
            self._event_records.clear()
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
    """Writes queued log records on its own thread until `finish` drains the queue.

    The thread is a daemon: a sink that blocks forever must not keep the process
    alive, and `finish` gives up after its deadline.
    """

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
        self._unwritten_batch = 0
        self._thread = threading.Thread(target=self.run, name="log-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def run(self) -> None:
        try:
            while True:
                done = self._done.is_set()
                records = self.queue.take_logs(0.0 if done else CONSUMER_POLL_S)
                self._unwritten_batch = len(records)
                for record in records:
                    self.output.event(record.event, **record.fields)
                    self._unwritten_batch -= 1
                    if self.observe is not None:
                        self.observe(record)
                if done and not records:
                    return
        except BaseException as error:
            self.error = error
            print(f"log writing failed: {describe(error)}", file=sys.stderr)
            self.on_failure()

    def finish(self, timeout_s: float) -> int | None:
        """Write everything still queued, then stop.

        Returns None once drained, or the number of records left unwritten when
        the writer is still busy after `timeout_s` (a stalled sink).
        """
        self._done.set()
        if self._thread.ident is None:
            return None
        self._thread.join(timeout_s)
        if not self._thread.is_alive():
            return None
        return self._unwritten_batch + self.queue.pending_logs


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
        follow_session: FollowSession | None = None,
        replay_fps: float | None = None,
        trigger_classes: tuple[str, ...] = ("pigeon",),
        publisher: FramePublisher | None = None,
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
        self.follow_session = follow_session
        self.replay_fps = replay_fps
        self.publisher = publisher
        self.motion_tracker = (
            MotionTracker(
                config.confirmation,
                trigger_classes,
                config.model.confidence,
                config.follow.maintenance_confidence,
                config.follow.retention_s,
            )
            if config.follow.enabled
            else None
        )
        self.target_selector = (
            TargetSelector(
                config.follow.retention_s,
                config.follow.switch_area_ratio,
                config.follow.switch_hold_s,
            )
            if config.follow.enabled
            else None
        )
        self._camera: str | None = None
        self._has_camera = False
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
                if self.follow_session is not None:
                    self.follow_session.check()
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
            if self.follow_session is not None:
                self.follow_session.request_stop()

    def follow_time(self, packet: FramePacket) -> tuple[float, float, ObservationTiming]:
        if self.replay_fps is not None:
            source_time_s = packet.sequence / self.replay_fps
            return source_time_s, source_time_s, "replay"
        now_s = time.monotonic()
        if packet.delivered_monotonic_s is None:
            return now_s, now_s, "unavailable"
        if packet.delivery_ms is None:
            return now_s, packet.delivered_monotonic_s, "delivery"
        observed_at_s = packet.delivered_monotonic_s - packet.delivery_ms / MILLISECONDS_PER_SECOND
        return now_s, observed_at_s, "sensor"

    def process(self, packet: FramePacket) -> None:
        result = self.result
        picked_up_monotonic_s = time.monotonic()
        target_events: list[TargetEvent] = []
        if self._has_camera and packet.camera != self._camera:
            if self.tracker is not None:
                self.tracker.reset()
            if self.motion_tracker is not None:
                self.motion_tracker.reset()
            if self.target_selector is not None:
                target_events.extend(self.target_selector.reset())
        self._camera = packet.camera
        self._has_camera = True
        inference_started_at_monotonic_s = time.monotonic()
        detections = self.detector.detect(packet)
        inference_finished_at_monotonic_s = time.monotonic()
        inference_s = inference_finished_at_monotonic_s - inference_started_at_monotonic_s
        triggers: list[TriggerEvent] = []
        confirmed_track_ids: frozenset[int] = frozenset()
        track_ms: float | None = None
        target: TargetSnapshot | None = None
        controller: ControllerStatus | None = None
        timing_source: ObservationTiming | None = None
        following_fields: dict[str, JsonValue] = {}
        if self.motion_tracker is not None and self.target_selector is not None:
            now_s, observed_at_s, timing_source = self.follow_time(packet)
            motion_update = self.motion_tracker.update(packet, detections, now_s, observed_at_s)
            detections, triggers = list(motion_update.detections), list(motion_update.triggers)
            confirmed_track_ids = self.motion_tracker.confirmed_track_ids
            selection = self.target_selector.update(motion_update.tracks, now_s)
            target = selection.target
            target_events.extend(selection.events)
            if self.follow_session is not None:
                height_px, width_px = packet.frame.shape[:2]
                controller = self.follow_session.submit(
                    target, width_px, height_px, now_s, timing_source
                )
            following_fields = {
                "target": target.as_dict(),
                "controller": controller.as_dict() if controller is not None else None,
                "follow_timebase": timing_source,
                "sensor_timing_available": packet.delivery_ms is not None,
                "acquisition_confidence": self.config.model.confidence,
                "maintenance_confidence": self.config.follow.maintenance_confidence,
                "pigeon_count_semantics": "confidence_floor_candidates",
            }
            track_ms = milliseconds(time.monotonic() - inference_finished_at_monotonic_s)
        elif self.tracker is not None:
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
        if self.render_frames or self.publisher is not None:
            job = RenderJob(
                packet.frame,
                detections,
                processing_fps,
                capture_fps,
                confirmed_track_ids,
                target,
                controller,
                packet.sequence,
                packet.captured_at,
                packet.delivered_monotonic_s,
                packet.sequence / self.replay_fps if self.replay_fps is not None else None,
            )
            if self.publisher is not None:
                self.publisher.publish(job)
            if self.render_frames:
                self.sink.render(job)
        self.sink.event(
            "frame",
            frame_number=packet.sequence,
            captured_at=packet.captured_at,
            inference_s=inference_s,
            pigeon_count=sum(detection.label == "pigeon" for detection in detections),
            detections=[
                {
                    **detection.as_dict(),
                    **(
                        {"maintenance_only": detection.confidence < self.config.model.confidence}
                        if self.config.follow.enabled
                        else {}
                    ),
                }
                for detection in detections
            ],
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
            **following_fields,
        )
        for trigger in triggers:
            self.sink.event("trigger", **trigger.as_dict())
        for target_event in target_events:
            self.sink.event(target_event.name, **target_event.as_dict(), timebase=timing_source)
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
