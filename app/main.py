from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Protocol, TypeAlias, cast

from .capture import LatestFrameStream, Picamera2Capture, VideoFileCapture
from .config import AppConfig, CameraConfig, ConfigError, ImageSize, ModelConfig
from .detector import DetectorError, FrameDetector, UltralyticsDetector, load_model_info
from .json_data import JsonValue
from .output import RunOutput, health_snapshot
from .pipeline import (
    CONSUMER_POLL_S,
    LOG_EVENT_ALLOWANCE,
    LOG_QUEUE_CAPACITY,
    FrameProcessor,
    LogWriter,
    OutputQueue,
    describe,
)
from .reaction import ReactionPlan, ReactionTest
from .switching import SwitchingCameraStream
from .tracking import ConfirmationTracker

REACTION_TRIALS = 10
REACTION_SUMMARY_KEYS = (
    "trials",
    "triggered",
    "missed",
    "false_triggers",
    "reaction_s",
    "first_detection_s",
)


@dataclass(frozen=True)
class CommandOptions:
    config: str = "app/config.yaml"
    model: str | None = None
    video: str | None = None
    confidence: float | None = None
    record: bool = False
    max_frames: int = 0
    reaction_test: str | None = None
    reaction_trials: int = REACTION_TRIALS


def load_configuration(options: CommandOptions) -> AppConfig:
    configuration = AppConfig.load(options.config)
    model = replace(
        configuration.model,
        path=options.model if options.model is not None else configuration.model.path,
        confidence=(
            options.confidence if options.confidence is not None else configuration.model.confidence
        ),
    )
    configuration = replace(
        configuration,
        model=model,
        output=replace(configuration.output, record=options.record or configuration.output.record),
    )
    if options.reaction_test is not None:
        if options.video:
            raise ConfigError("--reaction-test needs the live camera, not --video")
        if not Path(options.reaction_test).is_file():
            raise ConfigError(f"reaction-test image does not exist: {options.reaction_test}")
        if options.reaction_trials <= 0:
            raise ConfigError("--reaction-trials must be positive")
        # the monitor shows the stimulus, so the annotated view stays closed, and
        # reaction is measured to confirmed triggers.
        configuration = replace(
            configuration,
            display=replace(configuration.display, enabled=False),
            confirmation=replace(configuration.confirmation, enabled=True),
        )
    configuration.validate()
    if options.max_frames < 0:
        raise ConfigError("--max-frames must be non-negative")
    return configuration


class NativeDetectorFactory(Protocol):
    def __call__(
        self, model_path: str, image_size: ImageSize, confidence: float, threads: int = 4
    ) -> FrameDetector: ...


def create_camera(camera: CameraConfig) -> Picamera2Capture:
    return Picamera2Capture(
        camera.width,
        camera.height,
        camera.fps,
        stream=camera.stream,
        main_size=camera.main_size,
        lens_position=camera.lens_position,
        max_exposure_us=camera.max_exposure_us,
        tuning_file=camera.tuning_file,
        camera_index=camera.camera_index,
        name=camera.label,
    )


FrameStream: TypeAlias = VideoFileCapture | LatestFrameStream | SwitchingCameraStream


def create_source(config: AppConfig, video: str | None) -> FrameStream:
    if video:
        return VideoFileCapture(video)
    if config.cameras:
        return SwitchingCameraStream(
            {entry.role: config.camera_for(entry) for entry in config.cameras},
            config.camera_switch,
            lambda camera: LatestFrameStream(create_camera(camera), camera.label),
        )
    return LatestFrameStream(create_camera(config.camera), config.camera.label)


def create_detector(model: ModelConfig) -> tuple[FrameDetector, tuple[str, ...]]:
    """Build the configured backend; return it with the model's trigger classes."""
    if model.backend == "native":
        model_info = load_model_info(model.path, model.image_size)
        try:
            # WS8 owns app/native_detector.py; import by name so this module type-checks
            # before it exists.
            module = importlib.import_module(".native_detector", __package__)
            native_detector = cast(NativeDetectorFactory, module.NativeDetector)
        except (ImportError, AttributeError) as error:
            raise DetectorError(
                f"model.backend native is unavailable: {error}; build columba_native "
                "(native/) or use model.backend: ultralytics"
            ) from error
        native = native_detector(
            model.path, model.image_size, model.confidence, threads=model.threads
        )
        return native, model_info.trigger_classes
    detector = UltralyticsDetector(model.path, model.image_size, model.confidence)
    return detector, detector.trigger_classes


def finalize_output(
    output: RunOutput,
    reason: str,
    failure: str | None,
    processed_frames: int,
    dropped_frames: int,
    started_at_monotonic_s: float | None,
    **fields: JsonValue,
) -> bool:
    successful = True
    try:
        output.close_recording_and_display()
    except Exception as error:
        successful = False
        reason = "cleanup_error"
        cleanup_failure = f"output cleanup failed: {error}"
        failure = f"{failure}; {cleanup_failure}" if failure else cleanup_failure
        print(f"output cleanup error: {error}", file=sys.stderr)
    try:
        elapsed_s = (
            time.monotonic() - started_at_monotonic_s if started_at_monotonic_s is not None else 0.0
        )
        output.event(
            "run_end",
            reason=reason,
            error=failure,
            processed_frames=processed_frames,
            dropped_frames=dropped_frames,
            elapsed_s=elapsed_s,
            processing_fps=processed_frames / elapsed_s if elapsed_s else 0.0,
            **fields,
            **health_snapshot(),
        )
    except Exception as error:
        successful = False
        print(f"log finalization error: {error}", file=sys.stderr)
    finally:
        try:
            output.close()
        except Exception as error:
            successful = False
            print(f"output cleanup error: {error}", file=sys.stderr)
    return successful


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi 5 pigeon detection (no actuation)")
    parser.add_argument("--config", default="app/config.yaml")
    parser.add_argument("--model", help=".pt checkpoint or NCNN directory")
    parser.add_argument("--video", help="process all frames of a recorded video")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--record", action="store_true", help="save diagnostic annotated AVI")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--reaction-test",
        metavar="IMAGE",
        help="show IMAGE fullscreen at logged times for a camera facing the monitor "
        "and report display-to-trigger reaction times",
    )
    parser.add_argument("--reaction-trials", type=int, default=REACTION_TRIALS)
    return parser


def display_frames(
    output: RunOutput,
    queue: OutputQueue,
    processing: threading.Thread,
    stop: threading.Event,
    reaction: ReactionTest | None = None,
) -> bool:
    """Render frames (and drive the reaction test) on the main thread until processing
    ends. Logging runs separately in a LogWriter.

    Returns True when the window was closed or q was pressed.
    """
    window_closed = False
    while True:
        finished = not processing.is_alive()
        job = queue.take_render(0.0 if finished else CONSUMER_POLL_S)
        if reaction is not None and not finished:
            stimulus = reaction.tick()
            if stimulus is not None:
                queue.event("stimulus", **stimulus)
            if reaction.finished:
                stop.set()
        if window_closed:
            pass
        elif job is not None:
            window_closed = not output.render(
                job.frame,
                job.detections,
                job.processing_fps,
                job.capture_fps,
                confirmed_track_ids=job.confirmed_track_ids,
            )
        elif not finished:
            window_closed = not output.poll()
        if window_closed:
            stop.set()
            queue.close()
        if finished and job is None:
            return window_closed


def run(arguments: argparse.Namespace | CommandOptions) -> int:
    options = (
        arguments if isinstance(arguments, CommandOptions) else CommandOptions(**vars(arguments))
    )
    config = load_configuration(options)
    stop = threading.Event()
    handlers: dict[signal.Signals, int | Callable[[int, FrameType | None], None] | None] = {}
    source: FrameStream | None = None
    output: RunOutput | None = None
    queue: OutputQueue | None = None
    processor: FrameProcessor | None = None
    reaction: ReactionTest | None = None
    started_at_monotonic_s: float | None = None
    status = 0
    reason = "stopped"
    failure: str | None = None

    def request_stop(_signal_number: int, _frame: FrameType | None) -> None:
        stop.set()

    try:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            handlers[signal_number] = signal.signal(signal_number, request_stop)
        detector, trigger_classes = create_detector(config.model)
        detector.warmup(config.camera.width, config.camera.height)
        if stop.is_set():
            return 0
        tracker = (
            ConfirmationTracker(config.confirmation, trigger_classes)
            if config.confirmation.enabled
            else None
        )
        output = RunOutput(config)
        print(f"Run output: {output.directory}", flush=True)
        output.event(
            "run_start",
            config=cast(dict[str, JsonValue], asdict(config)),
            source=options.video or "csi_camera",
            timestamp_semantics="Unix seconds at frame delivery; replay is not source time",
        )
        output.start()
        if options.reaction_test is not None:
            reaction = ReactionTest(
                options.reaction_test,
                ReactionPlan(trials=options.reaction_trials),
                trigger_classes,
            )
            reaction.start()
        source = create_source(config, options.video)
        source.start()
        started_at_monotonic_s = time.monotonic()
        output.event("health", processed_frames=0, dropped_frames=0, **health_snapshot())
        live_source = (
            source if isinstance(source, (LatestFrameStream, SwitchingCameraStream)) else None
        )
        queue = OutputQueue(
            drop_oldest=not options.video,
            log_capacity=LOG_QUEUE_CAPACITY,
            event_allowance=LOG_EVENT_ALLOWANCE,
            on_overflow=stop.set,
        )
        processor = FrameProcessor(
            config,
            source,
            detector,
            tracker,
            queue,
            stop,
            started_at_monotonic_s,
            options.max_frames,
            capture_fps=(lambda: live_source.capture_fps) if live_source else lambda: None,
            camera_events=(
                source.take_events if isinstance(source, SwitchingCameraStream) else list
            ),
        )
        log_writer = LogWriter(
            output,
            queue,
            stop.set,
            observe=(
                (lambda record: reaction.analysis.observe(record.event, record.fields))
                if reaction is not None
                else None
            ),
        )
        processing = threading.Thread(target=processor.run, name="frame-processing")
        try:
            log_writer.start()
            processing.start()
            window_closed = display_frames(output, queue, processing, stop, reaction)
        finally:
            # stop and join the worker, then write every record it queued (even after a
            # display failure) before anything else is logged.
            stop.set()
            queue.close()
            if processing.ident is not None:
                processing.join()
            log_writer.finish()
        result = processor.result
        reason = result.reason
        processing_error = result.error
        if processing_error is None and not result.completed:
            processing_error = RuntimeError("frame processing ended without completing")
        if processing_error is not None:
            reason = "error"
            status, failure = 1, describe(processing_error)
            print(f"error: {failure}", file=sys.stderr)
        elif window_closed and reason == "stopped":
            reason = "window_closed_or_q"
        elif reaction is not None and reaction.finished and reason == "stopped":
            reason = "reaction_test_complete"
        log_failures: list[str] = []
        if log_writer.error is not None:
            log_failures.append(f"log writing failed: {describe(log_writer.error)}")
        if queue.overflow_error is not None:
            log_failures.append(queue.overflow_error)
        for log_failure in log_failures:
            print(f"error: {log_failure}", file=sys.stderr)
            status, reason = 1, "error"
            failure = f"{failure}; {log_failure}" if failure else log_failure
        if reaction is not None:
            report = reaction.analysis.report()
            output.event("reaction_test", **report)
            print(f"Reaction test report: {output.write_json('reaction_test.json', report)}")
            print(json.dumps({key: report[key] for key in REACTION_SUMMARY_KEYS}, indent=2))
    except Exception as error:
        status, reason, failure = 1, "error", str(error)
        print(f"error: {error}", file=sys.stderr)
    finally:
        if reaction is not None:
            try:
                reaction.close()
            except Exception as error:
                print(f"reaction-test window cleanup error: {error}", file=sys.stderr)
        if source is not None:
            try:
                source.close()
            except Exception as error:
                cleanup_failure = f"capture cleanup failed: {error}"
                status, reason = 1, "cleanup_error"
                failure = f"{failure}; {cleanup_failure}" if failure else str(error)
                print(f"capture cleanup error: {error}", file=sys.stderr)
        if output is not None:
            finalized = finalize_output(
                output,
                reason,
                failure,
                processor.result.processed_frames if processor else 0,
                processor.result.dropped_frames if processor else 0,
                started_at_monotonic_s,
                render_dropped_frames=queue.dropped_renders if queue else 0,
                log_dropped_records=queue.dropped_records if queue else 0,
            )
            if not finalized:
                status = 1
        for signal_number, handler in handlers.items():
            signal.signal(signal_number, handler)
    return status


def cli() -> None:
    options = build_parser().parse_args()
    try:
        status = run(options)
    except (ConfigError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    cli()
