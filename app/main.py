from __future__ import annotations

import argparse
import importlib
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from types import FrameType
from typing import Protocol, cast

from .capture import LatestFrameStream, Picamera2Capture, VideoFileCapture
from .config import AppConfig, ConfigError, ImageSize, ModelConfig
from .detector import DetectorError, FrameDetector, UltralyticsDetector, load_model_info
from .json_data import JsonValue
from .output import RunOutput, health_snapshot
from .pipeline import CONSUMER_POLL_S, FrameProcessor, OutputQueue
from .tracking import ConfirmationTracker


@dataclass(frozen=True)
class CommandOptions:
    config: str = "app/config.yaml"
    model: str | None = None
    video: str | None = None
    confidence: float | None = None
    record: bool = False
    max_frames: int = 0


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
    configuration.validate()
    if options.max_frames < 0:
        raise ConfigError("--max-frames must be non-negative")
    return configuration


class NativeDetectorFactory(Protocol):
    def __call__(
        self, model_path: str, image_size: ImageSize, confidence: float, threads: int = 4
    ) -> FrameDetector: ...


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
    return parser


def consume_output(
    output: RunOutput,
    queue: OutputQueue,
    processing: threading.Thread,
    stop: threading.Event,
) -> bool:
    """Write logs and render frames on the main thread until processing ends.

    Returns True when the window was closed or q was pressed.
    """
    window_closed = False
    while True:
        finished = not processing.is_alive()
        records, job = queue.take(0.0 if finished else CONSUMER_POLL_S)
        for record in records:
            output.event(record.event, **record.fields)
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
        if finished and not records and job is None:
            return window_closed


def run(arguments: argparse.Namespace | CommandOptions) -> int:
    options = (
        arguments if isinstance(arguments, CommandOptions) else CommandOptions(**vars(arguments))
    )
    config = load_configuration(options)
    stop = threading.Event()
    handlers: dict[signal.Signals, int | Callable[[int, FrameType | None], None] | None] = {}
    source: VideoFileCapture | LatestFrameStream | None = None
    output: RunOutput | None = None
    queue: OutputQueue | None = None
    processor: FrameProcessor | None = None
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
        source = (
            VideoFileCapture(options.video)
            if options.video
            else LatestFrameStream(
                Picamera2Capture(config.camera.width, config.camera.height, config.camera.fps)
            )
        )
        source.start()
        started_at_monotonic_s = time.monotonic()
        output.event("health", processed_frames=0, dropped_frames=0, **health_snapshot())
        live_source = source if isinstance(source, LatestFrameStream) else None
        queue = OutputQueue(drop_oldest=not options.video)
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
        )
        processing = threading.Thread(target=processor.run, name="frame-processing")
        processing.start()
        try:
            window_closed = consume_output(output, queue, processing, stop)
        finally:
            stop.set()
            queue.close()
            processing.join()
        result = processor.result
        reason = result.reason
        if result.error is not None:
            status, failure = 1, str(result.error)
            print(f"error: {result.error}", file=sys.stderr)
        elif window_closed and reason == "stopped":
            reason = "window_closed_or_q"
    except Exception as error:
        status, reason, failure = 1, "error", str(error)
        print(f"error: {error}", file=sys.stderr)
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as error:
                status, reason, failure = 1, "cleanup_error", str(error)
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
