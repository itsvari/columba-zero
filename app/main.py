from __future__ import annotations

import argparse
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from types import FrameType
from typing import cast

from .capture import EndOfStream, LatestFrameStream, Picamera2Capture, VideoFileCapture
from .config import AppConfig, ConfigError
from .detector import UltralyticsDetector
from .json_data import JsonValue
from .output import RunOutput, health_snapshot


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


def finalize_output(
    output: RunOutput,
    reason: str,
    failure: str | None,
    processed_frames: int,
    dropped_frames: int,
    started_at_monotonic_s: float | None,
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


def run(arguments: argparse.Namespace | CommandOptions) -> int:
    options = (
        arguments if isinstance(arguments, CommandOptions) else CommandOptions(**vars(arguments))
    )
    config = load_configuration(options)
    stopping = False
    handlers: dict[signal.Signals, int | Callable[[int, FrameType | None], None] | None] = {}
    source: VideoFileCapture | LatestFrameStream | None = None
    output: RunOutput | None = None
    processed_frames = 0
    dropped_frames = 0
    started_at_monotonic_s: float | None = None
    status = 0
    reason = "stopped"
    failure: str | None = None

    def request_stop(_signal_number: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    try:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            handlers[signal_number] = signal.signal(signal_number, request_stop)
        detector = UltralyticsDetector(
            config.model.path, config.model.image_size, config.model.confidence
        )
        detector.warmup(config.camera.width, config.camera.height)
        if stopping:
            return 0
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
        last_health_at_monotonic_s = started_at_monotonic_s
        output.event("health", processed_frames=0, dropped_frames=0, **health_snapshot())
        while not stopping:
            packet = source.get(config.camera.frame_timeout_s)
            inference_started_at_monotonic_s = time.monotonic()
            detections = detector.detect(packet)
            inference_s = time.monotonic() - inference_started_at_monotonic_s
            dropped_frames = source.dropped_frames
            next_processed_frames = processed_frames + 1
            elapsed_s = time.monotonic() - started_at_monotonic_s
            processing_fps = next_processed_frames / elapsed_s if elapsed_s else 0.0
            capture_fps = source.capture_fps if isinstance(source, LatestFrameStream) else None
            keep_running = output.render(packet.frame, detections, processing_fps, capture_fps)
            processed_frames = next_processed_frames
            output.event(
                "frame",
                frame_number=packet.sequence,
                captured_at=packet.captured_at,
                inference_s=inference_s,
                pigeon_count=len(detections),
                detections=[detection.as_dict() for detection in detections],
                dropped_frames=dropped_frames,
                processing_fps=processing_fps,
                capture_fps=capture_fps,
            )
            now_monotonic_s = time.monotonic()
            if now_monotonic_s - last_health_at_monotonic_s >= config.output.health_interval_s:
                output.event(
                    "health",
                    processed_frames=processed_frames,
                    dropped_frames=dropped_frames,
                    **health_snapshot(),
                )
                last_health_at_monotonic_s = now_monotonic_s
            if not keep_running:
                reason = "window_closed_or_q"
                break
            if options.max_frames and processed_frames >= options.max_frames:
                reason = "max_frames"
                break
    except EndOfStream:
        reason = "video_complete"
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
                processed_frames,
                dropped_frames,
                started_at_monotonic_s,
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
