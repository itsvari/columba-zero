from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import asdict, replace

from .capture import EndOfStream, LatestFrameStream, Picamera2Capture, VideoFileCapture
from .config import AppConfig, ConfigError
from .detector import UltralyticsDetector
from .output import RunOutput, health_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi 5 pigeon detection (no actuation)")
    parser.add_argument("--config", default="app/config.yaml")
    parser.add_argument("--model", help=".pt checkpoint or NCNN directory")
    parser.add_argument("--video", help="process all frames of a recorded video")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--record", action="store_true", help="save diagnostic annotated AVI")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    return parser


def run(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    model = replace(config.model,
                    path=args.model if args.model is not None else config.model.path,
                    confidence=args.confidence if args.confidence is not None else config.model.confidence)
    config = replace(config, model=model,
                     output=replace(config.output, record=args.record or config.output.record))
    config.validate()
    if args.max_frames < 0:
        raise ConfigError("--max-frames must be non-negative")
    stopping = False
    handlers = {}
    source = None
    output = None
    processed = 0
    dropped = 0
    started = None
    status = 0
    reason = "stopped"
    failure = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        detector = UltralyticsDetector(model.path, model.image_size, model.confidence)
        detector.warmup(config.camera.width, config.camera.height)
        if stopping:
            return 0
        output = RunOutput(config)
        print(f"Run output: {output.directory}", flush=True)
        output.event("run_start", config=asdict(config), source=args.video or "csi_camera",
                     timestamp_semantics="Unix seconds at frame delivery; replay is not source time")
        output.start()
        source = (VideoFileCapture(args.video) if args.video else
                  LatestFrameStream(Picamera2Capture(config.camera.width, config.camera.height,
                                                     config.camera.fps)))
        source.start()
        started = time.monotonic()
        last_health = started
        output.event("health", processed_frames=0, dropped_frames=0, **health_snapshot())
        while not stopping:
            packet = source.get(config.camera.frame_timeout_s)
            inference_started = time.monotonic()
            detections = detector.detect(packet)
            inference_s = time.monotonic() - inference_started
            dropped = source.dropped_frames
            next_count = processed + 1
            elapsed = time.monotonic() - started
            fps = next_count / elapsed if elapsed else 0.0
            capture_fps = source.capture_fps if isinstance(source, LatestFrameStream) else None
            keep_running = output.render(packet.frame, detections, fps, capture_fps)
            processed = next_count
            output.event("frame", frame_number=packet.sequence, captured_at=packet.captured_at,
                         inference_s=inference_s, pigeon_count=len(detections),
                         detections=[d.as_dict() for d in detections],
                         dropped_frames=dropped, processing_fps=fps, capture_fps=capture_fps)
            now = time.monotonic()
            if now - last_health >= config.output.health_interval_s:
                output.event("health", processed_frames=processed, dropped_frames=dropped,
                             **health_snapshot())
                last_health = now
            if not keep_running:
                reason = "window_closed_or_q"
                break
            if args.max_frames and processed >= args.max_frames:
                reason = "max_frames"
                break
    except EndOfStream:
        reason = "video_complete"
    except Exception as exc:
        status, reason, failure = 1, "error", str(exc)
        print(f"error: {exc}", file=sys.stderr)
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                status, reason, failure = 1, "cleanup_error", str(exc)
                print(f"capture cleanup error: {exc}", file=sys.stderr)
        if output is not None:
            try:
                elapsed = time.monotonic() - started if started is not None else 0.0
                output.event("run_end", reason=reason, error=failure, processed_frames=processed,
                             dropped_frames=dropped, elapsed_s=elapsed,
                             processing_fps=processed / elapsed if elapsed else 0.0,
                             **health_snapshot())
            except Exception as exc:
                status = 1
                print(f"log finalization error: {exc}", file=sys.stderr)
            finally:
                try:
                    output.close()
                except Exception as exc:
                    status = 1
                    print(f"output cleanup error: {exc}", file=sys.stderr)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return status


def cli() -> None:
    args = build_parser().parse_args()
    try:
        status = run(args)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    cli()
