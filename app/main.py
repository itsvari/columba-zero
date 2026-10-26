from __future__ import annotations

import argparse
import importlib
import json
import math
import shlex
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
from .demo_recording import FrameRecorder, RecordingStatus, export_recording
from .detector import DetectorError, FrameDetector, UltralyticsDetector, load_model_info
from .json_data import JsonValue
from .output import RunOutput, health_snapshot
from .pan_tilt_control import FollowSession
from .pipeline import (
    CONSUMER_POLL_S,
    LOG_DRAIN_TIMEOUT_S,
    LOG_EVENT_ALLOWANCE,
    LOG_QUEUE_CAPACITY,
    FrameProcessor,
    LogWriter,
    OutputQueue,
    describe,
)
from .preview import PreviewService
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
    configuration = AppConfig.load(
        options.config, follow_enabled=False if options.reaction_test is not None else None
    )
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
            follow=replace(configuration.follow, enabled=False),
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
        rotation_deg=camera.rotation_deg,
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
            # Imported on demand; columba_native itself loads when NativeDetector is built.
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


def replay_frame_rate(source: FrameStream, fallback_fps: float) -> tuple[float, str]:
    source_fps = getattr(source, "fps", None)
    if (
        isinstance(source_fps, (int, float))
        and not isinstance(source_fps, bool)
        and math.isfinite(source_fps)
        and source_fps > 0
    ):
        return float(source_fps), "source_fps"
    return fallback_fps, "configured_camera_fps"


def finish_demo_outputs(
    preview: PreviewService | None, recorder: FrameRecorder | None
) -> tuple[tuple[str, ...], RecordingStatus | None]:
    failures: list[str] = []
    if preview is not None:
        try:
            preview.close()
        except Exception as error:
            failures.append(f"preview failed: {describe(error)}")
        if preview.failure is not None and not failures:
            failures.append(f"preview failed: {preview.failure}")
    recording: RecordingStatus | None = None
    if recorder is not None:
        try:
            recording = recorder.finish()
            if recording.error is not None:
                failures.append(f"demonstration recording failed: {recording.error}")
        except Exception as error:
            failures.append(f"demonstration recording failed: {describe(error)}")
    return tuple(failures), recording


def finalize_output(
    output: RunOutput,
    reason: str,
    failure: str | None,
    processed_frames: int,
    dropped_frames: int,
    started_at_monotonic_s: float | None,
    *,
    log_available: bool = True,
    finished_at_monotonic_s: float | None = None,
    **fields: JsonValue,
) -> bool:
    """Close recording and display, then log run_end and close the log.

    With `log_available` false (the log writer is stalled inside the sink), run_end
    goes to stderr instead and the log is left alone, so that nothing waits on it.
    """
    successful = True
    try:
        output.close_recording_and_display()
    except Exception as error:
        successful = False
        reason = "cleanup_error"
        cleanup_failure = f"output cleanup failed: {error}"
        failure = f"{failure}; {cleanup_failure}" if failure else cleanup_failure
        print(f"output cleanup error: {error}", file=sys.stderr)
    if not log_available:
        print(
            f"run_end not logged: reason={reason}, error={failure}, "
            f"processed_frames={processed_frames}, dropped_frames={dropped_frames}",
            file=sys.stderr,
        )
        return False
    try:
        finished_at_s = (
            finished_at_monotonic_s if finished_at_monotonic_s is not None else time.monotonic()
        )
        elapsed_s = (
            finished_at_s - started_at_monotonic_s if started_at_monotonic_s is not None else 0.0
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
    parser = argparse.ArgumentParser(description="Pi 5 pigeon detection with opt-in following")
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
    follow_session: FollowSession | None = None,
) -> bool:
    """Render frames (and drive the reaction test) on the main thread until processing
    ends. Logging runs separately in a LogWriter.

    Returns True when the window was closed or q was pressed.
    """
    window_closed = False
    while True:
        if follow_session is not None:
            follow_session.check()
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
                target=job.target,
                controller=job.controller,
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
    log_writer: LogWriter | None = None
    processor: FrameProcessor | None = None
    reaction: ReactionTest | None = None
    follow_session: FollowSession | None = None
    follow_failure: str | None = None
    follow_cleanup_failure: str | None = None
    follow_close_attempted = False
    preview: PreviewService | None = None
    recorder: FrameRecorder | None = None
    recording: RecordingStatus | None = None
    demo_failures: tuple[str, ...] = ()
    demo_finish_attempted = False
    replay_fps: float | None = None
    log_unwritten: int | None = None
    started_at_monotonic_s: float | None = None
    processing_finished_at_monotonic_s: float | None = None
    status = 0
    reason = "stopped"
    failure: str | None = None
    export_in_progress = False
    export_cancel_requested = threading.Event()

    def request_stop(_signal_number: int, _frame: FrameType | None) -> None:
        stop.set()
        if export_in_progress:
            export_cancel_requested.set()

    try:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            handlers[signal_number] = signal.signal(signal_number, request_stop)
        detection_model = (
            replace(config.model, confidence=config.follow.maintenance_confidence)
            if config.follow.enabled
            else config.model
        )
        detector, trigger_classes = create_detector(detection_model)
        detector.warmup(config.camera.width, config.camera.height)
        if stop.is_set():
            return 0
        tracker = (
            ConfirmationTracker(config.confirmation, trigger_classes)
            if config.confirmation.enabled and not config.follow.enabled
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
        if options.video:
            replay_fps, replay_clock = replay_frame_rate(source, config.camera.fps)
            if config.follow.enabled:
                queue.event("follow_replay_clock", fps=replay_fps, source=replay_clock)
        if config.follow.enabled and not stop.is_set():
            follow_session = FollowSession(
                config.pan_tilt,
                config.follow.retention_s,
                replay=bool(options.video),
                stop=stop,
                events=queue,
                image_rotation_deg=0 if options.video else config.camera.rotation_deg,
            )
        if config.preview.record and not stop.is_set():
            recorder = FrameRecorder(config.preview, output.directory, queue)
            recorder.start()
        if (config.preview.enabled or config.preview.record) and not stop.is_set():
            preview = PreviewService(
                config,
                queue,
                recorder=recorder,
                control=follow_session.request if follow_session is not None else None,
                controller_state=follow_session.snapshot if follow_session is not None else None,
            )
            preview.start()
            if preview.address is not None:
                host, port = preview.address
                print(f"Browser preview: http://{host}:{port}", flush=True)
        if follow_session is not None:
            follow_session.start()
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
            follow_session=follow_session,
            replay_fps=replay_fps,
            trigger_classes=trigger_classes,
            publisher=preview,
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
            window_closed = display_frames(
                output, queue, processing, stop, reaction, follow_session
            )
        finally:
            # stop and join the worker, then write every record it queued (even after a
            # display failure) before anything else is logged.
            stop.set()
            queue.close()
            if processing.ident is not None:
                processing.join()
            processing_finished_at_monotonic_s = time.monotonic()
            if follow_session is not None:
                try:
                    follow_session.check()
                except Exception as error:
                    follow_failure = describe(error)
                follow_close_attempted = True
                try:
                    follow_session.close()
                except Exception as error:
                    follow_cleanup_failure = describe(error)
            demo_finish_attempted = True
            demo_failures, recording = finish_demo_outputs(preview, recorder)
            log_unwritten = log_writer.finish(LOG_DRAIN_TIMEOUT_S)
            if log_unwritten is not None:
                print(
                    f"error: log writer stalled; {log_unwritten} records unwritten",
                    file=sys.stderr,
                )
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
        if reaction is not None:
            report = reaction.analysis.report()
            if log_unwritten is None:
                output.event("reaction_test", **report)
            print(f"Reaction test report: {output.write_json('reaction_test.json', report)}")
            print(json.dumps({key: report[key] for key in REACTION_SUMMARY_KEYS}, indent=2))
    except Exception as error:
        status, reason, failure = 1, "error", str(error)
        print(f"error: {error}", file=sys.stderr)
    finally:
        if follow_session is not None and not follow_close_attempted:
            try:
                follow_session.close()
            except Exception as error:
                follow_cleanup_failure = follow_cleanup_failure or describe(error)
        if follow_failure is not None and follow_failure not in (failure or ""):
            status, reason = 1, "error"
            failure = f"{failure}; {follow_failure}" if failure else follow_failure
        log_failures: list[str] = []
        if log_writer is not None and log_writer.error is not None:
            log_failures.append(f"log writing failed: {describe(log_writer.error)}")
        if queue is not None and queue.overflow_error is not None:
            log_failures.append(queue.overflow_error)
        if log_unwritten is not None:
            log_failures.append(f"log writer stalled; {log_unwritten} records unwritten")
        for log_failure in log_failures:
            status, reason = 1, "error"
            if log_failure not in (failure or ""):
                failure = f"{failure}; {log_failure}" if failure else log_failure
            print(f"error: {log_failure}", file=sys.stderr)
        if follow_cleanup_failure is not None:
            status = 1
            reason = reason if failure else "cleanup_error"
            cleanup_message = f"pan-tilt cleanup failed: {follow_cleanup_failure}"
            failure = f"{failure}; {cleanup_message}" if failure else cleanup_message
            print(f"error: {cleanup_message}", file=sys.stderr)
        if not demo_finish_attempted:
            demo_failures, recording = finish_demo_outputs(preview, recorder)
        for demo_failure in demo_failures:
            status, reason = 1, "error"
            failure = f"{failure}; {demo_failure}" if failure else demo_failure
            print(f"error: {demo_failure}", file=sys.stderr)
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
            demo_fields: dict[str, JsonValue] = {}
            if recording is not None:
                demo_fields["demo_recording"] = recording.as_dict()
            if recording is not None and recording.error is None and recording.frames_written:
                final_frame_duration_s = 1.0 / (
                    replay_fps if replay_fps is not None else config.preview.max_fps
                )
                print(
                    "Exporting demonstration MP4. Ctrl+C cancels export and keeps source frames.",
                    flush=True,
                )
                export_in_progress = True
                try:
                    export = export_recording(
                        output.directory / "demo",
                        output.directory / "annotated.mp4",
                        timeout_s=config.preview.export_timeout_s,
                        default_frame_duration_s=final_frame_duration_s,
                        max_bytes=config.preview.record_max_bytes,
                        cancel_requested=export_cancel_requested.is_set,
                    )
                    demo_fields["demo_export"] = export.as_dict()
                    if log_unwritten is None:
                        output.event("demo_export", **export.as_dict())
                    export_failure = (
                        None if export.success else f"demonstration export failed: {export.error}"
                    )
                except Exception as error:
                    export_failure = f"demonstration export failed: {describe(error)}"
                finally:
                    export_in_progress = False
                if export_failure is not None:
                    status, reason = 1, "error"
                    failure = f"{failure}; {export_failure}" if failure else export_failure
                    print(f"error: {export_failure}", file=sys.stderr)
                    retry_command = shlex.join(
                        [
                            "python",
                            "tools/export_demo.py",
                            str(output.directory / "demo"),
                            str(output.directory / "annotated.mp4"),
                            "--frame-duration-s",
                            str(final_frame_duration_s),
                            "--timeout-s",
                            str(config.preview.export_timeout_s),
                            "--max-bytes",
                            str(config.preview.record_max_bytes),
                        ]
                    )
                    print(
                        f"Retry after resolving the export error: {retry_command}", file=sys.stderr
                    )
            finalized = finalize_output(
                output,
                reason,
                failure,
                processor.result.processed_frames if processor else 0,
                processor.result.dropped_frames if processor else 0,
                started_at_monotonic_s,
                log_available=log_unwritten is None,
                finished_at_monotonic_s=processing_finished_at_monotonic_s,
                render_dropped_frames=queue.dropped_renders if queue else 0,
                log_dropped_records=queue.dropped_records if queue else 0,
                **demo_fields,
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
