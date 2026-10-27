"""Bounded JPEG spooling and post-workload variable-frame-rate MP4 export."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

from .config import PreviewConfig
from .demo_frames import EncodedFrame
from .json_data import JsonValue

RECORDING_QUEUE_CAPACITY = 8
WRITER_POLL_INTERVAL_S = 0.05
DEFAULT_FINISH_TIMEOUT_S = 2.0
DEFAULT_EXPORT_TIMEOUT_S = 900.0
DEFAULT_FRAME_DURATION_S = 0.1
DEFAULT_MAX_BYTES = 536870912
INPUT_TIMEBASE_HZ = 1000
MAXIMUM_CONSECUTIVE_B_FRAMES = 0
DURATION_TOLERANCE_S = 0.01
SUBPROCESS_POLL_INTERVAL_S = 0.02
SUBPROCESS_STOP_TIMEOUT_S = 1.0
MAX_PROBE_BYTES = 1048576
FRAME_NAME_PATTERN = re.compile(r"frame_[0-9]{9}\.jpg")
RecordingState = Literal["idle", "recording", "finished", "limited", "error"]


class RecordingEventSink(Protocol):
    def event(self, event: str, **fields: JsonValue) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordingStatus:
    state: RecordingState = "idle"
    frames_written: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    first_monotonic_s: float | None = None
    last_monotonic_s: float | None = None
    error: str | None = None
    limit_reason: str | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "state": self.state,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "bytes_written": self.bytes_written,
            "first_monotonic_s": self.first_monotonic_s,
            "last_monotonic_s": self.last_monotonic_s,
            "error": self.error,
            "limit_reason": self.limit_reason,
        }


class FrameRecorder:
    def __init__(self, config: PreviewConfig, directory: Path, events: RecordingEventSink) -> None:
        self._config = config
        self._spool_directory = directory / "demo"
        self._events = events
        self._pending: queue.Queue[EncodedFrame] = queue.Queue(RECORDING_QUEUE_CAPACITY)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._status = RecordingStatus()
        self._worker: threading.Thread | None = None
        self._finish_called = False
        self._timebase: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._worker is not None or self._finish_called:
                return
            self._status = replace(self._status, state="recording")
            self._worker = threading.Thread(target=self._write, name="demo-recorder", daemon=True)
            self._worker.start()

    def offer(self, frame: EncodedFrame) -> None:
        with self._lock:
            if self._closing.is_set() or self._status.state != "recording":
                return
            try:
                self._pending.put_nowait(frame)
            except queue.Full:
                self._status = replace(self._status, frames_dropped=self._status.frames_dropped + 1)

    def snapshot(self) -> RecordingStatus:
        with self._lock:
            return self._status

    def finish(self, timeout_s: float = DEFAULT_FINISH_TIMEOUT_S) -> RecordingStatus:
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("finish timeout_s must be finite and non-negative")
        with self._lock:
            if self._finish_called:
                return self._status
            self._finish_called = True
            self._closing.set()
            worker = self._worker
        if worker is not None:
            worker.join(timeout_s)
            if worker.is_alive():
                self._stop("error", "recording writer did not stop before timeout")
        with self._lock:
            if self._status.state in ("idle", "recording"):
                self._status = replace(self._status, state="finished")
            return self._status

    def _stop(self, state: Literal["limited", "error"], reason: str) -> None:
        with self._lock:
            if self._status.state in ("error", "limited"):
                return
            self._closing.set()
            self._status = replace(
                self._status,
                state=state,
                error=reason if state == "error" else None,
                limit_reason=reason if state == "limited" else None,
            )
        self._events.event(
            "recording_limit" if state == "limited" else "recording_error", reason=reason
        )

    def _write(self) -> None:
        current_frame_pending = False
        try:
            self._spool_directory.mkdir(parents=True, exist_ok=True)
            with (self._spool_directory / "index.jsonl").open("xb") as index:
                while not self._closing.is_set() or not self._pending.empty():
                    if self.snapshot().state != "recording":
                        break
                    try:
                        frame = self._pending.get(timeout=WRITER_POLL_INTERVAL_S)
                    except queue.Empty:
                        continue
                    current_frame_pending = True
                    record = self._prepare(frame)
                    if record is None:
                        current_frame_pending = False
                        break
                    filename, encoded_metadata = record
                    source_path = self._spool_directory / filename
                    if source_path.exists():
                        raise FileExistsError(f"recording source already exists: {filename}")
                    source_path.write_bytes(frame.jpeg)
                    index.write(encoded_metadata)
                    index.flush()
                    self._record_written(frame, len(frame.jpeg) + len(encoded_metadata))
                    current_frame_pending = False
        except (OSError, ValueError, TypeError) as error:
            if current_frame_pending:
                self._drop_current()
            self._stop("error", str(error))
        finally:
            self._discard_pending()

    def _prepare(self, frame: EncodedFrame) -> tuple[str, bytes] | None:
        status = self.snapshot()
        if not math.isfinite(frame.monotonic_s) or not math.isfinite(frame.captured_at):
            raise ValueError("recording timestamps must be finite")
        if status.last_monotonic_s is not None and frame.monotonic_s <= status.last_monotonic_s:
            raise ValueError("recording timestamps must increase strictly")
        if self._timebase is not None and self._timebase != frame.timebase:
            raise ValueError("recording presentation timebase changed")
        if status.first_monotonic_s is not None:
            if frame.monotonic_s - status.first_monotonic_s > self._config.record_max_duration_s:
                self._stop("limited", "recording duration limit reached")
                self._drop_current()
                return None
        filename = f"frame_{status.frames_written:09d}.jpg"
        metadata: dict[str, JsonValue] = {
            "filename": filename,
            "sequence": frame.sequence,
            "captured_at": frame.captured_at,
            "monotonic_s": frame.monotonic_s,
            "timebase": frame.timebase,
            "metadata": dict(frame.metadata),
        }
        encoded_metadata = (
            json.dumps(metadata, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()
        additional_bytes = len(frame.jpeg) + len(encoded_metadata)
        if status.bytes_written + additional_bytes > self._config.record_max_bytes:
            self._stop("limited", "recording byte limit reached")
            self._drop_current()
            return None
        if (
            shutil.disk_usage(self._spool_directory).free - additional_bytes
            < self._config.record_min_free_bytes
        ):
            self._stop("limited", "recording free-space limit reached")
            self._drop_current()
            return None
        self._timebase = frame.timebase
        return filename, encoded_metadata

    def _record_written(self, frame: EncodedFrame, size_bytes: int) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                frames_written=self._status.frames_written + 1,
                bytes_written=self._status.bytes_written + size_bytes,
                first_monotonic_s=(
                    frame.monotonic_s
                    if self._status.first_monotonic_s is None
                    else self._status.first_monotonic_s
                ),
                last_monotonic_s=frame.monotonic_s,
            )

    def _drop_current(self) -> None:
        with self._lock:
            self._status = replace(self._status, frames_dropped=self._status.frames_dropped + 1)

    def _discard_pending(self) -> None:
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                return
            self._drop_current()


@dataclass(frozen=True, slots=True)
class ExportResult:
    success: bool
    output_path: Path | None
    duration_s: float | None
    expected_duration_s: float
    error: str | None
    source_retained: bool

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "duration_s": self.duration_s,
            "expected_duration_s": self.expected_duration_s,
            "error": self.error,
            "source_retained": self.source_retained,
        }


@dataclass(frozen=True, slots=True)
class _SourceFrame:
    filename: str
    monotonic_s: float


def _read_sources(spool_directory: Path, max_bytes: int) -> tuple[_SourceFrame, ...]:
    index_path = spool_directory / "index.jsonl"
    with index_path.open("rb") as index:
        encoded_index = index.read(max_bytes + 1)
    total_bytes = len(encoded_index)
    if total_bytes > max_bytes:
        raise ValueError("recording index exceeds byte limit")
    frames: list[_SourceFrame] = []
    timebase: JsonValue = None
    for line in encoded_index.splitlines():
        record = cast(JsonValue, json.loads(line))
        if not isinstance(record, dict):
            raise ValueError("invalid recording index entry")
        filename = record.get("filename")
        timestamp_s = record.get("monotonic_s")
        if not isinstance(filename, str) or FRAME_NAME_PATTERN.fullmatch(filename) is None:
            raise ValueError("unsafe recording frame filename")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
            raise ValueError("invalid recording timestamp")
        if not math.isfinite(timestamp_s) or (frames and timestamp_s <= frames[-1].monotonic_s):
            raise ValueError("recording timestamps must be finite and increase strictly")
        if frames and record.get("timebase") != timebase:
            raise ValueError("recording presentation timebase changed")
        timebase = record.get("timebase")
        source_path = spool_directory / filename
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("recording source frame missing or unsafe")
        total_bytes += source_path.stat().st_size
        if total_bytes > max_bytes:
            raise ValueError("recording sources exceed byte limit")
        frames.append(_SourceFrame(filename, float(timestamp_s)))
    if not frames:
        raise ValueError("recording index is empty")
    return tuple(frames)


def _check_export_interruption(
    deadline_s: float, cancel_requested: Callable[[], bool] | None
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ValueError("recording export cancelled")
    if time.monotonic() >= deadline_s:
        raise TimeoutError("recording export timed out")


def _run_command(
    arguments: list[str],
    deadline_s: float,
    bounded_path: Path,
    max_bytes: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> bytes:
    with tempfile.TemporaryFile() as output:
        _check_export_interruption(deadline_s, cancel_requested)
        process = subprocess.Popen(arguments, stdout=output, stderr=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                _check_export_interruption(deadline_s, cancel_requested)
                if bounded_path.exists() and bounded_path.stat().st_size > max_bytes:
                    raise ValueError("recording export exceeds byte limit")
                if output.tell() > MAX_PROBE_BYTES:
                    raise ValueError("recording probe output exceeds byte limit")
                time.sleep(min(SUBPROCESS_POLL_INTERVAL_S, max(0.0, deadline_s - time.monotonic())))
            _check_export_interruption(deadline_s, cancel_requested)
            if process.returncode != 0:
                raise ValueError(
                    f"{Path(arguments[0]).name} failed with status {process.returncode}"
                )
            if bounded_path.stat().st_size > max_bytes:
                raise ValueError("recording export exceeds byte limit")
            output.seek(0)
            result = output.read(MAX_PROBE_BYTES + 1)
            if len(result) > MAX_PROBE_BYTES:
                raise ValueError("recording probe output exceeds byte limit")
            return result
        finally:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=SUBPROCESS_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    "recording export process did not stop after termination"
                ) from error


def _probe_duration(encoded_probe: bytes, expected_duration_s: float, frame_count: int) -> float:
    probe = cast(JsonValue, json.loads(encoded_probe))
    if not isinstance(probe, dict):
        raise ValueError("invalid ffprobe output")
    streams = probe.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError("export has no single playable video stream")
    stream = streams[0]
    if stream.get("codec_type") != "video" or not stream.get("codec_name"):
        raise ValueError("export has no playable video stream")
    for field in ("width", "height", "nb_read_frames"):
        value = stream.get(field)
        if not isinstance(value, (str, int)) or int(value) <= 0:
            raise ValueError("export contains no decodable video frames")
    if int(str(stream["nb_read_frames"])) != frame_count:
        raise ValueError("export decoded frame count differs from recording")
    container = probe.get("format")
    if not isinstance(container, dict):
        raise ValueError("export duration is unavailable")
    duration = container.get("duration")
    if not isinstance(duration, (str, int, float)):
        raise ValueError("export duration is unavailable")
    duration_s = float(duration)
    if (
        not math.isfinite(duration_s)
        or abs(duration_s - expected_duration_s) > DURATION_TOLERANCE_S
    ):
        raise ValueError(
            f"export duration {duration_s:g}s differs from expected {expected_duration_s:g}s"
        )
    return duration_s


def export_recording(
    spool_directory: Path,
    output_path: Path,
    timeout_s: float = DEFAULT_EXPORT_TIMEOUT_S,
    default_frame_duration_s: float = DEFAULT_FRAME_DURATION_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> ExportResult:
    expected_duration_s = 0.0
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("export timeout_s must be finite and positive")
        if not math.isfinite(default_frame_duration_s) or default_frame_duration_s <= 0:
            raise ValueError("final frame duration must be finite and positive")
        if max_bytes <= 0:
            raise ValueError("export max_bytes must be positive")
        if "\n" in str(spool_directory) or "\r" in str(spool_directory):
            raise ValueError("recording source path contains a line break")
        deadline_s = time.monotonic() + timeout_s
        _check_export_interruption(deadline_s, cancel_requested)
        frames = _read_sources(spool_directory, max_bytes)
        protected_paths = (spool_directory / "index.jsonl",) + tuple(
            spool_directory / frame.filename for frame in frames
        )
        if any(output_path.resolve() == source.resolve() for source in protected_paths):
            raise ValueError("recording output would replace a source file")
        expected_duration_s = (
            frames[-1].monotonic_s - frames[0].monotonic_s + default_frame_duration_s
        )
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise ValueError("FFmpeg and ffprobe are required for recording export")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".demo-export-", dir=output_path.parent
        ) as temporary:
            temporary_directory = Path(temporary)
            temporary_video = temporary_directory / "recording.mp4"
            concat_path = temporary_directory / "frames.ffconcat"
            lines = ["ffconcat version 1.0"]
            for position, frame in enumerate(frames):
                duration_s = default_frame_duration_s
                if position + 1 < len(frames):
                    duration_s = frames[position + 1].monotonic_s - frame.monotonic_s
                source = str((spool_directory / frame.filename).resolve()).replace("'", "'\\''")
                lines.extend(
                    [
                        f"file '{source}'",
                        f"option framerate {INPUT_TIMEBASE_HZ}",
                        f"duration {duration_s:.9f}",
                    ]
                )
            lines.extend([f"file '{source}'", f"option framerate {INPUT_TIMEBASE_HZ}"])
            concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _run_command(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path),
                    "-an",
                    "-fps_mode",
                    "vfr",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    # reordered variable-rate packets can truncate the MP4 duration.
                    "-bf",
                    str(MAXIMUM_CONSECUTIVE_B_FRAMES),
                    "-pix_fmt",
                    "yuv420p",
                    "-enc_time_base",
                    f"1:{INPUT_TIMEBASE_HZ}",
                    "-video_track_timescale",
                    str(INPUT_TIMEBASE_HZ),
                    "-movflags",
                    "+faststart",
                    "-fs",
                    str(max_bytes),
                    str(temporary_video),
                ],
                deadline_s,
                temporary_video,
                max_bytes,
                cancel_requested=cancel_requested,
            )
            probe = _run_command(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_type,codec_name,width,height,nb_read_frames:format=duration",
                    "-of",
                    "json",
                    str(temporary_video),
                ],
                deadline_s,
                temporary_video,
                max_bytes,
                cancel_requested=cancel_requested,
            )
            duration_s = _probe_duration(probe, expected_duration_s, len(frames) + 1)
            _check_export_interruption(deadline_s, cancel_requested)
            os.replace(temporary_video, output_path)
        return ExportResult(True, output_path, duration_s, expected_duration_s, None, True)
    except Exception as error:
        # export failures must not mask the original workload error during cleanup.
        return ExportResult(False, None, None, expected_duration_s, str(error), True)
