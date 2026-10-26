from __future__ import annotations

import errno
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from app.config import PreviewConfig
from app.demo_frames import EncodedFrame
from app.demo_recording import (
    RECORDING_QUEUE_CAPACITY,
    FrameRecorder,
    RecordingStatus,
    export_recording,
)
from app.json_data import JsonValue

TEST_TIMEOUT_S = 2.0
NONBLOCKING_BUDGET_S = 0.25
FRAME_COUNT = 4


class Events:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, JsonValue]]] = []

    def event(self, event: str, **fields: JsonValue) -> None:
        self.records.append((event, fields))


def frame(sequence: int = 0, monotonic_s: float = 10.0, jpeg: bytes = b"jpeg") -> EncodedFrame:
    return EncodedFrame(
        sequence=sequence,
        captured_at=1000.0 + sequence,
        monotonic_s=monotonic_s,
        timebase="replay_source",
        jpeg=jpeg,
        metadata={"target": {"track_id": sequence}, "processing_fps": 7.5},
    )


def wait_written(recorder: FrameRecorder, count: int) -> RecordingStatus:
    deadline_s = time.monotonic() + TEST_TIMEOUT_S
    while recorder.snapshot().frames_written < count and time.monotonic() < deadline_s:
        time.sleep(0.005)
    status = recorder.snapshot()
    assert status.frames_written == count, status
    return status


def configuration() -> PreviewConfig:
    return PreviewConfig(record=True, record_min_free_bytes=0)


@pytest.fixture
def events() -> Events:
    return Events()


@pytest.fixture
def jpeg() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (64, 48), color="red")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def make_spool(directory: Path, events: Events, jpeg: bytes) -> Path:
    recorder = FrameRecorder(configuration(), directory, events)
    recorder.start()
    for sequence, timestamp_s in enumerate((4.0, 4.137, 4.608, 5.019)):
        recorder.offer(frame(sequence, timestamp_s, jpeg))
    status = recorder.finish()
    assert status.frames_written == FRAME_COUNT
    assert status.state == "finished"
    return directory / "demo"


def test_spool_pairs_metadata_timestamps_and_jpeg(
    tmp_path: Path, events: Events, jpeg: bytes
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    records = [
        cast(dict[str, JsonValue], json.loads(line))
        for line in (spool / "index.jsonl").read_text().splitlines()
    ]
    assert [record["monotonic_s"] for record in records] == [4.0, 4.137, 4.608, 5.019]
    for sequence, record in enumerate(records):
        assert record["sequence"] == sequence
        assert record["captured_at"] == 1000.0 + sequence
        assert record["timebase"] == "replay_source"
        assert record["metadata"] == {"target": {"track_id": sequence}, "processing_fps": 7.5}
        assert (spool / str(record["filename"])).read_bytes() == jpeg


def test_lifecycle_is_idempotent_and_offer_after_finish_is_ignored(
    tmp_path: Path, events: Events
) -> None:
    recorder = FrameRecorder(configuration(), tmp_path, events)
    assert recorder.snapshot().limit_reason is None
    recorder.start()
    recorder.start()
    recorder.offer(frame())
    status = recorder.finish()
    assert status.state == "finished"
    assert status.limit_reason is None
    assert status.frames_written == 1
    assert status.bytes_written == sum(
        path.stat().st_size for path in (tmp_path / "demo").iterdir()
    )
    recorder.offer(frame(1, 11.0))
    recorder.start()
    assert recorder.finish(0.0) == status
    assert recorder.snapshot().state == "finished"


def test_finish_without_start_never_accepts_frames(tmp_path: Path, events: Events) -> None:
    recorder = FrameRecorder(configuration(), tmp_path, events)
    recorder.offer(frame())
    assert recorder.finish().state == "finished"
    recorder.start()
    recorder.offer(frame())
    assert recorder.snapshot().frames_written == 0
    assert not (tmp_path / "demo").exists()


@pytest.mark.parametrize("bound", ["duration", "bytes", "free_space"])
def test_recording_limits_are_nonfatal_and_preserve_sources(
    tmp_path: Path, events: Events, bound: str
) -> None:
    config = configuration()
    if bound == "duration":
        config = replace(config, record_max_duration_s=0.2)
    elif bound == "bytes":
        config = replace(config, record_max_bytes=400)
    recorder = FrameRecorder(config, tmp_path, events)
    recorder.start()
    recorder.offer(frame())
    wait_written(recorder, 1)
    with (
        patch(
            "app.demo_recording.shutil.disk_usage",
            return_value=shutil._ntuple_diskusage(100, 100, 0),
        )
        if bound == "free_space"
        else patch("app.demo_recording.shutil.disk_usage", wraps=shutil.disk_usage)
    ):
        recorder.offer(frame(1, 11.0, b"x" * 500 if bound == "bytes" else b"jpeg"))
        status = recorder.finish()
    assert status.state == "limited"
    assert status.error is None
    assert status.limit_reason == events.records[0][1]["reason"]
    assert status.limit_reason is not None
    assert status.frames_written == 1
    assert status.frames_dropped == 1
    assert events.records[0][0] == "recording_limit"
    assert (tmp_path / "demo" / "frame_000000000.jpg").read_bytes() == b"jpeg"


def test_stalled_writer_has_bounded_queue_finish_and_offer(tmp_path: Path, events: Events) -> None:
    entered = threading.Event()
    release = threading.Event()
    original_write_bytes = Path.write_bytes

    def blocked_write(path: Path, data: bytes) -> int:
        entered.set()
        assert release.wait(TEST_TIMEOUT_S)
        return original_write_bytes(path, data)

    recorder = FrameRecorder(configuration(), tmp_path, events)
    with patch.object(Path, "write_bytes", blocked_write):
        recorder.start()
        recorder.offer(frame())
        assert entered.wait(TEST_TIMEOUT_S)
        started_s = time.monotonic()
        for sequence in range(RECORDING_QUEUE_CAPACITY + FRAME_COUNT):
            recorder.offer(frame(sequence + 1, 11.0 + sequence))
        assert time.monotonic() - started_s < NONBLOCKING_BUDGET_S
        assert recorder.snapshot().frames_dropped == FRAME_COUNT
        started_s = time.monotonic()
        status = recorder.finish(0.01)
        assert time.monotonic() - started_s < NONBLOCKING_BUDGET_S
        assert status.state == "error"
        assert "timeout" in str(status.error)
        started_s = time.monotonic()
        assert recorder.finish() == status
        recorder.offer(frame(100, 200.0))
        assert time.monotonic() - started_s < NONBLOCKING_BUDGET_S
        release.set()
        wait_written(recorder, 1)
    assert events.records[0][0] == "recording_error"
    assert recorder.snapshot().state == "error"


def test_queue_drops_preserve_accepted_frame_time_gaps(tmp_path: Path, events: Events) -> None:
    entered = threading.Event()
    release = threading.Event()
    original_write_bytes = Path.write_bytes

    def blocked_first_write(path: Path, data: bytes) -> int:
        if path.name == "frame_000000000.jpg":
            entered.set()
            assert release.wait(TEST_TIMEOUT_S)
        return original_write_bytes(path, data)

    recorder = FrameRecorder(configuration(), tmp_path, events)
    with patch.object(Path, "write_bytes", blocked_first_write):
        recorder.start()
        recorder.offer(frame())
        assert entered.wait(TEST_TIMEOUT_S)
        for sequence in range(RECORDING_QUEUE_CAPACITY + FRAME_COUNT):
            recorder.offer(frame(sequence + 1, 11.0 + sequence))
        release.set()
        wait_written(recorder, RECORDING_QUEUE_CAPACITY + 1)
        recorder.offer(frame(100, 50.123))
        status = recorder.finish()
    assert status.frames_dropped == FRAME_COUNT
    assert status.last_monotonic_s == 50.123
    records = [
        cast(dict[str, JsonValue], json.loads(line))
        for line in (tmp_path / "demo" / "index.jsonl").read_text().splitlines()
    ]
    assert records[-1]["monotonic_s"] == 50.123
    assert records[-2]["monotonic_s"] == 18.0


def test_enospc_preserves_previous_frame_and_rejects_further_offers(
    tmp_path: Path, events: Events
) -> None:
    recorder = FrameRecorder(configuration(), tmp_path, events)
    recorder.start()
    recorder.offer(frame())
    wait_written(recorder, 1)
    with patch.object(Path, "write_bytes", side_effect=OSError(errno.ENOSPC, "disk full")):
        recorder.offer(frame(1, 11.0))
        status = recorder.finish()
    assert status.state == "error"
    assert "disk full" in str(status.error)
    assert status.frames_dropped == 1
    recorder.offer(frame(2, 12.0))
    assert recorder.snapshot() == status
    assert (tmp_path / "demo" / "frame_000000000.jpg").read_bytes() == b"jpeg"
    assert len((tmp_path / "demo" / "index.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("timestamp_s", [float("nan"), float("inf"), 9.0, 10.0])
def test_invalid_timestamps_fail_without_losing_prior_frames(
    tmp_path: Path, events: Events, timestamp_s: float
) -> None:
    recorder = FrameRecorder(configuration(), tmp_path, events)
    recorder.start()
    recorder.offer(frame())
    wait_written(recorder, 1)
    recorder.offer(frame(1, timestamp_s))
    status = recorder.finish()
    assert status.state == "error"
    assert status.frames_written == 1


def test_missing_ffmpeg_retains_sources(tmp_path: Path, events: Events, jpeg: bytes) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    with patch("app.demo_recording.shutil.which", return_value=None):
        result = export_recording(spool, tmp_path / "recording.mp4")
    assert not result.success
    assert result.source_retained
    assert "FFmpeg" in str(result.error)
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before


@pytest.mark.parametrize("corruption", ["empty", "invalid", "unsafe", "missing", "nonmonotonic"])
def test_invalid_index_returns_clear_failure(
    tmp_path: Path, events: Events, jpeg: bytes, corruption: str
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    index = spool / "index.jsonl"
    lines = index.read_text().splitlines()
    if corruption == "empty":
        index.write_text("")
    elif corruption == "invalid":
        index.write_text("{broken json")
    elif corruption == "unsafe":
        index.write_text('{"filename":"../secret.jpg","monotonic_s":1}\n')
    elif corruption == "missing":
        (spool / "frame_000000000.jpg").unlink()
    else:
        index.write_text(lines[0] + "\n" + lines[0] + "\n")
    result = export_recording(spool, tmp_path / "recording.mp4")
    assert not result.success
    assert result.error
    assert result.source_retained
    assert not (tmp_path / "recording.mp4").exists()


@pytest.mark.parametrize("corruption", ["overflowing_timestamp", "deep_json"])
def test_malformed_index_exceptions_return_export_failure(
    tmp_path: Path, events: Events, jpeg: bytes, corruption: str
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    index_path = spool / "index.jsonl"
    if corruption == "overflowing_timestamp":
        index_path.write_text(
            json.dumps(
                {
                    "filename": "frame_000000000.jpg",
                    "monotonic_s": 10**400,
                    "timebase": "replay_source",
                }
            )
            + "\n"
        )
    else:
        index_path.write_text("[" * 2000 + "0" + "]" * 2000 + "\n")
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    output_path = tmp_path / "annotated.mp4"
    output_path.write_bytes(b"old video")

    result = export_recording(spool, output_path)

    assert not result.success
    assert result.error
    assert result.source_retained
    assert output_path.read_bytes() == b"old video"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable"
)
def test_real_export_preserves_irregular_gaps_and_explicit_final_duration(
    tmp_path: Path, events: Events, jpeg: bytes
) -> None:
    spool = make_spool(tmp_path / "source's directory", events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    output_path = tmp_path / "annotated.mp4"
    output_path.write_bytes(b"previous output")
    result = export_recording(spool, output_path, default_frame_duration_s=0.173)
    assert result.success, result.error
    assert result.duration_s == pytest.approx(1.192, abs=0.003)
    assert result.expected_duration_s == pytest.approx(1.192)
    assert result.output_path == output_path
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))
    probe = subprocess.run(
        [
            str(shutil.which("ffprobe")),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        timeout=TEST_TIMEOUT_S,
    )
    data = cast(dict[str, list[dict[str, str]]], json.loads(probe.stdout))
    assert [float(item["best_effort_timestamp_time"]) for item in data["frames"]] == pytest.approx(
        [0.0, 0.137, 0.608, 1.019, 1.192], abs=0.002
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable"
)
@pytest.mark.parametrize("width_px,height_px", [(65, 48), (64, 49)])
def test_real_export_pads_odd_dimensions_and_retains_sources(
    tmp_path: Path, events: Events, width_px: int, height_px: int
) -> None:
    from PIL import Image

    image = Image.new("RGB", (width_px, height_px), color="red")
    encoded_image = io.BytesIO()
    image.save(encoded_image, format="JPEG")
    spool = make_spool(tmp_path, events, encoded_image.getvalue())
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    output_path = tmp_path / "odd-dimensions.mp4"

    result = export_recording(spool, output_path, default_frame_duration_s=0.173)

    assert result.success, result.error
    assert result.source_retained
    assert result.duration_s == pytest.approx(1.192, abs=0.003)
    probe = subprocess.run(
        [
            str(shutil.which("ffprobe")),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        timeout=TEST_TIMEOUT_S,
    )
    data = cast(dict[str, list[dict[str, str | int]]], json.loads(probe.stdout))
    assert data["streams"] == [
        {
            "width": width_px + width_px % 2,
            "height": height_px + height_px % 2,
            "nb_read_frames": str(FRAME_COUNT + 1),
        }
    ]
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable"
)
def test_failed_conversion_preserves_source_and_existing_output(
    tmp_path: Path, events: Events
) -> None:
    spool = make_spool(tmp_path, events, b"corrupt jpeg")
    output_path = tmp_path / "annotated.mp4"
    output_path.write_bytes(b"previous output")
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    result = export_recording(spool, output_path)
    assert not result.success
    assert result.source_retained
    assert output_path.read_bytes() == b"previous output"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))


@pytest.mark.parametrize("failure", ["timeout", "byte_limit", "nonzero"])
def test_subprocess_failure_is_bounded_and_retains_files(
    tmp_path: Path, events: Events, jpeg: bytes, failure: str
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    executable = tmp_path / "fake-ffmpeg"
    if failure == "timeout":
        script = "import time\ntime.sleep(10)\n"
    elif failure == "byte_limit":
        script = "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes(b'x' * 100000)\n"
    else:
        script = "raise SystemExit(7)\n"
    executable.write_text(f"#!{sys.executable}\n{script}")
    executable.chmod(0o700)
    output = tmp_path / "annotated.mp4"
    output.write_bytes(b"old video")
    started_s = time.monotonic()
    with patch("app.demo_recording.shutil.which", return_value=str(executable)):
        result = export_recording(
            spool,
            output,
            timeout_s=0.15 if failure == "timeout" else TEST_TIMEOUT_S,
            max_bytes=10000,
        )
    assert time.monotonic() - started_s < TEST_TIMEOUT_S + NONBLOCKING_BUDGET_S
    assert not result.success
    expected_error = {"timeout": "timed out", "byte_limit": "byte limit", "nonzero": "status 7"}
    assert expected_error[failure] in str(result.error)
    assert result.source_retained
    assert output.read_bytes() == b"old video"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))


@pytest.mark.parametrize("failure", ["duration", "no_video", "undecodable", "missing_frames"])
def test_probe_rejects_bad_duration_or_unplayable_video(
    tmp_path: Path, events: Events, jpeg: bytes, failure: str
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    expected_frames = FRAME_COUNT + 1
    stream: dict[str, JsonValue] = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 64,
        "height": 48,
        "nb_read_frames": str(expected_frames),
    }
    if failure == "no_video":
        stream["codec_type"] = "audio"
    elif failure == "undecodable":
        stream["nb_read_frames"] = "0"
    elif failure == "missing_frames":
        stream["nb_read_frames"] = "2"
    probe: dict[str, JsonValue] = {
        "streams": [stream],
        "format": {"duration": "100.0" if failure == "duration" else "1.119"},
    }
    output = tmp_path / "annotated.mp4"
    output.write_bytes(b"old video")

    def command_result(
        arguments: list[str],
        deadline_s: float,
        bounded_path: Path,
        max_bytes: int,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> bytes:
        bounded_path.write_bytes(b"invalid mp4")
        return json.dumps(probe).encode()

    with (
        patch("app.demo_recording.shutil.which", return_value="fake-tool"),
        patch("app.demo_recording._run_command", side_effect=command_result),
    ):
        result = export_recording(spool, output)
    assert not result.success
    assert result.error
    assert result.source_retained
    assert output.read_bytes() == b"old video"
    assert len(list(spool.glob("*.jpg"))) == FRAME_COUNT


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable"
)
def test_single_frame_export_has_explicit_final_duration(
    tmp_path: Path, events: Events, jpeg: bytes
) -> None:
    recorder = FrameRecorder(configuration(), tmp_path, events)
    recorder.start()
    recorder.offer(frame(jpeg=jpeg))
    assert recorder.finish().frames_written == 1
    result = export_recording(
        tmp_path / "demo", tmp_path / "single.mp4", default_frame_duration_s=0.267
    )
    assert result.success, result.error
    assert result.duration_s == pytest.approx(0.267, abs=0.003)
    assert result.expected_duration_s == pytest.approx(0.267)


def test_export_cannot_replace_source_index(tmp_path: Path, events: Events, jpeg: bytes) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    index = spool / "index.jsonl"
    original = index.read_bytes()
    result = export_recording(spool, index)
    assert not result.success
    assert index.read_bytes() == original


def test_existing_spool_is_preserved(tmp_path: Path, events: Events, jpeg: bytes) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    recorder = FrameRecorder(configuration(), tmp_path, events)
    recorder.start()
    recorder.offer(frame())
    assert recorder.finish().state == "error"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before


def test_export_cancellation_before_launch_spawns_no_child(
    tmp_path: Path, events: Events, jpeg: bytes
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    output_path = tmp_path / "annotated.mp4"
    output_path.write_bytes(b"old video")
    cancellation = threading.Event()
    cancellation.set()

    with patch("app.demo_recording.subprocess.Popen") as launch:
        result = export_recording(spool, output_path, cancel_requested=cancellation.is_set)

    launch.assert_not_called()
    assert not result.success
    assert result.error == "recording export cancelled"
    assert result.source_retained
    assert output_path.read_bytes() == b"old video"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))


@pytest.mark.parametrize("interruption_stage", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("interruption", ["cancellation", "timeout"])
def test_export_interruption_kills_and_reaps_real_child(
    tmp_path: Path, events: Events, jpeg: bytes, interruption_stage: str, interruption: str
) -> None:
    spool = make_spool(tmp_path, events, jpeg)
    before = {path.name: path.read_bytes() for path in spool.iterdir()}
    marker_path = tmp_path / "running-child"
    for tool in ("ffmpeg", "ffprobe"):
        executable = tmp_path / tool
        script = "import os, pathlib, sys, time\n"
        if tool == interruption_stage:
            script += (
                f"pathlib.Path({str(marker_path)!r}).write_text(str(os.getpid()))\ntime.sleep(10)\n"
            )
        else:
            script += "pathlib.Path(sys.argv[-1]).write_bytes(b'video')\n"
        executable.write_text(f"#!{sys.executable}\n{script}")
        executable.chmod(0o700)
    output_path = tmp_path / "annotated.mp4"
    output_path.write_bytes(b"old video")
    started_s = time.monotonic()
    with patch("app.demo_recording.shutil.which", side_effect=lambda name: str(tmp_path / name)):
        result = export_recording(
            spool,
            output_path,
            timeout_s=TEST_TIMEOUT_S,
            cancel_requested=marker_path.exists if interruption == "cancellation" else None,
        )

    assert time.monotonic() - started_s < TEST_TIMEOUT_S + NONBLOCKING_BUDGET_S
    assert marker_path.exists()
    assert not result.success
    expected_error = (
        "recording export cancelled"
        if interruption == "cancellation"
        else "recording export timed out"
    )
    assert result.error == expected_error
    child_process_id = int(marker_path.read_text())
    with pytest.raises(ChildProcessError):
        os.waitpid(child_process_id, os.WNOHANG)
    assert result.source_retained
    assert output_path.read_bytes() == b"old video"
    assert {path.name: path.read_bytes() for path in spool.iterdir()} == before
    assert not list(tmp_path.glob(".demo-export-*"))
