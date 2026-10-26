from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import BinaryIO, ParamSpec, TypedDict, Unpack, cast
from unittest.mock import patch

import cv2
import numpy
import pytest

import app.preview as preview_module
from app.config import AppConfig, PreviewConfig
from app.demo_frames import EncodedFrame
from app.demo_recording import FrameRecorder, RecordingStatus
from app.json_data import JsonValue
from app.models import Detection, Frame
from app.output import annotate_frame
from app.pan_tilt_control import ControlAction, ControllerStatus
from app.pipeline import RenderJob
from app.preview import PreviewService

TEST_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.01
FRAME_WIDTH_PX = 640
FRAME_HEIGHT_PX = 480
UNCAPPED_FPS = 1000.0
P = ParamSpec("P")
CONTROL_HEADERS = {"Content-Type": "application/json", "X-Columba-Control": "1"}
JPEG_SIGNATURE = b"\xff\xd8"
PUBLISH_BUDGET_S = 0.5
SHORT_SHUTDOWN_S = 0.2


class Events:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, JsonValue]]] = []

    def event(self, event: str, **fields: JsonValue) -> None:
        self.records.append((event, fields))

    def names(self) -> list[str]:
        return [name for name, _ in self.records]


class RecorderDouble(FrameRecorder):
    def __init__(self, directory: Path, events: Events) -> None:
        super().__init__(PreviewConfig(), directory, events)
        self.frames: list[EncodedFrame] = []
        self.offer_gate: threading.Event | None = None
        self.offer_error: Exception | None = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def offer(self, frame: EncodedFrame) -> None:
        if self.offer_gate is not None:
            self.offer_gate.wait(TEST_TIMEOUT_S)
        if self.offer_error is not None:
            raise self.offer_error
        self.frames.append(frame)

    def snapshot(self) -> RecordingStatus:
        return RecordingStatus(state="recording", frames_written=len(self.frames))


class PreviewOverrides(TypedDict, total=False):
    enabled: bool
    host: str
    port: int
    max_fps: float
    width_px: int
    height_px: int
    jpeg_quality: int
    record: bool


class JobOverrides(TypedDict, total=False):
    detections: list[Detection]
    processing_fps: float
    capture_fps: float | None
    controller: ControllerStatus | None
    delivered_monotonic_s: float | None
    presentation_time_s: float | None


def preview_config(**overrides: Unpack[PreviewOverrides]) -> AppConfig:
    defaults = replace(PreviewConfig(), enabled=True, port=0, max_fps=UNCAPPED_FPS)
    return replace(AppConfig(), preview=replace(defaults, **overrides))


def make_frame(width_px: int = FRAME_WIDTH_PX, height_px: int = FRAME_HEIGHT_PX) -> Frame:
    return numpy.full((height_px, width_px, 3), 128, dtype=numpy.uint8)


def make_job(
    sequence: int = 1, frame: Frame | None = None, **overrides: Unpack[JobOverrides]
) -> RenderJob:
    job = RenderJob(
        frame=make_frame() if frame is None else frame,
        detections=[Detection("pigeon", 0.9, 10.0, 10.0, 100.0, 100.0, track_id=3)],
        processing_fps=12.5,
        capture_fps=15.0,
        sequence=sequence,
        captured_at=1_700_000_000.0 + sequence,
        delivered_monotonic_s=500.0 + sequence,
    )
    return replace(job, **overrides)


def gated(function: Callable[P, Frame], gate: threading.Event) -> Callable[P, Frame]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Frame:
        gate.wait(TEST_TIMEOUT_S)
        return function(*args, **kwargs)

    return wrapper


def wait_until(predicate: Callable[[], bool], timeout_s: float = TEST_TIMEOUT_S) -> None:
    deadline_s = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline_s:
            raise AssertionError("condition not reached in time")
        time.sleep(POLL_INTERVAL_S)


def frame_age_ms(service: PreviewService) -> float:
    preview = service.state()["preview"]
    assert isinstance(preview, dict)
    age_ms = preview["frame_age_ms"]
    assert isinstance(age_ms, float)
    return age_ms


def decode_jpeg(jpeg: bytes) -> Frame:
    image = cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return numpy.asarray(image, dtype=numpy.uint8)


def encoded_frames(service: PreviewService) -> int:
    performance = service.state()["performance"]
    assert isinstance(performance, dict)
    frames = performance["encoded_frames"]
    assert isinstance(frames, int)
    return frames


def recorded_count_is(recorder: RecorderDouble, expected: int) -> bool:
    return len(recorder.frames) == expected


def encoded_count_is(service: PreviewService, expected: int) -> bool:
    return encoded_frames(service) == expected


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.events = Events()
        self.services: list[PreviewService] = []

    def start(
        self,
        config: AppConfig | None = None,
        recorder: FrameRecorder | None = None,
        control: Callable[[ControlAction], None] | None = None,
        controller_state: Callable[[], ControllerStatus] | None = None,
    ) -> PreviewService:
        service = PreviewService(
            preview_config() if config is None else config,
            self.events,
            recorder,
            control,
            controller_state,
        )
        self.services.append(service)
        service.start()
        return service

    def request(
        self,
        service: PreviewService,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes, http.client.HTTPMessage]:
        address = service.address
        assert address is not None
        connection = http.client.HTTPConnection(*address, timeout=TEST_TIMEOUT_S)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), response.msg
        finally:
            connection.close()

    def post(
        self, service: PreviewService, payload: JsonValue, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, JsonValue]]:
        status, body, _ = self.request(
            service,
            "POST",
            "/api/control",
            json.dumps(payload).encode(),
            CONTROL_HEADERS if headers is None else headers,
        )
        decoded = json.loads(body)
        assert isinstance(decoded, dict)
        return status, decoded

    def json_state(self, service: PreviewService) -> dict[str, JsonValue]:
        status, body, _ = self.request(service, "GET", "/api/state")
        assert status == 200
        decoded = json.loads(body)
        assert isinstance(decoded, dict)
        return decoded


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(tmp_path)
    yield harness
    for service in harness.services:
        service.close()


def open_stream(service: PreviewService, receive_buffer_bytes: int | None = None) -> socket.socket:
    stream = socket.socket()
    if receive_buffer_bytes is not None:
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes)
    stream.settimeout(TEST_TIMEOUT_S)
    address = service.address
    assert address is not None
    stream.connect(address)
    hostname, port = address
    authority = f"[{hostname}]:{port}" if ":" in hostname else f"{hostname}:{port}"
    stream.sendall(f"GET /stream.mjpg HTTP/1.0\r\nHost: {authority}\r\n\r\n".encode())
    return stream


def read_stream_part(reader: BinaryIO) -> bytes:
    length = 0
    while True:
        line = reader.readline()
        if not line:
            raise AssertionError("stream closed")
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
        if line == b"\r\n" and length:
            return reader.read(length)


class TestFrameEncoding:
    def test_letterboxes_to_the_configured_size_with_black_bars(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(preview_config(width_px=320, height_px=240), recorder)
        service.publish(make_job(frame=make_frame(width_px=300, height_px=600)))
        wait_until(lambda: len(recorder.frames) == 1)
        image = decode_jpeg(recorder.frames[0].jpeg)
        assert image.shape == (240, 320, 3)
        assert int(image[120, 2].max()) < 10
        assert int(image[120, 160].min()) > 100

    def test_one_jpeg_is_shared_by_http_stream_and_recorder(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        with patch.object(preview_module, "encode_jpeg", wraps=preview_module.encode_jpeg) as spy:
            service = harness.start(recorder=recorder)
            stream = open_stream(service)
            reader = stream.makefile("rb")
            while reader.readline().strip() != b"":
                pass
            service.publish(make_job())
            part = read_stream_part(reader)
            wait_until(lambda: len(recorder.frames) == 1)
            _, snapshot, _ = harness.request(service, "GET", "/snapshot.jpg")
            stream.close()
        assert part.startswith(JPEG_SIGNATURE)
        assert part == snapshot == recorder.frames[0].jpeg
        assert spy.call_count == 1

    def test_metadata_is_read_only_and_carries_the_detections(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(recorder=recorder)
        service.publish(make_job(sequence=9))
        wait_until(lambda: len(recorder.frames) == 1)
        frame = recorder.frames[0]
        assert frame.sequence == 9
        assert frame.captured_at == 1_700_000_009.0
        assert frame.metadata["processing_fps"] == 12.5
        detections = frame.metadata["detections"]
        assert isinstance(detections, list) and len(detections) == 1
        mutable = cast(MutableMapping[str, JsonValue], frame.metadata)
        with pytest.raises(TypeError):
            mutable["new"] = 1

    def test_timebase_follows_the_available_clock(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(preview_config(max_fps=UNCAPPED_FPS), recorder)
        jobs = (
            make_job(1, presentation_time_s=42.0),
            make_job(2),
            make_job(3, delivered_monotonic_s=None),
        )
        for expected_count, job in enumerate(jobs, start=1):
            service.publish(job)
            wait_until(partial(recorded_count_is, recorder, expected_count))
        replay, delivery, processing = recorder.frames
        assert (replay.timebase, replay.monotonic_s) == ("replay_source", 42.0)
        assert (delivery.timebase, delivery.monotonic_s) == ("delivery_monotonic", 502.0)
        assert processing.timebase == "processing_monotonic"
        assert abs(processing.monotonic_s - time.monotonic()) < TEST_TIMEOUT_S


class TestPacing:
    def test_encoding_is_capped_and_latest_wins(self, harness: Harness) -> None:
        maximum_fps = 5.0
        service = harness.start(preview_config(max_fps=maximum_fps))
        duration_s = 0.6
        started_s = time.monotonic()
        sequence = 0
        while time.monotonic() - started_s < duration_s:
            sequence += 1
            service.publish(make_job(sequence))
            time.sleep(0.005)
        encoded = encoded_frames(service)
        assert 1 <= encoded <= int(duration_s * maximum_fps) + 1
        performance = service.state()["performance"]
        assert isinstance(performance, dict)
        overwritten = performance["overwritten_frames"]
        assert isinstance(overwritten, int) and overwritten > sequence // 2

    def test_close_drains_the_latest_accepted_frame_without_waiting_for_pacing(
        self, harness: Harness
    ) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(preview_config(max_fps=0.5), recorder)
        service.publish(make_job(1))
        wait_until(lambda: len(recorder.frames) == 1)
        service.publish(make_job(2))
        service.publish(make_job(3))
        started_s = time.monotonic()
        service.close()
        assert time.monotonic() - started_s < 1.0
        assert [frame.sequence for frame in recorder.frames] == [1, 3]
        assert service.failure is None
        service.close()

    def test_publish_after_close_is_ignored(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(recorder=recorder)
        service.close()
        service.publish(make_job())
        time.sleep(0.05)
        assert recorder.frames == []


class TestBoundedWork:
    def test_publish_never_waits_for_a_blocked_encoder(self, harness: Harness) -> None:
        gate = threading.Event()
        with patch.object(preview_module, "annotate_frame", gated(annotate_frame, gate)):
            service = harness.start()
            started_s = time.monotonic()
            for sequence in range(1, 51):
                service.publish(make_job(sequence))
            assert time.monotonic() - started_s < PUBLISH_BUDGET_S
            performance = service.state()["performance"]
            assert isinstance(performance, dict)
            assert performance["overwritten_frames"] == 48
            gate.set()
            wait_until(lambda: encoded_frames(service) == 2)

    def test_blocked_recorder_offer_does_not_stall_publish_or_http(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        recorder.offer_gate = threading.Event()
        service = harness.start(recorder=recorder)
        started_s = time.monotonic()
        for sequence in range(1, 31):
            service.publish(make_job(sequence))
        assert time.monotonic() - started_s < PUBLISH_BUDGET_S
        wait_until(lambda: encoded_frames(service) == 1)
        status, _, _ = harness.request(service, "GET", "/snapshot.jpg")
        assert status == 200
        with patch.object(preview_module, "ENCODER_SHUTDOWN_TIMEOUT_S", SHORT_SHUTDOWN_S):
            started_s = time.monotonic()
            service.close()
            assert time.monotonic() - started_s < 2.0
        recorder.offer_gate.set()
        assert service.failure is not None and "did not stop" in service.failure
        assert "preview_failure" in harness.events.names()

    def test_recorder_failure_is_stored_and_the_preview_continues(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        recorder.offer_error = OSError("disk full")
        service = harness.start(recorder=recorder)
        service.publish(make_job(1))
        wait_until(lambda: service.failure is not None)
        service.publish(make_job(2))
        wait_until(lambda: encoded_frames(service) == 2)
        recorder.offer_error = None
        service.publish(make_job(3))
        wait_until(lambda: encoded_frames(service) == 3)
        assert recorder.frames == []
        assert service.failure is not None and "disk full" in service.failure
        preview = harness.json_state(service)["preview"]
        assert isinstance(preview, dict)
        assert preview["running"] is True
        assert preview["failure"] == service.failure

    def test_encoder_failure_is_stored_and_never_raised_to_publish(self, harness: Harness) -> None:
        with patch.object(preview_module, "annotate_frame", side_effect=RuntimeError("boom")):
            service = harness.start()
            service.publish(make_job())
            wait_until(lambda: service.failure is not None)
        service.publish(make_job(2))
        assert service.failure is not None and "boom" in service.failure
        state = harness.json_state(service)
        preview = state["preview"]
        assert isinstance(preview, dict)
        assert preview["running"] is False
        assert preview["failure"] == service.failure
        assert "preview_failure" in harness.events.names()

    def test_recorder_start_failure_is_nonfatal(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        with patch.object(recorder, "start", side_effect=OSError("no space")):
            service = harness.start(recorder=recorder)
        service.publish(make_job())
        wait_until(lambda: encoded_frames(service) == 1)
        assert service.failure is not None and "no space" in service.failure
        assert recorder.frames == []

    def test_bind_failure_raises_and_is_stored(self, harness: Harness) -> None:
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            service = PreviewService(preview_config(host="127.0.0.1", port=port), harness.events)
            with pytest.raises(OSError):
                service.start()
        assert service.failure is not None and "cannot bind" in service.failure
        service.close()


class TestRecordingOnly:
    def test_no_http_server_is_bound_but_frames_are_encoded_and_recorded(
        self, harness: Harness
    ) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        config = preview_config(enabled=False, record=True)
        service = harness.start(config, recorder)
        assert recorder.started
        assert service.address is None
        service.publish(make_job())
        wait_until(lambda: len(recorder.frames) == 1)
        assert recorder.frames[0].jpeg.startswith(JPEG_SIGNATURE)

    def test_disabled_without_recording_does_nothing(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(preview_config(enabled=False, record=False), recorder)
        service.publish(make_job())
        time.sleep(0.05)
        assert service.address is None
        assert recorder.frames == []
        assert not recorder.started
        preview = service.state()["preview"]
        assert isinstance(preview, dict) and preview["running"] is False

    def test_real_recorder_spools_frames_and_reports_in_state(self, harness: Harness) -> None:
        config = preview_config(enabled=False, record=True)
        recorder = FrameRecorder(config.preview, harness.tmp_path, harness.events)
        service = harness.start(config, recorder)
        for sequence in range(1, 4):
            service.publish(make_job(sequence))
            wait_until(partial(encoded_count_is, service, sequence))
        service.close()
        status = recorder.finish()
        assert status.frames_written == 3
        assert status.error is None
        assert len(list((harness.tmp_path / "demo").glob("*.jpg"))) == 3


class TestState:
    def test_state_is_served_before_the_first_frame(self, harness: Harness) -> None:
        service = harness.start()
        state = harness.json_state(service)
        assert state["target"] is None
        assert state["controller"] is None
        assert state["preview"] == {"running": True, "frame_age_ms": None, "failure": None}

    def test_state_reflects_the_published_job(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(recorder=recorder, control=lambda action: None)
        service.publish(make_job(controller=ControllerStatus(state="tracking", mode="simulate")))
        wait_until(lambda: encoded_frames(service) == 1)
        state = harness.json_state(service)
        performance = state["performance"]
        assert isinstance(performance, dict)
        assert performance["processing_fps"] == 12.5
        assert performance["capture_fps"] == 15.0
        assert isinstance(performance["last_encode_ms"], float)
        controller = state["controller"]
        assert isinstance(controller, dict) and controller["state"] == "tracking"
        recording = state["recording"]
        assert isinstance(recording, dict) and recording["state"] == "recording"
        assert state["controls_enabled"] is True

    def test_live_controller_callback_is_preferred_and_failures_fall_back(
        self, harness: Harness
    ) -> None:
        live = ControllerStatus(state="paused")
        failing = False

        def controller_state() -> ControllerStatus:
            if failing:
                raise RuntimeError("controller gone")
            return live

        service = harness.start(controller_state=controller_state)
        service.publish(make_job(controller=ControllerStatus(state="tracking")))
        wait_until(lambda: encoded_frames(service) == 1)
        controller = harness.json_state(service)["controller"]
        assert isinstance(controller, dict) and controller["state"] == "paused"
        failing = True
        controller = harness.json_state(service)["controller"]
        assert isinstance(controller, dict) and controller["state"] == "tracking"

    def test_frame_age_is_wall_time_and_ignores_the_replay_clock(self, harness: Harness) -> None:
        service = harness.start()
        service.publish(
            make_job(delivered_monotonic_s=time.monotonic(), presentation_time_s=1_000_000.0)
        )
        wait_until(lambda: encoded_frames(service) == 1)
        first_age = frame_age_ms(service)
        assert 0 <= first_age < TEST_TIMEOUT_S * 1000
        time.sleep(0.25)
        assert frame_age_ms(service) >= first_age + 200

    def test_frame_age_counts_from_delivery_not_from_encoding_finish(
        self, harness: Harness
    ) -> None:
        service = harness.start()
        service.publish(make_job(delivered_monotonic_s=time.monotonic() - 2.0))
        wait_until(lambda: encoded_frames(service) == 1)
        assert frame_age_ms(service) >= 2000

    def test_encoded_metadata_uses_the_job_controller_not_the_live_callback(
        self, harness: Harness
    ) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(
            recorder=recorder, controller_state=lambda: ControllerStatus(state="paused")
        )
        service.publish(make_job(controller=ControllerStatus(state="tracking")))
        wait_until(lambda: len(recorder.frames) == 1)
        controller = recorder.frames[0].metadata["controller"]
        assert isinstance(controller, dict) and controller["state"] == "tracking"
        live = harness.json_state(service)["controller"]
        assert isinstance(live, dict) and live["state"] == "paused"

    def test_non_finite_rates_are_reported_as_null(self, harness: Harness) -> None:
        service = harness.start()
        service.publish(make_job(processing_fps=float("nan"), capture_fps=float("inf")))
        wait_until(lambda: encoded_frames(service) == 1)
        performance = harness.json_state(service)["performance"]
        assert isinstance(performance, dict)
        assert performance["processing_fps"] is None
        assert performance["capture_fps"] is None


class TestHttp:
    def test_serves_the_page_and_rejects_unknown_paths_and_wrong_methods(
        self, harness: Harness
    ) -> None:
        service = harness.start(control=lambda action: None)
        status, _, headers = harness.request(service, "GET", "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert harness.request(service, "GET", "/missing")[0] == 404
        assert harness.request(service, "GET", "/api/control")[0] == 404
        assert harness.request(service, "POST", "/api/state")[0] == 404

    def test_snapshot_is_unavailable_before_the_first_frame(self, harness: Harness) -> None:
        service = harness.start()
        assert harness.request(service, "GET", "/snapshot.jpg")[0] == 503

    def test_slow_stream_client_neither_slows_publishing_nor_other_clients(
        self, harness: Harness
    ) -> None:
        service = harness.start(preview_config(width_px=1280, height_px=720, jpeg_quality=100))
        slow = open_stream(service, receive_buffer_bytes=1024)
        noise = numpy.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=numpy.uint8)
        publish_durations_s: list[float] = []
        for sequence in range(1, 31):
            started_s = time.monotonic()
            service.publish(make_job(sequence, frame=noise))
            publish_durations_s.append(time.monotonic() - started_s)
            time.sleep(0.02)
        assert max(publish_durations_s) < PUBLISH_BUDGET_S
        wait_until(lambda: encoded_frames(service) >= 5)
        started_s = time.monotonic()
        status, body, _ = harness.request(service, "GET", "/snapshot.jpg")
        assert status == 200 and body.startswith(JPEG_SIGNATURE)
        assert time.monotonic() - started_s < 1.0
        started_s = time.monotonic()
        service.close()
        assert time.monotonic() - started_s < 2.0
        slow.close()

    def test_late_stream_reader_receives_the_newest_frame(self, harness: Harness) -> None:
        recorder = RecorderDouble(harness.tmp_path, harness.events)
        service = harness.start(recorder=recorder)
        service.publish(make_job(1))
        wait_until(lambda: len(recorder.frames) == 1)
        stream = open_stream(service)
        reader = stream.makefile("rb")
        part = read_stream_part(reader)
        assert part == recorder.frames[0].jpeg
        stream.close()

    def test_client_concurrency_is_bounded(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(preview_module, "MAX_HTTP_CLIENTS", 2)
        service = harness.start()
        streams = [open_stream(service) for _ in range(2)]
        time.sleep(0.2)
        assert harness.request(service, "GET", "/api/state")[0] == 503
        streams[0].close()
        wait_until(lambda: harness.request(service, "GET", "/api/state")[0] == 200)
        for stream in streams:
            stream.close()

    def test_closed_stream_clients_release_their_slot_while_idle(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(preview_module, "MAX_HTTP_CLIENTS", 1)
        monkeypatch.setattr(preview_module, "STREAM_WAIT_S", 0.05)
        service = harness.start()
        stream = open_stream(service)
        time.sleep(0.1)
        stream.close()
        wait_until(lambda: harness.request(service, "GET", "/api/state")[0] == 200)

    def test_idle_and_stalled_requests_time_out(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(preview_module.PreviewRequestHandler, "timeout", 0.3)
        service = harness.start()
        address = service.address
        assert address is not None
        with socket.create_connection(address, timeout=TEST_TIMEOUT_S) as idle:
            started_s = time.monotonic()
            assert idle.recv(1) == b""
            assert time.monotonic() - started_s < 2.0
        with socket.create_connection(address, timeout=TEST_TIMEOUT_S) as stalled:
            stalled.sendall(
                f"POST /api/control HTTP/1.0\r\nHost: {address[0]}:{address[1]}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                b'X-Columba-Control: 1\r\nContent-Length: 20\r\n\r\n{"ac'
            )
            response = stalled.recv(4096)
        assert response.startswith(b"HTTP/1.0 408")

    def test_close_stops_serving_and_ends_streams(self, harness: Harness) -> None:
        service = harness.start()
        address = service.address
        assert address is not None
        stream = open_stream(service)
        time.sleep(0.1)
        service.close()
        assert stream.recv(65536).startswith(b"HTTP/1.0 200")
        stream.settimeout(TEST_TIMEOUT_S)
        assert stream.recv(65536) == b""
        stream.close()
        with pytest.raises(OSError):
            socket.create_connection(address, timeout=1.0)


class TestControl:
    def test_rebinding_hostname_is_rejected_even_with_matching_origin(
        self, harness: Harness
    ) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        assert service.address is not None
        authority = f"attacker.example:{service.address[1]}"
        headers = {**CONTROL_HEADERS, "Host": authority, "Origin": f"http://{authority}"}
        status, body = harness.post(service, {"action": "home"}, headers)
        assert status == 403 and "error" in body
        assert not actions

    def test_rebinding_hostname_cannot_read_state_or_stream(self, harness: Harness) -> None:
        service = harness.start()
        for path in ("/", "/api/state", "/stream.mjpg", "/snapshot.jpg"):
            status, _, _ = harness.request(
                service, "GET", path, headers={"Host": "attacker.example"}
            )
            assert status == 403

    def test_valid_actions_reach_the_callback_once(self, harness: Harness) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        for action in ("pause", "resume", "home"):
            assert harness.post(service, {"action": action}) == (200, {"ok": True})
        assert actions == ["pause", "resume", "home"]

    def test_same_origin_is_accepted_and_foreign_origin_rejected(self, harness: Harness) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        assert service.address is not None
        host = f"127.0.0.1:{service.address[1]}"
        same = {**CONTROL_HEADERS, "Origin": f"http://{host}"}
        assert harness.post(service, {"action": "pause"}, same)[0] == 200
        for origin in ("http://evil.example", "null"):
            foreign = {**CONTROL_HEADERS, "Origin": origin}
            status, body = harness.post(service, {"action": "home"}, foreign)
            assert status == 403 and "error" in body
        assert actions == ["pause"]

    def test_missing_custom_header_and_wrong_content_type_are_rejected(
        self, harness: Harness
    ) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        no_header = {"Content-Type": "application/json"}
        assert harness.post(service, {"action": "home"}, no_header)[0] == 403
        wrong_header = {**CONTROL_HEADERS, "X-Columba-Control": "0"}
        assert harness.post(service, {"action": "home"}, wrong_header)[0] == 403
        form = {"Content-Type": "text/plain", "X-Columba-Control": "1"}
        assert harness.post(service, {"action": "home"}, form)[0] == 415
        charset = {**CONTROL_HEADERS, "Content-Type": "application/json; charset=utf-8"}
        assert harness.post(service, {"action": "pause"}, charset)[0] == 200
        assert actions == ["pause"]

    def test_invalid_bodies_are_rejected(self, harness: Harness) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        payloads: list[JsonValue] = [{"action": "fly"}, {"action": 1}, {}, ["pause"], "pause", None]
        for payload in payloads:
            status, body = harness.post(service, payload)
            assert status == 400 and "error" in body
        status, _, _ = harness.request(
            service, "POST", "/api/control", b"{not json", CONTROL_HEADERS
        )
        assert status == 400
        assert actions == []

    def test_oversized_and_unsized_bodies_are_rejected_before_reading(
        self, harness: Harness
    ) -> None:
        actions: list[ControlAction] = []
        service = harness.start(control=actions.append)
        address = service.address
        assert address is not None
        padding = b"x" * 5000
        status, _, _ = harness.request(
            service,
            "POST",
            "/api/control",
            b'{"action":"pause","pad":"' + padding + b'"}',
            CONTROL_HEADERS,
        )
        assert status == 413
        with socket.create_connection(address, timeout=TEST_TIMEOUT_S) as raw:
            raw.sendall(
                f"POST /api/control HTTP/1.0\r\nHost: {address[0]}:{address[1]}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                b"X-Columba-Control: 1\r\n\r\n"
            )
            assert raw.recv(4096).startswith(b"HTTP/1.0 411")
        assert actions == []

    def test_controls_are_unavailable_without_a_callback(self, harness: Harness) -> None:
        service = harness.start()
        status, body = harness.post(service, {"action": "pause"})
        assert status == 409 and "error" in body

    def test_a_failing_callback_is_reported_without_killing_the_server(
        self, harness: Harness
    ) -> None:
        def failing(action: ControlAction) -> None:
            raise RuntimeError("controller closed")

        service = harness.start(control=failing)
        status, body = harness.post(service, {"action": "pause"})
        assert status == 500 and "controller closed" in str(body["error"])
        assert harness.request(service, "GET", "/api/state")[0] == 200
