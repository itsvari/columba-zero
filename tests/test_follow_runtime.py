from __future__ import annotations

import errno
import json
import signal
import socket
import tempfile
import threading
import time
import unittest
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy
import yaml

from app.capture import EndOfStream
from app.config import AppConfig, ConfirmationConfig, DisplayConfig, FollowConfig, ModelConfig
from app.demo_recording import ExportResult
from app.json_data import JsonValue
from app.main import CommandOptions, FrameStream, load_configuration, replay_frame_rate, run
from app.models import Detection, FramePacket
from app.output import RunOutput
from app.pantilt import PCA9685, Calibration
from app.pipeline import FrameProcessor, OutputQueue

FRAME_WIDTH_PX = 320
FRAME_HEIGHT_PX = 180
REPLAY_FPS = 20.0
FRAME_INTERVAL_S = 1 / REPLAY_FPS
ACQUISITION_CONFIDENCE = 0.6
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.2
SENSOR_DELIVERY_MS = 200.0
PICKUP_AGE_S = 0.05
WORKER_OPPORTUNITY_S = 0.02
MILLISECONDS_PER_SECOND = 1000.0
MEDIA_FAULT_FRAME_COUNT = 30
MEDIA_FAULT_FRAME_INTERVAL_S = 0.002


def frame_packet(sequence: int, camera: str | None = None) -> FramePacket:
    return FramePacket(
        sequence,
        1000.0 + sequence * FRAME_INTERVAL_S,
        numpy.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=numpy.uint8),
        delivered_monotonic_s=time.monotonic(),
        camera=camera,
    )


def pigeon(confidence: float = HIGH_CONFIDENCE, left_px: float = 50) -> Detection:
    return Detection("pigeon", confidence, left_px, 50, left_px + 40, 90)


class ScriptedSource:
    dropped_frames = 0

    def __init__(
        self,
        packets: Sequence[FramePacket],
        on_start: Callable[[], None] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.packets = deque(packets)
        self.fps = REPLAY_FPS
        self.on_start = on_start
        self.delay_s = delay_s
        self.closed = False

    def start(self) -> None:
        if self.on_start is not None:
            self.on_start()

    def get(self, timeout_s: float) -> FramePacket:
        if self.delay_s:
            time.sleep(self.delay_s)
        if not self.packets:
            raise EndOfStream()
        return self.packets.popleft()

    def close(self) -> None:
        self.closed = True


class ScriptedDetector:
    names = {0: "pigeon"}

    def __init__(self, observations: Mapping[int, tuple[Detection, ...]]) -> None:
        self.observations = observations

    def warmup(self, width: int, height: int) -> None:
        pass

    def detect(self, packet: FramePacket) -> list[Detection]:
        return list(self.observations.get(packet.sequence, ()))


class FollowRuntimeTests(unittest.TestCase):
    def configuration(self, following: bool = True) -> AppConfig:
        return AppConfig(
            display=DisplayConfig(False),
            model=ModelConfig(confidence=ACQUISITION_CONFIDENCE),
            confirmation=ConfirmationConfig(enabled=True),
            follow=FollowConfig(enabled=following),
        )

    def run_script(
        self,
        configuration: AppConfig,
        source: ScriptedSource,
        observations: Mapping[int, tuple[Detection, ...]],
        replay: bool = True,
    ) -> tuple[int, list[dict[str, JsonValue]], ModelConfig]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = replace(
                configuration, output=replace(configuration.output, directory=str(root / "runs"))
            )
            configuration_path = root / "config.yaml"
            configuration_path.write_text(yaml.safe_dump(asdict(configuration)), encoding="utf-8")
            with (
                patch("app.main.create_source", return_value=source),
                patch(
                    "app.main.create_detector",
                    return_value=(ScriptedDetector(observations), ("pigeon",)),
                ) as detector_factory,
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                status = run(
                    CommandOptions(
                        config=str(configuration_path), video="scripted.avi" if replay else None
                    )
                )
            logs = list((root / "runs").glob("*/detections.jsonl"))
            self.assertEqual(len(logs), 1)
            records = [
                cast(dict[str, JsonValue], json.loads(line))
                for line in logs[0].read_text(encoding="utf-8").splitlines()
            ]
            detector_factory.assert_called_once()
            return status, records, cast(ModelConfig, detector_factory.call_args.args[0])

    def test_replay_uses_shared_identity_and_never_opens_hardware_or_calibration(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration,
            pan_tilt=replace(configuration.pan_tilt, mode="hardware", calibration="missing.yaml"),
        )
        source = ScriptedSource([frame_packet(sequence) for sequence in range(3)])
        with (
            patch.object(PCA9685, "__init__", side_effect=AssertionError("physical driver opened")),
            patch.object(Calibration, "load", side_effect=AssertionError("profile loaded")),
        ):
            status, records, detector_model = self.run_script(
                configuration,
                source,
                {
                    0: (pigeon(),),
                    1: (pigeon(),),
                    2: (pigeon(LOW_CONFIDENCE), pigeon(LOW_CONFIDENCE, 200)),
                },
            )
        self.assertEqual(status, 0)
        self.assertTrue(source.closed)
        self.assertEqual(detector_model.confidence, configuration.follow.maintenance_confidence)
        frames = [record for record in records if record["event"] == "frame"]
        triggers = [record for record in records if record["event"] == "trigger"]
        acquired = [record for record in records if record["event"] == "target_acquired"]
        self.assertEqual(len(triggers), 1)
        self.assertEqual(acquired[0]["track_id"], triggers[0]["track_id"])
        target = cast(dict[str, JsonValue], frames[-1]["target"])
        selected = cast(dict[str, JsonValue], target["track"])
        self.assertEqual(selected["track_id"], triggers[0]["track_id"])
        detections = cast(list[dict[str, JsonValue]], frames[-1]["detections"])
        self.assertEqual(detections[0]["track_id"], triggers[0]["track_id"])
        self.assertIsNone(detections[1]["track_id"])
        self.assertEqual(frames[-1]["follow_timebase"], "replay")
        controller = cast(dict[str, JsonValue], frames[-1]["controller"])
        self.assertEqual(controller["mode"], "simulate")
        self.assertIsNone(records[-1]["error"])

    def test_detection_only_keeps_acquisition_threshold_and_constructs_no_session(self) -> None:
        configuration = self.configuration(following=False)
        configuration = replace(
            configuration, pan_tilt=replace(configuration.pan_tilt, mode="hardware")
        )
        with patch("app.main.FollowSession", side_effect=AssertionError("session constructed")):
            status, records, detector_model = self.run_script(
                configuration, ScriptedSource([frame_packet(0)]), {0: (pigeon(),)}
            )
        self.assertEqual(status, 0)
        self.assertEqual(detector_model.confidence, ACQUISITION_CONFIDENCE)
        self.assertNotIn("target", next(record for record in records if record["event"] == "frame"))

    def test_cancellation_during_camera_start_prevents_driver_construction(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration, pan_tilt=replace(configuration.pan_tilt, mode="hardware")
        )
        source = ScriptedSource([], on_start=lambda: signal.raise_signal(signal.SIGTERM))
        with (
            patch.object(PCA9685, "__init__", side_effect=AssertionError("physical driver opened")),
            patch.object(Calibration, "load", side_effect=AssertionError("profile loaded")),
        ):
            status, records, _ = self.run_script(configuration, source, {}, replay=False)
        self.assertEqual(status, 0)
        self.assertTrue(source.closed)
        self.assertEqual(records[-1]["processed_frames"], 0)
        self.assertIsNone(records[-1]["error"])

    def test_worker_failure_is_reported_once_and_source_is_closed(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration, pan_tilt=replace(configuration.pan_tilt, mode="hardware")
        )
        source = ScriptedSource([frame_packet(0)], delay_s=WORKER_OPPORTUNITY_S)
        with (
            patch.object(Calibration, "load", return_value=Calibration()),
            patch.object(PCA9685, "__init__", return_value=None),
            patch.object(PCA9685, "move", side_effect=RuntimeError("first command failed")),
            patch.object(PCA9685, "close"),
        ):
            status, records, _ = self.run_script(configuration, source, {}, replay=False)
        self.assertEqual(status, 1)
        self.assertTrue(source.closed)
        failure = str(records[-1]["error"])
        self.assertEqual(failure.count("first command failed"), 1)
        self.assertNotIn("cleanup", failure)

    def test_worker_failure_keeps_a_simultaneous_log_failure_without_duplication(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration, pan_tilt=replace(configuration.pan_tilt, mode="hardware")
        )
        original_event = RunOutput.event

        def fail_log_thread(output: RunOutput, event: str, **fields: JsonValue) -> None:
            if threading.current_thread().name == "log-writer":
                raise OSError("log sink failed")
            original_event(output, event, **fields)

        with (
            patch.object(Calibration, "load", return_value=Calibration()),
            patch.object(PCA9685, "__init__", return_value=None),
            patch.object(PCA9685, "move", side_effect=RuntimeError("first command failed")),
            patch.object(PCA9685, "close"),
            patch.object(RunOutput, "event", fail_log_thread),
        ):
            status, records, _ = self.run_script(
                configuration,
                ScriptedSource([frame_packet(0)], delay_s=WORKER_OPPORTUNITY_S),
                {},
                replay=False,
            )
        self.assertEqual(status, 1)
        failure = str(records[-1]["error"])
        self.assertEqual(failure.count("first command failed"), 1)
        self.assertEqual(failure.count("log sink failed"), 1)

    def test_preview_bind_failure_prevents_mount_initialisation(self) -> None:
        configuration = self.configuration()
        source = ScriptedSource([])
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            _, port = cast(tuple[str, int], listener.getsockname())
            configuration = replace(
                configuration,
                pan_tilt=replace(configuration.pan_tilt, mode="hardware"),
                preview=replace(configuration.preview, enabled=True, port=port),
            )
            with (
                patch.object(
                    PCA9685, "__init__", side_effect=AssertionError("physical driver opened")
                ),
                patch.object(Calibration, "load", side_effect=AssertionError("profile loaded")),
            ):
                status, records, _ = self.run_script(configuration, source, {}, replay=False)
        self.assertEqual(status, 1)
        self.assertTrue(source.closed)
        self.assertEqual(records[-1]["event"], "run_end")
        self.assertIn("bind", str(records[-1]["error"]))

    def test_unexpected_export_failure_still_finalises_the_run(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration,
            preview=replace(configuration.preview, record=True, record_min_free_bytes=0),
        )
        with patch(
            "app.main.export_recording", side_effect=RuntimeError("export crashed")
        ) as exporter:
            status, records, _ = self.run_script(
                configuration,
                ScriptedSource([frame_packet(sequence) for sequence in range(3)]),
                {sequence: (pigeon(),) for sequence in range(3)},
            )
        exporter.assert_called_once()
        self.assertEqual(status, 1)
        self.assertEqual(records[-1]["event"], "run_end")
        self.assertEqual(records[-1]["processed_frames"], 3)
        self.assertIn("export crashed", str(records[-1]["error"]))
        recording = cast(dict[str, JsonValue], records[-1]["demo_recording"])
        self.assertGreater(cast(int, recording["frames_written"]), 0)

    def test_sigterm_during_export_requests_cancellation_and_restores_handlers(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration,
            preview=replace(configuration.preview, record=True, record_min_free_bytes=0),
        )
        previous_handler = signal.getsignal(signal.SIGTERM)
        cancellation_states: list[bool] = []

        def export_after_signal(
            spool_directory: Path,
            output_path: Path,
            *,
            timeout_s: float,
            default_frame_duration_s: float,
            max_bytes: int,
            cancel_requested: Callable[[], bool],
        ) -> ExportResult:
            self.assertTrue((spool_directory / "index.jsonl").is_file())
            cancellation_states.append(cancel_requested())
            signal.raise_signal(signal.SIGTERM)
            cancellation_states.append(cancel_requested())
            return ExportResult(
                success=False,
                output_path=None,
                duration_s=None,
                expected_duration_s=default_frame_duration_s,
                error="recording export cancelled",
                source_retained=True,
            )

        with patch("app.main.export_recording", side_effect=export_after_signal):
            status, records, _ = self.run_script(
                configuration, ScriptedSource([frame_packet(0)]), {0: (pigeon(),)}
            )
        self.assertEqual(status, 1)
        self.assertEqual(cancellation_states, [False, True])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertIn("recording export cancelled", str(records[-1]["error"]))
        self.assertEqual(records[-1]["event"], "run_end")

    def test_storage_failure_does_not_stop_frame_processing(self) -> None:
        configuration = self.configuration()
        configuration = replace(
            configuration,
            preview=replace(configuration.preview, record=True, record_min_free_bytes=0),
        )
        source = ScriptedSource(
            [frame_packet(sequence) for sequence in range(MEDIA_FAULT_FRAME_COUNT)],
            delay_s=MEDIA_FAULT_FRAME_INTERVAL_S,
        )
        observations = {sequence: (pigeon(),) for sequence in range(MEDIA_FAULT_FRAME_COUNT)}
        with patch.object(Path, "write_bytes", side_effect=OSError(errno.ENOSPC, "disk full")):
            status, records, _ = self.run_script(configuration, source, observations)
        self.assertEqual(status, 1)
        self.assertTrue(source.closed)
        self.assertEqual(records[-1]["processed_frames"], MEDIA_FAULT_FRAME_COUNT)
        self.assertEqual(
            sum(record["event"] == "frame" for record in records), MEDIA_FAULT_FRAME_COUNT
        )
        self.assertIn("disk full", str(records[-1]["error"]))

    def test_camera_change_clears_selection_even_when_previous_name_was_missing(self) -> None:
        configuration = self.configuration()
        queue = OutputQueue()
        processor = FrameProcessor(
            configuration,
            ScriptedSource([]),
            ScriptedDetector({sequence: (pigeon(),) for sequence in range(3)}),
            None,
            queue,
            threading.Event(),
            time.monotonic(),
            replay_fps=REPLAY_FPS,
        )
        for packet in (frame_packet(0), frame_packet(1), frame_packet(2, camera="wide")):
            processor.process(packet)
        records = queue.take_logs()
        frames = [record.fields for record in records if record.event == "frame"]
        first_identity = cast(list[dict[str, JsonValue]], frames[0]["detections"])[0]["track_id"]
        changed_identity = cast(list[dict[str, JsonValue]], frames[2]["detections"])[0]["track_id"]
        self.assertNotEqual(first_identity, changed_identity)
        self.assertIsNone(cast(dict[str, JsonValue], frames[2]["target"])["track"])
        lost = [record.fields for record in records if record.event == "target_lost"]
        self.assertEqual(lost[0]["track_id"], first_identity)

    def test_live_observation_age_includes_sensor_to_delivery_duration(self) -> None:
        processor = FrameProcessor(
            self.configuration(),
            ScriptedSource([]),
            ScriptedDetector({}),
            None,
            OutputQueue(),
            threading.Event(),
            time.monotonic(),
        )
        packet = replace(
            frame_packet(0),
            delivered_monotonic_s=time.monotonic() - PICKUP_AGE_S,
            delivery_ms=SENSOR_DELIVERY_MS,
        )
        now_s, observed_at_s, timing = processor.follow_time(packet)
        self.assertEqual(timing, "sensor")
        self.assertGreaterEqual(
            now_s - observed_at_s, PICKUP_AGE_S + SENSOR_DELIVERY_MS / MILLISECONDS_PER_SECOND
        )
        self.assertEqual(processor.follow_time(replace(packet, delivery_ms=None))[2], "delivery")
        self.assertEqual(
            processor.follow_time(replace(packet, delivered_monotonic_s=None))[2], "unavailable"
        )

    def test_reaction_test_disables_following_before_camera_combination_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "pigeon.jpg"
            image.write_bytes(b"image")
            configuration_path = root / "config.yaml"
            configuration: dict[str, JsonValue] = {
                "follow": {"enabled": True},
                "pan_tilt": {"mode": "hardware", "calibration": "missing.yaml"},
                "cameras": [{"name": "wide", "role": "day"}, {"name": "noir", "role": "night"}],
            }
            configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            loaded = load_configuration(
                CommandOptions(config=str(configuration_path), reaction_test=str(image))
            )
        self.assertFalse(loaded.follow.enabled)
        self.assertTrue(loaded.confirmation.enabled)

    def test_invalid_source_fps_uses_explicit_configured_fallback(self) -> None:
        source = ScriptedSource([])
        source.fps = float("nan")
        self.assertEqual(
            replay_frame_rate(cast(FrameStream, source), REPLAY_FPS),
            (REPLAY_FPS, "configured_camera_fps"),
        )


if __name__ == "__main__":
    unittest.main()
