from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .actuators import build_actuator
from .capture import (
    CaptureError,
    EndOfStream,
    LatestFrameStream,
    Picamera2Capture,
    VideoFileCapture,
)
from .config import AppConfig, ConfigError
from .controller import RuntimeController
from .detector import DetectorError, ReplayDetector, UltralyticsDetector
from .geometry import TargetGeometry
from .logger import EventLogger, health_snapshot
from .safety import GPIOInputs, SafetyGate
from .turret import build_turret
from .watchdog import SoftwareWatchdog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Columba Zero safe prototype runtime")
    parser.add_argument("--config", default="app/config.yaml", help="YAML configuration path")
    parser.add_argument("--video", help="replay a video instead of using Picamera2")
    parser.add_argument(
        "--detections", help="frame-indexed JSON detections for deterministic replay"
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="enable configured GPIO inputs/outputs; omitted means simulation outputs",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = AppConfig.load(args.config)
    logger = EventLogger(config.logging.path)

    if config.actuator.mode == "laser" and config.actuator.laser_enabled:
        if not args.hardware:
            raise ConfigError("enabled laser mode requires --hardware")
        if config.safety.emergency_stop_pin is None or config.safety.interlock_pin is None:
            raise ConfigError("enabled laser mode requires physical E-stop and interlock pins")

    # Load fallible vision resources before opening any GPIO outputs.
    detector = (
        ReplayDetector.from_json(args.detections)
        if args.detections
        else UltralyticsDetector(
            config.model.path,
            config.model.image_size,
            min(config.model.confidence, config.model.safety_confidence),
        )
    )
    capture = (
        VideoFileCapture(args.video, realtime=True)
        if args.video
        else Picamera2Capture(
            config.camera.width, config.camera.height, config.camera.fps, config.camera.rotation
        )
    )

    gpio_inputs = (
        GPIOInputs(
            config.safety.emergency_stop_pin,
            config.safety.interlock_pin,
            config.safety.interlock_active_high,
        )
        if args.hardware
        else None
    )
    emergency_stop = gpio_inputs.emergency_pressed if gpio_inputs else lambda: False
    interlock_closed = gpio_inputs.interlock_closed if gpio_inputs else lambda: True

    safety = SafetyGate(
        config.safety,
        config.model,
        config.camera.stale_after_s,
        emergency_stop=emergency_stop,
        interlock_closed=interlock_closed,
        require_hardware_interlock=config.actuator.mode == "laser",
    )
    turret = build_turret(config.turret)

    def output_permitted() -> bool:
        return interlock_closed() and not emergency_stop()

    actuator = build_actuator(config.actuator, args.hardware, output_permitted)
    geometry = TargetGeometry(config.geometry)
    controller = RuntimeController(config, safety, geometry, turret, actuator, logger)

    stream = LatestFrameStream(capture)
    watchdog = SoftwareWatchdog(
        timeout_s=config.safety.watchdog_timeout_s,
        on_timeout=lambda: controller.fail_safe("software_watchdog_timeout"),
    )
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    frames = 0
    last_health = time.monotonic()
    try:
        if not controller.arm():
            return 2
        stream.start()
        watchdog.start()
        while not stopping:
            watchdog.beat()
            packet = stream.get(config.camera.frame_timeout_s)
            started = time.monotonic()
            detections = detector.detect(packet)
            inference_s = time.monotonic() - started
            controller.process(packet, detections)
            watchdog.beat()
            if controller.state.value == "SAFE":
                return 1
            frames += 1
            if time.monotonic() - last_health >= config.logging.health_interval_s:
                health = health_snapshot(stream.dropped_frames)
                logger.event(
                    "health",
                    state=controller.state.value,
                    inference_s=inference_s,
                    **health,
                )
                if health.get("cpu_temperature_c", 0) >= config.safety.max_cpu_temperature_c:
                    controller.fail_safe("over_temperature")
                    return 1
                last_health = time.monotonic()
            if args.max_frames and frames >= args.max_frames:
                break
    except EndOfStream:
        logger.event("replay_complete", frames=frames)
    except (CaptureError, DetectorError) as exc:
        controller.fail_safe("runtime_fault", exc)
        return 1
    except Exception as exc:
        controller.fail_safe("unhandled_exception", exc)
        return 1
    finally:
        watchdog.close()
        controller.shutdown()
        stream.close()
        actuator.close()
        turret.close()
        if gpio_inputs is not None:
            gpio_inputs.close()
    return 0


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.detections and not args.video:
        parser.error("--detections requires --video")
    try:
        raise SystemExit(run(args))
    except (ConfigError, DetectorError, CaptureError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    cli()
