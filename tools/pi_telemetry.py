"""Record process and Pi health separately from the capture and control threads."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType

import psutil

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.benchmark_live_tracking import device_sample

DEFAULT_INTERVAL_S = 1.0
DEFAULT_DURATION_S = 7200.0
MAXIMUM_INTERVAL_S = 60.0
SCHEMA_VERSION = 1
ARM_CLOCK_PATTERN = re.compile(r"frequency\(\d+\)=(\d+)")
CLOCK_QUERY_TIMEOUT_S = 1.0


def measured_arm_frequency_hz() -> int | None:
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_clock", "arm"],
            capture_output=True,
            text=True,
            timeout=CLOCK_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    matched = ARM_CLOCK_PATTERN.fullmatch(result.stdout.strip())
    if result.returncode != 0 or matched is None:
        return None
    frequency_hz = int(matched.group(1))
    return frequency_hz if frequency_hz > 0 else None


def record_process(
    process_id: int,
    destination: Path,
    interval_s: float,
    maximum_duration_s: float,
    stop: threading.Event,
) -> None:
    process = psutil.Process(process_id)
    started_at_s = time.monotonic()
    previous_sample_at_s: float | None = None
    stop_reason = "duration_limit"
    process.cpu_percent()
    with destination.open("x", encoding="utf-8") as output:
        session: dict[str, JsonValue] = {
            "kind": "session",
            "schema_version": SCHEMA_VERSION,
            "process_id": process_id,
            "process_created_unix_s": process.create_time(),
            "started_at_monotonic_s": started_at_s,
            "requested_interval_s": interval_s,
            "maximum_duration_s": maximum_duration_s,
            "scope": "read-only sidecar; process lifetime includes any export after capture stops",
        }
        output.write(json.dumps(session, allow_nan=False) + "\n")
        while time.monotonic() - started_at_s < maximum_duration_s:
            if stop.is_set():
                stop_reason = "signal"
                break
            try:
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    stop_reason = "process_exited"
                    break
                sample = device_sample()
                sample["cpu_measured_frequency_hz"] = measured_arm_frequency_hz()
                sample.update(
                    kind="sample",
                    process_id=process_id,
                    process_memory_rss_bytes=process.memory_info().rss,
                    process_cpu_percent=(
                        None if previous_sample_at_s is None else process.cpu_percent()
                    ),
                    previous_sample_at_monotonic_s=previous_sample_at_s,
                )
            except psutil.NoSuchProcess:
                stop_reason = "process_exited"
                break
            output.write(json.dumps(sample, allow_nan=False) + "\n")
            output.flush()
            previous_sample_at_s = time.monotonic()
            stop.wait(interval_s)
        output.write(
            json.dumps(
                {"kind": "end", "monotonic_s": time.monotonic(), "reason": stop_reason},
                allow_nan=False,
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    options = parser.parse_args()
    if options.process_id <= 0:
        parser.error("--process-id must be positive")
    if not math.isfinite(options.interval_s) or not 0 < options.interval_s <= MAXIMUM_INTERVAL_S:
        parser.error("--interval-s must be finite and in (0, 60]")
    if not math.isfinite(options.duration_s) or options.duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    stop = threading.Event()

    def request_stop(_signal_number: int, _frame: FrameType | None) -> None:
        stop.set()

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, request_stop)
    try:
        record_process(
            options.process_id, options.output, options.interval_s, options.duration_s, stop
        )
    except (OSError, ValueError, psutil.Error) as error:
        parser.exit(1, f"telemetry failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
