"""Probe live-camera throughput with a simulated mount and fixed visual input."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import yaml

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig
from app.json_data import JsonValue
from tools.summarize_run import summarize_log

DEFAULT_FRAME_COUNT = 180
DEFAULT_REPETITIONS = 5
DEFAULT_THREAD_COUNT = 4
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_WARMUP_FRAME_COUNT = 300
SAMPLE_INTERVAL_S = 1.0
SHUTDOWN_TIMEOUT_S = 10.0
CPU_FREQUENCY_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
TEMPERATURE_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def probe_configuration(configuration: AppConfig, directory: Path, threads: int) -> AppConfig:
    return replace(
        configuration,
        model=replace(configuration.model, threads=threads),
        display=replace(configuration.display, enabled=False),
        output=replace(configuration.output, directory=str(directory), record=False),
        confirmation=replace(configuration.confirmation, enabled=True),
        follow=replace(configuration.follow, enabled=True),
        pan_tilt=replace(configuration.pan_tilt, mode="simulate"),
        preview=replace(configuration.preview, enabled=False, record=False),
    )


def read_number(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def device_sample() -> dict[str, JsonValue]:
    sample: dict[str, JsonValue] = {
        "monotonic_s": time.monotonic(),
        "temperature_millicelsius": read_number(TEMPERATURE_PATH),
        "cpu_frequency_khz": read_number(CPU_FREQUENCY_PATH),
    }
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=SAMPLE_INTERVAL_S
        )
        sample["throttled"] = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        sample["throttled"] = None
    return sample


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_probe(
    configuration: AppConfig,
    directory: Path,
    threads: int,
    opencv_threads: int,
    frames: int,
    timeout_s: float,
) -> dict[str, JsonValue]:
    directory.mkdir(parents=True, exist_ok=False)
    configured = probe_configuration(configuration, directory / "runs", threads)
    configured.validate()
    configuration_path = directory / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(asdict(configured)), encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        f"import cv2; cv2.setNumThreads({opencv_threads}); from app.main import cli; cli()",
        "--config",
        str(configuration_path),
        "--max-frames",
        str(frames),
    ]
    started_at_s = time.monotonic()
    timed_out = False
    with (directory / "console.txt").open("wb") as console:
        process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT)
        try:
            with (directory / "device.jsonl").open("w", encoding="utf-8") as samples:
                while process.poll() is None:
                    samples.write(json.dumps(device_sample(), allow_nan=False) + "\n")
                    samples.flush()
                    if time.monotonic() - started_at_s >= timeout_s:
                        timed_out = True
                        break
                    time.sleep(SAMPLE_INTERVAL_S)
        finally:
            stop_process(process)
    logs = tuple((directory / "runs").glob("*/detections.jsonl"))
    summary = summarize_log(logs[0]) if len(logs) == 1 else None
    run_end = summary.get("run_end") if summary is not None else None
    valid = (
        not timed_out
        and process.returncode == 0
        and isinstance(run_end, dict)
        and run_end.get("error") is None
        and run_end.get("processed_frames") == frames
        and run_end.get("log_dropped_records") == 0
    )
    result: dict[str, JsonValue] = {
        "threads": threads,
        "opencv_threads": opencv_threads,
        "directory": str(directory),
        "process_exit_status": process.returncode,
        "timed_out": timed_out,
        "valid": valid,
        "wall_s_including_startup": time.monotonic() - started_at_s,
        "summary": summary,
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="app/config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threads", type=positive_integer, nargs="+", default=[DEFAULT_THREAD_COUNT]
    )
    parser.add_argument(
        "--opencv-threads", type=positive_integer, nargs="+", default=[DEFAULT_THREAD_COUNT]
    )
    parser.add_argument("--frames", type=positive_integer, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--repetitions", type=positive_integer, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--warmup-frames", type=positive_integer, default=DEFAULT_WARMUP_FRAME_COUNT
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    options = parser.parse_args()
    if not math.isfinite(options.timeout_s) or options.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    configuration = AppConfig.load(options.config)
    if configuration.model.backend != "native":
        parser.error("the live probe requires the native backend")
    options.output.mkdir(parents=True, exist_ok=False)
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    configuration_sha256 = hashlib.sha256(Path(options.config).read_bytes()).hexdigest()
    source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    probe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_hashes: dict[str, JsonValue] = {
        name: hashlib.sha256((Path(configuration.model.path) / name).read_bytes()).hexdigest()
        for name in ("metadata.yaml", "model.ncnn.param", "model.ncnn.bin")
    }
    results: list[dict[str, JsonValue]] = []
    cases = tuple(itertools.product(options.threads, options.opencv_threads))
    warmup_threads, warmup_opencv_threads = cases[0]
    warmup = run_probe(
        configuration,
        options.output / "warmup",
        warmup_threads,
        warmup_opencv_threads,
        options.warmup_frames,
        options.timeout_s,
    )
    if not warmup["valid"]:
        return 1
    orders = itertools.cycle((cases, tuple(reversed(cases))))
    for repetition, ordered_cases in zip(range(options.repetitions), orders, strict=False):
        for threads, opencv_threads in ordered_cases:
            name = f"repeat-{repetition + 1}-ncnn-{threads}-opencv-{opencv_threads}"
            directory = options.output / name
            print(str(directory), flush=True)
            result = run_probe(
                configuration, directory, threads, opencv_threads, options.frames, options.timeout_s
            )
            results.append(result)
            report: dict[str, JsonValue] = {
                "scope": "live camera; simulated mount; outputs off; no acceptance score",
                "method": "warm-up then alternating case order; fixed external scene required",
                "source_revision": source_revision,
                "source_dirty": source_dirty,
                "probe_sha256": probe_sha256,
                "configuration_sha256": configuration_sha256,
                "model_sha256": model_hashes,
                "openmp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                "warmup": warmup,
                "frames_per_case": options.frames,
                "results": cast(list[JsonValue], results),
            }
            (options.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            if not result["valid"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
