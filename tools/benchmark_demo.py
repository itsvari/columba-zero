"""Compare replay processing with tracking, preview, recording, and both outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import yaml

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig
from app.json_data import JsonValue
from app.main import CommandOptions, run
from tools.summarize_run import summarize_log

Mode = Literal["tracking", "preview", "recording", "both"]
MODES: tuple[Mode, ...] = ("tracking", "preview", "recording", "both")
DEFAULT_FRAME_LIMIT = 300


@dataclass
class BenchmarkOptions:
    video: str = ""
    config: str = "app/config.yaml"
    frames: int = DEFAULT_FRAME_LIMIT
    output: str = "outputs/demo-benchmark"
    backend: Literal["native", "ultralytics"] | None = None


def mode_configuration(
    configuration: AppConfig, options: BenchmarkOptions, mode: Mode
) -> AppConfig:
    return replace(
        configuration,
        model=replace(
            configuration.model,
            backend=options.backend if options.backend is not None else configuration.model.backend,
        ),
        display=replace(configuration.display, enabled=False),
        output=replace(
            configuration.output, directory=str(Path(options.output) / mode), record=False
        ),
        confirmation=replace(configuration.confirmation, enabled=True),
        follow=replace(configuration.follow, enabled=True),
        pan_tilt=replace(configuration.pan_tilt, mode="simulate"),
        preview=replace(
            configuration.preview,
            enabled=mode in ("preview", "both"),
            record=mode in ("recording", "both"),
            host="127.0.0.1",
            port=0,
        ),
    )


def benchmark_mode(
    configuration: AppConfig, options: BenchmarkOptions, mode: Mode
) -> dict[str, JsonValue]:
    configured = mode_configuration(configuration, options, mode)
    configured.validate()
    root = Path(options.output)
    root.mkdir(parents=True, exist_ok=True)
    configuration_path = root / f"{mode}.yaml"
    configuration_path.write_text(yaml.safe_dump(asdict(configured)), encoding="utf-8")
    output_directory = Path(configured.output.directory)
    previous_logs = frozenset(output_directory.glob("*/detections.jsonl"))
    started_at_s = time.monotonic()
    status = run(
        CommandOptions(
            config=str(configuration_path), video=options.video, max_frames=options.frames
        )
    )
    result: dict[str, JsonValue] = {
        "mode": mode,
        "status": status,
        "wall_s": time.monotonic() - started_at_s,
    }
    logs = frozenset(output_directory.glob("*/detections.jsonl")) - previous_logs
    if len(logs) != 1:
        return {**result, "status": status or 1, "error": "runtime did not create one run log"}
    log = next(iter(logs))
    result["log"] = str(log)
    result["summary"] = summarize_log(log)
    media: dict[str, JsonValue] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        record = cast(dict[str, JsonValue], json.loads(line))
        if record["event"] in ("preview_stopped", "demo_export"):
            media[str(record["event"])] = record
    result["media"] = media
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="recorded video to process sequentially in simulation")
    parser.add_argument("--config", default=BenchmarkOptions.config)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_LIMIT)
    parser.add_argument("--output", default=BenchmarkOptions.output)
    parser.add_argument("--backend", choices=("native", "ultralytics"))
    options = BenchmarkOptions()
    parser.parse_args(namespace=options)
    if options.frames <= 0:
        parser.error("--frames must be positive")
    configuration = AppConfig.load(options.config)
    results = [benchmark_mode(configuration, options, mode) for mode in MODES]
    report: dict[str, JsonValue] = {
        "scope": "desktop or device replay; simulated mount; no camera or closed-loop acceptance",
        "preview_clients": 0,
        "processing_elapsed_excludes_export": True,
        "video": options.video,
        "results": cast(list[JsonValue], results),
    }
    destination = Path(options.output) / "benchmark.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Replay comparison: {destination}")
    raise SystemExit(1 if any(result["status"] != 0 for result in results) else 0)


if __name__ == "__main__":
    main()
