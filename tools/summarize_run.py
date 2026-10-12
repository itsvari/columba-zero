"""Summarise runtime logs with bounded memory, including health samples."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import cast

from app.json_data import JsonValue

MILLISECONDS_PER_SECOND = 1000
RECENT_FRAME_LIMIT = 1000
LATENCY_PERCENTILE = 0.95


def numeric_field(record: dict[str, JsonValue], name: str) -> int | float | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def summarize_log(path: Path) -> dict[str, JsonValue]:
    frame_count = 0
    inference_total_ms = 0.0
    inference_maximum_ms = 0.0
    memory_first_bytes: int | float | None = None
    memory_last_bytes: int | float | None = None
    memory_maximum_bytes: int | float | None = None
    temperature_maximum_c: int | float | None = None
    run_end: dict[str, JsonValue] | None = None
    recent_inference_ms: deque[float] = deque(maxlen=RECENT_FRAME_LIMIT)
    with path.open(encoding="utf-8") as log:
        for line_number, line in enumerate(log, start=1):
            record = cast(dict[str, JsonValue], json.loads(line))
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: log record must be a JSON mapping")
            if record.get("event") == "frame":
                inference_s = numeric_field(record, "inference_s")
                if inference_s is None:
                    raise ValueError(f"{path}:{line_number}: frame is missing inference_s")
                inference_ms = inference_s * MILLISECONDS_PER_SECOND
                frame_count += 1
                inference_total_ms += inference_ms
                inference_maximum_ms = max(inference_maximum_ms, inference_ms)
                recent_inference_ms.append(inference_ms)
            memory_bytes = numeric_field(record, "memory_rss_bytes")
            if memory_bytes is not None:
                if memory_first_bytes is None:
                    memory_first_bytes = memory_bytes
                memory_last_bytes = memory_bytes
                memory_maximum_bytes = (
                    memory_bytes
                    if memory_maximum_bytes is None
                    else max(memory_bytes, memory_maximum_bytes)
                )
            temperature_c = numeric_field(record, "temperature_c")
            if temperature_c is not None:
                temperature_maximum_c = (
                    temperature_c
                    if temperature_maximum_c is None
                    else max(temperature_c, temperature_maximum_c)
                )
            if record.get("event") == "run_end":
                run_end = record
    ordered_inference_ms = sorted(recent_inference_ms)
    return {
        "frames": frame_count,
        "inference_mean_ms": inference_total_ms / frame_count if frame_count else None,
        "inference_max_ms": inference_maximum_ms,
        "inference_p95_last_1000_ms": (
            ordered_inference_ms[round((len(ordered_inference_ms) - 1) * LATENCY_PERCENTILE)]
            if ordered_inference_ms
            else None
        ),
        "memory_first_bytes": memory_first_bytes,
        "memory_last_bytes": memory_last_bytes,
        "memory_max_bytes": memory_maximum_bytes,
        "temperature_max_c": temperature_maximum_c,
        "completed_cleanly": run_end is not None and run_end.get("error") is None,
        "run_end": run_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    arguments = parser.parse_args()
    try:
        report = summarize_log(Path(arguments.log))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot summarise log: {error}") from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
