"""Summarise runtime logs with bounded memory, including health samples.

Timing distributions (per-stage, frame delivery, decision and trigger reaction
times) use fixed 0.1 ms histogram bins up to 10 s, so memory does not grow with
the length of the run; percentiles are reported at that resolution. Values of
10 s or more are kept exactly while there are at most 1000 of them; beyond that,
percentiles falling among them are reported as 10000 ms and listed in
`lower_bound`. The maximum is always exact.
"""

from __future__ import annotations

import argparse
import json
import math
from array import array
from collections import Counter, deque
from pathlib import Path
from typing import cast

from app.json_data import JsonValue

MILLISECONDS_PER_SECOND = 1000
RECENT_FRAME_LIMIT = 1000
LATENCY_PERCENTILE = 0.95
HISTOGRAM_BIN_MS = 0.1
HISTOGRAM_RANGE_MS = 10_000.0
OVERFLOW_EXACT_LIMIT = 1000
REPORTED_PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
STAGES = ("wait", "preprocess", "inference", "postprocess", "track")


def numeric_field(record: dict[str, JsonValue], name: str) -> int | float | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


class Distribution:
    """Count, mean, maximum and histogram percentiles of non-negative milliseconds."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.bins = array("Q", bytes(8 * round(HISTOGRAM_RANGE_MS / HISTOGRAM_BIN_MS)))
        self.overflow_count = 0
        self.overflow_values = array("d")

    def add(self, value_ms: float) -> None:
        value_ms = max(0.0, value_ms)
        self.count += 1
        self.total += value_ms
        self.maximum = max(self.maximum, value_ms)
        index = int(value_ms / HISTOGRAM_BIN_MS)
        if index < len(self.bins):
            self.bins[index] += 1
            return
        self.overflow_count += 1
        if len(self.overflow_values) < OVERFLOW_EXACT_LIMIT:
            self.overflow_values.append(value_ms)

    def percentiles(self) -> tuple[dict[str, float], list[str]]:
        """Nearest-rank percentiles, as the upper edge of their bin (capped at the maximum),
        and the names of those that are only lower bounds."""
        ranks = [
            (name, max(1, math.ceil(fraction * self.count)))
            for name, fraction in REPORTED_PERCENTILES
        ]
        results: dict[str, float] = {}
        lower_bounds: list[str] = []
        in_range = self.count - self.overflow_count
        exact = sorted(self.overflow_values) if self.overflow_count <= OVERFLOW_EXACT_LIMIT else []
        for name, rank in ranks:
            if rank > in_range:
                if exact:
                    results[name] = exact[rank - in_range - 1]
                else:
                    results[name] = HISTOGRAM_RANGE_MS
                    lower_bounds.append(name)
        ranks = [(name, rank) for name, rank in ranks if rank <= in_range]
        seen = 0
        for index, count in enumerate(self.bins):
            if not count:
                continue
            seen += count
            while ranks and seen >= ranks[0][1]:
                results[ranks.pop(0)[0]] = min(
                    round((index + 1) * HISTOGRAM_BIN_MS, 1), self.maximum
                )
            if not ranks:
                break
        return results, lower_bounds

    def summary(self) -> dict[str, JsonValue] | None:
        if not self.count:
            return None
        percentiles, lower_bounds = self.percentiles()
        summary: dict[str, JsonValue] = {
            "count": self.count,
            "mean": self.total / self.count,
            **{name: percentiles[name] for name, _ in REPORTED_PERCENTILES},
            "max": self.maximum,
        }
        if lower_bounds:
            summary["lower_bound"] = cast(JsonValue, lower_bounds)
        return summary


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
    distributions = {
        name: Distribution()
        for name in ("inference_total", *STAGES, "delivery", "decision", "reaction")
    }
    frames_by_camera: Counter[str] = Counter()
    trigger_count = 0
    camera_switch_count = 0
    with path.open(encoding="utf-8") as log:
        for line_number, line in enumerate(log, start=1):
            record = cast(dict[str, JsonValue], json.loads(line))
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: log record must be a JSON mapping")
            event = record.get("event")
            if event == "frame":
                inference_s = numeric_field(record, "inference_s")
                if inference_s is None:
                    raise ValueError(f"{path}:{line_number}: frame is missing inference_s")
                inference_ms = inference_s * MILLISECONDS_PER_SECOND
                frame_count += 1
                inference_total_ms += inference_ms
                inference_maximum_ms = max(inference_maximum_ms, inference_ms)
                recent_inference_ms.append(inference_ms)
                distributions["inference_total"].add(inference_ms)
                stages = record.get("stage_ms")
                if isinstance(stages, dict):
                    for stage in STAGES:
                        value = numeric_field(stages, stage)
                        if value is not None:
                            distributions[stage].add(value)
                for name in ("delivery", "decision"):
                    value = numeric_field(record, f"{name}_ms")
                    if value is not None:
                        distributions[name].add(value)
                camera = record.get("camera")
                if isinstance(camera, str):
                    frames_by_camera[camera] += 1
            elif event == "trigger":
                trigger_count += 1
                first_seen_s = numeric_field(record, "first_seen_captured_at")
                confirmed_s = numeric_field(record, "confirmed_at")
                if first_seen_s is not None and confirmed_s is not None:
                    distributions["reaction"].add(
                        (confirmed_s - first_seen_s) * MILLISECONDS_PER_SECOND
                    )
            elif event == "camera_switch":
                camera_switch_count += 1
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
            if event == "run_end":
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
        # wait: frame delivery to pickup by the processing loop; delivery: sensor exposure
        # start to delivery; decision: delivery to decision; reaction: first frame of a
        # track to its confirmed trigger.
        "distributions_ms": {
            name: distribution.summary() for name, distribution in distributions.items()
        },
        "triggers": trigger_count,
        "camera_switches": camera_switch_count,
        "frames_by_camera": cast(JsonValue, dict(frames_by_camera)),
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
