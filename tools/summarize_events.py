#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize runtime safety and performance logs")
    parser.add_argument("events_jsonl")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.events_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("event log is empty")
    event_counts = Counter(str(record.get("event")) for record in records)
    faults = Counter(
        str(record.get("reason")) for record in records if record.get("event") == "fault"
    )
    rejections = Counter(
        str(record.get("reason"))
        for record in records
        if record.get("event") == "activation_rejected"
    )
    latencies = [
        float(record["inference_s"])
        for record in records
        if record.get("event") == "health" and "inference_s" in record
    ]
    elapsed_hours = (
        max(0.0, float(records[-1]["timestamp"]) - float(records[0]["timestamp"])) / 3600
    )
    activations = event_counts["activation"]
    summary = {
        "elapsed_hours": elapsed_hours,
        "activations": activations,
        "activations_per_hour": activations / elapsed_hours if elapsed_hours else None,
        "faults": faults,
        "activation_rejections": rejections,
        "inference_mean_ms": statistics.mean(latencies) * 1000 if latencies else None,
        "inference_p95_ms": (percentile(latencies, 0.95) * 1000 if latencies else None),
        "maximum_memory_rss_bytes": max(
            (int(record["memory_rss_bytes"]) for record in records if "memory_rss_bytes" in record),
            default=None,
        ),
        "maximum_temperature_c": max(
            (
                float(record["cpu_temperature_c"])
                for record in records
                if "cpu_temperature_c" in record
            ),
            default=None,
        ),
    }
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
