"""Summarize runtime logs without retaining every frame in memory."""
import argparse
import json
from collections import deque
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    count = 0
    total = 0.0
    maximum = 0.0
    memory = []
    temperature = []
    end = None
    recent = deque(maxlen=1000)
    with Path(args.log).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["event"] == "frame":
                value = record["inference_s"] * 1000
                count += 1
                total += value
                maximum = max(maximum, value)
                recent.append(value)
            if "memory_rss_bytes" in record:
                memory.append(record["memory_rss_bytes"])
            if "temperature_c" in record:
                temperature.append(record["temperature_c"])
            if record["event"] == "run_end":
                end = record
    ordered = sorted(recent)
    report = {"frames": count, "inference_mean_ms": total / count if count else None,
              "inference_max_ms": maximum,
              "inference_p95_last_1000_ms": ordered[round((len(ordered)-1)*0.95)] if ordered else None,
              "memory_first_bytes": memory[0] if memory else None,
              "memory_last_bytes": memory[-1] if memory else None,
              "memory_max_bytes": max(memory) if memory else None,
              "temperature_max_c": max(temperature) if temperature else None,
              "completed_cleanly": end is not None and end.get("error") is None,
              "run_end": end}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
