#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end YOLO video inference")
    parser.add_argument("model")
    parser.add_argument("video")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--output", default="benchmark.json")
    args = parser.parse_args()

    import cv2
    import psutil
    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    all_results: list[dict[str, object]] = []
    for size in args.sizes:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise SystemExit(f"cannot open {args.video}")
        latencies: list[float] = []
        started = time.monotonic()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_times()
        count = 0
        while count < args.frames:
            read_started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                break
            model.predict(frame, imgsz=size, conf=args.confidence, verbose=False, device="cpu", rect=False)
            latencies.append(time.monotonic() - read_started)
            count += 1
        elapsed = time.monotonic() - started
        cpu = process.cpu_times()
        capture.release()
        memory = process.memory_info().rss
        result = {
            "image_size": size,
            "frames": count,
            "elapsed_s": elapsed,
            "fps": count / elapsed if elapsed else 0.0,
            "latency_mean_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
            "latency_p95_ms": percentile(latencies, 0.95) * 1000 if latencies else 0.0,
            "process_cpu_s": (cpu.user + cpu.system) - (start_cpu.user + start_cpu.system),
            "memory_rss_bytes": memory,
            "memory_growth_bytes": memory - start_memory,
        }
        thermal = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal.exists():
            result["temperature_c"] = int(thermal.read_text().strip()) / 1000.0
        all_results.append(result)
        print(json.dumps(result))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


if __name__ == "__main__":
    main()
