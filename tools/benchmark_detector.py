#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import cast

from app.capture import CaptureError, EndOfStream, VideoFileCapture
from app.inference import NMS_INTERSECTION_OVER_UNION, PredictionModel
from app.json_data import JsonValue
from app.models import Frame
from app.output import health_snapshot

MILLISECONDS_PER_SECOND = 1000
LATENCY_PERCENTILE = 0.95
VIDEO_FRAME_TIMEOUT_S = 2.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def benchmark_video(
    model: PredictionModel,
    video: str,
    image_size_px: int,
    frame_limit: int,
    confidence: float,
) -> dict[str, JsonValue]:
    if frame_limit <= 0:
        raise ValueError("frame limit must be positive")
    import psutil

    def predict_frame(frame: Frame) -> None:
        model.predict(
            frame,
            imgsz=image_size_px,
            conf=confidence,
            verbose=False,
            device="cpu",
            rect=False,
            iou=NMS_INTERSECTION_OVER_UNION,
        )

    source = VideoFileCapture(video)
    try:
        source.start()
        predict_frame(source.get(VIDEO_FRAME_TIMEOUT_S).frame)
        source.close()
        source = VideoFileCapture(video)
        source.start()
        latencies_s: list[float] = []
        process = psutil.Process()
        initial_memory_bytes = process.memory_info().rss
        initial_cpu_times = process.cpu_times()
        processed_frames = 0
        started_at_monotonic_s = time.monotonic()
        while processed_frames < frame_limit:
            read_started_at_monotonic_s = time.monotonic()
            try:
                packet = source.get(VIDEO_FRAME_TIMEOUT_S)
            except EndOfStream:
                break
            predict_frame(packet.frame)
            latencies_s.append(time.monotonic() - read_started_at_monotonic_s)
            processed_frames += 1
        elapsed_s = time.monotonic() - started_at_monotonic_s
        cpu_times = process.cpu_times()
        memory_bytes = process.memory_info().rss
        result: dict[str, JsonValue] = {
            "image_size": image_size_px,
            "frames": processed_frames,
            "elapsed_s": elapsed_s,
            "fps": processed_frames / elapsed_s if elapsed_s else 0.0,
            "latency_mean_ms": (
                statistics.mean(latencies_s) * MILLISECONDS_PER_SECOND if latencies_s else 0.0
            ),
            "latency_p95_ms": (
                percentile(latencies_s, LATENCY_PERCENTILE) * MILLISECONDS_PER_SECOND
                if latencies_s
                else 0.0
            ),
            "process_cpu_s": (
                cpu_times.user
                + cpu_times.system
                - initial_cpu_times.user
                - initial_cpu_times.system
            ),
            "memory_rss_bytes": memory_bytes,
            "memory_growth_bytes": memory_bytes - initial_memory_bytes,
        }
        temperature_c = health_snapshot().get("temperature_c")
        if temperature_c is not None:
            result["temperature_c"] = temperature_c
        return result
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end YOLO video inference")
    parser.add_argument("model")
    parser.add_argument("video")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--output", default="benchmark.json")
    arguments = parser.parse_args()
    if arguments.frames <= 0:
        parser.error("--frames must be positive")

    from ultralytics import YOLO

    model = cast(PredictionModel, YOLO(arguments.model, task="detect"))
    results: list[dict[str, JsonValue]] = []
    for image_size_px in arguments.sizes:
        try:
            result = benchmark_video(
                model,
                arguments.video,
                image_size_px,
                arguments.frames,
                arguments.confidence,
            )
        except CaptureError as error:
            raise SystemExit(f"benchmark failed: {error}") from error
        results.append(result)
        print(json.dumps(result))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
