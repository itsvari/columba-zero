"""Measure full-call latency minus NCNN extract in isolated, alternating workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal, Protocol, cast
from unittest.mock import patch

import numpy

from app.models import Frame
from native.verify_native import (
    NativeModel,
    describe_package,
    load_reference,
    predict_reference,
)

Backend = Literal["native", "ultralytics"]
NANOSECONDS_PER_MILLISECOND = 1_000_000
WORKER_STOP_TIMEOUT_S = 10


@dataclass(frozen=True)
class BenchmarkOptions:
    model: Path
    image: Path
    threads: int
    confidence: float
    frame_hw: tuple[int, int]
    compute: str


@dataclass(frozen=True)
class Sample:
    total_ms: float
    extract_ms: float


class NcnnMatrix(Protocol):
    dims: int
    w: int
    h: int


class NcnnExtractor(Protocol):
    def extract(self, name: str) -> tuple[int, NcnnMatrix]: ...


def statistics(samples_ms: list[float]) -> dict[str, float]:
    return {
        "p50": float(numpy.percentile(samples_ms, 50)),
        "p95": float(numpy.percentile(samples_ms, 95)),
        "mean": float(numpy.mean(samples_ms)),
        "std": float(numpy.std(samples_ms)),
    }


def backend_sampler(backend: Backend, options: BenchmarkOptions) -> Callable[[], Sample]:
    import cv2

    cv2.setNumThreads(1)
    frame = cast(Frame | None, cv2.imread(str(options.image)))
    if frame is None:
        raise ValueError(f"cannot read image: {options.image}")
    frame = cast(Frame, cv2.resize(frame, (options.frame_hw[1], options.frame_hw[0])))
    if backend == "native":
        import columba_native

        native = cast(
            NativeModel,
            columba_native.Detector(str(options.model), options.threads, compute=options.compute),
        )

        def native_sample() -> Sample:
            started_ns = time.perf_counter_ns()
            _, timings = native.detect_with_timings(frame, options.confidence)
            total_ms = (time.perf_counter_ns() - started_ns) / NANOSECONDS_PER_MILLISECOND
            return Sample(total_ms, timings["extract"])

        return native_sample
    import ncnn
    import torch

    torch.set_num_threads(options.threads)
    description = describe_package(options.model)
    reference = load_reference(
        options.model, description, frame, options.confidence, options.threads, options.compute
    )
    original_extract = cast(
        Callable[[NcnnExtractor, str], tuple[int, NcnnMatrix]], ncnn.Extractor.extract
    )
    extraction_samples_ms: list[float] = []

    def timed_extract(extractor: NcnnExtractor, name: str) -> tuple[int, NcnnMatrix]:
        started_ns = time.perf_counter_ns()
        result = original_extract(extractor, name)
        extraction_samples_ms.append(
            (time.perf_counter_ns() - started_ns) / NANOSECONDS_PER_MILLISECOND
        )
        return result

    def reference_sample() -> Sample:
        extraction_samples_ms.clear()
        with patch.object(ncnn.Extractor, "extract", timed_extract):
            started_ns = time.perf_counter_ns()
            predict_reference(reference, frame, description, options.confidence)
            total_ms = (time.perf_counter_ns() - started_ns) / NANOSECONDS_PER_MILLISECOND
        return Sample(total_ms, sum(extraction_samples_ms))

    return reference_sample


def sample_worker(backend: Backend, options: BenchmarkOptions, connection: Connection) -> None:
    try:
        sampler = backend_sampler(backend, options)
        connection.send("ready")
        while connection.recv() == "sample":
            connection.send(sampler())
    except Exception as error:
        connection.send(f"{backend} worker failed: {error}")
    finally:
        connection.close()


def measure(
    options: BenchmarkOptions, warmup: int, iterations: int
) -> dict[str, dict[str, list[float]]]:
    context = multiprocessing.get_context("spawn")
    backends: tuple[Backend, ...] = ("native", "ultralytics")
    connections = {backend: context.Pipe() for backend in backends}
    workers = {
        backend: context.Process(
            target=sample_worker, args=(backend, options, connections[backend][1])
        )
        for backend in backends
    }
    records: dict[str, dict[str, list[float]]] = {
        backend: {stage: [] for stage in ("total", "extract", "overhead")} for backend in backends
    }
    try:
        for backend, worker in workers.items():
            worker.start()
            connections[backend][1].close()
            ready = connections[backend][0].recv()
            if ready != "ready":
                raise RuntimeError(str(ready))
        for iteration in range(warmup + iterations):
            order = backends if iteration % 2 else tuple(reversed(backends))
            for backend in order:
                connection = connections[backend][0]
                connection.send("sample")
                sample = connection.recv()
                if not isinstance(sample, Sample):
                    raise RuntimeError(str(sample))
                if iteration < warmup:
                    continue
                records[backend]["total"].append(sample.total_ms)
                records[backend]["extract"].append(sample.extract_ms)
                records[backend]["overhead"].append(sample.total_ms - sample.extract_ms)
    finally:
        for backend, worker in workers.items():
            if worker.is_alive():
                connections[backend][0].send("stop")
                worker.join(WORKER_STOP_TIMEOUT_S)
            if worker.is_alive():
                worker.terminate()
                worker.join(WORKER_STOP_TIMEOUT_S)
            connections[backend][0].close()
            connections[backend][1].close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compute", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--frame-size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.warmup < 1 or arguments.iterations < 1:
        parser.error("warmup and iterations must be positive")
    description = describe_package(arguments.model)
    if description.decode != "yolo11":
        parser.error("benchmark Ultralytics predict on a standard NMS NCNN package")
    options = BenchmarkOptions(
        arguments.model.resolve(),
        arguments.image.resolve(),
        arguments.threads,
        arguments.confidence,
        tuple(arguments.frame_size),
        arguments.compute,
    )
    records = measure(options, arguments.warmup, arguments.iterations)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    report = {
        "label": "measured",
        "host": platform.node(),
        "architecture": platform.machine(),
        "shared_machine_noise": platform.system() == "Darwin",
        "threads": options.threads,
        "compute": options.compute,
        "compute_flags": {
            name: options.compute == "fp16"
            for name in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic")
        },
        "pinned_cores": affinity,
        "model": str(options.model),
        "input_hw": description.input_hw,
        "frame_hw": options.frame_hw,
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
        "native_ncnn_tag": "20260526",
        "method": (
            "isolated alternating workers; full call minus instrumented extract; IPC excluded"
        ),
        "milliseconds": {
            backend: {stage: statistics(samples) for stage, samples in stages.items()}
            for backend, stages in records.items()
        },
        "samples_ms": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"host": report["host"], "milliseconds": report["milliseconds"]}))


if __name__ == "__main__":
    main()
