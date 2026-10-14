"""Benchmark packed challenger heads with WS4's custom NCNN bridge.

Usage: python -m training.adapters.bench_challenger --help
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy
import yaml

from app.json_data import JsonValue
from tools import bench_ncnn
from training.adapters import dfine, nanodet, rfdetr

Framework = Literal["dfine", "rfdetr", "nanodet"]
MODEL_IDS = {
    "dfine": "dfine-n-coco",
    "rfdetr": "rfdetr-nano-coco",
    "nanodet": "nanodet-plus-m-coco",
}
DETR_QUERIES = 300
RFDETR_CLASS_COUNT = 91


def preprocess(
    framework: Framework, frame: bench_ncnn.Frame, input_hw: tuple[int, int]
) -> bench_ncnn.InferenceValues:
    return {"dfine": dfine.preprocess, "rfdetr": rfdetr.preprocess, "nanodet": nanodet.preprocess}[
        framework
    ](frame, input_hw)


def decode(
    framework: Framework,
    prediction: bench_ncnn.InferenceValues,
    camera_hw: tuple[int, int],
    input_hw: tuple[int, int],
    confidence: float,
) -> bench_ncnn.InferenceValues:
    if framework == "nanodet":
        return nanodet.decode(prediction, camera_hw, input_hw, confidence)
    return dfine.decode(prediction, camera_hw, confidence)


def benchmark(
    framework: Framework,
    model: Path,
    prefix: Path,
    host_profile: str,
    precision: str,
    input_hw: tuple[int, int],
    frame_path: Path,
    camera_hw: tuple[int, int] = (480, 640),
    confidence: float = 0.05,
) -> dict[str, JsonValue]:
    import cv2

    evidence = (prefix / "build-evidence.txt").read_text()
    if (
        f"profile={host_profile}" not in evidence
        or f"commit={bench_ncnn.NCNN_COMMIT}" not in evidence
    ):
        raise ValueError("NCNN build profile/commit mismatch")
    metadata: JsonValue = yaml.safe_load((model / "metadata.yaml").read_text())
    if not isinstance(metadata, dict) or metadata.get("imgsz") != list(input_hw):
        raise ValueError("metadata export shape must match input_hw")
    frame = cast(bench_ncnn.Frame, cv2.imread(str(frame_path)))
    if frame is None:
        raise ValueError(f"cannot read {frame_path}")
    cv2.setNumThreads(1)
    frame = cast(bench_ncnn.Frame, cv2.resize(frame, (camera_hw[1], camera_hw[0])))
    class_count = RFDETR_CLASS_COUNT if framework == "rfdetr" else nanodet.CLASS_COUNT
    output_capacity = DETR_QUERIES * (class_count + bench_ncnn.BOX_COORDINATES)
    if framework == "nanodet":
        output_capacity = len(nanodet.centre_priors(input_hw)) * (
            class_count + nanodet.BOX_SIDES * nanodet.REGRESSION_BINS
        )
    output = numpy.empty(output_capacity, dtype=numpy.float32)
    engine = bench_ncnn.Engine(prefix, model, precision)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    end_to_end_ms: list[float] = []
    load_before = os.getloadavg()
    try:
        for iteration in range(bench_ncnn.DEFAULT_WARMUP + bench_ncnn.DEFAULT_ITERATIONS):
            started_s = time.perf_counter()
            image = preprocess(framework, frame, input_hw)
            preprocessed_s = time.perf_counter()
            prediction, elapsed_ms = engine.infer(image, output)
            inferred_s = time.perf_counter()
            detections = decode(framework, prediction, camera_hw, input_hw, confidence)
            bool(len(detections))
            finished_s = time.perf_counter()
            if iteration < bench_ncnn.DEFAULT_WARMUP:
                continue
            preprocess_ms.append((preprocessed_s - started_s) * bench_ncnn.MILLISECONDS_PER_SECOND)
            inference_ms.append(elapsed_ms)
            postprocess_ms.append((finished_s - inferred_s) * bench_ncnn.MILLISECONDS_PER_SECOND)
            end_to_end_ms.append((finished_s - started_s) * bench_ncnn.MILLISECONDS_PER_SECOND)
        features = int(engine.library.cz_features())
    finally:
        engine.close()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "model_id": MODEL_IDS[framework],
        "format": f"ncnn-{precision}",
        "input_hw": list(input_hw),
        "camera_hw": list(camera_hw),
        "host": platform.node(),
        "host_kind": "xcng-n1" if host_profile == "xcng" else host_profile,
        "threads": bench_ncnn.THREAD_COUNT,
        "pinned_cores": ",".join(map(str, affinity)) or "unavailable",
        "ncnn_version": bench_ncnn.NCNN_VERSION,
        "ncnn_build_flags": evidence,
        "warmup": bench_ncnn.DEFAULT_WARMUP,
        "iterations": bench_ncnn.DEFAULT_ITERATIONS,
        "inference_ms": bench_ncnn.summary(inference_ms),
        "preprocess_ms": bench_ncnn.summary(preprocess_ms),
        "postprocess_ms": bench_ncnn.summary(postprocess_ms),
        "end_to_end_ms": bench_ncnn.summary(end_to_end_ms),
        "samples_ms": list(end_to_end_ms),
        "label": "measured",
        "created_at": datetime.now(UTC).isoformat(),
        "classes": class_count,
        "confidence": confidence,
        "frame_source": str(frame_path),
        "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256((model / "model.ncnn.bin").read_bytes()).hexdigest(),
        "parameters_sha256": hashlib.sha256((model / "model.ncnn.param").read_bytes()).hexdigest(),
        "load_average_before": list(load_before),
        "load_average_after": list(os.getloadavg()),
        "cpu_feature_report_raw": features,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
        },
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_NODELIST",
            )
        },
        "measurement_notes": (
            "Mac M2 Pro a76compat, shared machine, noisy"
            if host_profile == "mac-a76compat"
            else "Neoverse-N1 stand-in; Pi estimates use YOLO calibration; op mix may differ"
        ),
        "inference_boundary": "WS4 C++ Mat copy + extractor input/extract",
        "end_to_end_boundary": (
            "model preprocess + bridge + own decode/NMS + original-frame boxes + decision"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--host-profile", choices=("mac-a76compat", "xcng", "pi5"), required=True)
    parser.add_argument("--format", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--input-hw", type=int, nargs=2, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("experiments/latency"))
    parser.add_argument(
        "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
    )
    arguments = parser.parse_args()
    if not 0 <= arguments.confidence <= 1 or any(size <= 0 for size in arguments.input_hw):
        parser.error("invalid confidence or dimensions")
    try:
        report = benchmark(
            arguments.framework,
            arguments.model,
            arguments.prefix,
            arguments.host_profile,
            arguments.format,
            tuple(arguments.input_hw),
            arguments.frame,
            confidence=arguments.confidence,
        )
        if arguments.host_profile != "pi5":
            report = bench_ncnn.apply_calibration(
                report, bench_ncnn.read_report(arguments.calibration)
            )
        print(bench_ncnn.write_report(report, arguments.output))
        estimate = report.get("estimated_pi5", report["end_to_end_ms"])
        assert isinstance(estimate, dict)
        p95_ms = bench_ncnn.number(estimate.get("end_to_end_p95_ms", estimate.get("p95")), "p95")
        print(
            f"Pi p95 {p95_ms:.2f} ms; hard <=100 "
            f"{'PASS' if p95_ms <= bench_ncnn.HARD_P95_MS else 'FAIL'}; "
            f"target <=70 {'PASS' if p95_ms <= bench_ncnn.TARGET_P95_MS else 'FAIL'}"
        )
        raise SystemExit(0 if p95_ms <= bench_ncnn.HARD_P95_MS else 1)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
