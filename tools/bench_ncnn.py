"""Measure custom-build NCNN and gate estimated Pi 5 end-to-end p95.

Build first with tools/build_ncnn_a76.sh. No pip ncnn kernels are used.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, cast

import numpy
import yaml
from numpy.typing import NDArray

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue

InferenceValues: TypeAlias = NDArray[numpy.float32]
Frame: TypeAlias = NDArray[numpy.uint8]
NCNN_VERSION = "1.0.20260526"
NCNN_COMMIT = "e54f7b1f88434e1d844ea0551b880a1cfb079ce1"
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20
THREAD_COUNT = 4
MILLISECONDS_PER_SECOND = 1000.0
TARGET_P95_MS = 70.0
HARD_P95_MS = 100.0
PAD_VALUE = 114
COLOUR_CHANNELS = 3
BOX_COORDINATES = 4
MAXIMUM_DETECTIONS = 300
MAXIMUM_NMS_CANDIDATES = 30000
DETECTION_STRIDES = (8, 16, 32)
PIXEL_MAXIMUM = 255.0
NMS_IOU = 0.7
DEFAULT_CONFIDENCE = 0.5


def read_report(path: Path) -> dict[str, JsonValue]:
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def number(value: JsonValue, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def summary(values_ms: Sequence[float]) -> dict[str, JsonValue]:
    if not values_ms or any(not math.isfinite(value) or value < 0 for value in values_ms):
        raise ValueError("timings must be a nonempty sequence of finite nonnegative values")
    return {
        "p50": float(numpy.percentile(values_ms, 50)),
        "p95": float(numpy.percentile(values_ms, 95)),
        "mean": statistics.mean(values_ms),
        "std": statistics.pstdev(values_ms),
    }


def calibration_factor(calibration: dict[str, JsonValue], host_kind: str) -> float:
    factors = calibration.get("host_factors")
    value = factors.get(host_kind) if isinstance(factors, dict) else None
    if value is None and host_kind == "xcng-n1":
        value = calibration.get("factor")
    if value is None:
        raise ValueError(f"calibration has no factor for {host_kind}")
    factor = number(value, "calibration factor")
    if factor <= 0:
        raise ValueError("calibration factor must be positive")
    return factor


def apply_calibration(
    report: dict[str, JsonValue], calibration: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    factor = calibration_factor(calibration, str(report["host_kind"]))
    timings = report["end_to_end_ms"]
    if not isinstance(timings, dict):
        raise ValueError("end_to_end_ms must be an object")
    return {
        **report,
        "estimated_pi5": {
            "end_to_end_p95_ms": number(timings["p95"], "p95") * factor,
            "calibration_id": calibration["id"],
            "factor": factor,
            "label": "estimated",
        },
    }


def make_calibration(
    reference: dict[str, JsonValue],
    source: str,
    published_ms: float = 94.03,
    mac_reference: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    if not math.isfinite(published_ms) or published_ms <= 0:
        raise ValueError("published inference mean must be finite and positive")
    host_factors: dict[str, JsonValue] = {}
    reference_means: dict[str, JsonValue] = {}
    references = [reference] if mac_reference is None else [reference, mac_reference]
    for measurement in references:
        if (
            measurement.get("model_id") != "yolo11n-coco"
            or measurement.get("format") != "ncnn-fp32"
            or measurement.get("input_hw") != [640, 640]
        ):
            raise ValueError("calibration requires COCO YOLO11n FP32 at 640x640")
        timings = measurement.get("inference_ms")
        if not isinstance(timings, dict):
            raise ValueError("calibration requires inference_ms mean")
        mean_ms = number(timings["mean"], "reference inference mean")
        if mean_ms <= 0:
            raise ValueError("reference inference mean must be positive")
        host_kind = str(measurement["host_kind"])
        host_factors[host_kind] = published_ms / mean_ms
        reference_means[host_kind] = mean_ms
    if reference.get("host_kind") != "xcng-n1":
        raise ValueError("primary calibration reference must be xcng-n1")
    return {
        "id": "cal-v0-published",
        "label": "estimated",
        "factor": host_factors["xcng-n1"],
        "host_factors": host_factors,
        "method": "published Pi5 mean inference / matched-host mean inference",
        "source": source,
        "date": datetime.now(UTC).date().isoformat(),
        "published": {
            "model": "YOLO11n COCO80",
            "input_hw": [640, 640],
            "format": "NCNN FP32",
            "inference_mean_ms": published_ms,
            "ultralytics_version": "8.3.152",
            "excludes": "pre/post-processing",
        },
        "reference_inference_mean_ms": reference_means,
        "reference_host": reference["host"],
        "ncnn_version": NCNN_VERSION,
        "limitations": (
            "Single-model, single-size inference-mean calibration. Scaling the whole pipeline "
            "and fp16/int8 is an estimate; no precision-specific Pi calibration exists. "
            "Mac factor is separately anchored and noisy on a shared machine. "
            "Published NCNN version, clock, thermal state and thread count are unspecified. "
            "Reference includes Mat copy; published Ultralytics backend also constructs Mat. "
            "WS10 must replace this record with paired Pi measurements."
        ),
    }


def write_report(report: dict[str, JsonValue], directory: Path) -> Path:
    height_width = report["input_hw"]
    if not isinstance(height_width, list) or len(height_width) != 2:
        raise ValueError("input_hw must have two values")
    model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(report["model_id"]))
    host = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(report["host"]))
    filename = f"{host}-{model_id}-{report['format']}-{height_width[0]}x{height_width[1]}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def letterbox(frame: Frame, input_hw: tuple[int, int]) -> InferenceValues:
    import cv2

    height_px, width_px = input_hw
    scale = min(height_px / frame.shape[0], width_px / frame.shape[1])
    resized_hw = (round(frame.shape[0] * scale), round(frame.shape[1] * scale))
    resized = cv2.resize(frame, (resized_hw[1], resized_hw[0]), interpolation=cv2.INTER_LINEAR)
    vertical_pad_px = height_px - resized_hw[0]
    horizontal_pad_px = width_px - resized_hw[1]
    padded = cv2.copyMakeBorder(
        resized,
        vertical_pad_px // 2,
        vertical_pad_px - vertical_pad_px // 2,
        horizontal_pad_px // 2,
        horizontal_pad_px - horizontal_pad_px // 2,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * COLOUR_CHANNELS,
    )
    return (
        numpy.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=numpy.float32)
        / PIXEL_MAXIMUM
    )


def decode(
    prediction: InferenceValues, confidence: float = DEFAULT_CONFIDENCE, nms_free: bool = False
) -> InferenceValues:
    if prediction.ndim != 2 or prediction.shape[0] <= BOX_COORDINATES:
        raise ValueError("expected C8 (4+nc, N) cxcywh pixels and sigmoid scores")
    candidates = prediction[:, prediction[BOX_COORDINATES:].max(axis=0) >= confidence].T
    if not len(candidates):
        return numpy.empty((0, 6), dtype=numpy.float32)
    classes = candidates[:, BOX_COORDINATES:].argmax(axis=1)
    scores = candidates[numpy.arange(len(candidates)), classes + BOX_COORDINATES]
    order = scores.argsort()[::-1][:MAXIMUM_NMS_CANDIDATES]
    centres = candidates[:, :2]
    sizes = candidates[:, 2:BOX_COORDINATES]
    boxes = numpy.concatenate((centres - sizes / 2, centres + sizes / 2), axis=1)
    if nms_free:
        class_count = prediction.shape[0] - BOX_COORDINATES
        flattened_scores = candidates[:, BOX_COORDINATES:].reshape(-1)
        ranked = flattened_scores.argsort()[::-1][:MAXIMUM_DETECTIONS]
        ranked = ranked[flattened_scores[ranked] >= confidence]
        anchors = ranked // class_count
        return numpy.asarray(
            numpy.column_stack((boxes[anchors], flattened_scores[ranked], ranked % class_count)),
            dtype=numpy.float32,
        )
    kept: list[int] = []
    while len(order) and len(kept) < MAXIMUM_DETECTIONS:
        current = int(order[0])
        kept.append(current)
        others = order[1:]
        intersection_hw = numpy.maximum(
            0,
            numpy.minimum(boxes[current, 2:], boxes[others, 2:])
            - numpy.maximum(boxes[current, :2], boxes[others, :2]),
        )
        intersections = intersection_hw.prod(axis=1)
        areas = sizes.prod(axis=1)
        unions = areas[current] + areas[others] - intersections
        overlaps = intersections / numpy.maximum(unions, numpy.finfo(numpy.float32).eps)
        order = others[(overlaps <= NMS_IOU) | (classes[others] != classes[current])]
    return numpy.asarray(
        numpy.column_stack((boxes[kept], scores[kept], classes[kept])), dtype=numpy.float32
    )


class Engine:
    """One bridge owns one RAII network; input/output buffers belong to Python."""

    def __init__(self, prefix: Path, model: Path, precision: str) -> None:
        suffix = "dylib" if platform.system() == "Darwin" else "so"
        self.library = ctypes.CDLL(str(prefix / "lib" / f"libcolumba_bench.{suffix}"))
        self.library.cz_load.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.cz_load.restype = ctypes.c_int
        self.library.cz_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.library.cz_infer.restype = ctypes.c_int
        self.library.cz_close.argtypes = []
        self.library.cz_close.restype = None
        self.library.cz_features.argtypes = []
        self.library.cz_features.restype = ctypes.c_int
        parameters = model / "model.ncnn.param"
        weights = model / "model.ncnn.bin"
        if precision == "int8" and not re.search(
            r"^(?:Convolution|ConvolutionDepthWise|InnerProduct).*\b8=[12]\b",
            parameters.read_text(),
            re.MULTILINE,
        ):
            raise ValueError("int8 needs a quantised param/bin model, not just an option")
        status = self.library.cz_load(
            os.fsencode(parameters),
            os.fsencode(weights),
            {"fp32": 32, "fp16": 16, "int8": 8}[precision],
            THREAD_COUNT,
        )
        if status != 0:
            raise ValueError(f"NCNN load failed with status {status}")

    def infer(
        self, image: InferenceValues, output: InferenceValues
    ) -> tuple[InferenceValues, float]:
        rows, columns = ctypes.c_int(), ctypes.c_int()
        elapsed_ms = ctypes.c_double()
        status = self.library.cz_infer(
            image.ctypes.data,
            image.shape[1],
            image.shape[2],
            output.ctypes.data,
            output.size,
            ctypes.byref(rows),
            ctypes.byref(columns),
            ctypes.byref(elapsed_ms),
        )
        if status != 0:
            raise ValueError(f"NCNN inference failed with status {status}")
        return output[: rows.value * columns.value].reshape(
            rows.value, columns.value
        ), elapsed_ms.value

    def close(self) -> None:
        self.library.cz_close()


def benchmark(
    model: Path,
    prefix: Path,
    input_hw: tuple[int, int],
    host_profile: str,
    model_id: str,
    precision: str = "fp16",
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    camera_hw: tuple[int, int] = (480, 640),
    frame_path: Path | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    nms_free: bool = False,
) -> dict[str, JsonValue]:
    import cv2

    if iterations < DEFAULT_ITERATIONS or warmup < 1:
        raise ValueError("benchmark requires >=200 iterations and >=1 warm-up")
    if any(size <= 0 for size in (*input_hw, *camera_hw)) or any(size % 32 for size in input_hw):
        raise ValueError("positive dimensions required; model dimensions must be multiples of 32")
    evidence = (prefix / "build-evidence.txt").read_text()
    if f"profile={host_profile}" not in evidence or f"commit={NCNN_COMMIT}" not in evidence:
        raise ValueError("NCNN build evidence does not match host profile and pinned commit")
    metadata: JsonValue = yaml.safe_load((model / "metadata.yaml").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a mapping")
    exported_hw = metadata.get("imgsz")
    if exported_hw != list(input_hw):
        raise ValueError(f"export shape {exported_hw} differs from requested {input_hw}")
    names = metadata.get("names")
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError("metadata needs class names")
    frame = cast(Frame, cv2.imread(str(frame_path))) if frame_path else None
    if frame_path and frame is None:
        raise ValueError(f"cannot read {frame_path}")
    if frame is None:
        frame = numpy.random.default_rng(0).integers(
            0, 256, (*camera_hw, COLOUR_CHANNELS), numpy.uint8
        )
    else:
        frame = cast(Frame, cv2.resize(frame, (camera_hw[1], camera_hw[0])))
    cv2.setNumThreads(1)
    prediction_capacity = (BOX_COORDINATES + len(names)) * sum(
        (input_hw[0] // stride) * (input_hw[1] // stride) for stride in DETECTION_STRIDES
    )
    output = numpy.empty(prediction_capacity, dtype=numpy.float32)
    engine = Engine(prefix, model, precision)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    end_to_end_ms: list[float] = []
    load_before = os.getloadavg()
    try:
        for iteration in range(warmup + iterations):
            started_s = time.perf_counter()
            image = letterbox(frame, input_hw)
            preprocessed_s = time.perf_counter()
            prediction, inference_elapsed_ms = engine.infer(image, output)
            inferred_s = time.perf_counter()
            detections = decode(prediction, confidence, nms_free)
            scale = min(input_hw[0] / camera_hw[0], input_hw[1] / camera_hw[1])
            padding_xy = numpy.asarray(
                [
                    (input_hw[1] - round(camera_hw[1] * scale)) // 2,
                    (input_hw[0] - round(camera_hw[0] * scale)) // 2,
                ]
                * 2,
                dtype=numpy.float32,
            )
            detections[:, :BOX_COORDINATES] = (detections[:, :BOX_COORDINATES] - padding_xy) / scale
            detections[:, (0, 2)] = detections[:, (0, 2)].clip(0, camera_hw[1])
            detections[:, (1, 3)] = detections[:, (1, 3)].clip(0, camera_hw[0])
            bool(len(detections))
            finished_s = time.perf_counter()
            if iteration < warmup:
                continue
            preprocess_ms.append((preprocessed_s - started_s) * MILLISECONDS_PER_SECOND)
            inference_ms.append(inference_elapsed_ms)
            postprocess_ms.append((finished_s - inferred_s) * MILLISECONDS_PER_SECOND)
            end_to_end_ms.append((finished_s - started_s) * MILLISECONDS_PER_SECOND)
        cpu_features = int(engine.library.cz_features())
    finally:
        engine.close()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    weights_hash = hashlib.sha256((model / "model.ncnn.bin").read_bytes()).hexdigest()
    return {
        "model_id": model_id,
        "format": f"ncnn-{precision}",
        "input_hw": list(input_hw),
        "host": platform.node(),
        "host_kind": {"xcng": "xcng-n1", "mac-a76compat": "mac-a76compat", "pi5": "pi5"}[
            host_profile
        ],
        "threads": THREAD_COUNT,
        "pinned_cores": ",".join(map(str, affinity)) or "unavailable",
        "ncnn_version": NCNN_VERSION,
        "ncnn_build_flags": evidence,
        "warmup": warmup,
        "iterations": iterations,
        "inference_ms": summary(inference_ms),
        "preprocess_ms": summary(preprocess_ms),
        "postprocess_ms": summary(postprocess_ms),
        "end_to_end_ms": summary(end_to_end_ms),
        "samples_ms": list(end_to_end_ms),
        "label": "measured",
        "created_at": datetime.now(UTC).isoformat(),
        "camera_hw": list(camera_hw),
        "classes": len(names),
        "confidence": confidence,
        "nms_free": nms_free,
        "weights_sha256": weights_hash,
        "parameters_sha256": hashlib.sha256((model / "model.ncnn.param").read_bytes()).hexdigest(),
        "int8_calibration": (
            read_report(model / "latency-only-calibration.json")
            if (model / "latency-only-calibration.json").exists()
            else None
        ),
        "frame_source": str(frame_path)
        if frame_path
        else "seed-0 synthetic uint8 camera-shaped noise",
        "load_average_before": list(load_before),
        "load_average_after": list(os.getloadavg()),
        "cpu_feature_report_raw": cpu_features,
        "cpu_feature_report_encoding": "fp16 query + 2 * dotprod query; platform-specific values",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
        },
        "measurement_notes": (
            "Mac M2 Pro a76compat build, shared machine; relative noisy measurements"
            if host_profile == "mac-a76compat"
            else "ARM stand-in, not measured Pi latency"
        ),
        "inference_boundary": "C++ Mat copy + extractor input/extract; excludes output copy/FFI",
        "end_to_end_boundary": "letterbox/normalise + bridge + decode/NMS + decision",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    calibration_parser = subcommands.add_parser(
        "calibrate", help="anchor to official YOLO11n Pi5 FP32 640 inference mean"
    )
    calibration_parser.add_argument("--reference", type=Path, required=True)
    calibration_parser.add_argument("--mac-reference", type=Path)
    calibration_parser.add_argument("--published-ms", type=float, default=94.03)
    calibration_parser.add_argument(
        "--source",
        default=(
            "https://github.com/ultralytics/ultralytics/blob/v8.3.200/docs/en/guides/raspberry-pi.md"
        ),
    )
    calibration_parser.add_argument(
        "--output", type=Path, default=Path("experiments/latency/calibration.json")
    )
    for command in ("run", "gate"):
        command_parser = subcommands.add_parser(
            command,
            help="measure C7" if command == "run" else "PASS <=100 ms; TARGET <=70 ms",
            description="Gate estimates Pi p95 using calibration matched to host.",
        )
        command_parser.add_argument("--model", required=True, type=Path)
        command_parser.add_argument("--model-id")
        command_parser.add_argument(
            "--input-hw", required=True, type=int, nargs=2, metavar=("H", "W")
        )
        command_parser.add_argument(
            "--host-profile", required=True, choices=("xcng", "mac-a76compat", "pi5")
        )
        command_parser.add_argument(
            "--prefix", type=Path, help="custom build prefix; default native/build/ncnn-PROFILE"
        )
        command_parser.add_argument("--format", choices=("fp32", "fp16", "int8"), default="fp16")
        command_parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
        command_parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
        command_parser.add_argument("--camera-hw", type=int, nargs=2, default=(480, 640))
        command_parser.add_argument(
            "--frame", type=Path, help="camera-like source image; otherwise seed-0 noise"
        )
        command_parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
        command_parser.add_argument(
            "--nms-free", action="store_true", help="only for verified one-to-one C8 export"
        )
        command_parser.add_argument("--output", type=Path, default=Path("experiments/latency"))
        command_parser.add_argument(
            "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
        )
    arguments = parser.parse_args()
    if arguments.command == "calibrate":
        try:
            calibration = make_calibration(
                read_report(arguments.reference),
                arguments.source,
                arguments.published_ms,
                read_report(arguments.mac_reference) if arguments.mac_reference else None,
            )
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
            print(json.dumps(calibration, indent=2))
            return
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if not 0 <= arguments.confidence <= 1:
        parser.error("confidence must be in [0,1]")
    if (
        arguments.command == "gate"
        and arguments.host_profile != "pi5"
        and not arguments.calibration.exists()
    ):
        parser.error("gate requires a calibration file")
    try:
        report = benchmark(
            arguments.model,
            arguments.prefix or Path(f"native/build/ncnn-{arguments.host_profile}"),
            tuple(arguments.input_hw),
            arguments.host_profile,
            arguments.model_id or arguments.model.name,
            arguments.format,
            arguments.iterations,
            arguments.warmup,
            tuple(arguments.camera_hw),
            arguments.frame,
            arguments.confidence,
            arguments.nms_free,
        )
        if arguments.calibration.exists() and arguments.host_profile != "pi5":
            report = apply_calibration(report, read_report(arguments.calibration))
        path = write_report(report, arguments.output)
        print(path)
        if arguments.command != "gate":
            return
        estimates = report.get("estimated_pi5", report["end_to_end_ms"])
        assert isinstance(estimates, dict)
        p95_ms = number(estimates.get("end_to_end_p95_ms", estimates.get("p95")), "p95")
        label = "measured" if arguments.host_profile == "pi5" else "estimated"
        print(
            f"{label} Pi p95 {p95_ms:.2f} ms: "
            f"{'PASS' if p95_ms <= HARD_P95_MS else 'FAIL'} hard <=100 ms; "
            f"{'PASS' if p95_ms <= TARGET_P95_MS else 'FAIL'} target <=70 ms"
        )
        raise SystemExit(0 if p95_ms <= HARD_P95_MS else 1)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
