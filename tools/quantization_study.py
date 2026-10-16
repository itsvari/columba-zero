#!/usr/bin/env python
"""Run reproducible ARM-only NCNN PTQ comparisons without changing deployment."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

from app.json_data import JsonValue
from app.models import Frame
from tools.bench_ncnn import read_report
from tools.evaluate_predictions import load_dataset, mapping, number, sidecar_path
from tools.predict_to_coco import letterbox, model_settings
from tools.quantize_ncnn import calibration_images, digest, layer_inventory
from tools.select_model import (
    accuracy_gate,
    affinity_cores,
    benchmark_native,
    native_predictions,
    parity,
    score,
    write_json,
)

THREAD_COUNT = 4
EQUALISATION_TIMEOUT_SECONDS = 900
FP16_REFERENCE_P95_MS = 36.543667408497654
CLEAR_SPEED_RATIO = 0.95


def require_arm() -> None:
    if platform.machine() != "aarch64" or platform.node().split(".")[0] not in ("xcng0", "xcng1"):
        raise ValueError("study must run in an allocated xcng ARM CPU job")
    if len(affinity_cores()) != THREAD_COUNT or "SLURM_JOB_ID" not in os.environ:
        raise ValueError("study requires exactly four allocated, pinned CPU cores")
    if os.environ.get("OMP_WAIT_POLICY") != "ACTIVE":
        raise ValueError("study latency requires OMP_WAIT_POLICY=ACTIVE before NCNN loads")


def validate_reference(reference: Path) -> None:
    for split in ("val", "test"):
        metadata = read_report(sidecar_path(reference / f"predictions_{split}.json"))
        if (
            metadata.get("device") != "cpu"
            or metadata.get("batch") != 1
            or metadata.get("format") != "pt"
            or metadata.get("input_hw") != [448, 800]
            or metadata.get("decode") != "yolo11"
        ):
            raise ValueError(
                "study requires the published rectangular CPU/batch-1 PyTorch reference"
            )


def compare_reference(directory: Path, reference: Path, frozen_test: Path) -> None:
    validate_reference(reference)
    precision = str(
        read_report(sidecar_path(directory / "predictions_test.json"))["format"]
    ).removeprefix("ncnn-")
    write_json(directory / "accuracy_gate.json", accuracy_gate(reference, directory))
    write_json(
        directory / "parity.json",
        parity(
            reference / "predictions_test.json",
            directory / "predictions_test.json",
            precision,
            [image.image_id for image in load_dataset(frozen_test, "test").images],
        ),
    )


def audit_preprocessing(arguments: argparse.Namespace) -> None:
    import columba_native
    import cv2

    cv2.setNumThreads(1)
    settings = model_settings(arguments.model, None, None)
    selected = calibration_images(
        load_dataset(arguments.dataset, "val"),
        load_dataset(arguments.test_dataset, "test"),
        arguments.count,
        42,
    )
    maximum_pixel_delta = 0
    maximum_tensor_delta = 0.0
    input_histogram = np.zeros(256, dtype=np.int64)
    padding_channel_values = 0
    images: list[JsonValue] = []
    for image in selected:
        frame = cv2.imread(str(image.path))
        if frame is None:
            raise ValueError(f"cannot read {image.path}")
        tensor, _ = letterbox(cast(Frame, frame), settings.input_hw)
        image_height_px, image_width_px = frame.shape[:2]
        scale = min(settings.input_hw[0] / image_height_px, settings.input_hw[1] / image_width_px)
        content_area_px = round(image_height_px * scale) * round(image_width_px * scale)
        padding_channel_values += (
            settings.input_hw[0] * settings.input_hw[1] - content_area_px
        ) * 3
        padded = np.rint(tensor.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        native_padded = np.asarray(columba_native._letterbox(frame, *settings.input_hw))
        input_histogram += np.bincount(native_padded.reshape(-1), minlength=len(input_histogram))
        pixel_delta = int(np.max(np.abs(padded.astype(np.int16) - native_padded.astype(np.int16))))
        native_tensor = native_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255
        tensor_delta = float(np.max(np.abs(tensor - native_tensor)))
        maximum_pixel_delta = max(maximum_pixel_delta, pixel_delta)
        maximum_tensor_delta = max(maximum_tensor_delta, tensor_delta)
        images.append(
            {"id": image.image_id, "sha256": digest(image.path), "pixel_delta": pixel_delta}
        )
    write_json(
        arguments.output / "preprocessing.json",
        {
            "label": "measured",
            "host": platform.node(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "count": len(selected),
            "input_hw": list(settings.input_hw),
            "maximum_pixel_delta": maximum_pixel_delta,
            "maximum_tensor_delta": maximum_tensor_delta,
            "passed": maximum_pixel_delta == 0 and maximum_tensor_delta == 0,
            "test_overlap_sha256": False,
            "input_rgb_byte_histogram": [int(value) for value in input_histogram],
            "padding_channel_values": padding_channel_values,
            "images": images,
        },
    )
    if maximum_pixel_delta or maximum_tensor_delta:
        raise ValueError("calibration preprocessing differs from deployed native preprocessing")


def calibrate(
    arguments: argparse.Namespace,
    variant: str,
    method: str,
    count: int,
    *,
    table_source: Path | None = None,
    float_layers: tuple[str, ...] = (),
    no_resize: bool = False,
    activation_ranges: tuple[tuple[str, float], ...] = (),
) -> Path | None:
    package = Path(arguments.output) / "packages" / f"{variant}_ncnn_model"
    if package.exists():
        provenance = read_report(package / "calibration.json")
        expected: dict[str, JsonValue] = {
            "algorithm": method,
            "count": count,
            "float_layers": list(float_layers),
            "activation_absolute_maxima": dict(activation_ranges),
            "source_weights_sha256": digest(arguments.model / "model.ncnn.bin"),
            "table_source_sha256": digest(table_source) if table_source else None,
        }
        defaults: dict[str, JsonValue] = {"activation_absolute_maxima": {}}
        if any(provenance.get(key, defaults.get(key)) != value for key, value in expected.items()):
            raise ValueError(f"cached package differs from requested variant: {package}")
        return package
    command = [
        sys.executable,
        "-m",
        "tools.quantize_ncnn",
        str(arguments.model),
        "--output",
        str(package),
        "--dataset",
        str(arguments.dataset),
        "--test-dataset",
        str(arguments.test_dataset),
        "--prefix",
        str(arguments.prefix),
        "--method",
        method,
        "--count",
        str(count),
    ]
    if table_source:
        command.extend(["--table-source", str(table_source)])
    if float_layers:
        command.extend(["--float-layers", *float_layers])
    if no_resize:
        command.append("--no-resize")
    if method == "eq":
        command.extend(["--timeout-seconds", str(EQUALISATION_TIMEOUT_SECONDS)])
    for name, maximum in activation_ranges:
        command.extend(["--activation-range", name, str(maximum)])
    logs = arguments.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started_s = time.monotonic()
    failure: str | None = None
    with (logs / f"{variant}.log").open("w") as log:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            failure = f"calibration exceeded {EQUALISATION_TIMEOUT_SECONDS} seconds"
        except subprocess.CalledProcessError as error:
            failure = f"calibration exited {error.returncode}; see logs/{variant}.log"
    write_json(
        arguments.output / "receipts" / f"{variant}.json",
        {
            "variant": variant,
            "method": method,
            "count": count,
            "float_layers": list(float_layers),
            "elapsed_seconds": time.monotonic() - started_s,
            "failure": failure,
            "host": platform.node(),
            "label": "measured",
            "job_id": os.environ["SLURM_JOB_ID"],
        },
    )
    return None if failure else package


def evaluate_variant(arguments: argparse.Namespace, variant: str, package: Path) -> None:
    directory = arguments.output / "results" / variant
    if (directory / "summary.json").exists():
        summary = read_report(directory / "summary.json")
        if summary["parameters_sha256"] != digest(package / "model.ncnn.param") or summary[
            "weights_sha256"
        ] != digest(package / "model.ncnn.bin"):
            raise ValueError(f"cached results differ from requested model: {directory}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for split, dataset in (("val", arguments.dataset), ("test", arguments.test_dataset)):
        predictions = directory / f"predictions_{split}.json"
        if not predictions.exists():
            native_predictions(
                argparse.Namespace(
                    model=package,
                    dataset=dataset,
                    split=split,
                    out=predictions,
                    model_id=f"ws7-quant/{variant}",
                )
            )
    score(directory, arguments.dataset, arguments.test_dataset)
    compare_reference(directory, arguments.reference, arguments.test_dataset)
    validation = read_report(directory / "metrics_val.json")
    threshold = mapping(validation["operating_threshold"], "threshold")["value"]
    benchmark_native(
        argparse.Namespace(
            model=package,
            model_id=f"ws7-quant/{variant}",
            dataset=arguments.dataset,
            prefix=arguments.prefix,
            confidence=float(cast(float, threshold)),
            calibration=arguments.calibration,
            out=directory / "latency.json",
        )
    )
    latency = read_report(directory / "latency.json")
    estimate = mapping(latency["estimated_pi5"], "Pi estimate")
    estimated_p95_ms = number(estimate["end_to_end_p95_ms"], "Pi p95")
    test = read_report(directory / "metrics_test.json")
    write_json(
        directory / "summary.json",
        {
            "variant": variant,
            "host": platform.node(),
            "label": "measured",
            "val": mapping(validation["pigeon"], "val"),
            "test": mapping(test["pigeon"], "test"),
            "threshold": threshold,
            "val_empty_far": validation["empty_false_alarm_rate"],
            "val_confuser_far": validation["confuser_false_alarm_rate"],
            "test_empty_far": test["empty_false_alarm_rate"],
            "test_confuser_far": test["confuser_false_alarm_rate"],
            "estimated_pi_p95_ms": estimated_p95_ms,
            "clearly_faster_than_historical_fp16": estimated_p95_ms
            < FP16_REFERENCE_P95_MS * CLEAR_SPEED_RATIO,
            "parameters_sha256": digest(package / "model.ncnn.param"),
            "weights_sha256": digest(package / "model.ncnn.bin"),
            "layers": cast(list[JsonValue], layer_inventory(package / "model.ncnn.param")),
        },
    )
    print(f"completed {variant}", flush=True)


def scan_groups(model: Path) -> dict[str, tuple[str, ...]]:
    inventory = layer_inventory(model / "model.ncnn.param")
    names = tuple(str(layer["name"]) for layer in inventory)
    groups: dict[str, tuple[str, ...]] = {"first": ("conv_3",)}
    for start, stop in ((4, 15), (15, 27), (27, 39), (39, 51), (51, 63), (63, 79)):
        groups[f"body-{start}-{stop - 1}"] = tuple(
            name for name in names if name.startswith("conv_") and start <= int(name[5:]) < stop
        )
    groups["box"] = tuple(
        name for name in names if name.startswith("conv_") and 79 <= int(name[5:]) <= 87
    )
    groups["class"] = tuple(
        name
        for name in names
        if (name.startswith("conv_") and 88 <= int(name[5:]) <= 96)
        or (name.startswith("convdw_") and 212 <= int(name[7:]) <= 217)
    )
    accounted = {name for group in groups.values() for name in group}
    groups["other-depthwise"] = tuple(name for name in names if name not in accounted)
    return {name: layers for name, layers in groups.items() if layers}


def head_variants(model: Path) -> dict[str, tuple[str, ...]]:
    groups = scan_groups(model)
    return {
        "head": groups["box"] + groups["class"],
        "terminal-class": ("conv_90", "conv_93", "conv_96"),
        "terminal-box": ("conv_81", "conv_84", "conv_87"),
        "terminal-head": ("conv_81", "conv_84", "conv_87", "conv_90", "conv_93", "conv_96"),
        "class-p3": ("conv_90",),
        "class-p4": ("conv_93",),
        "class-p5": ("conv_96",),
    }


def run_study(arguments: argparse.Namespace) -> None:
    require_arm()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "audit":
        audit_preprocessing(arguments)
        return
    validate_reference(arguments.reference)
    if arguments.phase == "recompare":
        for summary_path in sorted((arguments.output / "results").glob("*/summary.json")):
            compare_reference(summary_path.parent, arguments.reference, arguments.test_dataset)
        return
    if arguments.phase == "calibration":
        for method in arguments.methods:
            for count in arguments.counts:
                variant = f"{method}-{count}"
                package = calibrate(arguments, variant, method, count)
                if package:
                    evaluate_variant(arguments, variant, package)
        return
    if arguments.phase == "reference":
        evaluate_variant(arguments, arguments.variant, arguments.model)
        return
    if arguments.phase == "optimise-reference":
        package = arguments.output / "packages" / "optimised-fp32_ncnn_model"
        if not package.exists():
            package.mkdir(parents=True)
            subprocess.run(
                [
                    str(arguments.prefix / "bin/ncnnoptimize"),
                    str(arguments.model / "model.ncnn.param"),
                    str(arguments.model / "model.ncnn.bin"),
                    str(package / "model.ncnn.param"),
                    str(package / "model.ncnn.bin"),
                    "0",
                ],
                check=True,
            )
            shutil.copy2(arguments.model / "metadata.yaml", package / "metadata.yaml")
        evaluate_variant(arguments, "optimised-fp32", package)
        return
    if arguments.phase == "convert":
        package = calibrate(
            arguments,
            arguments.variant,
            arguments.method,
            arguments.count,
            table_source=arguments.table_source,
            float_layers=tuple(arguments.float_layers),
            no_resize=arguments.no_resize,
            activation_ranges=tuple(
                (name, float(value)) for name, value in arguments.activation_range
            ),
        )
        if package:
            evaluate_variant(arguments, arguments.variant, package)
        return
    groups = (
        head_variants(arguments.model)
        if arguments.phase == "heads"
        else scan_groups(arguments.model)
    )
    write_json(
        arguments.output / f"{arguments.phase}_groups.json",
        cast(dict[str, JsonValue], {name: list(layers) for name, layers in groups.items()}),
    )
    for name, layers in groups.items():
        variant = f"{arguments.method}-{arguments.count}-float-{name}"
        package = calibrate(
            arguments,
            variant,
            arguments.method,
            arguments.count,
            table_source=arguments.table_source,
            float_layers=layers,
        )
        if package:
            evaluate_variant(arguments, variant, package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "audit",
            "calibration",
            "scan",
            "heads",
            "convert",
            "reference",
            "optimise-reference",
            "recompare",
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--counts", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--method", choices=("kl", "aciq", "eq"), default="kl")
    parser.add_argument(
        "--methods", nargs="+", choices=("kl", "aciq", "eq"), default=["kl", "aciq"]
    )
    parser.add_argument("--table-source", type=Path)
    parser.add_argument("--float-layers", nargs="*", default=[])
    parser.add_argument("--variant", default="variant")
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument(
        "--activation-range", nargs=2, action="append", default=[], metavar=("LAYER", "ABSMAX")
    )
    arguments = parser.parse_args()
    try:
        run_study(arguments)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
