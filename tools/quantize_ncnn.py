#!/usr/bin/env python
"""Calibrate NCNN int8 with deterministic, letterboxed validation images only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.evaluate_predictions import Dataset, ImageRecord, load_dataset, mapping
from tools.predict_to_coco import letterbox, model_settings

CALIBRATION_IMAGES = 200
CALIBRATION_SEED = 42
THREAD_COUNT = 4
OPTIMISE_FP32 = "0"
CalibrationMethod = Literal["kl", "aciq", "eq"]
QUANTISABLE_LAYERS = frozenset({"Convolution", "ConvolutionDepthWise", "InnerProduct"})


def layer_inventory(parameters: Path) -> list[dict[str, JsonValue]]:
    inventory: list[dict[str, JsonValue]] = []
    for line in parameters.read_text().splitlines()[2:]:
        fields = line.split()
        if not fields or fields[0] not in QUANTISABLE_LAYERS:
            continue
        input_count, output_count = int(fields[2]), int(fields[3])
        options = dict(field.split("=", 1) for field in fields[4 + input_count + output_count :])
        inventory.append(
            {
                "name": fields[1],
                "kind": fields[0],
                "inputs": cast(list[JsonValue], fields[4 : 4 + input_count]),
                "outputs": cast(
                    list[JsonValue], fields[4 + input_count : 4 + input_count + output_count]
                ),
                "output_channels": int(options["0"]),
                "int8_scale_term": int(options.get("8", "0")),
                "quantised": int(options.get("8", "0")) != 0,
            }
        )
    return inventory


def filter_table(source: Path, destination: Path, float_layers: tuple[str, ...]) -> None:
    lines = source.read_text().splitlines()
    names = {line.split()[0].split("_param_", 1)[0] for line in lines if line.strip()}
    unknown = set(float_layers) - names
    if unknown:
        raise ValueError(f"float layers absent from calibration table: {sorted(unknown)}")
    retained = [
        line
        for line in lines
        if line.strip() and line.split()[0].split("_param_", 1)[0] not in float_layers
    ]
    if not retained:
        raise ValueError("at least one calibrated layer must remain quantised")
    destination.write_text("\n".join(retained) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_images(
    validation: Dataset, frozen_test: Dataset, count: int, seed: int
) -> tuple[ImageRecord, ...]:
    if count < 1 or count > len(validation.images):
        raise ValueError("calibration count must fit the validation split")
    test_hashes = {digest(image.path) for image in frozen_test.images}
    selected = tuple(
        sorted(
            validation.images,
            key=lambda image: hashlib.sha256(f"{seed}/{image.image_id}".encode()).hexdigest(),
        )[:count]
    )
    if any(digest(image.path) in test_hashes for image in selected):
        raise ValueError("calibration image overlaps frozen test by SHA-256")
    return selected


def quantize_package(
    model: Path,
    destination: Path,
    validation: Path,
    frozen_test: Path,
    prefix: Path,
    *,
    count: int = CALIBRATION_IMAGES,
    seed: int = CALIBRATION_SEED,
    require_arm: bool = True,
    method: CalibrationMethod = "kl",
    table_source: Path | None = None,
    float_layers: tuple[str, ...] = (),
    no_resize: bool = False,
    timeout_seconds: int | None = None,
) -> Path:
    if require_arm and not (
        platform.machine() == "aarch64" and platform.node().split(".")[0] in ("xcng0", "xcng1")
    ):
        raise ValueError("deployment calibration must run in an xcng ARM allocation")
    settings = model_settings(model, None, None)
    if settings.format != "ncnn-fp32":
        raise ValueError("calibrate from the fp32 export, before fp16 weight rounding")
    if method not in ("kl", "aciq", "eq"):
        raise ValueError("calibration method must be kl, aciq or eq")
    selected = calibration_images(
        load_dataset(validation, "val"), load_dataset(frozen_test, "test"), count, seed
    )
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="cz-calibration-") as temporary:
        scratch = Path(temporary)
        image_paths: list[str] = []
        for index, image in enumerate(selected):
            frame = cv2.imread(str(image.path))
            if frame is None:
                raise ValueError(f"cannot read {image.path}")
            tensor, _ = letterbox(cast(NDArray[np.uint8], frame), settings.input_hw)
            padded = np.rint(tensor.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
            image_path = scratch / f"{index:04d}.png"
            if not cv2.imwrite(str(image_path), padded):
                raise OSError(f"cannot write {image_path}")
            image_paths.append(str(image_path))
        image_list = scratch / "validation.txt"
        image_list.write_text("\n".join(image_paths) + "\n")
        parameters = scratch / "model.param"
        weights = scratch / "model.bin"
        table = scratch / "model.table"
        output = scratch / "package"
        output.mkdir()
        commands = [
            [
                str(prefix / "bin/ncnnoptimize"),
                str(model / "model.ncnn.param"),
                str(model / "model.ncnn.bin"),
                str(parameters),
                str(weights),
                OPTIMISE_FP32,
            ],
            [
                str(prefix / "bin/ncnn2table"),
                str(parameters),
                str(weights),
                str(image_list),
                str(table),
                "mean=[0,0,0]",
                "norm=[0.003921568627,0.003921568627,0.003921568627]",
                "shape=[0,0,3]"
                if no_resize
                else f"shape=[{settings.input_hw[1]},{settings.input_hw[0]},3]",
                "pixel=RGB",
                f"thread={THREAD_COUNT}",
                f"method={method}",
            ],
            [
                str(prefix / "bin/ncnn2int8"),
                str(parameters),
                str(weights),
                str(output / "model.ncnn.param"),
                str(output / "model.ncnn.bin"),
                str(table),
            ],
        ]
        subprocess.run(commands[0], check=True)
        if table_source is None:
            subprocess.run(commands[1], check=True, timeout=timeout_seconds)
        else:
            calibration = mapping(
                json.loads((table_source.parent / "calibration.json").read_text()), "calibration"
            )
            if (
                calibration["source_weights_sha256"] != digest(model / "model.ncnn.bin")
                or calibration["validation_manifest_sha256"]
                != digest(validation / "manifest.jsonl")
                or calibration["test_manifest_sha256"] != digest(frozen_test / "manifest.jsonl")
                or calibration["count"] != count
                or calibration["seed"] != seed
                or calibration.get("algorithm", "kl") != method
            ):
                raise ValueError("reused table provenance differs from the requested calibration")
            shutil.copy2(table_source, table)
        unfiltered = scratch / "unfiltered.table"
        shutil.copy2(table, unfiltered)
        filter_table(unfiltered, table, float_layers)
        subprocess.run(commands[2], check=True)
        inventory = layer_inventory(output / "model.ncnn.param")
        if not any(layer["quantised"] for layer in inventory):
            raise ValueError("NCNN conversion produced no int8 scale terms")
        metadata = mapping(yaml.safe_load((model / "metadata.yaml").read_text()), "metadata")
        columba = mapping(metadata["columba"], "columba")
        columba["quantization"] = "int8"
        metadata["columba"] = columba
        (output / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        provenance: dict[str, JsonValue] = {
            "label": "measured",
            "host": platform.node(),
            "purpose": "deployment",
            "dataset": validation.name,
            "split": "val",
            "count": len(selected),
            "seed": seed,
            "method": f"NCNN {method.upper()}",
            "algorithm": method,
            "float_layers": list(float_layers),
            "table_source_sha256": digest(table_source) if table_source else None,
            "table_sha256": digest(table),
            "calibrator_shape": [0, 0, 3]
            if no_resize
            else [settings.input_hw[1], settings.input_hw[0], 3],
            "preprocessing": "centred OpenCV letterbox, RGB /255, pad 114",
            "input_hw": list(settings.input_hw),
            "test_overlap_sha256": False,
            "source_weights_sha256": digest(model / "model.ncnn.bin"),
            "test_manifest_sha256": digest(frozen_test / "manifest.jsonl"),
            "validation_manifest_sha256": digest(validation / "manifest.jsonl"),
            "images": [{"id": image.image_id, "sha256": digest(image.path)} for image in selected],
            "tool_sha256": {
                name: digest(prefix / "bin" / name)
                for name in ("ncnnoptimize", "ncnn2table", "ncnn2int8")
            },
        }
        (output / "calibration.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (output / "layers.json").write_text(json.dumps(inventory, indent=2) + "\n")
        shutil.copy2(table, output / "calibration.table")
        shutil.move(str(output), destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="C1 training dataset; val only")
    parser.add_argument("--test-dataset", type=Path, required=True, help="overlap exclusion only")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--count", type=int, default=CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--method", choices=("kl", "aciq", "eq"), default="kl")
    parser.add_argument(
        "--table-source", type=Path, help="reuse a table with verified calibration provenance"
    )
    parser.add_argument("--float-layers", nargs="*", default=[])
    parser.add_argument("--timeout-seconds", type=int, help="bound the calibration subprocess")
    parser.add_argument(
        "--no-resize", action="store_true", help="pass shape=[0,0,3] for already letterboxed PNGs"
    )
    arguments = parser.parse_args()
    try:
        print(
            quantize_package(
                arguments.model,
                arguments.output,
                arguments.dataset,
                arguments.test_dataset,
                arguments.prefix,
                count=arguments.count,
                seed=arguments.seed,
                method=arguments.method,
                table_source=arguments.table_source,
                float_layers=tuple(arguments.float_layers),
                no_resize=arguments.no_resize,
                timeout_seconds=arguments.timeout_seconds,
            )
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
