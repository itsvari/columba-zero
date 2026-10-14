"""Compare native predictions against Ultralytics on C1 splits or local images."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast
from unittest.mock import patch

import numpy
import yaml
from numpy.typing import NDArray

from app.inference import Prediction
from app.models import Frame
from app.native_detector import MetadataValue
from tools.compare_exports import (
    IMAGE_SUFFIXES,
    MAXIMUM_MATCH_CONFIDENCE_DELTA,
    MINIMUM_MATCH_INTERSECTION_OVER_UNION,
    ExportPrediction,
    MatchedDetection,
    match_predictions,
)


class Description(Protocol):
    @property
    def input_hw(self) -> tuple[int, int]: ...
    @property
    def names(self) -> dict[int, str]: ...
    @property
    def decode(self) -> str: ...


class NativeModel(Description, Protocol):
    def detect(self, frame: Frame, conf: float) -> NDArray[numpy.float32]: ...
    def detect_with_timings(
        self, frame: Frame, conf: float
    ) -> tuple[NDArray[numpy.float32], dict[str, float]]: ...


@dataclass(frozen=True)
class ModelDescription:
    input_hw: tuple[int, int]
    names: dict[int, str]
    decode: str
    quantisation: str = "fp32"


class NativeOutput(TypedDict):
    input_hw: list[int]
    decode: str
    predictions: list[list[ExportPrediction]]


def describe_package(model: Path) -> ModelDescription:
    metadata = cast(
        dict[str, MetadataValue], yaml.safe_load((model / "metadata.yaml").read_text("utf-8"))
    )
    input_hw = metadata.get("imgsz")
    if (
        not isinstance(input_hw, list)
        or len(input_hw) != 2
        or not all(isinstance(side, int) for side in input_hw)
    ):
        raise ValueError("invalid export imgsz")
    names = metadata.get("names")
    if not isinstance(names, dict):
        raise ValueError("names must be a mapping")
    columba = metadata.get("columba", {})
    if not isinstance(columba, dict):
        raise ValueError("columba must be a mapping")
    return ModelDescription(
        (cast(int, input_hw[0]), cast(int, input_hw[1])),
        {int(index): str(name) for index, name in names.items()},
        str(columba.get("decode", "yolo11")),
        str(columba.get("quantization", "fp32")),
    )


@contextmanager
def ultralytics_package_path(path: Path) -> Iterator[Path]:
    if not path.is_dir() or path.name.endswith("_ncnn_model"):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="cz-ws8-ncnn-") as directory:
        compatible = Path(directory) / "model_ncnn_model"
        compatible.symlink_to(path.resolve(), target_is_directory=True)
        yield compatible


class RuntimeOptions(Protocol):
    num_threads: int
    use_fp16_packed: bool
    use_fp16_storage: bool
    use_fp16_arithmetic: bool


class RuntimeNetwork(Protocol):
    opt: RuntimeOptions


class NcnnBackend(Protocol):
    net: RuntimeNetwork


class AutoBackend(Protocol):
    backend: NcnnBackend


class Predictor(Protocol):
    model: AutoBackend


class ReferenceModel(Protocol):
    predictor: Predictor

    def predict(
        self,
        source: Frame,
        *,
        imgsz: tuple[int, int],
        conf: float,
        device: str,
        rect: bool,
        iou: float,
        verbose: bool,
        max_det: int,
        nms: bool,
    ) -> Sequence[Prediction]: ...


def load_reference(
    path: Path, description: ModelDescription, frame: Frame, confidence: float, threads: int
) -> ReferenceModel:
    from ultralytics import YOLO

    if not path.is_dir():
        return cast(ReferenceModel, YOLO(str(path), task="detect"))
    if description.decode == "yolo26":
        raise ValueError("one-to-one parity requires the original PyTorch checkpoint")
    import ncnn

    original_network = cast(Callable[[], RuntimeNetwork], ncnn.Net)

    def network_with_options() -> RuntimeNetwork:
        network = original_network()
        network.opt.num_threads = threads
        network.opt.use_fp16_packed = description.quantisation == "fp16"
        network.opt.use_fp16_storage = description.quantisation == "fp16"
        network.opt.use_fp16_arithmetic = False
        return network

    with (
        ultralytics_package_path(path) as compatible,
        patch.object(ncnn, "Net", network_with_options),
    ):
        reference = cast(ReferenceModel, YOLO(str(compatible), task="detect"))
        predict_reference(reference, frame, description, confidence)
    return reference


class ManifestRecord(TypedDict):
    id: str
    file: str
    split: str


def dataset_images(dataset: Path, split: str) -> list[Path]:
    manifest_path = dataset / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"C1 dataset has no manifest.jsonl: {dataset}")
    records = [
        cast(ManifestRecord, json.loads(line))
        for line in manifest_path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    paths: list[Path] = []
    for record in records:
        if record["split"] != split:
            continue
        path = (dataset / record["file"]).resolve()
        if not path.is_relative_to(dataset.resolve()) or not path.is_file():
            raise ValueError(f"invalid manifest image: {record['file']}")
        paths.append(path)
    return paths


def predict_reference(
    model: ReferenceModel, frame: Frame, native: Description, confidence: float
) -> Sequence[Prediction]:
    return model.predict(
        frame,
        imgsz=native.input_hw,
        conf=confidence,
        device="cpu",
        rect=False,
        iou=0.7,
        verbose=False,
        max_det=300,
        nms=native.decode != "yolo26",
    )


def reference_predictions(
    model: ReferenceModel, frame: Frame, native: Description, confidence: float
) -> list[ExportPrediction]:
    result = predict_reference(model, frame, native, confidence)[0]
    return [
        {
            "label": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "box": box.xyxy[0].tolist(),
        }
        for box in result.boxes or ()
    ]


def native_predictions(
    native: NativeModel, frame: Frame, confidence: float
) -> list[ExportPrediction]:
    return [
        {
            "label": native.names[int(row[5])],
            "confidence": float(row[4]),
            "box": [float(coordinate) for coordinate in row[:4]],
        }
        for row in native.detect(frame, confidence)
    ]


def match_clipped_boxes(
    expected: Sequence[ExportPrediction], actual: Sequence[ExportPrediction]
) -> list[MatchedDetection]:
    matches = match_predictions(expected, actual)
    matched_expected = {match["pt_index"] for match in matches}
    matched_actual = {match["ncnn_index"] for match in matches}
    for expected_index, prediction in enumerate(expected):
        box = prediction["box"]
        if expected_index in matched_expected or (box[2] > box[0] and box[3] > box[1]):
            continue
        for actual_index, candidate in enumerate(actual):
            confidence_delta = abs(prediction["confidence"] - candidate["confidence"])
            expected_coordinates = numpy.asarray(box, dtype=numpy.float32)
            actual_coordinates = numpy.asarray(candidate["box"], dtype=numpy.float32)
            coordinate_tolerance = numpy.maximum(
                numpy.spacing(expected_coordinates), numpy.spacing(actual_coordinates)
            )
            coordinates_match = bool(
                numpy.all(
                    numpy.abs(expected_coordinates - actual_coordinates) <= coordinate_tolerance
                )
            )
            if (
                actual_index in matched_actual
                or prediction["label"] != candidate["label"]
                or not coordinates_match
                or confidence_delta > MAXIMUM_MATCH_CONFIDENCE_DELTA
            ):
                continue
            matches.append(
                {
                    "pt_index": expected_index,
                    "ncnn_index": actual_index,
                    "iou": 0.0,
                    "confidence_delta": confidence_delta,
                    "box_delta_px": [
                        actual_coordinate - expected_coordinate
                        for expected_coordinate, actual_coordinate in zip(
                            box, candidate["box"], strict=True
                        )
                    ],
                }
            )
            matched_actual.add(actual_index)
            break
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="original .pt checkpoint")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="C1 dataset root with manifest.jsonl")
    source.add_argument(
        "--images", type=Path, help="directory of local real images, searched recursively"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-only", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    import cv2

    cv2.setNumThreads(1)
    images = (
        dataset_images(arguments.dataset, arguments.split)
        if arguments.dataset
        else sorted(
            path for path in arguments.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not images:
        raise SystemExit("no images to compare")
    if arguments.native_only:
        import columba_native

        native = cast(
            NativeModel, columba_native.Detector(str(arguments.model.resolve()), arguments.threads)
        )
        predictions = []
        for image in images:
            frame = cast(Frame | None, cv2.imread(str(image)))
            if frame is None:
                raise ValueError(f"cannot read image: {image}")
            predictions.append(native_predictions(native, frame, arguments.confidence))
        arguments.output.write_text(
            json.dumps(
                {
                    "input_hw": native.input_hw,
                    "decode": native.decode,
                    "predictions": predictions,
                }
            ),
            encoding="utf-8",
        )
        return
    with tempfile.TemporaryDirectory(prefix="cz-ws8-parity-") as temporary_directory:
        native_output = Path(temporary_directory) / "native.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "native.verify_native",
                *sys.argv[1:],
                "--native-only",
                "--output",
                str(native_output),
            ],
            check=True,
        )
        actual_predictions = cast(NativeOutput, json.loads(native_output.read_text("utf-8")))
    import torch

    torch.set_num_threads(arguments.threads)
    description = describe_package(arguments.model)
    first_frame = cast(Frame, cv2.imread(str(images[0])))
    reference = load_reference(
        arguments.reference.resolve(),
        description,
        first_frame,
        arguments.confidence,
        arguments.threads,
    )
    records = []
    for image, actual in zip(images, actual_predictions["predictions"], strict=True):
        frame = cast(Frame | None, cv2.imread(str(image)))
        if frame is None:
            raise ValueError(f"cannot read image: {image}")
        expected = reference_predictions(reference, frame, description, arguments.confidence)
        matches = match_clipped_boxes(expected, actual)
        records.append(
            {
                "image": str(image),
                "reference_count": len(expected),
                "native_count": len(actual),
                "reference": expected,
                "native": actual,
                "matches": matches,
                "degenerate_matches": sum(match["iou"] == 0.0 for match in matches),
                "passed": len(matches) == len(expected) == len(actual),
            }
        )
    failures = sum(not record["passed"] for record in records)
    report = {
        "label": "measured",
        "host": platform.node(),
        "architecture": platform.machine(),
        "model": str(arguments.model.resolve()),
        "reference": str(arguments.reference.resolve()),
        "reference_backend": "Ultralytics NCNN"
        if arguments.reference.is_dir()
        else "Ultralytics PyTorch",
        "dataset": str(arguments.dataset or arguments.images),
        "split": arguments.split,
        "input_hw": description.input_hw,
        "decode": description.decode,
        "images": len(images),
        "failures": failures,
        "confidence": arguments.confidence,
        "min_iou": MINIMUM_MATCH_INTERSECTION_OVER_UNION,
        "max_confidence_delta": MAXIMUM_MATCH_CONFIDENCE_DELTA,
        "degenerate_box_rule": (
            "zero-area clipped boxes require coordinates within one float32 ULP and equal class"
        ),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("host", "decode", "input_hw", "images", "failures")}
        )
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
