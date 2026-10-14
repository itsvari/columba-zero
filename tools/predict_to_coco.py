#!/usr/bin/env python
"""Produce C3 predictions in original pixels from PyTorch or C8 NCNN models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import yaml
from numpy.typing import NDArray

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import Prediction as UltralyticsPrediction
from app.json_data import JsonValue
from tools.evaluate_predictions import (
    Box,
    Dataset,
    ImageRecord,
    InputShape,
    Prediction,
    class_names,
    input_shape,
    integer,
    load_dataset,
    mapping,
    number,
    overlap,
    sidecar_path,
)

DEFAULT_INPUT_HW = (640, 640)
DEFAULT_CONFIDENCE = 0.001
DEFAULT_NMS_IOU = 0.7
DEFAULT_MAX_DETECTIONS = 300
MODEL_STRIDE_PX = 32
PAD_VALUE = 114
COLOUR_CHANNELS = 3
NORMALISATION_DIVISOR = 255.0
PADDING_ROUNDING_OFFSET = 0.1
MAX_THREADS = 4

PREDICTION_HELP = """
--imgsz H [W] is height then width; one number means square. For NCNN,
C8 columba.input_hw and columba.decode are authoritative defaults; an override
must agree with the package. Legacy packages use metadata.imgsz and yolo11.
PyTorch defaults to 640x640 and yolo11, which also selects YOLO26's one-to-many
head. --decode yolo26 selects the PyTorch one-to-one head (nms=False), or decodes
a C8 one-to-one NCNN head by top-k class scores without NMS. yolo11 uses
class-aware NMS at --iou. NCNN uses a raw in0/out0 loop with centred rectangular
letterboxing, RGB /255 and padding=114 as C8 specifies. Unsupported C8
preprocessing/layout is refused. NCNN is CPU-only and batch=1; --batch bounds
PyTorch batches. Dimensions must be multiples of 32.
NCNN retains its runtime's default fp16 compute flags. --ncnn-fp32 disables
fp16 packed/storage/arithmetic for close parity checks against CPU PyTorch;
it does not change the package's weight quantisation. Compute mode is recorded.
Output boxes are original-image [left,top,width,height] pixels, clipped to the
image. IDs come from manifest.jsonl; absent a manifest, unique image stems are
used and recorded in the sidecar. Class names/order must match data.yaml.
Confidence floor defaults to 0.001 and max detections to 300. The sidecar records
input_hw, decode, confidence floor, NMS, host, versions and image-ID provenance.
Set YOLO_CONFIG_DIR to job scratch before invoking; if unset this tool creates
isolated settings under outputs/ws1/ultralytics with sync=false. No downloads
are needed. Model paths are resolved absolutely and must exist.
Evaluation definitions: COCO AP50/AP50-95 and pigeon IoU>=0.5 PR metrics are
computed by evaluate_predictions.py. That tool defines small boxes, negative
image FAR, score ties and validation-only threshold selection in --help.
The legacy v1-2 wrapper directory containing best_ncnn_model is also accepted.
"""


class Checkpoint(Protocol):
    end2end: bool


class PredictModel(Protocol):
    @property
    def names(self) -> dict[int, str]: ...

    @property
    def model(self) -> Checkpoint: ...

    def predict(
        self,
        source: list[str],
        *,
        imgsz: list[int],
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        rect: bool,
        batch: int,
        verbose: bool,
        stream: bool,
        nms: bool,
    ) -> Sequence[UltralyticsPrediction]: ...


@dataclass(frozen=True)
class ModelSettings:
    input_hw: InputShape
    decode: str
    format: str
    names: tuple[str, ...] | None


@dataclass(frozen=True)
class Letterbox:
    scale: float
    left_px: int
    top_px: int
    width_px: int
    height_px: int

    def original_box(self, bbox: Box) -> Box:
        left_px = max(0.0, min(self.width_px, (bbox[0] - self.left_px) / self.scale))
        top_px = max(0.0, min(self.height_px, (bbox[1] - self.top_px) / self.scale))
        right_px = max(0.0, min(self.width_px, (bbox[0] + bbox[2] - self.left_px) / self.scale))
        bottom_px = max(0.0, min(self.height_px, (bbox[1] + bbox[3] - self.top_px) / self.scale))
        return left_px, top_px, right_px - left_px, bottom_px - top_px


def letterbox(
    frame: NDArray[np.uint8], input_hw: InputShape
) -> tuple[NDArray[np.float32], Letterbox]:
    import cv2

    height_px, width_px = frame.shape[:2]
    scale = min(input_hw[0] / height_px, input_hw[1] / width_px)
    resized_width_px, resized_height_px = round(width_px * scale), round(height_px * scale)
    horizontal_pad_px = (input_hw[1] - resized_width_px) / 2
    vertical_pad_px = (input_hw[0] - resized_height_px) / 2
    left_px = round(horizontal_pad_px - PADDING_ROUNDING_OFFSET)
    top_px = round(vertical_pad_px - PADDING_ROUNDING_OFFSET)
    resized = cv2.resize(
        frame, (resized_width_px, resized_height_px), interpolation=cv2.INTER_LINEAR
    )
    padded = cv2.copyMakeBorder(
        resized,
        top_px,
        input_hw[0] - resized_height_px - top_px,
        left_px,
        input_hw[1] - resized_width_px - left_px,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * COLOUR_CHANNELS,
    )
    tensor = (
        np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        / NORMALISATION_DIVISOR
    )
    return tensor, Letterbox(scale, left_px, top_px, width_px, height_px)


def configure_ultralytics() -> None:
    root = Path(
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("outputs/ws1/ultralytics").resolve()))
    )
    settings = root / "Ultralytics/settings.json"
    if settings.exists():
        configured = mapping(json.loads(settings.read_text()), "Ultralytics settings")
    else:
        configured = {}
    configured["sync"] = False
    for key in ("datasets_dir", "weights_dir", "runs_dir"):
        configured.setdefault(key, str((root / key.removesuffix("_dir")).resolve()))
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(configured, indent=2) + "\n")


def validate_preprocessing(columba: dict[str, JsonValue]) -> None:
    defaults: dict[str, JsonValue] = {
        "letterbox": True,
        "pad_value": PAD_VALUE,
        "color_order": "rgb",
        "normalize": {"mean": [0, 0, 0], "std": [255, 255, 255]},
        "output_layout": "1x(4+nc)xN cxcywh-px sigmoid-scores",
    }
    for key, expected in defaults.items():
        if columba.get(key, expected) != expected:
            raise ValueError(f"unsupported C8 {key}: {columba.get(key)!r}; expected {expected!r}")


def model_settings(model: Path, imgsz: Sequence[int] | None, decode: str | None) -> ModelSettings:
    if not model.exists():
        raise ValueError(f"model does not exist: {model}")
    requested = (
        input_shape([imgsz[0], imgsz[0]] if imgsz is not None and len(imgsz) == 1 else list(imgsz))
        if imgsz is not None
        else None
    )
    if model.is_file():
        if model.suffix != ".pt" or model.stat().st_size == 0:
            raise ValueError("model must be a non-empty .pt or an NCNN directory")
        settings = ModelSettings(requested or DEFAULT_INPUT_HW, decode or "yolo11", "pt", None)
    else:
        for name in ("metadata.yaml", "model.ncnn.param", "model.ncnn.bin"):
            if not (model / name).is_file() or not (model / name).stat().st_size:
                raise ValueError(f"incomplete NCNN package: {name}")
        metadata = mapping(yaml.safe_load((model / "metadata.yaml").read_text()), "metadata")
        if metadata.get("task") != "detect":
            raise ValueError("NCNN metadata must have task: detect")
        columba = mapping(metadata.get("columba", {}), "columba")
        validate_preprocessing(columba)
        exported = input_shape(columba.get("input_hw", metadata.get("imgsz")))
        if (
            "input_hw" in columba
            and metadata.get("imgsz") is not None
            and input_shape(metadata["imgsz"]) != exported
        ):
            raise ValueError("C8 input_hw disagrees with metadata.imgsz")
        if requested is not None and requested != exported:
            raise ValueError(f"requested input {requested} differs from NCNN export {exported}")
        package_decode = str(columba.get("decode", "yolo11"))
        if decode is not None and decode != package_decode:
            raise ValueError("--decode disagrees with C8 metadata/legacy yolo11 default")
        settings = ModelSettings(
            exported,
            package_decode,
            f"ncnn-{columba.get('quantization', 'fp32')}",
            class_names(metadata.get("names")),
        )
    if settings.decode not in ("yolo11", "yolo26"):
        raise ValueError(f"unsupported decode: {settings.decode}")
    if any(dimension % MODEL_STRIDE_PX for dimension in settings.input_hw):
        raise ValueError("input dimensions must be multiples of 32")
    return settings


def decode_output(
    output: NDArray[np.float32],
    image_id: str,
    names: tuple[str, ...],
    transform: Letterbox,
    decode: str,
    confidence: float,
    iou: float,
    max_det: int,
) -> tuple[Prediction, ...]:
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2 or output.shape[0] != 4 + len(names):
        raise ValueError(f"unsupported NCNN output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("NCNN output must be finite")
    scores = output[4:]
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("NCNN class scores must be sigmoid probabilities")
    if decode == "yolo26":
        categories, anchors = np.nonzero(scores >= confidence)
    else:
        best_categories = scores.argmax(axis=0)
        anchors = np.flatnonzero(scores[best_categories, np.arange(scores.shape[1])] >= confidence)
        categories = best_categories[anchors]
    ranking = np.argsort(-scores[categories, anchors], kind="stable")
    kept_input: list[tuple[int, Box]] = []
    predictions: list[Prediction] = []
    for candidate in ranking:
        category_id, anchor = int(categories[candidate]), int(anchors[candidate])
        centre_x_px, centre_y_px, width_px, height_px = (
            float(value) for value in output[:4, anchor]
        )
        if width_px <= 0 or height_px <= 0:
            continue
        bbox = (centre_x_px - width_px / 2, centre_y_px - height_px / 2, width_px, height_px)
        if decode == "yolo11" and any(
            category_id == previous_category and overlap(bbox, previous_bbox) > iou
            for previous_category, previous_bbox in kept_input
        ):
            continue
        original = transform.original_box(bbox)
        if original[2] <= 0 or original[3] <= 0:
            continue
        kept_input.append((category_id, bbox))
        predictions.append(
            Prediction(image_id, category_id, original, float(scores[category_id, anchor]))
        )
        limit = min(max_det, output.shape[1]) if decode == "yolo26" else max_det
        if len(predictions) == limit:
            break
    return tuple(predictions)


def ncnn_predictions(
    model: Path,
    dataset: Dataset,
    settings: ModelSettings,
    confidence: float,
    iou: float,
    max_det: int,
    ncnn_fp32: bool = False,
) -> Iterator[Prediction]:
    import cv2
    import ncnn

    cv2.setNumThreads(MAX_THREADS)
    with ncnn.Net() as network:
        network.opt.num_threads = MAX_THREADS
        network.opt.use_vulkan_compute = False
        if ncnn_fp32:
            network.opt.use_fp16_packed = False
            network.opt.use_fp16_storage = False
            network.opt.use_fp16_arithmetic = False
        if (
            network.load_param(str(model / "model.ncnn.param")) != 0
            or network.load_model(str(model / "model.ncnn.bin")) != 0
        ):
            raise ValueError("failed to load NCNN package")
        for image in dataset.images:
            frame = cv2.imread(str(image.path))
            if frame is None or frame.shape[:2] != (image.height_px, image.width_px):
                raise ValueError(
                    f"cannot read image or dimensions disagree with manifest: {image.path}"
                )
            tensor, transform = letterbox(np.asarray(frame, dtype=np.uint8), settings.input_hw)
            with network.create_extractor() as extractor:
                if extractor.input("in0", ncnn.Mat(tensor)) != 0:
                    raise ValueError("NCNN input blob in0 is unavailable")
                status, result = extractor.extract("out0")
                if status != 0:
                    raise ValueError("NCNN output blob out0 is unavailable")
                output = np.array(result, dtype=np.float32, copy=True)
            yield from decode_output(
                output,
                image.image_id,
                dataset.names,
                transform,
                settings.decode,
                confidence,
                iou,
                max_det,
            )


def result_predictions(result: UltralyticsPrediction, image: ImageRecord) -> Iterator[Prediction]:
    for box in result.boxes or ():
        left_px, top_px, right_px, bottom_px = box.xyxy[0].tolist()
        left_px, right_px = (max(0.0, min(image.width_px, value)) for value in (left_px, right_px))
        top_px, bottom_px = (max(0.0, min(image.height_px, value)) for value in (top_px, bottom_px))
        if right_px > left_px and bottom_px > top_px:
            yield Prediction(
                image.image_id,
                int(box.cls.item()),
                (left_px, top_px, right_px - left_px, bottom_px - top_px),
                float(box.conf.item()),
            )


def pytorch_predictions(
    model: Path,
    dataset: Dataset,
    settings: ModelSettings,
    confidence: float,
    iou: float,
    max_det: int,
    device: str,
    batch: int,
) -> Iterator[Prediction]:
    configure_ultralytics()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(MAX_THREADS)
    predictor = cast(PredictModel, YOLO(str(model), task="detect"))
    if class_names(predictor.names) != dataset.names:
        raise ValueError("model class names/order differ from dataset")
    predictor.model.end2end = settings.decode == "yolo26"
    if predictor.model.end2end != (settings.decode == "yolo26"):
        raise ValueError(f"checkpoint does not support the requested {settings.decode} head")
    for start in range(0, len(dataset.images), batch):
        images = dataset.images[start : start + batch]
        results = predictor.predict(
            [str(image.path) for image in images],
            imgsz=list(settings.input_hw),
            conf=confidence,
            iou=iou,
            max_det=max_det,
            device=device,
            rect=False,
            batch=batch,
            verbose=False,
            stream=False,
            nms=settings.decode != "yolo26",
        )
        if len(results) != len(images):
            raise ValueError("model returned a different number of images")
        for image, result in zip(images, results, strict=True):
            if class_names(result.names) != dataset.names:
                raise ValueError("prediction class names/order differ from dataset")
            yield from result_predictions(result, image)


def predict_to_coco(
    model: str | Path,
    dataset: str | Path,
    split: str,
    out: str | Path,
    *,
    imgsz: Sequence[int] | None = None,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_NMS_IOU,
    max_det: int = DEFAULT_MAX_DETECTIONS,
    device: str = "cpu",
    model_id: str | None = None,
    decode: str | None = None,
    batch: int = 1,
    host: str | None = None,
    ncnn_fp32: bool = False,
) -> dict[str, JsonValue]:
    """Write predictions plus sidecar and return the sidecar metadata."""
    model_path = Path(model).resolve()
    if model_path.is_dir() and not (model_path / "metadata.yaml").exists():
        nested = model_path / "best_ncnn_model"
        if nested.is_dir():
            model_path = nested
    settings = model_settings(model_path, imgsz, decode)
    for value, name in ((conf, "confidence"), (iou, "NMS IoU")):
        if not 0 <= number(value, name) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    integer(batch, "batch", 1)
    integer(max_det, "max_det", 1)
    if max_det > DEFAULT_MAX_DETECTIONS:
        raise ValueError("C3 max_det cannot exceed 300")
    loaded = load_dataset(Path(dataset), split, classes=None)
    if settings.names is not None and settings.names != loaded.names:
        raise ValueError("model class names/order differ from dataset")
    if model_path.is_dir():
        if device != "cpu" or batch != 1:
            raise ValueError("raw NCNN prediction supports device=cpu and batch=1 only")
        detections = tuple(
            ncnn_predictions(model_path, loaded, settings, conf, iou, max_det, ncnn_fp32)
        )
        framework = "ncnn"
    else:
        if ncnn_fp32:
            raise ValueError("--ncnn-fp32 only applies to NCNN packages")
        detections = tuple(
            pytorch_predictions(model_path, loaded, settings, conf, iou, max_det, device, batch)
        )
        framework = "ultralytics"
    metadata: dict[str, JsonValue] = {
        "model_id": model_id or model_path.stem,
        "model_path": str(model_path),
        "framework": framework,
        "format": settings.format,
        "input_hw": list(settings.input_hw),
        "decode": settings.decode,
        "conf_floor": conf,
        "nms_iou": iou if settings.decode == "yolo11" else None,
        "agnostic_nms": False,
        "max_det": max_det,
        "host": host or platform.node(),
        "dataset": loaded.name,
        "split": split,
        "created_at": datetime.now(UTC).isoformat(),
        "image_id_source": loaded.id_source,
        "images": len(loaded.images),
        "label": "measured",
        "device": device,
        "batch": batch,
        "ncnn_compute": ("fp32" if ncnn_fp32 else "runtime_default")
        if framework == "ncnn"
        else None,
    }
    from importlib.metadata import version

    metadata["versions"] = {
        package: version(package)
        for package in (("ncnn", "numpy") if framework == "ncnn" else ("ultralytics", "torch"))
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [
                {
                    "image_id": detection.image_id,
                    "category_id": detection.category_id,
                    "bbox": list(detection.bbox),
                    "score": detection.score,
                }
                for detection in detections
            ],
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    sidecar_path(output).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=PREDICTION_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", nargs="+", type=int, metavar="PX")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DETECTIONS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id")
    parser.add_argument("--decode", choices=("yolo11", "yolo26"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--host")
    parser.add_argument("--ncnn-fp32", action="store_true")
    arguments = parser.parse_args()
    try:
        predict_to_coco(
            arguments.model,
            arguments.dataset,
            arguments.split,
            arguments.out,
            imgsz=arguments.imgsz,
            conf=arguments.conf,
            iou=arguments.iou,
            max_det=arguments.max_det,
            device=arguments.device,
            model_id=arguments.model_id,
            decode=arguments.decode,
            batch=arguments.batch,
            host=arguments.host,
            ncnn_fp32=arguments.ncnn_fp32,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
