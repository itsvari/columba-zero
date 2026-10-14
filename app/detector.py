from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml

from .config import ImageSize, input_hw
from .inference import COLOUR_CHANNEL_COUNT, NMS_INTERSECTION_OVER_UNION, PredictionModel
from .json_data import JsonValue
from .models import Detection, FramePacket

DEFAULT_TRIGGER_CLASSES = ("pigeon",)
DEFAULT_DECODE = "yolo11"
DECODERS = frozenset({"yolo11", "yolo26", "dfine", "rfdetr", "nanodet"})
# Ultralytics applies NMS itself; for one-to-one (yolo26) heads that removes almost nothing.
ULTRALYTICS_DECODERS = frozenset({"yolo11", "yolo26"})
STAGE_NAMES = ("preprocess", "inference", "postprocess")
NCNN_DIRECTORY_SUFFIX = "_ncnn_model"


class DetectorError(RuntimeError):
    pass


class FrameDetector(Protocol):
    names: dict[int, str]

    def warmup(self, width: int, height: int) -> None: ...
    def detect(self, packet: FramePacket) -> list[Detection]: ...


@dataclass(frozen=True)
class ModelInfo:
    """What the deploy package says about a model (CONTRACTS.md C8).

    Packages without a `columba:` block (such as pigeon_baseline_v1-2) are read as
    `decode: yolo11` with `trigger_classes: [pigeon]`. Checkpoints carry no metadata
    file, so their names are known only after loading.
    """

    path: Path
    names: dict[int, str]
    trigger_classes: tuple[str, ...] = DEFAULT_TRIGGER_CLASSES
    decode: str = DEFAULT_DECODE
    input_hw: tuple[int, int] | None = None


def class_names(names: JsonValue | Mapping[int, str] | tuple[str, ...]) -> dict[int, str]:
    """Return lower-case names by contiguous index, with pigeon as class 0."""
    if isinstance(names, Mapping):
        try:
            indexed = {int(index): str(label).lower() for index, label in names.items()}
        except (TypeError, ValueError) as error:
            raise DetectorError("class indices must be integers") from error
    elif isinstance(names, (list, tuple)):
        indexed = {index: str(label).lower() for index, label in enumerate(names)}
    else:
        raise DetectorError("model has no class names")
    if not indexed or sorted(indexed) != list(range(len(indexed))):
        raise DetectorError("class indices must run from 0 without gaps")
    if indexed[0] != "pigeon":
        raise DetectorError("pigeon must have class index 0")
    if len(set(indexed.values())) != len(indexed):
        raise DetectorError("class names must be unique")
    return indexed


def validate_names(names: JsonValue | Mapping[int, str] | tuple[str, ...]) -> None:
    class_names(names)


def trigger_classes(value: JsonValue, names: dict[int, str]) -> tuple[str, ...]:
    if value is None:
        classes: tuple[str, ...] = DEFAULT_TRIGGER_CLASSES
    elif isinstance(value, list) and value and all(isinstance(label, str) for label in value):
        classes = tuple(str(label).lower() for label in value)
    else:
        raise ValueError("columba.trigger_classes must be a non-empty list of class names")
    unknown = set(classes) - set(names.values())
    if unknown:
        raise ValueError(f"trigger classes are not model classes: {', '.join(sorted(unknown))}")
    return classes


def read_model_info(model: Path, image_size: ImageSize) -> ModelInfo:
    metadata = yaml.safe_load((model / "metadata.yaml").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("task") != "detect":
        raise ValueError("metadata must describe a detection model")
    names = class_names(metadata.get("names", {}))
    expected_hw = list(input_hw(image_size))
    if metadata.get("imgsz") != expected_hw:
        raise ValueError(
            f"export size {metadata.get('imgsz')} differs from {expected_hw} ([h, w]); "
            "export a separate model for this size"
        )
    columba = metadata.get("columba")
    if columba is None:
        return ModelInfo(model, names, input_hw=input_hw(image_size))
    if not isinstance(columba, dict):
        raise ValueError("columba metadata must be a mapping")
    decode = columba.get("decode", DEFAULT_DECODE)
    if decode not in DECODERS:
        raise ValueError(f"unknown columba.decode {decode!r}")
    if "input_hw" in columba and columba["input_hw"] != expected_hw:
        raise ValueError(f"columba.input_hw {columba['input_hw']} differs from imgsz")
    return ModelInfo(
        model,
        names,
        trigger_classes(columba.get("trigger_classes"), names),
        str(decode),
        input_hw(image_size),
    )


def load_model_info(path: str, image_size: ImageSize) -> ModelInfo:
    model = Path(path)
    if not model.exists():
        raise DetectorError(f"model does not exist: {model}; see deploy/model_manifest.json")
    if model.is_dir():
        for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
            if not (model / name).is_file() or (model / name).stat().st_size == 0:
                raise DetectorError(f"incomplete NCNN export: missing or empty {model / name}")
        try:
            return read_model_info(model, image_size)
        except (OSError, ValueError, DetectorError, yaml.YAMLError) as error:
            raise DetectorError(f"invalid NCNN metadata: {error}") from error
    if model.suffix.lower() != ".pt" or model.stat().st_size == 0:
        raise DetectorError("model must be a non-empty .pt checkpoint or complete NCNN directory")
    return ModelInfo(model, {})


def check_model(path: str, image_size: ImageSize) -> Path:
    return load_model_info(path, image_size).path


def stage_timings(speed: object) -> dict[str, float | None]:
    """Read Ultralytics' per-result `speed` (milliseconds), ignoring anything malformed."""
    timings: dict[str, float | None] = dict.fromkeys(STAGE_NAMES)
    if isinstance(speed, dict):
        for name in STAGE_NAMES:
            value = speed.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timings[name] = float(value)
    return timings


class UltralyticsDetector:
    def __init__(self, model_path: str, image_size: ImageSize, confidence: float) -> None:
        self.model_info = load_model_info(model_path, image_size)
        self.image_size = image_size
        self.confidence = confidence
        self.last_stage_ms: dict[str, float | None] = dict.fromkeys(STAGE_NAMES)
        if self.model_info.decode not in ULTRALYTICS_DECODERS:
            raise DetectorError(f"decode {self.model_info.decode!r} needs model.backend: native")
        try:
            from ultralytics import YOLO

            self._model = cast(PredictionModel, YOLO(str(self.ultralytics_path()), task="detect"))
            self.names = class_names(self._model.names)
        except DetectorError:
            raise
        except Exception as error:
            raise DetectorError(
                f"cannot load model: {error}; install the vision dependencies"
            ) from error
        if self.model_info.names and self.names != self.model_info.names:
            raise DetectorError("model class names differ from metadata.yaml")
        try:
            self.trigger_classes = trigger_classes(
                list(self.model_info.trigger_classes), self.names
            )
        except ValueError as error:
            raise DetectorError(str(error)) from error

    def ultralytics_path(self) -> Path:
        """Ultralytics recognises NCNN exports only by a `_ncnn_model` directory suffix.

        A C8 package (`models/<model_id>/`) is linked under such a name in a
        temporary directory that lives as long as this detector.
        """
        path = self.model_info.path
        if not path.is_dir() or path.name.endswith(NCNN_DIRECTORY_SUFFIX):
            return path
        self._link_directory = tempfile.TemporaryDirectory(prefix="columba-model-")
        link = Path(self._link_directory.name) / f"{path.name}{NCNN_DIRECTORY_SUFFIX}"
        link.symlink_to(path.resolve(), target_is_directory=True)
        return link

    def warmup(self, width: int, height: int) -> None:
        import numpy

        self.detect(
            FramePacket(
                -1, 0.0, numpy.zeros((height, width, COLOUR_CHANNEL_COUNT), dtype=numpy.uint8)
            )
        )

    def detect(self, packet: FramePacket) -> list[Detection]:
        try:
            image_size = self.image_size
            results = self._model.predict(
                source=packet.frame,
                imgsz=image_size if isinstance(image_size, int) else list(image_size),
                conf=self.confidence,
                device="cpu",
                rect=False,
                iou=NMS_INTERSECTION_OVER_UNION,
                verbose=False,
                stream=False,
            )
            if not results:
                self.last_stage_ms = dict.fromkeys(STAGE_NAMES)
                return []
            result = results[0]
            self.last_stage_ms = stage_timings(getattr(result, "speed", None))
            names = class_names(result.names)
            detections: list[Detection] = []
            for box in result.boxes or ():
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if confidence >= self.confidence:
                    left_px, top_px, right_px, bottom_px = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            names[class_id],
                            confidence,
                            left_px,
                            top_px,
                            right_px,
                            bottom_px,
                            class_id=class_id,
                        )
                    )
            return detections
        except Exception as error:
            raise DetectorError(f"model inference failed: {error}") from error
