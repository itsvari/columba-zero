"""The optional C++ NCNN backend for the deployment detector."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy
import yaml
from numpy.typing import NDArray

from .detector import DetectorError, validate_names
from .inference import COLOUR_CHANNEL_COUNT
from .models import Detection, Frame, FramePacket

MetadataValue: TypeAlias = (
    None | bool | int | float | str | list["MetadataValue"] | dict[str | int, "MetadataValue"]
)


class NativeModel(Protocol):
    @property
    def input_hw(self) -> tuple[int, int]: ...
    @property
    def names(self) -> dict[int, str]: ...
    @property
    def decode(self) -> str: ...
    def detect(self, frame: Frame, conf: float) -> NDArray[numpy.float32]: ...
    def detect_with_timings(
        self, frame: Frame, conf: float
    ) -> tuple[NDArray[numpy.float32], dict[str, float]]: ...


def _check_package(model_path: str, image_size: int | tuple[int, int]) -> Path:
    model = Path(model_path)
    if not model.exists():
        raise DetectorError(f"model does not exist: {model}; see deploy/model_manifest.json")
    if not model.is_dir():
        raise DetectorError("native model must be a complete NCNN directory")
    for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise DetectorError(f"incomplete NCNN export: missing or empty {path}")
    try:
        metadata = cast(MetadataValue, yaml.safe_load((model / "metadata.yaml").read_text("utf-8")))
        if not isinstance(metadata, dict) or metadata.get("task") != "detect":
            raise ValueError("metadata must describe a detection model")
        configured_size = (
            [image_size, image_size] if isinstance(image_size, int) else list(image_size)
        )
        exported_size = metadata.get("imgsz")
        if exported_size != configured_size:
            raise ValueError(
                f"export size {exported_size} differs from {image_size}; "
                "export a separate model for this size"
            )
        names = metadata.get("names", {})
        validate_names(cast(dict[int, str], names))
        columba = metadata.get("columba")
        if "columba" not in metadata:
            if exported_size != [640, 640]:
                raise ValueError("legacy NCNN export without columba metadata must be 640x640")
            return model
        if not isinstance(columba, dict):
            raise ValueError("columba must be a mapping")
        if columba.get("input_hw") != exported_size:
            raise ValueError("columba.input_hw differs from export imgsz")
        if columba.get("decode") not in ("yolo11", "yolo26"):
            raise ValueError(f"unsupported decode: {columba.get('decode')}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise DetectorError(f"invalid NCNN metadata: {error}") from error
    return model


class NativeDetector:
    def __init__(
        self,
        model_path: str,
        image_size: int | tuple[int, int],
        confidence: float,
        threads: int = 4,
    ) -> None:
        path = _check_package(model_path, image_size)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise DetectorError("confidence must be finite and between 0 and 1")
        if threads <= 0:
            raise DetectorError("threads must be positive")
        self.image_size = image_size
        self.confidence = confidence
        self._last_class_ids: tuple[int, ...] = ()
        self.last_stage_ms: dict[str, float] = {}
        try:
            extension = importlib.import_module("columba_native")
            self._model = cast(NativeModel, extension.Detector(str(path), threads=threads))
            self.names = self._model.names
            validate_names(self.names)
        except DetectorError:
            raise
        except Exception as error:
            raise DetectorError(
                f"cannot load model: {error}; build columba_native as described in native/README.md"
            ) from error

    def warmup(self, width: int, height: int) -> None:
        self.detect(
            FramePacket(
                -1, 0.0, numpy.zeros((height, width, COLOUR_CHANNEL_COUNT), dtype=numpy.uint8)
            )
        )

    def detect(self, packet: FramePacket) -> list[Detection]:
        try:
            predictions, timings = self._model.detect_with_timings(packet.frame, self.confidence)
            self.last_stage_ms = {
                "preprocess": timings["preprocess"],
                "inference": timings["extract"],
                "postprocess": timings["postprocess"],
            }
            detections: list[Detection] = []
            class_ids: list[int] = []
            for left_px, top_px, right_px, bottom_px, score, predicted_class in predictions:
                class_id = int(predicted_class)
                label = self.names[class_id].lower()
                confidence = float(score)
                if label != "pigeon" or confidence < self.confidence:
                    continue
                detections.append(
                    Detection(
                        label,
                        confidence,
                        float(left_px),
                        float(top_px),
                        float(right_px),
                        float(bottom_px),
                    )
                )
                class_ids.append(class_id)
            self._last_class_ids = tuple(class_ids)
            return detections
        except Exception as error:
            raise DetectorError(f"model inference failed: {error}") from error
