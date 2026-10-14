"""The optional C++ NCNN backend for the deployment detector."""

from __future__ import annotations

import importlib
import math
from typing import Protocol, TypeAlias, cast

import numpy
import yaml
from numpy.typing import NDArray

from .detector import DetectorError, ModelInfo, class_names, load_model_info
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


def _check_package(model_path: str, image_size: int | tuple[int, int]) -> ModelInfo:
    information = load_model_info(model_path, image_size)
    if not information.path.is_dir():
        raise DetectorError("native model must be a complete NCNN directory")
    if information.decode not in ("yolo11", "yolo26"):
        raise DetectorError(f"invalid NCNN metadata: unsupported decode: {information.decode}")
    try:
        metadata = cast(
            MetadataValue, yaml.safe_load((information.path / "metadata.yaml").read_text("utf-8"))
        )
        if (
            isinstance(metadata, dict)
            and "columba" not in metadata
            and information.input_hw != (640, 640)
        ):
            raise ValueError("legacy NCNN export without columba metadata must be 640x640")
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise DetectorError(f"invalid NCNN metadata: {error}") from error
    return information


class NativeDetector:
    def __init__(
        self,
        model_path: str,
        image_size: int | tuple[int, int],
        confidence: float,
        threads: int = 4,
    ) -> None:
        self.model_info = _check_package(model_path, image_size)
        path = self.model_info.path
        self.trigger_classes = self.model_info.trigger_classes
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise DetectorError("confidence must be finite and between 0 and 1")
        if threads <= 0:
            raise DetectorError("threads must be positive")
        self.image_size = image_size
        self.confidence = confidence
        self.last_stage_ms: dict[str, float] = {}
        try:
            extension = importlib.import_module("columba_native")
            self._model = cast(NativeModel, extension.Detector(str(path), threads=threads))
            self.names = class_names(self._model.names)
            if self.names != self.model_info.names:
                raise DetectorError("model class names differ from metadata.yaml")
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
            for left_px, top_px, right_px, bottom_px, score, predicted_class in predictions:
                class_id = int(predicted_class)
                label = self.names[class_id].lower()
                confidence = float(score)
                if confidence < self.confidence:
                    continue
                detections.append(
                    Detection(
                        label,
                        confidence,
                        float(left_px),
                        float(top_px),
                        float(right_px),
                        float(bottom_px),
                        class_id=class_id,
                    )
                )
            return detections
        except Exception as error:
            raise DetectorError(f"model inference failed: {error}") from error
