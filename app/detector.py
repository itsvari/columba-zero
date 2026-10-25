from __future__ import annotations

from pathlib import Path

import yaml

from .models import Detection, FramePacket


class DetectorError(RuntimeError):
    pass


def check_model(path: str, image_size: int) -> Path:
    model = Path(path)
    if not model.exists():
        raise DetectorError(f"model does not exist: {model}; see deploy/model_manifest.json")
    if model.is_dir():
        for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
            if not (model / name).is_file() or (model / name).stat().st_size == 0:
                raise DetectorError(f"incomplete NCNN export: missing or empty {model / name}")
        try:
            metadata = yaml.safe_load((model / "metadata.yaml").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("task") != "detect":
                raise ValueError("metadata must describe a detection model")
            validate_names(metadata.get("names", {}))
            if metadata.get("imgsz") != [image_size, image_size]:
                raise ValueError(f"export size {metadata.get('imgsz')} differs from {image_size}; "
                                 "export a separate model for this size")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise DetectorError(f"invalid NCNN metadata: {exc}") from exc
    elif model.suffix.lower() != ".pt" or model.stat().st_size == 0:
        raise DetectorError("model must be a non-empty .pt checkpoint or complete NCNN directory")
    return model


def validate_names(names: object) -> None:
    labels = list(names.values()) if isinstance(names, dict) else names
    if not isinstance(labels, (list, tuple)) or [str(n).lower() for n in labels] != ["pigeon"]:
        raise DetectorError("expected a single-class model with class 0 named 'pigeon'")
    if isinstance(names, dict) and set(names) not in ({0}, {"0"}):
        raise DetectorError("pigeon must have class index 0")


class UltralyticsDetector:
    def __init__(self, model_path: str, image_size: int, confidence: float) -> None:
        path = check_model(model_path, image_size)
        self.image_size = image_size
        self.confidence = confidence
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(path), task="detect")
            validate_names(self._model.names)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"cannot load model: {exc}; "
                                "install the vision dependencies") from exc

    def warmup(self, width: int, height: int) -> None:
        import numpy as np
        self.detect(FramePacket(-1, 0.0, np.zeros((height, width, 3), dtype=np.uint8)))

    def detect(self, packet: FramePacket) -> list[Detection]:
        try:
            results = self._model.predict(source=packet.frame, imgsz=self.image_size,
                                          conf=self.confidence, device="cpu", rect=False,
                                          iou=0.7, verbose=False, stream=False)
            if not results:
                return []
            result = results[0]
            validate_names(result.names)
            detections = []
            for box in result.boxes:
                label = str(result.names[int(box.cls.item())]).lower()
                confidence = float(box.conf.item())
                if label == "pigeon" and confidence >= self.confidence:
                    coords = [float(value) for value in box.xyxy[0].tolist()]
                    detections.append(Detection(label, confidence, *coords))
            return detections
        except Exception as exc:
            raise DetectorError(f"model inference failed: {exc}") from exc
