from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Detection, FramePacket


class DetectorError(RuntimeError):
    pass


class Detector(ABC):
    @abstractmethod
    def detect(self, packet: FramePacket) -> list[Detection]: ...


class UltralyticsDetector(Detector):
    def __init__(self, model_path: str, image_size: int, confidence: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "Ultralytics is unavailable; install the 'vision' dependency group"
            ) from exc
        path = Path(model_path)
        if not path.exists():
            raise DetectorError(f"model does not exist: {path}")
        self._model = YOLO(str(path), task="detect")
        self.image_size = image_size
        self.confidence = confidence

    def warmup(self, width: int, height: int) -> None:
        """Initialize lazy inference backends before acquiring timestamped frames."""
        import numpy as np

        self.detect(
            FramePacket(
                sequence=-1,
                captured_at=time.monotonic(),
                frame=np.zeros((height, width, 3), dtype=np.uint8),
            )
        )

    def detect(self, packet: FramePacket) -> list[Detection]:
        try:
            results = self._model.predict(
                source=packet.frame,
                imgsz=self.image_size,
                conf=self.confidence,
                verbose=False,
                stream=False,
            )
            if not results:
                return []
            result = results[0]
            names = result.names
            detections: list[Detection] = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = str(names[class_id]).lower()
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(label, confidence, x1, y1, x2, y2))
            return detections
        except Exception as exc:
            raise DetectorError(f"model inference failed: {exc}") from exc


class ReplayDetector(Detector):
    """Deterministic frame-indexed detections for recorded closed-loop tests."""

    def __init__(self, detections_by_frame: dict[int, list[Detection]]) -> None:
        self.detections_by_frame = detections_by_frame

    @classmethod
    def from_json(cls, path: str | Path) -> ReplayDetector:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        frames = raw.get("frames", raw)
        if not isinstance(frames, dict):
            raise DetectorError("replay detections must be a frame-indexed mapping")
        parsed: dict[int, list[Detection]] = {}
        for frame_number, items in frames.items():
            parsed[int(frame_number)] = [
                Detection(
                    label=str(item["label"]).lower(),
                    confidence=float(item["confidence"]),
                    x1=float(item["bbox"][0]),
                    y1=float(item["bbox"][1]),
                    x2=float(item["bbox"][2]),
                    y2=float(item["bbox"][3]),
                )
                for item in items
            ]
        return cls(parsed)

    def detect(self, packet: FramePacket) -> list[Detection]:
        return list(self.detections_by_frame.get(packet.sequence, []))


class NullDetector(Detector):
    def detect(self, packet: FramePacket) -> list[Detection]:
        return []
