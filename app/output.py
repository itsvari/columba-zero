from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import AppConfig
from .models import Detection


def health_snapshot() -> dict:
    result = {}
    try:
        import psutil
        process = psutil.Process()
        result["memory_rss_bytes"] = process.memory_info().rss
        result["cpu_percent"] = psutil.cpu_percent(interval=None)
    except ImportError:
        pass
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal.exists():
        try:
            result["temperature_c"] = float(thermal.read_text().strip()) / 1000
        except (OSError, ValueError):
            pass
    return result


class RunOutput:
    window = "Columba Zero - pigeon detection (q to quit)"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(config.output.directory) / f"{stamp}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._log = (self.directory / "detections.jsonl").open("w", encoding="utf-8")
        self._writer: Any = None
        self._shape = None
        self._window_open = False

    def event(self, event: str, **fields: Any) -> None:
        self._log.write(json.dumps({"event": event, "timestamp": time.time(), **fields},
                                  allow_nan=False) + "\n")
        self._log.flush()

    def start(self) -> None:
        if self.config.display.enabled:
            import cv2
            self._window_open = True
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def render(self, frame: Any, detections: list[Detection], processing_fps: float,
               capture_fps: float | None, tracking: dict | None = None) -> bool:
        if not self.config.display.enabled and not self.config.output.record:
            return True
        import cv2
        image = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = (round(v) for v in
                              (detection.x1, detection.y1, detection.x2, detection.y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"pigeon {detection.confidence:.2f}"
            position = (max(0, x1), min(image.shape[0] - 8, max(70, y1 + 18)))
            cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        text = f"Pigeons: {len(detections)} | Processing: {processing_fps:.1f} FPS"
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if capture_fps is not None:
            text = f"Capture: {capture_fps:.1f} FPS (requested {self.config.camera.fps:g})"
            cv2.putText(image, text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if tracking is not None:
            height, width = image.shape[:2]
            cv2.drawMarker(image, (width // 2, height // 2), (255, 255, 0),
                           cv2.MARKER_CROSS, 16, 1)
            target = tracking["selected_target"]
            if target is not None and tracking["state"] == "tracking":
                x1, y1, x2, y2 = (round(v) for v in target["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            text = (f'{tracking["mode"]}: {tracking["state"]} | Command pan '
                    f'{tracking["commanded_pan_deg"]:.1f} tilt '
                    f'{tracking["commanded_tilt_deg"]:.1f} deg')
            cv2.putText(image, text, (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 0), 1)
        if self.config.output.record:
            height, width = image.shape[:2]
            if self._writer is None:
                self._shape = (width, height)
                self._writer = cv2.VideoWriter(str(self.directory / "annotated.avi"),
                                               cv2.VideoWriter_fourcc(*"MJPG"),
                                               self.config.output.recording_fps, self._shape)
                if not self._writer.isOpened():
                    raise RuntimeError("cannot open annotated video writer")
            if (width, height) != self._shape:
                raise RuntimeError("frame size changed during recording")
            self._writer.write(image)
        if self.config.display.enabled:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                return False
            cv2.imshow(self.window, image)
            key = cv2.waitKey(1) & 0xFF
            return key != ord("q") and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1
        return True

    def close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.release()
        finally:
            try:
                if self._window_open:
                    import cv2
                    cv2.destroyAllWindows()
            finally:
                self._log.close()
