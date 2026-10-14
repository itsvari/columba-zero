from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .config import AppConfig
from .json_data import JsonValue
from .models import Detection, Frame

if TYPE_CHECKING:
    import cv2

MILLIUNITS_PER_UNIT = 1000
RUN_SUFFIX_LENGTH = 8
BOX_COLOUR_BGR = (0, 255, 0)
OTHER_CLASS_COLOUR_BGR = (0, 165, 255)
CONFIRMED_COLOUR_BGR = (0, 0, 255)
TEXT_COLOUR_BGR = (0, 255, 255)
TEXT_OUTLINE_COLOUR_BGR = (0, 0, 0)
BOX_THICKNESS_PX = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS_PX = 1
TEXT_OUTLINE_THICKNESS_PX = 3
LABEL_BOTTOM_MARGIN_PX = 8
LABEL_MINIMUM_TOP_PX = 70
LABEL_OFFSET_PX = 18
PROCESSING_TEXT_POSITION_PX = (10, 24)
CAPTURE_TEXT_POSITION_PX = (10, 46)
WINDOW_KEY_MASK = 0xFF
WINDOW_EVENT_DELAY_MS = 1


def process_health() -> dict[str, JsonValue]:
    try:
        import psutil
    except ImportError:
        return {}
    try:
        process = psutil.Process()
        return {
            "memory_rss_bytes": process.memory_info().rss,
            "cpu_percent": psutil.cpu_percent(interval=None),
        }
    except (OSError, psutil.Error):
        return {}


def health_snapshot() -> dict[str, JsonValue]:
    result = process_health()
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        result["temperature_c"] = float(thermal.read_text().strip()) / MILLIUNITS_PER_UNIT
    except (OSError, ValueError):
        pass
    return result


def annotate_frame(
    frame: Frame,
    detections: Sequence[Detection],
    processing_fps: float,
    capture_fps: float | None,
    requested_capture_fps: float,
    confirmed_track_ids: frozenset[int] = frozenset(),
) -> Frame:
    import cv2

    image = frame.copy()
    for detection in detections:
        colour = BOX_COLOUR_BGR if detection.label == "pigeon" else OTHER_CLASS_COLOUR_BGR
        confirmed = detection.track_id is not None and detection.track_id in confirmed_track_ids
        if confirmed:
            colour = CONFIRMED_COLOUR_BGR
        left_px, top_px, right_px, bottom_px = (
            round(coordinate)
            for coordinate in (
                detection.left_px,
                detection.top_px,
                detection.right_px,
                detection.bottom_px,
            )
        )
        cv2.rectangle(
            image,
            (left_px, top_px),
            (right_px, bottom_px),
            colour,
            BOX_THICKNESS_PX,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"#{detection.track_id} {label}"
        if confirmed:
            label = f"{label} CONFIRMED"
        position_px = (
            max(0, left_px),
            min(
                image.shape[0] - LABEL_BOTTOM_MARGIN_PX,
                max(LABEL_MINIMUM_TOP_PX, top_px + LABEL_OFFSET_PX),
            ),
        )
        cv2.putText(
            image,
            label,
            position_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_OUTLINE_COLOUR_BGR,
            TEXT_OUTLINE_THICKNESS_PX,
        )
        cv2.putText(
            image,
            label,
            position_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            colour,
            TEXT_THICKNESS_PX,
        )
    pigeon_count = sum(detection.label == "pigeon" for detection in detections)
    text = f"Pigeons: {pigeon_count} | Processing: {processing_fps:.1f} FPS"
    cv2.putText(
        image,
        text,
        PROCESSING_TEXT_POSITION_PX,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_COLOUR_BGR,
        TEXT_THICKNESS_PX,
    )
    if capture_fps is not None:
        text = f"Capture: {capture_fps:.1f} FPS (requested {requested_capture_fps:g})"
        cv2.putText(
            image,
            text,
            CAPTURE_TEXT_POSITION_PX,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_COLOUR_BGR,
            TEXT_THICKNESS_PX,
        )
    return image


class RunOutput:
    window = "Columba Zero - pigeon detection (q to quit)"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.directory = (
            Path(config.output.directory) / f"{timestamp}-{uuid4().hex[:RUN_SUFFIX_LENGTH]}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self._log = (self.directory / "detections.jsonl").open("w", encoding="utf-8", newline="\n")
        self._writer: cv2.VideoWriter | None = None
        self._recording_size_px: tuple[int, int] | None = None
        self._window_open = False

    def event(self, event: str, **fields: JsonValue) -> None:
        self._log.write(
            json.dumps(
                {"event": event, "timestamp": time.time(), **fields},
                allow_nan=False,
            )
            + "\n"
        )
        self._log.flush()

    def write_json(self, name: str, data: JsonValue) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    def start(self) -> None:
        if self.config.display.enabled:
            import cv2

            self._window_open = True
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def _record(self, image: Frame) -> None:
        import cv2

        height_px, width_px = image.shape[:2]
        frame_size_px = (width_px, height_px)
        if self._writer is None:
            self._recording_size_px = frame_size_px
            self._writer = cv2.VideoWriter(
                str(self.directory / "annotated.avi"),
                cv2.VideoWriter.fourcc(*"MJPG"),
                self.config.output.recording_fps,
                frame_size_px,
            )
            if not self._writer.isOpened():
                raise RuntimeError("cannot open annotated video writer")
        if frame_size_px != self._recording_size_px:
            raise RuntimeError("frame size changed during recording")
        self._writer.write(image)

    def render(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        processing_fps: float,
        capture_fps: float | None,
        confirmed_track_ids: frozenset[int] = frozenset(),
    ) -> bool:
        if not self.config.display.enabled and not self.config.output.record:
            return True
        image = annotate_frame(
            frame,
            detections,
            processing_fps,
            capture_fps,
            self.config.camera.fps,
            confirmed_track_ids,
        )
        if self.config.output.record:
            self._record(image)
        if not self.config.display.enabled:
            return True
        import cv2

        if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
            return False
        cv2.imshow(self.window, image)
        key = cv2.waitKey(WINDOW_EVENT_DELAY_MS) & WINDOW_KEY_MASK
        return key != ord("q") and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1

    def poll(self) -> bool:
        """Handle window events between frames; False when the window closed or q."""
        if not self.config.display.enabled or not self._window_open:
            return True
        import cv2

        key = cv2.waitKey(WINDOW_EVENT_DELAY_MS) & WINDOW_KEY_MASK
        return key != ord("q") and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1

    def close_recording_and_display(self) -> None:
        writer, self._writer = self._writer, None
        window_open, self._window_open = self._window_open, False
        try:
            if writer is not None:
                writer.release()
        finally:
            if window_open:
                import cv2

                cv2.destroyAllWindows()

    def close(self) -> None:
        try:
            self.close_recording_and_display()
        finally:
            self._log.close()
