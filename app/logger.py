from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any


class EventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._console = logging.getLogger("columba_zero")

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._console.info("%s %s", event, fields)


def health_snapshot(dropped_frames: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"dropped_frames": dropped_frames}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        result.update(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_rss_bytes=process.memory_info().rss,
            memory_percent=process.memory_percent(),
        )
    except ImportError:
        pass
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            result["cpu_temperature_c"] = int(thermal_path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    throttled_path = Path("/sys/devices/platform/soc/soc:firmware/get_throttled")
    if throttled_path.exists():
        try:
            result["throttling"] = throttled_path.read_text().strip()
        except OSError:
            pass
    return result
