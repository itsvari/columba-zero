from __future__ import annotations

import threading
import time
from collections.abc import Callable


class SoftwareWatchdog:
    """Independent heartbeat monitor that invokes a one-shot fail-safe callback."""

    def __init__(
        self,
        timeout_s: float,
        on_timeout: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout_s = timeout_s
        self.on_timeout = on_timeout
        self.clock = clock
        self._last_beat = clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tripped = False

    def start(self) -> None:
        self._last_beat = self.clock()
        self._thread = threading.Thread(target=self._run, name="software-watchdog", daemon=True)
        self._thread.start()

    def beat(self) -> None:
        self._last_beat = self.clock()

    def _run(self) -> None:
        interval = min(0.25, self.timeout_s / 4)
        while not self._stop.wait(interval):
            if not self._tripped and self.clock() - self._last_beat > self.timeout_s:
                self._tripped = True
                self.on_timeout()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
