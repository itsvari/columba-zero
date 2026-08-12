from __future__ import annotations

import threading

from app.watchdog import SoftwareWatchdog


def test_watchdog_invokes_fail_safe_when_heartbeat_stops() -> None:
    tripped = threading.Event()
    watchdog = SoftwareWatchdog(0.03, tripped.set)
    try:
        watchdog.start()
        assert tripped.wait(0.5)
    finally:
        watchdog.close()
