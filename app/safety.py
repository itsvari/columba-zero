from __future__ import annotations

import time
from collections.abc import Callable

from .config import ModelConfig, SafetyConfig
from .models import Detection, FramePacket, SafetyDecision, Target
from .tracking import point_in_polygon


class SafetyGate:
    def __init__(
        self,
        config: SafetyConfig,
        model_config: ModelConfig,
        stale_after_s: float,
        emergency_stop: Callable[[], bool] = lambda: False,
        interlock_closed: Callable[[], bool] = lambda: True,
        require_hardware_interlock: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.stale_after_s = stale_after_s
        self.emergency_stop = emergency_stop
        self.interlock_closed = interlock_closed
        self.require_hardware_interlock = require_hardware_interlock
        self.clock = clock

    def evaluate(
        self,
        packet: FramePacket,
        detections: list[Detection],
        target: Target | None,
        require_interlock: bool = False,
    ) -> SafetyDecision:
        if self.emergency_stop():
            return SafetyDecision(False, "emergency_stop", critical=True)
        age = self.clock() - packet.captured_at
        if age < -0.05:
            return SafetyDecision(False, "frame_timestamp_in_future", critical=True)
        if age > self.stale_after_s:
            return SafetyDecision(False, "stale_frame", critical=True)
        for detection in detections:
            if (
                detection.label.lower() in self.model_config.blocked_labels
                and detection.confidence >= self.model_config.safety_confidence
            ):
                return SafetyDecision(False, f"blocked_object:{detection.label}")
        if target is not None and not point_in_polygon(
            (target.pixel_x, target.pixel_y), self.config.roi
        ):
            return SafetyDecision(False, "target_outside_roi")
        if require_interlock or self.require_hardware_interlock:
            if not self.interlock_closed():
                return SafetyDecision(False, "hardware_interlock_open", critical=True)
        return SafetyDecision(True)


class GPIOInputs:
    def __init__(
        self,
        emergency_stop_pin: int | None,
        interlock_pin: int | None,
        interlock_active_high: bool,
    ) -> None:
        try:
            from gpiozero import Button, DigitalInputDevice
        except ImportError as exc:
            raise RuntimeError("gpiozero is required for Raspberry Pi safety inputs") from exc
        self._emergency = (
            None if emergency_stop_pin is None else Button(emergency_stop_pin, pull_up=True)
        )
        self._interlock = (
            None
            if interlock_pin is None
            else DigitalInputDevice(interlock_pin, pull_up=not interlock_active_high)
        )
        self._interlock_active_high = interlock_active_high

    def emergency_pressed(self) -> bool:
        return self._emergency is not None and bool(self._emergency.is_pressed)

    def interlock_closed(self) -> bool:
        if self._interlock is None:
            return False
        return bool(self._interlock.value)

    def close(self) -> None:
        if self._emergency is not None:
            self._emergency.close()
        if self._interlock is not None:
            self._interlock.close()
