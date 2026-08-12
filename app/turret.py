from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from .config import TurretConfig
from .models import AimCommand


class TurretError(RuntimeError):
    pass


class TurretController(ABC):
    def __init__(
        self,
        config: TurretConfig,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.clock = clock
        self.sleeper = sleeper
        self.yaw, self.pitch = config.safe_deg
        self.stopped = True

    def set_limits(
        self,
        yaw_min: float,
        yaw_max: float,
        pitch_min: float,
        pitch_max: float,
    ) -> None:
        if yaw_min >= yaw_max or pitch_min >= pitch_max:
            raise TurretError("invalid turret limits")
        self.config = TurretConfig(
            driver=self.config.driver,
            yaw_pin=self.config.yaw_pin,
            pitch_pin=self.config.pitch_pin,
            yaw_limits_deg=(yaw_min, yaw_max),
            pitch_limits_deg=(pitch_min, pitch_max),
            home_deg=self.config.home_deg,
            safe_deg=self.config.safe_deg,
            max_velocity_deg_s=self.config.max_velocity_deg_s,
            command_stale_after_s=self.config.command_stale_after_s,
            move_timeout_s=self.config.move_timeout_s,
        )

    def _clamp(self, yaw: float, pitch: float) -> tuple[float, float]:
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise TurretError("non-finite turret command")
        yaw = min(max(yaw, self.config.yaw_limits_deg[0]), self.config.yaw_limits_deg[1])
        pitch = min(max(pitch, self.config.pitch_limits_deg[0]), self.config.pitch_limits_deg[1])
        return yaw, pitch

    def move_to(self, command: AimCommand) -> tuple[float, float]:
        age = self.clock() - command.issued_at
        if age < -0.05 or age > self.config.command_stale_after_s:
            raise TurretError(f"rejected stale turret command ({age:.3f}s old)")
        target_yaw, target_pitch = self._clamp(command.yaw, command.pitch)
        distance = max(abs(target_yaw - self.yaw), abs(target_pitch - self.pitch))
        duration = distance / self.config.max_velocity_deg_s
        if duration > self.config.move_timeout_s:
            raise TurretError("turret move would exceed timeout")
        steps = max(1, math.ceil(duration / 0.02))
        start_yaw, start_pitch = self.yaw, self.pitch
        started_at = self.clock()
        self.stopped = False
        for step in range(1, steps + 1):
            if self.clock() - started_at > self.config.move_timeout_s:
                self.stop()
                raise TurretError("turret move timed out")
            fraction = step / steps
            yaw = start_yaw + (target_yaw - start_yaw) * fraction
            pitch = start_pitch + (target_pitch - start_pitch) * fraction
            self._write_angles(yaw, pitch)
            self.yaw, self.pitch = yaw, pitch
            if step < steps and duration > 0:
                self.sleeper(duration / steps)
        self.stopped = True
        return self.yaw, self.pitch

    def home(self) -> tuple[float, float]:
        return self.move_to(AimCommand(*self.config.home_deg, issued_at=self.clock()))

    def safe_position(self) -> tuple[float, float]:
        return self.move_to(AimCommand(*self.config.safe_deg, issued_at=self.clock()))

    def stop(self) -> None:
        self.stopped = True
        self._stop_output()

    @abstractmethod
    def _write_angles(self, yaw: float, pitch: float) -> None: ...

    @abstractmethod
    def _stop_output(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class SimulationTurret(TurretController):
    def __init__(self, config: TurretConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self.movements: list[tuple[float, float]] = []

    def _write_angles(self, yaw: float, pitch: float) -> None:
        self.movements.append((yaw, pitch))

    def _stop_output(self) -> None:
        pass

    def close(self) -> None:
        self.stop()


class GPIOZeroTurret(TurretController):
    def __init__(self, config: TurretConfig, **kwargs: object) -> None:
        try:
            from gpiozero import AngularServo
        except ImportError as exc:
            raise TurretError("gpiozero is required for the gpiozero turret driver") from exc
        super().__init__(config, **kwargs)
        self._yaw_servo = AngularServo(
            config.yaw_pin,
            min_angle=config.yaw_limits_deg[0],
            max_angle=config.yaw_limits_deg[1],
        )
        self._pitch_servo = AngularServo(
            config.pitch_pin,
            min_angle=config.pitch_limits_deg[0],
            max_angle=config.pitch_limits_deg[1],
        )
        self._write_angles(*config.safe_deg)

    def _write_angles(self, yaw: float, pitch: float) -> None:
        self._yaw_servo.angle = yaw
        self._pitch_servo.angle = pitch

    def _stop_output(self) -> None:
        self._yaw_servo.detach()
        self._pitch_servo.detach()

    def close(self) -> None:
        self.stop()
        self._yaw_servo.close()
        self._pitch_servo.close()


def build_turret(config: TurretConfig) -> TurretController:
    if config.driver == "simulation":
        return SimulationTurret(config)
    if config.driver == "gpiozero":
        return GPIOZeroTurret(config)
    raise TurretError(f"unsupported turret driver: {config.driver}")
