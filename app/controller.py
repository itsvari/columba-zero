from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from .actuators import DeterrentActuator
from .config import AppConfig
from .geometry import TargetGeometry
from .logger import EventLogger
from .models import Detection, FramePacket, RuntimeState, Target
from .safety import SafetyGate
from .tracking import TemporalConfirmer, point_in_polygon, select_target
from .turret import TurretController, TurretError


class RuntimeController:
    def __init__(
        self,
        config: AppConfig,
        safety: SafetyGate,
        geometry: TargetGeometry,
        turret: TurretController,
        actuator: DeterrentActuator,
        logger: EventLogger,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.safety = safety
        self.geometry = geometry
        self.turret = turret
        self.actuator = actuator
        self.logger = logger
        self.clock = clock
        self.sleeper = sleeper
        self.state = RuntimeState.SAFE
        self.confirmer = TemporalConfirmer(
            config.tracking.window_frames,
            config.tracking.activation_frames,
            config.tracking.release_frames,
        )
        self.activation_times: deque[float] = deque()
        self.cooldown_until = 0.0
        self.settled_at = 0.0
        self.pending_target: Target | None = None

    def _transition(self, state: RuntimeState, reason: str) -> None:
        previous = self.state
        self.state = state
        self.logger.event(
            "state_transition", previous=previous.value, state=state.value, reason=reason
        )

    def arm(self) -> bool:
        self.actuator.all_off()
        if self.safety.emergency_stop():
            self._transition(RuntimeState.SAFE, "emergency_stop")
            return False
        if self.config.actuator.mode == "laser" and not self.safety.interlock_closed():
            self._transition(RuntimeState.SAFE, "hardware_interlock_open")
            return False
        self.confirmer.reset()
        self._transition(RuntimeState.IDLE, "startup_checks_passed")
        return True

    def fail_safe(self, reason: str, error: Exception | None = None) -> None:
        try:
            self.actuator.all_off()
        finally:
            self.turret.stop()
            self.pending_target = None
            self.confirmer.reset()
            self._transition(RuntimeState.SAFE, reason)
            self.logger.event("fault", reason=reason, error=None if error is None else repr(error))

    def _reject_event(self, reason: str) -> None:
        self.actuator.all_off()
        self.turret.stop()
        self.pending_target = None
        self.confirmer.reset()
        destination = (
            RuntimeState.COOLDOWN if self.clock() < self.cooldown_until else RuntimeState.IDLE
        )
        self._transition(destination, reason)
        self.logger.event("activation_rejected", reason=reason)

    def _rate_limit_allows(self, now: float) -> bool:
        while self.activation_times and now - self.activation_times[0] >= 60.0:
            self.activation_times.popleft()
        return len(self.activation_times) < self.config.safety.max_activations_per_minute

    def _eligible_pigeons(self, detections: list[Detection]) -> list[Detection]:
        return [
            detection
            for detection in detections
            if detection.label.lower() in self.config.model.pigeon_labels
            and detection.confidence >= self.config.model.confidence
            and point_in_polygon(detection.centroid, self.config.safety.roi)
        ]

    def process(self, packet: FramePacket, detections: list[Detection]) -> RuntimeState:
        if self.state == RuntimeState.SAFE:
            return self.state
        now = self.clock()
        target = select_target(
            self._eligible_pigeons(detections), self.config.tracking.cluster_distance_px
        )
        decision = self.safety.evaluate(packet, detections, target)
        if not decision.allowed:
            if decision.critical:
                self.fail_safe(decision.reason)
            else:
                self._reject_event(decision.reason)
            return self.state

        rising_confirmation = self.confirmer.update(target is not None)

        if self.state == RuntimeState.COOLDOWN:
            if now >= self.cooldown_until:
                self._transition(RuntimeState.IDLE, "cooldown_complete")
            return self.state

        if self.state == RuntimeState.AIMING:
            self._finish_aiming(packet, detections, target, now)
            return self.state

        if target is None:
            if self.state == RuntimeState.VERIFYING and self.confirmer.hit_count == 0:
                self._transition(RuntimeState.IDLE, "target_lost")
            return self.state

        if not rising_confirmation:
            if self.state == RuntimeState.IDLE:
                self._transition(RuntimeState.VERIFYING, "candidate_seen")
            return self.state

        if not self._rate_limit_allows(now):
            self._reject_event("activation_rate_limit")
            return self.state
        self._begin_aiming(target, now)
        return self.state

    def _begin_aiming(self, target: Target, now: float) -> None:
        try:
            command = self.geometry.target_to_command(target, issued_at=now)
            self.actuator.all_off()
            self._transition(RuntimeState.AIMING, "target_confirmed")
            actual_yaw, actual_pitch = self.turret.move_to(command)
            self.pending_target = target
            self.settled_at = self.clock() + self.config.actuator.settle_s
            self.logger.event(
                "turret_aimed",
                requested_yaw=command.yaw,
                requested_pitch=command.pitch,
                actual_yaw=actual_yaw,
                actual_pitch=actual_pitch,
                target_x=target.pixel_x,
                target_y=target.pixel_y,
                target_count=target.member_count,
            )
        except Exception as exc:
            self.fail_safe("aiming_fault", exc)

    def _finish_aiming(
        self,
        packet: FramePacket,
        detections: list[Detection],
        target: Target | None,
        now: float,
    ) -> None:
        if target is None or self.pending_target is None:
            self._reject_event("target_lost_while_aiming")
            return
        displacement = (
            (target.pixel_x - self.pending_target.pixel_x) ** 2
            + (target.pixel_y - self.pending_target.pixel_y) ** 2
        ) ** 0.5
        if displacement > self.config.tracking.cluster_distance_px:
            self._reject_event("target_moved_while_aiming")
            return
        if now < self.settled_at:
            return
        decision = self.safety.evaluate(
            packet,
            detections,
            target,
            require_interlock=self.config.actuator.mode == "laser",
        )
        if not decision.allowed:
            if decision.critical:
                self.fail_safe(decision.reason)
            else:
                self._reject_event(decision.reason)
            return
        try:
            self._transition(RuntimeState.ACTIVE, "fresh_frame_and_interlock_confirmed")
            self.actuator.pulse()
            activated_at = self.clock()
            self.activation_times.append(activated_at)
            self.cooldown_until = activated_at + self.config.actuator.cooldown_s
            self.pending_target = None
            self._transition(RuntimeState.COOLDOWN, "pulse_complete")
            self.logger.event("activation", duration_s=self.config.actuator.pulse_s)
        except Exception as exc:
            self.fail_safe("actuator_fault", exc)

    def shutdown(self, reason: str = "shutdown") -> None:
        self.actuator.all_off()
        was_faulted = self.state == RuntimeState.SAFE
        if not was_faulted and not self.safety.emergency_stop():
            try:
                self.turret.safe_position()
            except TurretError as exc:
                self.logger.event("shutdown_position_failed", error=repr(exc))
        self.turret.stop()
        self.pending_target = None
        self._transition(RuntimeState.SAFE, reason)
