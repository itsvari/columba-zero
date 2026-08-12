from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two numbers")
    return float(value[0]), float(value[1])


@dataclass(frozen=True, slots=True)
class CameraConfig:
    width: int
    height: int
    fps: float
    frame_timeout_s: float
    stale_after_s: float
    rotation: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    path: str
    image_size: int
    confidence: float
    safety_confidence: float
    pigeon_labels: tuple[str, ...]
    blocked_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    window_frames: int
    activation_frames: int
    release_frames: int
    cluster_distance_px: float


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    roi: tuple[tuple[float, float], ...]
    emergency_stop_pin: int | None
    interlock_pin: int | None
    interlock_active_high: bool
    max_activations_per_minute: int
    watchdog_timeout_s: float
    max_cpu_temperature_c: float


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    calibrated: bool
    homography: tuple[tuple[float, float, float], ...]
    turret_origin_m: tuple[float, float, float]
    target_plane_z_m: float
    aim_mode: str
    approved_target_plane_m: tuple[float, float]
    yaw_offset_deg: float
    pitch_offset_deg: float


@dataclass(frozen=True, slots=True)
class TurretConfig:
    driver: str
    yaw_pin: int
    pitch_pin: int
    yaw_limits_deg: tuple[float, float]
    pitch_limits_deg: tuple[float, float]
    home_deg: tuple[float, float]
    safe_deg: tuple[float, float]
    max_velocity_deg_s: float
    command_stale_after_s: float
    move_timeout_s: float


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    mode: str
    output_pin: int
    shutter_pin: int
    active_high: bool
    bench_mode: bool
    laser_enabled: bool
    pulse_s: float
    max_pulse_s: float
    settle_s: float
    cooldown_s: float


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    path: str
    health_interval_s: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    tracking: TrackingConfig
    safety: SafetyConfig
    geometry: GeometryConfig
    turret: TurretConfig
    actuator: ActuatorConfig
    logging: LoggingConfig

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AppConfig:
        camera = _section(data, "camera")
        model = _section(data, "model")
        tracking = _section(data, "tracking")
        safety = _section(data, "safety")
        geometry = _section(data, "geometry")
        turret = _section(data, "turret")
        actuator = _section(data, "actuator")
        logging = _section(data, "logging")

        roi_value = safety.get("roi", [])
        if not isinstance(roi_value, list) or len(roi_value) < 3:
            raise ConfigError("safety.roi must contain at least three [x, y] points")
        roi = tuple(_pair(point, "safety.roi point") for point in roi_value)

        matrix_value = geometry.get("homography", [])
        if not isinstance(matrix_value, list) or len(matrix_value) != 3:
            raise ConfigError("geometry.homography must be a 3x3 matrix")
        matrix = tuple(
            tuple(float(item) for item in row)
            for row in matrix_value
            if isinstance(row, list) and len(row) == 3
        )
        if len(matrix) != 3:
            raise ConfigError("geometry.homography must be a 3x3 matrix")

        cfg = cls(
            camera=CameraConfig(
                width=int(camera.get("width", 640)),
                height=int(camera.get("height", 480)),
                fps=float(camera.get("fps", 15)),
                frame_timeout_s=float(camera.get("frame_timeout_s", 2.0)),
                stale_after_s=float(camera.get("stale_after_s", 0.75)),
                rotation=int(camera.get("rotation", 0)),
            ),
            model=ModelConfig(
                path=str(model.get("path", "models/pigeon_ncnn_model")),
                image_size=int(model.get("image_size", 416)),
                confidence=float(model.get("confidence", 0.55)),
                safety_confidence=float(model.get("safety_confidence", 0.35)),
                pigeon_labels=tuple(str(x).lower() for x in model.get("pigeon_labels", ["pigeon"])),
                blocked_labels=tuple(
                    str(x).lower()
                    for x in model.get(
                        "blocked_labels",
                        [
                            "person",
                            "vehicle",
                            "car",
                            "truck",
                            "bus",
                            "motorcycle",
                            "bicycle",
                            "pet",
                            "dog",
                            "cat",
                        ],
                    )
                ),
            ),
            tracking=TrackingConfig(
                window_frames=int(tracking.get("window_frames", 5)),
                activation_frames=int(tracking.get("activation_frames", 3)),
                release_frames=int(tracking.get("release_frames", 1)),
                cluster_distance_px=float(tracking.get("cluster_distance_px", 120)),
            ),
            safety=SafetyConfig(
                roi=roi,
                emergency_stop_pin=(
                    None
                    if safety.get("emergency_stop_pin") is None
                    else int(safety["emergency_stop_pin"])
                ),
                interlock_pin=(
                    None if safety.get("interlock_pin") is None else int(safety["interlock_pin"])
                ),
                interlock_active_high=bool(safety.get("interlock_active_high", True)),
                max_activations_per_minute=int(safety.get("max_activations_per_minute", 3)),
                watchdog_timeout_s=float(safety.get("watchdog_timeout_s", 5.0)),
                max_cpu_temperature_c=float(safety.get("max_cpu_temperature_c", 80.0)),
            ),
            geometry=GeometryConfig(
                calibrated=bool(geometry.get("calibrated", False)),
                homography=matrix,
                turret_origin_m=tuple(
                    float(x) for x in geometry.get("turret_origin_m", [0.0, 0.0, 0.4])
                ),
                target_plane_z_m=float(geometry.get("target_plane_z_m", 0.0)),
                aim_mode=str(geometry.get("aim_mode", "fixed_safe_surface")),
                approved_target_plane_m=_pair(
                    geometry.get("approved_target_plane_m", [1.5, 0.0]),
                    "geometry.approved_target_plane_m",
                ),
                yaw_offset_deg=float(geometry.get("yaw_offset_deg", 0.0)),
                pitch_offset_deg=float(geometry.get("pitch_offset_deg", 0.0)),
            ),
            turret=TurretConfig(
                driver=str(turret.get("driver", "simulation")),
                yaw_pin=int(turret.get("yaw_pin", 12)),
                pitch_pin=int(turret.get("pitch_pin", 13)),
                yaw_limits_deg=_pair(turret.get("yaw_limits_deg", [-20, 20]), "yaw_limits_deg"),
                pitch_limits_deg=_pair(
                    turret.get("pitch_limits_deg", [-10, 10]), "pitch_limits_deg"
                ),
                home_deg=_pair(turret.get("home_deg", [0, 0]), "home_deg"),
                safe_deg=_pair(turret.get("safe_deg", [0, 10]), "safe_deg"),
                max_velocity_deg_s=float(turret.get("max_velocity_deg_s", 25)),
                command_stale_after_s=float(turret.get("command_stale_after_s", 0.5)),
                move_timeout_s=float(turret.get("move_timeout_s", 3.0)),
            ),
            actuator=ActuatorConfig(
                mode=str(actuator.get("mode", "indicator")),
                output_pin=int(actuator.get("output_pin", 23)),
                shutter_pin=int(actuator.get("shutter_pin", 24)),
                active_high=bool(actuator.get("active_high", True)),
                bench_mode=bool(actuator.get("bench_mode", False)),
                laser_enabled=bool(actuator.get("laser_enabled", False)),
                pulse_s=float(actuator.get("pulse_s", 0.25)),
                max_pulse_s=float(actuator.get("max_pulse_s", 0.5)),
                settle_s=float(actuator.get("settle_s", 0.2)),
                cooldown_s=float(actuator.get("cooldown_s", 10.0)),
            ),
            logging=LoggingConfig(
                path=str(logging.get("path", "logs/events.jsonl")),
                health_interval_s=float(logging.get("health_interval_s", 30.0)),
            ),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be a mapping")
        return cls.from_mapping(data)

    def validate(self) -> None:
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ConfigError("camera dimensions and fps must be positive")
        if self.camera.rotation not in (0, 180):
            raise ConfigError("camera.rotation must be 0 or 180")
        if any(
            not (0 <= x <= self.camera.width and 0 <= y <= self.camera.height)
            for x, y in self.safety.roi
        ):
            raise ConfigError("safety.roi points must lie within the configured frame")
        if not 0.0 <= self.model.confidence <= 1.0:
            raise ConfigError("model.confidence must be between 0 and 1")
        if not 0.0 <= self.model.safety_confidence <= 1.0:
            raise ConfigError("model.safety_confidence must be between 0 and 1")
        if not 1 <= self.tracking.activation_frames <= self.tracking.window_frames:
            raise ConfigError("activation_frames must be between 1 and window_frames")
        if not 0 <= self.tracking.release_frames < self.tracking.activation_frames:
            raise ConfigError("release_frames must be lower than activation_frames")
        if self.turret.yaw_limits_deg[0] >= self.turret.yaw_limits_deg[1]:
            raise ConfigError("yaw limits must be increasing")
        if self.turret.pitch_limits_deg[0] >= self.turret.pitch_limits_deg[1]:
            raise ConfigError("pitch limits must be increasing")
        if self.turret.max_velocity_deg_s <= 0 or self.turret.move_timeout_s <= 0:
            raise ConfigError("turret velocity and timeout must be positive")
        for name, (yaw, pitch) in (
            ("home_deg", self.turret.home_deg),
            ("safe_deg", self.turret.safe_deg),
        ):
            if not self.turret.yaw_limits_deg[0] <= yaw <= self.turret.yaw_limits_deg[1]:
                raise ConfigError(f"turret.{name} yaw lies outside limits")
            if not self.turret.pitch_limits_deg[0] <= pitch <= self.turret.pitch_limits_deg[1]:
                raise ConfigError(f"turret.{name} pitch lies outside limits")
        if self.actuator.pulse_s <= 0 or self.actuator.max_pulse_s <= 0:
            raise ConfigError("pulse durations must be positive")
        if self.actuator.pulse_s > self.actuator.max_pulse_s:
            raise ConfigError("pulse_s cannot exceed max_pulse_s")
        longest_blocking_operation = max(
            self.camera.frame_timeout_s,
            self.turret.move_timeout_s + self.actuator.settle_s + self.actuator.max_pulse_s,
        )
        if self.safety.watchdog_timeout_s <= longest_blocking_operation:
            raise ConfigError("watchdog_timeout_s must exceed all bounded runtime operations")
        if not 40 <= self.safety.max_cpu_temperature_c <= 90:
            raise ConfigError("max_cpu_temperature_c must be between 40 and 90")
        if self.actuator.mode == "laser" and not self.actuator.bench_mode:
            raise ConfigError("laser mode is permitted only when bench_mode is true")
        if self.actuator.mode == "laser" and self.geometry.aim_mode != "fixed_safe_surface":
            raise ConfigError("laser mode requires geometry.aim_mode=fixed_safe_surface")
        if self.actuator.laser_enabled and self.actuator.mode != "laser":
            raise ConfigError("laser_enabled requires actuator.mode=laser")
        if len(self.geometry.turret_origin_m) != 3:
            raise ConfigError("geometry.turret_origin_m must contain three numbers")
        if self.geometry.aim_mode not in ("fixed_safe_surface", "direct_target"):
            raise ConfigError("geometry.aim_mode is unsupported")
        if self.safety.max_activations_per_minute <= 0:
            raise ConfigError("max_activations_per_minute must be positive")
        if self.tracking.cluster_distance_px <= 0:
            raise ConfigError("cluster_distance_px must be positive")
