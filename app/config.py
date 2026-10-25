from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: float = 15.0
    frame_timeout_s: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/pigeon_baseline_v1-2/best_ncnn_model"
    image_size: int = 640
    confidence: float = 0.5


@dataclass(frozen=True)
class DisplayConfig:
    enabled: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs"
    record: bool = False
    recording_fps: float = 15.0
    health_interval_s: float = 30.0


@dataclass(frozen=True)
class TrackingConfig:
    calibration: str = "calibration.local.yaml"
    match_distance: float = 0.25  # center distance in frame-width/height units
    gain: float = 30.0  # degrees/second at normalized error 1
    deadband: float = 0.05  # fraction of full frame dimension on each side
    max_speed: float = 30.0  # degrees/second
    loss_timeout_s: float = 2.0
    observation_timeout_s: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    display: DisplayConfig = DisplayConfig()
    output: OutputConfig = OutputConfig()
    tracking: TrackingConfig = TrackingConfig()

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ConfigError("configuration must be a YAML mapping")
            obsolete = set(raw) & {"safety", "geometry", "turret", "actuator", "logging"}
            if obsolete:
                raise ConfigError("obsolete actuation configuration; migrate to app/config.yaml "
                                  f"(remove {', '.join(sorted(obsolete))})")
            unknown = set(raw) - {"camera", "model", "display", "output", "tracking"}
            if unknown:
                raise ConfigError(f"unknown configuration sections: {', '.join(sorted(unknown))}")
            config = cls(CameraConfig(**raw.get("camera", {})),
                         ModelConfig(**raw.get("model", {})),
                         DisplayConfig(**raw.get("display", {})),
                         OutputConfig(**raw.get("output", {})),
                         TrackingConfig(**raw.get("tracking", {})))
            config.validate()
            return config
        except (OSError, yaml.YAMLError, TypeError) as exc:
            raise ConfigError(f"cannot load configuration {path}: {exc}") from exc

    def validate(self) -> None:
        for name, value in (("width", self.camera.width), ("height", self.camera.height),
                            ("image_size", self.model.image_size)):
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if self.model.image_size % 32:
            raise ConfigError("image_size must be a multiple of 32")
        for name, value in (("fps", self.camera.fps),
                            ("frame_timeout_s", self.camera.frame_timeout_s),
                            ("recording_fps", self.output.recording_fps),
                            ("health_interval_s", self.output.health_interval_s)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be finite and positive")
        confidence = self.model.confidence
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 < confidence <= 1):
            raise ConfigError("confidence must be in (0, 1]")
        if type(self.display.enabled) is not bool or type(self.output.record) is not bool:
            raise ConfigError("display.enabled and output.record must be YAML booleans")
        if not isinstance(self.model.path, str) or not self.model.path.strip():
            raise ConfigError("model.path must be a non-empty path")
        if not isinstance(self.output.directory, str) or not self.output.directory.strip():
            raise ConfigError("output.directory must be a non-empty path")
        tracking = self.tracking
        if not isinstance(tracking.calibration, str) or not tracking.calibration.strip():
            raise ConfigError("tracking.calibration must be a non-empty path")
        for name in ("match_distance", "gain", "max_speed", "loss_timeout_s",
                     "observation_timeout_s"):
            value = getattr(tracking, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"tracking.{name} must be finite and positive")
        if tracking.match_distance > 2 ** 0.5:
            raise ConfigError("tracking.match_distance cannot exceed the frame diagonal")
        if (type(tracking.deadband) not in (int, float)
                or not math.isfinite(tracking.deadband) or not 0 <= tracking.deadband < 0.5):
            raise ConfigError("tracking.deadband must be in [0, 0.5)")
