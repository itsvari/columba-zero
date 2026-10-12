from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from .json_data import JsonValue

MODEL_STRIDE_PX = 32
CONFIG_SECTIONS = frozenset({"camera", "model", "display", "output"})
OBSOLETE_SECTIONS = frozenset({"tracking", "safety", "geometry", "turret", "actuator", "logging"})


class ConfigError(ValueError):
    pass


def read_mapping(value: JsonValue, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a YAML mapping with string keys")
    return value


def read_section(
    configuration: dict[str, JsonValue],
    name: str,
    fields: frozenset[str],
) -> dict[str, JsonValue]:
    section = read_mapping(configuration.get(name, {}), name)
    unknown_fields = section.keys() - fields
    if unknown_fields:
        raise ConfigError(f"unknown {name} settings: {', '.join(sorted(unknown_fields))}")
    return section


def positive_integer(value: JsonValue, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def finite_positive_number(value: JsonValue, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{name} must be finite and positive")
    return float(value)


def boolean(value: JsonValue, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a YAML boolean")
    return value


def non_empty_path(value: JsonValue, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path")
    return value


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
class AppConfig:
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    display: DisplayConfig = DisplayConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        try:
            raw = read_mapping(
                cast(JsonValue, yaml.safe_load(Path(path).read_text(encoding="utf-8"))),
                "configuration",
            )
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"cannot load configuration {path}: {error}") from error
        obsolete_sections = raw.keys() & OBSOLETE_SECTIONS
        if obsolete_sections:
            raise ConfigError(
                "obsolete actuation configuration; migrate to app/config.yaml "
                f"(remove {', '.join(sorted(obsolete_sections))})"
            )
        unknown_sections = raw.keys() - CONFIG_SECTIONS
        if unknown_sections:
            raise ConfigError(
                f"unknown configuration sections: {', '.join(sorted(unknown_sections))}"
            )
        camera = read_section(
            raw, "camera", frozenset({"width", "height", "fps", "frame_timeout_s"})
        )
        model = read_section(raw, "model", frozenset({"path", "image_size", "confidence"}))
        display = read_section(raw, "display", frozenset({"enabled"}))
        output = read_section(
            raw,
            "output",
            frozenset({"directory", "record", "recording_fps", "health_interval_s"}),
        )
        defaults = cls()
        configuration = cls(
            camera=CameraConfig(
                width=positive_integer(camera.get("width", defaults.camera.width), "width"),
                height=positive_integer(camera.get("height", defaults.camera.height), "height"),
                fps=finite_positive_number(camera.get("fps", defaults.camera.fps), "fps"),
                frame_timeout_s=finite_positive_number(
                    camera.get("frame_timeout_s", defaults.camera.frame_timeout_s),
                    "frame_timeout_s",
                ),
            ),
            model=ModelConfig(
                path=non_empty_path(model.get("path", defaults.model.path), "model.path"),
                image_size=positive_integer(
                    model.get("image_size", defaults.model.image_size),
                    "image_size",
                ),
                confidence=finite_positive_number(
                    model.get("confidence", defaults.model.confidence),
                    "confidence",
                ),
            ),
            display=DisplayConfig(
                boolean(display.get("enabled", defaults.display.enabled), "display.enabled")
            ),
            output=OutputConfig(
                directory=non_empty_path(
                    output.get("directory", defaults.output.directory), "output.directory"
                ),
                record=boolean(output.get("record", defaults.output.record), "output.record"),
                recording_fps=finite_positive_number(
                    output.get("recording_fps", defaults.output.recording_fps),
                    "recording_fps",
                ),
                health_interval_s=finite_positive_number(
                    output.get("health_interval_s", defaults.output.health_interval_s),
                    "health_interval_s",
                ),
            ),
        )
        configuration.validate()
        return configuration

    def validate(self) -> None:
        for name, value in (
            ("width", self.camera.width),
            ("height", self.camera.height),
            ("image_size", self.model.image_size),
        ):
            positive_integer(value, name)
        if self.model.image_size % MODEL_STRIDE_PX:
            raise ConfigError(f"image_size must be a multiple of {MODEL_STRIDE_PX}")
        for name, measurement in (
            ("fps", self.camera.fps),
            ("frame_timeout_s", self.camera.frame_timeout_s),
            ("recording_fps", self.output.recording_fps),
            ("health_interval_s", self.output.health_interval_s),
        ):
            finite_positive_number(measurement, name)
        finite_positive_number(self.model.confidence, "confidence")
        if self.model.confidence > 1:
            raise ConfigError("confidence must be in (0, 1]")
        boolean(self.display.enabled, "display.enabled")
        boolean(self.output.record, "output.record")
        non_empty_path(self.model.path, "model.path")
        non_empty_path(self.output.directory, "output.directory")
