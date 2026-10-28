from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from .json_data import JsonValue

MODEL_STRIDE_PX = 32
CONFIG_SECTIONS = frozenset(
    {
        "camera",
        "model",
        "display",
        "output",
        "confirmation",
        "cameras",
        "camera_switch",
        "follow",
        "pan_tilt",
        "preview",
        "zone",
    }
)
OBSOLETE_SECTIONS = frozenset({"tracking", "safety", "geometry", "turret", "actuator", "logging"})
CAMERA_STREAMS = ("main", "lores")
CAMERA_ROLES = ("day", "night")
MODEL_BACKENDS = ("ultralytics", "native")
CAMERA_FIELDS = frozenset(
    {
        "width",
        "height",
        "fps",
        "frame_timeout_s",
        "stream",
        "main_size",
        "lens_position",
        "max_exposure_us",
        "exposure_compensation_ev",
        "tuning_file",
        "camera_index",
        "name",
        "rotation_deg",
    }
)
CAMERA_ENTRY_FIELDS = frozenset(
    {
        "name",
        "role",
        "camera_index",
        "tuning_file",
        "lens_position",
        "max_exposure_us",
        "exposure_compensation_ev",
        "rotation_deg",
    }
)
PAN_TILT_MODES = ("simulate", "hardware")
PREVIEW_PORT_MAX = 65535
PREVIEW_FPS_MAX = 10.0
PREVIEW_QUALITY_MIN = 1
PREVIEW_QUALITY_MAX = 100
PAN_TILT_DEADBAND_MAX = 0.5

ImageSize: TypeAlias = int | tuple[int, int]
CameraRotation: TypeAlias = Literal[0, 180]


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


def non_negative_integer(value: JsonValue, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def rotation_degrees(value: JsonValue, name: str) -> CameraRotation:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 180):
        raise ConfigError(f"{name} must be an integer of 0 or 180 degrees")
    return cast(CameraRotation, value)


def bounded_integer(value: JsonValue, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def finite_number(value: JsonValue, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    return float(value)


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


def choice(value: JsonValue, name: str, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"{name} must be one of: {', '.join(options)}")
    return value


def size_pair(value: JsonValue | tuple[int, int], name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must be a list of two positive integers")
    first, second = value
    return positive_integer(first, name), positive_integer(second, name)


def image_size(value: JsonValue | ImageSize, name: str = "image_size") -> ImageSize:
    """An integer means a square input; a pair is [height, width], as in NCNN metadata."""
    if isinstance(value, (list, tuple)):
        return size_pair(value, name)
    return positive_integer(value, name)


def input_hw(size: ImageSize) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else size


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: float = 15.0
    frame_timeout_s: float = 2.0
    stream: str = "main"
    main_size: tuple[int, int] | None = None
    lens_position: float | None = None
    max_exposure_us: int | None = None
    exposure_compensation_ev: float | None = None
    tuning_file: str | None = None
    camera_index: int = 0
    name: str | None = None
    rotation_deg: CameraRotation = 0

    @property
    def label(self) -> str:
        return self.name or f"camera{self.camera_index}"


@dataclass(frozen=True)
class CameraEntry:
    """One camera of a day/night pair; unset values inherit from `camera:`."""

    name: str
    role: str
    camera_index: int | None = None
    tuning_file: str | None = None
    lens_position: float | None = None
    max_exposure_us: int | None = None
    exposure_compensation_ev: float | None = None
    rotation_deg: CameraRotation | None = None


@dataclass(frozen=True)
class CameraSwitchConfig:
    night_below_lux: float = 10.0
    day_above_lux: float = 40.0
    hold_s: float = 5.0


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/pigeon_baseline_v1-2/best_ncnn_model"
    image_size: ImageSize = 640
    confidence: float = 0.5
    backend: str = "ultralytics"
    threads: int = 4


@dataclass(frozen=True)
class ConfirmationConfig:
    enabled: bool = False
    k: int = 2
    n: int = 3
    iou: float = 0.3
    max_age_frames: int = 2


@dataclass(frozen=True)
class FollowConfig:
    enabled: bool = False
    maintenance_confidence: float = 0.15
    retention_s: float = 0.75
    switch_area_ratio: float = 1.25
    switch_hold_s: float = 0.5


@dataclass(frozen=True)
class PanTiltConfig:
    mode: Literal["simulate", "hardware"] = "simulate"
    calibration: str = "calibration.local.yaml"
    control_hz: float = 20.0
    deadband: float = 0.05
    gain_deg_s: float = 60.0
    max_speed_deg_s: float = 30.0
    max_acceleration_deg_s2: float = 60.0
    observation_timeout_s: float = 0.25
    home_speed_deg_s: float = 10.0


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    max_fps: float = 10.0
    width_px: int = 800
    height_px: int = 450
    jpeg_quality: int = 80
    record: bool = False
    record_max_duration_s: float = 600.0
    record_max_bytes: int = 536870912
    record_min_free_bytes: int = 268435456
    export_timeout_s: float = 900.0
    record_unannotated: bool = False


@dataclass(frozen=True)
class ZoneConfig:
    enabled: bool = False


@dataclass(frozen=True)
class DisplayConfig:
    enabled: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs"
    record: bool = False
    recording_fps: float = 15.0
    health_interval_s: float = 30.0


def read_camera(section: dict[str, JsonValue], defaults: CameraConfig) -> CameraConfig:
    main_size = section.get("main_size")
    lens_position = section.get("lens_position")
    max_exposure_us = section.get("max_exposure_us")
    exposure_compensation_ev = section.get("exposure_compensation_ev")
    tuning_file = section.get("tuning_file")
    name = section.get("name")
    return CameraConfig(
        width=positive_integer(section.get("width", defaults.width), "width"),
        height=positive_integer(section.get("height", defaults.height), "height"),
        fps=finite_positive_number(section.get("fps", defaults.fps), "fps"),
        frame_timeout_s=finite_positive_number(
            section.get("frame_timeout_s", defaults.frame_timeout_s), "frame_timeout_s"
        ),
        stream=choice(section.get("stream", defaults.stream), "camera.stream", CAMERA_STREAMS),
        main_size=None if main_size is None else size_pair(main_size, "camera.main_size"),
        lens_position=(
            None if lens_position is None else finite_number(lens_position, "lens_position")
        ),
        max_exposure_us=(
            None
            if max_exposure_us is None
            else positive_integer(max_exposure_us, "max_exposure_us")
        ),
        exposure_compensation_ev=(
            None
            if exposure_compensation_ev is None
            else finite_number(exposure_compensation_ev, "exposure_compensation_ev")
        ),
        tuning_file=None if tuning_file is None else non_empty_path(tuning_file, "tuning_file"),
        camera_index=non_negative_integer(
            section.get("camera_index", defaults.camera_index), "camera_index"
        ),
        name=None if name is None else non_empty_path(name, "camera.name"),
        rotation_deg=rotation_degrees(
            section.get("rotation_deg", defaults.rotation_deg), "camera.rotation_deg"
        ),
    )


def read_camera_entries(value: JsonValue) -> tuple[CameraEntry, ...]:
    if not isinstance(value, list):
        raise ConfigError("cameras must be a list of camera mappings")
    entries: list[CameraEntry] = []
    for index, item in enumerate(value):
        entry = read_mapping(item, f"cameras[{index}]")
        unknown_fields = entry.keys() - CAMERA_ENTRY_FIELDS
        if unknown_fields:
            raise ConfigError(
                f"unknown cameras[{index}] settings: {', '.join(sorted(unknown_fields))}"
            )
        camera_index = entry.get("camera_index")
        tuning_file = entry.get("tuning_file")
        lens_position = entry.get("lens_position")
        max_exposure_us = entry.get("max_exposure_us")
        exposure_compensation_ev = entry.get("exposure_compensation_ev")
        rotation_deg = entry.get("rotation_deg")
        entries.append(
            CameraEntry(
                name=non_empty_path(entry.get("name"), f"cameras[{index}].name"),
                role=choice(entry.get("role"), f"cameras[{index}].role", CAMERA_ROLES),
                camera_index=(
                    None
                    if camera_index is None
                    else non_negative_integer(camera_index, "camera_index")
                ),
                tuning_file=(
                    None if tuning_file is None else non_empty_path(tuning_file, "tuning_file")
                ),
                lens_position=(
                    None if lens_position is None else finite_number(lens_position, "lens_position")
                ),
                max_exposure_us=(
                    None
                    if max_exposure_us is None
                    else positive_integer(max_exposure_us, "max_exposure_us")
                ),
                exposure_compensation_ev=(
                    None
                    if exposure_compensation_ev is None
                    else finite_number(exposure_compensation_ev, "exposure_compensation_ev")
                ),
                rotation_deg=(
                    None
                    if rotation_deg is None
                    else rotation_degrees(rotation_deg, f"cameras[{index}].rotation_deg")
                ),
            )
        )
    return tuple(entries)


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    display: DisplayConfig = DisplayConfig()
    output: OutputConfig = OutputConfig()
    confirmation: ConfirmationConfig = ConfirmationConfig()
    cameras: tuple[CameraEntry, ...] = ()
    camera_switch: CameraSwitchConfig = CameraSwitchConfig()
    follow: FollowConfig = FollowConfig()
    pan_tilt: PanTiltConfig = PanTiltConfig()
    preview: PreviewConfig = PreviewConfig()
    zone: ZoneConfig = ZoneConfig()

    @classmethod
    def load(cls, path: str | Path, *, follow_enabled: bool | None = None) -> AppConfig:
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
        camera = read_section(raw, "camera", CAMERA_FIELDS)
        model = read_section(
            raw,
            "model",
            frozenset({"path", "image_size", "confidence", "backend", "threads"}),
        )
        display = read_section(raw, "display", frozenset({"enabled"}))
        output = read_section(
            raw,
            "output",
            frozenset({"directory", "record", "recording_fps", "health_interval_s"}),
        )
        confirmation = read_section(
            raw, "confirmation", frozenset({"enabled", "k", "n", "iou", "max_age_frames"})
        )
        camera_switch = read_section(
            raw, "camera_switch", frozenset({"night_below_lux", "day_above_lux", "hold_s"})
        )
        follow = read_section(
            raw,
            "follow",
            frozenset(
                {
                    "enabled",
                    "maintenance_confidence",
                    "retention_s",
                    "switch_area_ratio",
                    "switch_hold_s",
                }
            ),
        )
        pan_tilt = read_section(
            raw,
            "pan_tilt",
            frozenset(
                {
                    "mode",
                    "calibration",
                    "control_hz",
                    "deadband",
                    "gain_deg_s",
                    "max_speed_deg_s",
                    "max_acceleration_deg_s2",
                    "observation_timeout_s",
                    "home_speed_deg_s",
                }
            ),
        )
        preview = read_section(
            raw,
            "preview",
            frozenset(
                {
                    "enabled",
                    "host",
                    "port",
                    "max_fps",
                    "width_px",
                    "height_px",
                    "jpeg_quality",
                    "record",
                    "record_unannotated",
                    "record_max_duration_s",
                    "record_max_bytes",
                    "record_min_free_bytes",
                    "export_timeout_s",
                }
            ),
        )
        zone = read_section(raw, "zone", frozenset({"enabled"}))
        if "camera_switch" in raw and raw.get("cameras") is None:
            raise ConfigError("camera_switch requires a cameras list")
        defaults = cls()
        configuration = cls(
            camera=read_camera(camera, defaults.camera),
            model=ModelConfig(
                path=non_empty_path(model.get("path", defaults.model.path), "model.path"),
                image_size=image_size(model.get("image_size", defaults.model.image_size)),
                confidence=finite_positive_number(
                    model.get("confidence", defaults.model.confidence),
                    "confidence",
                ),
                backend=choice(
                    model.get("backend", defaults.model.backend), "model.backend", MODEL_BACKENDS
                ),
                threads=positive_integer(
                    model.get("threads", defaults.model.threads), "model.threads"
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
            confirmation=ConfirmationConfig(
                enabled=boolean(
                    confirmation.get("enabled", defaults.confirmation.enabled),
                    "confirmation.enabled",
                ),
                k=positive_integer(confirmation.get("k", defaults.confirmation.k), "k"),
                n=positive_integer(confirmation.get("n", defaults.confirmation.n), "n"),
                iou=finite_positive_number(
                    confirmation.get("iou", defaults.confirmation.iou), "confirmation.iou"
                ),
                max_age_frames=non_negative_integer(
                    confirmation.get("max_age_frames", defaults.confirmation.max_age_frames),
                    "max_age_frames",
                ),
            ),
            cameras=(() if raw.get("cameras") is None else read_camera_entries(raw.get("cameras"))),
            camera_switch=CameraSwitchConfig(
                night_below_lux=finite_number(
                    camera_switch.get("night_below_lux", defaults.camera_switch.night_below_lux),
                    "night_below_lux",
                ),
                day_above_lux=finite_number(
                    camera_switch.get("day_above_lux", defaults.camera_switch.day_above_lux),
                    "day_above_lux",
                ),
                hold_s=finite_number(
                    camera_switch.get("hold_s", defaults.camera_switch.hold_s), "hold_s"
                ),
            ),
            follow=FollowConfig(
                enabled=boolean(follow.get("enabled", defaults.follow.enabled), "follow.enabled"),
                maintenance_confidence=finite_positive_number(
                    follow.get("maintenance_confidence", defaults.follow.maintenance_confidence),
                    "follow.maintenance_confidence",
                ),
                retention_s=finite_positive_number(
                    follow.get("retention_s", defaults.follow.retention_s), "follow.retention_s"
                ),
                switch_area_ratio=finite_positive_number(
                    follow.get("switch_area_ratio", defaults.follow.switch_area_ratio),
                    "follow.switch_area_ratio",
                ),
                switch_hold_s=finite_number(
                    follow.get("switch_hold_s", defaults.follow.switch_hold_s),
                    "follow.switch_hold_s",
                ),
            ),
            pan_tilt=PanTiltConfig(
                mode=cast(
                    Literal["simulate", "hardware"],
                    choice(
                        pan_tilt.get("mode", defaults.pan_tilt.mode),
                        "pan_tilt.mode",
                        PAN_TILT_MODES,
                    ),
                ),
                calibration=non_empty_path(
                    pan_tilt.get("calibration", defaults.pan_tilt.calibration),
                    "pan_tilt.calibration",
                ),
                control_hz=finite_positive_number(
                    pan_tilt.get("control_hz", defaults.pan_tilt.control_hz),
                    "pan_tilt.control_hz",
                ),
                deadband=finite_number(
                    pan_tilt.get("deadband", defaults.pan_tilt.deadband), "pan_tilt.deadband"
                ),
                gain_deg_s=finite_positive_number(
                    pan_tilt.get("gain_deg_s", defaults.pan_tilt.gain_deg_s),
                    "pan_tilt.gain_deg_s",
                ),
                max_speed_deg_s=finite_positive_number(
                    pan_tilt.get("max_speed_deg_s", defaults.pan_tilt.max_speed_deg_s),
                    "pan_tilt.max_speed_deg_s",
                ),
                max_acceleration_deg_s2=finite_positive_number(
                    pan_tilt.get(
                        "max_acceleration_deg_s2", defaults.pan_tilt.max_acceleration_deg_s2
                    ),
                    "pan_tilt.max_acceleration_deg_s2",
                ),
                observation_timeout_s=finite_positive_number(
                    pan_tilt.get("observation_timeout_s", defaults.pan_tilt.observation_timeout_s),
                    "pan_tilt.observation_timeout_s",
                ),
                home_speed_deg_s=finite_positive_number(
                    pan_tilt.get("home_speed_deg_s", defaults.pan_tilt.home_speed_deg_s),
                    "pan_tilt.home_speed_deg_s",
                ),
            ),
            preview=PreviewConfig(
                enabled=boolean(
                    preview.get("enabled", defaults.preview.enabled), "preview.enabled"
                ),
                host=non_empty_path(preview.get("host", defaults.preview.host), "preview.host"),
                port=bounded_integer(
                    preview.get("port", defaults.preview.port),
                    "preview.port",
                    minimum=0,
                    maximum=PREVIEW_PORT_MAX,
                ),
                max_fps=finite_positive_number(
                    preview.get("max_fps", defaults.preview.max_fps), "preview.max_fps"
                ),
                width_px=positive_integer(
                    preview.get("width_px", defaults.preview.width_px), "preview.width_px"
                ),
                height_px=positive_integer(
                    preview.get("height_px", defaults.preview.height_px), "preview.height_px"
                ),
                jpeg_quality=bounded_integer(
                    preview.get("jpeg_quality", defaults.preview.jpeg_quality),
                    "preview.jpeg_quality",
                    minimum=PREVIEW_QUALITY_MIN,
                    maximum=PREVIEW_QUALITY_MAX,
                ),
                record=boolean(preview.get("record", defaults.preview.record), "preview.record"),
                record_unannotated=boolean(
                    preview.get("record_unannotated", defaults.preview.record_unannotated),
                    "preview.record_unannotated",
                ),
                record_max_duration_s=finite_positive_number(
                    preview.get("record_max_duration_s", defaults.preview.record_max_duration_s),
                    "preview.record_max_duration_s",
                ),
                record_max_bytes=positive_integer(
                    preview.get("record_max_bytes", defaults.preview.record_max_bytes),
                    "preview.record_max_bytes",
                ),
                record_min_free_bytes=non_negative_integer(
                    preview.get("record_min_free_bytes", defaults.preview.record_min_free_bytes),
                    "preview.record_min_free_bytes",
                ),
                export_timeout_s=finite_positive_number(
                    preview.get("export_timeout_s", defaults.preview.export_timeout_s),
                    "preview.export_timeout_s",
                ),
            ),
            zone=ZoneConfig(
                enabled=boolean(zone.get("enabled", defaults.zone.enabled), "zone.enabled")
            ),
        )
        if follow_enabled is not None:
            configuration = replace(
                configuration, follow=replace(configuration.follow, enabled=follow_enabled)
            )
        configuration.validate()
        return configuration

    def camera_for(self, entry: CameraEntry) -> CameraConfig:
        camera = self.camera
        return CameraConfig(
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            frame_timeout_s=camera.frame_timeout_s,
            stream=camera.stream,
            main_size=camera.main_size,
            lens_position=(
                entry.lens_position if entry.lens_position is not None else camera.lens_position
            ),
            max_exposure_us=(
                entry.max_exposure_us
                if entry.max_exposure_us is not None
                else camera.max_exposure_us
            ),
            exposure_compensation_ev=(
                entry.exposure_compensation_ev
                if entry.exposure_compensation_ev is not None
                else camera.exposure_compensation_ev
            ),
            tuning_file=entry.tuning_file if entry.tuning_file is not None else camera.tuning_file,
            camera_index=(
                entry.camera_index if entry.camera_index is not None else camera.camera_index
            ),
            name=entry.name,
            rotation_deg=(
                entry.rotation_deg if entry.rotation_deg is not None else camera.rotation_deg
            ),
        )

    def validate(self) -> None:
        for name, value in (
            ("width", self.camera.width),
            ("height", self.camera.height),
        ):
            positive_integer(value, name)
        for side in input_hw(image_size(self.model.image_size)):
            if side % MODEL_STRIDE_PX:
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
        if self.camera.exposure_compensation_ev is not None:
            finite_number(self.camera.exposure_compensation_ev, "exposure_compensation_ev")
        choice(self.model.backend, "model.backend", MODEL_BACKENDS)
        positive_integer(self.model.threads, "model.threads")
        choice(self.camera.stream, "camera.stream", CAMERA_STREAMS)
        rotation_degrees(self.camera.rotation_deg, "camera.rotation_deg")
        if self.camera.main_size is not None:
            main_width, main_height = size_pair(self.camera.main_size, "camera.main_size")
            if main_width < self.camera.width or main_height < self.camera.height:
                raise ConfigError("camera.main_size must not be smaller than width x height")
        if self.camera.lens_position is not None and self.camera.lens_position < 0:
            raise ConfigError("lens_position must be non-negative (dioptres; 0 is infinity)")
        boolean(self.display.enabled, "display.enabled")
        boolean(self.output.record, "output.record")
        non_empty_path(self.model.path, "model.path")
        non_empty_path(self.output.directory, "output.directory")
        confirmation = self.confirmation
        boolean(confirmation.enabled, "confirmation.enabled")
        if not 1 <= confirmation.k <= confirmation.n:
            raise ConfigError("confirmation needs 1 <= k <= n")
        if not 0 < confirmation.iou <= 1:
            raise ConfigError("confirmation.iou must be in (0, 1]")
        non_negative_integer(confirmation.max_age_frames, "max_age_frames")
        follow = self.follow
        boolean(follow.enabled, "follow.enabled")
        finite_positive_number(follow.maintenance_confidence, "follow.maintenance_confidence")
        finite_positive_number(follow.retention_s, "follow.retention_s")
        finite_positive_number(follow.switch_area_ratio, "follow.switch_area_ratio")
        finite_number(follow.switch_hold_s, "follow.switch_hold_s")
        if follow.maintenance_confidence > 1:
            raise ConfigError("follow.maintenance_confidence must be in (0, 1]")
        if follow.enabled and follow.maintenance_confidence > self.model.confidence:
            raise ConfigError("follow.maintenance_confidence must not exceed model.confidence")
        if follow.switch_area_ratio <= 1:
            raise ConfigError("follow.switch_area_ratio must exceed 1")
        if follow.switch_hold_s < 0:
            raise ConfigError("follow.switch_hold_s must be non-negative")
        if follow.enabled and not confirmation.enabled:
            raise ConfigError("following requires confirmation.enabled: true")
        pan_tilt = self.pan_tilt
        choice(pan_tilt.mode, "pan_tilt.mode", PAN_TILT_MODES)
        non_empty_path(pan_tilt.calibration, "pan_tilt.calibration")
        finite_positive_number(pan_tilt.control_hz, "pan_tilt.control_hz")
        finite_number(pan_tilt.deadband, "pan_tilt.deadband")
        if not 0 <= pan_tilt.deadband < PAN_TILT_DEADBAND_MAX:
            raise ConfigError("pan_tilt.deadband must be in [0, 0.5)")
        for name, measurement in (
            ("gain_deg_s", pan_tilt.gain_deg_s),
            ("max_speed_deg_s", pan_tilt.max_speed_deg_s),
            ("max_acceleration_deg_s2", pan_tilt.max_acceleration_deg_s2),
            ("observation_timeout_s", pan_tilt.observation_timeout_s),
            ("home_speed_deg_s", pan_tilt.home_speed_deg_s),
        ):
            finite_positive_number(measurement, f"pan_tilt.{name}")
        if pan_tilt.home_speed_deg_s > pan_tilt.max_speed_deg_s:
            raise ConfigError("pan_tilt.home_speed_deg_s must not exceed max_speed_deg_s")
        if follow.enabled and pan_tilt.observation_timeout_s > follow.retention_s:
            raise ConfigError(
                "pan_tilt.observation_timeout_s must not exceed follow.retention_s when following"
            )
        preview = self.preview
        boolean(preview.enabled, "preview.enabled")
        non_empty_path(preview.host, "preview.host")
        bounded_integer(preview.port, "preview.port", 0, PREVIEW_PORT_MAX)
        finite_positive_number(preview.max_fps, "preview.max_fps")
        if preview.max_fps > PREVIEW_FPS_MAX:
            raise ConfigError(f"preview.max_fps must not exceed {PREVIEW_FPS_MAX:g}")
        positive_integer(preview.width_px, "preview.width_px")
        positive_integer(preview.height_px, "preview.height_px")
        bounded_integer(
            preview.jpeg_quality,
            "preview.jpeg_quality",
            PREVIEW_QUALITY_MIN,
            PREVIEW_QUALITY_MAX,
        )
        boolean(preview.record, "preview.record")
        boolean(preview.record_unannotated, "preview.record_unannotated")
        finite_positive_number(preview.record_max_duration_s, "preview.record_max_duration_s")
        positive_integer(preview.record_max_bytes, "preview.record_max_bytes")
        non_negative_integer(preview.record_min_free_bytes, "preview.record_min_free_bytes")
        finite_positive_number(preview.export_timeout_s, "preview.export_timeout_s")
        boolean(self.zone.enabled, "zone.enabled")
        if self.zone.enabled:
            raise ConfigError(
                "zone.enabled is deferred until fixed-zone hardware acceptance is complete"
            )
        if self.cameras:
            roles = sorted(entry.role for entry in self.cameras)
            if roles != sorted(CAMERA_ROLES):
                raise ConfigError("cameras must list exactly one day and one night camera")
            if len({entry.name for entry in self.cameras}) != len(self.cameras):
                raise ConfigError("camera names must be unique")
            for index, entry in enumerate(self.cameras):
                if entry.rotation_deg is not None:
                    rotation_degrees(entry.rotation_deg, f"cameras[{index}].rotation_deg")
                if entry.exposure_compensation_ev is not None:
                    finite_number(entry.exposure_compensation_ev, "exposure_compensation_ev")
                if entry.lens_position is not None and entry.lens_position < 0:
                    raise ConfigError("lens_position must be non-negative")
            if follow.enabled:
                raise ConfigError(
                    "follow.enabled cannot be used with cameras; camera switching is unsupported"
                )
            switch = self.camera_switch
            finite_number(switch.hold_s, "hold_s")
            if switch.hold_s < 0:
                raise ConfigError("camera_switch.hold_s must be non-negative")
            if not switch.night_below_lux < switch.day_above_lux:
                raise ConfigError("camera_switch needs night_below_lux < day_above_lux")
