"""Report Pi runtime prerequisites without starting capture, inference, or servo output."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig, ConfigError
from app.json_data import JsonValue
from app.pantilt import Calibration

MINIMUM_PYTHON_VERSION = (3, 11)
REQUIRED_POINTER_BITS = 64
BITS_PER_BYTE = 8
IMPORT_TIMEOUT_S = 20.0
PROBE_DETAIL_LIMIT = 2000
DEVICE_MODEL_PATH = Path("/proc/device-tree/model")
BASE_MODULES = ("numpy", "yaml", "cv2", "picamera2", "libcamera", "psutil")
NON_IMPORTING_MODULES = frozenset({"picamera2", "libcamera", "smbus2"})
IMPORT_PROBE = (
    "import importlib, sys; module = importlib.import_module(sys.argv[1]); print(module.__file__)"
)
AVAILABILITY_PROBE = (
    "import importlib.util, sys; specification = importlib.util.find_spec(sys.argv[1]); "
    "print(f'{specification.origin}; runtime importability unverified' "
    "if specification else 'not installed'); sys.exit(0 if specification else 1)"
)
MODEL_PROBE = (
    "import sys; from app.config import AppConfig; "
    "from app.detector import load_model_info; "
    "configuration = AppConfig.load(sys.argv[1]); "
    "print(load_model_info(configuration.model.path, configuration.model.image_size).path)"
)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def platform_check() -> PreflightCheck:
    try:
        device_model = DEVICE_MODEL_PATH.read_text(encoding="utf-8").rstrip("\x00\n")
    except OSError as error:
        return PreflightCheck("platform", False, f"cannot read Pi model: {error}")
    pointer_bits = struct.calcsize("P") * BITS_PER_BYTE
    passed = (
        platform.system() == "Linux"
        and platform.machine() == "aarch64"
        and pointer_bits == REQUIRED_POINTER_BITS
        and device_model.startswith("Raspberry Pi 5 ")
    )
    return PreflightCheck(
        "platform", passed, f"{device_model}; {platform.machine()}; {pointer_bits}-bit Python"
    )


def probe_python(name: str, source: str, *arguments: str) -> PreflightCheck:
    try:
        result = subprocess.run(
            (sys.executable, "-c", source, *arguments),
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return PreflightCheck(name, False, str(error))
    detail = result.stdout if result.returncode == 0 else result.stderr or result.stdout
    return PreflightCheck(name, result.returncode == 0, detail.strip()[-PROBE_DETAIL_LIMIT:])


def executable_check(name: str) -> PreflightCheck:
    executable = shutil.which(name)
    return PreflightCheck(name, executable is not None, executable or "not found on PATH")


def probe_module(name: str) -> PreflightCheck:
    # Picamera2 queries video devices during import, before capture is constructed.
    source = AVAILABILITY_PROBE if name in NON_IMPORTING_MODULES else IMPORT_PROBE
    return probe_python(name, source, name)


def mount_checks(configuration: AppConfig) -> tuple[PreflightCheck, ...]:
    try:
        calibration = Calibration.load(configuration.pan_tilt.calibration)
        calibration.validate_image_rotation(configuration.camera.rotation_deg)
    except ConfigError as error:
        return (PreflightCheck("calibration", False, str(error)),)
    device_path = Path(f"/dev/i2c-{calibration.bus}")
    try:
        accessible = device_path.is_char_device() and os.access(device_path, os.R_OK | os.W_OK)
    except OSError:
        accessible = False
    return (
        PreflightCheck("calibration", True, configuration.pan_tilt.calibration),
        PreflightCheck(
            "i2c_access",
            accessible,
            f"{device_path}; requires a character device and read/write permission; "
            "HAT presence and PCA9685 response are not tested",
        ),
    )


def preflight(configuration_path: Path) -> tuple[PreflightCheck, ...]:
    checks = [
        platform_check(),
        PreflightCheck(
            "python",
            sys.version_info >= MINIMUM_PYTHON_VERSION,
            f"{platform.python_version()}; {sys.executable}",
        ),
    ]
    try:
        configuration = AppConfig.load(configuration_path)
    except (OSError, ConfigError) as error:
        return (*checks, PreflightCheck("configuration", False, str(error)))
    checks.append(PreflightCheck("configuration", True, str(configuration_path)))
    modules: tuple[str, ...] = BASE_MODULES
    modules += (
        ("columba_native",)
        if configuration.model.backend == "native"
        else (
            "ultralytics",
            "ncnn",
        )
    )
    if configuration.follow.enabled:
        modules += ("scipy",)
        if configuration.pan_tilt.mode == "hardware":
            modules += ("smbus2",)
    checks.extend(probe_module(module) for module in modules)
    checks.append(probe_python("model_metadata", MODEL_PROBE, str(configuration_path)))
    checks.append(executable_check("rpicam-hello"))
    if configuration.preview.record:
        checks.extend(executable_check(name) for name in ("ffmpeg", "ffprobe"))
    if configuration.follow.enabled and configuration.pan_tilt.mode == "hardware":
        checks.extend(mount_checks(configuration))
    return tuple(checks)


@dataclass
class PreflightOptions:
    config: Path = Path("app/config.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PreflightOptions.config)
    options = PreflightOptions()
    parser.parse_args(namespace=options)
    checks = preflight(options.config)
    passed = all(check.passed for check in checks)
    report: dict[str, JsonValue] = {
        "scope": "environment prerequisites only; no capture, inference, or servo commands",
        "preflight_passed": passed,
        "hardware_validated": False,
        "checks": [check.as_dict() for check in checks],
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
