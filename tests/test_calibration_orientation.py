from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from app.calibrate import CalibrationArguments
from app.config import ConfigError, PanTiltConfig
from app.json_data import JsonValue
from app.pan_tilt_control import FollowSession
from app.pantilt import Calibration

RETENTION_S = 0.75


class RotationEvents:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, JsonValue]]] = []

    def event(self, event: str, **fields: JsonValue) -> None:
        self.records.append((event, fields))


def test_profile_rotation_round_trips_and_old_profiles_default_to_zero() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.yaml"
        profile = replace(Calibration(), image_rotation_deg=180)
        profile.save(path)
        assert Calibration.load(path) == profile
        assert CalibrationArguments(image_rotation_deg=180).profile().image_rotation_deg == 180
        profile.validate_image_rotation(180)
        with pytest.raises(ConfigError, match="camera.rotation_deg"):
            profile.validate_image_rotation(0)
        legacy = profile.as_dict()
        legacy.pop("image_rotation_deg")
        path.write_text(yaml.safe_dump(legacy))
        assert Calibration.load(path).image_rotation_deg == 0


def test_profile_rotation_rejects_malformed_external_values() -> None:
    invalid_values: tuple[JsonValue, ...] = (90, True, 180.0, "180", None)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.yaml"
        for value in invalid_values:
            profile = Calibration().as_dict()
            profile["image_rotation_deg"] = value
            path.write_text(yaml.safe_dump(profile))
            with pytest.raises(ConfigError, match="image_rotation_deg"):
                Calibration.load(path)


def test_mismatched_capture_rotation_never_constructs_a_hardware_driver() -> None:
    factory = Mock(side_effect=AssertionError("driver must not be opened"))
    session = FollowSession(
        PanTiltConfig(mode="hardware"),
        RETENTION_S,
        driver_factory=factory,
        image_rotation_deg=180,
    )
    with patch.object(Calibration, "load", return_value=Calibration()):
        session.start()
        with pytest.raises(RuntimeError, match="camera.rotation_deg"):
            session.close()
    factory.assert_not_called()
    assert session.snapshot().state == "fault"


def test_replay_ignores_live_rotation_without_loading_a_profile_or_driver() -> None:
    events = RotationEvents()
    factory = Mock(side_effect=AssertionError("replay must not open a driver"))
    session = FollowSession(
        PanTiltConfig(mode="hardware"),
        RETENTION_S,
        replay=True,
        events=events,
        driver_factory=factory,
        image_rotation_deg=180,
    )
    with patch.object(Calibration, "load", side_effect=AssertionError("replay loaded profile")):
        session.start()
        session.close()
    factory.assert_not_called()
    startup = next(fields for name, fields in events.records if name == "controller_start")
    assert startup["mode"] == "simulate"
    assert startup["image_rotation_deg"] == 0
    assert startup["image_orientation"] == "as_stored"
    calibration = startup["calibration"]
    assert isinstance(calibration, dict)
    assert calibration["image_rotation_deg"] == 0
