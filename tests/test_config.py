from __future__ import annotations

import pytest
from conftest import config_mapping

from app.config import AppConfig, ConfigError


def test_laser_mode_requires_bench_mode() -> None:
    data = config_mapping(actuator={"mode": "laser", "laser_enabled": True})
    with pytest.raises(ConfigError, match="bench_mode"):
        AppConfig.from_mapping(data)


def test_temporal_thresholds_are_validated() -> None:
    data = config_mapping(tracking={"activation_frames": 6})
    with pytest.raises(ConfigError, match="activation_frames"):
        AppConfig.from_mapping(data)


def test_laser_mode_cannot_follow_detected_animal_coordinate() -> None:
    data = config_mapping(
        actuator={"mode": "laser", "bench_mode": True, "laser_enabled": True},
        geometry={"aim_mode": "direct_target"},
    )
    with pytest.raises(ConfigError, match="fixed_safe_surface"):
        AppConfig.from_mapping(data)
