from __future__ import annotations

import pytest
from conftest import make_config

from app.geometry import CalibrationError, TargetGeometry
from app.models import AimCommand, Target
from app.turret import SimulationTurret, TurretError


def test_geometry_maps_pixel_to_angles() -> None:
    config = make_config(geometry={"aim_mode": "direct_target"})
    command = TargetGeometry(config.geometry).target_to_command(Target(100, 100, 1, 1), 0)
    assert command.yaw == pytest.approx(45)
    assert command.pitch == pytest.approx(0)


def test_uncalibrated_geometry_refuses_to_aim() -> None:
    config = make_config(geometry={"calibrated": False})
    with pytest.raises(CalibrationError, match="not been calibrated"):
        TargetGeometry(config.geometry).target_to_command(Target(100, 100, 1, 1), 0)


def test_fixed_safe_surface_does_not_follow_detection_coordinate() -> None:
    config = make_config(
        geometry={"aim_mode": "fixed_safe_surface", "approved_target_plane_m": [2, 0]}
    )
    geometry = TargetGeometry(config.geometry)
    first = geometry.target_to_command(Target(100, 100, 1, 1), 0)
    second = geometry.target_to_command(Target(300, 200, 1, 1), 0)
    assert first.yaw == second.yaw == 0
    assert first.pitch == second.pitch == 0


def test_turret_clamps_and_rejects_stale_commands() -> None:
    now = [0.0]
    config = make_config()
    turret = SimulationTurret(config.turret, clock=lambda: now[0], sleeper=lambda _: None)
    assert turret.move_to(AimCommand(180, -90, 0)) == (90, -45)
    now[0] = 1.0
    with pytest.raises(TurretError, match="stale"):
        turret.move_to(AimCommand(0, 0, 0))
