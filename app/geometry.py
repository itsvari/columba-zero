from __future__ import annotations

import math

import numpy as np

from .config import GeometryConfig
from .models import AimCommand, Target


class CalibrationError(RuntimeError):
    pass


class TargetGeometry:
    def __init__(self, config: GeometryConfig) -> None:
        self.config = config
        self._homography = np.asarray(config.homography, dtype=float)
        if self._homography.shape != (3, 3):
            raise CalibrationError("homography must be 3x3")

    def pixel_to_plane(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        if not self.config.calibrated:
            raise CalibrationError("camera-to-target geometry has not been calibrated")
        mapped = self._homography @ np.array([pixel_x, pixel_y, 1.0])
        if abs(mapped[2]) < 1e-9 or not np.all(np.isfinite(mapped)):
            raise CalibrationError("homography produced an invalid target point")
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def target_to_command(self, target: Target, issued_at: float) -> AimCommand:
        plane_x, plane_y = self.pixel_to_plane(target.pixel_x, target.pixel_y)
        if self.config.aim_mode == "fixed_safe_surface":
            plane_x, plane_y = self.config.approved_target_plane_m
        elif self.config.aim_mode != "direct_target":
            raise CalibrationError(f"unsupported aim mode: {self.config.aim_mode}")
        origin_x, origin_y, origin_z = self.config.turret_origin_m
        delta_x = plane_x - origin_x
        delta_y = plane_y - origin_y
        delta_z = self.config.target_plane_z_m - origin_z
        horizontal = math.hypot(delta_x, delta_y)
        if horizontal < 1e-6:
            raise CalibrationError("target is too close to the turret origin")
        yaw = math.degrees(math.atan2(delta_y, delta_x)) + self.config.yaw_offset_deg
        pitch = math.degrees(math.atan2(delta_z, horizontal)) + self.config.pitch_offset_deg
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise CalibrationError("geometry produced non-finite angles")
        return AimCommand(yaw=yaw, pitch=pitch, issued_at=issued_at)
