from __future__ import annotations

from typing import Any

from app.config import AppConfig


def config_mapping(**overrides: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 15,
            "frame_timeout_s": 1,
            "stale_after_s": 0.5,
            "rotation": 0,
        },
        "model": {
            "path": "unused",
            "image_size": 416,
            "confidence": 0.5,
            "safety_confidence": 0.3,
            "pigeon_labels": ["pigeon"],
            "blocked_labels": ["person", "car", "dog"],
        },
        "tracking": {
            "window_frames": 5,
            "activation_frames": 3,
            "release_frames": 1,
            "cluster_distance_px": 80,
        },
        "safety": {
            "roi": [[0, 0], [640, 0], [640, 480], [0, 480]],
            "emergency_stop_pin": None,
            "interlock_pin": None,
            "max_activations_per_minute": 3,
            "watchdog_timeout_s": 2,
            "max_cpu_temperature_c": 80,
        },
        "geometry": {
            "calibrated": True,
            "homography": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
            "turret_origin_m": [0, 0, 0],
            "target_plane_z_m": 0,
            "aim_mode": "fixed_safe_surface",
            "approved_target_plane_m": [2, 0],
            "yaw_offset_deg": 0,
            "pitch_offset_deg": 0,
        },
        "turret": {
            "driver": "simulation",
            "yaw_limits_deg": [-90, 90],
            "pitch_limits_deg": [-45, 45],
            "home_deg": [0, 0],
            "safe_deg": [0, 10],
            "max_velocity_deg_s": 1000,
            "command_stale_after_s": 0.5,
            "move_timeout_s": 1,
        },
        "actuator": {
            "mode": "indicator",
            "bench_mode": False,
            "laser_enabled": False,
            "pulse_s": 0.1,
            "max_pulse_s": 0.2,
            "settle_s": 0,
            "cooldown_s": 2,
        },
        "logging": {"path": "unused.jsonl", "health_interval_s": 30},
    }
    for section, values in overrides.items():
        data[section].update(values)
    return data


def make_config(**overrides: dict[str, Any]) -> AppConfig:
    return AppConfig.from_mapping(config_mapping(**overrides))
