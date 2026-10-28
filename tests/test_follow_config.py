from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import yaml

from app.config import (
    AppConfig,
    ConfigError,
    ConfirmationConfig,
    FollowConfig,
    PanTiltConfig,
    ZoneConfig,
)
from app.json_data import JsonValue


class FollowConfigTests(unittest.TestCase):
    def load(self, configuration: dict[str, JsonValue]) -> AppConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
            return AppConfig.load(path)

    def test_enabling_following_keeps_the_acquisition_threshold_above_maintenance(self) -> None:
        baseline = self.load({})
        configuration = self.load({"follow": {"enabled": True}, "confirmation": {"enabled": True}})
        self.assertEqual(configuration.model.confidence, baseline.model.confidence)
        self.assertLess(configuration.follow.maintenance_confidence, configuration.model.confidence)

    def test_follow_policy_round_trips(self) -> None:
        configuration = self.load(
            {
                "confirmation": {"enabled": True},
                "follow": {
                    "enabled": True,
                    "maintenance_confidence": 0.2,
                    "retention_s": 1.0,
                    "switch_area_ratio": 1.5,
                    "switch_hold_s": 0.25,
                },
            }
        )
        self.assertEqual(configuration.follow, FollowConfig(True, 0.2, 1.0, 1.5, 0.25))

    def test_malformed_or_unknown_follow_settings_are_rejected(self) -> None:
        invalid_sections: tuple[JsonValue, ...] = (
            {"enabled": "true"},
            {"enabled": True, "maintenance_confidence": 0.6},
            {"maintenance_confidence": 0},
            {"maintenance_confidence": True},
            {"maintenance_confidence": 1.1},
            {"retention_s": float("nan")},
            {"switch_area_ratio": 1},
            {"switch_hold_s": -1},
            {"switch_hold_s": float("inf")},
            {"unexpected": True},
            [],
        )
        for section in invalid_sections:
            with self.subTest(section=section), self.assertRaises(ConfigError):
                self.load({"follow": section})

    def test_disabled_following_keeps_existing_low_acquisition_settings_valid(self) -> None:
        configuration = self.load({"model": {"confidence": 0.1}})
        self.assertFalse(configuration.follow.enabled)
        with self.assertRaisesRegex(ConfigError, "must not exceed"):
            replace(configuration, follow=replace(configuration.follow, enabled=True)).validate()

    def test_following_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(ConfigError, "confirmation.enabled"):
            self.load({"follow": {"enabled": True}, "confirmation": {"enabled": False}})

    def test_following_hardware_preview_and_zone_are_off_by_default(self) -> None:
        configuration = self.load({})
        self.assertFalse(configuration.follow.enabled)
        self.assertEqual(configuration.pan_tilt.mode, "simulate")
        self.assertFalse(configuration.preview.enabled)
        self.assertFalse(configuration.preview.record)
        self.assertFalse(configuration.zone.enabled)

    def test_follow_sections_load_typed_values_and_allow_recording_without_http(self) -> None:
        configuration = self.load(
            {
                "pan_tilt": {
                    "mode": "hardware",
                    "calibration": "profiles/demo.yaml",
                    "control_hz": 15,
                    "deadband": 0.1,
                    "gain_deg_s": 20,
                    "max_speed_deg_s": 25,
                    "max_acceleration_deg_s2": 50,
                    "observation_timeout_s": 0.2,
                    "home_speed_deg_s": 5,
                },
                "preview": {"enabled": False, "record": True, "port": 0},
            }
        )
        self.assertEqual(
            configuration.pan_tilt,
            PanTiltConfig("hardware", "profiles/demo.yaml", 15.0, 0.1, 20.0, 25.0, 50.0, 0.2, 5.0),
        )
        self.assertFalse(configuration.preview.enabled)
        self.assertTrue(configuration.preview.record)
        self.assertEqual(configuration.preview.port, 0)

    def test_unannotated_recording_is_explicit_and_field_checked(self) -> None:
        self.assertFalse(self.load({}).preview.record_unannotated)
        configuration = self.load({"preview": {"record": True, "record_unannotated": True}})
        self.assertTrue(configuration.preview.record_unannotated)
        with self.assertRaisesRegex(ConfigError, "preview.record_unannotated"):
            self.load({"preview": {"record_unannotated": "yes"}})

    def test_new_sections_reject_unknown_fields_and_malformed_sections(self) -> None:
        invalid_configurations: tuple[dict[str, JsonValue], ...] = (
            {"pan_tilt": []},
            {"pan_tilt": {"mode": "manual"}},
            {"pan_tilt": {"calibration": "  "}},
            {"pan_tilt": {"unexpected": True}},
            {"preview": []},
            {"preview": {"enabled": "false"}},
            {"preview": {"host": ""}},
            {"preview": {"unexpected": True}},
            {"zone": []},
            {"zone": {"unexpected": True}},
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration), self.assertRaises(ConfigError):
                self.load(configuration)

    def test_pan_tilt_range_and_cross_section_rules_are_enforced(self) -> None:
        invalid_settings: tuple[dict[str, JsonValue], ...] = (
            {"control_hz": 0},
            {"control_hz": float("inf")},
            {"deadband": -0.01},
            {"deadband": 0.5},
            {"deadband": True},
            {"gain_deg_s": 0},
            {"max_speed_deg_s": float("nan")},
            {"max_acceleration_deg_s2": -1},
            {"observation_timeout_s": 0},
            {"home_speed_deg_s": 0},
            {"home_speed_deg_s": 31},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                self.load({"pan_tilt": settings})

        with self.assertRaisesRegex(ConfigError, "when following"):
            self.load(
                {
                    "follow": {"enabled": True},
                    "confirmation": {"enabled": True},
                    "pan_tilt": {"observation_timeout_s": 0.8},
                }
            )
        with self.assertRaisesRegex(ConfigError, "camera switching"):
            self.load(
                {
                    "follow": {"enabled": True},
                    "confirmation": {"enabled": True},
                    "cameras": [
                        {"name": "day", "role": "day"},
                        {"name": "night", "role": "night"},
                    ],
                }
            )

    def test_preview_ranges_and_zone_deferral_are_enforced(self) -> None:
        invalid_settings: tuple[dict[str, JsonValue], ...] = (
            {"port": -1},
            {"port": 65536},
            {"port": True},
            {"max_fps": 10.1},
            {"max_fps": float("nan")},
            {"width_px": 0},
            {"height_px": -1},
            {"jpeg_quality": 0},
            {"jpeg_quality": 101},
            {"jpeg_quality": 80.0},
            {"record_max_duration_s": 0},
            {"record_max_duration_s": float("inf")},
            {"record_max_bytes": 0},
            {"record_max_bytes": -1},
            {"record_min_free_bytes": -1},
            {"export_timeout_s": 0},
            {"record": 1},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ConfigError):
                self.load({"preview": settings})
        with self.assertRaisesRegex(ConfigError, "hardware acceptance"):
            self.load({"zone": {"enabled": True}})

    def test_directly_constructed_configuration_is_validated(self) -> None:
        configuration = AppConfig(
            follow=FollowConfig(enabled=True, switch_hold_s=0),
            confirmation=ConfirmationConfig(enabled=True),
        )
        configuration.validate()
        invalid_configurations: tuple[AppConfig, ...] = (
            replace(configuration, follow=FollowConfig(retention_s=-1)),
            replace(
                configuration,
                pan_tilt=replace(
                    configuration.pan_tilt,
                    mode=cast(Literal["simulate", "hardware"], "invalid"),
                ),
            ),
            replace(configuration, preview=replace(configuration.preview, port=-1)),
            replace(configuration, zone=ZoneConfig(enabled=True)),
        )
        for invalid_configuration in invalid_configurations:
            with self.subTest(configuration=invalid_configuration), self.assertRaises(ConfigError):
                invalid_configuration.validate()


if __name__ == "__main__":
    unittest.main()
