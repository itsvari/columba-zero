from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from app.json_data import JsonValue
from tools.summarize_run import HISTOGRAM_RANGE_MS, summarize_log

MILLISECONDS_PER_SECOND = 1000.0
COMMAND_DURATION_S = 0.01
SENSOR_AGE_MS = 150.0
DELIVERY_AGE_MS = 75.0


def controller_record(
    started_at_s: float | None,
    *,
    mode: str = "hardware",
    state: str = "tracking",
    action: str | None = "move",
    timing_source: str = "sensor",
    age_ms: float | None = SENSOR_AGE_MS,
) -> dict[str, JsonValue]:
    return {
        "event": "controller_state",
        "mode": mode,
        "state": state,
        "driver_action": action,
        "timing_source": timing_source,
        "tick_started_at_monotonic_s": started_at_s,
        "driver_action_completed_at_monotonic_s": (
            started_at_s + COMMAND_DURATION_S if started_at_s is not None else None
        ),
        "tracking_observation_age_at_command_ms": age_ms,
    }


def summarise_records(path: Path, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return summarize_log(path)


def mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def test_controller_metrics_separate_modes_sources_and_missing_timing(tmp_path: Path) -> None:
    records: list[dict[str, JsonValue]] = [
        controller_record(None, state="acquiring", action=None, age_ms=None),
        controller_record(0.0),
        controller_record(10.0, mode="simulate", age_ms=1.0),
        controller_record(100.0, mode="fake", age_ms=1.0),
        controller_record(200.0, mode="replay", age_ms=1.0),
        controller_record(0.05, action="hold", age_ms=1.0),
        controller_record(0.1, timing_source="delivery", age_ms=DELIVERY_AGE_MS),
        controller_record(10.0625, mode="simulate", age_ms=1.0),
        controller_record(0.3, state="returning"),
        controller_record(0.35, state="paused"),
        controller_record(0.4, timing_source="unavailable"),
        controller_record(0.45, age_ms=None),
        controller_record(0.5) | {"driver_action_completed_at_monotonic_s": None},
        controller_record(None),
        controller_record(0.55, timing_source="replay"),
        {"event": "controller_state", "mode": "hardware", "state": "tracking"},
        {"event": "run_end", "error": None, "log_dropped_records": 4},
    ]
    report = summarise_records(tmp_path / "mixed.jsonl", records)
    metrics = mapping(report["controller_metrics"])
    assert metrics["live_samples_by_mode"] == {"hardware": 9, "simulate": 2}
    intervals = mapping(metrics["observed_tick_intervals_ms_by_mode"])
    hardware_intervals = mapping(intervals["hardware"])
    assert hardware_intervals["count"] == 8
    assert hardware_intervals["max"] == pytest.approx(200.0)
    assert hardware_intervals["mean"] == pytest.approx(0.55 * MILLISECONDS_PER_SECOND / 8)
    assert mapping(intervals["simulate"])["max"] == 62.5
    assert metrics["hardware_tracking_position_moves"] == 7
    assert metrics["untimed_hardware_tracking_position_moves"] == 5
    command_ages = mapping(metrics["tracking_observation_age_at_command_ms_by_timing_source"])
    sensor_ages = mapping(command_ages["sensor"])
    delivery_ages = mapping(command_ages["delivery"])
    assert sensor_ages["count"] == delivery_ages["count"] == 1
    assert sensor_ages["mean"] == sensor_ages["p95"] == SENSOR_AGE_MS
    assert delivery_ages["mean"] == delivery_ages["p95"] == DELIVERY_AGE_MS
    assert "run_end.log_dropped_records" in str(metrics["basis"])
    assert "widen observed tick intervals" in str(metrics["basis"])
    assert "Counts are lower bounds" in str(metrics["basis"])
    assert "upper bound on controller sample loss" in str(metrics["basis"])
    assert "Ages are per command" in str(metrics["basis"])
    assert mapping(report["run_end"])["log_dropped_records"] == 4
    assert report["completed_cleanly"] is True


def test_replay_legacy_and_simulation_provide_no_hardware_latency_evidence(tmp_path: Path) -> None:
    legacy_records: list[dict[str, JsonValue]] = [
        {"event": "controller_state", "mode": "hardware", "state": "tracking"},
        {
            "event": "controller_state",
            "mode": "simulate",
            "state": "tracking",
            "timing_source": "replay",
            "observation_age_ms": 50.0,
        },
        {"event": "frame", "inference_s": 0.01, "stage_ms": {"track": 2.0}},
        {"event": "run_end", "error": None},
    ]
    report = summarise_records(tmp_path / "legacy.jsonl", legacy_records)
    metrics = mapping(report["controller_metrics"])
    assert metrics["live_samples_by_mode"] == {"hardware": 0, "simulate": 0}
    assert metrics["observed_tick_intervals_ms_by_mode"] == {"hardware": None, "simulate": None}
    assert metrics["hardware_tracking_position_moves"] == 0
    assert metrics["untimed_hardware_tracking_position_moves"] == 0
    assert metrics["tracking_observation_age_at_command_ms_by_timing_source"] == {
        "sensor": None,
        "delivery": None,
    }
    assert report["frames"] == 1
    assert report["inference_mean_ms"] == 10.0
    distributions = mapping(report["distributions_ms"])
    assert set(distributions) == {
        "inference_total",
        "wait",
        "preprocess",
        "inference",
        "postprocess",
        "track",
        "delivery",
        "decision",
        "reaction",
    }
    assert mapping(distributions["track"])["max"] == 2.0

    simulated = summarise_records(
        tmp_path / "simulated.jsonl",
        [controller_record(1.0, mode="simulate"), controller_record(1.05, mode="simulate")],
    )
    simulated_metrics = mapping(simulated["controller_metrics"])
    assert mapping(simulated_metrics["live_samples_by_mode"])["simulate"] == 2
    assert simulated_metrics["hardware_tracking_position_moves"] == 0
    assert simulated_metrics["tracking_observation_age_at_command_ms_by_timing_source"] == {
        "sensor": None,
        "delivery": None,
    }


def test_all_explicit_hardware_tracking_moves_without_usable_timing_are_counted(
    tmp_path: Path,
) -> None:
    records: list[dict[str, JsonValue]] = [
        {
            "event": "controller_state",
            "mode": "hardware",
            "state": "tracking",
            "driver_action": "move",
        },
        controller_record(None) | {"driver_action_completed_at_monotonic_s": 1.0},
        controller_record(1.0) | {"driver_action_completed_at_monotonic_s": None},
        controller_record(1.05, age_ms=None),
        controller_record(1.1, timing_source="unavailable"),
        controller_record(1.15, timing_source="replay"),
        controller_record(1.2) | {"timing_source": None},
    ]
    metrics = mapping(summarise_records(tmp_path / "untimed.jsonl", records)["controller_metrics"])
    assert metrics["hardware_tracking_position_moves"] == len(records)
    assert metrics["untimed_hardware_tracking_position_moves"] == len(records)
    assert metrics["tracking_observation_age_at_command_ms_by_timing_source"] == {
        "sensor": None,
        "delivery": None,
    }


def test_controller_numeric_boundary_rejects_bad_values_even_for_excluded_modes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.jsonl"
    timing_fields = (
        "tick_started_at_monotonic_s",
        "driver_action_completed_at_monotonic_s",
        "tracking_observation_age_at_command_ms",
    )
    malformed_values: tuple[JsonValue, ...] = (
        -1.0,
        True,
        "unknown",
        float("nan"),
        float("inf"),
        float("-inf"),
        [],
        {},
    )
    for field in timing_fields:
        for value in malformed_values:
            record = controller_record(1.0, mode="simulate") | {field: value}
            with pytest.raises(ValueError, match=field) as raised:
                summarise_records(path, [record])
            assert str(raised.value).startswith(f"{path}:1: ")

    with pytest.raises(ValueError, match="must be >= tick start"):
        summarise_records(
            path,
            [controller_record(1.0) | {"driver_action_completed_at_monotonic_s": 0.5}],
        )
    with pytest.raises(ValueError, match="tick intervals must be finite and non-negative"):
        summarise_records(path, [controller_record(1.0), controller_record(0.5)])


def test_controller_histograms_use_bounded_overflow_for_intervals_and_command_ages(
    tmp_path: Path,
) -> None:
    overflow_exact_limit = 2
    overflow_age_ms = HISTOGRAM_RANGE_MS * 2
    interval_s = overflow_age_ms / MILLISECONDS_PER_SECOND
    sample_count = 6
    with patch("tools.summarize_run.OVERFLOW_EXACT_LIMIT", overflow_exact_limit):
        report = summarise_records(
            tmp_path / "overflow.jsonl",
            [
                controller_record(index * interval_s, age_ms=overflow_age_ms)
                for index in range(sample_count)
            ],
        )
    metrics = mapping(report["controller_metrics"])
    intervals = mapping(mapping(metrics["observed_tick_intervals_ms_by_mode"])["hardware"])
    sensor_ages = mapping(
        mapping(metrics["tracking_observation_age_at_command_ms_by_timing_source"])["sensor"]
    )
    assert intervals["count"] == sample_count - 1
    assert sensor_ages["count"] == sample_count
    for distribution in (intervals, sensor_ages):
        assert distribution["p95"] == HISTOGRAM_RANGE_MS
        assert distribution["max"] == overflow_age_ms
        assert "p95" in cast(list[str], distribution["lower_bound"])
