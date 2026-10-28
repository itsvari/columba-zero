"""Assess a captured endurance run offline against its pinned policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import cast

from app.json_data import JsonValue

SOURCE_TIMESTAMP_TOLERANCE_S = 0.000001
MAXIMUM_SAMPLE_INTERVAL_MULTIPLIER = 2
MILLISECONDS_PER_SECOND = 1000.0
UNDERVOLTAGE_CURRENT = 1
UNDERVOLTAGE_LATCH = 1 << 16
ROOT = Path(__file__).resolve().parents[1]


def mapping(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def read_json(path: Path) -> dict[str, JsonValue]:
    return mapping(cast(JsonValue, json.loads(path.read_bytes())))


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    with path.open(encoding="utf-8") as stream:
        return [mapping(cast(JsonValue, json.loads(line))) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def verdict(checks: list[bool | None]) -> str:
    if False in checks:
        return "fail"
    return "inconclusive" if None in checks or not checks else "pass"


def throttle(value: JsonValue) -> int | None:
    if isinstance(value, str):
        try:
            return int(value.removeprefix("throttled="), 0)
        except ValueError:
            return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def sensor_ticks(
    records: list[dict[str, JsonValue]],
) -> tuple[list[dict[str, JsonValue]], int, list[float]]:
    ticks: list[dict[str, JsonValue]] = []
    invalid = 0
    intervals: list[float] = []
    previous_tick: float | None = None
    newest_observation: float | None = None
    for record in records:
        if record.get("event") != "controller_state" or record.get("mode") != "hardware":
            continue
        tick_s = number(record.get("tick_started_at_monotonic_s"))
        state = record.get("state")
        sensor = record.get("timing_source") == "sensor" and state in ("tracking", "stale")
        if tick_s is None:
            # startup and close status events are not controller ticks
            if sensor:
                invalid += 1
            continue
        if previous_tick is not None:
            interval = tick_s - previous_tick
            if interval <= 0:
                invalid += 1
            else:
                intervals.append(interval)
        previous_tick = tick_s
        if state in ("tracking", "stale") and not sensor:
            invalid += 1
        age_ms = number(record.get("observation_age_ms")) if sensor else None
        first_use = False
        if sensor:
            if age_ms is None or age_ms < 0:
                invalid += 1
            else:
                observed_at_s = tick_s - age_ms / MILLISECONDS_PER_SECOND
                if newest_observation is None or observed_at_s > (
                    newest_observation + SOURCE_TIMESTAMP_TOLERANCE_S
                ):
                    first_use = True
                    newest_observation = observed_at_s
                elif observed_at_s < newest_observation - SOURCE_TIMESTAMP_TOLERANCE_S:
                    invalid += 1
        ticks.append(
            {
                "at_s": tick_s,
                "sensor": sensor,
                "state": state,
                "age_ms": age_ms,
                "first_use": first_use,
            }
        )
    return ticks, invalid, intervals


def window_metrics(
    start_s: float,
    end_s: float,
    frames_s: list[float],
    ticks: list[dict[str, JsonValue]],
    limits: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    ages: list[float] = []
    first_use: list[float] = []
    tracked_s = 0.0
    stale_s = 0.0
    for index, tick in enumerate(ticks):
        tick_s = number(tick["at_s"])
        assert tick_s is not None
        if tick["sensor"] is not True:
            continue
        age = number(tick["age_ms"])
        if start_s <= tick_s < end_s and age is not None:
            ages.append(age)
            if tick["first_use"] is True:
                first_use.append(age)
        if index + 1 == len(ticks):
            continue
        next_s = number(ticks[index + 1]["at_s"])
        assert next_s is not None
        duration_s = max(0.0, min(end_s, next_s) - max(start_s, tick_s))
        tracked_s += duration_s
        if tick["state"] == "stale":
            stale_s += duration_s
    first_p95 = percentile_95(first_use)
    age_p95 = percentile_95(ages)
    stale_fraction = stale_s / tracked_s if tracked_s else None
    frame_count = sum(start_s <= frame < end_s for frame in frames_s)
    fps = frame_count / (end_s - start_s)
    checks: dict[str, bool | None] = {
        "first_use_age": first_p95
        <= float(cast(float, limits["first_use_sensor_age_at_tick_p95_ms_max"]))
        if first_p95 is not None
        else None,
        "uncensored_age": age_p95
        <= float(cast(float, limits["uncensored_tracking_stale_age_p95_ms_max"]))
        if age_p95 is not None
        else None,
        "stale_fraction": stale_fraction
        <= float(cast(float, limits["sensor_tracking_stale_fraction_max"]))
        if stale_fraction is not None
        else None,
        "processed_fps": fps >= float(cast(float, limits["processed_fps_min"])),
    }
    tick_coverage = bool(ticks) and (
        cast(float, ticks[0]["at_s"]) <= start_s and cast(float, ticks[-1]["at_s"]) >= end_s
    )
    if not tick_coverage:
        checks["uncensored_age"] = None
        checks["first_use_age"] = None
        checks["stale_fraction"] = None
    return {
        "tick_coverage": tick_coverage,
        "start_monotonic_s": start_s,
        "end_monotonic_s": end_s,
        "duration_s": end_s - start_s,
        "first_use_count": len(first_use),
        "first_use_p95_ms": first_p95,
        "uncensored_tick_count": len(ages),
        "uncensored_age_p95_ms": age_p95,
        "sensor_tracking_stale_duration_s": tracked_s,
        "sensor_stale_duration_s": stale_s,
        "stale_fraction": stale_fraction,
        "processed_frame_count": frame_count,
        "processed_fps": fps,
        "checks": cast(JsonValue, checks),
        "verdict": verdict(list(checks.values())),
    }


def recording_integrity(records: list[dict[str, JsonValue]]) -> dict[str, bool | None]:
    ends = [record for record in records if record.get("event") == "run_end"]
    if len(ends) != 1:
        return {"run_end": None}
    end = ends[0]
    demo = end.get("demo_recording")
    export = end.get("demo_export")
    if not isinstance(demo, dict) or not isinstance(export, dict):
        return {"recording": None}
    return {
        "clean_run": (end.get("reason") == "stopped" and end.get("error") is None)
        if "reason" in end and "error" in end
        else None,
        "log_drops": end.get("log_dropped_records") == 0
        if isinstance(end.get("log_dropped_records"), int)
        else None,
        "render_drops": end.get("render_dropped_frames") == 0
        if isinstance(end.get("render_dropped_frames"), int)
        else None,
        "record_drops": demo.get("frames_dropped") == 0
        if isinstance(demo.get("frames_dropped"), int)
        else None,
        "recording_finished": demo.get("state") == "finished" and demo.get("error") is None,
        "record_limit": demo.get("limit_reason") is None if "limit_reason" in demo else None,
        "export": export.get("success") is True and export.get("error") is None,
    }


def browser_integrity(records: list[dict[str, JsonValue]], repeats: int) -> bool | None:
    sessions = [record for record in records if record.get("kind") == "session"]
    completes = [record for record in records if record.get("event") == "complete"]
    if len(sessions) != 1 or len(completes) != 1:
        return None
    manifest = sessions[0].get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("scenes"), list):
        return None
    scenes = cast(list[JsonValue], manifest["scenes"])
    if not all(
        isinstance(scene, dict) and isinstance(scene.get("duration_frames"), int)
        for scene in scenes
    ):
        return None
    frames_per_cycle = sum(
        cast(int, cast(dict[str, JsonValue], scene)["duration_frames"]) for scene in scenes
    )
    complete = completes[0]
    observed = {
        record.get("repeat_index")
        for record in records
        if record.get("kind") == "frame" and record.get("run_number") == complete.get("run_number")
    }
    return (
        manifest.get("repeat_count") == repeats
        and observed == set(range(repeats))
        and complete.get("repeat_index") == repeats - 1
        and complete.get("sequence_frame") == repeats * frames_per_cycle - 1
        and complete.get("presentation_id") == sessions[0].get("presentation_id")
    )


def telemetry_metrics(
    records: list[dict[str, JsonValue]],
    baseline: dict[str, JsonValue],
    final_voltage: dict[str, JsonValue],
    start_s: float,
    end_s: float,
    hot_start_s: float,
    window_s: float,
) -> tuple[dict[str, JsonValue], list[bool | None]]:
    samples = [record for record in records if record.get("kind") == "sample"]
    valid_times = [number(record.get("monotonic_s")) for record in samples]
    if not samples or any(value is None for value in valid_times):
        return {"error": "telemetry samples or their timestamps unavailable"}, [None]
    times = [cast(float, value) for value in valid_times]
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        return {"error": "telemetry timestamps are not strictly increasing"}, [None]
    in_capture = [
        (record, at_s)
        for record, at_s in zip(samples, times, strict=True)
        if start_s <= at_s <= end_s
    ]
    flags = [throttle(record.get("throttled")) for record, _ in in_capture]
    baseline_flags = throttle(baseline.get("throttled"))
    baseline_at_s = number(baseline.get("monotonic_s"))
    baseline_clear = (
        baseline_flags is not None
        and baseline_at_s is not None
        and baseline_at_s <= start_s
        and not baseline_flags & (UNDERVOLTAGE_CURRENT | UNDERVOLTAGE_LATCH)
    )
    covered = bool(in_capture) and times[0] <= start_s and times[-1] >= end_s
    final_flags = throttle(final_voltage.get("throttled"))
    final_at_s = number(final_voltage.get("monotonic_s"))
    boot_id = baseline.get("boot_id")
    same_boot = (
        isinstance(boot_id, str) and bool(boot_id) and final_voltage.get("boot_id") == boot_id
    )
    latch_coverage = (
        baseline_clear
        and same_boot
        and final_flags is not None
        and final_at_s is not None
        and final_at_s >= end_s
    )
    sampled_current = any(flag is not None and flag & UNDERVOLTAGE_CURRENT for flag in flags)
    sampled_latch = any(flag is not None and flag & UNDERVOLTAGE_LATCH for flag in flags)
    current_ok: bool | None = (
        False
        if sampled_current
        else (
            not bool(cast(int, final_flags) & (UNDERVOLTAGE_CURRENT | UNDERVOLTAGE_LATCH))
            if latch_coverage
            else None
        )
    )
    latch_ok: bool | None = (
        False
        if sampled_latch and baseline_clear and same_boot
        else (not bool(cast(int, final_flags) & UNDERVOLTAGE_LATCH) if latch_coverage else None)
    )
    sessions = [record for record in records if record.get("kind") == "session"]
    requested_interval_s = (
        number(sessions[0].get("requested_interval_s")) if len(sessions) == 1 else None
    )
    hot_gaps_s = [
        right - left
        for left, right in zip(times, times[1:], strict=False)
        if right > hot_start_s and left < end_s
    ]
    max_hot_gap_s = max(hot_gaps_s) if hot_gaps_s else None
    hot_coverage = (
        times[0] <= hot_start_s
        and times[-1] >= end_s
        and requested_interval_s is not None
        and requested_interval_s > 0
        and max_hot_gap_s is not None
        and max_hot_gap_s <= requested_interval_s * MAXIMUM_SAMPLE_INTERVAL_MULTIPLIER
    )
    first_rss = [
        number(record.get("process_memory_rss_bytes"))
        for record, at_s in in_capture
        if hot_start_s <= at_s < hot_start_s + window_s
    ]
    last_rss = [
        number(record.get("process_memory_rss_bytes"))
        for record, at_s in in_capture
        if end_s - window_s <= at_s <= end_s and at_s >= hot_start_s
    ]
    rss_valid = (
        end_s - hot_start_s >= window_s
        and hot_coverage
        and bool(first_rss)
        and bool(last_rss)
        and all(value is not None and value > 0 for value in first_rss + last_rss)
    )
    first_median = statistics.median(cast(list[float], first_rss)) if rss_valid else None
    last_median = statistics.median(cast(list[float], last_rss)) if rss_valid else None
    temperatures = [number(record.get("temperature_millicelsius")) for record, _ in in_capture]
    clocks = [number(record.get("cpu_frequency_khz")) for record, _ in in_capture]
    return (
        {
            "capture_samples": len(in_capture),
            "capture_covered": covered,
            "hot_sampling_covered": hot_coverage,
            "requested_interval_s": requested_interval_s,
            "max_hot_sample_gap_s": max_hot_gap_s,
            "baseline_clear": baseline_clear,
            "same_boot": same_boot,
            "latch_covers_capture": latch_coverage,
            "final_throttled": final_flags,
            "current_undervoltage_clear": current_ok,
            "new_latch_clear": latch_ok,
            "first_rss_median_bytes": first_median,
            "last_rss_median_bytes": last_median,
            "rss_median_increase_bytes": last_median - first_median
            if first_median is not None and last_median is not None
            else None,
            "temperature_millicelsius_range": [
                min(cast(list[float], temperatures)),
                max(cast(list[float], temperatures)),
            ]
            if temperatures and all(x is not None for x in temperatures)
            else None,
            "cpu_frequency_khz_range": [
                min(cast(list[float], clocks)),
                max(cast(list[float], clocks)),
            ]
            if clocks and all(x is not None for x in clocks)
            else None,
            "throttle_values": cast(
                JsonValue, sorted({flag for flag in flags if flag is not None})
            ),
        },
        [current_ok, latch_ok, True if hot_coverage else None],
    )


def evaluate(
    detections: list[dict[str, JsonValue]],
    telemetry: list[dict[str, JsonValue]],
    supervisor: dict[str, JsonValue],
    full_off: dict[str, JsonValue],
    browser: list[dict[str, JsonValue]],
    baseline: dict[str, JsonValue],
    final_voltage: dict[str, JsonValue],
    decode: dict[str, JsonValue],
    video_hash: str,
    policy: dict[str, JsonValue],
    truth_hashes_valid: bool,
) -> dict[str, JsonValue]:
    limits = mapping(policy["engineering_gates"])
    start_unix = number(supervisor.get("started_at_unix_s"))
    start_monotonic = number(supervisor.get("started_at_monotonic_s"))
    if start_unix is None or start_monotonic is None:
        raise ValueError("supervisor clock pair unavailable; capture cannot be located")
    offset_s = start_unix - start_monotonic
    frames_s: list[float] = []
    invalid_frames = 0
    for record in detections:
        if record.get("event") != "frame":
            continue
        captured_at = number(record.get("captured_at"))
        if captured_at is None:
            invalid_frames += 1
        else:
            frames_s.append(captured_at - offset_s)
    if len(frames_s) < 2 or any(b <= a for a, b in zip(frames_s, frames_s[1:], strict=False)):
        raise ValueError("processed frame timestamps missing or non-increasing")
    capture_start_s, capture_end_s = frames_s[0], frames_s[-1]
    ticks, invalid_ticks, intervals = sensor_ticks(detections)
    capture_intervals = [
        (cast(float, b["at_s"]) - cast(float, a["at_s"])) * MILLISECONDS_PER_SECOND
        for a, b in zip(ticks, ticks[1:], strict=False)
        if cast(float, a["at_s"]) >= capture_start_s
        and cast(float, b["at_s"]) <= capture_end_s
        and cast(float, b["at_s"]) > cast(float, a["at_s"])
    ]
    max_interval = max(capture_intervals) if capture_intervals else None
    window_s = float(cast(float, limits["window_s"]))
    hot_start_s = capture_start_s + float(cast(float, limits["warmup_s"]))
    minimum_window_s = float(cast(float, limits["minimum_final_window_s"]))
    windows: list[dict[str, JsonValue]] = []
    cursor_s = hot_start_s
    while cursor_s < capture_end_s:
        end_s = min(cursor_s + window_s, capture_end_s)
        result = window_metrics(cursor_s, end_s, frames_s, ticks, limits)
        if end_s - cursor_s < minimum_window_s:
            result["verdict"] = "excluded_short_final_window"
        windows.append(result)
        cursor_s = end_s
    whole_hot = (
        window_metrics(hot_start_s, capture_end_s, frames_s, ticks, limits)
        if hot_start_s < capture_end_s
        else None
    )
    metrics, telemetry_checks = telemetry_metrics(
        telemetry, baseline, final_voltage, capture_start_s, capture_end_s, hot_start_s, window_s
    )
    increase = number(metrics.get("rss_median_increase_bytes"))
    memory_ok = (
        increase <= float(cast(float, limits["rss_median_increase_bytes_max"]))
        if increase is not None
        else None
    )
    max_ok = (
        max_interval <= float(cast(float, limits["controller_interval_ms_max"]))
        if max_interval is not None
        else None
    )
    integrity = recording_integrity(detections)
    registers = full_off.get("registers")
    if isinstance(registers, dict):
        pan = registers.get("9")
        tilt = registers.get("13")
        registers_off = (
            bool(pan & 16) and bool(tilt & 16)
            if isinstance(pan, int) and isinstance(tilt, int)
            else None
        )
    else:
        registers_off = None
    integrity.update(
        {
            "supervisor_exit": supervisor.get("child_exit_status") == 0
            if isinstance(supervisor.get("child_exit_status"), int)
            else None,
            "term_once": supervisor.get("term_sent_count") == 1
            if isinstance(supervisor.get("term_sent_count"), int)
            else None,
            "no_forced_kill": supervisor.get("forced_kill") is False
            if isinstance(supervisor.get("forced_kill"), bool)
            else None,
            "full_off": (full_off.get("both_full_off") is True and registers_off)
            if isinstance(full_off.get("both_full_off"), bool) and registers_off is not None
            else None,
            "playlist": browser_integrity(
                browser, int(cast(int, limits["all_playlist_repeats_required"]))
            ),
            "independent_decode": (
                decode.get("success") is True
                and decode.get("video_sha256") == video_hash
                and decode.get("independent") is True
            )
            if all(key in decode for key in ("success", "video_sha256", "independent"))
            else None,
        }
    )
    window_verdicts = [
        window["verdict"]
        for window in windows
        if window["verdict"] != "excluded_short_final_window"
    ]
    checks: list[bool | None] = (
        list(integrity.values()) + telemetry_checks + [memory_ok, max_ok, truth_hashes_valid]
    )
    checks.append(True if invalid_frames == 0 and invalid_ticks == 0 and intervals else None)
    checks.append(
        False
        if "fail" in window_verdicts
        else None
        if not window_verdicts or "inconclusive" in window_verdicts
        else True
    )
    if whole_hot is None or whole_hot["verdict"] == "inconclusive":
        checks.append(None)
    else:
        checks.append(whole_hot["verdict"] == "pass")
    return {
        "scope": "endurance timing and stability only",
        "endurance_verdict": verdict(checks),
        "global_acceptance": "inconclusive",
        "integrity": cast(JsonValue, integrity),
        "invalid_frame_timestamps": invalid_frames,
        "invalid_controller_ticks": invalid_ticks,
        "clock_offset_unix_minus_monotonic_s": offset_s,
        "capture_start_monotonic_s": capture_start_s,
        "capture_end_monotonic_s": capture_end_s,
        "whole_capture_max_tick_interval_ms": max_interval,
        "whole_capture_interval_check": max_ok,
        "whole_hot": whole_hot,
        "windows": cast(JsonValue, windows),
        "telemetry": metrics,
        "rss_check": memory_ok,
        "frozen_truth_hashes_match": truth_hashes_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--full-off", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--voltage-final", type=Path, required=True)
    parser.add_argument("--decode-proof", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "deploy/monitor-endurance-policy.json"
    )
    parser.add_argument("--policy-sha256", required=True, help="hash frozen in the run plan")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "detections": args.run_directory / "detections.jsonl",
        "telemetry": args.telemetry,
        "supervisor": args.supervisor,
        "full_off": args.full_off,
        "browser": args.browser,
        "baseline": args.baseline,
        "voltage_final": args.voltage_final,
        "decode_proof": args.decode_proof,
        "video": args.video,
        "policy": args.policy,
    }
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if hashes["policy"] != args.policy_sha256:
        raise ValueError("policy SHA-256 does not match the pinned acceptance policy")
    policy = read_json(args.policy)
    assessment_tool = mapping(policy.get("assessment_tool"))
    if assessment_tool.get("sha256") != sha256(Path(__file__)):
        raise ValueError("assessment tool differs from the policy's frozen version")
    truth = mapping(mapping(policy["centring_confirmation"])["frozen_files_sha256"])
    truth_hashes: dict[str, JsonValue] = {}
    for name, expected in truth.items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else None
        truth_hashes[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    result = evaluate(
        read_jsonl(inputs["detections"]),
        read_jsonl(args.telemetry),
        read_json(args.supervisor),
        read_json(args.full_off),
        read_jsonl(args.browser),
        read_json(args.baseline),
        read_json(args.voltage_final),
        read_json(args.decode_proof),
        hashes["video"],
        policy,
        all(mapping(value)["matches"] is True for value in truth_hashes.values()),
    )
    result["truth_hashes"] = truth_hashes
    args.output_directory.mkdir(parents=True, exist_ok=False)
    report_path = args.output_directory / "assessment.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": hashes[name]}
            for name, path in inputs.items()
        },
        "outputs": {"assessment.json": sha256(report_path)},
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "endurance_verdict": result["endurance_verdict"],
                "assessment": str(report_path),
                "global_acceptance": "inconclusive",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
