from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from tools import assess_monitor_endurance as assessment


def evidence(
    duration_s: int = 770,
) -> tuple[
    list[dict[str, JsonValue]],
    list[dict[str, JsonValue]],
    dict[str, JsonValue],
    dict[str, JsonValue],
    list[dict[str, JsonValue]],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
]:
    offset_s = 1000000
    start_s = 1000
    records: list[dict[str, JsonValue]] = [
        {"event": "frame", "captured_at": offset_s + start_s + index / 11}
        for index in range(duration_s * 11 + 1)
    ]
    records.extend(
        {
            "event": "controller_state",
            "mode": "hardware",
            "state": "tracking",
            "timing_source": "sensor",
            "tick_started_at_monotonic_s": start_s + index / 20,
            "observation_age_ms": 100 if index % 2 == 0 else 150,
        }
        for index in range(duration_s * 20 + 1)
    )
    records.append(
        {
            "event": "run_end",
            "reason": "stopped",
            "error": None,
            "log_dropped_records": 0,
            "render_dropped_frames": 0,
            "demo_recording": {
                "state": "finished",
                "frames_dropped": 0,
                "error": None,
                "limit_reason": None,
            },
            "demo_export": {"success": True, "error": None},
        }
    )
    telemetry: list[dict[str, JsonValue]] = [
        {
            "kind": "sample",
            "monotonic_s": start_s + index,
            "throttled": "throttled=0x0",
            "process_memory_rss_bytes": 1000000,
            "temperature_millicelsius": 50000,
            "cpu_frequency_khz": 2000000,
        }
        for index in range(duration_s + 1)
    ]
    telemetry.insert(0, {"kind": "session", "requested_interval_s": 1.0})
    telemetry.append(
        {
            "kind": "sample",
            "monotonic_s": start_s + duration_s + 1,
            "throttled": "throttled=0x0",
            "process_memory_rss_bytes": 999999999,
        }
    )
    supervisor: dict[str, JsonValue] = {
        "started_at_unix_s": offset_s + start_s,
        "started_at_monotonic_s": start_s,
        "child_exit_status": 0,
        "term_sent_count": 1,
        "forced_kill": False,
    }
    browser: list[dict[str, JsonValue]] = [
        {
            "kind": "session",
            "presentation_id": "abc",
            "manifest": {"repeat_count": 18, "scenes": [{"duration_frames": 1}]},
        },
        {
            "kind": "event",
            "event": "complete",
            "run_number": 1,
            "repeat_index": 17,
            "sequence_frame": 17,
            "presentation_id": "abc",
        },
    ]
    browser.extend({"kind": "frame", "run_number": 1, "repeat_index": index} for index in range(18))
    policy = cast(
        dict[str, JsonValue],
        json.loads((assessment.ROOT / "deploy/monitor-endurance-policy.json").read_text()),
    )
    return (
        records,
        telemetry,
        supervisor,
        {"both_full_off": True, "registers": {"9": 16, "13": 16}},
        browser,
        {"monotonic_s": start_s - 1, "throttled": "throttled=0x0", "boot_id": "boot"},
        {"success": True, "independent": True, "video_sha256": "video"},
        policy,
    )


def evaluate_fixture(
    data: tuple[
        list[dict[str, JsonValue]],
        list[dict[str, JsonValue]],
        dict[str, JsonValue],
        dict[str, JsonValue],
        list[dict[str, JsonValue]],
        dict[str, JsonValue],
        dict[str, JsonValue],
        dict[str, JsonValue],
    ],
) -> dict[str, JsonValue]:
    records, telemetry, supervisor, full_off, browser, baseline, decode, policy = data
    return assessment.evaluate(
        records,
        telemetry,
        supervisor,
        full_off,
        browser,
        baseline,
        {
            "monotonic_s": telemetry[-1]["monotonic_s"],
            "throttled": "throttled=0x0",
            "boot_id": "boot",
        },
        decode,
        "video",
        policy,
        True,
    )


def test_pass_and_excluded_partial_remains_in_whole_hot() -> None:
    report = evaluate_fixture(evidence())
    assert report["endurance_verdict"] == "pass"
    assert report["global_acceptance"] == "inconclusive"
    windows = cast(list[dict[str, JsonValue]], report["windows"])
    assert len(windows) == 2
    assert windows[0]["duration_s"] == 600
    assert windows[1]["duration_s"] == 50
    assert windows[1]["verdict"] == "excluded_short_final_window"
    assert assessment.mapping(report["whole_hot"])["duration_s"] == 650
    assert assessment.mapping(report["telemetry"])["rss_median_increase_bytes"] == 0


def test_minute_partial_is_gated() -> None:
    report = evaluate_fixture(evidence(780))
    windows = cast(list[dict[str, JsonValue]], report["windows"])
    assert windows[1]["duration_s"] == 60
    assert windows[1]["verdict"] == "pass"


def test_loss_and_missing_baseline_cannot_pass() -> None:
    data = evidence()
    data[0][-1]["log_dropped_records"] = 1
    assert evaluate_fixture(data)["endurance_verdict"] == "fail"
    data = evidence()
    data[5].clear()
    assert evaluate_fixture(data)["endurance_verdict"] == "inconclusive"


def test_timestamp_deduplication_does_not_depend_on_age_drop() -> None:
    records: list[dict[str, JsonValue]] = [
        {
            "event": "controller_state",
            "mode": "hardware",
            "state": "tracking",
            "timing_source": "sensor",
            "tick_started_at_monotonic_s": 10.0,
            "observation_age_ms": 100,
        },
        {
            "event": "controller_state",
            "mode": "hardware",
            "state": "tracking",
            "timing_source": "sensor",
            "tick_started_at_monotonic_s": 10.05,
            "observation_age_ms": 150,
        },
        {
            "event": "controller_state",
            "mode": "hardware",
            "state": "tracking",
            "timing_source": "sensor",
            "tick_started_at_monotonic_s": 10.10,
            "observation_age_ms": 150,
        },
    ]
    ticks, invalid, _ = assessment.sensor_ticks(records)
    assert invalid == 0
    assert [tick["first_use"] for tick in ticks] == [True, False, True]


def test_policy_hash_mismatch_refuses_assessment(tmp_path: Path) -> None:
    file = tmp_path / "evidence"
    file.write_text("{}")
    (tmp_path / "detections.jsonl").write_text("{}\n")
    command = [
        sys.executable,
        str(assessment.ROOT / "tools/assess_monitor_endurance.py"),
        "--run-directory",
        str(tmp_path),
        "--output-directory",
        str(tmp_path / "out"),
    ]
    for option in (
        "telemetry",
        "supervisor",
        "full-off",
        "browser",
        "baseline",
        "voltage-final",
        "decode-proof",
        "video",
        "policy",
    ):
        command.extend([f"--{option}", str(file)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "policy SHA-256 does not match" in result.stderr
    assert not (tmp_path / "out").exists()


def test_half_open_boundary_counts_each_frame_once() -> None:
    policy = evidence()[7]
    limits = assessment.mapping(policy["engineering_gates"])
    first = assessment.window_metrics(0, 1, [0, 1, 2], [], limits)
    second = assessment.window_metrics(1, 2, [0, 1, 2], [], limits)
    assert first["processed_frame_count"] == 1
    assert second["processed_frame_count"] == 1


def test_clear_same_boot_latch_covers_early_telemetry_gap() -> None:
    data = evidence()
    del data[1][1]
    report = evaluate_fixture(data)
    telemetry = assessment.mapping(report["telemetry"])
    assert telemetry["capture_covered"] is False
    assert telemetry["latch_covers_capture"] is True
    assert telemetry["current_undervoltage_clear"] is True
    assert telemetry["new_latch_clear"] is True


def test_asserted_final_latch_fails_and_changed_boot_is_inconclusive() -> None:
    data = evidence()
    records, samples, supervisor, full_off, browser, baseline, decode, policy = data
    failed = assessment.evaluate(
        records,
        samples,
        supervisor,
        full_off,
        browser,
        baseline,
        {"monotonic_s": 1771, "throttled": "0x10000", "boot_id": "boot"},
        decode,
        "video",
        policy,
        True,
    )
    assert failed["endurance_verdict"] == "fail"
    changed = assessment.evaluate(
        records,
        samples,
        supervisor,
        full_off,
        browser,
        baseline,
        {"monotonic_s": 1771, "throttled": "0x0", "boot_id": "other"},
        decode,
        "video",
        policy,
        True,
    )
    assert changed["endurance_verdict"] == "inconclusive"


def test_cli_writes_hashed_report_and_preserves_inputs(tmp_path: Path) -> None:
    import hashlib

    records, telemetry, supervisor, full_off, browser, baseline, decode, _ = evidence()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic fixture")
    decode["video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    final: dict[str, JsonValue] = {
        "monotonic_s": 1771,
        "throttled": "0x0",
        "boot_id": "boot",
    }
    jsonl = {"detections": records, "telemetry": telemetry, "browser": browser}
    for name, values in jsonl.items():
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    for name, value in {
        "supervisor": supervisor,
        "full-off": full_off,
        "baseline": baseline,
        "voltage-final": final,
        "decode-proof": decode,
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "assessment"
    command = [
        sys.executable,
        str(assessment.ROOT / "tools/assess_monitor_endurance.py"),
        "--run-directory",
        str(tmp_path),
        "--output-directory",
        str(output),
        "--video",
        str(video),
    ]
    for name in ("telemetry", "browser"):
        command.extend([f"--{name}", str(tmp_path / f"{name}.jsonl")])
    for name in ("supervisor", "full-off", "baseline", "voltage-final", "decode-proof"):
        command.extend([f"--{name}", str(tmp_path / f"{name}.json")])
    original = (tmp_path / "detections.jsonl").read_bytes()
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert (tmp_path / "detections.jsonl").read_bytes() == original
    manifest = assessment.read_json(output / "manifest.json")
    outputs = assessment.mapping(manifest["outputs"])
    assert outputs["assessment.json"] == assessment.sha256(output / "assessment.json")
    assert len(assessment.mapping(manifest["inputs"])) == 10
    assert assessment.read_json(output / "assessment.json")["global_acceptance"] == "inconclusive"


def test_missing_hot_telemetry_cannot_pass_memory_coverage() -> None:
    data = evidence()
    data[1][:] = [
        sample
        for sample in data[1]
        if sample.get("kind") != "sample" or not 1300 < cast(float, sample["monotonic_s"]) < 1400
    ]
    report = evaluate_fixture(data)
    assert report["endurance_verdict"] == "inconclusive"
    telemetry = assessment.mapping(report["telemetry"])
    assert telemetry["hot_sampling_covered"] is False
    assert report["rss_check"] is None
