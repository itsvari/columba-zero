from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tools.pi_telemetry import measured_arm_frequency_hz, record_process

SAMPLE_INTERVAL_S = 0.01
TEST_DURATION_S = 0.05


def test_process_telemetry_records_real_samples_and_a_declared_window(tmp_path: Path) -> None:
    destination = tmp_path / "telemetry.jsonl"
    record_process(os.getpid(), destination, SAMPLE_INTERVAL_S, TEST_DURATION_S, threading.Event())
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert records[0]["kind"] == "session"
    assert records[0]["process_id"] == os.getpid()
    samples = [record for record in records if record["kind"] == "sample"]
    assert samples
    assert samples[0]["process_cpu_percent"] is None
    assert all(sample["process_memory_rss_bytes"] > 0 for sample in samples)
    assert all("throttled" in sample and "temperature_millicelsius" in sample for sample in samples)
    assert all("cpu_measured_frequency_hz" in sample for sample in samples)
    assert records[-1]["reason"] == "duration_limit"


def test_a_stop_request_finishes_without_claiming_samples(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    destination = tmp_path / "telemetry.jsonl"
    record_process(os.getpid(), destination, SAMPLE_INTERVAL_S, TEST_DURATION_S, stop)
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [record["kind"] for record in records] == ["session", "end"]
    assert records[-1]["reason"] == "signal"


def test_invalid_interval_is_rejected_without_creating_evidence(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "pi_telemetry.py"
    destination = tmp_path / "telemetry.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--process-id",
            str(os.getpid()),
            "--output",
            str(destination),
            "--interval-s",
            "nan",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--interval-s must be finite" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize(
    ("response", "returncode", "expected_hz"),
    [
        ("frequency(48)=1500000000\n", 0, 1_500_000_000),
        ("frequency(48)=2400000000", 0, 2_400_000_000),
        ("frequency(48)=0", 0, None),
        ("frequency(48)=-1", 0, None),
        ("nominal=2400000000", 0, None),
        ("frequency(48)=1500000000", 1, None),
    ],
)
def test_measured_clock_is_distinct_from_nominal_frequency(
    response: str, returncode: int, expected_hz: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def query(
        arguments: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert arguments == ["vcgencmd", "measure_clock", "arm"]
        assert capture_output and text and timeout > 0 and check is False
        return subprocess.CompletedProcess(arguments, returncode, stdout=response)

    monkeypatch.setattr("tools.pi_telemetry.subprocess.run", query)
    assert measured_arm_frequency_hz() == expected_hz


def test_unavailable_measured_clock_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*_arguments: str, **_options: bool) -> None:
        raise FileNotFoundError("vcgencmd")

    monkeypatch.setattr("tools.pi_telemetry.subprocess.run", unavailable)
    assert measured_arm_frequency_hz() is None
