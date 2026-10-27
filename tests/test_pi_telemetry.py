from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from tools.pi_telemetry import record_process

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
