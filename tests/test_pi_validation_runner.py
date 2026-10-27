"""Exercise supervisor shutdown with a local fake exporter, without a camera or bus."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.json_data import JsonValue
from tools import run_pi_validation as runner

READY_TIMEOUT_S = 5.0
TEST_POLL_INTERVAL_S = 0.01
TEST_CLEANUP_GRACE_S = 0.1
FAKE_EXPORT_DURATION_S = 0.7

FAKE_APPLICATION = """
import signal
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
stalled = sys.argv[2] == "stalled"
term_count = 0
export_deadline_s = None

def request_stop(signal_number, frame):
    global term_count, export_deadline_s
    term_count += 1
    (directory / "term-count").write_text(str(term_count))
    if term_count > 1:
        print("export cancelled by additional TERM", flush=True)
        raise SystemExit(3)
    export_deadline_s = time.monotonic() + float(sys.argv[3])
    (directory / "exporting").touch()
    print("export started", flush=True)

signal.signal(signal.SIGTERM, request_stop)
(directory / "ready").touch()
print("capture started", flush=True)
while stalled or export_deadline_s is None or time.monotonic() < export_deadline_s:
    time.sleep(0.01)
print("export complete", flush=True)
"""


def fake_command(directory: Path, stalled: bool = False) -> tuple[str, ...]:
    script = directory / "fake_application.py"
    script.write_text(FAKE_APPLICATION, encoding="utf-8")
    return (
        sys.executable,
        "-u",
        str(script),
        str(directory),
        "stalled" if stalled else "exports",
        str(FAKE_EXPORT_DURATION_S),
    )


def await_file(path: Path) -> None:
    deadline_s = time.monotonic() + READY_TIMEOUT_S
    while not path.exists():
        if time.monotonic() >= deadline_s:
            raise AssertionError(f"fake child did not create {path}")
        time.sleep(TEST_POLL_INTERVAL_S)


def read_manifest(directory: Path) -> dict[str, JsonValue]:
    manifest: JsonValue = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    return manifest


def test_fake_export_reproduces_double_term_cancellation(tmp_path: Path) -> None:
    child = subprocess.Popen(
        fake_command(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        await_file(tmp_path / "ready")
        child.terminate()
        await_file(tmp_path / "exporting")
        child.terminate()
        console, _ = child.communicate(timeout=READY_TIMEOUT_S)
        assert child.returncode == 3
        assert (tmp_path / "term-count").read_text() == "2"
        assert b"export cancelled by additional TERM" in console
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=READY_TIMEOUT_S)


def test_stop_file_repeated_signals_and_capture_deadline_send_one_term(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "CLEANUP_GRACE_S", TEST_CLEANUP_GRACE_S)
    command = fake_command(tmp_path)
    stop_file = tmp_path / "stop"

    def trigger_stop_requests() -> None:
        await_file(tmp_path / "ready")
        stop_file.touch()
        await_file(tmp_path / "exporting")
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(TEST_POLL_INTERVAL_S)
        os.kill(os.getpid(), signal.SIGINT)
        stop_file.touch()

    previous_term_handler = signal.getsignal(signal.SIGTERM)
    previous_interrupt_handler = signal.getsignal(signal.SIGINT)
    with ThreadPoolExecutor(max_workers=1) as executor:
        requests = executor.submit(trigger_stop_requests)
        assert runner.supervise(command, tmp_path, stop_file, 0.4, 2.0, {}) == 0
        requests.result(timeout=READY_TIMEOUT_S)
    manifest = read_manifest(tmp_path)
    assert (tmp_path / "term-count").read_text() == "1"
    assert "export complete" in (tmp_path / "console.log").read_text()
    assert manifest["term_sent_count"] == 1
    assert manifest["supervisor_signal_count"] == 2
    assert manifest["first_supervisor_signal"] in (signal.SIGTERM, signal.SIGINT)
    assert manifest["stop_reason"] == "stop_file"
    assert manifest["forced_kill"] is False
    assert manifest["child_exit_status"] == 0
    start_s = manifest["started_at_monotonic_s"]
    capture_deadline_s = manifest["capture_deadline_monotonic_s"]
    export_deadline_s = manifest["export_deadline_monotonic_s"]
    requested_stop_s = manifest["requested_stop_monotonic_s"]
    finished_s = manifest["finished_at_monotonic_s"]
    assert isinstance(start_s, float) and isinstance(capture_deadline_s, float)
    assert isinstance(export_deadline_s, float) and isinstance(requested_stop_s, float)
    assert isinstance(finished_s, float)
    assert capture_deadline_s == pytest.approx(start_s + 0.4)
    assert export_deadline_s == pytest.approx(requested_stop_s + 2.0 + TEST_CLEANUP_GRACE_S)
    assert capture_deadline_s < finished_s < export_deadline_s
    assert signal.getsignal(signal.SIGTERM) == previous_term_handler
    assert signal.getsignal(signal.SIGINT) == previous_interrupt_handler


def test_capture_deadline_allows_export_to_finish(tmp_path: Path) -> None:
    assert runner.supervise(fake_command(tmp_path), tmp_path, tmp_path / "stop", 0.3, 2.0, {}) == 0
    manifest = read_manifest(tmp_path)
    assert manifest["stop_reason"] == "capture_deadline"
    assert manifest["term_sent_count"] == 1
    assert manifest["child_exit_status"] == 0
    assert "export complete" in (tmp_path / "console.log").read_text()


def test_natural_child_exit_needs_no_termination(tmp_path: Path) -> None:
    command = (sys.executable, "-u", "-c", "print('completed without a capture stop')")
    assert runner.supervise(command, tmp_path, tmp_path / "stop", 1.0, 2.0, {}) == 0
    manifest = read_manifest(tmp_path)
    assert manifest["child_exit_status"] == 0
    assert manifest["term_sent_count"] == 0
    assert manifest["stop_reason"] is None
    assert manifest["export_deadline_monotonic_s"] is None


def test_stalled_child_is_killed_only_after_cleanup_export_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "CLEANUP_GRACE_S", TEST_CLEANUP_GRACE_S)
    assert (
        runner.supervise(
            fake_command(tmp_path, stalled=True), tmp_path, tmp_path / "stop", 0.3, 0.15, {}
        )
        == 1
    )
    manifest = read_manifest(tmp_path)
    assert manifest["stop_reason"] == "capture_deadline"
    assert manifest["term_sent_count"] == 1
    assert manifest["forced_kill"] is True
    assert manifest["child_exit_status"] == -signal.SIGKILL
    assert (tmp_path / "term-count").read_text() == "1"
    deadline_s = manifest["export_deadline_monotonic_s"]
    finished_s = manifest["finished_at_monotonic_s"]
    assert isinstance(deadline_s, float) and isinstance(finished_s, float)
    assert finished_s >= deadline_s


@pytest.mark.parametrize("failure_phase", ("launched", "stopping"))
def test_supervisor_exception_still_sends_only_one_term_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    monkeypatch.setattr(runner, "CLEANUP_GRACE_S", TEST_CLEANUP_GRACE_S)
    write_manifest = runner._write_manifest
    failure_injected = False

    def fail_first_stop_write(output: Path, manifest: dict[str, JsonValue]) -> None:
        nonlocal failure_injected
        expected_term_count = 0 if failure_phase == "launched" else 1
        if (
            not failure_injected
            and manifest["child_pid"] is not None
            and manifest["term_sent_count"] == expected_term_count
        ):
            await_file(tmp_path / "ready")
            failure_injected = True
            raise OSError("synthetic manifest failure during export")
        write_manifest(output, manifest)

    monkeypatch.setattr(runner, "_write_manifest", fail_first_stop_write)
    with pytest.raises(OSError, match="synthetic manifest failure"):
        runner.supervise(fake_command(tmp_path), tmp_path, tmp_path / "stop", 0.3, 2.0, {})
    manifest = read_manifest(tmp_path)
    assert manifest["child_exit_status"] == 0
    assert manifest["term_sent_count"] == 1
    assert manifest["forced_kill"] is False
    assert (tmp_path / "term-count").read_text() == "1"
    assert "synthetic manifest failure" in str(manifest["supervisor_error"])


@pytest.mark.parametrize(
    "invalid",
    (
        "duration-zero",
        "duration-negative",
        "duration-nan",
        "duration-inf",
        "config",
        "hardware",
        "stop",
        "stop-symlink",
        "stop-output",
        "stop-console",
        "stop-parent",
        "output",
    ),
)
def test_invalid_cli_cannot_start_a_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    config = tmp_path / "profile.yaml"
    config.write_text("{}", encoding="utf-8")
    stop_file = tmp_path / "stop"
    output = tmp_path / "supervisor"
    duration_s = "1"
    if invalid.startswith("duration-"):
        duration_s = {"zero": "0", "negative": "-1", "nan": "nan", "inf": "inf"}[
            invalid.removeprefix("duration-")
        ]
    elif invalid == "config":
        config.write_text("preview: {export_timeout_s: .nan}", encoding="utf-8")
    elif invalid == "hardware":
        config.write_text(
            "confirmation: {enabled: true}\nfollow: {enabled: true}\npan_tilt: {mode: hardware}",
            encoding="utf-8",
        )
    elif invalid == "stop":
        stop_file.touch()
    elif invalid == "stop-symlink":
        stop_file.symlink_to(tmp_path / "missing-target")
    elif invalid == "stop-output":
        stop_file = output
    elif invalid == "stop-console":
        stop_file = output / "console.log"
    elif invalid == "stop-parent":
        stop_file = tmp_path / "missing-directory" / "stop"
    elif invalid == "output":
        output.mkdir()

    def unexpected_launch(
        command: Sequence[str],
        destination: Path,
        requested_stop: Path,
        capture_duration_s: float,
        export_timeout_s: float,
        provenance: dict[str, JsonValue],
    ) -> int:
        raise AssertionError("invalid arguments must not launch a child")

    monkeypatch.setattr(runner, "supervise", unexpected_launch)
    with pytest.raises(SystemExit) as failure:
        runner.main(
            [
                "--config",
                str(config),
                "--duration-s",
                duration_s,
                "--stop-file",
                str(stop_file),
                "--output",
                str(output),
            ]
        )
    assert failure.value.code == 2


@pytest.mark.parametrize("hardware", (False, True))
def test_cli_snapshots_config_and_launches_only_the_explicit_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hardware: bool
) -> None:
    config = tmp_path / "profile.yaml"
    config.write_text(
        "confirmation: {enabled: true}\nfollow: {enabled: true}\npan_tilt: {mode: "
        + ("hardware" if hardware else "simulate")
        + "}\npreview: {export_timeout_s: 12}\n",
        encoding="utf-8",
    )
    output = tmp_path / "supervisor"
    stop_file = tmp_path / "stop"

    def inspect_launch(
        command: Sequence[str],
        destination: Path,
        requested_stop: Path,
        duration_s: float,
        export_timeout_s: float,
        provenance: dict[str, JsonValue],
    ) -> int:
        assert tuple(command) == (
            sys.executable,
            "-u",
            "-m",
            "app.main",
            "--config",
            str(output / "config.yaml"),
        )
        assert destination == output and requested_stop == stop_file
        assert duration_s == 270 and export_timeout_s == 12
        assert provenance["hardware_authorised"] is hardware
        assert provenance["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        source_hash = provenance["source_sha256"]
        assert isinstance(source_hash, str) and len(source_hash) == 64
        assert (output / "config.yaml").read_bytes() == config.read_bytes()
        return 0

    monkeypatch.setattr(runner, "supervise", inspect_launch)
    arguments = [
        "--config",
        str(config),
        "--duration-s",
        "270",
        "--stop-file",
        str(stop_file),
        "--output",
        str(output),
    ]
    if hardware:
        arguments.append("--hardware")
    assert runner.main(arguments) == 0
