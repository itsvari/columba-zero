"""Bound one validation capture while preserving its separate export window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig
from app.json_data import JsonValue

CLEANUP_GRACE_S = 30.0
POLL_INTERVAL_S = 0.1
KILL_REAP_TIMEOUT_S = 5.0
SCHEMA_VERSION = 1


def _write_manifest(output: Path, manifest: dict[str, JsonValue]) -> None:
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "app").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def supervise(
    command: Sequence[str],
    output: Path,
    stop_file: Path,
    duration_s: float,
    export_timeout_s: float,
    provenance: dict[str, JsonValue],
) -> int:
    started_at_monotonic_s = time.monotonic()
    capture_deadline_monotonic_s = started_at_monotonic_s + duration_s
    export_deadline_monotonic_s: float | None = None
    child: subprocess.Popen[bytes] | None = None
    supervisor_signal_count = 0
    first_supervisor_signal: int | None = None
    manifest: dict[str, JsonValue] = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "command": list(command),
        "working_directory": str(Path.cwd()),
        "supervisor_pid": os.getpid(),
        "child_pid": None,
        "started_at_unix_s": time.time(),
        "started_at_monotonic_s": started_at_monotonic_s,
        "duration_s": duration_s,
        "capture_deadline_monotonic_s": capture_deadline_monotonic_s,
        "export_timeout_s": export_timeout_s,
        "cleanup_grace_s": CLEANUP_GRACE_S,
        "export_deadline_monotonic_s": None,
        "stop_file": str(stop_file),
        "stop_reason": None,
        "requested_stop_monotonic_s": None,
        "supervisor_signal_count": 0,
        "first_supervisor_signal": None,
        "term_sent_count": 0,
        "forced_kill": False,
        "child_exit_status": None,
        "supervisor_error": None,
    }

    def request_stop(signal_number: int, _frame: FrameType | None) -> None:
        nonlocal supervisor_signal_count, first_supervisor_signal
        supervisor_signal_count += 1
        if first_supervisor_signal is None:
            first_supervisor_signal = signal_number

    def terminate_once(reason: str) -> None:
        nonlocal export_deadline_monotonic_s
        if export_deadline_monotonic_s is not None:
            return
        requested_at_monotonic_s = time.monotonic()
        export_deadline_monotonic_s = requested_at_monotonic_s + export_timeout_s + CLEANUP_GRACE_S
        manifest.update(
            stop_reason=reason,
            requested_stop_monotonic_s=requested_at_monotonic_s,
            export_deadline_monotonic_s=export_deadline_monotonic_s,
            term_sent_count=1,
        )
        assert child is not None
        try:
            child.terminate()
        except ProcessLookupError:
            pass
        _write_manifest(output, manifest)

    def kill_child() -> None:
        assert child is not None
        manifest["forced_kill"] = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=KILL_REAP_TIMEOUT_S)

    def monitor_child() -> None:
        assert child is not None
        while child.poll() is None:
            now_monotonic_s = time.monotonic()
            if export_deadline_monotonic_s is not None:
                if now_monotonic_s >= export_deadline_monotonic_s:
                    kill_child()
                    return
            elif supervisor_signal_count:
                terminate_once("supervisor_signal")
            elif stop_file.exists() or stop_file.is_symlink():
                terminate_once("stop_file")
            elif now_monotonic_s >= capture_deadline_monotonic_s:
                terminate_once("capture_deadline")
            time.sleep(POLL_INTERVAL_S)

    def cleanup_child() -> None:
        if child is None or child.poll() is not None:
            return
        try:
            terminate_once("supervisor_exception")
        finally:
            assert export_deadline_monotonic_s is not None
            remaining_s = max(0.0, export_deadline_monotonic_s - time.monotonic())
            try:
                child.wait(timeout=remaining_s)
            except subprocess.TimeoutExpired:
                kill_child()

    previous_handlers = {
        signal_number: signal.signal(signal_number, request_stop)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        _write_manifest(output, manifest)
        with (output / "console.log").open("xb") as console:
            child = subprocess.Popen(
                command, stdout=console, stderr=subprocess.STDOUT, start_new_session=True
            )
            manifest["child_pid"] = child.pid
            _write_manifest(output, manifest)
            monitor_child()
    except BaseException as error:
        manifest["supervisor_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            cleanup_child()
            manifest.update(
                child_exit_status=child.returncode if child is not None else None,
                supervisor_signal_count=supervisor_signal_count,
                first_supervisor_signal=first_supervisor_signal,
                finished_at_monotonic_s=time.monotonic(),
                finished_at_unix_s=time.time(),
            )
            _write_manifest(output, manifest)
        finally:
            for signal_number, handler in previous_handlers.items():
                signal.signal(signal_number, handler)
    return 0 if child is not None and child.returncode == 0 else 1


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", action="store_true")
    options = parser.parse_args(arguments)
    if not math.isfinite(options.duration_s) or options.duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    try:
        if options.stop_file.exists() or options.stop_file.is_symlink():
            raise ValueError("--stop-file must be new; choose a unique absent path")
        reserved_paths = (
            options.output,
            *(
                options.output / name
                for name in ("config.yaml", "manifest.json", "manifest.json.tmp", "console.log")
            ),
        )
        if options.stop_file.resolve() in (path.resolve() for path in reserved_paths):
            raise ValueError("--stop-file must differ from the supervisor's output paths")
        if not options.stop_file.parent.is_dir() and options.stop_file.parent != options.output:
            raise ValueError("--stop-file parent directory must exist or be --output")
        contents = options.config.read_bytes()
        options.output.mkdir()
        snapshot = options.output.resolve() / "config.yaml"
        snapshot.write_bytes(contents)
        configuration = AppConfig.load(snapshot)
        if configuration.follow.enabled and configuration.pan_tilt.mode == "hardware":
            if not options.hardware:
                raise ValueError("hardware following requires explicit --hardware")
        provenance: dict[str, JsonValue] = {
            "config_source_path": str(options.config.resolve()),
            "config_snapshot_path": str(snapshot),
            "config_sha256": hashlib.sha256(contents).hexdigest(),
            "source_sha256": _source_hash(),
            "source_hash_scope": "app/**/*.py and tools/run_pi_validation.py",
            "hardware_authorised": options.hardware,
        }
        return supervise(
            (sys.executable, "-u", "-m", "app.main", "--config", str(snapshot)),
            options.output.resolve(),
            options.stop_file.absolute(),
            options.duration_s,
            configuration.preview.export_timeout_s,
            provenance,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
