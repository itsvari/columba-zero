from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from app.config import AppConfig
from tools.benchmark_live_tracking import probe_configuration, stop_process


def test_live_probe_cannot_retain_hardware_or_output_modes(tmp_path: Path) -> None:
    original = AppConfig()
    original = replace(
        original,
        pan_tilt=replace(original.pan_tilt, mode="hardware", calibration="missing.local.yaml"),
        output=replace(original.output, record=True),
        preview=replace(original.preview, enabled=True, record=True),
    )
    configured = probe_configuration(original, tmp_path, threads=2)
    configured.validate()
    assert configured.pan_tilt.mode == "simulate"
    assert not configured.display.enabled
    assert not configured.output.record
    assert not configured.preview.enabled
    assert not configured.preview.record
    assert configured.follow.enabled
    assert configured.confirmation.enabled
    assert configured.model.confidence == original.model.confidence
    assert configured.model.image_size == original.model.image_size
    assert configured.camera == original.camera
    assert original.pan_tilt.mode == "hardware"
    assert original.preview.record


def test_probe_shutdown_reaps_its_child_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stop_process(process)
        assert process.poll() is not None
        stop_process(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_live_probe_rejects_nonfinite_deadlines_before_loading_configuration(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "benchmark_live_tracking.py"
    for value in ("nan", "inf", "0", "-1"):
        result = subprocess.run(
            [sys.executable, str(script), "--output", str(tmp_path), "--timeout-s", value],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert result.returncode == 2
        assert "--timeout-s must be finite and positive" in result.stderr
