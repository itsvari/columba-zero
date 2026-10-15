"""Probe every configured GPU in one minute without a vision environment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import subprocess
from pathlib import Path

from app.json_data import JsonValue
from cluster.artifacts import atomic_json
from cluster.device_faults import (
    DEVICE_FAILURE_EXIT_STATUS,
    clear_bad_node,
    expected_gpu_count,
    node_resources,
)


def probe(clear_on_success: bool = False) -> dict[str, JsonValue]:
    root = (Path.home() / "columba").resolve()
    destination = Path(os.environ["COLUMBA_RUN_PATH"]).resolve()
    if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-") or not destination.is_relative_to(
        root
    ):
        raise ValueError("probe requires an owned cz- job and project run path")
    node = platform.node().split(".")[0]
    resource_type, expected_count = node_resources(node)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    devices = (
        list(csv.DictReader(io.StringIO(result.stdout), skipinitialspace=True))
        if result.returncode == 0
        else []
    )
    allocated_count = expected_gpu_count()
    valid = all(
        device.get("name", "").strip()
        and not any(
            token in str(device).lower() for token in ("error", "n/a", "lost", "no devices")
        )
        for device in devices
    )
    success = result.returncode == 0 and valid and len(devices) == expected_count == allocated_count
    report: dict[str, JsonValue] = {
        "host": node,
        "architecture": platform.machine(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "label": "measured",
        "nvidia_smi_exit_status": result.returncode,
        "nvidia_smi": result.stdout,
        "expected_gpu_count": expected_count,
        "allocated_gpu_count": allocated_count,
        "visible_gpu_count": len(devices),
        "gres_type": resource_type,
        "status": "probe_complete" if success else "probe_failed",
        "cuda_check": "not run: system-Python one-minute probe",
        "cleared_dynamic_exclusion": success and clear_on_success,
    }
    if success and clear_on_success:
        clear_bad_node(root, node)
    atomic_json(destination / "run.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-on-success", action="store_true")
    arguments = parser.parse_args()
    report = probe(arguments.clear_on_success)
    if report["status"] != "probe_complete":
        raise SystemExit(DEVICE_FAILURE_EXIT_STATUS)


if __name__ == "__main__":
    main()
